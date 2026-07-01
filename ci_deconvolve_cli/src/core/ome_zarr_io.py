"""OME-Zarr helpers for the focused ci_deconvolve CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _metadata_warnings(metadata: dict[str, Any]) -> list[str]:
    warnings = metadata.get("metadata_warnings")
    if isinstance(warnings, list):
        return [str(item) for item in warnings if str(item).strip()]
    return []


def _metadata_description(metadata: dict[str, Any]) -> str:
    parts = []
    defaulted = sorted(str(item) for item in metadata.get("_defaulted_keys", set()) if item)
    if defaulted:
        parts.append("CIDeconvolve metadata defaults: " + ", ".join(defaulted))
    warnings = _metadata_warnings(metadata)
    if warnings:
        parts.append("CIDeconvolve metadata warnings: " + " | ".join(warnings))
    return "\n".join(parts)


def _coerce_rgb(color: Any) -> tuple[int, int, int] | None:
    if isinstance(color, str):
        text = color.strip().lstrip("#")
        if len(text) == 6:
            try:
                return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return None
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            return tuple(max(0, min(255, int(v))) for v in color[:3])
        except (TypeError, ValueError):
            return None
    return None


def _rgb_to_ome_hex(color: Any) -> str | None:
    rgb = _coerce_rgb(color)
    if rgb is None:
        return None
    return "".join(f"{v:02X}" for v in rgb)


def _ome_zarr_name(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (".ome.zarr", ".zarr"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _resolve_channel_display_colors(metadata: dict[str, Any], n_channels: int) -> list[Any]:
    channels = metadata.get("channels") or []
    colors = []
    default_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
    for i in range(n_channels):
        ch = channels[i] if i < len(channels) and isinstance(channels[i], dict) else {}
        colors.append(ch.get("color") or default_colors[i % len(default_colors)])
    return colors


def zarr_attrs(path: Path) -> dict[str, Any]:
    try:
        attrs_path = path / ".zattrs"
        if attrs_path.is_file():
            data = json.loads(attrs_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def is_ome_zarr_image_group(path: Path) -> bool:
    return isinstance(zarr_attrs(path).get("multiscales"), list)


def bioformats2raw_primary_series_path(path: Path) -> Optional[Path]:
    attrs = zarr_attrs(path)
    if "bioformats2raw.layout" not in attrs:
        return None

    series: list[str] = []
    raw_series = zarr_attrs(path / "OME").get("series")
    if isinstance(raw_series, list):
        series.extend(str(item) for item in raw_series if str(item))

    if not series:
        try:
            series.extend(
                child.name
                for child in sorted(path.iterdir(), key=lambda p: p.name)
                if child.is_dir() and is_ome_zarr_image_group(child)
            )
        except Exception:
            pass

    for series_name in series:
        candidate = path / series_name
        if is_ome_zarr_image_group(candidate):
            return candidate
    return None


def resolve_ome_zarr_image_path(path: str | Path) -> Path:
    path = Path(path)
    if is_ome_zarr_image_group(path):
        return path
    if path.is_dir() and path.suffix.lower() == ".zarr":
        series_path = bioformats2raw_primary_series_path(path)
        if series_path is not None:
            return series_path
    return path


def ome_zarr_format_for_path(path: str | Path):
    from ome_zarr.format import CurrentFormat, FormatV04

    attrs = zarr_attrs(resolve_ome_zarr_image_path(path))
    for multiscale in attrs.get("multiscales") or []:
        if str(multiscale.get("version", "")).startswith("0.4"):
            return FormatV04()
    return CurrentFormat()


def open_ome_zarr_image_node(path: str | Path):
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    image_path = resolve_ome_zarr_image_path(path)
    loc = parse_url(str(image_path), fmt=ome_zarr_format_for_path(image_path))
    if loc is None:
        raise ValueError(f"Could not open OME-Zarr path: {image_path}")
    for node in Reader(loc)():
        if isinstance(getattr(node, "data", None), list) and node.data:
            return image_path, node
    raise ValueError(f"No OME-Zarr image node found in {image_path}")


def is_hcs_plate(path: str | Path) -> bool:
    """Return True when *path* is an OME-Zarr HCS plate root."""
    try:
        import zarr

        root = zarr.open(str(path), mode="r")
        return "plate" in dict(root.attrs)
    except Exception:
        return False


def _result_to_tczyx(result: dict[str, Any]) -> np.ndarray:
    channels = [np.asarray(ch, dtype=np.float32) for ch in result["channels"]]
    if not channels:
        raise ValueError("No channels to write")
    if channels[0].ndim == 3:
        stack = np.stack(channels, axis=0)
        return stack[np.newaxis, ...].astype(np.float32, copy=False)
    if channels[0].ndim == 2:
        stack = np.stack(channels, axis=0)
        return stack[np.newaxis, :, np.newaxis, :, :].astype(np.float32, copy=False)
    raise ValueError(f"OME-Zarr output expects 2D or 3D channels, got {channels[0].ndim}D")


def _axes_metadata() -> list[dict[str, str]]:
    return [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def _scale_transform(px_z: float, px_y: float, px_x: float) -> list[dict[str, Any]]:
    return [{
        "type": "scale",
        "scale": [1.0, 1.0, float(px_z), float(px_y), float(px_x)],
    }]


def _omero_metadata(metadata: dict[str, Any], n_channels: int) -> dict[str, Any]:
    names = list(metadata.get("channel_names") or [])
    source_channels = [
        dict(ch) if isinstance(ch, dict) else {}
        for ch in (metadata.get("channels") or [])
    ]
    colors = _resolve_channel_display_colors(metadata, n_channels)
    channels = []
    for i in range(n_channels):
        src = source_channels[i] if i < len(source_channels) else {}
        label = src.get("name") or src.get("label") or (names[i] if i < len(names) else f"Ch{i}")
        color = _rgb_to_ome_hex(colors[i] if i < len(colors) else src.get("color")) or "FFFFFF"
        window_start = src.get("window_start")
        window_end = src.get("window_end")
        try:
            start = float(window_start) if window_start is not None else 0.0
            end = float(window_end) if window_end is not None else max(start, 1.0)
        except (TypeError, ValueError):
            start, end = 0.0, 1.0
        if end <= start:
            end = start + 1.0
        channels.append({
            "label": str(label),
            "color": color,
            "active": bool(src.get("active", True)),
            "coefficient": 1,
            "family": "linear",
            "inverted": False,
            "window": {"start": start, "end": end, "min": min(start, 0.0), "max": max(end, 1.0)},
        })
    omero = {
        "channels": channels,
        "name": str(metadata.get("name") or "CIDeconvolve"),
        "rdefs": {
            "defaultT": int(metadata.get("default_t", 0) or 0),
            "defaultZ": int(metadata.get("default_z", 0) or 0),
            "model": "color",
        },
    }
    description = _metadata_description(metadata)
    if description:
        omero["description"] = description
    return omero


def save_result_ome_zarr(
    result: dict[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Write a deconvolution result as OME-Zarr v0.4 / Zarr v2.

    This targets the common compatibility floor for QuPath 0.7 and OMERO:
    `.zgroup`/`.zattrs` metadata, NGFF multiscales version 0.4, and OMERO
    channel display metadata at the image root.
    """
    output_path = Path(output_path)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(output_path)
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import zarr
    except Exception as exc:
        raise ImportError(
            "OME-Zarr output requires zarr. "
            "Install the CLI package dependencies or run: pip install 'zarr>=2.16,<4'."
        ) from exc

    data = _result_to_tczyx(result)
    metadata = dict(result.get("metadata") or {})
    px_x = float(metadata.get("pixel_size_x") or 1.0)
    px_y = float(metadata.get("pixel_size_y") or px_x)
    px_z = float(metadata.get("pixel_size_z") or 1.0)
    chunks = (1, 1, min(data.shape[2], 16), min(data.shape[3], 512), min(data.shape[4], 512))

    try:
        root = zarr.open_group(str(output_path), mode="w", zarr_format=2)
    except TypeError:
        root = zarr.open_group(str(output_path), mode="w", zarr_version=2)

    create_kwargs = {
        "shape": data.shape,
        "chunks": chunks,
        "dtype": data.dtype,
    }
    try:
        array = root.create_dataset("0", dimension_separator="/", **create_kwargs)
    except TypeError:
        array = root.create_dataset("0", **create_kwargs)
    array[:] = data

    name = str(metadata.get("name") or _ome_zarr_name(output_path))
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": name,
        "axes": _axes_metadata(),
        "datasets": [{
            "path": "0",
            "coordinateTransformations": _scale_transform(px_z, px_y, px_x),
        }],
    }]
    root.attrs["omero"] = _omero_metadata(metadata, data.shape[1])
    root.attrs["cideconvolve"] = {
        "metadata": metadata,
        "physical_pixel_sizes_um": {"x": px_x, "y": px_y, "z": px_z},
    }
    return output_path
