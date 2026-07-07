"""Small OME-Zarr helpers shared by GUI, wrapper, and streaming code."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape

import numpy as np


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


def _ome_zarr_name(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (".ome.zarr", ".zarr"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


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
    metadata = dict(result.get("metadata") or {})
    if channels[0].ndim == 4:
        return np.stack(channels, axis=1).astype(np.float32, copy=False)
    if (
        channels[0].ndim == 3
        and int(metadata.get("size_t", 1) or 1) > 1
        and int(metadata.get("size_z", 1) or 1) == 1
    ):
        stack = np.stack(channels, axis=1)
        return stack[:, :, np.newaxis, :, :].astype(np.float32, copy=False)
    if channels[0].ndim == 3:
        stack = np.stack(channels, axis=0)
        return stack[np.newaxis, ...].astype(np.float32, copy=False)
    if channels[0].ndim == 2:
        stack = np.stack(channels, axis=0)
        return stack[np.newaxis, :, np.newaxis, :, :].astype(np.float32, copy=False)
    raise ValueError(f"OME-Zarr output expects 2D or 3D channels, got {channels[0].ndim}D")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted((_jsonable(v) for v in value), key=str)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if number <= 0 or not np.isfinite(number):
        return float(default)
    return number


def _cideconvolve_metadata_payload(metadata: dict[str, Any], shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "creator": "CIDeconvolve",
        "shape_tczyx": list(shape),
        "metadata": _jsonable(metadata),
        "physical_pixel_sizes_um": {
            "x": _positive_float(metadata.get("pixel_size_x"), 1.0),
            "y": _positive_float(
                metadata.get("pixel_size_y"),
                _positive_float(metadata.get("pixel_size_x"), 1.0),
            ),
            "z": _positive_float(metadata.get("pixel_size_z"), 1.0),
        },
        "channels": _jsonable(metadata.get("channels") or []),
        "processing": _jsonable(metadata.get("cideconvolve_processing") or {}),
        "source": {
            "id": metadata.get("id"),
            "name": metadata.get("name"),
            "source_id": metadata.get("source_id"),
        },
    }


def _ome_dtype(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float32"):
        return "float"
    if dtype == np.dtype("float64"):
        return "double"
    if dtype == np.dtype("uint8"):
        return "uint8"
    if dtype == np.dtype("uint16"):
        return "uint16"
    if dtype == np.dtype("uint32"):
        return "uint32"
    if dtype == np.dtype("int8"):
        return "int8"
    if dtype == np.dtype("int16"):
        return "int16"
    if dtype == np.dtype("int32"):
        return "int32"
    return "float"


def _xml_escape_attr(value: Any) -> str:
    return escape(str(value), {'"': "&quot;"})


def _xml_attr(name: str, value: Any) -> str:
    if value in (None, ""):
        return ""
    return f' {name}="{_xml_escape_attr(value)}"'


def _ome_xml_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _ome_acquisition_mode(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    mapping = {
        "wide field": "WideField",
        "widefield": "WideField",
        "confocal": "LaserScanningConfocalMicroscopy",
        "laser scanning confocal": "LaserScanningConfocalMicroscopy",
        "spinning disk": "SpinningDiskConfocal",
        "spinning disk confocal": "SpinningDiskConfocal",
        "tirm": "TIRF",
        "tirf": "TIRF",
        "sted": "STED",
    }
    return mapping.get(text)


def _ome_channel_color(color: Any) -> int | None:
    rgb = None
    if isinstance(color, str):
        text = color.strip().lstrip("#")
        if len(text) == 6:
            try:
                rgb = tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                rgb = None
    elif isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            rgb = tuple(max(0, min(255, int(v))) for v in color[:3])
        except (TypeError, ValueError):
            rgb = None
    if rgb is None:
        return None
    r, g, b = rgb
    rgba = (r << 24) | (g << 16) | (b << 8) | 255
    return rgba - (1 << 32) if rgba >= (1 << 31) else rgba


def _channel_xml_attrs(channel: dict[str, Any], index: int) -> str:
    attrs = [
        _xml_attr("ID", channel.get("id") or f"Channel:{index}"),
        _xml_attr("Name", channel.get("name") or channel.get("label") or f"Channel {index + 1}"),
    ]
    color = _ome_channel_color(channel.get("color"))
    if color is not None:
        attrs.append(_xml_attr("Color", color))
    for src, dst in (
        ("emission_wavelength", "EmissionWavelength"),
        ("excitation_wavelength", "ExcitationWavelength"),
    ):
        value = _ome_xml_float(channel.get(src))
        if value is not None:
            attrs.append(_xml_attr(dst, value))
            attrs.append(_xml_attr(f"{dst}Unit", "nm"))
    return "".join(attrs)


def _metadata_xml_annotation(metadata: dict[str, Any], shape_tczyx: tuple[int, ...]) -> str:
    payload = _cideconvolve_metadata_payload(metadata, shape_tczyx)
    payload["ome_xml_note"] = "Full CIDeconvolve metadata serialized as JSON because not every key maps to a native OME-XML field."
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    return (
        '<StructuredAnnotations>'
        '<CommentAnnotation ID="Annotation:CIDeconvolve:0" Namespace="CIDeconvolve">'
        f"<Value>{escape(text)}</Value>"
        '</CommentAnnotation>'
        '</StructuredAnnotations>'
    )


def _write_ome_xml_metadata(
    output_path: Path,
    metadata: dict[str, Any],
    shape_tczyx: tuple[int, int, int, int, int],
    dtype: np.dtype,
    image_name: str,
) -> None:
    t, c, z, y, x = shape_tczyx
    ome_dir = output_path / "OME"
    ome_dir.mkdir(parents=True, exist_ok=True)
    (ome_dir / ".zgroup").write_text(json.dumps({"zarr_format": 2}, indent=2), encoding="utf-8")

    objective_attrs = [_xml_attr("ID", "Objective:0")]
    na = _ome_xml_float(metadata.get("na"))
    if na is not None:
        objective_attrs.append(_xml_attr("LensNA", na))
    mag = _ome_xml_float(metadata.get("magnification"))
    if mag is not None:
        objective_attrs.append(_xml_attr("NominalMagnification", mag))
    immersion = str(metadata.get("immersion") or "").lower()
    if "oil" in immersion:
        objective_attrs.append(_xml_attr("Immersion", "Oil"))
    elif "water" in immersion:
        objective_attrs.append(_xml_attr("Immersion", "Water"))
    elif "air" in immersion:
        objective_attrs.append(_xml_attr("Immersion", "Air"))

    objective_settings_attrs = [_xml_attr("ID", "Objective:0")]
    sample_ri = _ome_xml_float(metadata.get("sample_refractive_index"))
    if sample_ri is not None:
        objective_settings_attrs.append(_xml_attr("RefractiveIndex", sample_ri))

    channels = [
        dict(ch) if isinstance(ch, dict) else {}
        for ch in (metadata.get("channels") or [])
    ]
    if len(channels) < c:
        channels.extend({} for _ in range(c - len(channels)))
    channel_xml = "".join(
        f"<Channel{_channel_xml_attrs(channels[i], i)}/>"
        for i in range(c)
    )
    description = escape(json.dumps(_jsonable(metadata.get("cideconvolve_processing") or {}), ensure_ascii=False))
    structured_annotations = _metadata_xml_annotation(metadata, shape_tczyx)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f"<Instrument ID=\"Instrument:0\"><Objective{''.join(objective_attrs)}/></Instrument>"
        f"<Image ID=\"Image:0\" Name=\"{_xml_escape_attr(image_name)}\">"
        f"<Description>{description}</Description>"
        '<InstrumentRef ID="Instrument:0"/>'
        f"<ObjectiveSettings{''.join(objective_settings_attrs)}/>"
        f"<Pixels DimensionOrder=\"XYZCT\" ID=\"Pixels:0\""
        f" PhysicalSizeX=\"{_positive_float(metadata.get('pixel_size_x'), 1.0)}\" PhysicalSizeXUnit=\"µm\""
        f" PhysicalSizeY=\"{_positive_float(metadata.get('pixel_size_y'), _positive_float(metadata.get('pixel_size_x'), 1.0))}\" PhysicalSizeYUnit=\"µm\""
        f" PhysicalSizeZ=\"{_positive_float(metadata.get('pixel_size_z'), 1.0)}\" PhysicalSizeZUnit=\"µm\""
        f" SizeC=\"{c}\" SizeT=\"{t}\" SizeX=\"{x}\" SizeY=\"{y}\" SizeZ=\"{z}\" Type=\"{_ome_dtype(dtype)}\">"
        f"{channel_xml}</Pixels></Image>{structured_annotations}</OME>"
    )
    (ome_dir / "METADATA.ome.xml").write_text(xml, encoding="utf-8")


def _axes_metadata(include_time: bool) -> list[dict[str, str]]:
    axes = []
    if include_time:
        axes.append({"name": "t", "type": "time"})
    axes.extend([
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ])
    return axes


def _scale_transform(px_z: float, px_y: float, px_x: float, include_time: bool) -> list[dict[str, Any]]:
    scale = [1.0, float(px_z), float(px_y), float(px_x)]
    if include_time:
        scale.insert(0, 1.0)
    return [{
        "type": "scale",
        "scale": scale,
    }]


def _channel_display_window(data_tczyx: np.ndarray, channel: int) -> tuple[float, float, float, float] | None:
    try:
        sample = np.asarray(data_tczyx[:, int(channel), :, :, :], dtype=np.float32)
        while sample.size > 2_000_000:
            axes = [idx for idx, size in enumerate(sample.shape) if size > 1]
            if not axes:
                break
            axis = max(axes, key=lambda idx: sample.shape[idx])
            slices = [slice(None)] * sample.ndim
            slices[axis] = slice(0, sample.shape[axis], 2)
            sample = sample[tuple(slices)]
    except Exception:
        return None
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return None
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if max_value <= min_value:
        end = max(max_value, min_value + 1.0)
        return min_value, end, min_value, end
    start = float(np.percentile(finite, 0.1))
    end = float(np.percentile(finite, 99.9))
    if end <= start:
        start, end = min_value, max_value
    if end <= start:
        end = start + 1.0
    return start, end, min_value, max_value


def _metadata_with_display_windows(metadata: dict[str, Any], data_tczyx: np.ndarray) -> dict[str, Any]:
    meta = dict(metadata)
    n_channels = int(data_tczyx.shape[1])
    channels = [
        dict(ch) if isinstance(ch, dict) else {}
        for ch in (meta.get("channels") or [])
    ]
    if len(channels) < n_channels:
        channels.extend({} for _ in range(n_channels - len(channels)))
    for i in range(n_channels):
        window = _channel_display_window(data_tczyx, i)
        if window is None:
            continue
        start, end, min_value, max_value = window
        channels[i]["window_start"] = start
        channels[i]["window_end"] = end
        channels[i]["window_min"] = min_value
        channels[i]["window_max"] = max_value
    meta["channels"] = channels[:n_channels]
    return meta


def _downsample_xy_mean(data: np.ndarray) -> np.ndarray:
    """Downsample TCZYX data by 2x in XY using block means."""
    y_even = data.shape[3] - (data.shape[3] % 2)
    x_even = data.shape[4] - (data.shape[4] % 2)
    cropped = data[:, :, :, :y_even, :x_even]
    return cropped.reshape(
        cropped.shape[0],
        cropped.shape[1],
        cropped.shape[2],
        y_even // 2,
        2,
        x_even // 2,
        2,
    ).mean(axis=(4, 6), dtype=np.float32)


def _pyramid_levels(data: np.ndarray, mode: str) -> list[np.ndarray]:
    mode = str(mode or "auto").lower()
    if mode in {"off", "none", "false", "0"}:
        return [data]
    if mode not in {"auto", "on", "true", "1"}:
        raise ValueError(f"Unknown OME-Zarr pyramid mode: {mode}")

    levels = [data]
    current = data
    should_continue = lambda arr: arr.shape[3] >= 512 and arr.shape[4] >= 512
    if mode == "auto" and max(data.shape[3], data.shape[4]) < 2048:
        return levels
    while len(levels) < 5 and should_continue(current):
        current = _downsample_xy_mean(current)
        if current.shape[3] < 1 or current.shape[4] < 1:
            break
        levels.append(current.astype(np.float32, copy=False))
    return levels


def _omero_metadata(metadata: dict[str, Any], n_channels: int) -> dict[str, Any]:
    from .metadata import (
        _metadata_description,
        _resolve_channel_display_colors,
        _rgb_to_ome_hex,
    )

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
        window_min = src.get("window_min")
        window_max = src.get("window_max")
        try:
            start = float(window_start) if window_start is not None else 0.0
            end = float(window_end) if window_end is not None else max(start, 1.0)
        except (TypeError, ValueError):
            start, end = 0.0, 1.0
        if end <= start:
            end = start + 1.0
        try:
            min_value = float(window_min) if window_min is not None else min(start, 0.0)
        except (TypeError, ValueError):
            min_value = min(start, 0.0)
        try:
            max_value = float(window_max) if window_max is not None else max(end, 1.0)
        except (TypeError, ValueError):
            max_value = max(end, 1.0)
        min_value = min(min_value, start)
        max_value = max(max_value, end)
        channels.append({
            "label": str(label),
            "color": color,
            "active": bool(src.get("active", True)),
            "coefficient": 1,
            "family": "linear",
            "inverted": False,
            "window": {"start": start, "end": end, "min": min_value, "max": max_value},
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
    pyramid: str = "auto",
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
    metadata = _metadata_with_display_windows(metadata, data)
    px_x = float(metadata.get("pixel_size_x") or 1.0)
    px_y = float(metadata.get("pixel_size_y") or px_x)
    px_z = float(metadata.get("pixel_size_z") or 1.0)
    full_levels = _pyramid_levels(data, pyramid)
    levels = [level_data[0] if level_data.shape[0] == 1 else level_data for level_data in full_levels]
    include_time = levels[0].ndim == 5
    axes = ["t", "c", "z", "y", "x"] if include_time else ["c", "z", "y", "x"]

    try:
        root = zarr.open_group(str(output_path), mode="w", zarr_format=2)
    except TypeError:
        root = zarr.open_group(str(output_path), mode="w", zarr_version=2)

    datasets = []
    for level, level_data in enumerate(levels):
        path = str(level)
        if include_time:
            chunks = (
                1,
                1,
                min(level_data.shape[2], 16),
                min(level_data.shape[3], 512),
                min(level_data.shape[4], 512),
            )
        else:
            chunks = (
                1,
                min(level_data.shape[1], 16),
                min(level_data.shape[2], 512),
                min(level_data.shape[3], 512),
            )
        create_kwargs = {
            "shape": level_data.shape,
            "chunks": chunks,
            "dtype": level_data.dtype,
        }
        try:
            array = root.create_dataset(path, dimension_separator="/", **create_kwargs)
        except TypeError:
            array = root.create_dataset(path, **create_kwargs)
        array[:] = level_data
        array.attrs["_ARRAY_DIMENSIONS"] = axes
        scale = 2 ** level
        datasets.append({
            "path": path,
            "coordinateTransformations": _scale_transform(px_z, px_y * scale, px_x * scale, include_time),
        })

    name = output_path.name
    display_metadata = dict(metadata)
    display_metadata["name"] = name
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": name,
        "axes": _axes_metadata(include_time),
        "datasets": datasets,
        "type": "mean",
        "metadata": {"method": "chunked 2x2 XY mean" if len(datasets) > 1 else "single-scale"},
    }]
    root.attrs["omero"] = _omero_metadata(display_metadata, data.shape[1])
    payload = _cideconvolve_metadata_payload(metadata, tuple(int(v) for v in data.shape))
    payload["streaming"] = False
    root.attrs["_creator"] = payload
    root.attrs["cideconvolve"] = payload
    _write_ome_xml_metadata(output_path, metadata, tuple(int(v) for v in data.shape), data.dtype, name)
    return output_path
