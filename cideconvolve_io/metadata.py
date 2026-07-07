"""Shared metadata helpers for CIDeconvolve image writers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _coerce_rgb(color: Any) -> tuple[int, int, int] | None:
    if isinstance(color, str):
        text = color.strip().lstrip("#")
        if len(text) == 6:
            try:
                return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                return None
        return None
    if isinstance(color, Sequence) and not isinstance(color, (bytes, bytearray)):
        vals = list(color)
        if len(vals) >= 3:
            try:
                return tuple(max(0, min(255, int(v))) for v in vals[:3])  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None
    return None


def _rgb_to_ome_hex(color: Any) -> str | None:
    rgb = _coerce_rgb(color)
    if rgb is None:
        return None
    return "".join(f"{v:02X}" for v in rgb)


def _rgb_to_ome_int(color: Any) -> int | None:
    rgb = _coerce_rgb(color)
    if rgb is None:
        return None
    r, g, b = rgb
    return int((r << 24) | (g << 16) | (b << 8) | 255)


_FALLBACK_COLORS = [
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
]

_BGRCYM = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
]


def _emission_to_rgb(wavelength_nm: Any) -> tuple[int, int, int]:
    try:
        wl = float(wavelength_nm)
    except (TypeError, ValueError):
        return (255, 255, 255)
    r = g = b = 0.0
    if 380 <= wl < 440:
        r = -(wl - 440) / 60.0
        b = 1.0
    elif 440 <= wl < 490:
        g = (wl - 440) / 50.0
        b = 1.0
    elif 490 <= wl < 510:
        g = 1.0
        b = -(wl - 510) / 20.0
    elif 510 <= wl < 580:
        r = (wl - 510) / 70.0
        g = 1.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / 65.0
    elif 645 <= wl <= 780:
        r = 1.0
    else:
        return (255, 255, 255)
    return (int(r * 255), int(g * 255), int(b * 255))


def _channel_display_color(metadata: dict[str, Any], channel: int) -> tuple[int, int, int]:
    channels = metadata.get("channels", [])
    ch = channels[channel] if channel < len(channels) and isinstance(channels[channel], dict) else {}
    rgb = _coerce_rgb(ch.get("color"))
    if rgb is not None and rgb != (255, 255, 255):
        return rgb
    rgb = _emission_to_rgb(ch.get("emission_wavelength"))
    if rgb == (255, 255, 255):
        rgb = _FALLBACK_COLORS[channel % len(_FALLBACK_COLORS)]
    return rgb


def _resolve_channel_display_colors(metadata: dict[str, Any], count: int) -> list[tuple[int, int, int]]:
    colors = [_channel_display_color(metadata, i) for i in range(count)]
    if count > 1 and len(set(colors)) == 1:
        colors = [_BGRCYM[i % len(_BGRCYM)] for i in range(count)]
    return colors


def _positive_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out) or out <= 0:
        return float(default)
    return out


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


def _metadata_warnings(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("metadata_warnings")
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    return []


def _metadata_description(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    if metadata.get("na") is not None:
        parts.append(f"NA={metadata['na']}")
    if metadata.get("refractive_index") is not None:
        parts.append(f"RI={metadata['refractive_index']}")
    if metadata.get("sample_refractive_index") is not None:
        parts.append(f"SampleRI={metadata['sample_refractive_index']}")
    if metadata.get("microscope_type"):
        parts.append(f"Microscope={metadata['microscope_type']}")
    defaulted = sorted(str(key) for key in (metadata.get("_defaulted_keys") or []))
    if defaulted:
        parts.append("CIDeconvolve metadata defaults: " + ", ".join(defaulted))
    warnings = _metadata_warnings(metadata)
    if warnings:
        parts.append("CIDeconvolve metadata warnings: " + " | ".join(warnings))
    return "; ".join(parts)


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

