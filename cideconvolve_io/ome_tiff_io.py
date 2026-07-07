"""Shared OME-TIFF writing helpers for CIDeconvolve outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .streaming import TiledOmeTiffSink


def _channel_names(metadata: dict[str, Any], count: int) -> list[str]:
    names = metadata.get("channel_names") or []
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes, bytearray)):
        names = []
    out = [str(name) for name in names[:count]]
    out.extend(f"Ch{i}" for i in range(len(out), count))
    return out


def write_tczyx_ome_tiff(
    data: np.ndarray,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    levels: int = 1,
    compression: str | None = "lzw",
) -> Path:
    """Write a TCZYX float image through the shared tiled OME-TIFF sink."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 5:
        raise ValueError(f"Expected TCZYX data for OME-TIFF export, got {arr.shape}")

    meta = dict(metadata or {})
    meta.setdefault("channel_names", _channel_names(meta, arr.shape[1]))
    out_path = Path(path)
    sink = TiledOmeTiffSink(
        out_path,
        shape=tuple(int(v) for v in arr.shape),
        metadata=meta,
        levels=int(levels),
        compression=compression,
    )
    try:
        for t in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                sink.write_tile(
                    t=t,
                    c=c,
                    z=slice(0, arr.shape[2]),
                    y=slice(0, arr.shape[3]),
                    x=slice(0, arr.shape[4]),
                    data=arr[t, c],
                )
        sink.build_pyramids()
        sink.validate()
        sink.close()
    except Exception:
        sink.abort()
        raise
    return out_path

