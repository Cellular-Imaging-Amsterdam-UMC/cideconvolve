"""Compatibility shim for shared CIDeconvolve streaming I/O helpers."""

from __future__ import annotations

from cideconvolve_io import streaming as _impl

globals().update({
    name: getattr(_impl, name)
    for name in dir(_impl)
    if not name.startswith("__")
})

__all__ = [name for name in globals() if not name.startswith("__")]

