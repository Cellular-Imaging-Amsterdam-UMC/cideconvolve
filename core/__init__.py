# core package — CI deconvolution engine
from .deconvolve import (
    deconvolve,
    deconvolve_image,
    generate_psf,
    load_image,
    save_mip_png,
    save_result,
    _DEFAULT_PINHOLE_AIRY_UNITS,
    _apply_pinhole_airy_units,
)

_STREAMING_EXPORTS = {
    "InMemoryPyramidSink",
    "InMemoryRegionSource",
    "ZarrPyramidSink",
    "deconvolve_streaming",
    "open_region_source",
    "should_stream_source",
}


def __getattr__(name):
    if name in _STREAMING_EXPORTS:
        from . import streaming as _streaming

        return getattr(_streaming, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "deconvolve",
    "deconvolve_image",
    "generate_psf",
    "load_image",
    "save_mip_png",
    "save_result",
    "_DEFAULT_PINHOLE_AIRY_UNITS",
    "_apply_pinhole_airy_units",
    "InMemoryPyramidSink",
    "InMemoryRegionSource",
    "ZarrPyramidSink",
    "deconvolve_streaming",
    "open_region_source",
    "should_stream_source",
]
