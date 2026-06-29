from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
tifffile = pytest.importorskip("tifffile")

from core.deconvolve import load_image, save_result  # noqa: E402
from core.streaming import ZarrPyramidSink  # noqa: E402


def test_missing_metadata_defaults_are_written_to_ome_description(tmp_path):
    src = tmp_path / "missing_meta.tif"
    tifffile.imwrite(src, np.ones((8, 9), dtype=np.uint16))

    loaded = load_image(src, overrule_metadata=False)
    metadata = loaded["metadata"]

    assert metadata["pixel_size_x"] == pytest.approx(0.065)
    assert metadata["na"] == pytest.approx(1.4)
    assert "metadata_warnings" in metadata
    assert "pixel_size_x" in metadata["_defaulted_keys"]

    out = tmp_path / "out.ome.tiff"
    save_result(
        {
            "channels": [np.ones((8, 9), dtype=np.float32)],
            "metadata": metadata,
        },
        out,
    )

    with tifffile.TiffFile(out) as tif:
        ome_xml = tif.ome_metadata or ""
    assert "CIDeconvolve metadata defaults" in ome_xml
    assert "CIDeconvolve metadata warnings" in ome_xml


def test_incomplete_excitation_wavelengths_do_not_break_ome_write(tmp_path):
    out = tmp_path / "incomplete_ex.ome.tiff"
    metadata = {
        "pixel_size_x": 0.065,
        "pixel_size_y": 0.065,
        "pixel_size_z": 0.2,
        "na": 1.4,
        "refractive_index": 1.515,
        "sample_refractive_index": 1.47,
        "microscope_type": "widefield",
        "channels": [
            {"emission_wavelength": 520.0, "excitation_wavelength": 488.0},
            {"emission_wavelength": 520.0},
            {"emission_wavelength": 520.0},
        ],
    }

    save_result(
        {
            "channels": [np.ones((4, 5), dtype=np.float32) for _ in range(3)],
            "metadata": metadata,
        },
        out,
    )

    with tifffile.TiffFile(out) as tif:
        ome_xml = tif.ome_metadata or ""
    assert "ExcitationWavelength" not in ome_xml
    assert "Incomplete excitation wavelength metadata" in ome_xml


def test_streaming_zarr_omero_description_contains_metadata_warnings(tmp_path):
    pytest.importorskip("zarr")
    metadata = {
        "pixel_size_x": "not-a-number",
        "pixel_size_y": None,
        "pixel_size_z": -1,
        "na": None,
        "channels": [{"emission_wavelength": "bad"}],
    }
    sink = ZarrPyramidSink(
        tmp_path / "out.ome.zarr",
        shape=(1, 1, 1, 8, 9),
        metadata=metadata,
        levels=1,
    )
    sink.write_tile(
        t=0,
        c=0,
        z=slice(0, 1),
        y=slice(0, 8),
        x=slice(0, 9),
        data=np.ones((1, 8, 9), dtype=np.float32),
    )
    sink.validate()
    sink.close()

    import zarr

    root = zarr.open(str(tmp_path / "out.ome.zarr"), mode="r")
    description = root.attrs["omero"]["description"]
    assert "CIDeconvolve metadata defaults" in description
    assert "CIDeconvolve metadata warnings" in description
    assert root.attrs["cideconvolve"]["metadata"]["pixel_size_x"] == 0.065
