from pathlib import Path

import numpy as np
import pytest


def _write_sidecar(path: Path) -> None:
    ome_dir = path / "OME"
    ome_dir.mkdir(parents=True, exist_ok=True)
    (ome_dir / "METADATA.ome.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Instrument ID="Instrument:0">
    <Objective ID="Objective:0" LensNA="1.25" NominalMagnification="63" Immersion="Oil"/>
  </Instrument>
  <Image ID="Image:0" Name="slurm-input">
    <ObjectiveSettings ID="Objective:0" RefractiveIndex="1.515"/>
    <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
            SizeX="5" SizeY="4" SizeZ="3" SizeC="2" SizeT="1"
            PhysicalSizeX="0.066813" PhysicalSizeY="0.066813" PhysicalSizeZ="0.2">
      <Channel ID="Channel:0" Name="676.0" ExcitationWavelength="621" EmissionWavelength="676" PinholeSize="62.72"/>
      <Channel ID="Channel:1" Name="435.0" ExcitationWavelength="400" EmissionWavelength="435" PinholeSize="62.72"/>
    </Pixels>
  </Image>
</OME>""",
        encoding="utf-8",
    )


def test_zarr_region_source_reads_ome_xml_sidecar_wavelengths(tmp_path):
    zarr = pytest.importorskip("zarr")
    from cideconvolve_io.streaming import ZarrRegionSource

    path = tmp_path / "slurm_input.ome.zarr"
    root = zarr.open(str(path), mode="w", zarr_format=2)
    root.create_array(
        "0",
        data=np.zeros((1, 2, 3, 4, 5), dtype=np.uint16),
        chunks=(1, 1, 3, 4, 5),
    )
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 0.2, 0.066813, 0.066813]}],
        }],
    }]
    root.attrs["omero"] = {"channels": [
        {"label": "fallback label 0", "color": "FF0000", "window": {"min": 0, "max": 1, "start": 0, "end": 1}},
        {"label": "fallback label 1", "color": "00FF00", "window": {"min": 0, "max": 1, "start": 0, "end": 1}},
    ]}
    _write_sidecar(path)

    source = ZarrRegionSource(path)

    assert source.metadata["na"] == pytest.approx(1.25)
    assert source.metadata["pixel_size_x"] == pytest.approx(0.066813)
    assert source.metadata["channel_names"] == ["676.0", "435.0"]
    assert source.metadata["channels"][0]["emission_wavelength"] == pytest.approx(676.0)
    assert source.metadata["channels"][0]["excitation_wavelength"] == pytest.approx(621.0)
    assert source.metadata["channels"][1]["emission_wavelength"] == pytest.approx(435.0)
    assert source.metadata["channels"][1]["excitation_wavelength"] == pytest.approx(400.0)
    assert source.metadata["channels"][0]["color"] == (255, 0, 0)


def test_core_zarr_sidecar_merge_preserves_rendering_fields(tmp_path):
    from core.deconvolve import _merge_ome_zarr_sidecar_metadata

    path = tmp_path / "slurm_input.ome.zarr"
    path.mkdir()
    _write_sidecar(path)
    meta = {
        "channel_names": ["old"],
        "channels": [{"color": (1, 2, 3), "window_start": 10}],
    }

    _merge_ome_zarr_sidecar_metadata(meta, path)

    assert meta["channel_names"] == ["676.0", "435.0"]
    assert meta["channels"][0]["color"] == (1, 2, 3)
    assert meta["channels"][0]["window_start"] == 10
    assert meta["channels"][0]["emission_wavelength"] == pytest.approx(676.0)
    assert meta["channels"][1]["excitation_wavelength"] == pytest.approx(400.0)
