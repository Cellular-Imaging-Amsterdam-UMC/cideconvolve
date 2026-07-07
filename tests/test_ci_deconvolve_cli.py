from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_SRC = ROOT / "ci_deconvolve_cli" / "src"
if str(CLI_SRC) not in sys.path:
    sys.path.insert(0, str(CLI_SRC))

from ci_deconvolve import cli  # noqa: E402


def test_cli_help_smoke(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.run(["--help"])

    assert exc.value.code == 0
    assert "ci_deconvolve" in capsys.readouterr().out


def test_discover_single_ome_tiff(tmp_path):
    path = tmp_path / "image.ome.tiff"
    path.write_bytes(b"")

    assert cli._discover_inputs(path) == [path]


def test_discover_single_ome_zarr(tmp_path):
    path = tmp_path / "image.ome.zarr"
    path.mkdir()

    assert cli._discover_inputs(path) == [path]


def test_discover_folder_filters_to_supported_children(tmp_path):
    supported_tiff = tmp_path / "a.ome.tiff"
    supported_tiff.write_bytes(b"")
    supported_zarr = tmp_path / "b.ome.zarr"
    supported_zarr.mkdir()
    (tmp_path / "plain.tif").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert cli._discover_inputs(tmp_path) == [supported_tiff, supported_zarr]


def test_discover_rejects_unsupported_file(tmp_path):
    path = tmp_path / "plain.tif"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported input"):
        cli._discover_inputs(path)


def test_output_name_uses_decon_stem(tmp_path):
    assert cli._output_path(tmp_path / "stack.ome.tiff", tmp_path, "ome-tiff").name == (
        "stack_decon.ome.tiff"
    )
    assert cli._output_path(tmp_path / "stack.ome.zarr", tmp_path, "ome-zarr").name == (
        "stack_decon.ome.zarr"
    )


def test_projection_max_z_only_changes_3d_channels():
    np = pytest.importorskip("numpy")

    result = {
        "channels": [np.arange(24, dtype=np.float32).reshape(2, 3, 4)],
        "source_channels": [np.arange(24, dtype=np.float32).reshape(2, 3, 4)],
        "metadata": {"size_z": 2},
    }

    projected = cli._apply_projection(result, "max-z")

    assert projected is not result
    assert projected["channels"][0].shape == (3, 4)
    assert projected["channels"][0][0, 0] == 12
    assert projected["source_channels"][0].shape == (3, 4)
    assert projected["metadata"]["size_z"] == 1
    assert projected["metadata"]["projection"] == {"axis": "z", "method": "max"}

    two_d = {"channels": [np.ones((3, 4), dtype=np.float32)], "metadata": {}}
    assert cli._apply_projection(two_d, "max-z") is two_d


def test_project_dependencies_do_not_declare_excluded_packages():
    text = (ROOT / "ci_deconvolve_cli" / "pyproject.toml").read_text(encoding="utf-8")
    lower = text.lower()

    assert '"torch' not in lower
    assert "bioio-ome-zarr" not in lower
    assert "pyqt6" not in lower


def test_load_image_tracks_metadata_provenance(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    tifffile = pytest.importorskip("tifffile")

    from core.deconvolve import load_image

    src = tmp_path / "meta.ome.tiff"
    ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Instrument ID="Instrument:0">
    <Objective ID="Objective:0" Immersion="Oil" LensNA="1.25"/>
  </Instrument>
  <Image ID="Image:0">
    <ObjectiveSettings ID="Objective:0" RefractiveIndex="1.515"/>
    <Pixels ID="Pixels:0" DimensionOrder="XYZTC" Type="uint16"
            SizeX="5" SizeY="4" SizeZ="1" SizeC="1" SizeT="1"
            PhysicalSizeX="0.066813" PhysicalSizeY="0.066813">
      <Channel ID="Channel:0" EmissionWavelength="676"/>
    </Pixels>
  </Image>
</OME>"""
    tifffile.imwrite(src, np.ones((4, 5), dtype=np.uint16), description=ome_xml)

    loaded = load_image(
        src,
        overrule_metadata=False,
        na=1.4,
        pixel_size_xy=0.5,
        pixel_size_z=0.7,
        excitation_wavelengths=[488.0],
        pinhole_airy_units=[1.2],
    )
    metadata = loaded["metadata"]
    provenance = metadata["_metadata_provenance"]

    assert metadata["pixel_size_x"] == pytest.approx(0.066813)
    assert metadata["pixel_size_z"] == pytest.approx(0.7)
    assert provenance["fields"]["pixel_size_x"] == "image metadata"
    assert provenance["fields"]["pixel_size_z"] == "user setting fallback"
    assert provenance["fields"]["na"] == "image metadata"
    assert provenance["channels"][0]["emission_wavelength"] == "image metadata"
    assert provenance["channels"][0]["excitation_wavelength"] == "user setting fallback"
    assert provenance["channels"][0]["pinhole_airy_units"] == "user setting fallback"


def test_metadata_report_prints_sources(capsys):
    cli._print_metadata_report({
        "pixel_size_x": 0.12,
        "pixel_size_y": 0.13,
        "pixel_size_z": 0.5,
        "na": 1.4,
        "refractive_index": 1.515,
        "sample_refractive_index": 1.47,
        "microscope_type": "confocal",
        "channels": [{
            "name": "CH1",
            "emission_wavelength": 520.0,
            "excitation_wavelength": 488.0,
            "pinhole_airy_units": 1.0,
        }],
        "_metadata_provenance": {
            "fields": {
                "pixel_size_x": "image metadata",
                "pixel_size_y": "image metadata",
                "pixel_size_z": "user setting fallback",
                "na": "image metadata",
            },
            "channels": [{
                "emission_wavelength": "image metadata",
                "excitation_wavelength": "user setting fallback",
                "pinhole_airy_units": "built-in default",
            }],
        },
    })

    out = capsys.readouterr().out
    assert "metadata:" in out
    assert "pixel size X" in out
    assert "(image metadata)" in out
    assert "(user setting fallback)" in out
    assert "channel 0 (CH1)" in out


def test_save_result_ome_zarr_smoke(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")

    from core.ome_zarr_io import save_result_ome_zarr

    path = tmp_path / "out.ome.zarr"
    result = {
        "channels": [np.arange(72, dtype=np.float32).reshape(1, 8, 9)],
        "metadata": {
            "pixel_size_x": 0.12,
            "pixel_size_y": 0.12,
            "pixel_size_z": 0.5,
            "channel_names": ["CH1"],
            "channels": [{"name": "CH1", "color": (255, 0, 0)}],
            "_defaulted_keys": {"pixel_size_x", "pixel_size_z"},
            "source_path": Path("input.ome.tiff"),
        },
    }

    save_result_ome_zarr(result, path)

    assert (path / ".zgroup").is_file()
    assert (path / ".zattrs").is_file()
    assert (path / "OME" / ".zgroup").is_file()
    assert (path / "OME" / "METADATA.ome.xml").is_file()
    assert not (path / "zarr.json").exists()

    import json

    attrs = json.loads((path / ".zattrs").read_text(encoding="utf-8"))
    assert attrs["multiscales"][0]["version"] == "0.4"
    assert attrs["multiscales"][0]["datasets"][0]["path"] == "0"
    assert attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [
        1.0,
        0.5,
        0.12,
        0.12,
    ]
    assert attrs["omero"]["channels"][0]["label"] == "CH1"
    assert attrs["omero"]["channels"][0]["color"] == "FF0000"
    assert attrs["omero"]["channels"][0]["window"]["min"] == 0.0
    assert attrs["omero"]["channels"][0]["window"]["max"] == 71.0
    assert attrs["omero"]["channels"][0]["window"]["end"] > 1.0
    assert attrs["_creator"]["physical_pixel_sizes_um"] == {"x": 0.12, "y": 0.12, "z": 0.5}
    assert attrs["cideconvolve"]["streaming"] is False

    array_attrs = json.loads((path / "0" / ".zattrs").read_text(encoding="utf-8"))
    assert array_attrs["_ARRAY_DIMENSIONS"] == ["c", "z", "y", "x"]
    assert attrs["cideconvolve"]["metadata"]["_defaulted_keys"] == [
        "pixel_size_x",
        "pixel_size_z",
    ]
    assert attrs["cideconvolve"]["metadata"]["source_path"] == "input.ome.tiff"

    import zarr

    root = zarr.open(str(path), mode="r")
    assert tuple(root["0"].shape) == (1, 1, 8, 9)

    ome_xml = (path / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")
    assert 'PhysicalSizeX="0.12"' in ome_xml
    assert 'PhysicalSizeZ="0.5"' in ome_xml
    assert 'EmissionWavelength' not in ome_xml
    assert 'CommentAnnotation ID="Annotation:CIDeconvolve:0"' in ome_xml
    assert "source_path" in ome_xml
