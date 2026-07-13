from pathlib import Path

import pytest

from tools.omero_import_metadata_probe.omero_import_metadata_probe import (
    StackConfig,
    _compare_source_omero_to_zarr,
    _masked_env,
    _map_host_path_to_container,
    _parse_ome_xml_summary,
    _parse_target,
    _read_env_file,
)


def test_parse_target_accepts_dataset_and_screen():
    assert _parse_target("Dataset:123") == ("Dataset", 123)
    assert _parse_target("screen:456") == ("Screen", 456)


def test_parse_target_rejects_unknown_format():
    with pytest.raises(ValueError):
        _parse_target("Image:1")


def test_read_env_file_masks_and_strips_inline_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OMERO_ROOT_PASSWORD=omero # TODO\n"
        "OMERO_PORT=4064\n"
        "# COMMENTED=yes\n",
        encoding="utf-8",
    )
    env = _read_env_file(env_file)
    assert env["OMERO_ROOT_PASSWORD"] == "omero"
    assert env["OMERO_PORT"] == "4064"
    assert _masked_env(env)["OMERO_ROOT_PASSWORD"] == "******"


def test_map_host_path_to_container_uses_biomero_mount(tmp_path):
    root = tmp_path / "nl"
    data = root / "data"
    sample = data / "folder" / "image.ome.tif"
    sample.parent.mkdir(parents=True)
    sample.write_text("x", encoding="utf-8")
    stack = StackConfig(
        root=root,
        compose_file=root / "docker-compose.yml",
        env_file=root / ".env",
        env={
            "INPLACE_STORAGE_HOST_PATH": "./data",
            "IMPORT_MOUNT_PATH": "/data",
        },
    )
    assert _map_host_path_to_container(sample, stack) == "/data/folder/image.ome.tif"


def test_parse_ome_xml_summary_extracts_pixels_and_channels():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Image ID="Image:0" Name="probe">
        <Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
                SizeX="8" SizeY="7" SizeZ="6" SizeC="2" SizeT="1"
                PhysicalSizeX="0.1" PhysicalSizeY="0.2" PhysicalSizeZ="0.3">
          <Channel ID="Channel:0" Name="a" EmissionWavelength="520"/>
          <Channel ID="Channel:1" Name="b" EmissionWavelength="600"/>
        </Pixels>
      </Image>
    </OME>"""
    summary = _parse_ome_xml_summary(xml)
    assert summary["image_name"] == "probe"
    assert summary["pixels"]["SizeX"] == "8"
    assert [ch["Name"] for ch in summary["channels"]] == ["a", "b"]


def test_compare_source_omero_to_slurm_zarr_flags_metadata_changes():
    source = {
        "pixels": {
            "size_t": 1,
            "size_c": 2,
            "size_z": 3,
            "size_y": 4,
            "size_x": 5,
            "physical_size_x": {"value": 0.1},
        },
        "channels": [{"name": "DAPI"}, {"name": "FITC"}],
    }
    zarr_info = {
        "arrays": {"0": {"shape": (1, 2, 3, 4, 5)}},
        "attrs": {"omero": {"channels": [{"label": "DAPI"}, {"label": "GFP"}]}},
        "ome_xml_summary": {"pixels": {"PhysicalSizeX": "0.1"}},
    }
    comparison = _compare_source_omero_to_zarr(source, zarr_info)
    assert comparison["SizeX"]["status"] == "matched"
    assert comparison["Channel1Name"]["status"] == "matched"
    assert comparison["Channel2Name"]["status"] == "changed"
    assert comparison["PhysicalSizeX"]["status"] == "matched"
