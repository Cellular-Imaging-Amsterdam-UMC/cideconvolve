import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI_SRC = ROOT / "src"
if str(CLI_SRC) not in sys.path:
    # The wheel maps ``core`` to the repository's single shared implementation.
    sys.path.append(str(CLI_SRC))


def test_cli_uses_single_shared_core_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"core" = "../core"' in pyproject
    assert not list((ROOT / "src" / "core").glob("*.py"))

from ci_deconvolve import __version__
from ci_deconvolve.cli import _apply_projection, _write_manifest
from core.ome_zarr_io import _result_to_tczyx, save_result_ome_zarr


def test_version_matches_cli_contract():
    assert __version__ == (ROOT / "version.txt").read_text(encoding="utf-8").strip()


def test_result_to_tczyx_preserves_time_series_3d():
    result = {
        "channels": [
            np.zeros((2, 3, 4, 5), dtype=np.float32),
            np.ones((2, 3, 4, 5), dtype=np.float32),
        ],
        "metadata": {"size_t": 2, "size_z": 3},
    }

    data = _result_to_tczyx(result)

    assert data.shape == (2, 2, 3, 4, 5)
    assert np.all(data[:, 0] == 0)
    assert np.all(data[:, 1] == 1)


def test_result_to_tczyx_preserves_time_series_2d_with_singleton_z():
    result = {
        "channels": [np.zeros((2, 4, 5), dtype=np.float32)],
        "metadata": {"size_t": 2, "size_z": 1},
    }

    data = _result_to_tczyx(result)

    assert data.shape == (2, 1, 1, 4, 5)


def test_max_z_projection_handles_time_series_3d():
    channel = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    result = {
        "channels": [channel],
        "source_channels": [channel],
        "metadata": {"size_t": 2, "size_z": 3},
    }

    projected = _apply_projection(result, "max-z")

    assert projected["channels"][0].shape == (2, 4, 5)
    assert projected["source_channels"][0].shape == (2, 4, 5)
    assert projected["metadata"]["size_z"] == 1


def test_write_manifest(tmp_path):
    args = Namespace(output_format="ome-zarr", projection="none", iterations=[4, 8])
    _write_manifest(tmp_path, [{"input": "in.ome.tiff", "status": "success"}], args)

    manifest = json.loads((tmp_path / "ci_deconvolve_manifest.json").read_text())

    assert manifest["ci_deconvolve_version"] == (ROOT / "version.txt").read_text(encoding="utf-8").strip()
    assert manifest["output_format"] == "ome-zarr"
    assert manifest["records"][0]["status"] == "success"


def test_ome_zarr_pyramid_scales_xy_pixel_sizes(tmp_path):
    output = tmp_path / "out.ome.zarr"
    result = {
        "channels": [np.zeros((1024, 1024), dtype=np.float32)],
        "metadata": {
            "name": "pyramid",
            "pixel_size_x": 0.1,
            "pixel_size_y": 0.2,
            "pixel_size_z": 0.5,
        },
    }

    save_result_ome_zarr(result, output, pyramid="on")
    attrs = json.loads((output / ".zattrs").read_text())
    datasets = attrs["multiscales"][0]["datasets"]

    assert [dataset["path"] for dataset in datasets] == ["0", "1", "2"]
    assert datasets[0]["coordinateTransformations"][0]["scale"] == [1.0, 0.5, 0.2, 0.1]
    assert datasets[1]["coordinateTransformations"][0]["scale"] == [1.0, 0.5, 0.4, 0.2]
