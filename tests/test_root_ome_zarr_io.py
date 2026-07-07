from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_SRC = ROOT / "ci_deconvolve_cli" / "src"


def _same_path(value: str, path: Path) -> bool:
    try:
        return Path(value).resolve() == path.resolve()
    except Exception:
        return False


@contextmanager
def _root_ome_zarr_io():
    old_path = list(sys.path)
    old_core_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "core" or name.startswith("core.")
    }
    for name in list(old_core_modules):
        sys.modules.pop(name, None)

    sys.path[:] = [
        str(ROOT),
        *(entry for entry in old_path if not _same_path(entry, CLI_SRC)),
    ]
    try:
        yield importlib.import_module("core.ome_zarr_io")
    finally:
        for name in list(sys.modules):
            if name == "core" or name.startswith("core."):
                sys.modules.pop(name, None)
        sys.modules.update(old_core_modules)
        sys.path[:] = old_path


def test_root_result_to_tczyx_matches_cli_time_series_layout():
    np = pytest.importorskip("numpy")

    with _root_ome_zarr_io() as ome_zarr_io:
        data = ome_zarr_io._result_to_tczyx({
            "channels": [
                np.zeros((2, 3, 4, 5), dtype=np.float32),
                np.ones((2, 3, 4, 5), dtype=np.float32),
            ],
            "metadata": {"size_t": 2, "size_z": 3},
        })

    assert data.shape == (2, 2, 3, 4, 5)
    assert np.all(data[:, 0] == 0)
    assert np.all(data[:, 1] == 1)


def test_root_result_to_tczyx_preserves_2d_time_series_with_singleton_z():
    np = pytest.importorskip("numpy")

    with _root_ome_zarr_io() as ome_zarr_io:
        data = ome_zarr_io._result_to_tczyx({
            "channels": [np.zeros((2, 4, 5), dtype=np.float32)],
            "metadata": {"size_t": 2, "size_z": 1},
        })

    assert data.shape == (2, 1, 1, 4, 5)


def test_root_ome_zarr_pyramid_scales_xy_pixel_sizes(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")

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

    with _root_ome_zarr_io() as ome_zarr_io:
        ome_zarr_io.save_result_ome_zarr(result, output, pyramid="on")

    attrs = json.loads((output / ".zattrs").read_text(encoding="utf-8"))
    datasets = attrs["multiscales"][0]["datasets"]

    assert [dataset["path"] for dataset in datasets] == ["0", "1", "2"]
    assert datasets[0]["coordinateTransformations"][0]["scale"] == [1.0, 0.5, 0.2, 0.1]
    assert datasets[1]["coordinateTransformations"][0]["scale"] == [1.0, 0.5, 0.4, 0.2]
