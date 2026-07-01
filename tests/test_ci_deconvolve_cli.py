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


def test_project_dependencies_do_not_declare_excluded_packages():
    text = (ROOT / "ci_deconvolve_cli" / "pyproject.toml").read_text(encoding="utf-8")
    lower = text.lower()

    assert '"torch' not in lower
    assert "bioio-ome-zarr" not in lower
    assert "pyqt6" not in lower


def test_save_result_ome_zarr_smoke(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("zarr")

    from core.ome_zarr_io import save_result_ome_zarr

    path = tmp_path / "out.ome.zarr"
    result = {
        "channels": [np.ones((1, 8, 9), dtype=np.float32)],
        "metadata": {
            "pixel_size_x": 0.12,
            "pixel_size_y": 0.12,
            "pixel_size_z": 0.5,
            "channel_names": ["CH1"],
            "channels": [{"name": "CH1", "color": (255, 0, 0)}],
        },
    }

    save_result_ome_zarr(result, path)

    assert (path / ".zgroup").is_file()
    assert (path / ".zattrs").is_file()
    assert not (path / "zarr.json").exists()

    import json

    attrs = json.loads((path / ".zattrs").read_text(encoding="utf-8"))
    assert attrs["multiscales"][0]["version"] == "0.4"
    assert attrs["multiscales"][0]["datasets"][0]["path"] == "0"
    assert attrs["omero"]["channels"][0]["label"] == "CH1"
    assert attrs["omero"]["channels"][0]["color"] == "FF0000"

    import zarr

    root = zarr.open(str(path), mode="r")
    assert tuple(root["0"].shape) == (1, 1, 1, 8, 9)
