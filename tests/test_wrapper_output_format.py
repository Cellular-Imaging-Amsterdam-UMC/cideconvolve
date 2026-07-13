from pathlib import Path

import numpy as np


def test_projected_wrapper_result_respects_ome_zarr_output_format(tmp_path, monkeypatch):
    import wrapper

    captured = {}

    def fake_save_zarr(result, path, *, output_dtype="float32"):
        captured["zarr"] = {"result": result, "path": Path(path), "dtype": output_dtype}
        Path(path).mkdir(parents=True)

    def fake_save_tiff(*args, **kwargs):
        raise AssertionError("TIFF writer should not be used for OME-Zarr output")

    monkeypatch.setattr(wrapper, "save_result_ome_zarr", fake_save_zarr)
    monkeypatch.setattr(wrapper, "save_result", fake_save_tiff)

    result = {
        "channels": [np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)],
        "source_channels": [np.ones((3, 4, 5), dtype=np.float32)],
        "metadata": {"size_z": 3},
    }

    out_name, out_path = wrapper._save_wrapper_result(
        result,
        tmp_path,
        "sample",
        projection="mip",
        output_format="ome-zarr",
        output_dtype="float32",
    )

    assert out_name == "sample_decon_mip-proj.ome.zarr"
    assert out_path == tmp_path / out_name
    assert captured["zarr"]["result"]["channels"][0].shape == (4, 5)
    assert captured["zarr"]["result"]["metadata"]["size_z"] == 1
    assert captured["zarr"]["result"]["metadata"]["projection"] == "mip"


def test_projected_wrapper_result_keeps_tiff_when_requested(tmp_path, monkeypatch):
    import wrapper

    captured = {}

    def fake_save_tiff(result, path, *, output_dtype="float32"):
        captured["tiff"] = {"result": result, "path": Path(path), "dtype": output_dtype}
        Path(path).write_bytes(b"tiff")

    def fake_save_zarr(*args, **kwargs):
        raise AssertionError("OME-Zarr writer should not be used for TIFF output")

    monkeypatch.setattr(wrapper, "save_result", fake_save_tiff)
    monkeypatch.setattr(wrapper, "save_result_ome_zarr", fake_save_zarr)

    result = {
        "channels": [np.ones((3, 4, 5), dtype=np.float32)],
        "metadata": {"size_z": 3},
    }

    out_name, out_path = wrapper._save_wrapper_result(
        result,
        tmp_path,
        "sample",
        projection="sum",
        output_format="ome-tiff",
        output_dtype="uint16",
    )

    assert out_name == "sample_decon_sum-proj.ome.tiff"
    assert out_path == tmp_path / out_name
    assert captured["tiff"]["result"]["channels"][0].shape == (4, 5)
    assert captured["tiff"]["result"]["metadata"]["projection"] == "sum"
    assert captured["tiff"]["dtype"] == "uint16"
