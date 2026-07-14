from __future__ import annotations

from pathlib import Path

import pytest

from core import deconvolve_ci


def test_backend_cpu_forces_cpu_without_loading_extension():
    device, ops, selected = deconvolve_ci._resolve_backend("cpu", "cuda")

    assert device.type == "cpu"
    assert ops is None
    assert selected == "cpu"


def test_backend_alias_cuda_means_pytorch_cuda(monkeypatch):
    monkeypatch.setattr(deconvolve_ci.torch.cuda, "is_available", lambda: True)

    device, ops, selected = deconvolve_ci._resolve_backend("cuda", None)

    assert device.type == "cuda"
    assert ops is None
    assert selected == "pytorch_cuda"


def test_smooth_work_dimensions_use_only_small_prime_factors():
    assert deconvolve_ci._next_smooth(113) == 120
    assert deconvolve_ci._next_smooth(430) == 432


def test_gui_declares_requested_backend_choices():
    root = Path(__file__).resolve().parents[1]
    source = (root / "gui" / "gui_deconvolve_ci.py").read_text(encoding="utf-8")

    for label in (
        "Auto",
        "Optimized CUDA",
        "PyTorch CUDA",
        "CPU",
    ):
        assert f'"{label}"' in source
    assert 'aml.addRow("Backend:", self._device_combo)' in source


def test_cli_wheel_declares_optimized_cuda_sources():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "ci_deconvolve_cli" / "pyproject.toml").read_text(encoding="utf-8")

    assert '"core.optimized_cuda"' in pyproject
    assert '"core.optimized_cuda.cuda" = ["*.cpp", "*.cu"]' in pyproject


def test_explicit_cuda_backends_reject_missing_cuda(monkeypatch):
    monkeypatch.setattr(deconvolve_ci.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires a CUDA-capable"):
        deconvolve_ci._resolve_backend("optimized_cuda", None)
