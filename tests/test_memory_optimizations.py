import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.deconvolve import generate_psf
from core.deconvolve_ci import (
    _ci_deconvolve_tiled,
    _pick_dtype,
    ci_generate_psf,
    ci_rl_deconvolve,
    ci_sparse_hessian_deconvolve,
)


def _identity_psf_2d() -> np.ndarray:
    psf = np.zeros((1, 1), dtype=np.float32)
    psf[0, 0] = 1.0
    return psf


def test_ci_compute_defaults_to_float32_on_cpu(monkeypatch) -> None:
    monkeypatch.delenv("CIDE_CONVOLVE_CPU_FLOAT64", raising=False)
    assert _pick_dtype(torch.device("cpu")) == torch.float32


def test_ci_generate_psf_returns_float32() -> None:
    psf = ci_generate_psf(
        na=0.8,
        wavelength_nm=520.0,
        pixel_size_xy_nm=120.0,
        pixel_size_z_nm=300.0,
        n_xy=9,
        n_z=3,
        n_pupil=9,
        integrate_pixels=False,
        device="cpu",
    )

    assert psf.dtype == np.float32
    assert psf.shape == (3, 9, 9)
    np.testing.assert_allclose(psf.sum(), 1.0, rtol=1e-5, atol=1e-6)


def test_generate_psf_returns_float32() -> None:
    metadata = {
        "na": 0.8,
        "refractive_index": 1.33,
        "sample_refractive_index": 1.33,
        "pixel_size_x": 0.12,
        "pixel_size_z": 0.3,
        "size_z": 1,
        "microscope_type": "widefield",
        "channels": [{"emission_wavelength": 520.0}],
    }

    psf = generate_psf(
        metadata,
        psf_size_xy=9,
        n_pix_pupil=9,
        two_d_mode="legacy_2d",
    )

    assert psf.dtype == np.float32
    assert psf.shape == (9, 9)


def test_ci_solvers_return_float32() -> None:
    image = np.full((9, 9), 3.0, dtype=np.float32)
    psf = _identity_psf_2d()

    rl = ci_rl_deconvolve(
        image,
        psf,
        niter=1,
        background=1e-6,
        offset=0.0,
        start="observed",
        convergence="fixed",
        device="cpu",
        tiling="none",
        two_d_mode="legacy_2d",
    )
    sparse = ci_sparse_hessian_deconvolve(
        image,
        psf,
        niter=1,
        background=1e-6,
        offset=0.0,
        start="observed",
        convergence="fixed",
        device="cpu",
        tiling="none",
    )

    assert rl["result"].dtype == np.float32
    assert sparse["result"].dtype == np.float32


def test_release_cache_called_after_solver(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_release() -> None:
        calls["count"] += 1

    monkeypatch.setattr("core.deconvolve_ci._release_cuda_cache", fake_release)

    ci_rl_deconvolve(
        np.ones((5, 5), dtype=np.float32),
        _identity_psf_2d(),
        niter=1,
        background=1e-6,
        offset=0.0,
        start="observed",
        convergence="fixed",
        device="cpu",
        tiling="none",
        two_d_mode="legacy_2d",
    )

    assert calls["count"] >= 1


def test_release_cache_called_after_each_internal_tile(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_release() -> None:
        calls["count"] += 1

    def solver(tile_img: np.ndarray, _psf: np.ndarray, **_kwargs):
        return {
            "result": np.asarray(tile_img, dtype=np.float32),
            "convergence": [],
            "iterations_used": 1,
        }

    monkeypatch.setattr("core.deconvolve_ci._release_cuda_cache", fake_release)

    _ci_deconvolve_tiled(
        np.ones((1, 12, 12), dtype=np.float32),
        np.ones((1, 1, 1), dtype=np.float32),
        n_tiles=4,
        solver=solver,
    )

    assert calls["count"] >= 4
