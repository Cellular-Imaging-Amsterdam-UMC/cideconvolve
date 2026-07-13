import numpy as np
import pytest

from core.deconvolve_ci import (
    _resolve_snr_settings,
    _scale_aware_auto_offset,
    _snr_prefilter_sigma,
    ci_rl_deconvolve,
    estimate_image_snr,
)


def test_sparse_photon_snr_does_not_collapse_on_zero_background() -> None:
    rng = np.random.default_rng(4)
    image = np.zeros((16, 64, 64), dtype=np.float32)
    selected = rng.choice(image.size, size=2500, replace=False)
    image.flat[selected] = rng.poisson(2.0, size=selected.size).astype(np.float32)

    estimate = estimate_image_snr(image)

    assert estimate["mode"] == "photon-count"
    assert estimate["reliability"] in {"medium", "high"}
    assert estimate["intensity_step"] == pytest.approx(1.0)
    assert 1.0 <= estimate["snr"] < 20.0


def test_continuous_snr_fallback_is_finite() -> None:
    rng = np.random.default_rng(7)
    image = rng.normal(10.0, 1.5, size=(8, 32, 32)).astype(np.float32)
    image[:, 12:20, 12:20] += 8.0

    estimate = estimate_image_snr(image)

    assert estimate["mode"] == "continuous"
    assert np.isfinite(estimate["snr"])
    assert estimate["snr"] > 0.0
    assert estimate["noise_sigma"] > 0.0


@pytest.mark.parametrize(
    ("snr", "expected"),
    [(4.0, 0.8), (8.0, 0.5), (15.0, 0.25), (30.0, 0.0), (50.0, 0.0)],
)
def test_snr_prefilter_boundaries(snr: float, expected: float) -> None:
    assert _snr_prefilter_sigma(snr) == pytest.approx(expected)


def test_acuity_monotonically_changes_automatic_smoothing_and_stopping() -> None:
    image = np.ones((8, 8), dtype=np.float32)
    smooth = _resolve_snr_settings(image, 6.0, -50.0, 0.0, 0.005)
    neutral = _resolve_snr_settings(image, 6.0, 0.0, 0.0, 0.005)
    sharp = _resolve_snr_settings(image, 6.0, 50.0, 0.0, 0.005)

    assert smooth["prefilter_sigma"] > neutral["prefilter_sigma"] > sharp["prefilter_sigma"]
    assert smooth["rel_threshold"] > neutral["rel_threshold"] > sharp["rel_threshold"]


def test_explicit_prefilter_overrides_snr_automatic_value() -> None:
    settings = _resolve_snr_settings(np.ones((8, 8), dtype=np.float32), 4.0, 0.0, 0.3, 0.005)
    assert settings["prefilter_sigma"] == pytest.approx(0.3)


def test_photon_step_makes_auto_offset_scale_aware() -> None:
    image = np.array([0, 2, 4, 6], dtype=np.float32)
    assert _scale_aware_auto_offset(image, {"intensity_step": 2.0}) == pytest.approx(0.1)


def test_snr_off_retains_legacy_defaults() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 10.0
    psf = np.zeros((3, 3), dtype=np.float32)
    psf[1, 1] = 1.0

    implicit = ci_rl_deconvolve(
        image, psf, niter=3, device="cpu", tiling="none", convergence="fixed",
        microscope_type="confocal",
    )
    explicit_off = ci_rl_deconvolve(
        image, psf, niter=3, device="cpu", tiling="none", convergence="fixed",
        snr=None, acuity=0.0, microscope_type="confocal",
    )

    np.testing.assert_array_equal(implicit["result"], explicit_off["result"])
    assert explicit_off["effective_parameters"]["offset"] == pytest.approx(5.0)
    assert explicit_off["effective_parameters"]["snr"] is None


def test_auto_snr_reports_effective_parameters_without_damping() -> None:
    image = np.zeros((9, 9), dtype=np.float32)
    image[3:6, 3:6] = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]], dtype=np.float32)
    psf = np.zeros((3, 3), dtype=np.float32)
    psf[1, 1] = 1.0

    output = ci_rl_deconvolve(
        image, psf, niter=2, device="cpu", tiling="none", convergence="fixed", snr="auto"
    )
    effective = output["effective_parameters"]

    assert effective["snr"] > 0.0
    assert effective["snr_mode"] in {"photon-count", "continuous"}
    assert effective["prefilter_sigma"] >= 0.0
    assert 0.0 < effective["offset"] < 5.0


def test_auto_snr_improves_sparse_low_photon_reconstruction() -> None:
    scipy_ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(2)
    truth = np.zeros((12, 32, 32), dtype=np.float32)
    truth[5:8, 12:20, 12:20] = 2.0
    z, y, x = np.mgrid[-2:3, -3:4, -3:4]
    psf = np.exp(-(z * z / 2.0 + y * y / 4.0 + x * x / 4.0)).astype(np.float32)
    psf /= psf.sum()
    observed = rng.poisson(scipy_ndimage.convolve(truth / 2.0, psf) * 2.0).astype(np.float32)
    options = dict(
        niter=20, device="cpu", tiling="none", microscope_type="confocal", convergence="fixed"
    )

    legacy = ci_rl_deconvolve(observed, psf, **options)["result"]
    aware = ci_rl_deconvolve(observed, psf, snr="auto", **options)["result"]

    assert np.mean((aware - truth) ** 2) < np.mean((legacy - truth) ** 2)
    assert float(aware.max()) < float(legacy.max())
