from __future__ import annotations

import math

import numpy as np


def _sample_pair(reference: np.ndarray, candidate: np.ndarray, max_values: int = 10_000_000):
    ref32 = np.asarray(reference, dtype=np.float32)
    got32 = np.asarray(candidate, dtype=np.float32)
    if ref32.shape != got32.shape:
        raise ValueError(f"shape mismatch: {ref32.shape} != {got32.shape}")
    stride = max(1, math.ceil(ref32.size / max_values))
    return ref32, got32, ref32.reshape(-1)[::stride].astype(np.float64), got32.reshape(-1)[::stride].astype(np.float64)


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool]:
    ref32, got32, ref, got = _sample_pair(reference, candidate)
    delta = got - ref
    dynamic = max(float(np.max(ref) - np.min(ref)), 1e-12)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    nrmse = rmse / dynamic
    flux_ref = float(np.sum(ref32, dtype=np.float64))
    flux_diff = abs(float(np.sum(got32, dtype=np.float64)) - flux_ref) / max(abs(flux_ref), 1e-12)
    # Global SSIM is deterministic and inexpensive for large volumes. Slice SSIMs
    # are reported separately by the runner when scikit-image is available.
    mu_x, mu_y = float(np.mean(ref)), float(np.mean(got))
    vx, vy = float(np.var(ref)), float(np.var(got))
    covariance = float(np.mean((ref - mu_x) * (got - mu_y)))
    c1 = (0.01 * dynamic) ** 2
    c2 = (0.03 * dynamic) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (vx + vy + c2)
    )
    finite = bool(np.isfinite(got32).all())
    nonnegative = bool(float(np.min(got32)) >= 0.0)
    passed = finite and nonnegative and ssim >= 0.9999 and nrmse <= 0.001 and flux_diff <= 0.001
    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "ssim_global": float(ssim),
        "flux_relative_difference": flux_diff,
        "finite": finite,
        "nonnegative": nonnegative,
        "quality_pass": passed,
    }


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    dynamic = max(float(np.max(ref) - np.min(ref)), 1e-12)
    mse = float(np.mean((got - ref) ** 2))
    return math.inf if mse == 0.0 else 20.0 * math.log10(dynamic / math.sqrt(mse))
