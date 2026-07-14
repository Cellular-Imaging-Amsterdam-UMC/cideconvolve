"""CUDA/container acceptance smoke test for local Docker and Slurm nodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE if (HERE / "core").is_dir() else HERE.parent
sys.path.insert(0, str(APP_ROOT))

from core.deconvolve_ci import ci_rl_deconvolve, ci_sparse_hessian_deconvolve


def _synthetic_case() -> tuple[np.ndarray, np.ndarray]:
    z, y, x = np.mgrid[-2:3, -4:5, -4:5]
    psf = np.exp(-(z * z / 2.0 + y * y / 4.0 + x * x / 4.0)).astype(np.float32)
    psf /= psf.sum(dtype=np.float64)
    image = np.full((9, 32, 32), 4.0, dtype=np.float32)
    image[4, 16, 16] = 100.0
    image[3:6, 10:13, 22:25] += 20.0
    return image, psf


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    array = np.asarray(result["result"])
    return {
        "backend": str(result.get("backend", "unknown")),
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "nonnegative": bool(float(array.min()) >= 0.0),
        "iterations_used": int(result.get("iterations_used", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-solvers", action="store_true")
    args = parser.parse_args()

    expected_version = os.environ.get("CIDECONVOLVE_PYTORCH_VERSION", "2.13.0")
    expected_cuda = os.environ.get("CIDECONVOLVE_PYTORCH_CUDA", "cu132").removeprefix("cu")
    expected_cuda = f"{expected_cuda[:-1]}.{expected_cuda[-1]}"
    if not torch.__version__.startswith(f"{expected_version}+cu"):
        raise RuntimeError(f"unexpected PyTorch version: {torch.__version__}")
    if torch.version.cuda != expected_cuda:
        raise RuntimeError(f"unexpected PyTorch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    major, minor = torch.cuda.get_device_capability(0)
    native_arch = f"sm_{major}{minor}"
    wheel_arches = torch.cuda.get_arch_list()
    if native_arch not in wheel_arches:
        raise RuntimeError(f"wheel lacks native {native_arch}; architectures={wheel_arches}")

    source = torch.rand((9, 33, 35), dtype=torch.float32, device="cuda")
    restored = torch.fft.irfftn(torch.fft.rfftn(source), s=source.shape)
    torch.cuda.synchronize()
    fft_error = float((source - restored).abs().max())
    if not torch.isfinite(restored).all() or fft_error > 1e-5:
        raise RuntimeError(f"FFT round trip failed: max_abs={fft_error}")

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "native_arch": native_arch,
        "wheel_arches": wheel_arches,
        "fft_max_abs": fft_error,
    }
    if not args.skip_solvers:
        image, psf = _synthetic_case()
        common = {
            "niter": 2,
            "offset": 0.0,
            "background": 1e-3,
            "convergence": "fixed",
            "device": "cuda",
            "backend": "optimized_cuda",
            "tiling": "none",
        }
        report["ci_rl"] = _result_summary(ci_rl_deconvolve(image, psf, **common))
        report["ci_rl_tv"] = _result_summary(
            ci_rl_deconvolve(image, psf, tv_lambda=1e-4, **common)
        )
        report["ci_sparse_hessian"] = _result_summary(
            ci_sparse_hessian_deconvolve(image, psf, **common)
        )
        if not all(
            result["backend"] == "optimized_cuda"
            and result["finite"]
            and result["nonnegative"]
            for result in (report["ci_rl"], report["ci_rl_tv"], report["ci_sparse_hessian"])
        ):
            raise RuntimeError("one or more optimized solver smoke tests failed")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
