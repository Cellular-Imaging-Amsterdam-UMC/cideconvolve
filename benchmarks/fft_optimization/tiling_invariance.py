"""Benchmark result invariance when the XY compute-tile count changes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from core.deconvolve import generate_psf, load_image  # noqa: E402
from core.deconvolve_ci import (  # noqa: E402
    ci_rl_deconvolve,
    clear_optimized_context_cache,
    estimate_image_snr,
)
from metrics import comparison_metrics  # noqa: E402
from solver import crop_psf_to_image  # noqa: E402


def _global_parameters(image: np.ndarray) -> tuple[float, float, float]:
    flat = np.asarray(image, dtype=np.float32).reshape(-1)
    sample = flat[::max(1, int(np.ceil(flat.size / 10_000_000)))]
    background = float(np.percentile(sample, 5.0))
    dynamic = max(
        float(np.percentile(sample, 99.9)) - float(np.percentile(sample, 1.0)), 0.0,
    )
    offset = float(np.clip(1e-6 * dynamic, 1e-6, 5.0))
    return background, offset, float(estimate_image_snr(image)["snr"])


def _run(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    tiles: int,
    iterations: int,
    background: str | float,
    offset: str | float,
    snr: float,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    clear_optimized_context_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = ci_rl_deconvolve(
        image,
        psf,
        niter=iterations,
        snr=snr,
        start="auto",
        background=background,
        offset=offset,
        convergence="auto",
        rel_threshold=0.001,
        check_every=5,
        pixel_size_xy=metadata.get("pixel_size_x"),
        pixel_size_z=metadata.get("pixel_size_z"),
        microscope_type=metadata.get("microscope_type", "widefield"),
        device="cuda",
        backend="optimized_cuda",
        tiling=tiles,
    )
    torch.cuda.synchronize()
    summary = {
        "elapsed_s": time.perf_counter() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "backend": result.get("backend"),
        "tile_count": result.get("tile_count", 1),
        "work_shape": result.get("work_shape"),
    }
    return np.asarray(result["result"], dtype=np.float32), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "localdata" / "DNA.ome.tiff")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--reference-tiles", type=int, default=9)
    parser.add_argument("--candidate-tiles", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    data = load_image(args.input)
    image = np.asarray(data["images"][args.channel], dtype=np.float32)
    psf = crop_psf_to_image(generate_psf(data["metadata"], channel_idx=args.channel), image.shape)
    background, offset, snr = _global_parameters(image)
    common = dict(
        image=image,
        psf=psf,
        iterations=args.iterations,
        snr=snr,
        metadata=data["metadata"],
    )
    accepted, accepted_stats = _run(
        tiles=args.reference_tiles, background="auto", offset="auto", **common,
    )
    global_reference, global_reference_stats = _run(
        tiles=args.reference_tiles, background=background, offset=offset, **common,
    )
    global_candidate, global_candidate_stats = _run(
        tiles=args.candidate_tiles, background=background, offset=offset, **common,
    )
    report = {
        "input": str(args.input),
        "channel": args.channel,
        "shape": image.shape,
        "psf_shape": psf.shape,
        "iterations": args.iterations,
        "global_parameters": {"background": background, "offset": offset, "snr": snr},
        "accepted_reference": accepted_stats,
        "global_reference": global_reference_stats,
        "global_candidate": global_candidate_stats,
        "global_reference_vs_accepted": comparison_metrics(accepted, global_reference),
        "global_candidate_vs_accepted": comparison_metrics(accepted, global_candidate),
        "global_candidate_vs_global_reference": comparison_metrics(global_reference, global_candidate),
    }
    output = args.output or HERE / "results" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    (output / "tiling_invariance.json").write_text(
        json.dumps(report, indent=2, default=list), encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=list))
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
