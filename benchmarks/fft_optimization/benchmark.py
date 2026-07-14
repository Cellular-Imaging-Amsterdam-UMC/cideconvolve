from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from core.deconvolve import generate_psf, load_image, save_mip_png  # noqa: E402
from fft_shapes import candidate_shapes, named_shapes, next_smooth, padding_ratio  # noqa: E402
from fused_extension import load_fused_extension, lto_callback_capability  # noqa: E402
from metrics import comparison_metrics, psnr  # noqa: E402
from solver import (  # noqa: E402
    crop_psf_to_image,
    minimum_work_shape,
    run_buffered,
    run_buffered_tiled,
    run_direct_cufft,
    run_direct_regularized,
    run_direct_tiled,
    run_direct_z_partitioned,
    run_production,
    validate_fused_regularizers,
)


def _command(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def system_info() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    lto_ok, lto_reason = lto_callback_capability()
    mathdx_root = Path(os.environ.get("MATHDX_ROOT", "")) if os.environ.get("MATHDX_ROOT") else None
    cufftdx_header = mathdx_root / "include" / "cufftdx.hpp" if mathdx_root else None
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_cuda_arch_list": torch.cuda.get_arch_list(),
        "extension_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "gpu_memory_gib": props.total_memory / 1024**3,
        "cuda_home": os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH")),
        "nvcc": _command(["nvcc", "--version"]),
        "driver": _command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "cufft_lto_callback_available": lto_ok,
        "cufft_lto_callback_reason": lto_reason,
        "cufftdx_available": bool(cufftdx_header and cufftdx_header.is_file()),
        "cufftdx_reason": "available" if cufftdx_header and cufftdx_header.is_file() else "cufftdx.hpp not found; cuFFTDx is distributed separately in NVIDIA MathDx",
    }


def fft_shape_scan(minimum: tuple[int, ...], max_shapes: int = 12) -> list[dict[str, Any]]:
    shapes = candidate_shapes(minimum, max_padding=0.10, max_shapes=max_shapes)
    for shape in named_shapes(minimum).values():
        if shape not in shapes:
            shapes.append(shape)
    rows = []
    for shape in shapes:
        try:
            torch.cuda.empty_cache()
            real = torch.rand(shape, device="cuda", dtype=torch.float32)
            for _ in range(2):
                frequency = torch.fft.rfftn(real)
                torch.fft.irfftn(frequency, s=shape)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                frequency = torch.fft.rfftn(real)
                torch.fft.irfftn(frequency, s=shape)
            end.record()
            torch.cuda.synchronize()
            rows.append({
                "shape": list(shape),
                "padding_ratio": padding_ratio(shape, minimum),
                "fft_pair_ms": start.elapsed_time(end) / 5.0,
                "status": "ok",
            })
            del real, frequency
        except torch.cuda.OutOfMemoryError:
            rows.append({"shape": list(shape), "padding_ratio": padding_ratio(shape, minimum), "status": "oom"})
            torch.cuda.empty_cache()
    return sorted(rows, key=lambda row: row.get("fft_pair_ms", float("inf")))


def qc_images(out: Path, metadata: dict[str, Any], channel: int, reference: np.ndarray, candidate: np.ndarray, label: str) -> None:
    ref_mip = np.max(reference, axis=0) if reference.ndim == 3 else reference
    got_mip = np.max(candidate, axis=0) if candidate.ndim == 3 else candidate
    diff_mip = np.max(np.abs(candidate - reference), axis=0) if reference.ndim == 3 else np.abs(candidate - reference)
    save_mip_png(ref_mip, out / f"ch{channel}_reference.png", metadata, channel_indices=[channel])
    save_mip_png(got_mip, out / f"ch{channel}_{label}.png", metadata, channel_indices=[channel])
    save_mip_png(diff_mip, out / f"ch{channel}_{label}_absdiff.png", metadata, channel_indices=[channel])


def result_row(channel: int, variant: str, result, reference: np.ndarray | None) -> dict[str, Any]:
    row = {"channel": channel, "variant": variant, "status": "ok", **result.metrics()}
    row["work_shape"] = "x".join(map(str, result.work_shape))
    if reference is not None:
        row.update(comparison_metrics(reference, result.result))
        row["psnr_db"] = psnr(reference, result.result)
    return row


def write_outputs(out: Path, info: dict[str, Any], rows: list[dict[str, Any]], shape_scans: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"system": info, "shape_scans": shape_scans, "results": rows}
    (out / "report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# FFT deconvolution benchmark", "", f"GPU: {info['gpu']}", f"PyTorch: {info['torch']} (CUDA {info['torch_cuda']})", "", "| dataset | ch | method | variant | wall s | GPU s | data s | regularizer s | peak MB | SSIM | NRMSE | pass |", "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in rows:
        lines.append(f"| {row.get('dataset','')} | {row.get('channel','')} | {row.get('method','')} | {row.get('variant','')} | {row.get('wall_time_s',0):.3f} | {row.get('gpu_time_s',0):.3f} | {row.get('data_step_time_s',0):.3f} | {row.get('regularizer_time_s',0):.3f} | {row.get('peak_allocated_mb',0):.0f} | {row.get('ssim_global',0):.6f} | {row.get('nrmse',0):.3g} | {row.get('quality_pass','')} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_quick(data: dict[str, Any], out: Path, fused_ops: Any | None, iterations: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scans: dict[str, Any] = {}
    metadata = data["metadata"]
    for channel, image in enumerate(data["images"]):
        psf = crop_psf_to_image(generate_psf(metadata, channel_idx=channel), image.shape)
        minimum = minimum_work_shape(image.shape, psf.shape)
        scans[f"channel_{channel}"] = fft_shape_scan(minimum)
        print(f"Channel {channel}: production reference ({iterations} iterations)", flush=True)
        reference_result = run_production(image, psf, niter=iterations, tiling="none")
        rows.append(result_row(channel, "production_exact", reference_result, reference_result.result))
        variants = [
            ("buffered_exact", named_shapes(minimum)["exact"], None, False),
            ("buffered_smooth", named_shapes(minimum)["smooth"], None, False),
            ("buffered_power2", named_shapes(minimum)["power2"], None, False),
        ]
        if fused_ops is not None:
            variants.extend([
                ("fused_exact", named_shapes(minimum)["exact"], fused_ops, False),
                ("fused_smooth", named_shapes(minimum)["smooth"], fused_ops, False),
                ("fused_smooth_graph", named_shapes(minimum)["smooth"], fused_ops, True),
            ])
        best = None
        for label, shape, extension, graph in variants:
            print(f"Channel {channel}: {label} work={shape}", flush=True)
            try:
                result = run_buffered(image, psf, work_shape=shape, niter=iterations, fused_ops=extension, use_graph=graph)
                row = result_row(channel, label, result, reference_result.result)
                rows.append(row)
                if row["quality_pass"] and (best is None or row["wall_time_s"] < best[0]):
                    best = (row["wall_time_s"], label, result.result.copy())
            except Exception as exc:
                rows.append({"channel": channel, "variant": label, "status": "error", "error": repr(exc)})
                torch.cuda.empty_cache()
        if fused_ops is not None:
            for label, function in [
                ("overwrite_smooth", lambda: run_buffered(image, psf, work_shape=named_shapes(minimum)["smooth"], niter=iterations, fused_ops=fused_ops, overwrite_state=True)),
                ("direct_cufft_fp32", lambda: run_direct_cufft(image, psf, work_shape=named_shapes(minimum)["smooth"], niter=iterations, fused_ops=fused_ops, static_precision="fp32")),
                ("direct_cufft_static_fp16", lambda: run_direct_cufft(image, psf, work_shape=named_shapes(minimum)["smooth"], niter=iterations, fused_ops=fused_ops, static_precision="fp16")),
            ]:
                print(f"Channel {channel}: {label}", flush=True)
                try:
                    result = function()
                    row = result_row(channel, label, result, reference_result.result)
                    rows.append(row)
                    if row["quality_pass"] and (best is None or row["wall_time_s"] < best[0]):
                        best = (row["wall_time_s"], label, result.result.copy())
                except Exception as exc:
                    rows.append({"channel": channel, "variant": label, "status": "error", "error": repr(exc)})
                    torch.cuda.empty_cache()
        cufftdx_header = Path(os.environ.get("MATHDX_ROOT", "")) / "include" / "cufftdx.hpp" if os.environ.get("MATHDX_ROOT") else None
        rows.append({
            "channel": channel,
            "variant": "cufftdx_fused_3d",
            "status": "unavailable" if not cufftdx_header or not cufftdx_header.is_file() else "not_built",
            "error": "cuFFTDx is a separate MathDx package; cufftdx.hpp was not found" if not cufftdx_header or not cufftdx_header.is_file() else "MATHDX_ROOT found but backend compilation is not enabled",
        })
        if best:
            qc_images(out, metadata, channel, reference_result.result, best[2], best[1])
        del reference_result, psf
        gc.collect()
        torch.cuda.empty_cache()
    return rows, scans


def run_full(data: dict[str, Any], out: Path, fused_ops: Any | None, iterations: int, tiles: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata = data["metadata"]
    for channel, image in enumerate(data["images"]):
        psf = crop_psf_to_image(generate_psf(metadata, channel_idx=channel), image.shape)
        print(f"Channel {channel}: full production auto-tiling ({iterations} iterations)", flush=True)
        reference_result = run_production(image, psf, niter=iterations, tiling="auto")
        rows.append(result_row(channel, "production_auto", reference_result, reference_result.result))
        smooth = lambda shape: tuple(next_smooth(v) for v in shape)
        variants = []
        if fused_ops is not None:
            variants.extend([
                ("direct9_uncached_fp32", lambda: run_direct_tiled(image, psf, niter=iterations, n_tiles=9, shape_policy=smooth, fused_ops=fused_ops, static_precision="fp32", margin=16, cache_static=False)),
                ("direct9_cached_fp32", lambda: run_direct_tiled(image, psf, niter=iterations, n_tiles=9, shape_policy=smooth, fused_ops=fused_ops, static_precision="fp32", margin=16, cache_static=True)),
                ("direct9_cached_static_fp16", lambda: run_direct_tiled(image, psf, niter=iterations, n_tiles=9, shape_policy=smooth, fused_ops=fused_ops, static_precision="fp16", margin=16, cache_static=True)),
                ("direct4_margin16", lambda: run_direct_tiled(image, psf, niter=iterations, n_tiles=4, shape_policy=smooth, fused_ops=fused_ops, static_precision="fp32", margin=16, cache_static=True)),
                ("direct4_margin32", lambda: run_direct_tiled(image, psf, niter=iterations, n_tiles=4, shape_policy=smooth, fused_ops=fused_ops, static_precision="fp32", margin=32, cache_static=True)),
                ("direct4_margin64", lambda: run_direct_tiled(image, psf, niter=iterations, n_tiles=4, shape_policy=smooth, fused_ops=fused_ops, static_precision="fp32", margin=64, cache_static=True)),
                ("direct_untiled_fp32", lambda: run_direct_cufft(image, psf, work_shape=smooth(minimum_work_shape(image.shape, psf.shape)), niter=iterations, fused_ops=fused_ops, static_precision="fp32")),
                ("direct_z2_fp32", lambda: run_direct_z_partitioned(image, psf, niter=iterations, z_partitions=2, shape_policy=smooth, fused_ops=fused_ops, static_precision="fp32")),
            ])
        best = None
        for label, function in variants:
            print(f"Channel {channel}: {label}", flush=True)
            try:
                optimized = function()
                row = result_row(channel, label, optimized, reference_result.result)
                rows.append(row)
                if row["quality_pass"] and (best is None or row["wall_time_s"] < best[0]):
                    best = (row["wall_time_s"], label, optimized.result.copy())
                del optimized
            except Exception as exc:
                rows.append({"channel": channel, "variant": label, "status": "error", "error": repr(exc)})
                gc.collect()
                torch.cuda.empty_cache()
        rows.append({
            "channel": channel, "variant": "cufftdx_fused_3d", "status": "unavailable",
            "error": "cuFFTDx MathDx headers are not installed; no licensed package was downloaded automatically",
        })
        if best:
            qc_images(out, metadata, channel, reference_result.result, best[2], best[1])
        del reference_result, psf
        gc.collect()
    return rows, {}


def run_regularizers(data: dict[str, Any], out: Path, fused_ops: Any | None, iterations: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare each regularized solver with the optimized FP32 FFT data step."""
    if fused_ops is None:
        raise RuntimeError("the regularizer preset requires the CUDA benchmark extension")
    rows: list[dict[str, Any]] = []
    validation = validate_fused_regularizers(fused_ops)
    if not validation["tv_pass"] or not validation["sparse_pass"] or not validation["sparse_2d_pass"]:
        raise RuntimeError(f"fused regularizer validation failed: {validation}")
    metadata = data["metadata"]
    for channel, image in enumerate(data["images"]):
        psf = crop_psf_to_image(generate_psf(metadata, channel_idx=channel), image.shape)
        work = named_shapes(minimum_work_shape(image.shape, psf.shape))["smooth"]
        for method in ("ci_rl_tv", "ci_sparse_hessian"):
            print(f"Channel {channel}: {method} production reference ({iterations} iterations)", flush=True)
            reference = run_production(image, psf, niter=iterations, tiling="none", method=method)
            reference_row = result_row(channel, "production_exact", reference, reference.result)
            rows.append(reference_row)
            for label, use_fused in (("direct_pytorch_regularizer", False), ("direct_fused_regularizer", True)):
                print(f"Channel {channel}: {method} {label} FP32 work={work}", flush=True)
                try:
                    optimized = run_direct_regularized(
                        image, psf, work_shape=work, niter=iterations,
                        fused_ops=fused_ops, method=method,
                        fused_regularizer=use_fused,
                    )
                    row = result_row(channel, label, optimized, reference.result)
                    rows.append(row)
                    if use_fused:
                        qc_images(out, metadata, channel, reference.result, optimized.result, f"{method}_fused")
                    del optimized
                except Exception as exc:
                    rows.append({"channel": channel, "method": method, "variant": label, "status": "error", "error": repr(exc)})
                    torch.cuda.empty_cache()
            del reference
            gc.collect()
            torch.cuda.empty_cache()
        del psf
    return rows, {"fused_regularizer_validation": validation}


def run_sparse_datasets(out: Path, fused_ops: Any | None, iterations: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate fused sparse Hessian on representative local 2D and 3D files."""
    if fused_ops is None:
        raise RuntimeError("the sparse dataset preset requires the CUDA benchmark extension")
    validation = validate_fused_regularizers(fused_ops)
    if not validation["sparse_pass"] or not validation["sparse_2d_pass"]:
        raise RuntimeError(f"fused sparse-Hessian validation failed: {validation}")
    cases = [
        ("WF-2D-3Ch-Actin.ome.tiff", 0),
        ("cidecon/DNARepairSpots_decon.ome.tiff", 0),
        ("Vesicles.ome.tiff", 0),
        ("DividingCellcrop.ome.tiff", 0),
        ("U2OS.ome.tiff", 0),
    ]
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for relative, channel in cases:
        print(f"Dataset {relative}: loading", flush=True)
        data = load_image(ROOT / "localdata" / relative)
        image = np.ascontiguousarray(data["images"][channel], dtype=np.float32)
        original_ndim = image.ndim
        if original_ndim == 2:
            image = image[np.newaxis, ...]
        psf = generate_psf(data["metadata"], channel_idx=channel)
        if original_ndim == 2:
            if psf.ndim == 3:
                psf = psf[psf.shape[0] // 2]
            psf = psf[np.newaxis, ...]
        psf = crop_psf_to_image(psf, image.shape)
        # Use production-exact padding here so this suite isolates regularizer
        # equivalence from the separately benchmarked smooth-shape FFT choice.
        work = minimum_work_shape(image.shape, psf.shape)
        pixel_xy = data["metadata"].get("pixel_size_x")
        pixel_z = data["metadata"].get("pixel_size_z")
        z_scale = float(pixel_xy) / float(pixel_z) if original_ndim == 3 and pixel_xy and pixel_z else 1.0
        inventory.append({
            "dataset": relative, "source_shape": list(data["images"][channel].shape),
            "benchmark_shape": list(image.shape), "psf_shape": list(psf.shape),
            "work_shape": list(work), "dimension": "2D/singleton-Z" if original_ndim == 2 else "3D",
            "pixel_size_xy": pixel_xy, "pixel_size_z": pixel_z, "z_scale": z_scale,
        })
        print(f"Dataset {relative}: production and optimized sparse Hessian ({iterations} iterations)", flush=True)
        reference = run_production(
            image, psf, niter=iterations, method="ci_sparse_hessian",
            pixel_size_xy=pixel_xy, pixel_size_z=pixel_z,
        )
        ref_row = result_row(channel, "production_exact", reference, reference.result)
        ref_row.update(dataset=relative, dimension=inventory[-1]["dimension"], z_scale=z_scale)
        rows.append(ref_row)
        pytorch_regularizer_result = None
        for label, use_fused in (("direct_pytorch_regularizer", False), ("direct_fused_regularizer", True)):
            result = run_direct_regularized(
                image, psf, work_shape=work, niter=iterations, fused_ops=fused_ops,
                method="ci_sparse_hessian", fused_regularizer=use_fused,
                pixel_size_xy=pixel_xy, pixel_size_z=pixel_z,
            )
            row = result_row(channel, label, result, reference.result)
            row.update(dataset=relative, dimension=inventory[-1]["dimension"], z_scale=z_scale)
            if use_fused and pytorch_regularizer_result is not None:
                kernel_metrics = comparison_metrics(pytorch_regularizer_result, result.result)
                row.update({f"kernel_{key}": value for key, value in kernel_metrics.items()})
            elif not use_fused:
                pytorch_regularizer_result = result.result.copy()
            rows.append(row)
            del result
        del reference, data, image, psf
        gc.collect()
        torch.cuda.empty_cache()
    return rows, {"fused_regularizer_validation": validation, "dataset_inventory": inventory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "full", "regularizers", "sparse-datasets"), default="quick")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-extension", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch")
    default_name = "DNA.ome.tiff" if args.preset == "full" else "DNAcrop.ome.tiff"
    input_path = args.input or ROOT / "localdata" / default_name
    iterations = args.iterations or (20 if args.preset == "full" else 5)
    out = args.output or HERE / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    info = system_info()
    data = None
    if args.preset != "sparse-datasets":
        print(f"Loading {input_path}", flush=True)
        data = load_image(input_path)
    fused_ops, extension_error = (None, "disabled") if args.no_extension else load_fused_extension(verbose=True)
    info["fused_extension_loaded"] = fused_ops is not None
    info["fused_extension_error"] = extension_error
    started = time.perf_counter()
    if args.preset == "quick":
        assert data is not None
        rows, scans = run_quick(data, out, fused_ops, iterations)
    elif args.preset == "regularizers":
        assert data is not None
        rows, scans = run_regularizers(data, out, fused_ops, iterations)
    elif args.preset == "sparse-datasets":
        rows, scans = run_sparse_datasets(out, fused_ops, iterations)
    else:
        assert data is not None
        rows, scans = run_full(data, out, fused_ops, iterations, args.tiles)
    info["total_benchmark_seconds"] = time.perf_counter() - started
    info["input"] = "selected localdata suite" if args.preset == "sparse-datasets" else str(input_path)
    info["iterations"] = iterations
    write_outputs(out, info, rows, scans)
    print(f"Results: {out}", flush=True)
    return 0 if all(row.get("status") in {"ok", "unavailable"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
