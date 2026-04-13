"""
wrapper.py â€” BIAFLOWS-compatible entrypoint for CIDeconvolve.

Parses BIAFLOWS job parameters (--infolder, --outfolder, --gtfolder, etc.)
via bioflows_local, then processes each input image through the CI
deconvolution pipeline in deconvolve.py and writes results to the output
folder.

Usage (inside Docker):
    python wrapper.py --infolder /data/in --outfolder /data/out --gtfolder /data/gt --local

Usage (local):
    python wrapper.py --infolder ./infolder --outfolder ./outfolder --gtfolder ./gtfolder --local --iterations "40" --method ci_rl
"""
import csv
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

# Configure logging so deconvolve.py INFO messages are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from bioflows_local import (
    CLASS_SPTCNT,
    BiaflowsJob,
    get_discipline,
    prepare_data,
)

# Import deconvolve first (handles torch-before-numpy DLL load order)
from deconvolve import (
    MAX_TILE_XY,
    MAX_TILE_Z,
    deconvolve_image,
    load_image,
    save_mip_png,
    save_result,
)

import numpy as np

# ---------------------------------------------------------------------------
# RI lookup tables â€” value-choices in descriptor.json use "name (RI)" format
# ---------------------------------------------------------------------------
_IMMERSION_RI = {
    "air":   1.0003,
    "water": 1.333,
    "oil":   1.515,
}

_SAMPLE_RI = {
    "water":          1.333,
    "pbs":            1.334,
    "culture medium": 1.337,
    "vectashield":    1.45,
    "glycerol":       1.474,
    "oil":            1.515,
    "prolong glass":  1.52,
}

# Default sample RI when "auto" is chosen and metadata has no value
_SAMPLE_RI_DEFAULT = 1.45  # Vectashield


def _to_bool(value) -> bool:
    """Convert a value to bool, handling string 'True'/'False' from CLI."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _parse_ri_choice(raw: str, lookup: dict[str, float]) -> float | None:
    """Parse a RI choice string like 'oil (1.515)' or a bare float.

    Returns None for 'auto' (meaning: use image metadata / fallback).
    """
    s = str(raw).strip().lower()
    if s == "auto":
        return None
    # Try "name (1.234)" format â€” extract the name part
    name = s.split("(")[0].strip()
    if name in lookup:
        return lookup[name]
    # Try bare float
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Human-readable byte formatting
# ---------------------------------------------------------------------------
def _format_bytes(mb):
    """Format megabytes as a human-readable string."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _normalise_image(arr: np.ndarray) -> np.ndarray:
    """Normalise image to [0, 1] range."""
    img = np.asarray(arr, dtype=np.float64)
    lo = float(np.min(img))
    hi = float(np.max(img))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float64)
    return (img - lo) / (hi - lo)


def _mean_or_zero(values: list[float]) -> float:
    """Return mean of values or 0.0 if empty."""
    return float(np.mean(values)) if values else 0.0


def _no_reference_metrics(arr: np.ndarray) -> dict[str, float]:
    """Compute no-reference quality metrics for a single channel.
    
    Returns:
        dict with 'sharpness', 'contrast', 'noise_proxy' keys.
    """
    img = _normalise_image(arr)
    p1 = float(np.percentile(img, 1))
    p99 = float(np.percentile(img, 99))
    q1 = float(np.percentile(img, 25))
    q3 = float(np.percentile(img, 75))

    try:
        from scipy import ndimage

        if img.ndim == 3:
            lap_vars = [float(np.var(ndimage.laplace(z))) for z in img]
            sharpness = float(np.mean(lap_vars)) if lap_vars else 0.0
        else:
            sharpness = float(np.var(ndimage.laplace(img)))
    except ImportError:
        grads = np.gradient(img.astype(np.float64))
        sharpness = float(np.mean([np.var(g) for g in grads])) if grads else 0.0

    return {
        "sharpness": sharpness,
        "contrast": p99 - p1,
        "noise_proxy": q3 - q1,
    }


def _quality_metrics(
    result_channels: list[np.ndarray],
) -> dict[str, float | int]:
    """Compute aggregate quality metrics from deconvolved channels.
    
    Returns:
        dict with 'channels_compared', 'sharpness_mean', 'contrast_mean', 'noise_proxy_mean'.
    """
    sharpness_vals: list[float] = []
    contrast_vals: list[float] = []
    noise_vals: list[float] = []

    for result in result_channels:
        nr = _no_reference_metrics(result)
        sharpness_vals.append(nr["sharpness"])
        contrast_vals.append(nr["contrast"])
        noise_vals.append(nr["noise_proxy"])
    return {
        "channels_compared": len(result_channels),
        "sharpness_mean": _mean_or_zero(sharpness_vals),
        "contrast_mean": _mean_or_zero(contrast_vals),
        "noise_proxy_mean": _mean_or_zero(noise_vals),
    }


def main(argv):
    with BiaflowsJob.from_cli(argv) as bj:
        parameters = getattr(bj, "parameters", SimpleNamespace())

        # Extract parameters with defaults from descriptor.json
        iter_raw = str(getattr(parameters, "iterations", "40")).strip()
        niter_list = [max(1, int(s.strip())) for s in iter_raw.split(",") if s.strip()]
        if not niter_list:
            niter_list = [40]
        tiling_raw = getattr(parameters, "tiling", "custom")
        tile_limits_raw = str(getattr(parameters, "tile_limits", "512, 64"))
        method = getattr(parameters, "method", "ci_rl")
        device_param = getattr(parameters, "device", "auto")
        device = None if device_param in (None, "auto") else device_param

        # PSF metadata overrides
        na_raw = getattr(parameters, "na", "auto")
        na_override = None if str(na_raw).strip().lower() == "auto" else float(na_raw)
        ri_raw = str(getattr(parameters, "refractive_index", "auto"))
        ri_override = _parse_ri_choice(ri_raw, _IMMERSION_RI)
        sample_ri_raw = str(getattr(parameters, "sample_ri", "auto"))
        sample_ri_parsed = _parse_ri_choice(sample_ri_raw, _SAMPLE_RI)
        sample_ri = sample_ri_parsed if sample_ri_parsed is not None else _SAMPLE_RI_DEFAULT
        micro_raw = getattr(parameters, "microscope_type", "auto")
        micro_override = None if str(micro_raw).strip().lower() == "auto" else str(micro_raw)
        em_raw = str(getattr(parameters, "emission_wl", "auto")).strip()
        em_override = (
            None if em_raw.lower() == "auto"
            else [float(x.strip()) for x in em_raw.split(",") if x.strip()]
        )
        ex_raw = str(getattr(parameters, "excitation_wl", "auto")).strip()
        ex_override = (
            None if ex_raw.lower() == "auto" or not ex_raw
            else [float(x.strip()) for x in ex_raw.split(",") if x.strip()]
        ) or None

        # Deconvolution parameters
        tv_lambda = float(getattr(parameters, "tv_lambda", 0.001))
        bg_raw = str(getattr(parameters, "background", "auto")).strip()
        background = bg_raw if bg_raw.lower() == "auto" else float(bg_raw)
        damp_raw = str(getattr(parameters, "damping", "none")).strip().lower()
        if damp_raw in ("none", "0", "0.0"):
            damping = 0.0
        elif damp_raw == "auto":
            damping = "auto"
        else:
            damping = float(damp_raw)
        convergence = str(getattr(parameters, "convergence", "auto")).strip().lower()
        rel_threshold = float(getattr(parameters, "rel_threshold", 0.005))
        check_every = int(getattr(parameters, "check_every", 5))

        # PSF Gibson-Lanni parameters
        t_g = float(getattr(parameters, "t_g", 170000))
        t_g0 = float(getattr(parameters, "t_g0", 170000))
        t_i0 = float(getattr(parameters, "t_i0", 100000))
        z_p = float(getattr(parameters, "z_p", 0))

        # Pixel size overrides
        px_xy_raw = str(getattr(parameters, "pixel_size_xy", "auto")).strip()
        px_xy_override = None if px_xy_raw.lower() == "auto" else float(px_xy_raw) / 1000.0  # nm → µm
        px_z_raw = str(getattr(parameters, "pixel_size_z", "auto")).strip()
        px_z_override = None if px_z_raw.lower() == "auto" else float(px_z_raw) / 1000.0  # nm → µm

        projection = str(getattr(parameters, "projection", "none")).lower()
        benchmark = _to_bool(getattr(parameters, "benchmark", False))
        bench_crop = _to_bool(getattr(parameters, "bench_crop", False))

        # Parse tiling
        tiling = str(tiling_raw).strip().lower()
        if tiling not in ("none", "custom"):
            tiling = "none"  # fallback

        # Parse tile limits (max_xy, max_z)
        _lim_parts = [s.strip() for s in tile_limits_raw.split(",") if s.strip()]
        max_tile_xy = int(_lim_parts[0]) if len(_lim_parts) >= 1 else MAX_TILE_XY
        max_tile_z = int(_lim_parts[1]) if len(_lim_parts) >= 2 else MAX_TILE_Z

        print("=" * 70)
        print("CIDeconvolve - BIAFLOWS Workflow")
        print("=" * 70)
        print(f"  Input dir    : {bj.input_dir}")
        print(f"  Output dir   : {bj.output_dir}")
        print(f"  Method       : {method}")
        print(f"  Iterations   : {', '.join(str(n) for n in niter_list)}")
        print(f"  Tiling       : {tiling}")
        if tiling == "custom":
            print(f"  Tile limits  : max_xy={max_tile_xy}, max_z={max_tile_z}")
        print(f"  Device       : {device_param}")
        print(f"  Projection   : {projection}")
        if method == "ci_rl_tv":
            print(f"  TV lambda    : {tv_lambda}")
        print(f"  Background   : {background}")
        if damping != 0.0:
            print(f"  Damping      : {damping}")
        print(f"  Convergence  : {convergence} (threshold={rel_threshold}, every={check_every})")
        if na_override is not None:
            print(f"  NA           : {na_override}")
        if ri_override is not None:
            print(f"  Immersion    : {ri_raw} -> RI {ri_override}")
        if sample_ri_parsed is not None:
            print(f"  Sample medium: {sample_ri_raw} -> RI {sample_ri}")
        else:
            print(f"  Sample medium: auto -> vectashield (RI {sample_ri})")
        if micro_override is not None:
            print(f"  Microscope   : {micro_override}")
        if em_override is not None:
            print(f"  Emission WL  : {em_override}")
        if ex_override is not None:
            print(f"  Excitation WL: {ex_override}")
        if benchmark:
            print(f"  Benchmark    : ON (crop={bench_crop})")

        # Prepare data directories and collect input images
        in_imgs, _, in_path, _, out_path, tmp_path = prepare_data(
            get_discipline(bj, default=CLASS_SPTCNT), bj, is_2d=False, **bj.flags
        )

        if not in_imgs:
            print("No input images found. Exiting.")
            return

        print(f"\nFound {len(in_imgs)} input image(s).")

        # ---- Benchmark mode ----
        if benchmark:
            _run_benchmark(
                in_imgs[0], in_path, out_path,
                niter_list=niter_list,
                device=device,
                bench_crop=bench_crop,
                max_tile_xy=max_tile_xy,
                max_tile_z=max_tile_z,
                tiling=tiling,
                na=na_override,
                refractive_index=ri_override,
                sample_ri=sample_ri,
                microscope_type=micro_override,
                emission_wavelengths=em_override,
                excitation_wavelengths=ex_override,
                pixel_size_xy=px_xy_override,
                pixel_size_z=px_z_override,
                tv_lambda=tv_lambda,
                background=background,
                damping=damping,
                convergence=convergence,
                rel_threshold=rel_threshold,
                check_every=check_every,
                ri_coverslip=ri_override,
                ri_coverslip_design=ri_override,
                ri_immersion_design=ri_override,
                t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
            )
            print(f"\n{'=' * 70}")
            print("Benchmark complete.")
            print(f"{'=' * 70}")
            if tmp_path and Path(tmp_path).exists():
                shutil.rmtree(tmp_path, ignore_errors=True)
            return

        for img_resource in in_imgs:
            img_path = Path(in_path) / img_resource.filename
            print(f"\n{'=' * 60}")
            print(f"Processing: {img_resource.filename}")
            print(f"{'=' * 60}")

            t0 = time.time()

            try:
                # Load image and extract metadata
                data = load_image(img_path)
                meta = data["metadata"]
                images = data["images"]

                print(f"  Channels: {len(images)}")
                for i, img in enumerate(images):
                    print(f"    Ch{i}: shape={img.shape}, dtype={img.dtype}")

                # Create a temp dir alongside outfolder for intermediate files
                tmp_work = Path(out_path) / "tmp"
                tmp_work.mkdir(parents=True, exist_ok=True)

                # ----- Deconvolve -----
                result = deconvolve_image(
                    img_path,
                    method=method,
                    niter=niter_list,
                    tiling=tiling,
                    max_tile_xy=max_tile_xy,
                    max_tile_z=max_tile_z,
                    device=device,
                    na=na_override,
                    refractive_index=ri_override,
                    sample_refractive_index=sample_ri,
                    microscope_type=micro_override,
                    emission_wavelengths=em_override,
                    excitation_wavelengths=ex_override,
                    pixel_size_xy=px_xy_override,
                    pixel_size_z=px_z_override,
                    tv_lambda=tv_lambda,
                    background=background,
                    damping=damping,
                    convergence=convergence,
                    rel_threshold=rel_threshold,
                    check_every=check_every,
                    ri_coverslip=ri_override,
                    ri_coverslip_design=ri_override,
                    ri_immersion_design=ri_override,
                    t_g=t_g,
                    t_g0=t_g0,
                    t_i0=t_i0,
                    z_p=z_p,
                )

                if result is None:
                    print(f"  WARNING: deconvolve_image returned None for {img_resource.filename}")
                    shutil.rmtree(tmp_work, ignore_errors=True)
                    continue

                stem = _stem(img_resource.filename)
                is_3d = result["channels"][0].ndim == 3

                if projection in ("mip", "sum") and is_3d:
                    out_name = f"{stem}_decon_{projection}.ome.tiff"
                    tmp_file = tmp_work / out_name
                    proj_result = dict(result)
                    if projection == "mip":
                        proj_result["channels"] = [
                            ch.max(axis=0) for ch in result["channels"]
                        ]
                        if result.get("source_channels"):
                            proj_result["source_channels"] = [
                                ch.max(axis=0) for ch in result["source_channels"]
                            ]
                    else:  # sum
                        proj_result["channels"] = [
                            ch.astype(np.float32).sum(axis=0) for ch in result["channels"]
                        ]
                        if result.get("source_channels"):
                            proj_result["source_channels"] = [
                                ch.astype(np.float32).sum(axis=0) for ch in result["source_channels"]
                            ]
                    save_result(proj_result, str(tmp_file))
                    print(f"  Saved {projection.upper()}: {out_name}")
                else:
                    out_name = f"{stem}_decon.ome.tiff"
                    tmp_file = tmp_work / out_name
                    save_result(result, str(tmp_file))
                    print(f"  Saved: {out_name}")

                # Move only the deconvolved TIFF to the output folder
                dest = Path(out_path) / out_name
                shutil.move(str(tmp_file), str(dest))

                # Clean up the temp working directory
                shutil.rmtree(tmp_work, ignore_errors=True)

            except Exception as exc:
                print(f"  ERROR processing {img_resource.filename}: {exc}")
                import traceback
                traceback.print_exc()
                # Clean up temp dir on failure so no partial files remain
                tmp_work = Path(out_path) / "tmp"
                shutil.rmtree(tmp_work, ignore_errors=True)
                continue

            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.1f}s")

        print(f"\n{'=' * 70}")
        print("CIDeconvolve workflow complete.")
        print(f"{'=' * 70}")

        # Clean up tmp folder
        if tmp_path and Path(tmp_path).exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
            print(f"Cleaned up tmp folder: {tmp_path}")



# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
_BENCH_METHODS = ["ci_rl", "ci_rl_tv"]


def _method_device(method: str) -> str:
    """Return the compute device label for a benchmark method."""
    if method.startswith("ci_rl"):
        import torch
        return "CUDA" if torch.cuda.is_available() else "CPU"
    return "?"


# ---------------------------------------------------------------------------
# Background metrics monitor
# ---------------------------------------------------------------------------
class _MetricsMonitor:
    """Daemon thread that samples CPU/RAM and GPU metrics during a run."""

    def __init__(self, interval=0.1):
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = None

        # Sampled data
        self._cpu_percent: list[float] = []
        self._ram_bytes: list[int] = []
        self._gpu_util: list[float] = []
        self._gpu_mem_bytes: list[int] = []

        # Baselines
        self._ram_baseline = 0
        self._gpu_mem_baseline = 0
        self._torch_baseline = 0

        # Timing
        self._t0 = 0.0
        self._t1 = 0.0

        # Detect capabilities
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except ImportError:
            pass

        self._nvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def start(self):
        """Record baselines and begin sampling."""
        self._cpu_percent.clear()
        self._ram_bytes.clear()
        self._gpu_util.clear()
        self._gpu_mem_bytes.clear()
        self._stop_event.clear()

        if self._proc:
            self._proc.cpu_percent()          # prime
            self._ram_baseline = self._proc.memory_info().rss
        else:
            self._ram_baseline = 0

        if self._nvml_handle:
            import pynvml
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            self._gpu_mem_baseline = mem_info.used
        else:
            self._gpu_mem_baseline = 0

        self._torch_baseline = 0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._torch_baseline = torch.cuda.memory_allocated()
        except Exception:
            pass

        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        """Sampling loop running in background thread."""
        while not self._stop_event.is_set():
            if self._proc:
                try:
                    self._cpu_percent.append(self._proc.cpu_percent())
                    self._ram_bytes.append(self._proc.memory_info().rss)
                except Exception:
                    pass
            if self._nvml_handle:
                try:
                    import pynvml
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    self._gpu_util.append(util.gpu)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self._gpu_mem_bytes.append(mem_info.used)
                except Exception:
                    pass
            self._stop_event.wait(self._interval)

    def stop(self):
        """Stop sampling and return metrics dict."""
        self._t1 = time.perf_counter()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        elapsed = self._t1 - self._t0
        MB = 1024 * 1024

        m: dict[str, float] = {
            "time_s": elapsed,
            "cpu_percent_avg": 0.0,
            "cpu_percent_peak": 0.0,
            "ram_peak_mb": 0.0,
            "ram_avg_mb": 0.0,
            "ram_delta_peak_mb": 0.0,
            "gpu_util_avg": 0.0,
            "gpu_util_peak": 0.0,
            "gpu_mem_peak_mb": 0.0,
            "gpu_mem_avg_mb": 0.0,
            "gpu_mem_delta_peak_mb": 0.0,
            "torch_gpu_peak_mb": 0.0,
            "torch_gpu_delta_mb": 0.0,
            "gpu_total_mb": 0.0,
            "gpu_spill_mb": 0.0,
            "ram_total_mb": 0.0,
            "ram_percent": 0.0,
            "gpu_mem_percent": 0.0,
        }

        if self._cpu_percent:
            m["cpu_percent_avg"] = sum(self._cpu_percent) / len(self._cpu_percent)
            m["cpu_percent_peak"] = max(self._cpu_percent)

        if self._proc:
            import psutil
            m["ram_total_mb"] = psutil.virtual_memory().total / MB

        if self._ram_bytes:
            m["ram_peak_mb"] = max(self._ram_bytes) / MB
            m["ram_avg_mb"] = sum(self._ram_bytes) / len(self._ram_bytes) / MB
            m["ram_delta_peak_mb"] = (max(self._ram_bytes) - self._ram_baseline) / MB
            if m["ram_total_mb"] > 0:
                m["ram_percent"] = m["ram_peak_mb"] / m["ram_total_mb"] * 100

        if self._gpu_util:
            m["gpu_util_avg"] = sum(self._gpu_util) / len(self._gpu_util)
            m["gpu_util_peak"] = max(self._gpu_util)

        if self._gpu_mem_bytes:
            m["gpu_mem_peak_mb"] = max(self._gpu_mem_bytes) / MB
            m["gpu_mem_avg_mb"] = (
                sum(self._gpu_mem_bytes) / len(self._gpu_mem_bytes) / MB
            )
            m["gpu_mem_delta_peak_mb"] = (
                max(self._gpu_mem_bytes) - self._gpu_mem_baseline
            ) / MB

        if self._nvml_handle:
            try:
                import pynvml
                total_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                m["gpu_total_mb"] = total_info.total / MB
            except Exception:
                pass

        try:
            import torch
            if torch.cuda.is_available():
                m["torch_gpu_peak_mb"] = torch.cuda.max_memory_allocated() / MB
                m["torch_gpu_delta_mb"] = (
                    torch.cuda.max_memory_allocated() - self._torch_baseline
                ) / MB
        except Exception:
            pass

        if m["gpu_total_mb"] > 0 and m["gpu_mem_peak_mb"] > 0:
            m["gpu_mem_percent"] = (
                m["gpu_mem_peak_mb"] / m["gpu_total_mb"] * 100
            )

        if m["gpu_total_mb"] > 0 and m["torch_gpu_delta_mb"] > m["gpu_total_mb"]:
            m["gpu_spill_mb"] = m["torch_gpu_delta_mb"] - m["gpu_total_mb"]

        return m


# ---------------------------------------------------------------------------
# CSV & montage helpers
# ---------------------------------------------------------------------------

def _write_metrics_csv(csv_path: Path, all_metrics: dict[str, dict]):
    """Write benchmark metrics to a CSV file."""
    fieldnames = [
        "label", "device", "time_s",
        "cpu_percent_avg", "cpu_percent_peak",
        "ram_total_mb", "ram_peak_mb", "ram_percent", "ram_avg_mb",
        "ram_delta_peak_mb",
        "gpu_util_avg", "gpu_util_peak",
        "gpu_total_mb", "gpu_mem_peak_mb", "gpu_mem_percent", "gpu_mem_avg_mb",
        "gpu_mem_delta_peak_mb",
        "torch_gpu_peak_mb", "torch_gpu_delta_mb",
        "gpu_spill_mb",
        "channels_compared", "sharpness_mean", "contrast_mean", "noise_proxy_mean",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, m in sorted(all_metrics.items()):
            row = {"label": label, "device": m.get("device", "")}
            for k in fieldnames[2:]:
                row[k] = f"{m.get(k, 0.0):.2f}"
            writer.writerow(row)
    print(f"\n  Metrics CSV saved -> {csv_path}")


def _make_metadata_panel(meta, width, height, font):
    """Create a metadata text panel for the montage."""
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(panel)

    lines = [
        f"NA: {meta.get('na', '?')}",
        f"RI immersion: {meta.get('refractive_index', '?')}",
        f"RI sample: {meta.get('sample_refractive_index', '?')}",
        f"Pixel XY: {meta.get('pixel_size_x', '?')} \u00b5m",
        f"Pixel Z:  {meta.get('pixel_size_z', '?')} \u00b5m",
        f"Size: {meta.get('size_x', '?')}\u00d7{meta.get('size_y', '?')}"
        f"\u00d7{meta.get('size_z', '?')}",
        f"Microscope: {meta.get('microscope_type', '?')}",
    ]
    for i, ch in enumerate(meta.get("channels", [])):
        em = ch.get("emission_wavelength") or "?"
        lines.append(f"Ch{i}: Em {em} nm")

    draw.text((8, 8), "\n".join(lines), fill=(255, 255, 255), font=font)
    return panel


def _make_benchmark_montage(
    out_path,
    stem,
    available_methods,
    bench_iterations,
    all_metrics,
    metadata,
):
    """Create an RGB montage of all benchmark MIP PNGs.

    Layout: Row 0 = Source + metadata panel spanning remaining columns.
    Each subsequent row = one iteration config, one column per method.
    """
    from PIL import Image, ImageDraw, ImageFont

    out_dir = Path(out_path)

    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 18)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    # Column order
    col_order = [(m, None) for m in available_methods]

    # Row 0: source MIP
    rows = [[(out_dir / "mip_source.ome.png", "Source")]]

    # One row per iteration config
    for nit_tag, nit_label in bench_iterations:
        row = []
        for method, _variant in col_order:
            fname = f"mip_{stem}_{method}_{nit_tag}i.ome.png"
            metrics_key = f"{method}_{nit_tag}i"
            met = all_metrics.get(metrics_key)
            if met is not None:
                label = f"{method}\n{nit_label} iter  {met['time_s']:.1f}s"
            else:
                label = f"{method}\n{nit_label} iter"
            row.append((out_dir / fname, label))
        rows.append(row)

    # Load existing images, skip missing
    loaded_rows = []
    total = 0
    for row_entries in rows:
        row_images = []
        for path, label in row_entries:
            if path.exists():
                img = Image.open(path).convert("RGB")
                row_images.append((img, label))
                total += 1
        if row_images:
            loaded_rows.append(row_images)

    if total == 0:
        print("  No MIP PNG files found \u2014 skipping montage.")
        return None

    label_height = 78
    padding = 4

    all_imgs = [img for row in loaded_rows for img, _ in row]
    max_w = max(img.size[0] for img in all_imgs)
    max_h = max(img.size[1] for img in all_imgs)

    n_cols = max(len(row) for row in loaded_rows)
    n_rows = len(loaded_rows)
    cell_w = max_w + 2 * padding
    cell_h = max_h + label_height + 2 * padding

    # Metadata panel spans all columns to the right of Source
    span_cols = max(n_cols - 1, 1)
    meta_w = span_cols * cell_w - 2 * padding
    meta_panel = _make_metadata_panel(metadata, meta_w, max_h, font)

    montage_w = n_cols * cell_w
    montage_h = n_rows * cell_h

    montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(montage)

    for row_idx, row_images in enumerate(loaded_rows):
        for col_idx, (img, label) in enumerate(row_images):
            x0 = col_idx * cell_w + padding
            y0 = row_idx * cell_h + padding
            x_off = (max_w - img.size[0]) // 2
            y_off = (max_h - img.size[1]) // 2
            montage.paste(img, (x0 + x_off, y0 + y_off))

            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            tx = x0 + (max_w - tw) // 2
            ty = y0 + max_h + padding
            draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    # Paste metadata panel at row 0, starting at column 1
    meta_x = cell_w + padding
    meta_y = padding
    montage.paste(meta_panel, (meta_x, meta_y))

    montage_path = out_dir / f"decon_benchmark_{stem}.png"
    montage.save(str(montage_path))
    print(f"  Saved montage: {montage_path}  ({montage_w}x{montage_h})")
    return montage_path


def _make_per_channel_montages(
    out_path,
    stem,
    available_methods,
    bench_iterations,
    all_metrics,
    metadata,
):
    """Create one greyscale montage per channel from benchmark MIP TIFFs."""
    from PIL import Image, ImageDraw, ImageFont
    import tifffile

    out_dir = Path(out_path)
    n_ch = metadata.get("n_channels", 1)
    if n_ch < 1:
        n_ch = 1

    font = None
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(name, 18)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    col_order = [(m, None) for m in available_methods]

    # Row 0 = source, rows 1-N = per iteration config
    mip_rows = [
        [("Source", out_dir / "mip_source.ome.tiff")],
    ]
    for nit_tag, nit_label in bench_iterations:
        row = []
        for method, _variant in col_order:
            fname = f"mip_{stem}_{method}_{nit_tag}i.ome.tiff"
            metrics_key = f"{method}_{nit_tag}i"
            met = all_metrics.get(metrics_key)
            if met is not None:
                label = f"{method}\n{nit_label} iter  {met['time_s']:.1f}s"
            else:
                label = f"{method}\n{nit_label} iter"
            row.append((label, out_dir / fname))
        mip_rows.append(row)

    # Load TIFF arrays per row
    loaded_rows = []
    for row_entries in mip_rows:
        row_data = []
        for label, path in row_entries:
            if path.exists():
                arr = tifffile.imread(str(path))
                if arr.ndim == 2:
                    arr = arr[np.newaxis]
                row_data.append((label, arr))
        if row_data:
            loaded_rows.append(row_data)

    if not loaded_rows:
        return

    print(f"\n  Creating per-channel montages ({n_ch} channels)...")

    label_height = 78
    padding = 4

    for ch_idx in range(n_ch):
        panel_rows = []
        for row_data in loaded_rows:
            row_panels = []
            for label, arr in row_data:
                if ch_idx >= arr.shape[0]:
                    continue
                ch_data = arr[ch_idx].astype(np.float64)
                lo, hi = ch_data.min(), ch_data.max()
                if hi > lo:
                    ch_data = (ch_data - lo) / (hi - lo)
                else:
                    ch_data = np.zeros_like(ch_data)
                ch_uint8 = (ch_data * 255).astype(np.uint8)
                img = Image.fromarray(ch_uint8, mode="L").convert("RGB")
                img_draw = ImageDraw.Draw(img)
                img_draw.text(
                    (4, 2), f"Ch{ch_idx}",
                    fill=(255, 255, 255), font=font,
                )
                row_panels.append((img, label))
            if row_panels:
                panel_rows.append(row_panels)

        if not panel_rows:
            continue

        all_imgs = [img for row in panel_rows for img, _ in row]
        max_w = max(img.size[0] for img in all_imgs)
        max_h = max(img.size[1] for img in all_imgs)

        n_cols = max(len(row) for row in panel_rows)
        n_grid_rows = len(panel_rows)
        cell_w = max_w + 2 * padding
        cell_h = max_h + label_height + 2 * padding

        span_cols = max(n_cols - 1, 1)
        meta_w = span_cols * cell_w - 2 * padding
        meta_panel = _make_metadata_panel(metadata, meta_w, max_h, font)

        montage_w = n_cols * cell_w
        montage_h = n_grid_rows * cell_h

        montage = Image.new("RGB", (montage_w, montage_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(montage)

        for row_idx, row_panels_r in enumerate(panel_rows):
            for col_idx, (img, label) in enumerate(row_panels_r):
                x0 = col_idx * cell_w + padding
                y0 = row_idx * cell_h + padding
                x_off = (max_w - img.size[0]) // 2
                y_off = (max_h - img.size[1]) // 2
                montage.paste(img, (x0 + x_off, y0 + y_off))

                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                tx = x0 + (max_w - tw) // 2
                ty = y0 + max_h + padding
                draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

        meta_x = cell_w + padding
        meta_y = padding
        montage.paste(meta_panel, (meta_x, meta_y))

        ch_path = out_dir / f"decon_benchmark_{stem}_ch{ch_idx}.png"
        montage.save(str(ch_path))
        print(f"    Ch{ch_idx}: {ch_path}  ({montage_w}x{montage_h})")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_benchmark(
    img_resource,
    in_path,
    out_path,
    *,
    niter_list,
    device,
    bench_crop,
    max_tile_xy,
    max_tile_z,
    tiling,
    na,
    refractive_index,
    sample_ri,
    microscope_type,
    emission_wavelengths,
    excitation_wavelengths,
    pixel_size_xy,
    pixel_size_z,
    tv_lambda,
    background,
    damping,
    convergence,
    rel_threshold,
    check_every,
    ri_coverslip,
    ri_coverslip_design,
    ri_immersion_design,
    t_g,
    t_g0,
    t_i0,
    z_p,
):
    """Run benchmark on the first input image with both methods."""
    import gc
    import torch

    img_path = Path(in_path) / img_resource.filename
    stem = _stem(img_resource.filename)
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive iteration tag and label from descriptor iterations
    if len(set(niter_list)) == 1:
        nit_tag = str(niter_list[0])
        nit_label = str(niter_list[0])
    else:
        nit_tag = "-".join(str(n) for n in niter_list)
        nit_label = "/".join(str(n) for n in niter_list)
    bench_iterations = [(nit_tag, nit_label)]

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK: {img_resource.filename}")
    print(f"  Methods   : {', '.join(_BENCH_METHODS)}")
    print(f"  Iterations: {nit_label}")
    print(f"  Crop      : {bench_crop}")
    print(f"{'=' * 70}")

    # Load image once
    data = load_image(img_path)
    meta = data["metadata"]
    images = data["images"]

    # If bench_crop, centre-crop each channel to tile limits
    if bench_crop:
        import tifffile
        cropped = []
        for ch in images:
            if ch.ndim == 3:
                nz, ny, nx = ch.shape
                cz = min(nz, max_tile_z)
                cy = min(ny, max_tile_xy)
                cx = min(nx, max_tile_xy)
                z0 = (nz - cz) // 2
                y0 = (ny - cy) // 2
                x0 = (nx - cx) // 2
                cropped.append(ch[z0:z0+cz, y0:y0+cy, x0:x0+cx])
            else:
                ny, nx = ch.shape
                cy = min(ny, max_tile_xy)
                cx = min(nx, max_tile_xy)
                y0 = (ny - cy) // 2
                x0 = (nx - cx) // 2
                cropped.append(ch[y0:y0+cy, x0:x0+cx])
        images = cropped
        if images[0].ndim == 3:
            meta["size_z"], meta["size_y"], meta["size_x"] = images[0].shape
        else:
            meta["size_y"], meta["size_x"] = images[0].shape
        crop_path = out_dir / f"{stem}_bench_crop.ome.tiff"
        stack = np.stack(images, axis=0)
        tifffile.imwrite(str(crop_path), stack)
        img_path = crop_path
        print(f"  Cropped to: {images[0].shape}")

    all_metrics: dict[str, dict] = {}
    available_methods = list(_BENCH_METHODS)

    print(f"\n  Benchmarking {len(available_methods)} method(s)")

    common_kw = dict(
        tiling=tiling,
        max_tile_xy=max_tile_xy,
        max_tile_z=max_tile_z,
        device=device,
        na=na,
        refractive_index=refractive_index,
        sample_refractive_index=sample_ri,
        microscope_type=microscope_type,
        emission_wavelengths=emission_wavelengths,
        excitation_wavelengths=excitation_wavelengths,
        pixel_size_xy=pixel_size_xy,
        pixel_size_z=pixel_size_z,
        background=background,
        damping=damping,
        convergence=convergence,
        rel_threshold=rel_threshold,
        check_every=check_every,
        ri_coverslip=ri_coverslip,
        ri_coverslip_design=ri_coverslip_design,
        ri_immersion_design=ri_immersion_design,
        t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
    )

    for m in available_methods:
        label = f"{m}_{nit_tag}i"
        print(f"\n  -- {m}, {nit_label} iterations --")
        try:
            monitor = _MetricsMonitor()
            monitor.start()
            result = deconvolve_image(
                img_path,
                method=m,
                niter=niter_list,
                tv_lambda=tv_lambda if m == "ci_rl_tv" else 0.0,
                **common_kw,
            )
            out_name = f"{stem}_{m}_{nit_tag}i.ome.tiff"
            out_file = out_dir / out_name
            save_result(result, str(out_file), mip_only=True)
            metrics = monitor.stop()
            metrics["device"] = _method_device(m)
            
            # Compute image quality metrics
            quality = _quality_metrics(result["channels"])
            metrics.update(quality)
            
            all_metrics[label] = metrics
            print(f"    {metrics['time_s']:.1f}s"
                  f"  RAM d{_format_bytes(metrics['ram_delta_peak_mb'])}"
                  f"  GPU d{_format_bytes(metrics['gpu_mem_delta_peak_mb'])}"
                  f" -> {out_name}")
            del result

        except ValueError as exc:
            print(f"    SKIPPED: {exc}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            import traceback
            traceback.print_exc()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Wait for GPU memory to settle
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                prev_used = None
                for _ in range(10):
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if prev_used is not None and mem.used == prev_used:
                        break
                    prev_used = mem.used
                    time.sleep(0.5)
            except Exception:
                time.sleep(3)
        else:
            time.sleep(2)

    # --- Metrics summary ---
    if all_metrics:
        _print_metrics_summary(all_metrics)

    # --- CSV export ---
    csv_path = out_dir / f"benchmark_metrics_{stem}.csv"
    _write_metrics_csv(csv_path, all_metrics)

    # --- Montages ---
    _make_benchmark_montage(
        str(out_dir), stem, available_methods,
        bench_iterations, all_metrics, meta,
    )
    _make_per_channel_montages(
        str(out_dir), stem, available_methods,
        bench_iterations, all_metrics, meta,
    )

    # Clean up: keep only CSV and montage PNGs
    keep_prefixes = ("benchmark_metrics_", "decon_benchmark_")
    for f in out_dir.iterdir():
        if f.is_file() and not f.name.startswith(keep_prefixes):
            f.unlink()
            print(f"  Cleaned: {f.name}")


def _print_metrics_summary(all_metrics):
    """Print a formatted table of benchmark metrics to stdout."""
    hdr = (f"  {'Method':<25} {'Device':>6} {'Time':>7} {'CPU%':>6}"
           f" {'RAM pk':>8} {'RAM d':>8}"
           f" {'GPU%':>6} {'VRAM pk':>8} {'GPU d':>8}")
    sep = "  " + "-" * len(hdr.strip())
    print(f"\n{sep}")
    print(f"  Benchmark metrics summary:")
    print(f"{sep}")
    print(hdr)
    print(f"{sep}")
    for lbl, m in sorted(all_metrics.items()):
        gpu_delta = (m['torch_gpu_delta_mb']
                     if m.get('torch_gpu_delta_mb', 0) > 0
                     else m['gpu_mem_delta_peak_mb'])
        print(f"  {lbl:<25} {m.get('device', '?'):>6}"
              f" {m['time_s']:>6.1f}s"
              f" {m['cpu_percent_avg']:>5.0f}%"
              f" {_format_bytes(m['ram_peak_mb']):>8}"
              f" {_format_bytes(m['ram_delta_peak_mb']):>8}"
              f" {m['gpu_util_avg']:>5.0f}%"
              f" {_format_bytes(m['gpu_mem_peak_mb']):>8}"
              f" {_format_bytes(gpu_delta):>8}")
    print(f"{sep}")


def _stem(filename: str) -> str:
    """Derive a clean output stem from an image filename."""
    stem = Path(filename).stem
    for ext in (".tiff", ".tif", ".ome"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return stem


if __name__ == "__main__":
    main(sys.argv[1:])