from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from core.deconvolve_ci import (
    _bertero_weights,
    _blend_tile,
    _compute_tile_grid,
    _compute_tile_slices,
    _estimate_background,
    _initial_estimate,
    _prepare_otf,
    _resolve_start_mode,
    _resolve_tiling,
    _sparse_hessian_penalty,
    _to_numpy,
    _to_tensor,
    _tv_penalty,
    ci_rl_deconvolve,
    ci_sparse_hessian_deconvolve,
)


@dataclass
class SolverResult:
    result: np.ndarray
    wall_time_s: float
    gpu_time_s: float
    setup_time_s: float
    iteration_time_s: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    work_shape: tuple[int, ...]
    graph_used: bool
    extension_used: bool
    tile_count: int = 1
    tile_grid: tuple[int, int] = (1, 1)
    backend: str = "torch"
    workspace_mb: float = 0.0
    static_precision: str = "fp32"
    tile_margin: int = 16
    z_partitions: int = 1
    data_step_time_s: float = 0.0
    regularizer_time_s: float = 0.0
    method: str = "ci_rl"

    def metrics(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("result")
        return payload


def crop_psf_to_image(psf: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    slices = []
    for psf_size, image_size in zip(psf.shape, image_shape):
        if psf_size > image_size:
            start = (psf_size - image_size) // 2
            slices.append(slice(start, start + image_size))
        else:
            slices.append(slice(None))
    return np.ascontiguousarray(psf[tuple(slices)], dtype=np.float32)


def minimum_work_shape(image_shape: tuple[int, ...], psf_shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(i + p - 1 for i, p in zip(image_shape, psf_shape))


def _prepare(
    image: np.ndarray,
    psf: np.ndarray,
    work_shape: tuple[int, ...],
    *,
    start: str = "auto",
    background: str | float = "auto",
    offset: float = 5.0,
) -> dict[str, Any]:
    dev = torch.device("cuda")
    dtype = torch.float32
    img_t = _to_tensor(image, dev, dtype)
    psf_t = _to_tensor(psf, dev, dtype)
    bg = max(_estimate_background(img_t), 1e-6) if background == "auto" else max(float(background), 1e-6)
    if offset > 0:
        img_t = img_t + offset
        bg += offset
    start = _resolve_start_mode(start, img_t, bg, "widefield")
    otf, otf_conj = _prepare_otf(psf_t, work_shape)
    weights = _bertero_weights(otf, otf_conj, img_t.shape, work_shape)
    observed_work = torch.full(work_shape, bg, dtype=dtype, device=dev)
    support = tuple(slice(0, size) for size in img_t.shape)
    observed_work[support] = img_t
    x_prev, x_cur = _initial_estimate(start, img_t, observed_work, work_shape, support, bg, dtype, dev)
    return {
        "img": img_t,
        "psf": psf_t,
        "otf": otf,
        "otf_conj": otf_conj,
        "weights": weights,
        "x_prev": x_prev,
        "x_cur": x_cur,
        "bg": bg,
        "support": support,
        "offset": offset,
        "observed_work": observed_work,
    }


def _iteration(
    state: dict[str, Any],
    k: int,
    *,
    fused_ops: Any | None,
) -> None:
    x_prev, x_cur = state["x_prev"], state["x_cur"]
    if k >= 3:
        alpha_max = 1.0 - 2.0 / math.sqrt(k + 3.0)
        alpha = min((k - 1.0) / (k + 2.0), alpha_max)
    else:
        alpha = 0.0
    p, spatial, ratio, frequency = state["p"], state["spatial"], state.get("ratio"), state["frequency"]
    bg = state["bg"]
    if fused_ops is not None:
        fused_ops.momentum(x_cur, x_prev, alpha, bg, p)
    else:
        torch.sub(x_cur, x_prev, out=p)
        p.mul_(alpha).add_(x_cur).clamp_(min=bg)

    torch.fft.rfftn(p, out=frequency)
    frequency.mul_(state["otf"])
    torch.fft.irfftn(frequency, s=state["work_shape"], out=spatial)

    if fused_ops is not None:
        fused_ops.ratio(spatial, state["img"], *state["img"].shape, bg)
        ratio_source = spatial
    else:
        ratio.zero_()
        view = ratio[state["support"]]
        torch.div(state["img"], spatial[state["support"]].clamp(min=bg), out=view)
        ratio_source = ratio

    torch.fft.rfftn(ratio_source, out=frequency)
    frequency.mul_(state["otf_conj"])
    torch.fft.irfftn(frequency, s=state["work_shape"], out=spatial)
    if fused_ops is not None:
        fused_ops.update(p, spatial, state["weights"], bg, x_prev)
    else:
        torch.mul(p, spatial, out=x_prev)
        x_prev.mul_(state["weights"]).clamp_(min=bg)
    state["x_prev"], state["x_cur"] = x_cur, x_prev


def _iteration_overwrite(state: dict[str, Any], k: int, *, fused_ops: Any) -> None:
    x_prev, x_cur = state["x_prev"], state["x_cur"]
    alpha = min((k - 1.0) / (k + 2.0), 1.0 - 2.0 / math.sqrt(k + 3.0)) if k >= 3 else 0.0
    spatial, frequency = state["spatial"], state["frequency"]
    bg = state["bg"]
    # The previous estimate is dead after momentum. Reuse it as p and later as x_new.
    fused_ops.momentum(x_cur, x_prev, alpha, bg, x_prev)
    torch.fft.rfftn(x_prev, out=frequency)
    frequency.mul_(state["otf"])
    torch.fft.irfftn(frequency, s=state["work_shape"], out=spatial)
    fused_ops.ratio(spatial, state["img"], *state["img"].shape, bg)
    torch.fft.rfftn(spatial, out=frequency)
    frequency.mul_(state["otf_conj"])
    torch.fft.irfftn(frequency, s=state["work_shape"], out=spatial)
    fused_ops.update(x_prev, spatial, state["weights"], bg, x_prev)
    state["x_prev"], state["x_cur"] = x_cur, x_prev


def run_buffered(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    work_shape: tuple[int, ...],
    niter: int,
    fused_ops: Any | None = None,
    use_graph: bool = False,
    overwrite_state: bool = False,
) -> SolverResult:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    setup_start = time.perf_counter()
    state = _prepare(image, psf, work_shape)
    state["work_shape"] = work_shape
    state["p"] = None if overwrite_state else torch.empty(work_shape, dtype=torch.float32, device="cuda")
    state["spatial"] = torch.empty(work_shape, dtype=torch.float32, device="cuda")
    state["ratio"] = torch.empty(work_shape, dtype=torch.float32, device="cuda") if fused_ops is None else None
    frequency_shape = work_shape[:-1] + (work_shape[-1] // 2 + 1,)
    state["frequency"] = torch.empty(frequency_shape, dtype=torch.complex64, device="cuda")
    buffer_prev, buffer_cur = state["x_prev"], state["x_cur"]
    seed_prev = buffer_prev.clone() if use_graph else None
    seed_cur = buffer_cur.clone() if use_graph else None
    del state["observed_work"]
    torch.cuda.synchronize()
    setup_time = time.perf_counter() - setup_start

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    graph_used = False
    start_event.record()
    iteration_start = time.perf_counter()
    if use_graph:
        # Plan creation must happen outside capture. Reset estimates after warm-up.
        iteration = _iteration_overwrite if overwrite_state else _iteration
        iteration(state, 1, fused_ops=fused_ops)
        torch.cuda.synchronize()
        buffer_prev.copy_(seed_prev)
        buffer_cur.copy_(seed_cur)
        state["x_prev"], state["x_cur"] = buffer_prev, buffer_cur
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                for k in range(1, niter + 1):
                    iteration(state, k, fused_ops=fused_ops)
            buffer_prev.copy_(seed_prev)
            buffer_cur.copy_(seed_cur)
            graph.replay()
            graph_used = True
        except Exception:
            torch.cuda.synchronize()
            state["x_prev"] = seed_prev.clone()
            state["x_cur"] = seed_cur.clone()
            for k in range(1, niter + 1):
                iteration(state, k, fused_ops=fused_ops)
    else:
        iteration = _iteration_overwrite if overwrite_state else _iteration
        for k in range(1, niter + 1):
            iteration(state, k, fused_ops=fused_ops)
    end_event.record()
    torch.cuda.synchronize()
    iteration_time = time.perf_counter() - iteration_start
    gpu_time = start_event.elapsed_time(end_event) / 1000.0
    result = state["x_cur"][state["support"]]
    if state["offset"] > 0:
        result = (result - state["offset"]).clamp(min=0.0)
    result_np = _to_numpy(result).astype(np.float32, copy=False)
    wall_time = time.perf_counter() - wall_start
    return SolverResult(
        result=result_np,
        wall_time_s=wall_time,
        gpu_time_s=gpu_time,
        setup_time_s=setup_time,
        iteration_time_s=iteration_time,
        peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
        work_shape=work_shape,
        graph_used=graph_used,
        extension_used=fused_ops is not None,
        backend="torch_overwrite" if overwrite_state else "torch_buffered",
    )


def _direct_iteration(state: dict[str, Any], k: int, fused_ops: Any) -> None:
    x_prev, x_cur = state["x_prev"], state["x_cur"]
    alpha = min((k - 1.0) / (k + 2.0), 1.0 - 2.0 / math.sqrt(k + 3.0)) if k >= 3 else 0.0
    shape = state["work_shape"]
    pitch = state["pitch"]
    storage = state["storage"]
    plan, workspace = state["plan"], state["workspace"]
    fused_ops.momentum_pack(x_cur, x_prev, alpha, state["bg"], storage, *shape, pitch)
    plan.forward(storage, workspace)
    if state["static_precision"] == "fp16":
        fused_ops.multiply_otf_half(state["frequency"], state["otf"], False)
    else:
        fused_ops.multiply_otf(state["frequency"], state["otf"], False)
    plan.inverse(storage, workspace)
    fused_ops.ratio_pitched(storage, state["img"], *shape, pitch, state["bg"], state["normalization"])
    plan.forward(storage, workspace)
    if state["static_precision"] == "fp16":
        fused_ops.multiply_otf_half(state["frequency"], state["otf"], True)
    else:
        fused_ops.multiply_otf(state["frequency"], state["otf"], True)
    plan.inverse(storage, workspace)
    if state["static_precision"] == "fp16":
        fused_ops.update_pitched_half(x_prev, storage, state["weights"], state["bg"], state["normalization"], *shape, pitch)
    else:
        fused_ops.update_pitched(x_prev, storage, state["weights"], state["bg"], state["normalization"], *shape, pitch)
    state["x_prev"], state["x_cur"] = x_cur, x_prev


def prepare_direct_static(
    psf: np.ndarray,
    image_shape: tuple[int, ...],
    work_shape: tuple[int, ...],
    fused_ops: Any,
    static_precision: str,
) -> dict[str, Any]:
    psf_t = _to_tensor(psf, torch.device("cuda"), torch.float32)
    otf, otf_conj = _prepare_otf(psf_t, work_shape)
    weights = _bertero_weights(otf, otf_conj, image_shape, work_shape)
    plan = fused_ops.DirectFFTPlan(*work_shape)
    pitch = int(plan.physical_x)
    storage = torch.empty((work_shape[0], work_shape[1], pitch), dtype=torch.float32, device="cuda")
    frequency = storage.view(torch.complex64).view(work_shape[0], work_shape[1], work_shape[2] // 2 + 1)
    workspace = torch.empty(max(int(plan.workspace_bytes), 1), dtype=torch.uint8, device="cuda")
    if static_precision == "fp16":
        otf = torch.view_as_real(otf).to(torch.float16).contiguous()
        weights = weights.to(torch.float16).contiguous()
    elif static_precision == "fp32":
        otf = otf.resolve_conj().contiguous()
        weights = weights.contiguous()
    else:
        raise ValueError(f"unsupported static precision {static_precision}")
    return {
        "otf": otf,
        "weights": weights,
        "plan": plan,
        "pitch": pitch,
        "storage": storage,
        "frequency": frequency,
        "workspace": workspace,
        "work_shape": work_shape,
        "normalization": float(math.prod(work_shape)),
        "static_precision": static_precision,
        "workspace_mb": int(plan.workspace_bytes) / 1024**2,
    }


def _prepare_direct_dynamic(
    image: np.ndarray,
    static: dict[str, Any],
    *,
    offset: float = 5.0,
    microscope_type: str = "widefield",
) -> dict[str, Any]:
    img_t = _to_tensor(image, torch.device("cuda"), torch.float32)
    bg = max(_estimate_background(img_t), 1e-6)
    if offset > 0:
        img_t = img_t + offset
        bg += offset
    start = _resolve_start_mode("auto", img_t, bg, microscope_type)
    work_shape = static["work_shape"]
    support = tuple(slice(0, size) for size in img_t.shape)
    observed_work = torch.full(work_shape, bg, dtype=torch.float32, device="cuda")
    observed_work[support] = img_t
    x_prev, x_cur = _initial_estimate(start, img_t, observed_work, work_shape, support, bg, torch.float32, torch.device("cuda"))
    state = dict(static)
    state.update(img=img_t, bg=bg, support=support, offset=offset, x_prev=x_prev, x_cur=x_cur)
    del observed_work
    return state


def run_direct_cufft(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    work_shape: tuple[int, ...],
    niter: int,
    fused_ops: Any,
    static_precision: str = "fp32",
    static_context: dict[str, Any] | None = None,
    manage_memory: bool = True,
) -> SolverResult:
    if fused_ops is None or not hasattr(fused_ops, "DirectFFTPlan"):
        raise RuntimeError("direct cuFFT extension is unavailable")
    if len(work_shape) != 3:
        raise ValueError("direct cuFFT benchmark currently requires a 3D volume")
    if manage_memory:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    setup_start = time.perf_counter()
    static = static_context or prepare_direct_static(psf, image.shape, work_shape, fused_ops, static_precision)
    state = _prepare_direct_dynamic(image, static)
    torch.cuda.synchronize()
    setup_time = time.perf_counter() - setup_start
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    iteration_start = time.perf_counter()
    for k in range(1, niter + 1):
        _direct_iteration(state, k, fused_ops)
    end_event.record()
    torch.cuda.synchronize()
    iteration_time = time.perf_counter() - iteration_start
    result = state["x_cur"][state["support"]]
    if state["offset"] > 0:
        result = (result - state["offset"]).clamp(min=0.0)
    result_np = _to_numpy(result).astype(np.float32, copy=False)
    return SolverResult(
        result=result_np,
        wall_time_s=time.perf_counter() - wall_start,
        gpu_time_s=start_event.elapsed_time(end_event) / 1000.0,
        setup_time_s=setup_time,
        iteration_time_s=iteration_time,
        peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
        work_shape=work_shape,
        graph_used=False,
        extension_used=True,
        backend="direct_cufft_inplace",
        workspace_mb=static["workspace_mb"],
        static_precision=static_precision,
    )


def run_direct_regularized(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    work_shape: tuple[int, ...],
    niter: int,
    fused_ops: Any,
    method: str,
    tv_lambda: float = 1e-4,
    sparse_hessian_weight: float = 0.6,
    sparse_hessian_reg: float = 0.98,
    fused_regularizer: bool = False,
    pixel_size_xy: float | None = None,
    pixel_size_z: float | None = None,
) -> SolverResult:
    """Benchmark direct FP32 cuFFT with production TV or sparse-Hessian math."""
    if method not in {"ci_rl_tv", "ci_sparse_hessian"}:
        raise ValueError("method must be ci_rl_tv or ci_sparse_hessian")
    if fused_ops is None or not hasattr(fused_ops, "DirectFFTPlan"):
        raise RuntimeError("direct cuFFT extension is unavailable")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    setup_start = time.perf_counter()
    static = prepare_direct_static(psf, image.shape, work_shape, fused_ops, "fp32")
    state = _prepare_direct_dynamic(
        image, static,
        microscope_type="" if method == "ci_sparse_hessian" else "widefield",
    )
    z_scale = (
        max(float(pixel_size_xy), 1e-12) / max(float(pixel_size_z), 1e-12)
        if image.ndim == 3 and pixel_size_xy is not None and pixel_size_z is not None
        else 1.0
    )
    state["axis_scales"] = (z_scale, 1.0, 1.0)
    state["z_scale"] = z_scale
    state["sparse_gradient"] = (
        torch.empty(image.shape, dtype=torch.float32, device="cuda")
        if fused_regularizer and method == "ci_sparse_hessian" else None
    )
    torch.cuda.synchronize()
    setup_time = time.perf_counter() - setup_start

    data_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    reg_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()
    iteration_start = time.perf_counter()
    for k in range(1, niter + 1):
        data_start = torch.cuda.Event(enable_timing=True)
        data_end = torch.cuda.Event(enable_timing=True)
        reg_end = torch.cuda.Event(enable_timing=True)
        data_start.record()
        _direct_iteration(state, k, fused_ops)
        data_end.record()
        x_new = state["x_cur"]
        if method == "ci_rl_tv":
            if fused_regularizer:
                fused_ops.tv_update(
                    x_new, state["storage"], tv_lambda,
                    *state["axis_scales"], state["bg"],
                )
            else:
                x_new.mul_(_tv_penalty(x_new, tv_lambda, state["axis_scales"]))
                x_new.clamp_(min=state["bg"])
        else:
            support = state["support"]
            if fused_regularizer:
                prior_grad = state["sparse_gradient"]
                fused_ops.sparse_hessian_gradient(
                    x_new, prior_grad, *image.shape,
                    sparse_hessian_weight, state["z_scale"],
                )
                grad_scale = (torch.linalg.vector_norm(prior_grad, ord=1) / prior_grad.numel()).detach().clamp(min=1e-12)
            else:
                prior_probe = x_new[support].detach().requires_grad_(True)
                prior_loss = _sparse_hessian_penalty(
                    prior_probe, sparse_hessian_weight, z_scale=state["z_scale"],
                )
                prior_grad = torch.autograd.grad(prior_loss, prior_probe)[0]
                grad_scale = prior_grad.abs().mean().detach().clamp(min=1e-12)
            signal_scale = max(float((x_new[support].mean() - state["bg"]).detach()), 1.0)
            reg_step = 0.1 * max(1.0 - sparse_hessian_reg, 0.0) * signal_scale
            if fused_regularizer:
                fused_ops.sparse_hessian_update(
                    x_new, prior_grad, *image.shape,
                    reg_step / float(grad_scale), state["bg"],
                )
            else:
                with torch.no_grad():
                    x_new[support].sub_(reg_step * prior_grad / grad_scale).clamp_(min=state["bg"])
        reg_end.record()
        data_events.append((data_start, data_end))
        reg_events.append((data_end, reg_end))
    total_end.record()
    torch.cuda.synchronize()
    iteration_time = time.perf_counter() - iteration_start
    data_time = sum(start.elapsed_time(end) for start, end in data_events) / 1000.0
    regularizer_time = sum(start.elapsed_time(end) for start, end in reg_events) / 1000.0
    result = state["x_cur"][state["support"]]
    if state["offset"] > 0:
        result = (result - state["offset"]).clamp(min=0.0)
    return SolverResult(
        result=_to_numpy(result).astype(np.float32, copy=False),
        wall_time_s=time.perf_counter() - wall_start,
        gpu_time_s=total_start.elapsed_time(total_end) / 1000.0,
        setup_time_s=setup_time,
        iteration_time_s=iteration_time,
        peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
        work_shape=work_shape,
        graph_used=False,
        extension_used=True,
        backend=("direct_cufft_plus_fused_regularizer" if fused_regularizer else "direct_cufft_plus_production_regularizer"),
        workspace_mb=static["workspace_mb"],
        static_precision="fp32",
        data_step_time_s=data_time,
        regularizer_time_s=regularizer_time,
        method=method,
    )


def validate_fused_regularizers(fused_ops: Any) -> dict[str, float | bool]:
    """Compare benchmark CUDA kernels with the production PyTorch formulas."""
    torch.manual_seed(1234)
    floor = 0.25
    tv_source = torch.rand((7, 8, 9), device="cuda", dtype=torch.float32) * 20.0 + floor
    tv_expected = (tv_source * _tv_penalty(tv_source, 1e-4, (0.7, 1.0, 1.0))).clamp(min=floor)
    tv_actual = tv_source.clone()
    tv_scratch = torch.empty((7, 8, 11), device="cuda", dtype=torch.float32)
    fused_ops.tv_update(tv_actual, tv_scratch, 1e-4, 0.7, 1.0, 1.0, floor)

    full = torch.rand((8, 10, 12), device="cuda", dtype=torch.float32) * 20.0 + floor
    support_shape = (7, 8, 9)
    support = tuple(slice(0, size) for size in support_shape)
    probe = full[support].detach().requires_grad_(True)
    loss = _sparse_hessian_penalty(probe, 0.6, z_scale=0.7)
    expected_grad = torch.autograd.grad(loss, probe)[0]
    actual_grad = torch.empty(support_shape, device="cuda", dtype=torch.float32)
    fused_ops.sparse_hessian_gradient(full, actual_grad, *support_shape, 0.6, 0.7)
    expected_norm = expected_grad / expected_grad.abs().mean().clamp(min=1e-12)
    actual_norm = actual_grad / actual_grad.abs().mean().clamp(min=1e-12)
    torch.cuda.synchronize()
    tv_abs = float((tv_actual - tv_expected).abs().max())
    sparse_abs = float((actual_norm - expected_norm).abs().max())
    sparse_rmse = float(torch.sqrt(torch.mean((actual_norm - expected_norm) ** 2)))
    full_2d = torch.rand((1, 9, 11), device="cuda", dtype=torch.float32) * 20.0 + floor
    support_2d_shape = (1, 8, 9)
    support_2d = tuple(slice(0, size) for size in support_2d_shape)
    probe_2d = full_2d[support_2d].detach().requires_grad_(True)
    loss_2d = _sparse_hessian_penalty(probe_2d, 0.6, z_scale=0.3)
    expected_2d = torch.autograd.grad(loss_2d, probe_2d)[0]
    actual_2d = torch.empty(support_2d_shape, device="cuda", dtype=torch.float32)
    fused_ops.sparse_hessian_gradient(full_2d, actual_2d, *support_2d_shape, 0.6, 0.3)
    expected_2d_norm = expected_2d / expected_2d.abs().mean().clamp(min=1e-12)
    actual_2d_norm = actual_2d / actual_2d.abs().mean().clamp(min=1e-12)
    sparse_2d_abs = float((actual_2d_norm - expected_2d_norm).abs().max())
    sparse_2d_rmse = float(torch.sqrt(torch.mean((actual_2d_norm - expected_2d_norm) ** 2)))
    return {
        "tv_max_abs": tv_abs,
        "tv_pass": tv_abs <= 1e-4,
        "sparse_normalized_gradient_max_abs": sparse_abs,
        "sparse_normalized_gradient_rmse": sparse_rmse,
        "sparse_pass": sparse_abs <= 2e-4 and sparse_rmse <= 1e-5,
        "sparse_2d_normalized_gradient_max_abs": sparse_2d_abs,
        "sparse_2d_normalized_gradient_rmse": sparse_2d_rmse,
        "sparse_2d_pass": sparse_2d_abs <= 2e-4 and sparse_2d_rmse <= 1e-5,
    }


def run_direct_tiled(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    niter: int,
    n_tiles: int,
    shape_policy,
    fused_ops: Any,
    static_precision: str = "fp32",
    margin: int = 16,
    cache_static: bool = True,
) -> SolverResult:
    image3 = image if image.ndim == 3 else image[np.newaxis, ...]
    psf3 = psf if psf.ndim == 3 else psf[np.newaxis, ...]
    ny, nx = _compute_tile_grid(image3.shape[1:], n_tiles)
    descriptors = _compute_tile_slices(image3.shape, ny, nx, max(psf3.shape[-2:]) // 2)
    jobs = []
    for desc in descriptors:
        _, ey, ex = desc["extract"]
        y0, y1 = max(ey.start - margin, 0), min(ey.stop + margin, image3.shape[1])
        x0, x1 = max(ex.start - margin, 0), min(ex.stop + margin, image3.shape[2])
        shape = (image3.shape[0], y1 - y0, x1 - x0)
        tile_psf = crop_psf_to_image(psf3, shape)
        work = shape_policy(minimum_work_shape(shape, tile_psf.shape))
        jobs.append((desc, y0, y1, x0, x1, tile_psf, work))
    # Reordering equal geometries lets one OTF, weight map, plan and arena serve all matching tiles.
    jobs.sort(key=lambda job: (job[6], job[1] == 0, job[3] == 0))
    numerator = np.zeros_like(image3, dtype=np.float32)
    denominator = np.zeros_like(image3, dtype=np.float32)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    total_gpu = total_setup = total_iter = 0.0
    context = None
    context_key = None
    representative_work = jobs[0][6]
    max_workspace = 0.0
    for desc, y0, y1, x0, x1, tile_psf, work in jobs:
        tile = np.ascontiguousarray(image3[:, y0:y1, x0:x1], dtype=np.float32)
        key = (tile.shape, work)
        if not cache_static or context is None or key != context_key:
            del context
            torch.cuda.empty_cache()
            context = prepare_direct_static(tile_psf, tile.shape, work, fused_ops, static_precision)
            context_key = key
        result = run_direct_cufft(tile, tile_psf, work_shape=work, niter=niter, fused_ops=fused_ops, static_precision=static_precision, static_context=context, manage_memory=False)
        _, ey, ex = desc["extract"]
        cropped = result.result[:, ey.start - y0:ey.stop - y0, ex.start - x0:ex.stop - x0]
        weighted, weight = _blend_tile(cropped, desc)
        numerator[desc["extract"]] += weighted
        denominator[desc["extract"]] += weight[np.newaxis]
        total_gpu += result.gpu_time_s
        total_setup += result.setup_time_s
        total_iter += result.iteration_time_s
        max_workspace = max(max_workspace, result.workspace_mb)
        del result
    output = numerator / np.maximum(denominator, 1e-8)
    if image.ndim == 2:
        output = output[0]
    torch.cuda.synchronize()
    return SolverResult(
        result=output.astype(np.float32, copy=False), wall_time_s=time.perf_counter() - wall_start,
        gpu_time_s=total_gpu, setup_time_s=total_setup, iteration_time_s=total_iter,
        peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
        work_shape=representative_work, graph_used=False, extension_used=True,
        tile_count=len(jobs), tile_grid=(ny, nx), backend="direct_cufft_cached" if cache_static else "direct_cufft",
        workspace_mb=max_workspace, static_precision=static_precision, tile_margin=margin,
    )


def run_direct_z_partitioned(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    niter: int,
    z_partitions: int,
    shape_policy,
    fused_ops: Any,
    static_precision: str = "fp32",
) -> SolverResult:
    """Experimental axial overlap-save RL approximation.

    Each axial core is solved with a halo and a PSF cropped to that local axial
    support. This is intentionally quality-gated because partitioning the RL
    state itself is not algebraically identical to partitioning one convolution.
    """
    if image.ndim != 3 or psf.ndim != 3:
        raise ValueError("axial partition benchmark requires 3D image and PSF")
    boundaries = [round(i * image.shape[0] / z_partitions) for i in range(z_partitions + 1)]
    halo = max(1, min(psf.shape[0] // 4, image.shape[0] // 3))
    output = np.zeros_like(image, dtype=np.float32)
    total_wall = total_gpu = total_setup = total_iter = 0.0
    peak_alloc = peak_reserved = max_workspace = 0.0
    representative_work = None
    for index in range(z_partitions):
        core0, core1 = boundaries[index], boundaries[index + 1]
        ext0, ext1 = max(0, core0 - halo), min(image.shape[0], core1 + halo)
        tile = np.ascontiguousarray(image[ext0:ext1], dtype=np.float32)
        tile_psf = crop_psf_to_image(psf, tile.shape)
        work = shape_policy(minimum_work_shape(tile.shape, tile_psf.shape))
        representative_work = work
        result = run_direct_cufft(tile, tile_psf, work_shape=work, niter=niter, fused_ops=fused_ops, static_precision=static_precision)
        output[core0:core1] = result.result[core0 - ext0:core1 - ext0]
        total_wall += result.wall_time_s
        total_gpu += result.gpu_time_s
        total_setup += result.setup_time_s
        total_iter += result.iteration_time_s
        peak_alloc = max(peak_alloc, result.peak_allocated_mb)
        peak_reserved = max(peak_reserved, result.peak_reserved_mb)
        max_workspace = max(max_workspace, result.workspace_mb)
        del result
    return SolverResult(
        result=output, wall_time_s=total_wall, gpu_time_s=total_gpu,
        setup_time_s=total_setup, iteration_time_s=total_iter,
        peak_allocated_mb=peak_alloc, peak_reserved_mb=peak_reserved,
        work_shape=representative_work or image.shape, graph_used=False, extension_used=True,
        backend="direct_cufft_z_partitioned", workspace_mb=max_workspace,
        static_precision=static_precision, z_partitions=z_partitions,
    )


def run_production(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    niter: int,
    tiling: str | int = "none",
    method: str = "ci_rl",
    tv_lambda: float = 1e-4,
    sparse_hessian_weight: float = 0.6,
    sparse_hessian_reg: float = 0.98,
    pixel_size_xy: float | None = None,
    pixel_size_z: float | None = None,
) -> SolverResult:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    resolved_tiles = _resolve_tiling(
        tiling, image.shape, device="cuda", psf_xy_est=max(psf.shape[-2:]),
    )
    common = {
        "image": image,
        "psf": psf,
        "niter": niter,
        "background": "auto",
        "offset": 5.0,
        "start": "auto",
        "convergence": "fixed",
        "check_every": max(niter, 1),
        "device": "cuda",
        "backend": "pytorch_cuda",
        "tiling": tiling,
        "pixel_size_xy": pixel_size_xy,
        "pixel_size_z": pixel_size_z,
    }
    if method == "ci_sparse_hessian":
        output = ci_sparse_hessian_deconvolve(
            **common,
            sparse_hessian_weight=sparse_hessian_weight,
            sparse_hessian_reg=sparse_hessian_reg,
        )
    else:
        output = ci_rl_deconvolve(
            **common,
            tv_lambda=tv_lambda if method == "ci_rl_tv" else 0.0,
            microscope_type="widefield",
            two_d_mode="legacy_2d",
        )
    end_event.record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - wall_start
    return SolverResult(
        result=output["result"],
        wall_time_s=elapsed,
        gpu_time_s=start_event.elapsed_time(end_event) / 1000.0,
        setup_time_s=0.0,
        iteration_time_s=elapsed,
        peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
        work_shape=minimum_work_shape(image.shape, psf.shape),
        graph_used=False,
        extension_used=False,
        tile_count=resolved_tiles,
        tile_grid=_compute_tile_grid(image.shape[-2:], resolved_tiles),
        method=method,
    )


def run_buffered_tiled(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    niter: int,
    n_tiles: int,
    shape_policy,
    fused_ops: Any | None = None,
) -> SolverResult:
    image3 = image if image.ndim == 3 else image[np.newaxis, ...]
    psf3 = psf if psf.ndim == 3 else psf[np.newaxis, ...]
    ny, nx = _compute_tile_grid(image3.shape[1:], n_tiles)
    overlap = max(psf3.shape[-2:]) // 2
    tiles = _compute_tile_slices(image3.shape, ny, nx, overlap)
    numerator = np.zeros_like(image3, dtype=np.float32)
    denominator = np.zeros_like(image3, dtype=np.float32)
    total_wall = total_gpu = total_setup = total_iter = peak_alloc = peak_reserved = 0.0
    representative_work = None
    for desc in tiles:
        _, ey, ex = desc["extract"]
        margin = 16
        y0, y1 = max(ey.start - margin, 0), min(ey.stop + margin, image3.shape[1])
        x0, x1 = max(ex.start - margin, 0), min(ex.stop + margin, image3.shape[2])
        tile = np.ascontiguousarray(image3[:, y0:y1, x0:x1], dtype=np.float32)
        tile_psf = crop_psf_to_image(psf3, tile.shape)
        minimum = minimum_work_shape(tile.shape, tile_psf.shape)
        work = shape_policy(minimum)
        representative_work = work
        result = run_buffered(tile, tile_psf, work_shape=work, niter=niter, fused_ops=fused_ops)
        crop_y0 = ey.start - y0
        crop_y1 = crop_y0 + (ey.stop - ey.start)
        crop_x0 = ex.start - x0
        crop_x1 = crop_x0 + (ex.stop - ex.start)
        tile_cropped = result.result[:, crop_y0:crop_y1, crop_x0:crop_x1]
        weighted, weight = _blend_tile(tile_cropped, desc)
        numerator[desc["extract"]] += weighted
        denominator[desc["extract"]] += weight[np.newaxis]
        total_wall += result.wall_time_s
        total_gpu += result.gpu_time_s
        total_setup += result.setup_time_s
        total_iter += result.iteration_time_s
        peak_alloc = max(peak_alloc, result.peak_allocated_mb)
        peak_reserved = max(peak_reserved, result.peak_reserved_mb)
        del result
        torch.cuda.empty_cache()
    output = numerator / np.maximum(denominator, 1e-8)
    if image.ndim == 2:
        output = output[0]
    return SolverResult(
        result=output.astype(np.float32, copy=False),
        wall_time_s=total_wall,
        gpu_time_s=total_gpu,
        setup_time_s=total_setup,
        iteration_time_s=total_iter,
        peak_allocated_mb=peak_alloc,
        peak_reserved_mb=peak_reserved,
        work_shape=representative_work or minimum_work_shape(image3.shape, psf3.shape),
        graph_used=False,
        extension_used=fused_ops is not None,
        tile_count=len(tiles),
        tile_grid=(ny, nx),
    )
