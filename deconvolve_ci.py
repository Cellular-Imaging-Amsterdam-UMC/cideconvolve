"""CI Deconvolve — GPU-accelerated Richardson-Lucy with SHB momentum & PSF generation.

This module provides two public functions:

* ``ci_rl_deconvolve``  – Scaled Heavy Ball (SHB) accelerated Richardson-Lucy
  deconvolution with optional Total Variation regularisation, Bertero boundary
  weights, and I-divergence convergence monitoring.  All heavy lifting runs on
  GPU via PyTorch when a CUDA device is available; CPU fallback is automatic.

* ``ci_generate_psf``  – Physically accurate PSF generation using the vectorial
  Richards-Wolf model (high NA) or scalar Kirchhoff model (lower NA), with
  Gibson-Lanni refractive-index mismatch correction and optional sub-pixel
  integration.

References
----------
[1] Wang & Miller 2014, IEEE TIP 23(2):848-854  (SHB acceleration)
[2] Bertero & Boccacci 2005, A&A 437:369-374     (boundary weights)
[3] Dey et al. 2006, Microsc. Res. Tech. 69:260  (RLTV)
[4] Richards & Wolf 1959, Proc. R. Soc. A 253    (vectorial PSF)
[5] Gibson & Lanni 1991, JOSA A 8(10):1601       (RI mismatch)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.special import bessel_j0, bessel_j1

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — device / dtype
# ---------------------------------------------------------------------------

def _pick_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pick_dtype(dev: torch.device) -> torch.dtype:
    return torch.float32 if dev.type == "cuda" else torch.float64


def _to_tensor(arr: np.ndarray, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device=dev)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

# ===================================================================
#  PART 1 — Richardson-Lucy deconvolution engine
# ===================================================================

# ---------------------------------------------------------------------------
# FFT helpers (rfftn for real-valued data — halves memory vs fftn)
# ---------------------------------------------------------------------------

def _rfft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfftn(x)


def _irfft(X: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.fft.irfftn(X, s=shape)

# ---------------------------------------------------------------------------
# PSF → OTF preparation
# ---------------------------------------------------------------------------

def _prepare_otf(
    psf: torch.Tensor,
    work_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad & circularly-shift PSF, return (OTF, conj(OTF))."""
    # Normalise
    psf = psf / psf.sum()

    # Zero-pad PSF into the work volume
    padded = torch.zeros(work_shape, dtype=psf.dtype, device=psf.device)
    slices = tuple(slice(0, s) for s in psf.shape)
    padded[slices] = psf

    # Circular shift so PSF centre sits at (0, 0, ..., 0)
    shifts = [-(s // 2) for s in psf.shape]
    padded = torch.roll(padded, shifts=shifts, dims=list(range(psf.ndim)))

    otf = _rfft(padded)
    otf_conj = torch.conj(otf)
    return otf, otf_conj

# ---------------------------------------------------------------------------
# Bertero boundary weights  (Bertero & Boccacci 2005)
# ---------------------------------------------------------------------------

def _bertero_weights(
    otf: torch.Tensor,
    otf_conj: torch.Tensor,
    image_shape: tuple[int, ...],
    work_shape: tuple[int, ...],
    sigma: float = 0.01,
) -> torch.Tensor:
    """Compute boundary correction weights W = 1 / H^T(𝟏_M).

    A flat image of ones (image-sized, the data support 𝟏_M) is correlated
    with the PSF (H^T) in the work domain. Where the result exceeds *sigma*
    it is inverted to give W; elsewhere W is zero.  This corrects for
    partial PSF overlap near the data boundary (Bertero & Boccacci 2005).
    """
    ones = torch.zeros(work_shape, dtype=otf.real.dtype, device=otf.device)
    slices = tuple(slice(0, s) for s in image_shape)
    ones[slices] = 1.0

    ones_fft = _rfft(ones)
    # H^T(1_M) = IFFT(conj(H) * FFT(1_M))  — Bertero & Boccacci 2005
    denom_fft = ones_fft * otf_conj
    denom = _irfft(denom_fft, work_shape)

    W = torch.zeros_like(denom)
    mask = denom > sigma
    W[mask] = 1.0 / denom[mask]
    return W

# ---------------------------------------------------------------------------
# I-divergence  (Csiszár 1991)
# ---------------------------------------------------------------------------

def _i_divergence(observed: torch.Tensor, estimated: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute mean I-divergence (generalised KL) for Poisson data."""
    est_safe = estimated.clamp(min=eps)
    obs_safe = observed.clamp(min=eps)
    div = obs_safe * torch.log(obs_safe / est_safe) - obs_safe + est_safe
    return float(div.mean())

# ---------------------------------------------------------------------------
# Total-Variation multiplicative penalty  (Dey et al. 2006 / DL2 RLTV)
# ---------------------------------------------------------------------------

def _tv_penalty(x: torch.Tensor, tv_lambda: float) -> torch.Tensor:
    """Multiplicative TV correction factor.

    Returns a tensor of the same shape as *x* such that
    ``x_new = x * _tv_penalty(x, λ)`` applies one TV step.
    """
    ndim = x.ndim
    eps = 1e-8

    # Forward differences with zero-padded boundary
    grads = []
    for d in range(ndim):
        g = torch.zeros_like(x)
        slc_src = [slice(None)] * ndim
        slc_dst = [slice(None)] * ndim
        slc_src[d] = slice(0, -1)
        slc_dst[d] = slice(1, None)
        g[tuple(slc_dst)] = x[tuple(slc_dst)] - x[tuple(slc_src)]
        grads.append(g)

    # Gradient magnitude
    mag = torch.zeros_like(x)
    for g in grads:
        mag = mag + g * g
    mag = torch.sqrt(mag + eps)

    # Normalised gradients
    normed = [g / mag for g in grads]

    # Backward divergence
    div = torch.zeros_like(x)
    for d, gn in enumerate(normed):
        # Backward difference of normalised gradient
        bg = torch.zeros_like(x)
        slc_src = [slice(None)] * ndim
        slc_dst = [slice(None)] * ndim
        slc_src[d] = slice(1, None)
        slc_dst[d] = slice(0, -1)
        bg[tuple(slc_dst)] = gn[tuple(slc_dst)] - gn[tuple(slc_src)]
        div = div + bg

    # Multiplicative factor
    factor = 1.0 / (1.0 - tv_lambda * div)
    return factor.clamp(min=0.1, max=10.0)

# ---------------------------------------------------------------------------
# Background estimation
# ---------------------------------------------------------------------------

def _estimate_background(image: torch.Tensor) -> float:
    """Estimate background from the mode of the lowest 10% of voxels."""
    flat = image.flatten()
    n = max(int(flat.numel() * 0.1), 1)
    lowest, _ = torch.topk(flat, n, largest=False)
    # Approximate mode via median of lowest decile (robust)
    return float(lowest.median())

# ---------------------------------------------------------------------------
# Top-level RL deconvolution
# ---------------------------------------------------------------------------

def ci_rl_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    niter: int = 50,
    tv_lambda: float = 0.0,
    background: Union[str, float] = "auto",
    convergence: str = "fixed",
    rel_threshold: float = 0.001,
    check_every: int = 5,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """SHB-accelerated Richardson-Lucy deconvolution (GPU / CPU).

    Parameters
    ----------
    image : ndarray
        Observed (noisy) image, 2-D or 3-D.
    psf : ndarray
        Point Spread Function (same dimensionality as image).
    niter : int
        Maximum number of iterations.
    tv_lambda : float
        Total-Variation regularisation strength (0 = disabled).
    background : ``"auto"`` or float
        Background level used as positivity floor and safe-division epsilon.
    convergence : ``"fixed"`` or ``"auto"``
        ``"fixed"`` runs exactly *niter* iterations; ``"auto"`` stops when
        the relative I-divergence change drops below *rel_threshold*.
    rel_threshold : float
        Relative change threshold for auto-convergence.
    check_every : int
        Evaluate I-divergence every *check_every* iterations.
    device : str or None
        PyTorch device (``"cuda"``, ``"cpu"``).  ``None`` = auto.

    Returns
    -------
    dict
        ``"result"`` — deconvolved image (ndarray, same shape as input).
        ``"convergence"`` — list of I-divergence values at check-points.
        ``"iterations_used"`` — number of iterations actually performed.
    """
    dev = _pick_device(device)
    dtype = _pick_dtype(dev)
    ndim = image.ndim

    log.info("ci_rl_deconvolve  device=%s  dtype=%s  shape=%s  niter=%d  "
             "tv_lambda=%.4g  convergence=%s", dev, dtype, image.shape, niter,
             tv_lambda, convergence)

    # Move data to device
    img_t = _to_tensor(image.astype(np.float64), dev, dtype)
    psf_t = _to_tensor(psf.astype(np.float64), dev, dtype)

    # Background
    if background == "auto":
        bg = max(_estimate_background(img_t), 1e-6)
    else:
        bg = max(float(background), 1e-6)
    log.info("  background=%.4g", bg)

    # Work shape = image + psf - 1  (full linear convolution)
    work_shape = tuple(si + sp - 1 for si, sp in zip(img_t.shape, psf_t.shape))

    # Prepare OTF & weights
    otf, otf_conj = _prepare_otf(psf_t, work_shape)
    W = _bertero_weights(otf, otf_conj, img_t.shape, work_shape)

    # Zero-pad observed image into work domain
    d_work = torch.full(work_shape, bg, dtype=dtype, device=dev)
    slices = tuple(slice(0, s) for s in img_t.shape)
    d_work[slices] = img_t

    # Initialise estimate
    x_prev = d_work.clone()
    x_cur = d_work.clone()

    convergence_history: list[float] = []
    use_tv = tv_lambda > 0.0
    iterations_used = niter

    for k in range(1, niter + 1):
        # --- SHB momentum (Wang & Miller 2014) ---
        if k >= 3:
            alpha_max = 1.0 - 2.0 / math.sqrt(k + 3.0)
            alpha = min((k - 1.0) / (k + 2.0), alpha_max)
        else:
            alpha = 0.0
        p = x_cur + alpha * (x_cur - x_prev)
        p = p.clamp(min=bg)

        # --- Forward model: y = H ⊗ p ---
        P_fft = _rfft(p)
        Y_fft = P_fft * otf
        y = _irfft(Y_fft, work_shape)

        # --- Ratio: computed ONLY in the image domain (Bertero formulation) ---
        r = torch.zeros(work_shape, dtype=dtype, device=dev)
        r[slices] = img_t / y[slices].clamp(min=bg)

        # --- Back-project: IFFT(FFT(r) * conj(H)) ---
        R_fft = _rfft(r)
        corr = _irfft(R_fft * otf_conj, work_shape)

        # --- Multiplicative update with Bertero weights ---
        x_new = p * corr * W

        # --- TV regularisation ---
        if use_tv:
            x_new = x_new * _tv_penalty(x_new, tv_lambda)

        # --- Positivity ---
        x_new = x_new.clamp(min=bg)

        x_prev = x_cur
        x_cur = x_new

        # --- Convergence check ---
        if k % check_every == 0 or k == niter:
            # Recompute forward for I-divergence (reuse y from last iter if
            # it's a check iteration — here y is still valid for p, not x_new,
            # but the difference is small; for exactness re-project)
            fwd_fft = _rfft(x_cur) * otf
            fwd = _irfft(fwd_fft, work_shape)
            idiv = _i_divergence(img_t, fwd[slices].clamp(min=bg))
            convergence_history.append(idiv)
            log.info("  iter %4d/%d  I-div=%.6g", k, niter, idiv)

            if convergence == "auto" and len(convergence_history) >= 2:
                prev_idiv = convergence_history[-2]
                if prev_idiv > 0:
                    rel_change = (prev_idiv - idiv) / prev_idiv
                    if rel_change < rel_threshold:
                        log.info("  converged at iter %d (rel_change=%.4g)", k, rel_change)
                        iterations_used = k
                        break

    # Extract the image-sized region
    result = x_cur[slices]
    return {
        "result": _to_numpy(result),
        "convergence": convergence_history,
        "iterations_used": iterations_used,
    }


# ===================================================================
#  PART 2 — PSF generation
# ===================================================================

# ---------------------------------------------------------------------------
# Simpson's rule (1-D, torch)
# ---------------------------------------------------------------------------

def _simpsons(fs: torch.Tensor, dx: float) -> torch.Tensor:
    """Simpson's rule along dim 0.  *fs* must have odd size along dim 0."""
    return (fs[0] + 4.0 * torch.sum(fs[1:-1:2], dim=0)
            + 2.0 * torch.sum(fs[2:-1:2], dim=0) + fs[-1]) * dx / 3.0

# ---------------------------------------------------------------------------
# Gibson-Lanni OPD
# ---------------------------------------------------------------------------

def _gibson_lanni_opd(
    sin_t: torch.Tensor,
    *,
    z_p: float,
    n_s: float,
    n_i: float,
    n_i0: float,
    n_g: float,
    n_g0: float,
    t_g: float,
    t_g0: float,
    t_i0: float,
) -> torch.Tensor:
    """Optical Path Difference from Gibson-Lanni model.

    Parameters in **nanometres** (consistent with psf_generator convention).
    Returns the OPD tensor of the same shape as *sin_t*.
    """
    ni2_sin2 = n_i ** 2 * sin_t ** 2
    t_i = n_i * (t_g0 / n_g0 + t_i0 / n_i0 - t_g / n_g - z_p / n_s)

    opd = (z_p   * torch.sqrt((n_s  ** 2 - ni2_sin2).clamp(min=0))
         + t_i   * torch.sqrt((n_i  ** 2 - ni2_sin2).clamp(min=0))
         - t_i0  * torch.sqrt((n_i0 ** 2 - ni2_sin2).clamp(min=0))
         + t_g   * torch.sqrt((n_g  ** 2 - ni2_sin2).clamp(min=0))
         - t_g0  * torch.sqrt((n_g0 ** 2 - ni2_sin2).clamp(min=0)))
    return opd

# ---------------------------------------------------------------------------
# Scalar PSF slice (single z-plane)
# ---------------------------------------------------------------------------

def _scalar_psf_slice(
    k: float,
    thetas: torch.Tensor,
    dtheta: float,
    rs: torch.Tensor,
    pupil: torch.Tensor,
    defocus_phase: torch.Tensor,
) -> torch.Tensor:
    """Compute scalar PSF for unique radii at one z-plane.

    Returns complex field values for each unique radius.
    """
    sin_t = torch.sin(thetas)
    bessel_arg = k * rs[None, :] * sin_t[:, None]
    J0 = bessel_j0(bessel_arg)

    integrand = J0 * (pupil * defocus_phase * sin_t)[:, None]
    field = _simpsons(integrand, dtheta)
    return field

# ---------------------------------------------------------------------------
# Vectorial PSF slice (single z-plane)
# ---------------------------------------------------------------------------

def _vectorial_psf_slice(
    k: float,
    thetas: torch.Tensor,
    dtheta: float,
    rs: torch.Tensor,
    pupil: torch.Tensor,
    defocus_phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute I0, I1, I2 integrals for one z-plane (unpolarised)."""
    sin_t = torch.sin(thetas)
    cos_t = torch.cos(thetas)

    bessel_arg = k * rs[None, :] * sin_t[:, None]
    J0 = bessel_j0(bessel_arg)
    J1 = bessel_j1(bessel_arg)
    # J2 via recurrence: J2(x) = 2*J1(x)/x - J0(x)
    J2 = 2.0 * torch.where(
        bessel_arg.abs() > 1e-6,
        J1 / bessel_arg,
        torch.tensor(0.5, dtype=bessel_arg.dtype, device=bessel_arg.device),
    ) - J0

    base = pupil * defocus_phase * sin_t

    integrand_0 = J0 * (base * (1.0 + cos_t))[:, None]
    integrand_1 = J1 * (base * sin_t)[:, None]
    integrand_2 = J2 * (base * (1.0 - cos_t))[:, None]

    I0 = _simpsons(integrand_0, dtheta)
    I1 = _simpsons(integrand_1, dtheta)
    I2 = _simpsons(integrand_2, dtheta)
    return I0, I1, I2

# ---------------------------------------------------------------------------
# Sub-pixel integration wrapper
# ---------------------------------------------------------------------------

def _pixel_integrate_psf(
    psf_func,
    pixel_size_xy: float,
    n_xy: int,
    n_subpixels: int,
    **kwargs,
) -> torch.Tensor:
    """Compute PSF by averaging over sub-pixel grid positions.

    *psf_func(fov, n_xy, **kwargs)* returns a 3-D PSF for a given field-of-
    view and lateral pixel count. We shrink the effective pixel, evaluate on a
    finer grid, and block-average.
    """
    if n_subpixels <= 1:
        fov = pixel_size_xy * n_xy
        return psf_func(fov=fov, n_xy=n_xy, **kwargs)

    n_fine = n_xy * n_subpixels
    fov = pixel_size_xy * n_xy  # total field of view unchanged
    fine_psf = psf_func(fov=fov, n_xy=n_fine, **kwargs)  # (Z, fineY, fineX)

    # Block-average back to n_xy × n_xy
    nz = fine_psf.shape[0]
    fine_psf = fine_psf.reshape(nz, n_xy, n_subpixels, n_xy, n_subpixels)
    return fine_psf.mean(dim=(2, 4))

# ---------------------------------------------------------------------------
# Core PSF builder
# ---------------------------------------------------------------------------

def _build_psf_stack(
    *,
    fov: float,
    n_xy: int,
    n_z: int,
    wavelength_nm: float,
    na: float,
    ri_immersion: float,
    ri_sample: float,
    ri_coverslip: float,
    ri_coverslip_design: float,
    ri_immersion_design: float,
    t_g: float,
    t_g0: float,
    t_i0: float,
    z_p: float,
    pixel_size_z_nm: float,
    n_pupil: int,
    use_vectorial: bool,
    gibson_lanni: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 3-D PSF stack.  Returns tensor of shape (n_z, n_xy, n_xy)."""
    k = 2.0 * math.pi / wavelength_nm
    ri = ri_sample if gibson_lanni else ri_immersion

    # Theta grid (pupil samples)
    s_max = na / ri_immersion_design
    if s_max > 1.0:
        s_max = 1.0
    theta_max = math.asin(s_max)
    thetas = torch.linspace(0, theta_max, n_pupil, device=device, dtype=dtype)
    dtheta = theta_max / (n_pupil - 1)

    # PSF spatial coordinates — unique radii
    x = torch.linspace(-fov / 2.0, fov / 2.0, n_xy, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    r_unique, rr_inv = torch.unique(rr, return_inverse=True)
    rs = r_unique  # (n_unique,)

    # Bessel argument scaling is handled inside slice functions via k*ri

    # Correction / pupil
    sin_t = torch.sin(thetas)
    cos_t = torch.cos(thetas)
    pupil = torch.sqrt(cos_t).to(torch.complex128 if dtype == torch.float64 else torch.complex64)

    if gibson_lanni:
        clamp_val = min(ri_sample / ri_immersion, ri_coverslip / ri_immersion)
        sin_t_gl = sin_t.clamp(max=clamp_val)
        opd = _gibson_lanni_opd(
            sin_t_gl,
            z_p=z_p, n_s=ri_sample, n_i=ri_immersion, n_i0=ri_immersion_design,
            n_g=ri_coverslip, n_g0=ri_coverslip_design,
            t_g=t_g, t_g0=t_g0, t_i0=t_i0,
        )
        pupil = pupil * torch.exp(1j * k * opd.to(pupil.dtype))

    # Scale rs for Bessel arg — handled inside slice functions

    # Z planes
    z_min = -pixel_size_z_nm * (n_z // 2)
    zs = torch.linspace(z_min, -z_min, n_z, device=device, dtype=dtype)

    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    slices_out = []
    for zi in range(n_z):
        z = zs[zi]
        defocus = torch.exp(1j * k * z * cos_t * ri).to(cdtype)

        if use_vectorial:
            I0, I1, I2 = _vectorial_psf_slice(

                k * ri, thetas, dtheta, rs, pupil.to(cdtype), defocus,
            )
            # Intensity = |I0|^2 + 2|I1|^2 + |I2|^2  (unpolarised)
            intensity = (I0.abs() ** 2 + 2.0 * I1.abs() ** 2 + I2.abs() ** 2)
        else:
            field = _scalar_psf_slice(
                k * ri, thetas, dtheta, rs, pupil.to(cdtype), defocus,
            )
            intensity = field.abs() ** 2

        # Scatter from unique radii to 2-D grid
        plane = intensity[rr_inv.flatten()].reshape(n_xy, n_xy)
        slices_out.append(plane)

    psf_stack = torch.stack(slices_out, dim=0).to(dtype)  # (Z, Y, X)    
    return psf_stack

# ---------------------------------------------------------------------------
# Top-level PSF function
# ---------------------------------------------------------------------------

def ci_generate_psf(
    na: float,
    wavelength_nm: float,
    pixel_size_xy_nm: float,
    pixel_size_z_nm: float,
    n_xy: int,
    n_z: int,
    *,
    ri_immersion: float = 1.515,
    ri_sample: float = 1.33,
    ri_coverslip: float = 1.5,
    ri_coverslip_design: float = 1.5,
    ri_immersion_design: float = 1.515,
    t_g: float = 170e3,
    t_g0: float = 170e3,
    t_i0: float = 100e3,
    z_p: float = 0.0,
    microscope_type: str = "widefield",
    excitation_nm: Optional[float] = None,
    integrate_pixels: bool = True,
    n_subpixels: int = 3,
    n_pupil: int = 129,
    device: Optional[str] = None,
) -> np.ndarray:
    """Generate a physically accurate 3-D PSF.

    Parameters
    ----------
    na : float
        Numerical aperture.
    wavelength_nm : float
        Emission wavelength in nm.
    pixel_size_xy_nm, pixel_size_z_nm : float
        Pixel sizes in nm.
    n_xy, n_z : int
        Lateral / axial pixel counts (should be odd).
    ri_immersion, ri_sample, ri_coverslip : float
        Refractive indices for immersion medium, sample, and coverslip.
    ri_coverslip_design, ri_immersion_design : float
        Design (nominal) RI values for coverslip and immersion.
    t_g, t_g0 : float
        Actual and design coverslip thickness in nm.
    t_i0 : float
        Design immersion thickness in nm.
    z_p : float
        Depth of the particle below coverslip in nm.
    microscope_type : str
        ``"widefield"`` or ``"confocal"``.
    excitation_nm : float or None
        Excitation wavelength for confocal.
    integrate_pixels : bool
        Integrate over pixel area (more accurate, slower).
    n_subpixels : int
        Sub-pixel grid size per axis for pixel integration.
    n_pupil : int
        Number of pupil integration samples (should be odd).
    device : str or None
        PyTorch device.

    Returns
    -------
    ndarray
        Normalised PSF (sum = 1), shape ``(n_z, n_xy, n_xy)``.
    """
    dev = _pick_device(device)
    dtype = _pick_dtype(dev)

    use_vectorial = na >= 0.9
    gibson_lanni = (abs(ri_sample - ri_immersion) > 0.001
                    or abs(ri_coverslip - ri_coverslip_design) > 0.001
                    or z_p > 0)

    log.info("ci_generate_psf  NA=%.2f  λ=%gnm  pixel_xy=%gnm  pixel_z=%gnm  "
             "size=%dx%dx%d  vectorial=%s  GL=%s  device=%s",
             na, wavelength_nm, pixel_size_xy_nm, pixel_size_z_nm,
             n_xy, n_xy, n_z, use_vectorial, gibson_lanni, dev)

    common = dict(
        n_z=n_z,
        wavelength_nm=wavelength_nm,
        na=na,
        ri_immersion=ri_immersion,
        ri_sample=ri_sample,
        ri_coverslip=ri_coverslip,
        ri_coverslip_design=ri_coverslip_design,
        ri_immersion_design=ri_immersion_design,
        t_g=t_g, t_g0=t_g0, t_i0=t_i0, z_p=z_p,
        pixel_size_z_nm=pixel_size_z_nm,
        n_pupil=n_pupil,
        use_vectorial=use_vectorial,
        gibson_lanni=gibson_lanni,
        device=dev,
        dtype=dtype,
    )

    def _psf_func(*, fov: float, n_xy: int, **kw) -> torch.Tensor:
        return _build_psf_stack(fov=fov, n_xy=n_xy, **{**common, **kw})

    if integrate_pixels and n_subpixels > 1:
        psf = _pixel_integrate_psf(
            _psf_func,
            pixel_size_xy=pixel_size_xy_nm,
            n_xy=n_xy,
            n_subpixels=n_subpixels,
        )
    else:
        fov = pixel_size_xy_nm * n_xy
        psf = _psf_func(fov=fov, n_xy=n_xy)

    # Confocal: multiply emission × excitation PSFs
    if microscope_type == "confocal":
        if excitation_nm is not None and excitation_nm != wavelength_nm:
            common_ex = {**common, "wavelength_nm": excitation_nm}

            def _psf_ex(*, fov, n_xy, **kw):
                return _build_psf_stack(fov=fov, n_xy=n_xy, **{**common_ex, **kw})

            if integrate_pixels and n_subpixels > 1:
                psf_ex = _pixel_integrate_psf(
                    _psf_ex,
                    pixel_size_xy=pixel_size_xy_nm,
                    n_xy=n_xy,
                    n_subpixels=n_subpixels,
                )
            else:
                fov = pixel_size_xy_nm * n_xy
                psf_ex = _psf_ex(fov=fov, n_xy=n_xy)
            psf = psf * psf_ex
        else:
            psf = psf ** 2

    # Normalise
    psf = psf / psf.sum()

    result = _to_numpy(psf)
    log.info("  PSF range [%.3g, %.3g], sum=%.6f", result.min(), result.max(),
             result.sum())
    return result
