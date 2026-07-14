# FFT optimization benchmark: results and recommendations

Last updated: 2026-07-14

This document preserves the benchmark findings independently of the generated
`results/` directories, which are intentionally excluded from version control.
The experiments were originally isolated under `benchmarks/fft_optimization`.
The accepted implementation was promoted into production on 2026-07-14; the
benchmark now imports the maintained CUDA sources from `core/optimized_cuda`.

## Executive conclusion

The recommended implementation candidate is:

- direct FP32 cuFFT R2C/C2R execution;
- an in-place padded FFT buffer;
- one caller-owned workspace shared by forward and inverse plans;
- overwriting the dead previous RL estimate;
- smooth FFT work dimensions;
- cached OTF, Bertero weights, cuFFT plans, and allocations per tile geometry;
- the existing production-equivalent nine-tile geometry for this DNA volume.

On both channels of `DNA.ome.tiff`, this configuration was approximately 2x
faster than production, reduced peak allocated GPU memory by approximately 25%,
and passed every strict numerical gate.

Static FP16 storage for the OTF and boundary weights also passed the configured
gates, but its additional speedup was too small and its numerical error was
measurably larger. FP32 static storage therefore remains the default
recommendation.

## Production integration

The production core now provides an optional optimized CUDA backend for
`ci_rl`, `ci_rl_tv`, and `ci_sparse_hessian`. It uses direct FP32 in-place cuFFT,
a caller-owned shared workspace, overwrite-state SHB updates, smooth work
dimensions, cached plans/OTFs/Bertero weights/arenas, fused TV, and the explicit
sparse-Hessian gradient kernel. Static FP16, CUDA Graphs, four-tile substitution,
axial partitioning, and cuFFTDx remain disabled because the benchmarks did not
recommend them as defaults.

The GUI exposes Auto, Optimized CUDA, PyTorch CUDA, and CPU. Auto attempts a
compatible prebuilt module, may build a cache-isolated local module when a
matching toolkit/compiler is available, and otherwise retains the PyTorch
fallback. Output processing metadata records the requested/used backend, work
shape, and direct-cuFFT workspace size where available.

## Test environment

- GPU: NVIDIA RTX A5000, 24 GB, compute capability 8.6
- Driver observed during the run: 595.79
- PyTorch: 2.11.0+cu126
- PyTorch CUDA runtime: 12.6
- CUDA toolkit: 12.6 GA, nvcc 12.6.20
- Host: Windows, Visual Studio 2022 C++ toolchain
- Full data: `localdata/DNA.ome.tiff`, two channels, each `74 x 2048 x 2048`
- Quick data: `localdata/DNAcrop.ome.tiff`, two channels, each `57 x 366 x 366`
- Full benchmark iterations: 20
- Quick benchmark iterations: 5

The strict acceptance criteria were:

- global SSIM at least 0.9999;
- normalized RMSE at most 0.001;
- relative flux difference at most 0.001 (0.1%);
- all values finite and nonnegative.

## Final full-volume benchmark

Raw report: `results/20260713_195125/REPORT.md`

### Channel 0

| Variant | Wall time (s) | GPU time (s) | Peak allocated (MB) | Workspace (MB) | SSIM | NRMSE | Flux difference | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Production auto, 9 tiles | 21.186 | 21.186 | 6,938 | - | 1.000000 | 0 | 0 | Pass |
| Direct 9, uncached FP32 | 12.333 | 6.094 | 5,206 | 420 | 1.000000 | 4.19e-7 | 3.32e-8 | Pass |
| Direct 9, cached FP32 | 10.572 | 6.106 | 5,206 | 420 | 1.000000 | 4.19e-7 | 3.32e-8 | Pass |
| Direct 9, cached static FP16 | 10.271 | 5.968 | 5,206 | 420 | 0.999965 | 3.91e-4 | 2.69e-4 | Pass |
| Direct 4, margin 16 | 9.964 | 6.459 | 9,402 | 745 | 0.999872 | 7.33e-4 | 3.24e-3 | Fail |
| Direct 4, margin 32 | 9.903 | 6.456 | 9,539 | 745 | 0.999888 | 6.84e-4 | 3.52e-3 | Fail |
| Direct 4, margin 64 | 10.705 | 6.939 | 10,213 | 809 | 0.999890 | 6.79e-4 | 3.37e-3 | Fail |
| Direct untiled FP32 | 43.648 | 13.900 | 32,949* | 2,619 | 0.999726 | 1.04e-3 | 8.44e-3 | Fail |
| Direct axial 2-part prototype | 20.452 | 8.511 | 24,952* | 1,995 | 0.996883 | 3.59e-3 | 1.39e-2 | Fail |

### Channel 1

| Variant | Wall time (s) | GPU time (s) | Peak allocated (MB) | Workspace (MB) | SSIM | NRMSE | Flux difference | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Production auto, 9 tiles | 20.859 | 20.859 | 6,938 | - | 1.000000 | 0 | 0 | Pass |
| Direct 9, uncached FP32 | 11.813 | 6.184 | 5,206 | 420 | 1.000000 | 2.93e-7 | 1.07e-7 | Pass |
| Direct 9, cached FP32 | 10.445 | 6.182 | 5,206 | 420 | 1.000000 | 2.93e-7 | 1.07e-7 | Pass |
| Direct 9, cached static FP16 | 10.361 | 6.029 | 5,206 | 420 | 0.999937 | 2.57e-4 | 5.54e-4 | Pass |
| Direct 4, margin 16 | 9.747 | 6.454 | 9,404 | 745 | 0.999760 | 5.02e-4 | 1.07e-3 | Fail |
| Direct 4, margin 32 | 9.594 | 6.465 | 9,539 | 745 | 0.999762 | 5.00e-4 | 1.13e-3 | Fail |
| Direct 4, margin 64 | 10.457 | 6.948 | 10,213 | 809 | 0.999765 | 4.97e-4 | 1.04e-3 | Fail |
| Direct untiled FP32 | 26.190 | 13.273 | 32,950* | 2,619 | 0.999508 | 7.29e-4 | 4.33e-3 | Fail |
| Direct axial 2-part prototype | 20.980 | 9.092 | 24,952* | 1,995 | 0.998664 | 1.18e-3 | 3.26e-4 | Fail |

`*` Values above physical VRAM indicate Windows/CUDA oversubscription and paging,
not usable resident GPU capacity. Live `nvidia-smi` usage reached approximately
24.1 GB. These configurations are not safe defaults despite completing.

### Quantified improvements of the recommendation

Compared with production auto-tiling, cached direct FP32 cuFFT delivered:

- channel 0: 2.004x speedup, 50.10% less wall time;
- channel 1: 1.997x speedup, 49.93% less wall time;
- both channels: approximately 24.96% less peak allocated GPU memory;
- effectively identical numerical output under all configured gates.

Caching equal tile geometries improved wall time by another 14.28% on channel 0
and 11.59% on channel 1 relative to the uncached direct-cuFFT implementation.

## Earlier benchmark stages

These reports remain useful for separating individual changes:

- `results/20260713_192239`: original crop comparison, including fused kernels
  and CUDA Graph capture.
- `results/20260713_192400`: four-tile full-volume experiment; faster, but failed
  strict quality gates.
- `results/20260713_192734`: original nine-tile fused-smooth experiment; passed
  strict quality gates and was approximately 23-24% faster than production.
- `results/20260713_194953`: extended crop comparison with overwrite-state,
  direct cuFFT FP32, and static FP16 storage.
- `results/20260713_195125`: final full-volume comparison of all available
  recommendations.

The crop direct-cuFFT FP32 path reduced iterative GPU time from approximately
54 ms to 29 ms and passed the strict gates. CUDA Graph capture increased total
crop time and was not selected.

## Assessment of the eight recommendations

### 1. Overwrite the old RL estimate

Status: **working and recommended**.

The previous estimate is dead after SHB momentum is calculated. It can hold the
momentum and later the updated estimate. This removes one live work-sized tensor
without changing the algorithm. The isolated peak can still be dominated by OTF
and Bertero-weight setup, so the saving is clearest when combined with the direct
backend.

### 2. Direct in-place cuFFT

Status: **working and strongly recommended**.

The direct backend uses the same padded allocation for the real spatial domain
and Hermitian frequency domain. It approximately halved iterative GPU time on the
crop and is central to the final 2x full-volume result.

### 3. Explicit shared cuFFT workspace

Status: **working and recommended**.

Forward and inverse plans execute sequentially and share one caller-owned
workspace. This exposes actual workspace requirements to the tiling preflight and
prevents separate opaque work areas. Observed workspace sizes were approximately
420 MB for nine tiles, 745-809 MB for four tiles, and 2.62 GB untiled.

### 4. Cache plans, OTF, weights, and allocations

Status: **working and recommended**.

Grouping equal tile geometries and reusing their static preparation reduced wall
time by 11.6-14.3%. CUDA Graph capture was also tested on the crop, but its setup
cost exceeded its launch-overhead saving for this workload, so Graph capture is
not recommended at present.

### 5. FP16 storage for static tensors only

Status: **working, optional, not the default**.

FFT computation, observations, ratios, and RL state remained FP32. Only OTF and
boundary weights were stored as FP16 and converted inside fused kernels. It
passed the current gates, but improved wall time by only 0.8-2.8% over cached
FP32 while increasing numerical error. It may be useful on a tighter-memory GPU,
but should require an explicit opt-in and retain the quality checks.

### 6. cuFFTDx fusion

Status: **not benchmarked; optional future research**.

`cuFFTDx` was unavailable because MathDx headers were not installed. MathDx is a
separate NVIDIA SDK download and is governed by the NVIDIA Math Libraries SDK
license. The current winning backend does not use it. cuFFTDx should not become a
mandatory user dependency unless a later benchmark demonstrates a material gain
over direct cuFFT.

Relevant NVIDIA documentation:

- <https://developer.nvidia.com/cufftdx-downloads>
- <https://docs.nvidia.com/cuda/mathdx/26.03.0/installation.html>
- <https://docs.nvidia.com/cuda/mathdx/26.03.0/license.html>

For CUDA 12, the NVIDIA archive currently lists MathDx 25.12.1 CUDA 12 packages;
the current main download is CUDA 13. A matching package/toolkit combination is
required.

### 7. Larger tiles with larger halos

Status: **numerically viable with tile-invariant preprocessing, but not the
A5000 default**.

The original margins of 16, 32, and 64 pixels were tested while background and
offset were still re-estimated independently in every tile. Those four-tile runs
did not satisfy flux/SSIM gates. The later tile-invariance experiment froze the
nonlinear preprocessing globally; under that condition, four versus nine tiles
passed all gates. Four tiles nevertheless saved only 0.44 s and raised peak
allocation from 5.38 to 9.73 GB, so nine tiles retain the better safety/performance
balance on the A5000. See “Tile-count invariance follow-up” below.

### 8. Axial partitioning

Status: **prototype failed; not recommended**.

The explicitly approximate two-part axial RL prototype failed the quality gates.
It also retained a very large full-XY cuFFT workspace and triggered memory
oversubscription. A mathematically exact partitioned convolution would require a
substantially different overlap-add implementation inside every RL iteration; the
current result does not justify that engineering effort.

## Regularized methods: TV and sparse Hessian

This follow-up was run on 14 July 2026 on both channels of
`DNAcrop.ome.tiff`, using 20 fixed iterations on the RTX A5000. Each optimized
result used FP32 state, OTF and weights, the smooth work shape
`120 x 432 x 432`, and was compared with its own production-method reference at
the exact work shape `113 x 430 x 430`. Static FP16 and additional tile-boundary
experiments were intentionally not repeated.

| Method | Channel | Production wall (s) | Optimized wall (s) | Wall reduction | Optimized peak allocated (MB) | Production peak (MB) | Regularizer share of iteration GPU time | NRMSE | Quality gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `ci_rl_tv` | 0 | 0.846 | 0.486 | 42.6% | 1571 | 1948 | 69.1% | 1.13e-6 | pass |
| `ci_rl_tv` | 1 | 0.663 | 0.474 | 28.5% | 1571 | 1948 | 69.4% | 1.06e-6 | pass |
| `ci_sparse_hessian` | 0 | 0.863 | 0.635 | 26.4% | 1045 | 1038 | 78.2% | 8.21e-7 | pass |
| `ci_sparse_hessian` | 1 | 0.861 | 0.634 | 26.4% | 1045 | 1038 | 77.4% | 2.52e-7 | pass |

All four optimized outputs passed the existing strict gates: finite and
nonnegative output, global SSIM at least 0.9999, NRMSE at most 0.001, and relative
flux difference at most 0.001. Actual global SSIM values were greater than
0.9999999990 and relative flux differences were below 9.2e-7. QC MIPs and absolute
difference MIPs were also generated in the timestamped result directory.

### Decision on custom regularizer kernels

**Fused TV kernel: justified and recommended as the next prototype.** Once the
FFT path is optimized, the existing PyTorch TV stencil consumes about 69% of
iteration GPU time. The direct FFT path already reduces peak allocated memory by
about 19.3%, but TV still constructs several full-work-domain gradient,
magnitude, normalized-gradient and divergence tensors. A fused stencil with a
small fixed scratch allocation has credible speed and memory upside.

**Custom sparse-Hessian gradient kernel: justified as a separate, higher-risk
prototype.** The autograd-based prior consumes 77-78% of optimized iteration GPU
time. The FFT optimization reduces wall time, but peak allocated memory is about
0.7% higher and peak reserved memory rises from 1328 MB to 1548 MB because the
autograd graph and direct-cuFFT arena coexist. Replacing autograd and the
full-domain clone is therefore the main remaining opportunity. It requires more
careful numerical validation than TV because the exact finite-difference
gradient, mixed derivatives, edge treatment, normalization and anisotropic
Z scaling must be preserved.

The initial result justified implementing both as benchmark-only prototypes. The
following section records their measured outcome; neither has been added to the
production solver.

### Fused regularizer prototype results

The benchmark-only CUDA prototypes were run on 14 July 2026 with the same RTX
A5000, two DNA crop channels, FP32 arithmetic, smooth work shape, and 20 fixed
iterations. The comparison includes production, direct cuFFT with the PyTorch
regularizer, and direct cuFFT with the fused regularizer.

| Method | Channel | PyTorch-regularizer wall (s) | Fused wall (s) | Wall reduction vs optimized PyTorch | PyTorch regularizer GPU time (s) | Fused regularizer GPU time (s) | Regularizer reduction | Fused peak allocated (MB) | NRMSE vs production | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `ci_rl_tv` | 0 | 0.574 | 0.229 | 60.1% | 0.288 | 0.023 | 92.0% | 1045 | 1.13e-6 | pass |
| `ci_rl_tv` | 1 | 0.474 | 0.219 | 53.8% | 0.281 | 0.023 | 91.8% | 1045 | 1.06e-6 | pass |
| `ci_sparse_hessian` | 0 | 0.651 | 0.228 | 65.0% | 0.454 | 0.033 | 92.7% | 1045 | 8.26e-7 | pass |
| `ci_sparse_hessian` | 1 | 0.640 | 0.220 | 65.6% | 0.447 | 0.025 | 94.4% | 1045 | 2.50e-7 | pass |

The fused TV implementation writes its correction factors into the dead cuFFT
arena and applies them in a second kernel. It creates no additional full-volume
TV tensors. Peak allocated memory falls from 1571 MB for direct cuFFT plus the
PyTorch TV implementation to 1045 MB, a 33.5% reduction; relative to production
TV, the reduction is 46.4%.

The sparse-Hessian implementation analytically evaluates the exact gradient of
the existing 3D penalty, stores one image-support-sized gradient, normalizes it
with an L1 reduction, and applies the update in a fused kernel. It eliminates the
autograd graph and intermediates. Peak allocated memory remains 1045 MB—the
direct FFT arena is now the dominant allocation—and is only 0.7% above the
production sparse-Hessian peak of 1038 MB.

Before every benchmark, deterministic small-tensor checks compare the CUDA
kernels against the production PyTorch formulas. Final validation measured:

- TV updated-image maximum absolute difference: exactly `0`;
- sparse-Hessian normalized-gradient maximum absolute difference: `1.43e-6`;
- sparse-Hessian normalized-gradient RMSE: `1.64e-7`;
- all four full deconvolution outputs passed SSIM, NRMSE, flux, finiteness and
  nonnegativity gates.

**Updated recommendation:** both kernels are technically worthwhile. Fused TV is
the lower-risk production candidate. Sparse Hessian now demonstrates comparable
speed benefit, but still needs broader tests for anisotropic Z scaling, odd and
small shapes, 2D/singleton-Z handling, multiple microscopy datasets, and edge
conditions before production integration.

### Sparse-Hessian local dataset validation

The sparse-Hessian prototype was subsequently extended with an exact 2D stencil
for singleton-Z inputs and tested on five local OME-TIFF files for 20 fixed
iterations. Production-exact FFT dimensions were used so the test isolates the
regularizer rather than the smooth-padding optimization. Physical XY and Z
sampling was passed through for every 3D file.

| Dataset | Case | Shape | Z scale | Production wall (s) | Optimized PyTorch wall (s) | Fused wall (s) | Fused reduction vs optimized PyTorch | PyTorch regularizer GPU time (s) | Fused regularizer GPU time (s) | Fused peak MB | NRMSE vs production | Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `WF-2D-3Ch-Actin.ome.tiff` | 2D/singleton-Z | 1×443×374 | 1.000 | 0.357 | 0.127 | 0.060 | 52.8% | 0.091 | 0.025 | 31 | 3.16e-7 | pass |
| `DNARepairSpots_decon.ome.tiff` | 2D/singleton-Z | 1×800×800 | 1.000 | 0.141 | 0.111 | 0.039 | 64.9% | 0.082 | 0.009 | 118 | 4.21e-9 | pass |
| `Vesicles.ome.tiff` | anisotropic 3D | 11×302×367 | 0.536 | 0.282 | 0.225 | 0.098 | 56.4% | 0.132 | 0.007 | 151 | 9.52e-8 | pass |
| `DividingCellcrop.ome.tiff` | anisotropic odd 3D | 25×271×303 | 0.334 | 0.300 | 0.274 | 0.134 | 51.1% | 0.135 | 0.010 | 292 | 5.90e-8 | pass |
| `U2OS.ome.tiff` | anisotropic odd 3D | 30×369×301 | 0.323 | 0.432 | 0.430 | 0.225 | 47.7% | 0.212 | 0.015 | 457 | 3.34e-7 | pass |

All fused outputs also passed direct comparison with the optimized PyTorch
regularizer; the worst fused-vs-PyTorch NRMSE was `3.21e-7`. The deterministic
singleton-Z gradient check measured maximum normalized-gradient error `4.77e-7`
and RMSE `1.21e-7`. This exercises 2D edges, odd XY sizes, shallow Z, and 3D
anisotropy down to Z scale 0.323.

The first exploratory dataset run exposed a benchmark-only initialization
mismatch: the direct path forced the widefield auto-start policy while production
sparse Hessian uses the generic auto-start policy. Both fused and PyTorch-direct
variants showed the same deviation, proving it was outside the custom kernel.
After matching production initialization, all five production-relative gates
passed. Production code was not changed.

**Revised sparse-Hessian recommendation:** the tested 2D/singleton-Z and
anisotropic 3D cases no longer block production consideration. Remaining work is
broader regression coverage (especially very small dimensions below the Hessian
stencil width, more channels/timepoints, and additional PSFs), packaging, and a
maintained fallback—not a demonstrated numerical problem with the fused kernel.

## Production readiness audit (pre-integration record)

The findings below record the state before the production integration. See the
2026-07-14 resolution update after the rollout section for the current status.

### Meaning of the remaining concerns

“No demonstrated numerical problem” means that, for the tested FP32 inputs and
parameters, the custom kernels reproduce the production formulas and complete
deconvolution outputs within the strict gates. It does **not** by itself prove
that the benchmark extension can safely replace every installed production path.
The remaining concerns are integration and distribution properties that the
numerical benchmark does not exercise: GPU architecture coverage, PyTorch/CUDA
binary compatibility, Linux builds, multiple CUDA devices and streams, fallback
behavior, convergence callbacks, production tiling, and package-specific release
mechanisms.

### Concrete audit findings

1. **The current binary is A5000-only.** `cuobjdump` reports one
   `sm_86` cubin and no PTX. It cannot run on the H100 (`sm_90`). Production
   builds for the known targets must contain both `sm_86` and `sm_90` code (or a
   deliberately selected PTX fallback). PyTorch documents
   `TORCH_CUDA_ARCH_LIST` and the performance/forward-compatibility tradeoff of
   `+PTX` in its
   [C++ extension documentation](https://docs.pytorch.org/docs/stable/cpp_extension.html).
2. **The benchmark uses runtime JIT compilation.** End users would need NVCC,
   CUDA headers, a host C++ compiler, Ninja and a matching toolkit. The current
   Docker image is based on `python:3.11-slim` and intentionally contains none of
   those. Production must use ahead-of-time compiled binaries; PyTorch describes
   `CUDAExtension`/`BuildExtension` as its standard AOT path in the
   [custom C++/CUDA operator guide](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).
3. **The extension is coupled to its build environment.** It currently uses
   `torch::Tensor` and pybind11, not the newer stable LibTorch ABI. The simplest
   safe initial release therefore pins/builds against a specific PyTorch/CUDA and
   Python combination. A stable-ABI migration is possible but is a separate
   engineering task.
4. **Device and concurrency hardening is missing.** The CUDA entry points do not
   yet guard the tensor device or verify that all tensors share a device. The
   cuFFT plan owns mutable stream/work-area state, so one cached plan must not be
   used concurrently on multiple streams without per-stream plans or locking.
5. **The benchmark solver is not a complete production solver.** It exercises
   fixed iteration counts and direct untiled volumes. Production integration must
   retain automatic convergence, callbacks/previews, established tile counts and
   blending, memory preflight, cancellation/error handling, provenance, and a
   restart-on-PyTorch fallback if the optimized setup fails.
6. **The packages have different method contracts.** The root GUI, standalone
   and Docker/Bilayers paths expose `ci_rl`, `ci_rl_tv` and
   `ci_sparse_hessian`. The focused PyPI CLI intentionally exposes only `ci_rl`
   and contains a reduced copy of `core/deconvolve_ci.py`. Adding TV or sparse
   Hessian there would be a product/API expansion, not merely an optimization.
7. **The PyPI build configuration is not ready for this binary.** The CLI declares
   Python 3.12 or newer, but `publish.cmd` selects the repository's Python 3.11
   `deconvolve` environment. A pybind11 extension built there would have the wrong
   CPython tag for the declared package. The package also deliberately leaves
   PyTorch out of dependencies, so the supported Torch/CUDA wheel matrix and
   source-build behavior must be explicit.
8. **The release baseline has one unrelated failing test.** The targeted audit
   run completed 23 tests successfully, but
   `test_load_image_tracks_metadata_provenance` fails because
   `_metadata_provenance` is absent. This predates and is independent of the
   benchmark kernels, but a production release should start from a green or
   explicitly baselined suite.
9. **The benchmark folder is currently untracked by Git.** The validated source,
   tests and durable results must be reviewed and added deliberately before they
   can serve as the implementation record for production work.

### Core-unification update (2026-07-14)

The integration groundwork identified in findings 6–8 is now complete:

- the focused PyPI CLI packages the repository's main `core` source instead of a
  reduced duplicate, while its public interface remains `ci_rl` only;
- CLI-only metadata provenance, time-series handling, and cancellation behavior
  were merged into the shared core without removing HCS or TV/sparse-Hessian
  support;
- the package now declares Python 3.11+ consistently with the canonical and
  Docker runtimes;
- the complete repository suite passes, and a wheel built from the CLI project
  passes `twine check`, installs in isolation, imports the shared core, and has a
  byte-identical packaged `deconvolve_ci.py`.

This removes source divergence as a blocker, but does not resolve findings 1–5:
the fused CUDA backend still needs multi-architecture AOT packaging, device and
stream hardening, production tiling/callback integration, memory preflight, and a
tested pure-PyTorch fallback before it can become the default backend.

### Per-package decision

| Target | Integrate an optional backend now? | Make it the released default now? | Required next work |
|---|:---:|:---:|---|
| Root source/GUI development build | Yes | No | Move and harden backend, feature flag, fallback, full solver integration |
| Windows PyInstaller standalone on RTX A5000 | Yes, first candidate | Not before packaged regression | Bundle the matching `.pyd`, add it to the spec, test clean machine |
| Docker/Bilayers and H100 Slurm | Yes, after Linux build | No | AOT Linux build, exact `sm_90`, pinned Torch/CUDA, run on H100 |
| Focused PyPI `ci-deconvolve` CLI | Defer binary rollout | No | Resolve Python/build matrix; decide whether CLI remains `ci_rl`-only |
| CPU or unsupported GPU installations | Fallback only | PyTorch remains default | Import/capability checks and transparent pure-PyTorch fallback |

### Recommended rollout

The production code can now be updated as an **optional experimental backend**,
but the benchmark sources should not yet be copied wholesale or enabled as the
unconditional default. The smallest safe rollout is:

1. create a maintained internal CUDA-backend module and AOT build definition;
2. add device guards, same-device validation and non-shared/per-stream cuFFT plans;
3. build Windows `sm_86` and Linux `sm_90` artifacts against pinned
   PyTorch/CUDA versions;
4. integrate behind `auto`/`torch`/`cuda_direct` selection, initially defaulting
   to `torch`, with a logged restart-on-fallback path;
5. preserve the current production tiling, convergence and callback interfaces;
6. run method/parameter/tile regression tests on A5000 and H100;
7. promote `auto` to select the optimized backend only after packaged Windows,
   Docker and Slurm checks pass;
8. address or explicitly baseline the unrelated metadata-provenance test before
   publishing a release.

This is a **go for staged production integration**, but a **no-go for immediately
publishing all packages with the optimized backend enabled by default**.

### Resolution update (2026-07-14)

The staged integration described above has now been implemented:

- the CUDA sources live in `core/optimized_cuda` and are shared by production
  and the benchmark;
- device guards, same-device validation, per-device/thread/stream plan caches,
  production tiling, convergence callbacks, and the PyTorch fallback are in
  place;
- `Auto` prefers the compiled backend and falls back to PyTorch; the GUI also
  exposes forced Optimized CUDA, PyTorch CUDA, and CPU selections;
- the Docker multi-stage build emits native SM 86, 89, 90, 100, and 120 code and
  copies only the prebuilt extension into the slim runtime;
- on the RTX A5000, the cu132 container loaded `/app/core/_optimized_cuda.so`
  without a runtime toolkit/compiler and all three forced-optimized smoke tests
  produced finite, nonnegative results;
- the complete repository suite passes (121 tests), and the focused PyPI wheel
  includes the maintained loader and CUDA sources while retaining its ci_rl-only
  public CLI contract.

The remaining release qualification is hardware/distribution coverage rather
than a known solver defect: run the same container through BIOMERO/Apptainer on
the H100/H200, test an Ada and Blackwell device when available, and validate any
future Windows standalone precompiled binary on a clean machine. Until the H100
test passes, the `-cu130` cluster fallback remains required.

### Large tiled production follow-up (2026-07-14)

The first GUI run of `DNA.ome.tiff` exposed a production cache-policy mismatch:
four distinct edge/centre tile arenas were retained simultaneously. The accepted
benchmark instead groups equal tile geometries and retains only the active arena.
Production now follows that policy and releases the initialization padding before
the iteration loop.

On one `74 x 2048 x 2048` DNA channel with 20 iterations and the GUI's automatic
background/start/convergence behavior, forced Optimized CUDA completed in 29.4 s
versus 32.4 s for forced PyTorch CUDA. The optimized run used 5.38 GB peak
PyTorch allocation; the original GUI log reported 11.1 GB. The net end-to-end
speedup is smaller than the fixed-parameter kernel benchmark because GUI
preprocessing, convergence checks, host transfers, and tile blending are shared
overheads. The memory reduction now matches the benchmark expectation and is the
larger practical gain for this workload.

A later GUI timing audit separated cold-start overhead from iteration time. In a
fresh Windows Python process, loading the cached optimized extension took 6.95 s.
A zero-iteration nine-tile pass, which includes per-tile preprocessing, OTF and
weight preparation, plan/allocation setup, blending, and output transfer, took
11.93 s for Optimized CUDA and 12.85 s for PyTorch CUDA. The GUI's per-channel
and resource-monitor totals already included these costs. The GUI now reports
the optimized-backend preparation phase separately so the one-time cold load is
visible; subsequent channels and runs in the same process reuse the module.

Phase profiling of a zero-iteration nine-tile channel then identified automatic
offset estimation as the dominant setup cost: 6.93 s of a 12.03 s pass. Each
tile was converted in full to float64 for two percentiles. Production now uses a
deterministic sample of at most one million values before float64 conversion,
matching the existing statistics/SNR sampling policy. On the full DNA channel,
20 iterations fell from 26.56 s to 12.62 s in a warm process. The comparison
against the full-percentile output gave SSIM effectively 1.0, NRMSE `4.11e-8`,
relative flux difference `3.60e-8`, and maximum absolute difference 0.16.

Tile blending was also storing the identical XY feather denominator for every Z
plane. It is now a single 2D array broadcast during final division, removing
about 1.2 GB of host allocation for a `74 x 2048 x 2048` output. The measured
remaining setup components were comparatively small: optimized OTF/Bertero/
plan/arena preparation 1.15 s, blending 0.47 s, tile result transfers 0.25 s,
background estimation 0.10 s, and cache release 0.31 s. More aggressive GPU
blending or transfer overlap is therefore deferred because it would increase
VRAM/complexity for a sub-second opportunity on this workload.

The optimized backend accelerates every tile; tiling itself does not mean that
the PyTorch backend was selected. Logs now report `backend=optimized_cuda` per
tile, and the GUI completion line reports the backend and tile count.
Per-iteration full-image previews remain disabled for internal tiled execution
because there is no complete volume until all independently converged tiles have
been blended.

### Tile-count invariance follow-up (2026-07-14)

The earlier four-tile failures were not caused solely by the convolution halo.
Production estimated background and automatic offset independently inside every
compute tile. Changing the grid therefore changed nine nonlinear solver inputs,
so it could not be expected to reproduce the four-tile result.

A new `run.cmd tiling-invariance` experiment freezes background, offset, SNR,
and start policy once for the complete image, then changes only the XY compute
grid. On channel 0 of `DNA.ome.tiff`, 20 iterations produced:

| Variant | Wall time | Peak allocated | Comparison | SSIM | NRMSE | Flux difference | Gate |
|---|---:|---:|---|---:|---:|---:|:---:|
| Historical per-tile auto, 9 tiles | 26.94 s | 5,377 MB | accepted reference | 1 | 0 | 0 | Pass |
| Global preprocessing, 9 tiles | 12.23 s | 5,377 MB | vs historical 9 | 0.999734 | 0.001014 | 0.008057 | Fail |
| Global preprocessing, 4 tiles | 11.79 s | 9,727 MB | vs global 9 | 0.999978 | 0.000309 | 0.0000325 | Pass |

This confirms the user's expectation under tile-invariant preprocessing: four
and nine tiles pass the strict equivalence gates. It also shows why this should
not be silently enabled yet. Global preprocessing changes the historical result
by about 0.8% flux, and four tiles save only 0.44 s while consuming 4.35 GB more
VRAM. Nine tiles therefore remain the safer A5000 default until global
preprocessing is independently validated for both channels, RL-TV,
sparse-Hessian, and additional datasets. On H100/H200, larger tiles or untiled
execution may be worthwhile after the same invariance checks.

The axial/Z prototype remains rejected. RL is nonlinear and iterative, so
independently solving Z slabs is not equivalent to partitioning one convolution;
moreover, Z-only partitioning retains the dominant full-XY FFT planes. Reduced
arena caching does not change either limitation.

## NVIDIA GPU coverage strategy

The optimized algorithms use ordinary FP32 CUDA kernels and cuFFT; they do not
depend on an A5000-only hardware feature. Broad hardware support is therefore a
build, runtime-detection and testing problem rather than an algorithm rewrite.

NVIDIA's current
[compute-capability table](https://developer.nvidia.com/cuda/gpus) maps the
relevant production families as follows:

| Compute capability | Representative GPUs | Proposed support |
|---:|---|---|
| 8.0 | A100, A30 | Native cubin |
| 8.6 | RTX 3060 through RTX 3090, RTX A2000–A6000, A10/A40 | Native cubin |
| 8.9 | RTX 4050 through RTX 4090, RTX Ada, L4/L40/L40S | Native cubin |
| 9.0 | H100, H200, GH200 | Native cubin |
| 10.0 | B200, GB200 | Native cubin in modern build |
| 12.0 | RTX 50 series and RTX PRO Blackwell | Native cubin plus PTX |
| 10.3 / 12.1 | B300/GB300 and GB10-class newest devices | PTX fallback initially; native experimental build after validation |

The installed CUDA 12.6 compiler reports native targets only through `sm_90`.
It can build one useful compatibility artifact for Ampere, Ada and Hopper, but it
cannot natively target Blackwell. CUDA 12.8 introduced compiler support for
`sm_100`, `sm_101` and `sm_120`; CUDA 12.9 added `sm_103` and `sm_121`.
For current PyTorch packaging, CUDA 13.0 is the practical stable modern lane and
CUDA 13.2 is the optional newest-architecture lane.

### Recommended binary lanes

**Stable production lane (recommended default): PyTorch 2.11/2.12 + CUDA 13.0**

Build the extension with native targets and forward-compatible PTX equivalent to:

```text
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0+PTX"
```

This covers RTX 3060 through RTX 5090, A100/A30, A10/A40, L4/L40, H100/H200,
and B200/GB200 in one Windows or Linux fat binary. Exact cubins avoid PTX JIT on
known GPUs; the final PTX target provides a functional path for compatible newer
compute capabilities. NVIDIA notes that cubins are not compatible across major
compute-capability generations, while PTX is forward-compatible; native cubins
are still preferred for startup time and architecture-specific performance.

**Legacy transition lane: current PyTorch 2.11 + CUDA 12.6**

```text
TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX"
```

This supports Ampere through H100/H200 and lets development continue before the
toolchain migration. It does not provide native RTX 50/B200 support and should
not be advertised as the all-current-GPU package.

**Newest/experimental lane: PyTorch CUDA 13.2**

Use this to validate exact native coverage for 10.3/12.1-era devices such as
B300/GB300 and GB10. PyTorch currently describes its CUDA 13.2 builds as
experimental, so this should remain a separately tested image/artifact until the
upstream build becomes a stable release choice.

### Local and Docker implications

- **Local installed/standalone application:** ship the precompiled Windows fat
  binary; ordinary users need a sufficiently recent NVIDIA driver but no CUDA
  Toolkit, NVCC, Visual Studio or Ninja. Source compilation remains developer-only.
- **Docker/Bilayers/Slurm:** use a CUDA-devel builder stage to compile the Linux
  fat binary, then copy it into a smaller runtime image with the exactly pinned
  PyTorch CUDA build. The host supplies the NVIDIA driver through the container
  runtime. The same image can run on RTX, A-series, L-series and H100/H200 when
  its driver is new enough.
- **Driver baseline:** the CUDA 13 stable PyTorch lane requires a newer driver
  than the present CUDA 12.6 lane. PyTorch lists CUDA 13.0+ as the recommended
  route for Blackwell and gives minimum 580-series drivers for its modern builds.
  Installer diagnostics must report an actionable fallback rather than failing
  with “no kernel image”.
- **Memory scaling:** architecture compatibility does not mean identical tile
  capacity. A small RTX 3060 must use the memory preflight and more tiles, while
  H100/H200 can use much larger tiles. Backend selection and tile selection must
  remain separate decisions.
- **Runtime checks:** record GPU name, compute capability, driver, PyTorch CUDA
  version and backend artifact version; verify both PyTorch and the extension
  contain a usable cubin/PTX target; otherwise restart on the PyTorch fallback.
- **Multi-GPU/MIG:** H100/H200 MIG instances report the Hopper capability but
  expose less memory. Plans, workspaces and caches must be device-local and sized
  from the visible instance rather than the physical card model.

### Required hardware validation matrix

At minimum, release qualification should run one numerical and memory benchmark
on an RTX 3060 or equivalent 8.6 card, the existing RTX A5000, one Ada 8.9 card,
H100 or H200, and one Blackwell 12.0 card. H100 and H200 share `sm_90`, so one
validates kernel compatibility, but both should eventually receive memory/tile
profiling because their capacities and bandwidth differ. CI can inspect fatbins
with `cuobjdump`; actual hardware runs are still required before claiming tested
performance.

## PyTorch 2.13 / CUDA 13.2 migration validation (2026-07-14)

The local `deconvolve` environment now uses PyTorch `2.13.0+cu132`, CUDA Toolkit
`13.2.51`, NVIDIA driver `595.79`, and the RTX A5000 (`sm_86`). The extension
needed one source-level toolchain adjustment: CUDA 13.2 CCCL requires MSVC's
conforming preprocessor (`/Zc:preprocessor`). Extension build caches are now
isolated by Python, PyTorch, CUDA, and architecture, and the launcher refuses a
Toolkit/PyTorch CUDA-version mismatch.

Migration result directories are preserved alongside the CUDA 12.6 baselines:

- warmed crop: `results/20260714_114257`;
- warmed TV/sparse-Hessian: `results/20260714_114333`;
- warmed five-dataset sparse-Hessian: `results/20260714_114405`;
- full DNA volume: `results/20260714_113236`.

All recommended FP32 variants passed the existing SSIM, NRMSE, flux, finiteness,
and nonnegativity gates. The five sparse-Hessian datasets passed in both 2D and
3D. The ten expected failures in the full run were the already rejected
four-tile, untiled-oversubscribed, and approximate axial variants; CUDA 13.2 did
not change those conclusions. Static FP16 again passed but remains unrecommended.

Peak allocated memory was effectively unchanged. Performance was mixed:

- TV and most sparse-Hessian comparisons were unchanged or faster;
- the cached direct-cuFFT full-volume path was about 4.9-7.3% slower in wall
  time and about 5.2-5.3% slower in measured GPU time than the CUDA 12.6 run;
- full-volume production PyTorch was about 8.5-16.2% slower;
- the unsafe untiled path allocated about 32.9 GB, filled the 24 GB GPU, used a
  roughly 36 GB host working set, and remains disqualified.

The migration is therefore numerically accepted and memory-neutral, with a
documented RTX A5000 performance regression above the 5% review threshold for
some full-volume paths. The fused-kernel productionization decision remains a
separate change.

Docker validation built and tested these images:

| Image | Runtime | Size | Local validation |
|---|---|---:|---|
| `w_cideconvolve:v3.1.0` | PyTorch 2.13.0 + cu132 | 2.93 GB | GPU FFT, all three solvers, CPU fallback, DNAcrop OME-TIFF pass |
| `w_cideconvolve:v3.1.0-cu130` | PyTorch 2.13.0 + cu130 | 2.95 GB | GPU FFT, all three solvers, CPU fallback pass |
| `v3.1.0-gradio` / `v3.1.0-jupyter` | PyTorch 2.13.0 + cu132 | 3.07 / 3.01 GB | build and runtime import pass |

The PyTorch wheels contain native `sm_90`, but the H100 NVL with driver
`580.167.08` still requires the cluster smoke test before the unqualified cu132
tag is selected in BIOMERO. Use the cu130 tag if that compatibility-mode test
fails.

## Installation and distribution implications

### Recommended backend: no MathDx download for users

The selected direct-cuFFT backend uses standard cuFFT from CUDA/PyTorch, plus our
own small CUDA/C++ extension. It does **not** use cuFFTDx. Therefore, ordinary
`cideconvolve` users should not be asked to download MathDx.

The preferred distribution model is precompiled wheels for supported combinations
of operating system, Python, PyTorch/CUDA ABI, and GPU architecture. With such a
wheel, an end user needs:

- a sufficiently recent NVIDIA driver;
- the matching PyTorch CUDA installation;
- the `cideconvolve` wheel.

They should not need a local CUDA toolkit, NVCC, Visual Studio, Ninja, or MathDx.

For source builds or unsupported combinations, developers would need a compatible
CUDA toolkit, host C++ compiler, Ninja/build tooling, and headers/libraries matching
the PyTorch CUDA ABI. The existing pure-PyTorch production path should remain a
fallback when the compiled extension cannot be loaded.

### If cuFFTDx is pursued later

cuFFTDx is header-only but distributed as part of the separate MathDx archive.
NVIDIA states that downloading and using it accepts the NVIDIA Software License
Agreement. The license permits distribution of certain binaries, samples, and
headers when incorporated into an application and subject to its distribution
requirements; it does not justify silently downloading the SDK during a normal
package installation.

Recommended options, in order:

1. **Build-time dependency only:** project CI obtains the correct MathDx package
   under accepted terms, builds platform wheels, and users install those wheels
   without separately downloading MathDx.
2. **Optional developer source build:** developers download MathDx themselves,
   accept NVIDIA's terms, set `MATHDX_ROOT`, and build an optional cuFFTDx backend.
3. **Do not auto-download MathDx from `pip install`:** this creates license,
   network, reproducibility, CUDA-version, and offline-installation problems.

Before redistributing MathDx headers or derived binaries in public wheels, the
project should review the NVIDIA license and include required notices. This
document is an engineering recommendation, not legal advice.

### Other recommendations and user dependencies

| Feature | Additional end-user dependency if shipped in a prebuilt wheel? | Recommendation |
|---|---|---|
| Overwrite-state fused kernels | No | Enable in optimized backend |
| Direct cuFFT and shared workspace | No MathDx; compatible CUDA/PyTorch runtime required | Enable |
| Cached plans/OTF/weights | No | Enable |
| Static FP16 storage | No | Expert opt-in only |
| CUDA Graph capture | No | Leave disabled |
| Four-tile larger halos | No | Do not select automatically |
| Axial prototype | No | Do not ship as optimized path |
| cuFFTDx | MathDx needed at build time unless already compiled into wheels | Research-only |

## Productionization status

Completed: maintained production module, CUDA/PyTorch compatibility detection,
pure-PyTorch fallback, production tiling/callback integration, quality tests on
2D/3D and multiple microscopy datasets, workspace-aware memory handling, and
backend/work-shape/workspace provenance. Docker uses an AOT multi-architecture
extension; local source installations retain a cache-isolated JIT fallback.

Still required for a broad binary release: automated Linux/Windows binary CI,
clean-machine standalone testing, H100/H200 BIOMERO validation, and physical Ada
and Blackwell regression runs. These do not block the source/local A5000 path or
the validated cu132 Docker image, but they do limit which hardware can yet be
claimed as performance-tested.

## Reproduction

From `benchmarks/fft_optimization`:

```cmd
install.cmd
run.cmd quick
run.cmd full
run.cmd regularizers --iterations 20
run.cmd sparse-datasets --iterations 20
```

Generated JSON, CSV, Markdown, and QC PNG files are written under timestamped
`results/` directories. The benchmark uses the repository's `deconvolve` Conda
environment and loads the maintained extension sources from
`core/optimized_cuda`.
