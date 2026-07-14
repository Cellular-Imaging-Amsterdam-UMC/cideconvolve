# FFT optimization benchmark

The durable summary of all measured results, accepted/rejected experiments, and
installation implications is in [RESULTS_AND_RECOMMENDATIONS.md](RESULTS_AND_RECOMMENDATIONS.md).

This folder is deliberately independent of the production solver. It imports the
repository's readers, PSF generator, writers, and numerical helpers, but does not
modify them.

Install the one benchmark-only build dependency and run the crop benchmark:

```cmd
install.cmd
run.cmd quick
```

The launcher defaults to PyTorch 2.13 with CUDA Toolkit 13.2 and the local A5000
architecture (`8.6`). Override `PY`, `CUDA_VERSION`, `CUDA_HOME`, or
`TORCH_CUDA_ARCH_LIST` before calling `run.cmd` for another matched environment.
The preflight rejects a Toolkit/PyTorch CUDA mismatch before compilation.

Run the selected variants on the full two-channel DNA volume:

```cmd
run.cmd full
```

Profile the optimized FP32 FFT data step with the production TV and
sparse-Hessian regularizers on the two-channel crop:

```cmd
run.cmd regularizers --iterations 20
```

This preset compares each method against its own production output and reports
the FFT/data-step and regularizer GPU times separately. It benchmarks both the
production PyTorch regularizer and benchmark-only fused CUDA TV/sparse-Hessian
kernels. Small-tensor formula/gradient equivalence checks run first and abort the
benchmark on failure. The preset intentionally omits static FP16 and additional
tiling experiments because those options were already rejected by the `ci_rl`
study.

Validate the fused sparse-Hessian kernel on a representative localdata suite
containing two 2D/singleton-Z and three anisotropic odd-shaped 3D OME-TIFFs:

```cmd
run.cmd sparse-datasets --iterations 20
```

This preset uses production-exact FFT dimensions to isolate regularizer behavior,
passes physical XY/Z sampling into both implementations, and reports quality
against production plus a direct fused-vs-PyTorch-regularizer comparison.

The full suite includes both the production-equivalent nine-tile control and
four-tile variants for measuring the speed/detail tradeoff from larger tiles.

Test tile-count invariance separately by freezing automatic background and
offset once for the complete image, then comparing 9 and 4 compute tiles:

```cmd
run.cmd tiling-invariance
```

The default is channel 0 of `DNA.ome.tiff` with 20 iterations. Optional flags
include `--channel`, `--iterations`, `--reference-tiles`, and
`--candidate-tiles`.

The extended full preset benchmarks all eight follow-up ideas:

1. overwriting the dead previous RL estimate;
2. direct in-place cuFFT R2C/C2R transforms;
3. one explicit caller-owned cuFFT workspace;
4. cached OTF, boundary weights, plans and allocation arenas by tile geometry;
5. optional FP16 storage for static OTF/weight tensors with FP32 FFT/state math;
6. cuFFTDx capability detection (requires a separately installed MathDx package);
7. four-tile halo margins of 16, 32 and 64 pixels;
8. an explicitly approximate, quality-gated two-part axial RL prototype.

The untiled direct-cuFFT variant is also attempted. On a 24 GB RTX A5000 it can
invoke Windows/CUDA memory oversubscription, so both allocated/reserved counters
and wall time must be considered rather than treating completion as proof that it
is a safe production configuration.

Results are written below `results/<timestamp>/`. Full deconvolved volumes are not
saved unless `--save-full-volumes` is supplied. Run `benchmark.py --help` for all
options.

The LTO callback probe requires a CUDA toolkit whose `cufftXt.h` exports
`cufftXtSetJITCallback`. CUDA 12.6.20 does not export that API, so the report marks
the historical CUDA 12.6 baseline unavailable rather than substituting legacy
callbacks. Each current run probes the selected toolkit again.
