# CIDeconvolve

**GPU-accelerated 3-D / 2-D microscopy deconvolution with SHB Richardson-Lucy.**

CIDeconvolve is a [BIAFLOWS](https://biaflows.neubias.org/)-compatible
workflow that deconvolves widefield and confocal fluorescence microscopy
images.  It generates a physically accurate PSF from OME-TIFF metadata
(Gibson–Lanni model) and applies Scaled Heavy Ball (SHB) accelerated
Richardson-Lucy deconvolution with optional Total Variation regularisation —
all on the GPU via PyTorch.

| | |
|---|---|
| **Docker image** | `cellularimagingcf/w_cideconvolve` |
| **Version** | v1.0.0 |
| **Container type** | Singularity (pulled from Docker Hub) |
| **Methods** | `ci_rl` (SHB-accelerated RL) · `ci_rl_tv` (+ TV regularisation) |
| **Benchmark** | built-in with timing metrics CSV and MIP montages |

---

## Methods

### `ci_rl` — Scaled Heavy Ball Accelerated Richardson-Lucy

Standard Richardson-Lucy enhanced with **Scaled Heavy Ball (SHB) momentum
acceleration** (Wang & Miller 2014).  Achieves 5–10× faster convergence
than vanilla RL at no extra per-iteration cost.  Includes Bertero boundary
correction weights and I-divergence convergence monitoring.

### `ci_rl_tv` — SHB-RL with Total Variation Regularisation

Same as `ci_rl` with an additional **Total Variation (TV) penalty** after
each RL update (Dey et al. 2006).  Suppresses noise amplification at high
iteration counts while preserving edges.  Controlled by the `--tv_lambda`
parameter (typical range 0.0001–0.01).

For full algorithmic details see [DECONVOLVE_CI.MD](DECONVOLVE_CI.MD).

---

## Using CIDeconvolve with BIOMERO

[BIOMERO](https://github.com/NL-BioImaging/biomero) (BioImage Analysis in
OMERO) lets you run FAIR bioimage-analysis workflows from an OMERO server
on a SLURM-based HPC cluster.  CIDeconvolve is designed to plug directly
into this framework.

### How it works

1. The OMERO admin configures the workflow in
   **`slurm-config.ini`** on the SLURM submission host by adding a section
   for `W_CIDeconvolve`:

   ```ini
   [SLURM]
   # ... global SLURM settings ...

   [W_CIDeconvolve]
   # Override default SLURM resources for this workflow
   job_cpus=8
   job_memory=52G
   job_gres=gpu:2g.24gb
   ```

2. BIOMERO reads **`descriptor.json`** from the container to discover
   input parameters (method, iterations, device, PSF settings, benchmark
   options, etc.) and presents them in the OMERO web UI.

3. On submission, BIOMERO pulls the Singularity image from Docker Hub,
   transfers the selected images, and executes the workflow on the cluster.

4. Results (deconvolved images, benchmark montages, metrics CSV) are
   automatically uploaded back into OMERO.

> For full BIOMERO setup instructions see the
> [BIOMERO documentation](https://nl-bioimaging.github.io/biomero/)
> and the [NL-BIOMERO deployment repo](https://github.com/NL-BioImaging/NL-BIOMERO).

### SLURM job script

A ready-made SLURM script is provided for manual cluster submission
(outside of BIOMERO):

```bash
sbatch cideconvolve.slurm \
    --infolder /data/myimages \
    --outfolder /data/results \
    -- --method ci_rl --iterations 40 --benchmark True
```

See `cideconvolve.slurm` for full usage and resource settings.

---

## Building the Docker image locally

```bash
docker build -t w_cideconvolve:v1.0.0 -t w_cideconvolve:latest .
```

The Dockerfile builds on the **NVIDIA CUDA 12.6 runtime** image with
Python 3.11 and all pip dependencies — no Java, no conda, no compilation
step required.

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  (for GPU pass-through at runtime)
- A working `docker build` environment (Docker Desktop on Windows/macOS,
  or Docker Engine on Linux)

---

## Running locally with Docker

```bash
docker run --rm --gpus all \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    -v /tmp/gt:/data/gt \
    cellularimagingcf/w_cideconvolve \
    --infolder /data/in --outfolder /data/out --gtfolder /data/gt \
    --method ci_rl --iterations 40 \
    --na auto --refractive_index auto --sample_ri auto \
    --microscope_type auto --emission_wl auto --excitation_wl auto
```

Replace paths as needed.  The `--gpus all` flag enables NVIDIA GPU
pass-through.  Omit it to force CPU-only execution.

### Benchmark mode

```bash
docker run --rm --gpus all \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    -v /tmp/gt:/data/gt \
    cellularimagingcf/w_cideconvolve \
    --infolder /data/in --outfolder /data/out --gtfolder /data/gt \
    --benchmark True --bench_crop True
```

Benchmark mode deconvolves the first input image with both `ci_rl` and
`ci_rl_tv` at multiple iteration counts (20, 40, 60), writes a CSV with
timing and image-quality metrics (sharpness, contrast, noise proxy), and
generates MIP montage images.

---

## Running locally without Docker

### Requirements

- Python 3.10 or 3.11
- PyTorch 2.4+ with CUDA support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### CLI

```bash
python wrapper.py \
    --infolder ./infolder --outfolder ./outfolder --gtfolder ./gtfolder \
    --method ci_rl --iterations 40 \
    --na auto --refractive_index auto --sample_ri auto
```

### Launcher (GUI)

A PyQt6-based launcher provides a graphical interface with parameter
controls, folder pickers, and a live command preview:

```bash
python launcher.py
```

The launcher saves your last-used settings and can restore them on next
launch via the **Restore Last Settings** button.

---

## Parameters

All parameters are defined in `descriptor.json` and exposed on the
command line via `wrapper.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 40 | Number of RL iterations (comma-separated for per-channel) |
| `--tiling` | custom | Tiling mode: `none` or `custom` |
| `--tile_limits` | 512, 64 | Max tile dimensions `max_xy, max_z` (when tiling = `custom`) |
| `--method` | ci_rl | Deconvolution method: `ci_rl` or `ci_rl_tv` |
| `--device` | auto | Compute device: `auto`, `cpu`, `cuda` |
| `--na` | auto | Numerical aperture override |
| `--refractive_index` | auto | Immersion medium RI (`air`, `water`, `oil`) |
| `--sample_ri` | auto | Sample/mounting medium RI (named presets available) |
| `--microscope_type` | auto | `widefield` or `confocal` |
| `--emission_wl` | auto | Emission wavelengths in nm (comma-separated) |
| `--excitation_wl` | auto | Excitation wavelengths in nm (for confocal PSF) |
| `--tv_lambda` | 0.001 | TV regularisation strength (only for `ci_rl_tv`) |
| `--background` | auto | Background subtraction: `auto`, numeric value, or `0` to disable |
| `--damping` | none | Noise suppression damping factor: `none`, `auto`, or numeric |
| `--convergence` | auto | Early-stopping convergence: `auto` or `none` |
| `--rel_threshold` | 0.005 | Relative change threshold for early stopping |
| `--pixel_size_xy` | auto | Lateral pixel size in nm |
| `--pixel_size_z` | auto | Axial pixel size (Z step) in nm |
| `--projection` | none | Z-projection: `none`, `mip`, `sum` |
| `--benchmark` | false | Run benchmark mode |
| `--bench_crop` | false | Centre-crop image to tile-size limits before benchmarking |

---

## Project structure

```
wrapper.py              BIAFLOWS entrypoint — parameter parsing, benchmark runner, metrics
deconvolve.py           Core deconvolution engine + PSF generation
deconvolve_ci.py        CI SHB-RL / RLTV implementation (PyTorch)
launcher.py             PyQt6 GUI launcher
gui_deconvolve_ci.py    GUI deconvolution panel
descriptor.json         BIAFLOWS/BIOMERO parameter descriptor
bioflows_local.py       Local BIAFLOWS compatibility shim
cideconvolve.slurm      SLURM job script for HPC execution
Dockerfile              Docker build (CUDA 12.6 runtime + Python 3.11)
requirements.txt        Python dependencies (local install)
requirements_docker.txt Python dependencies (Docker)
vendor/                 Vendored libraries (psf_generator)
```

---

## References

- **SHB Acceleration:** Wang, Y. & Miller, E. L. (2014). "Scaled Heavy-Ball Acceleration of the Richardson-Lucy Algorithm for 3D Microscopy Image Restoration." *IEEE TIP* **23**(12), 5284–5297.
- **TV Regularisation:** Dey, N. et al. (2006). "Richardson-Lucy Algorithm With Total Variation Regularization for 3D Confocal Microscope Deconvolution." *Microsc. Res. Tech.* **69**(4), 260–266.
- **BIOMERO:** Luik, T. T., Rosas-Bertolini, R., Reits, E. A. J., Hoebe, R. A. & Krawczyk, P. M. (2024). "BIOMERO: A scalable and extensible image analysis framework." *Patterns* **5**(8), 101024. [doi:10.1016/j.patter.2024.101024](https://doi.org/10.1016/j.patter.2024.101024) · [GitHub](https://github.com/NL-BioImaging/biomero) · [Documentation](https://nl-bioimaging.github.io/biomero/)
- **BIAFLOWS:** Rubens, U. et al. (2020). "BIAFLOWS: A Collaborative Framework to Reproducibly Deploy and Benchmark Bioimage Analysis Workflows." *Patterns* **1**(3), 100040. [doi:10.1016/j.patter.2020.100040](https://doi.org/10.1016/j.patter.2020.100040)
- **PSF Generator:** Kirshner, H. et al. — [EPFL PSF Generator](https://bigwww.epfl.ch/algorithms/psfgenerator/)
- **Gibson–Lanni model:** Gibson, S. F. & Lanni, F. (1992). [doi:10.1364/JOSAA.9.000154](https://doi.org/10.1364/JOSAA.9.000154)
- **OMERO:** Allan, C. et al. (2012). "OMERO: flexible, model-driven data management for experimental biology." *Nat Methods* **9**, 245–253. [doi:10.1038/nmeth.1896](https://doi.org/10.1038/nmeth.1896)

---

## License

MIT — see [LICENSE](LICENSE).


