# CIDeconvolve

**GPU-accelerated 3-D / 2-D fluorescence microscopy deconvolution — SHB Richardson-Lucy, TV regularisation, and sparse-Hessian variational solver, all via PyTorch.**

| | |
|---|---|
| **Docker image** | `cellularimagingcf/w_cideconvolve` |
| **Website** | https://cellular-imaging-amsterdam-umc.github.io/cideconvolve/ |
| **Version** | v3.0.0 |
| **Container type** | Docker image; BIOMERO/HPC deployments can pull or convert it to Singularity / Apptainer |
| **Methods** | `ci_rl` · `ci_rl_tv` · `ci_sparse_hessian` |
| **Benchmark** | built-in with timing metrics CSV and MIP montages |

---

## Overview

CIDeconvolve is a Bilayers / BIOMERO-compatible workflow that deconvolves widefield and confocal fluorescence microscopy images. It reads OME-TIFF / OME-Zarr metadata where available, auto-generates a physically accurate PSF from the optical parameters, and applies native GPU-capable deconvolution methods.

**Three user-facing entry points:**

| Entry point | Purpose |
|---|---|
| `gui_deconvolve_ci.py` | Full standalone interactive GUI — open files, configure parameters, run deconvolution, inspect results side-by-side |
| `launcher.py` | Docker launcher GUI backed by Bilayers `config.yaml` |
| `wrapper.py` | Bilayers / BIOMERO CLI entrypoint for batch and HPC use |

---

## Deconvolution Methods

### `ci_rl` — Scaled Heavy Ball Accelerated Richardson-Lucy

Standard Richardson-Lucy enhanced with **Scaled Heavy Ball (SHB) momentum acceleration** (Wang & Miller 2014).  Achieves 5–10× faster convergence than vanilla RL at no extra per-iteration cost.  Includes Bertero boundary-correction weights and I-divergence convergence monitoring.

**Best for:** Fast, stable deconvolution of most microscopy images.

### `ci_rl_tv` — SHB-RL with Total Variation Regularisation

Same SHB-RL engine with an additional **Total Variation (TV) penalty** after each update (Dey et al. 2006).  Suppresses noise amplification at high iteration counts while preserving edges.  Controlled by `--tv_lambda` (typical range 0.00005–0.001).

**Best for:** Noisy data where edge preservation is important; higher iteration counts.

### `ci_sparse_hessian` — Sparse-Hessian Variational Deconvolution

A quality-focused **sparse-Hessian / SPITFIRE-style** variational method.  Combines the same FFT-based forward model and preprocessing stack with a sparse-Hessian prior that favours thin, high-contrast structures while suppressing noise.  Controlled by `--sparse_hessian_weight` (0–1) and `--sparse_hessian_reg` (0–1).

**Best for:** Filaments, membranes, and synapses; sparse structures that need to stand out against diffuse background.

### Stabilisation and PSF options

- **Positive offsetting** (`--offset`) — pre-shift before iteration to prevent division by zero
- **Anscombe prefiltering** (`--prefilter_sigma`) — variance-stabilised Gaussian smoothing before deconvolution
- **Initial estimate** (`--start auto|flat|percentile_flat|observed|observed_bgsub|lowpass|lowpass_bgsub|hybrid`)
- **Auto early stopping** (`--convergence auto`) — halts when relative I-divergence change < `--rel_threshold`
- **Enhanced 2D widefield mode** (`--two_d_mode auto`) — collapses a full 3D Gibson-Lanni PSF to 2D for single-plane widefield data with aggressiveness and background controls
- **Physically accurate PSF** — vectorial Richards-Wolf model (NA ≥ 0.9) or scalar Kirchhoff (NA < 0.9), Gibson-Lanni OPD aberration correction, sub-pixel integration, finite confocal pinhole convolution
- **Automatic memory tiling** — tiles large volumes with feathered overlap to fit GPU or CPU RAM
- **Streaming large tilescans** — optional OME-Zarr output path reads halo-extended regions and writes tiles/pyramids without materialising full 40k × 40k arrays

For full algorithmic details see [DECONVOLVE_CI.MD](docs/DECONVOLVE_CI.MD).

---

## GUI — Interactive Deconvolution (`gui_deconvolve_ci.py`)

![GUI Deconvolution Panel](docs/screenshots/gui_deconvolve_ci.png)

The standalone interactive GUI is the primary tool for exploratory deconvolution.  It now uses a top workflow bar for opening data, running jobs, exporting results, managing settings, opening the log/help, and launching batch mode.  The left pane contains deconvolution and optics parameters; the right pane is a synchronized 2D/3D viewer for original and deconvolved data.

### Running the GUI

```bash
python gui/gui_deconvolve_ci.py
```

The title bar shows the detected PyTorch version and GPU (e.g. `CI Deconvolve — torch 2.13.0 | NVIDIA RTX 4090 CUDA 13.2`).

#### GUI-only command-line flags

These flags are understood only by `gui_deconvolve_ci.py` and are **not** part of `config.yaml` or `wrapper.py`.

| Flag | Effect |
|---|---|
| `--movie` | Reveals the **Iteration Movie** panel in Advanced Parameters for exporting per-iteration MP4 / GIF recordings. |

Example — enable movie export:

```bash
python gui/gui_deconvolve_ci.py --movie
```

### Layout

| Area | What it contains |
|---|---|
| **Top workflow bar** | **Open…**, source **Recent**, run progress/status, **Save…**, **Save T-Series…**, **Settings…**, settings **Recent**, **Log**, **Help**, **Batch…**, and **Run Deconvolution** |
| **Left controls** | Loaded-file summary, metadata warning/reset controls, method settings, 2D widefield controls, optics/PSF, refractive indices, advanced parameters, optional movie controls, and run history |
| **Right viewer** | Channel buttons, 2D/3D view modes, projection, display scaling, navigator, scale bar, Z/T sliders, and linked original/deconvolved panes |
| **Status bar** | Cursor readout, messages, and live CPU/RAM/SWAP/GPU/VRAM/SPILL monitor |

Keyboard shortcuts: `Ctrl+O` open, `Ctrl+R` run, `Ctrl+S` save OME-TIFF, `Ctrl+Shift+S` save settings, `Ctrl+L` log, `F1` help.

### Image loading

Use the **Open…** menu in the top workflow bar:

| Menu item | Supported source |
|---|---|
| **Open File…** | OME-TIFF, TIFF, ND2, CZI, LIF, and any format supported by BioIO |
| **Open OME-Zarr…** | Local OME-Zarr folders, including HCS plates/fields |
| **Open Leica…** | Leica LIF / LOF / XLEF containers via `leica-browser-qt` |
| **Open OMERO…** | OMERO server — browse projects/datasets/images (requires `omero-browser-qt[viewer]`) |
| **Recent** | Reopen recent local files, Zarr folders, Leica entries, and supported recent sources |

Drag-and-drop uses the same loading path as the Open menu.  Large pyramidal OMERO images are opened through the OMERO tile/pyramid reader instead of downloading the full plane.  The viewer loads the overview quickly and requests higher-resolution tiles as you zoom, matching the behaviour of the OMERO viewer in `omero-browser-qt`.

### Deconvolution controls (left panel)

#### Method
| Control | Default | Options |
|---|---|---|
| Method | `ci_rl` | `ci_rl`, `ci_rl_tv`, `ci_sparse_hessian` |
| Iterations | `80` widefield / `50` confocal | comma-separated per-channel |
| Convergence | `auto` | `auto`, `fixed` |
| Rel. threshold | `0.001` | 1×10⁻⁸ – 1.0 |
| Start | `auto` | `auto`, `flat`, `percentile_flat`, `observed`, `observed_bgsub`, `lowpass`, `lowpass_bgsub`, `hybrid` |

#### 2D Widefield
| Control | Default | Notes |
|---|---|---|
| 2D WF model | `Auto` | Uses the widefield-aware collapsed-PSF model for 2D widefield images; legacy mode is retained for old settings |
| 2D WF aggressiveness | `Balanced` | `Very Conservative`, `Conservative`, `Balanced`, `Strong`, `Very Strong` |

#### Optics / PSF
| Control | Default | Notes |
|---|---|---|
| NA | `1.4` | 0.1 – 2.0 |
| Emission (nm) | `520` | comma-separated per channel |
| Excitation (nm) | `488` | comma-separated per channel (confocal only) |
| Pixel XY (nm) | `65.0` | lateral pixel size |
| Pixel Z (nm) | `200.0` | axial step size |
| Microscope | `confocal` | `widefield`, `confocal` |
| Pinhole (AU) | `1.0` | Airy disk units per channel; `0` = legacy point-detector; hidden for widefield |

#### Refractive indices
| Control | Default | Options |
|---|---|---|
| RI immersion | `1.515` | air, water, oil and more |
| Embedding medium | `prolong gold (1.47)` | 8 standard presets |
| RI sample | `1.47` | editable spin box |

#### Advanced parameters (collapsible)
- **Method Tuning** — TV lambda, sparse-Hessian weight/reg, background mode/value, offset mode/value, prefilter sigma, convergence check interval, and backend (`Auto`, `Optimized CUDA`, `PyTorch CUDA`, `CPU`). Auto uses the compiled direct-cuFFT backend when compatible and otherwise falls back safely.
- **2D Widefield Expert** — background estimator radius (`0.50 µm`) and auto background scale (`1.00`) for 2D widefield auto mode.
- **Coverslip / Depth** — actual/design coverslip thickness, design immersion thickness, and particle depth.
- **PSF Advanced** — pixel integration toggle, sub-pixel count, and pupil sampling density.
- **Iteration Movie** — shown only with `--movie`; exports MP4 and optional half-size GIF recordings of the iteration sequence.

### Dual-pane viewer (right panel)

| Control | Description |
|---|---|
| Channel buttons | Left-click toggles visibility; right-click changes channel colour |
| View mode | `2D Both`, `2D Original`, `2D Deconvolved`, `2D Linked split`, `2D Blink`, `2D Difference`, `2D Ratio`, `3D Both`, `3D Original`, `3D Deconvolved` |
| Projection | `Slice`, `MIP`, or `SUM` for Z data |
| Fit | `fitInView` on both panes simultaneously |
| Smooth zoom | Preview interpolation toggle; does not alter image data |
| Navigator | Small interactive overview/minimap for zoomed 2D views |
| Scale bar | Physical scale overlay when pixel size is known |
| Lo% / Hi% | Percentile-based contrast (defaults 0.1 % / 100 %) |
| Adv. Scaling | Opens dedicated scaling dialog |
| Z slider | Vertical slider for plane navigation |
| Log / Help | Top-bar buttons and shortcuts: `Ctrl+L` opens the log; `F1` opens the searchable in-app help |

In 3D mode additional controls appear:
- **Render method:** MIP, Attenuated MIP, MinIP, Translucent, Average, Isosurface, Additive
- **Gain / Threshold / Attenuation** slider
- **Downsample** (1×, 2×, 4×) and **Smooth** toggle
- **Reset View** (resets arcball camera)

Both 2D panes are linked for synchronized pan and zoom. In **Linked split** mode, left-drag moves the split position and right-drag pans the image. Difference and ratio modes are display-normalized previews for visual comparison, not quantitative output.

### Advanced scaling dialog

A detached 420×680 window with:
- Per-channel visibility checkboxes and colour pickers
- Per-pane (Original + Deconvolved) min/max sliders and spinboxes
- Gamma (0.10–5.00, default 1.0)
- Auto / Reset buttons
- Dual stacked histograms (Original / Deconvolved) with draggable range markers and log-scale toggle

### Resource monitor

A live status bar shows **CPU | RAM | SWAP | GPU | VRAM | SPILL** updated every 500 ms.  Bars are green < 70 %, orange 70–90 %, red ≥ 90 %.  A green activity dot (●) pulses during deconvolution.  PyTorch VRAM spill (Windows pagefile overflow) is tracked separately.

### Saving results

Use the **Save…** menu in the top workflow bar:

| Menu item | Action |
|---|---|
| **Save as OME-TIFF…** | Save the current timepoint deconvolution result as LZW-compressed OME-TIFF |
| **Save as OME-Zarr…** | Save the current timepoint deconvolution result as chunked OME-Zarr |
| **Save Views as PNG…** | Export the current visual original/deconvolved preview |
| **Save Comparison as PNG…** | Export the active comparison mode with display settings and scale bar |
| **Save T-Series…** | Export full T-series using memory-mapped staging |

For full-resolution streamed OMERO pyramid jobs, **Run Deconvolution** asks for an OME-Zarr output path and writes tiles directly to that directory. OME-TIFF and OME-Zarr exports preserve physical pixel metadata where available; PNG exports are display/rendering snapshots for reports.

### Settings and run history

The **Settings…** menu can restore the last GUI settings, save the current parameter set to JSON, or load a saved settings JSON.  The adjacent settings **Recent** menu reopens recently used settings files.

The **Run History** section records recent runs with time, image, method, status, and output path.  From there you can restore parameters, open the output folder, copy a run summary, or remove an entry.

### Batch Deconvolver

The **Batch…** button opens a session-only batch dialog for sequential GPU-safe processing.  It is intended for experiments where many images share the same optics and deconvolution settings.

Batch inputs can be added from:

| Button | Batch source |
|---|---|
| **Open…** | Multi-select regular image files |
| **Open Zarr…** | Multi-select OME-Zarr folders |
| **Open Leica…** | Multi-select images from Leica containers |
| **Open OMERO…** | Multi-select OMERO images using the existing OMERO login/session |

Batch workflow:

1. Add images to the list.
2. Choose a saved settings JSON.  These settings are the deconvolution/optics source of truth for the whole batch.
3. Choose an output folder and format.  **OME-TIFF** is the default; **OME-Zarr** is also available.
4. Optionally choose **Z output**: full stack, MIP, SUM, or Mean.  For 3D images, projection modes save only the projection.
5. Start the batch.  Processing is strictly sequential to protect GPU memory, OMERO sessions, Leica handles, and large-image I/O.

Each row stores the output folder that was selected when that image was added, so you can change the output folder and add another group of images to a different destination.  The table shows per-row status, progress, source, display name, shape, output folder, output filename, and messages such as current tile progress.  The bottom status bar shows elapsed time, ETA, predicted end date/time, and image progress, assuming images are roughly equal and refining the estimate after each tile update.

Default batch output names are concise:

```text
Position_3_decon_mip.ome.tiff
Position_3_decon.ome.zarr
```

For Leica sources, `save_child_name` is used as the table name and output filename base when available.

Batch OME-TIFF output uses streamed BigTIFF writing with LZW lossless compression.  OME-Zarr output is chunked and multiscale.  Existing outputs at the target path are overwritten.

### Image quality metrics

Computed on both input and output (≤ 32 Z-planes, ≤ 512 YX) and shown in the log:

| Metric | Description |
|---|---|
| `detail_energy` | FFT power fraction above 25 % of max frequency |
| `bright_detail_energy` | Same, restricted to top 5 % intensity pixels |
| `edge_strength` | Mean gradient magnitude |
| `signal_sparsity` | Gini coefficient approximation |
| `robust_range` | p99.5 − p0.5 |

---

## CLI — Running Locally Without Docker

### Installation

**Requirements:** Python 3.10 or 3.11 and a current NVIDIA driver. The default
GPU environment uses PyTorch 2.13 with CUDA 13.2.

```bash
pip install -r requirements_gui.txt
```

`requirements_gui.txt` installs exactly `torch==2.13.0+cu132`. CUDA 13.x needs
an NVIDIA driver from the 580 series or newer; driver 595.45.04 or newer is
recommended for full CUDA 13.2 support. The H100 NVL cluster driver 580.167.08
uses CUDA 13.x minor-version compatibility and must pass the documented cluster
smoke test before the CUDA 13.2 image is promoted there.

The **Open OMERO…** button requires `omero-browser-qt`, which depends on **ZeroC ICE**.  ZeroC ICE is not on PyPI and must be installed from a pre-built wheel matching your Python version and platform before running `pip install -r requirements_gui.txt`.

OME-TIFF compression uses `tifffile` plus `imagecodecs`; both are listed in `requirements_gui.txt`.

Download the wheel from the [zeroc-ice releases](https://github.com/zeroc-ice/ice/releases) or the [zeroc-ice PyPI mirror](https://zeroc.com/downloads/ice).  For the supported environment (Python 3.11, Windows x86-64):

```bash
# Example — adjust the filename to the exact release
pip install zeroc_ice-3.7.10-cp311-cp311-win_amd64.whl
pip install -r requirements_gui.txt
```

> **Note:** Use the `cp311` wheel — the `cp312` wheel is not compatible with Python 3.11.  If ZeroC ICE is not installed, all GUI features work normally except the OMERO browser.

### Basic usage

```bash
python wrapper.py \
    --infolder ./infolder \
    --outfolder ./outfolder \
    --method ci_rl --iterations 40
```

### Benchmark mode

```bash
python wrapper.py \
    --infolder ./infolder --outfolder ./outfolder \
    --benchmark True --bench_crop True --compute_metrics True
```

Runs the three methods (`ci_rl`, `ci_rl_tv`, and `ci_sparse_hessian`), writes `benchmark_metrics_*.csv` with per-method timing and quality metrics, and generates MIP montage comparison images.
See [metrics.md](docs/metrics.md) for metric formulas and interpretation.

### Parameters

The public CLI parameters are defined in `config.yaml` and exposed via `wrapper.py`:

#### Core parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--method` | `ci_rl` | `ci_rl`, `ci_rl_tv`, or `ci_sparse_hessian` |
| `--iterations` | `60` | RL iterations; comma-separated for per-channel |
| `--convergence` | `auto` | Early stopping: `auto` or fixed iteration count: `fixed` |
| `--rel_threshold` | `0.005` | Relative I-divergence change threshold for early stopping |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`; automatic CUDA execution prefers the compatible optimized backend |
| `--projection` | `none` | Z-projection: `none`, `mip`, `sum`, or `mean` |
| `--output_format` | `ome-zarr` | `ome-tiff` or chunked multiscale `ome-zarr` |
| `--output_dtype` | `float32` | `float32` for quantitative output, or globally scaled `uint16` to reduce size without clipping high values |
| `--streaming` | `auto` | `auto`, `always`, or `never`; auto enables region reads above `--streaming_threshold_gb` |
| `--streaming_threshold_gb` | `2.0` | Estimated full source-array size that triggers streaming auto mode |
| `--t_start` | `1` | First T frame to save, using 1-based inclusive indexing |
| `--t_stop` | `0` | Last T frame to save, using 1-based inclusive indexing; `0` means final frame |
| `--t_step` | `1` | Save every Nth T frame in the selected T range |
| `--hcs_field` | `auto` | Optional OME-Zarr HCS field path for single-field reads, for example `A/1/0`; full HCS plate inputs ignore this and process every field |

#### PSF / optics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--na` | `1.4` | Numerical aperture fallback / override |
| `--emission_wl` | `520` | Emission wavelength in nm; comma-separated per channel |
| `--excitation_wl` | `488` | Excitation wavelength in nm; comma-separated per channel |
| `--pixel_size_xy` | `65` | Lateral pixel size in nm |
| `--pixel_size_z` | `200` | Axial step size in nm |
| `--microscope_type` | `confocal` | `widefield` or `confocal` |
| `--pinhole_airy` | `1.00` | Confocal pinhole in Airy disk units; comma-separated per channel; `0` = point-detector |
| `--refractive_index` | `oil (1.515)` | Immersion medium RI |
| `--sample_ri` | `prolong gold (1.47)` | Sample / mounting medium RI |
| `--overrule_image_metadata` | `false` | When `true`, CLI values replace image metadata |

#### Stabilisation (RL-family)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tv_lambda` | `0.0001` | TV regularisation strength (for `ci_rl_tv`; typical 0.00005–0.001) |
| `--offset` | `auto` | Positive processing offset: `auto`, `none`, or numeric |
| `--prefilter_sigma` | `0.0` | Anscombe-domain Gaussian prefilter sigma in pixels |
| `--snr_mode` | `none` | Bilayers SNR selection: `none`, `auto`, or `manual` |
| `--snr_value` | `4.0` | Positive SNR used only when `--snr_mode manual` |
| `--acuity` | `0` | Smooth/sharp balance from `-100` to `+100` when SNR is enabled |
| `--start` | `auto` | Initial estimate: `auto`, `flat`, `percentile_flat`, `observed`, `observed_bgsub`, `lowpass`, `lowpass_bgsub`, or `hybrid` |
| `--background` | `auto` | Background subtraction: `auto`, numeric, or `0` to disable |
| `--two_d_mode` | `auto` | 2D widefield mode: `auto` (widefield-aware PSF) or `legacy_2d` |
| `--two_d_wf_aggressiveness` | `Balanced` | PSF collapse aggressiveness preset for 2D widefield auto mode |
| `--two_d_wf_bg_radius_um` | `0.5` | Background estimator radius in µm |
| `--two_d_wf_bg_scale` | `1.0` | Background estimator scale factor |

#### Sparse-Hessian

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sparse_hessian_weight` | `0.6` | Hessian-vs-sparsity balance (0–1) |
| `--sparse_hessian_reg` | `0.98` | Data-vs-regulariser balance (0–1) |

#### Benchmark

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--benchmark` | `false` | Run the three classical benchmark methods and write timing CSV + MIP montages |
| `--bench_crop` | `false` | Centre-crop to at most 512 × 512 pixels in XY and 64 Z slices before benchmarking |
| `--compute_metrics` | `false` | Compute optional FFT / gradient quality metrics |

---

### Large tilescans / streaming output

For very large images, use streaming output so CIDeconvolve can read and write
tiles instead of materialising the full image in RAM.  OME-Zarr remains the
preferred format for very large float32 tilescans because it is chunked,
resumable, and viewer-friendly, but the GUI batch workflow also supports tiled
OME-TIFF output.

```bash
python wrapper.py \
    --infolder ./infolder --outfolder ./outfolder \
    --method ci_rl --iterations 50 \
    --output_format ome-zarr --streaming always
```

Streaming mode reads halo-extended XY regions, deconvolves each tile with the
existing CI solver, and writes the tile core directly to the output.  OME-Zarr
and OME-TIFF exports build XY pyramid levels.  OME-TIFF exports are written as
tiled BigTIFFs without private tags for QuPath/Bio-Formats compatibility;
float32 TIFFs keep TIFF predictors off, while uint16 TIFFs use the standard
integer predictor.  `--output_dtype uint16` maps the full float output range to
`0..65535` and records the scale/offset in CIDeconvolve metadata so high values
are not clipped.  Tile size is selected
automatically from the source shape, method, device, and available memory.  For
3D data the current streaming implementation keeps the full Z extent per tile
to avoid axial boundary artefacts.

`--streaming auto` enables this path when the estimated full source array
exceeds `--streaming_threshold_gb`, or when a T subset is requested.  Use
`--projection mip`, `--projection sum`, or `--projection mean` to stream a
Z-projected output instead of the full stack.

---

## Docker Usage

### Building locally

```bash
docker build -t w_cideconvolve:<version> -t w_cideconvolve:latest .
```

On Windows you can use:

```powershell
.\builddocker.cmd
```

To also build the optional Bilayers Gradio or Jupyter images:

```powershell
.\builddocker_gradio.cmd
.\builddocker_jupyter.cmd
```

This creates the standard headless tags plus separate interface tags:

```text
w_cideconvolve:<version>
w_cideconvolve:latest
w_cideconvolve:<version>-gradio
w_cideconvolve:latest-gradio
w_cideconvolve:<version>-jupyter
w_cideconvolve:latest-jupyter
```

The unqualified tags contain PyTorch 2.13 with CUDA 13.2. `builddocker.cmd` also
builds temporary headless CUDA 13.0 fallback tags:

```text
w_cideconvolve:<version>-cu130
w_cideconvolve:latest-cu130
```

Validate the local GPU runtime and the compiled backend for all three solvers with:

```bash
docker run --rm --gpus all --entrypoint python \
    w_cideconvolve:<version> /app/cuda_smoke.py
```

The headless runtime is Python 3.11 with no Java, conda, compiler, or CUDA
toolkit. A multi-stage builder uses the selected CUDA toolkit to compile native
optimized kernels for SM 86, 89, 90, 100, and 120; only the resulting extension
is copied into the slim runtime. Users therefore need only a compatible NVIDIA
driver and container runtime. Interface images are intentionally separate and
larger because they add the Bilayers interface generator and the relevant UI
runtime.

**Prerequisites:**
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) for GPU pass-through
- Docker Desktop (Windows/macOS) or Docker Engine (Linux)

### Running with Docker

```bash
docker run --rm --gpus all \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    cellularimagingcf/w_cideconvolve \
    --infolder /data/in --outfolder /data/out \
    --method ci_rl --iterations 40
```

Omit `--gpus all` to force CPU-only execution.

By default, image metadata (NA, wavelengths, pixel sizes, microscope type, pinhole, refractive indices) is used where present; CLI values are fallbacks.  Pass `--overrule_image_metadata True` to force the CLI values.

### Bilayers Gradio image

The optional Gradio image follows the Bilayers convention of a separate interface-specific tag. It starts a web UI by default:

```bash
docker run --rm --gpus all -p 7878:7878 cellularimagingcf/w_cideconvolve:<version>-gradio
```

Open `http://localhost:7878`. Gradio uploads are copied into the container and outputs can be downloaded from the UI, so a volume mount is optional. For large data or persistent output storage, mount the same folders used by the headless workflow:

```bash
docker run --rm --gpus all -p 7878:7878 \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    cellularimagingcf/w_cideconvolve:<version>-gradio
```

### Bilayers Jupyter image

The optional Jupyter image follows the same Bilayers interface-tag convention. Jupyter needs a volume mount when you want notebooks, inputs, or outputs to persist outside the container:

```bash
docker run --rm --gpus all -p 8888:8888 \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    cellularimagingcf/w_cideconvolve:<version>-jupyter
```

Open `http://localhost:8888`. The generated notebook is available in the Jupyter file browser as `cideconvolve_bilayers.ipynb`.

### Bilayers config validation

The local helper validates `config.yaml` without requiring the full Bilayers toolchain:

```bash
python bilayers_cli.py validate
```

For upstream LinkML schema validation, install the optional validation dependencies and run strict mode:

```bash
pip install -r requirements_bilayers_validation.txt
python bilayers_cli.py validate --strict
```

### Docker benchmark mode

```bash
docker run --rm --gpus all \
    -v /path/to/input:/data/in \
    -v /path/to/output:/data/out \
    cellularimagingcf/w_cideconvolve \
    --infolder /data/in --outfolder /data/out \
    --benchmark True --bench_crop True --compute_metrics True
```

---

## BIOMERO — HPC / OMERO Workflow

[BIOMERO](https://github.com/NL-BioImaging/biomero) (BioImage Analysis in OMERO) lets you run FAIR bioimage-analysis workflows from an OMERO server on a SLURM-based HPC cluster.  CIDeconvolve is designed to plug directly into this framework.

### How it works

1. The OMERO admin configures the workflow in **`slurm-config.ini`** on the SLURM submission host:

   ```ini
   [SLURM]
   # ... global SLURM settings ...

   [W_CIDeconvolve]
   job_cpus=8
   job_memory=52G
   job_gres=gpu:2g.24gb
   ```

2. BIOMERO reads **`config.yaml`** to discover input parameters (method, iterations, device, PSF settings, benchmark options, etc.) and presents them in the OMERO web UI.

3. On submission, BIOMERO pulls the Singularity image from Docker Hub, transfers the selected images, and executes the workflow on the cluster.

4. Results (deconvolved images, benchmark montages, metrics CSV) are automatically uploaded back into OMERO.

Before selecting the CUDA 13.2 tag on a cluster, run the image through the same
Apptainer/Singularity path used by BIOMERO:

```bash
apptainer exec --nv docker://cellularimagingcf/w_cideconvolve:<version> \
    python /app/cuda_smoke.py
```

The output must report the expected PyTorch/CUDA versions, native `sm_90` on
H100/H200, a successful FFT round trip, and passing `ci_rl`, `ci_rl_tv`, and
`ci_sparse_hessian` checks. Use `<version>-cu130` if the cu132 image reports that
a newer driver or CUDA feature is required.

> For full setup instructions see the
> [BIOMERO documentation](https://nl-bioimaging.github.io/biomero/)
> and the [NL-BIOMERO deployment repo](https://github.com/NL-BioImaging/NL-BIOMERO).

---

## Launcher — Docker GUI (`launcher.py`)

![Launcher](docs/screenshots/launcher.png)

The Bilayers launcher provides a graphical interface that reads `config.yaml` at runtime, builds a matching parameter form, and generates / executes a `docker run` command — no command-line knowledge required.

```bash
python launcher.py
```

### Layout

1. **Header** — workflow name from `config.yaml`
2. **Data Folders** — input / output folder pickers with Browse… buttons
3. **Docker Runtime** — GPU toggle (`--gpus all`, enabled by default)
4. **Parameters** — two-column grid of all essential parameters with an expandable **Advanced** section for less-common settings
5. **Command Preview** — live-updated read-only console showing the exact `docker run` command that will be executed
6. **Buttons** — Restore Last Settings · Load Settings · Save Settings · **Run** · Close

### Widget types

| Bilayers type | Widget |
|---|---|
| Boolean | Pill toggle switch (grey / green) |
| String with choices | `QComboBox` |
| Float | `QDoubleSpinBox` |
| Integer | `QSpinBox` |
| Free text | `QLineEdit` |

### Settings persistence

Saved to `.last_launcher_settings.json` in the script directory (stores `values`, `folders`, `docker_options`).  **Restore Last Settings** reloads them on the next launch.

---

## Metadata Behaviour

When `--overrule_image_metadata false` (default), image metadata wins and CLI values are fallbacks.  When `true`, CLI values replace image metadata.

**OME-TIFF / OME-Zarr readers extract:** pixel size, objective NA, magnification, immersion RI from standard OME `ObjectiveSettings` or objective immersion, per-channel wavelengths, acquisition mode (widefield vs confocal), and confocal pinhole size.  Additionally parsed: benchmark-style `MapAnnotation` keys (`SampleRefractiveIndex`, `PinholeAiryUnits`).

Batch and streaming writers preserve metadata in the output:

- **OME-Zarr:** OME-NGFF 0.4 multiscales on Zarr v2, with physical pixel-size coordinate transforms; OMERO channel labels, colors, active state, and contrast windows; CIDeconvolve/source metadata in the root `cideconvolve` attribute; and `OME/METADATA.ome.xml` for Bio-Formats/QuPath readers that prefer OME-XML metadata. This layout targets QuPath 0.7 and OMERO compatibility.
- **OME-TIFF:** OME-XML physical pixel sizes, channel names, channel colors, and emission wavelengths where available; full CIDeconvolve/source metadata in private TIFF tag `65000`.

Confocal pinhole diameters in the metadata are converted to Airy disk units as:

```
AU = pinhole_µm / (1.22 × emission_µm × magnification / NA)
```

Use `--pinhole_airy 0` for the legacy point-detector confocal model.  Widefield PSFs ignore the pinhole parameter.

---

## Project Structure

| File | Purpose |
|------|---------|
| `gui_deconvolve_ci.py` | Standalone interactive deconvolution GUI |
| `ci_dual_viewer.py` | Synchronized dual-pane XYZT / 3D viewer widget |
| `launcher.py` | Docker launcher GUI backed by Bilayers `config.yaml` |
| `wrapper.py` | Bilayers / BIOMERO CLI entrypoint, benchmark runner, metrics |
| `cideconvolve_io/` | Shared OME-Zarr, OME-TIFF, metadata, and streaming I/O used by GUI, wrapper, Docker, and the focused CLI package |
| `deconvolve.py` | High-level pipeline: image loading, metadata extraction, PSF sizing, dispatch |
| `deconvolve_ci.py` | Core PyTorch engine: SHB-RL, RLTV, sparse-Hessian, PSF generation, tiling |
| `config.yaml` | Bilayers / BIOMERO parameter configuration |
| `bilayers_cli.py` | Bilayers CLI helper and wrapper argument parser |
| `Dockerfile` | Headless Docker build for BIOMERO and batch workflows |
| `Dockerfile.gradio` | Optional Bilayers Gradio Docker image built from the headless image |
| `Dockerfile.jupyter` | Optional Bilayers Jupyter Docker image built from the headless image |
| `requirements_gui.txt` | Python dependencies for GUI features |
| `requirements_docker.txt` | Python dependencies (Docker image) |
| `requirements_gradio.txt` | Extra dependencies for the optional Gradio image |
| `requirements_jupyter.txt` | Extra dependencies for the optional Jupyter image |
| `requirements_bilayers_validation.txt` | Optional strict LinkML validation dependencies |
| `version.txt` | Project version marker |

`core/ome_zarr_io.py`, `core/ome_tiff_io.py`, and `core/streaming.py` are compatibility shims that re-export the shared implementations from `cideconvolve_io`. Keep new image writer changes in `cideconvolve_io` so GUI, wrapper, Docker, and `ci_deconvolve_cli` stay in sync.

---

## References

- **SHB Acceleration:** Wang, Y. & Miller, E. L. (2014). "Scaled Heavy-Ball Acceleration of the Richardson-Lucy Algorithm for 3D Microscopy Image Restoration." *IEEE TIP* **23**(12), 5284–5297.
- **TV Regularisation:** Dey, N. et al. (2006). "Richardson-Lucy Algorithm With Total Variation Regularization for 3D Confocal Microscope Deconvolution." *Microsc. Res. Tech.* **69**(4), 260–266.
- **Content-Aware Image Restoration:** Weigert, M. et al. (2018). "Content-aware image restoration: pushing the limits of fluorescence microscopy." *Nat Methods* **15**, 1090–1097. [doi:10.1038/s41592-018-0216-7](https://doi.org/10.1038/s41592-018-0216-7)
- **BIOMERO:** Luik, T. T., Rosas-Bertolini, R., Reits, E. A. J., Hoebe, R. A. & Krawczyk, P. M. (2024). "BIOMERO: A scalable and extensible image analysis framework." *Patterns* **5**(8), 101024. [doi:10.1016/j.patter.2024.101024](https://doi.org/10.1016/j.patter.2024.101024) · [GitHub](https://github.com/NL-BioImaging/biomero) · [Documentation](https://nl-bioimaging.github.io/biomero/)
- **Gibson-Lanni model:** Gibson, S. F. & Lanni, F. (1992). [doi:10.1364/JOSAA.9.000154](https://doi.org/10.1364/JOSAA.9.000154)
- **PSF Generator:** Kirshner, H. et al. — [EPFL PSF Generator](https://bigwww.epfl.ch/algorithms/psfgenerator/)
- **OMERO:** Allan, C. et al. (2012). "OMERO: flexible, model-driven data management for experimental biology." *Nat Methods* **9**, 245–253. [doi:10.1038/nmeth.1896](https://doi.org/10.1038/nmeth.1896)

---

## Further Reading

- [DECONVOLVE_CI.MD](docs/DECONVOLVE_CI.MD) — full algorithmic documentation: SHB momentum derivation, TV and sparse-Hessian formulations, PSF model details, tiling strategy, and convergence criteria.
- [metrics.md](docs/metrics.md) — benchmark metric formulas and interpretation: timing CSV columns, FFT detail energy, edge strength, signal sparsity, and robust range.

---

## License

MIT — see [LICENSE](LICENSE).
