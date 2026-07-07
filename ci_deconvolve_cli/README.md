# ci-deconvolve

Focused command-line CI-RL deconvolution for OME-TIFF and OME-Zarr images.

This package installs the `ci_deconvolve` command. On Windows, `pip` also
creates a `ci_deconvolve.exe` console launcher in the active environment.

## Prerequisite: PyTorch

PyTorch is required but is not bundled in this package. Install the CPU or CUDA
build that matches your system before running `ci_deconvolve`.

Examples:

```bash
pip install torch
```

or install a CUDA build using the command from:

```text
https://pytorch.org/get-started/locally/
```

## Install

From this folder during development:

```bash
pip install -e .
```

From PyPI after publishing:

```bash
pip install ci-deconvolve
```

The package depends on `ome-zarr` for OME-Zarr reading and `zarr>=2.16,<4` for
writing QuPath- and OMERO-compatible OME-Zarr output.

## Usage

Run one OME-TIFF file:

```bash
ci_deconvolve --input image.ome.tiff --output out
```

Run one OME-Zarr image:

```bash
ci_deconvolve --input image.ome.zarr --output out --output-format ome-zarr
```

Run every supported immediate child in a folder:

```bash
ci_deconvolve --input in_folder --output out_folder --iterations 40
```

Only these inputs are accepted:

- `.ome.tif`
- `.ome.tiff`
- `.zarr` / `.ome.zarr` directories

The CLI always runs `ci_rl`. It intentionally has no `--method` option.

## Examples

```bash
ci_deconvolve --input image.ome.tiff --output out ^
  --iterations 40 ^
  --device auto ^
  --output-format ome-tiff ^
  --projection none ^
  --emission-wl 520 ^
  --excitation-wl 488 ^
  --pixel-size-xy 0.065 ^
  --pixel-size-z 0.2
```

Use `--overrule-metadata` when CLI metadata values should replace image
metadata. Without it, CLI metadata values are used as fallbacks.

Per-channel values can be comma-separated. If there are more channels than
values, the last value is reused for the remaining channels.

```bash
ci_deconvolve --input image.ome.tiff --output out ^
  --iterations 40,60,80 ^
  --emission-wl 520,610 ^
  --excitation-wl 488,561 ^
  --pinhole-airy 1.0,1.2
```

## Parameters

### Required Paths

| Option | Default | Description |
| --- | --- | --- |
| `input_path` | none | Optional positional input path. Use this or `--input`. |
| `--input PATH` | none | Input OME-TIFF file, OME-Zarr folder, or folder containing supported inputs. |
| `--output DIR` | required | Output folder. Created if it does not exist. |

Supported inputs are immediate files/folders only:

- `.ome.tif`
- `.ome.tiff`
- `.zarr` / `.ome.zarr` directories

HCS plate OME-Zarr inputs are detected and rejected clearly; this focused CLI
does not process plate fields.

### Output And Compute

| Option | Default | Description |
| --- | --- | --- |
| `--output-format ome-tiff\|ome-zarr` | `ome-tiff` | Writes `<stem>_decon.ome.tiff` or `<stem>_decon.ome.zarr`. |
| `--projection none\|max-z` | `none` | With `max-z`, writes a maximum-Z projection when the deconvolved result is 3D. 2D inputs are unchanged. |
| `--iterations N[,N...]` | `40` | CI-RL iteration count. A comma- or semicolon-separated list applies per channel, with the last value reused for extra channels. |
| `--device auto\|cpu\|cuda` | `auto` | Compute device. `auto` lets PyTorch choose CUDA when available, otherwise CPU. |
| `-v`, `--verbose` | off | Enable INFO logging from the CLI and core deconvolution code. |

### CI-RL Solver

| Option | Default | Description |
| --- | --- | --- |
| `--background VALUE\|auto` | `auto` | Background level used by the solver. `auto` estimates it from the image. Numeric values are accepted. |
| `--offset VALUE\|auto\|none` | `auto` | Positive offset added before iteration and removed afterwards. Use `none`, `0`, or `0.0` to disable. |
| `--damping VALUE\|auto\|none` | `none` | Noise-gated damping strength. Use `auto` for automatic damping, numeric values for manual damping, or `none`/`0` to disable. |
| `--prefilter-sigma VALUE` | `0.0` | Optional Anscombe/low-pass prefilter sigma. `0.0` disables prefiltering. |
| `--start MODE` | `auto` | Initial estimate mode. Choices: `auto`, `flat`, `percentile_flat`, `observed`, `observed_bgsub`, `lowpass`, `lowpass_bgsub`, `hybrid`. |
| `--convergence auto\|fixed\|none` | `auto` | `auto` enables early stopping based on convergence. `fixed` and `none` both run the requested iteration count. |
| `--rel-threshold VALUE` | `0.005` | Relative convergence threshold used when `--convergence auto`. Values are clamped to at least `1e-8`. |
| `--check-every N` | `5` | Check convergence every N iterations. Values below 1 are clamped to 1. |

### Metadata And PSF

These values are used to generate the point spread function. By default, image
metadata wins and CLI values fill in missing metadata. With
`--overrule-metadata`, CLI values replace image metadata.

| Option | Default | Description |
| --- | --- | --- |
| `--na VALUE` | `1.4` | Numerical aperture. |
| `--refractive-index VALUE` | `1.515` | Immersion medium refractive index, typically oil. |
| `--sample-ri VALUE` | `1.47` | Sample medium refractive index. |
| `--microscope-type widefield\|confocal` | `confocal` | Microscope model used for PSF generation. |
| `--emission-wl VALUE[,VALUE...]` | `520` | Emission wavelength(s), in nm. Per-channel list supported. |
| `--excitation-wl VALUE[,VALUE...]` | `488` | Excitation wavelength(s), in nm. Per-channel list supported. |
| `--pinhole-airy VALUE[,VALUE...]` | `1.0` | Confocal pinhole diameter in Airy units. Per-channel list supported. |
| `--pixel-size-xy VALUE` | `0.065` | Lateral pixel size in micrometers. Applied to X and Y. |
| `--pixel-size-z VALUE` | `0.2` | Axial pixel size in micrometers. |
| `--overrule-metadata` | off | Replace image metadata with CLI metadata values instead of only using CLI values as fallbacks. |

List syntax:

```bash
--emission-wl 520,610
--excitation-wl 488;561
--iterations 30,40,50
```

Commas and semicolons are both accepted.

### OME-Zarr Compatibility

OME-Zarr output is written as OME-NGFF 0.4 on Zarr v2, with `.zgroup` and
`.zattrs` metadata, OMERO channel display metadata, and
`OME/METADATA.ome.xml` for Bio-Formats/QuPath readers that prefer OME-XML
metadata. This layout is intended to open in QuPath 0.7 and remain compatible
with OMERO tooling.

### 2D Widefield Options

These only affect 2D widefield data when `--microscope-type widefield` and
`--two-d-mode auto` are used.

| Option | Default | Description |
| --- | --- | --- |
| `--two-d-mode auto\|legacy_2d` | `auto` | `auto` enables the enhanced 2D widefield path. `legacy_2d` uses the historical pure-2D PSF behavior. |
| `--two-d-wf-aggressiveness conservative\|balanced\|strong` | `balanced` | Strength preset for the enhanced 2D widefield model. |
| `--two-d-wf-bg-radius-um VALUE` | `0.5` | Background-estimation radius in micrometers. Values below `0.1` are clamped to `0.1`. |
| `--two-d-wf-bg-scale VALUE` | `1.0` | Multiplier for the estimated 2D widefield background. Values below `0.1` are clamped to `0.1`. |

## Metadata Override Behavior

Without `--overrule-metadata`:

- Existing OME metadata is preserved.
- CLI metadata values are used only when a value is missing or invalid.
- This is the recommended mode for well-annotated OME-TIFF or OME-Zarr data.

With `--overrule-metadata`:

- CLI metadata values replace image metadata.
- Use this when source metadata is known to be wrong, incomplete, or converted
  with incorrect physical pixel sizes or wavelengths.

Example:

```bash
ci_deconvolve --input image.ome.tiff --output out ^
  --overrule-metadata ^
  --microscope-type widefield ^
  --na 1.40 ^
  --refractive-index 1.515 ^
  --sample-ri 1.47 ^
  --pixel-size-xy 0.065 ^
  --pixel-size-z 0.200 ^
  --emission-wl 520,610
```
