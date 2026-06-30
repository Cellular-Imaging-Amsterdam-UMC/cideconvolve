from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger("ci_deconvolve")

OME_TIFF_SUFFIXES = (".ome.tif", ".ome.tiff")
ZARR_SUFFIXES = (".zarr", ".ome.zarr")


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required but is not installed or cannot be imported. "
            "Install a CPU or CUDA PyTorch build first, then rerun ci_deconvolve. "
            "See https://pytorch.org/get-started/locally/ for the current install command."
        ) from exc


def _is_ome_tiff(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(suffix) for suffix in OME_TIFF_SUFFIXES)


def _is_ome_zarr(path: Path) -> bool:
    name = path.name.lower()
    return path.is_dir() and any(name.endswith(suffix) for suffix in ZARR_SUFFIXES)


def _discover_inputs(path: Path) -> list[Path]:
    if _is_ome_tiff(path) or _is_ome_zarr(path):
        return [path]
    if path.is_dir():
        inputs = [
            child
            for child in sorted(path.iterdir(), key=lambda p: p.name.lower())
            if _is_ome_tiff(child) or _is_ome_zarr(child)
        ]
        if inputs:
            return inputs
        raise ValueError(f"No OME-TIFF files or OME-Zarr folders found in {path}")
    raise ValueError(
        f"Unsupported input {path}. Expected .ome.tif/.ome.tiff, .zarr/.ome.zarr, "
        "or a folder containing those inputs."
    )


def _stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr", ".zarr"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for item in str(raw or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(max(1, int(float(item))))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid iteration count: {item!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one iteration count is required")
    return values


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    values = []
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values or None


def _parse_float_or_auto(raw: str):
    text = str(raw).strip().lower()
    if text == "auto":
        return "auto"
    if text in {"none", "0", "0.0"}:
        return 0.0
    return float(text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ci_deconvolve",
        description="Run CI-RL deconvolution on OME-TIFF and OME-Zarr inputs.",
    )
    parser.add_argument("input_path", nargs="?", help="Input file/folder. Alias for --input.")
    parser.add_argument("--input", dest="input_option", help="OME-TIFF, OME-Zarr, or folder.")
    parser.add_argument("--output", required=True, help="Output folder.")
    parser.add_argument(
        "--output-format",
        choices=("ome-tiff", "ome-zarr"),
        default="ome-tiff",
        help="Output format. Default: ome-tiff.",
    )
    parser.add_argument("--iterations", type=_parse_int_list, default=[40])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--background", default="auto")
    parser.add_argument("--offset", default="auto")
    parser.add_argument("--damping", default="none")
    parser.add_argument("--prefilter-sigma", type=float, default=0.0)
    parser.add_argument(
        "--start",
        choices=(
            "auto",
            "flat",
            "percentile_flat",
            "observed",
            "observed_bgsub",
            "lowpass",
            "lowpass_bgsub",
            "hybrid",
        ),
        default="auto",
    )
    parser.add_argument("--convergence", choices=("auto", "fixed", "none"), default="auto")
    parser.add_argument("--rel-threshold", type=float, default=0.005)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--na", type=float, default=1.4)
    parser.add_argument("--refractive-index", type=float, default=1.515)
    parser.add_argument("--sample-ri", type=float, default=1.47)
    parser.add_argument("--microscope-type", choices=("widefield", "confocal"), default="confocal")
    parser.add_argument("--emission-wl", default="520")
    parser.add_argument("--excitation-wl", default="488")
    parser.add_argument("--pinhole-airy", default="1.0")
    parser.add_argument("--pixel-size-xy", type=float, default=0.065, help="Micrometers.")
    parser.add_argument("--pixel-size-z", type=float, default=0.2, help="Micrometers.")
    parser.add_argument(
        "--overrule-metadata",
        action="store_true",
        help="Use CLI metadata values even when image metadata is present.",
    )
    parser.add_argument(
        "--two-d-mode",
        choices=("auto", "legacy_2d"),
        default="auto",
    )
    parser.add_argument(
        "--two-d-wf-aggressiveness",
        choices=("conservative", "balanced", "strong"),
        default="balanced",
    )
    parser.add_argument("--two-d-wf-bg-radius-um", type=float, default=0.5)
    parser.add_argument("--two-d-wf-bg-scale", type=float, default=1.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _normalise_convergence(value: str) -> str:
    return "fixed" if value == "none" else value


def _output_path(input_path: Path, output_dir: Path, output_format: str) -> Path:
    if output_format == "ome-zarr":
        return output_dir / f"{_stem(input_path)}_decon.ome.zarr"
    return output_dir / f"{_stem(input_path)}_decon.ome.tiff"


def _run_one(input_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    from core.deconvolve import deconvolve_image, save_result
    from core.ome_zarr_io import is_hcs_plate, save_result_ome_zarr

    if _is_ome_zarr(input_path) and is_hcs_plate(input_path):
        if args.output_format == "ome-tiff":
            raise ValueError("HCS plate OME-Zarr input cannot be written as one OME-TIFF output.")
        raise ValueError("HCS plate OME-Zarr output is not supported by the focused CLI yet.")

    device = None if args.device == "auto" else args.device
    result = deconvolve_image(
        input_path,
        method="ci_rl",
        niter=args.iterations,
        na=args.na,
        refractive_index=args.refractive_index,
        sample_refractive_index=args.sample_ri,
        microscope_type=args.microscope_type,
        emission_wavelengths=_parse_float_list(args.emission_wl),
        excitation_wavelengths=_parse_float_list(args.excitation_wl),
        pinhole_airy_units=_parse_float_list(args.pinhole_airy),
        overrule_metadata=bool(args.overrule_metadata),
        pixel_size_xy=args.pixel_size_xy,
        pixel_size_z=args.pixel_size_z,
        background=_parse_float_or_auto(args.background),
        damping=_parse_float_or_auto(args.damping),
        offset=_parse_float_or_auto(args.offset),
        prefilter_sigma=max(0.0, float(args.prefilter_sigma)),
        start=args.start,
        convergence=_normalise_convergence(args.convergence),
        rel_threshold=max(1e-8, float(args.rel_threshold)),
        check_every=max(1, int(args.check_every)),
        two_d_mode=args.two_d_mode,
        two_d_wf_aggressiveness=args.two_d_wf_aggressiveness,
        two_d_wf_bg_radius_um=max(0.1, float(args.two_d_wf_bg_radius_um)),
        two_d_wf_bg_scale=max(0.1, float(args.two_d_wf_bg_scale)),
        device=device,
    )

    out_path = _output_path(input_path, output_dir, args.output_format)
    if args.output_format == "ome-zarr":
        return save_result_ome_zarr(result, out_path)

    tmp_dir = output_dir / ".ci_deconvolve_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / out_path.name
    try:
        save_result(result, tmp_path)
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(tmp_path), str(out_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path


def run(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_value = args.input_option or args.input_path
    if not input_value:
        parser.error("an input path is required via --input or positional input_path")
    input_path = Path(input_value)
    output_dir = Path(args.output)

    _require_torch()
    inputs = _discover_inputs(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("ci_deconvolve")
    print(f"  method       : ci_rl")
    print(f"  input count  : {len(inputs)}")
    print(f"  output       : {output_dir}")
    print(f"  output format: {args.output_format}")
    print(f"  iterations   : {', '.join(str(v) for v in args.iterations)}")

    failures: list[tuple[Path, Exception]] = []
    for index, item in enumerate(inputs, start=1):
        print(f"\n[{index}/{len(inputs)}] {item}")
        start = time.time()
        try:
            out_path = _run_one(item, output_dir, args)
            print(f"  saved: {out_path}")
            print(f"  time : {time.time() - start:.1f}s")
        except Exception as exc:
            failures.append((item, exc))
            print(f"  ERROR: {exc}", file=sys.stderr)

    if failures:
        print(f"\nFailed inputs: {len(failures)}", file=sys.stderr)
        for item, exc in failures:
            print(f"  {item}: {exc}", file=sys.stderr)
        return 1
    print("\nci_deconvolve complete.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    try:
        return run(argv)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ci_deconvolve failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
