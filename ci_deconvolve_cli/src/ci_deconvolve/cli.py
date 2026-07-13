from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Iterable

from ci_deconvolve import __version__

LOGGER = logging.getLogger("ci_deconvolve")
_CANCEL_REQUESTED = False

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


def _parse_snr(raw: str):
    text = str(raw or "off").strip().lower()
    if text in {"off", "none", ""}:
        return None
    if text == "auto":
        return "auto"
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("SNR must be off, auto, or a positive number")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ci_deconvolve",
        description="Run CI-RL deconvolution on OME-TIFF and OME-Zarr inputs.",
    )
    parser.add_argument("input_path", nargs="?", help="Input file/folder. Alias for --input.")
    parser.add_argument("--input", dest="input_option", help="OME-TIFF, OME-Zarr, or folder.")
    parser.add_argument("--output", help="Output folder.")
    parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Report Python/PyTorch/Zarr/CIDeconvolve environment details and exit.",
    )
    parser.add_argument(
        "--output-format",
        choices=("ome-tiff", "ome-zarr"),
        default="ome-tiff",
        help="Output format. Default: ome-tiff.",
    )
    parser.add_argument(
        "--output-dtype",
        "--output_dtype",
        choices=("float32", "uint16"),
        default="float32",
        help="Output pixel type. uint16 uses global scaling to avoid clipping high values.",
    )
    parser.add_argument(
        "--projection",
        choices=("none", "max-z"),
        default="none",
        help="Write a max-Z projection instead of the full 3D stack when input data is 3D.",
    )
    parser.add_argument(
        "--ome-zarr-pyramid",
        choices=("auto", "on", "off"),
        default="auto",
        help="Write XY pyramid levels for OME-Zarr output. Default: auto.",
    )
    parser.add_argument("--t-start", "--t_start", type=int, default=1, help="First T frame to save, 1-based inclusive.")
    parser.add_argument("--t-stop", "--t_stop", type=int, default=0, help="Last T frame to save, 1-based inclusive. Use 0 for the final frame.")
    parser.add_argument("--t-step", "--t_step", type=int, default=1, help="Save every Nth T frame in the selected range.")
    parser.add_argument("--iterations", type=_parse_int_list, default=[40])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--background", default="auto")
    parser.add_argument("--offset", default="auto")
    parser.add_argument("--prefilter-sigma", type=float, default=0.0)
    parser.add_argument("--snr", default="off", help="Noise-aware setup: off, auto, or a positive SNR.")
    parser.add_argument("--acuity", type=float, default=0.0, help="Sharpness balance from -100 to +100.")
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
    parser.add_argument("--save-qc-mips", dest="save_qc_mips", action="store_true", default=True)
    parser.add_argument("--no-qc-mips", dest="save_qc_mips", action="store_false")
    parser.add_argument("--write-manifest", dest="write_manifest", action="store_true", default=True)
    parser.add_argument("--no-manifest", dest="write_manifest", action="store_false")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _normalise_convergence(value: str) -> str:
    return "fixed" if value == "none" else value


def _output_path(input_path: Path, output_dir: Path, output_format: str) -> Path:
    if output_format == "ome-zarr":
        return output_dir / f"{_stem(input_path)}_decon.ome.zarr"
    return output_dir / f"{_stem(input_path)}_decon.ome.tiff"


def _apply_projection(result: dict, projection: str) -> dict:
    if projection != "max-z":
        return result
    channels = list(result.get("channels") or [])
    if not channels:
        return result
    if channels[0].ndim == 4 and channels[0].shape[1] > 1:
        import numpy as np

        projected = dict(result)
        projected["channels"] = [
            np.max(channel, axis=1).astype(np.float32, copy=False)
            for channel in channels
        ]
        if result.get("source_channels"):
            projected["source_channels"] = [
                np.max(channel, axis=1).astype(np.float32, copy=False)
                if getattr(channel, "ndim", 0) == 4 and channel.shape[1] > 1
                else channel
                for channel in result["source_channels"]
            ]
        metadata = dict(result.get("metadata") or {})
        metadata["projection"] = {"axis": "z", "method": "max"}
        metadata["size_z"] = 1
        projected["metadata"] = metadata
        return projected
    if channels[0].ndim != 3 or channels[0].shape[0] <= 1:
        return result

    import numpy as np

    projected = dict(result)
    projected["channels"] = [np.max(channel, axis=0).astype(np.float32, copy=False) for channel in channels]
    if result.get("source_channels"):
        projected["source_channels"] = [
            np.max(channel, axis=0).astype(np.float32, copy=False)
            if getattr(channel, "ndim", 0) == 3 and channel.shape[0] > 1
            else channel
            for channel in result["source_channels"]
        ]
    metadata = dict(result.get("metadata") or {})
    metadata["projection"] = {"axis": "z", "method": "max"}
    metadata["size_z"] = 1
    projected["metadata"] = metadata
    return projected


def _apply_cli_metadata_to_source(metadata: dict, args: argparse.Namespace, size_c: int) -> dict:
    meta = dict(metadata or {})

    def _use(key: str, value) -> None:
        if value is not None and (bool(args.overrule_metadata) or meta.get(key) in (None, "")):
            meta[key] = value

    _use("na", args.na)
    _use("refractive_index", args.refractive_index)
    _use("sample_refractive_index", args.sample_ri)
    _use("microscope_type", args.microscope_type)
    _use("pixel_size_x", args.pixel_size_xy)
    _use("pixel_size_y", args.pixel_size_xy)
    _use("pixel_size_z", args.pixel_size_z)

    channels = [dict(ch) if isinstance(ch, dict) else {} for ch in meta.get("channels", [])]
    if len(channels) < size_c:
        channels.extend({} for _ in range(size_c - len(channels)))

    def _list_value(raw: str | None) -> list[float] | None:
        return _parse_float_list(raw)

    def _value(values: list[float] | None, index: int):
        if not values:
            return None
        return values[index] if index < len(values) else values[-1]

    emissions = _list_value(args.emission_wl)
    excitations = _list_value(args.excitation_wl)
    pinholes = _list_value(args.pinhole_airy)
    for index, channel in enumerate(channels[:size_c]):
        for key, values in (
            ("emission_wavelength", emissions),
            ("excitation_wavelength", excitations),
            ("pinhole_airy_units", pinholes),
        ):
            value = _value(values, index)
            if value is not None and (bool(args.overrule_metadata) or channel.get(key) in (None, "")):
                channel[key] = value
    meta["channels"] = channels[:size_c]
    names = list(meta.get("channel_names") or [])
    names.extend(f"Ch{i}" for i in range(len(names), size_c))
    meta["channel_names"] = names[:size_c]
    return meta


def _run_streaming_one(input_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    import numpy as np

    from core.deconvolve import deconvolve, generate_psf
    from core.streaming import (
        ProjectionPyramidSink,
        TiledOmeTiffSink,
        TimepointSubsetRegionSource,
        ZarrPyramidSink,
        deconvolve_streaming,
        normalise_timepoint_indices,
        open_region_source,
        save_streaming_provenance,
        suggest_streaming_tile_size,
    )

    source = open_region_source(input_path)
    selected = normalise_timepoint_indices(
        source.shape[0],
        start=max(int(args.t_start), 1),
        stop=(None if int(args.t_stop) <= 0 else int(args.t_stop)),
        step=max(int(args.t_step), 1),
        one_based=True,
    )
    source.metadata = _apply_cli_metadata_to_source(source.metadata, args, source.shape[1])
    if selected != list(range(source.shape[0])):
        source = TimepointSubsetRegionSource(source, selected)

    out_path = _output_path(input_path, output_dir, args.output_format)
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()

    project_output = args.projection == "max-z" and source.shape[2] > 1
    sink_shape = (source.shape[0], source.shape[1], 1 if project_output else source.shape[2], source.shape[3], source.shape[4])
    sink_metadata = dict(source.metadata)
    if project_output:
        sink_metadata["size_z"] = 1
        sink_metadata["default_z"] = 0
        sink_metadata["projection"] = {"axis": "z", "method": "max"}

    if args.output_format == "ome-zarr":
        base_sink = ZarrPyramidSink(
            out_path,
            shape=sink_shape,
            metadata=sink_metadata,
            resume=False,
            output_dtype=args.output_dtype,
        )
    else:
        base_sink = TiledOmeTiffSink(
            out_path,
            shape=sink_shape,
            metadata=sink_metadata,
            tile_yx=(512, 512),
            levels=None,
            predictor=str(args.output_dtype or "float32").strip().lower() == "uint16",
            output_dtype=args.output_dtype,
            write_private_metadata=False,
        )
    sink = ProjectionPyramidSink(base_sink, source_shape=source.shape, mode="mip") if project_output else base_sink

    psf_cache: dict[int, np.ndarray] = {}
    frozen_snr: dict[int, float] = {}

    def _psf_for_channel(channel: int) -> np.ndarray:
        if channel not in psf_cache:
            psf_cache[channel] = generate_psf(source.metadata, channel_idx=channel)
        return psf_cache[channel]

    def _deconvolve_tile(tile_img: np.ndarray, psf: np.ndarray, channel: int) -> np.ndarray:
        effective_psf = psf
        if tile_img.ndim == 2 and effective_psf.ndim == 3:
            effective_psf = effective_psf[effective_psf.shape[0] // 2]
        elif tile_img.ndim == 3 and effective_psf.ndim == 2:
            effective_psf = effective_psf[np.newaxis, :, :]
        niter = args.iterations[channel] if channel < len(args.iterations) else args.iterations[-1]
        requested_snr = _parse_snr(args.snr)
        if requested_snr == "auto":
            if channel not in frozen_snr:
                from core.deconvolve_ci import estimate_image_snr
                frozen_snr[channel] = float(estimate_image_snr(tile_img)["snr"])
            requested_snr = frozen_snr[channel]
        return deconvolve(
            tile_img,
            effective_psf,
            method="ci_rl",
            niter=niter,
            background=_parse_float_or_auto(args.background),
            offset=_parse_float_or_auto(args.offset),
            prefilter_sigma=max(0.0, float(args.prefilter_sigma)),
            snr=requested_snr,
            acuity=max(-100.0, min(100.0, float(args.acuity))),
            start=args.start,
            convergence=_normalise_convergence(args.convergence),
            rel_threshold=max(1e-8, float(args.rel_threshold)),
            check_every=max(1, int(args.check_every)),
            device=None if args.device == "auto" else args.device,
            pixel_size_xy=source.metadata.get("pixel_size_x"),
            pixel_size_z=source.metadata.get("pixel_size_z"),
            microscope_type=source.metadata.get("microscope_type", "widefield"),
            two_d_mode=args.two_d_mode,
            two_d_wf_aggressiveness=args.two_d_wf_aggressiveness,
            two_d_wf_bg_radius_um=max(0.1, float(args.two_d_wf_bg_radius_um)),
            two_d_wf_bg_scale=max(0.1, float(args.two_d_wf_bg_scale)),
        )

    tile_xy = suggest_streaming_tile_size(source.shape, method="ci_rl", device=None if args.device == "auto" else args.device)
    summary = deconvolve_streaming(
        source,
        sink,
        psf_for_channel=_psf_for_channel,
        deconvolve_tile=_deconvolve_tile,
        tile_yx=(tile_xy, tile_xy),
        resume=False,
        build_pyramids=True,
    )
    save_streaming_provenance(
        out_path.with_suffix(out_path.suffix + ".provenance.json"),
        source=source,
        sink=sink,
        params={
            "projection": args.projection,
            "timepoints_zero_based": selected,
            "timepoints_one_based": [t + 1 for t in selected],
            "tile_xy": tile_xy,
            "output_dtype": args.output_dtype,
            "snr": args.snr,
            "acuity": args.acuity,
        },
        summary=summary,
    )
    return out_path


def _format_metadata_value(value) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _metadata_source(metadata: dict, key: str) -> str:
    provenance = metadata.get("_metadata_provenance") or {}
    fields = provenance.get("fields") if isinstance(provenance, dict) else {}
    return str((fields or {}).get(key) or "unknown")


def _channel_metadata_source(metadata: dict, index: int, key: str) -> str:
    provenance = metadata.get("_metadata_provenance") or {}
    channels = provenance.get("channels") if isinstance(provenance, dict) else []
    if isinstance(channels, list) and index < len(channels) and isinstance(channels[index], dict):
        return str(channels[index].get(key) or "unknown")
    return "unknown"


def _print_metadata_report(metadata: dict) -> None:
    fields = [
        ("pixel_size_x", "pixel size X", "um"),
        ("pixel_size_y", "pixel size Y", "um"),
        ("pixel_size_z", "pixel size Z", "um"),
        ("na", "NA", ""),
        ("refractive_index", "immersion RI", ""),
        ("sample_refractive_index", "sample RI", ""),
        ("microscope_type", "microscope", ""),
    ]
    print("  metadata:")
    for key, label, unit in fields:
        value = _format_metadata_value(metadata.get(key))
        suffix = f" {unit}" if unit else ""
        print(f"    {label:14}: {value}{suffix} ({_metadata_source(metadata, key)})")

    defaulted = sorted(str(key) for key in metadata.get("_defaulted_keys") or [])
    inferred = sorted(str(key) for key in metadata.get("_inferred_keys") or [])
    warnings = [str(item) for item in metadata.get("metadata_warnings") or [] if str(item).strip()]
    if defaulted:
        print("    defaults      : " + ", ".join(defaulted))
    if inferred:
        print("    inferred      : " + ", ".join(inferred))
    if warnings:
        print("    warnings      : " + " | ".join(warnings))

    channels = list(metadata.get("channels") or [])
    for index, channel in enumerate(channels):
        name = channel.get("name") or channel.get("label")
        if not name and index < len(metadata.get("channel_names") or []):
            name = metadata["channel_names"][index]
        print(f"    channel {index}{f' ({name})' if name else ''}:")
        for key, label, unit in (
            ("emission_wavelength", "emission", "nm"),
            ("excitation_wavelength", "excitation", "nm"),
            ("pinhole_airy_units", "pinhole", "AU"),
        ):
            value = _format_metadata_value(channel.get(key))
            print(
                f"      {label:10}: {value} {unit} "
                f"({_channel_metadata_source(metadata, index, key)})"
            )


def _run_one(input_path: Path, output_dir: Path, args: argparse.Namespace) -> Path:
    from core.deconvolve import deconvolve_image, save_result
    from core.ome_zarr_io import is_hcs_plate, save_result_ome_zarr
    from core.streaming import normalise_timepoint_indices, open_region_source

    if _is_ome_zarr(input_path) and is_hcs_plate(input_path):
        if args.output_format == "ome-tiff":
            raise ValueError("HCS plate OME-Zarr input cannot be written as one OME-TIFF output.")
        raise ValueError("HCS plate OME-Zarr output is not supported by the focused CLI yet.")

    try:
        probe = open_region_source(input_path)
        selected_t = normalise_timepoint_indices(
            probe.shape[0],
            start=max(int(args.t_start), 1),
            stop=(None if int(args.t_stop) <= 0 else int(args.t_stop)),
            step=max(int(args.t_step), 1),
            one_based=True,
        )
        if probe.shape[0] > 1 or selected_t != [0]:
            return _run_streaming_one(input_path, output_dir, args)
    except Exception as exc:
        if int(args.t_start) != 1 or int(args.t_stop) > 0 or int(args.t_step) != 1:
            raise
        LOGGER.info("Streaming probe unavailable; using eager path: %s", exc)

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
        offset=_parse_float_or_auto(args.offset),
        prefilter_sigma=max(0.0, float(args.prefilter_sigma)),
        snr=_parse_snr(args.snr),
        acuity=max(-100.0, min(100.0, float(args.acuity))),
        start=args.start,
        convergence=_normalise_convergence(args.convergence),
        rel_threshold=max(1e-8, float(args.rel_threshold)),
        check_every=max(1, int(args.check_every)),
        two_d_mode=args.two_d_mode,
        two_d_wf_aggressiveness=args.two_d_wf_aggressiveness,
        two_d_wf_bg_radius_um=max(0.1, float(args.two_d_wf_bg_radius_um)),
        two_d_wf_bg_scale=max(0.1, float(args.two_d_wf_bg_scale)),
        device=device,
        cancel_checker=lambda: _CANCEL_REQUESTED,
    )
    result = _apply_projection(result, args.projection)
    _print_metadata_report(result.get("metadata") or {})

    out_path = _output_path(input_path, output_dir, args.output_format)
    if args.output_format == "ome-zarr":
        return save_result_ome_zarr(result, out_path, pyramid=args.ome_zarr_pyramid, output_dtype=args.output_dtype)

    tmp_dir = output_dir / ".ci_deconvolve_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / out_path.name
    try:
        save_result(result, tmp_path, save_qc_mips=bool(args.save_qc_mips), output_dtype=args.output_dtype)
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(tmp_path), str(out_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path


def _validate_environment() -> int:
    print("CIDECONVOLVE_OK")
    print(f"ci_deconvolve={__version__}")
    print(f"python={sys.version.split()[0]}")
    try:
        import torch
        print(f"torch={getattr(torch, '__version__', 'unknown')}")
        print(f"cuda_available={bool(torch.cuda.is_available())}")
        print(f"cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    except Exception as exc:
        print(f"torch_error={type(exc).__name__}: {exc}")
        return 1
    try:
        import zarr
        print(f"zarr={getattr(zarr, '__version__', 'unknown')}")
        print("ome_zarr_v2_write=ok")
    except Exception as exc:
        print(f"zarr_error={type(exc).__name__}: {exc}")
        return 1
    return 0


def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        global _CANCEL_REQUESTED
        _CANCEL_REQUESTED = True
        print(f"Cancellation requested by signal {signum}.", file=sys.stderr)

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except Exception:
                pass


def _write_manifest(output_dir: Path, records: list[dict], args: argparse.Namespace) -> None:
    manifest = {
        "ci_deconvolve_version": __version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_format": args.output_format,
        "projection": args.projection,
        "t_start": getattr(args, "t_start", 1),
        "t_stop": getattr(args, "t_stop", 0),
        "t_step": getattr(args, "t_step", 1),
        "iterations": args.iterations,
        "snr": getattr(args, "snr", "off"),
        "acuity": getattr(args, "acuity", 0.0),
        "records": records,
    }
    (output_dir / "ci_deconvolve_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )


def run(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.validate_env:
        return _validate_environment()
    if not args.output:
        parser.error("--output is required unless --validate-env is used")
    _install_signal_handlers()

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
    print(f"  projection   : {args.projection}")
    print(f"  T range      : start={args.t_start}, stop={'last' if int(args.t_stop) <= 0 else args.t_stop}, step={args.t_step}")
    print(f"  iterations   : {', '.join(str(v) for v in args.iterations)}")

    failures: list[tuple[Path, Exception]] = []
    records: list[dict] = []
    for index, item in enumerate(inputs, start=1):
        if _CANCEL_REQUESTED:
            print("\nCancelled before remaining inputs.", file=sys.stderr)
            return 130
        print(f"\n[{index}/{len(inputs)}] {item}")
        start = time.time()
        try:
            out_path = _run_one(item, output_dir, args)
            print(f"  saved: {out_path}")
            elapsed = time.time() - start
            print(f"  time : {elapsed:.1f}s")
            records.append({
                "input": str(item),
                "output": str(out_path),
                "status": "success",
                "seconds": elapsed,
            })
        except Exception as exc:
            if _CANCEL_REQUESTED or str(exc) == "Cancelled":
                elapsed = time.time() - start
                print("  CANCELLED", file=sys.stderr)
                records.append({
                    "input": str(item),
                    "status": "cancelled",
                    "seconds": elapsed,
                })
                if args.write_manifest:
                    _write_manifest(output_dir, records, args)
                return 130
            failures.append((item, exc))
            print(f"  ERROR: {exc}", file=sys.stderr)
            records.append({
                "input": str(item),
                "status": "failed",
                "error": str(exc),
            })

    if args.write_manifest:
        _write_manifest(output_dir, records, args)
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
