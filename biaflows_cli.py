"""
biaflows_cli.py — Local BIAFLOWS CLI helper for CIDeconvolve.

Provides a BiaflowsJob class and helper functions that mirror the
Cytomine/BIAFLOWS runner API so that the workflow can run locally
(inside Docker or on the host) without any Cytomine dependencies.

Based on the pattern from W_CellExpansionAdvanced.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

CLASS_SPTCNT = "LOCAL_CLASS_SPTCNT"

KNOWN_JOB_ATTRS = {
    "input_dir",
    "output_dir",
    "gt_dir",
    "temp_dir",
    "suffixes",
    "local",
    "parameters",
    "parameters_json",
}

DEFAULT_SUFFIXES = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".npy",
)

_IMMERSION_RI = {
    "air": 1.0003,
    "water": 1.333,
    "oil": 1.515,
}

_SAMPLE_RI = {
    "water": 1.333,
    "pbs": 1.334,
    "culture medium": 1.337,
    "vectashield": 1.45,
    "prolong gold": 1.47,
    "glycerol": 1.474,
    "oil": 1.515,
    "prolong glass": 1.52,
}

_DEFAULT_NA = 1.4
_DEFAULT_EMISSION_WL = "520"
_DEFAULT_PIXEL_SIZE_XY_NM = 65.0
_DEFAULT_PIXEL_SIZE_Z_NM = 200.0
_DEFAULT_MICROSCOPE_TYPE = "confocal"
_DEFAULT_EXCITATION_WL = "488"
_DEFAULT_PINHOLE_AIRY = 1.0
_DEFAULT_IMMERSION_RI_CHOICE = "oil (1.515)"
_DEFAULT_SAMPLE_RI_CHOICE = "prolong gold (1.47)"
_SAMPLE_RI_DEFAULT = 1.47
_START_MODES = (
    "auto",
    "flat",
    "percentile_flat",
    "observed",
    "observed_bgsub",
    "lowpass",
    "lowpass_bgsub",
    "hybrid",
)


def _str_to_bool(value: str) -> bool:
    """Convert a string to a boolean for argparse."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def _to_bool(value: Any) -> bool:
    """Convert CLI, JSON, and descriptor boolean values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _parse_ri_choice(raw: str, lookup: dict[str, float]) -> float | None:
    """Parse a RI choice string like 'oil (1.515)' or a bare float."""
    text = str(raw).strip().lower()
    if not text:
        return None
    name = text.split("(")[0].strip()
    if name in lookup:
        return lookup[name]
    try:
        return float(text)
    except ValueError:
        return None


def _parse_float_or_default(raw: Any, default: float) -> float:
    """Parse a finite float, accepting non-numeric values as the default."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return value


def _parse_float_list_or_default(raw: Any, default: str) -> list[float]:
    """Parse comma- or semicolon-separated floats."""
    text = str(raw if raw is not None else default).strip()
    if not text or text.lower() == "auto":
        text = default
    values: list[float] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values or [float(default)]


def _parse_tile_limits(raw: Any, default: tuple[int, int] = (0, 64)) -> tuple[int, int]:
    """Parse tile limits as max_xy,max_z; XY <= 0 means auto tile sizing."""
    text = str(raw or "").strip()
    if not text or text.lower() == "auto":
        return default
    parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
    try:
        max_xy = int(parts[0]) if parts else default[0]
        max_z = int(parts[1]) if len(parts) > 1 else default[1]
    except ValueError:
        return default
    if max_xy <= 0:
        max_xy = 0
    return (max_xy if max_xy == 0 else max(max_xy, 64)), max(max_z, 1)


def _load_descriptor_inputs() -> List[dict]:
    """Return parameter definitions declared in descriptor.json if available."""
    descriptor_path = Path(__file__).with_name("descriptor.json")
    try:
        with descriptor_path.open("r", encoding="utf-8") as stream:
            descriptor = json.load(stream)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as exc:
        print(f"Warning: descriptor.json could not be parsed ({exc}); ignoring parameter metadata.")
        return []
    inputs = descriptor.get("inputs", [])
    if not isinstance(inputs, list):
        return []
    return inputs


@dataclass
class ImageResource:
    """Minimal image representation compatible with the BIAFLOWS wrapper."""

    filename: str
    filename_original: str
    filepath: Path

    def __post_init__(self) -> None:
        self.filepath = Path(self.filepath)
        self.path = str(self.filepath)


class BiaflowsJob:
    """Local stand-in for the Cytomine/BIAFLOWS job helper."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        parameters: Optional[SimpleNamespace] = None,
    ) -> None:
        if parameters is None:
            parameters = getattr(args, "parameters", None)
        if parameters is None:
            param_values = {
                key: value
                for key, value in vars(args).items()
                if key not in KNOWN_JOB_ATTRS
            }
            parameters = SimpleNamespace(**param_values)

        self.parameters = parameters
        self.flags = {}
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.gt_dir = Path(args.gt_dir)

        temp_dir_value = getattr(args, "temp_dir", None)
        if temp_dir_value is None:
            temp_dir_value = self.output_dir / "tmp"
        self.temp_dir = Path(temp_dir_value)
        self.suffixes = self._normalise_suffixes(args.suffixes)

    def __enter__(self) -> "BiaflowsJob":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    @classmethod
    def from_cli(
        cls,
        argv: Sequence[str],
        **overrides,
    ) -> "BiaflowsJob":
        args = _parse_args(argv)
        parameters = overrides.pop(
            "parameters",
            getattr(args, "parameters", None),
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return cls(args, parameters=parameters)

    @staticmethod
    def _normalise_suffixes(
        suffixes: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        if not suffixes:
            return list(DEFAULT_SUFFIXES)
        normalised: List[str] = []
        for suffix in suffixes:
            clean = suffix.strip().lower()
            if not clean:
                continue
            if not clean.startswith("."):
                clean = f".{clean}"
            normalised.append(clean)
        return normalised or list(DEFAULT_SUFFIXES)


def resolve_workflow_parameters(parameters: object | None) -> SimpleNamespace:
    """Resolve raw CLI/Bilayers parameters into wrapper-ready values."""
    if parameters is None:
        parameters = SimpleNamespace()

    iter_raw = str(getattr(parameters, "iterations", "40")).strip()
    niter_list: list[int] = []
    for item in iter_raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            niter_list.append(max(1, int(float(item))))
        except ValueError:
            continue
    if not niter_list:
        niter_list = [40]

    method = str(getattr(parameters, "method", "ci_rl") or "ci_rl").strip()
    if method not in ("ci_rl", "ci_rl_tv", "ci_sparse_hessian"):
        method = "ci_rl"

    device_param = getattr(parameters, "device", "auto")
    device = None if device_param in (None, "auto") else device_param

    overrule_metadata = _to_bool(getattr(parameters, "overrule_image_metadata", False))
    na_value = _parse_float_or_default(getattr(parameters, "na", _DEFAULT_NA), _DEFAULT_NA)
    ri_raw = str(getattr(parameters, "refractive_index", _DEFAULT_IMMERSION_RI_CHOICE))
    ri_value = _parse_ri_choice(ri_raw, _IMMERSION_RI) or 1.515
    sample_ri_raw = str(getattr(parameters, "sample_ri", _DEFAULT_SAMPLE_RI_CHOICE))
    sample_ri_value = _parse_ri_choice(sample_ri_raw, _SAMPLE_RI) or _SAMPLE_RI_DEFAULT
    micro_value = str(getattr(parameters, "microscope_type", _DEFAULT_MICROSCOPE_TYPE)).strip().lower()
    if micro_value == "auto":
        micro_value = _DEFAULT_MICROSCOPE_TYPE
    em_raw = str(getattr(parameters, "emission_wl", _DEFAULT_EMISSION_WL)).strip()
    em_value = _parse_float_list_or_default(em_raw, _DEFAULT_EMISSION_WL)
    ex_raw = str(getattr(parameters, "excitation_wl", _DEFAULT_EXCITATION_WL)).strip()
    ex_value = _parse_float_list_or_default(ex_raw, _DEFAULT_EXCITATION_WL)
    pinhole_airy = _parse_float_list_or_default(
        getattr(parameters, "pinhole_airy", str(_DEFAULT_PINHOLE_AIRY)),
        str(_DEFAULT_PINHOLE_AIRY),
    )

    tv_lambda = _parse_float_or_default(getattr(parameters, "tv_lambda", 0.0001), 0.0001)
    damping_raw = str(getattr(parameters, "damping", "none")).strip().lower()
    if damping_raw in ("none", "0", "0.0"):
        damping: float | str = 0.0
    elif damping_raw == "auto":
        damping = "auto"
    else:
        damping = _parse_float_or_default(damping_raw, 0.0)

    bg_raw = str(getattr(parameters, "background", "auto")).strip()
    background: float | str = "auto" if bg_raw.lower() == "auto" else _parse_float_or_default(bg_raw, 0.0)
    offset_raw = str(getattr(parameters, "offset", "auto")).strip().lower()
    if offset_raw in ("none", "0", "0.0"):
        offset: float | str = 0.0
    elif offset_raw == "auto":
        offset = "auto"
    else:
        offset = _parse_float_or_default(offset_raw, 0.0)

    prefilter_sigma = max(0.0, _parse_float_or_default(getattr(parameters, "prefilter_sigma", 0.0), 0.0))
    start = str(getattr(parameters, "start", "auto")).strip().lower()
    if start not in _START_MODES:
        start = "flat"
    sparse_hessian_weight = min(
        max(_parse_float_or_default(getattr(parameters, "sparse_hessian_weight", 0.6), 0.6), 0.0),
        1.0,
    )
    sparse_hessian_reg = min(
        max(_parse_float_or_default(getattr(parameters, "sparse_hessian_reg", 0.98), 0.98), 0.0),
        1.0,
    )
    convergence = str(getattr(parameters, "convergence", "auto")).strip().lower()
    if convergence in ("none", "fixed"):
        convergence = "fixed"
    elif convergence != "auto":
        convergence = "auto"
    rel_threshold = min(
        max(_parse_float_or_default(getattr(parameters, "rel_threshold", 0.005), 0.005), 1e-8),
        1.0,
    )
    check_every = 5

    t_g = 170000.0
    t_g0 = 170000.0
    t_i0 = 100000.0
    z_p = 0.0

    px_xy_raw = str(getattr(parameters, "pixel_size_xy", _DEFAULT_PIXEL_SIZE_XY_NM)).strip()
    px_xy_nm = _parse_float_or_default(px_xy_raw, _DEFAULT_PIXEL_SIZE_XY_NM)
    px_xy_value = px_xy_nm / 1000.0
    px_z_raw = str(getattr(parameters, "pixel_size_z", _DEFAULT_PIXEL_SIZE_Z_NM)).strip()
    px_z_nm = _parse_float_or_default(px_z_raw, _DEFAULT_PIXEL_SIZE_Z_NM)
    px_z_value = px_z_nm / 1000.0

    projection = str(getattr(parameters, "projection", "none")).lower()
    benchmark = _to_bool(getattr(parameters, "benchmark", False))
    bench_crop = _to_bool(getattr(parameters, "bench_crop", False))
    compute_metrics = _to_bool(getattr(parameters, "compute_metrics", False))
    output_format = str(getattr(parameters, "output_format", "ome-tiff")).strip().lower()
    if output_format in ("ome_zarr", "zarr"):
        output_format = "ome-zarr"
    streaming_mode = str(getattr(parameters, "streaming", "auto")).strip().lower()
    tile_limits = _parse_tile_limits(getattr(parameters, "tile_limits", "auto"))
    streaming_threshold_gb = max(
        _parse_float_or_default(getattr(parameters, "streaming_threshold_gb", 2.0), 2.0),
        0.01,
    )
    scene = getattr(parameters, "scene", None)
    scene = None if scene in (None, "", "auto") else scene
    hcs_field = getattr(parameters, "hcs_field", None)
    hcs_field = None if hcs_field in (None, "", "auto") else str(hcs_field)

    two_d_mode = str(getattr(parameters, "two_d_mode", "auto")).strip().lower()
    two_d_wf_aggressiveness = str(getattr(parameters, "two_d_wf_aggressiveness", "Balanced")).strip()
    two_d_wf_bg_radius_um = max(
        _parse_float_or_default(getattr(parameters, "two_d_wf_bg_radius_um", 0.5), 0.5),
        0.1,
    )
    two_d_wf_bg_scale = max(
        _parse_float_or_default(getattr(parameters, "two_d_wf_bg_scale", 1.0), 1.0),
        0.1,
    )

    return SimpleNamespace(
        niter_list=niter_list,
        method=method,
        device_param=device_param,
        device=device,
        overrule_metadata=overrule_metadata,
        na_value=na_value,
        ri_raw=ri_raw,
        ri_value=ri_value,
        sample_ri_raw=sample_ri_raw,
        sample_ri_value=sample_ri_value,
        micro_value=micro_value,
        em_value=em_value,
        ex_value=ex_value,
        pinhole_airy=pinhole_airy,
        tv_lambda=tv_lambda,
        damping=damping,
        background=background,
        offset=offset,
        prefilter_sigma=prefilter_sigma,
        start=start,
        sparse_hessian_weight=sparse_hessian_weight,
        sparse_hessian_reg=sparse_hessian_reg,
        convergence=convergence,
        rel_threshold=rel_threshold,
        check_every=check_every,
        t_g=t_g,
        t_g0=t_g0,
        t_i0=t_i0,
        z_p=z_p,
        px_xy_nm=px_xy_nm,
        px_z_nm=px_z_nm,
        na_override=na_value,
        ri_override=ri_value,
        sample_ri=sample_ri_value,
        micro_override=micro_value,
        em_override=em_value,
        ex_override=ex_value,
        pinhole_airy_override=pinhole_airy,
        px_xy_override=px_xy_value,
        px_z_override=px_z_value,
        projection=projection,
        benchmark=benchmark,
        bench_crop=bench_crop,
        compute_metrics=compute_metrics,
        output_format=output_format,
        streaming_mode=streaming_mode,
        tile_limits=tile_limits,
        streaming_threshold_gb=streaming_threshold_gb,
        scene=scene,
        hcs_field=hcs_field,
        two_d_mode=two_d_mode,
        two_d_wf_aggressiveness=two_d_wf_aggressiveness,
        two_d_wf_bg_radius_um=two_d_wf_bg_radius_um,
        two_d_wf_bg_scale=two_d_wf_bg_scale,
    )


def prepare_data(
    discipline: str,
    job: BiaflowsJob,
    *,
    is_2d: bool = True,
    **flags,
):
    """Prepare input/output directories and enumerate available images."""
    del discipline, is_2d, flags

    job.input_dir.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.temp_dir.mkdir(parents=True, exist_ok=True)

    in_imgs = _collect_images(job.input_dir, job.suffixes)
    gt_imgs = _collect_images(job.gt_dir, job.suffixes)

    return (
        in_imgs,
        gt_imgs,
        str(job.input_dir),
        str(job.gt_dir),
        str(job.output_dir),
        str(job.temp_dir),
    )


def get_discipline(job: BiaflowsJob, default: Optional[str] = None) -> Optional[str]:
    """Return the requested default discipline (placeholder for compatibility)."""
    del job
    return default


def _collect_images(directory: Path, suffixes: Optional[Sequence[str]]) -> List[ImageResource]:
    if not directory.exists():
        return []
    records: List[ImageResource] = []
    for entry in sorted(directory.iterdir()):
        # OME-Zarr stores are directories ending in .zarr
        if entry.is_dir() and entry.suffix.lower() == ".zarr":
            records.append(
                ImageResource(
                    filename=entry.name,
                    filename_original=entry.name,
                    filepath=entry,
                )
            )
            continue
        if not entry.is_file():
            continue
        if suffixes and entry.suffix.lower() not in suffixes:
            continue
        records.append(
            ImageResource(
                filename=entry.name,
                filename_original=entry.name,
                filepath=entry,
            )
        )
    return records


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local BIAFLOWS runner for CIDeconvolve."
    )
    # BIAFLOWS standard directory arguments
    parser.add_argument("--input-dir", dest="input_dir")
    parser.add_argument(
        "--infolder", dest="input_dir",
        help="Compatibility alias for --input-dir.",
    )
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument(
        "--outfolder", dest="output_dir",
        help="Compatibility alias for --output-dir.",
    )
    parser.add_argument("--gt-dir", dest="gt_dir", default="")
    parser.add_argument(
        "--gtfolder", dest="gt_dir",
        help="Compatibility alias for --gt-dir.",
    )
    parser.add_argument("--temp-dir", dest="temp_dir", default=None)
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally without Cytomine.",
    )
    parser.add_argument(
        "--suffixes", nargs="*", default=None,
        help="File suffixes to process (default: .tif .tiff .ome.tif .ome.tiff .png).",
    )
    parser.add_argument(
        "--parameters",
        dest="parameters_json",
        default=None,
        help="JSON object with parameter defaults/values, used by Bilayers-compatible launchers.",
    )

    # Descriptor-defined parameters (loaded from descriptor.json)
    descriptor_inputs = _load_descriptor_inputs()
    descriptor_param_ids: list[str] = []
    descriptor_defaults: dict[str, Any] = {}
    for inp in descriptor_inputs:
        param_id = inp.get("id")
        if not param_id:
            continue
        descriptor_param_ids.append(param_id)
        flag = inp.get("command-line-flag", f"--{param_id}")
        param_type = inp.get("type", "String")
        default = inp.get("default-value")
        descriptor_defaults[param_id] = default

        kwargs = {"default": argparse.SUPPRESS, "help": inp.get("description", "")}

        if param_type == "Boolean":
            kwargs["nargs"] = "?"
            kwargs["const"] = True
            kwargs["type"] = _str_to_bool
            kwargs["metavar"] = "BOOL"
        elif param_type == "Number":
            is_int = inp.get("integer", False)
            kwargs["type"] = int if is_int else float
        else:
            kwargs["type"] = str

        parser.add_argument(flag, dest=param_id, **kwargs)

    args, unknown = parser.parse_known_args(argv)

    parameter_values: dict[str, Any] = dict(descriptor_defaults)
    if args.parameters_json:
        try:
            loaded = json.loads(args.parameters_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--parameters must be a JSON object: {exc}")
        if not isinstance(loaded, dict):
            parser.error("--parameters must be a JSON object")
        parameter_values.update(loaded)
    for param_id in descriptor_param_ids:
        if hasattr(args, param_id):
            parameter_values[param_id] = getattr(args, param_id)
    if parameter_values:
        args.parameters = SimpleNamespace(**parameter_values)

    # Default directories for Docker convention
    if not args.input_dir:
        args.input_dir = "/data/in"
    if not args.output_dir:
        args.output_dir = "/data/out"
    if not args.gt_dir:
        args.gt_dir = "/data/gt"

    return args
