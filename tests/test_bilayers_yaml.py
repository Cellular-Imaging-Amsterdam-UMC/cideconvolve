import json
from pathlib import Path

import yaml

from bilayers_cli import generate_cli_command, validate_config


ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PARAMETER_ORDER = [
    "method",
    "iterations",
    "convergence",
    "rel_threshold",
    "start",
    "microscope_type",
    "na",
    "pinhole_airy",
    "refractive_index",
    "sample_ri",
    "excitation_wl",
    "emission_wl",
    "pixel_size_xy",
    "pixel_size_z",
    "overrule_image_metadata",
    "projection",
    "output_format",
    "streaming",
    "streaming_threshold_gb",
    "scene",
    "hcs_field",
    "benchmark",
    "bench_crop",
    "compute_metrics",
    "tv_lambda",
    "sparse_hessian_weight",
    "sparse_hessian_reg",
    "device",
    "background",
    "offset",
    "damping",
    "prefilter_sigma",
    "two_d_mode",
    "two_d_wf_aggressiveness",
    "two_d_wf_bg_radius_um",
    "two_d_wf_bg_scale",
]

ADVANCED_PARAMETERS = {
    "output_format",
    "streaming",
    "streaming_threshold_gb",
    "scene",
    "hcs_field",
    "benchmark",
    "bench_crop",
    "compute_metrics",
    "tv_lambda",
    "sparse_hessian_weight",
    "sparse_hessian_reg",
    "device",
    "background",
    "offset",
    "damping",
    "prefilter_sigma",
    "two_d_mode",
    "two_d_wf_aggressiveness",
    "two_d_wf_bg_radius_um",
    "two_d_wf_bg_scale",
}


def _load_descriptor() -> dict:
    return json.loads((ROOT / "descriptor.json").read_text(encoding="utf-8"))


def _load_bilayers() -> dict:
    return yaml.safe_load((ROOT / "bilayers.yaml").read_text(encoding="utf-8"))


def test_bilayers_yaml_is_complete_for_descriptor_parameters() -> None:
    descriptor = _load_descriptor()
    bilayers = _load_bilayers()

    descriptor_ids = {entry["id"] for entry in descriptor["inputs"]}
    bilayers_ids = [entry["name"] for entry in bilayers["parameters"]]

    assert set(bilayers_ids) == descriptor_ids
    assert bilayers_ids == EXPECTED_PARAMETER_ORDER


def test_bilayers_modes_match_launcher_layout() -> None:
    bilayers = _load_bilayers()

    bilayers_modes = {entry["name"]: entry["mode"] for entry in bilayers["parameters"]}
    output_format = next(entry for entry in bilayers["parameters"] if entry["name"] == "output_format")

    assert {name for name, mode in bilayers_modes.items() if mode == "advanced"} == ADVANCED_PARAMETERS
    assert output_format["default"] == "ome-zarr"


def test_bilayers_cli_helper_validates_and_generates_command() -> None:
    bilayers = _load_bilayers()

    assert validate_config(bilayers) == []
    command = generate_cli_command(bilayers)

    assert command.startswith("python wrapper.py --infolder /data/in --outfolder /data/out")
    assert "--gtfolder /data/gt" in command
    assert "--method ci_rl" in command
    assert "--output_format ome-zarr" in command
    assert "--two_d_wf_aggressiveness Balanced" in command
