from pathlib import Path

import yaml

from bilayers_cli import generate_cli_command, validate_config, validate_config_strict


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
    "device",
    "tv_lambda",
    "sparse_hessian_reg",
    "sparse_hessian_weight",
    "output_format",
    "output_dtype",
    "streaming",
    "streaming_threshold_gb",
    "t_start",
    "t_stop",
    "t_step",
    "hcs_field",
    "benchmark",
    "bench_crop",
    "compute_metrics",
    "background",
    "offset",
    "prefilter_sigma",
    "snr_mode",
    "snr_value",
    "acuity",
    "two_d_mode",
    "two_d_wf_aggressiveness",
    "two_d_wf_bg_radius_um",
    "two_d_wf_bg_scale",
]

ADVANCED_PARAMETERS = {
    "output_format",
    "output_dtype",
    "streaming",
    "streaming_threshold_gb",
    "t_start",
    "t_stop",
    "t_step",
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
    "prefilter_sigma",
    "two_d_mode",
    "two_d_wf_aggressiveness",
    "two_d_wf_bg_radius_um",
    "two_d_wf_bg_scale",
}

BILAYERS_TOP_LEVEL_KEYS = {
    "citations",
    "docker_image",
    "algorithm_folder_name",
    "exec_function",
    "inputs",
    "outputs",
    "parameters",
    "display_only",
}


def _load_bilayers() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_bilayers_yaml_has_expected_parameter_order() -> None:
    bilayers = _load_bilayers()

    bilayers_ids = [entry["name"] for entry in bilayers["parameters"]]

    assert bilayers_ids == EXPECTED_PARAMETER_ORDER


def test_config_yaml_has_bilayers_standard_sections() -> None:
    bilayers = _load_bilayers()

    assert BILAYERS_TOP_LEVEL_KEYS <= set(bilayers)


def test_config_yaml_has_public_citation_metadata() -> None:
    bilayers = _load_bilayers()

    citation = bilayers["citations"][0]

    assert citation["license"] == "MIT"
    assert citation["doi"].startswith("https://github.com/")


def test_config_yaml_image_entries_include_schema_required_fields() -> None:
    bilayers = _load_bilayers()

    for section in ("inputs", "outputs"):
        for entry in bilayers[section]:
            if entry["type"] != "image":
                continue
            assert entry["subtype"]
            assert entry["unique_string"]
            for key in ("depth", "timepoints", "tiled", "pyramidal"):
                assert key in entry


def test_bilayers_modes_match_launcher_layout() -> None:
    bilayers = _load_bilayers()

    bilayers_modes = {entry["name"]: entry["mode"] for entry in bilayers["parameters"]}
    output_format = next(entry for entry in bilayers["parameters"] if entry["name"] == "output_format")
    pinhole = next(entry for entry in bilayers["parameters"] if entry["name"] == "pinhole_airy")

    assert {name for name, mode in bilayers_modes.items() if mode == "advanced"} == ADVANCED_PARAMETERS
    assert output_format["default"] == "ome-zarr"
    assert pinhole["default"] == "1"


def test_bilayers_cli_helper_validates_and_generates_command() -> None:
    bilayers = _load_bilayers()

    assert validate_config(bilayers) == []
    command = generate_cli_command(bilayers)

    assert command.startswith("python wrapper.py --infolder /data/in --outfolder /data/out")
    assert command.split().count("--outfolder") == 1
    assert "--method ci_rl" in command
    assert "--output_format ome-zarr" in command
    assert "--output_dtype float32" in command
    assert "--snr_mode none" in command
    assert "--snr_value 4.0" in command
    assert "--two_d_wf_aggressiveness Balanced" in command


def test_strict_validator_reports_dependency_or_schema_result() -> None:
    bilayers = _load_bilayers()

    errors = validate_config_strict(bilayers)

    assert isinstance(errors, list)
