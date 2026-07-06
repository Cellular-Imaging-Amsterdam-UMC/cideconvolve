import json
from pathlib import Path

import yaml

from bilayers_cli import generate_cli_command, validate_config


ROOT = Path(__file__).resolve().parent.parent


def _load_descriptor() -> dict:
    return json.loads((ROOT / "descriptor.json").read_text(encoding="utf-8"))


def _load_bilayers() -> dict:
    return yaml.safe_load((ROOT / "bilayers.yaml").read_text(encoding="utf-8"))


def test_bilayers_yaml_is_complete_for_descriptor_parameters() -> None:
    descriptor = _load_descriptor()
    bilayers = _load_bilayers()

    descriptor_ids = [entry["id"] for entry in descriptor["inputs"]]
    bilayers_ids = [entry["name"] for entry in bilayers["parameters"]]

    assert bilayers_ids == descriptor_ids


def test_bilayers_beginner_mode_matches_non_advanced_descriptor_inputs() -> None:
    descriptor = _load_descriptor()
    bilayers = _load_bilayers()

    descriptor_modes = {
        entry["id"]: "advanced" if "(adv)" in entry.get("name", "").lower() else "beginner"
        for entry in descriptor["inputs"]
    }
    bilayers_modes = {entry["name"]: entry["mode"] for entry in bilayers["parameters"]}

    assert bilayers_modes == descriptor_modes


def test_bilayers_cli_helper_validates_and_generates_command() -> None:
    bilayers = _load_bilayers()

    assert validate_config(bilayers) == []
    command = generate_cli_command(bilayers)

    assert command.startswith("python wrapper.py --infolder /data/in --outfolder /data/out")
    assert "--gtfolder /data/gt" in command
    assert "--method ci_rl" in command
    assert "--two_d_wf_aggressiveness Balanced" in command
