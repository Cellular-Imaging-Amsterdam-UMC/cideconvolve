"""Small Bilayers-compatible CLI helper for CIDeconvolve.

This intentionally keeps the project independent from the external bilayers
package at runtime while supporting the same useful parse/generate/validate
workflow for the local ``bilayers.yaml`` file.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in incomplete envs
    yaml = None


DEFAULT_CONFIG = Path(__file__).with_name("bilayers.yaml")


def _load_config(config_path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install PyYAML")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Bilayers config must be a YAML mapping")
    return config


def _iter_cli_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in ("inputs", "parameters"):
        for spec in config.get(source, []) or []:
            item = dict(spec)
            item["source"] = source[:-1]
            items.append(item)
    for spec in config.get("exec_function", {}).get("hidden_args", []) or []:
        item = dict(spec)
        item["source"] = "hidden"
        items.append(item)
    return sorted(items, key=lambda item: int(item.get("cli_order", 0)))


def _format_cli_arg(cli_tag: str, value: Any) -> str:
    if cli_tag in ("", None):
        return shlex.quote(str(value))
    if "=" in cli_tag:
        return f"{cli_tag}{shlex.quote(str(value))}"
    return f"{cli_tag} {shlex.quote(str(value))}"


def generate_cli_command(config: dict[str, Any]) -> str:
    command = [str(config.get("exec_function", {}).get("cli_command", "python wrapper.py")).strip()]
    for item in _iter_cli_items(config):
        cli_tag = item.get("cli_tag")
        if cli_tag in (None, "None"):
            continue
        if item["source"] == "hidden":
            value = item.get("value")
            append_value = bool(item.get("append_value", True))
        else:
            value = item.get("default")
            append_value = bool(item.get("append_value", False))
            if item.get("type") in {"image", "file", "directory"} and item.get("folder_name"):
                value = item["folder_name"]

        if item.get("type") == "checkbox" or isinstance(value, bool):
            if append_value:
                command.append(_format_cli_arg(str(cli_tag), value))
            elif value:
                command.append(str(cli_tag))
            continue

        if value not in (None, ""):
            command.append(_format_cli_arg(str(cli_tag), value))
    return " ".join(part for part in command if part)


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("docker_image", "algorithm_folder_name", "exec_function", "inputs", "outputs", "parameters"):
        if key not in config:
            errors.append(f"Missing top-level key: {key}")
    for section in ("inputs", "outputs", "parameters"):
        value = config.get(section, [])
        if not isinstance(value, list):
            errors.append(f"{section} must be a list")
            continue
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"{section}[{index}] must be a mapping")
                continue
            name = item.get("name")
            if not name:
                errors.append(f"{section}[{index}] is missing name")
            elif name in seen:
                errors.append(f"{section} has duplicate name: {name}")
            seen.add(str(name))
            if section != "outputs" and not item.get("cli_tag"):
                errors.append(f"{section}.{name} is missing cli_tag")
            mode = item.get("mode")
            if mode not in ("beginner", "advanced"):
                errors.append(f"{section}.{name} mode must be beginner or advanced")
    return errors


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CIDeconvolve Bilayers YAML helper.",
    )
    parser.add_argument("-v", "--version", action="version", version="bilayers_cli 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a Bilayers YAML config file.")
    parse_parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))

    generate_parser = subparsers.add_parser("generate", help="Generate outputs from a Bilayers YAML config file.")
    generate_parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    generate_parser.add_argument("--cli", action="store_true", help="Generate the default CLI command.")

    validate_parser = subparsers.add_parser("validate", help="Validate a Bilayers YAML config file.")
    validate_parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))

    args = parser.parse_args(argv)
    try:
        config = _load_config(Path(args.config))
        if args.command == "parse":
            image = config["docker_image"]
            print(f"Inputs: {len(config.get('inputs', []))}")
            print(f"Outputs: {len(config.get('outputs', []))}")
            print(f"Parameters: {len(config.get('parameters', []))}")
            print(f"Docker Image: {image['org']}/{image['name']}:{image['tag']} ({image['platform']})")
            print(f"CLI Sequence Order: {[item.get('name', item.get('cli_tag')) for item in _iter_cli_items(config)]}")
        elif args.command == "generate":
            if not args.cli:
                print("Only --cli generation is implemented for this local helper.")
                return 1
            print("Generated CLI Command:")
            print(generate_cli_command(config))
        elif args.command == "validate":
            errors = validate_config(config)
            if errors:
                for error in errors:
                    print(f"[ERROR] {error}")
                return 1
            print("No issues found")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli())
