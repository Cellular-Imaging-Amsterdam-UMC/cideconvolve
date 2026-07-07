"""Generate Docker-friendly Bilayers interface artifacts for CIDeconvolve."""

from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_INTERFACES = {"gradio", "jupyter"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def build_interface_config(config: dict[str, Any], interface: str) -> dict[str, Any]:
    """Return a temporary config adjusted for generated interface containers."""
    if interface not in SUPPORTED_INTERFACES:
        raise ValueError(f"Unsupported interface: {interface}")
    interface_config = copy.deepcopy(config)
    _normalise_hidden_flags(interface_config)
    parameters = list(interface_config.get("parameters", []) or [])
    outputs = list(interface_config.get("outputs", []) or [])

    has_outfolder_parameter = any(
        isinstance(item, dict) and item.get("cli_tag") == "--outfolder"
        for item in parameters
    )
    if not has_outfolder_parameter:
        output = next(
            (
                item
                for item in outputs
                if isinstance(item, dict) and item.get("cli_tag") == "--outfolder"
            ),
            None,
        )
        if output is not None:
            parameters.insert(
                0,
                {
                    "name": "outfolder",
                    "type": "textbox",
                    "label": output.get("label", "Output Folder"),
                    "description": output.get(
                        "description",
                        "Output folder for CIDeconvolve results.",
                    ),
                    "output_dir_set": True,
                    "default": output.get("folder_name", "/data/out"),
                    "cli_tag": "--outfolder",
                    "cli_order": output.get("cli_order", 2),
                    "optional": bool(output.get("optional", False)),
                    "section_id": output.get("section_id", "outputs"),
                    "mode": output.get("mode", "beginner"),
                },
            )

    interface_config["parameters"] = parameters
    return interface_config


def _normalise_hidden_flags(config: dict[str, Any]) -> None:
    """Make schema-clean string hidden flags behave as booleans in interfaces."""
    hidden_args = config.get("exec_function", {}).get("hidden_args", []) or []
    for item in hidden_args:
        if not isinstance(item, dict) or item.get("append_value", True):
            continue
        value = item.get("value")
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
            item["value"] = True
        elif isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
            item["value"] = False


def build_gradio_config(config: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible helper for Gradio-specific tests/imports."""
    return build_interface_config(config, "gradio")


def build_jupyter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a temporary config for the generated Jupyter notebook."""
    return build_interface_config(config, "jupyter")


def patch_gradio_app(app_path: Path) -> None:
    """Patch the generated Gradio app for private Docker use and directory outputs."""
    text = app_path.read_text(encoding="utf-8")
    text = _add_shell_safe_quoting(text)
    text = _replace_create_zip_function(text)

    launch_call = 'app.launch(server_name="0.0.0.0", server_port=7878, share=True)'
    launch_replacement = """app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7878")),
        share=os.environ.get("GRADIO_SHARE", "false").strip().lower() in ("1", "true", "yes", "on"),
    )"""
    if launch_call not in text:
        raise RuntimeError("Could not find the expected Gradio launch call to patch.")
    text = text.replace(launch_call, launch_replacement)
    app_path.write_text(text, encoding="utf-8")


def _add_shell_safe_quoting(text: str) -> str:
    """Patch upstream's shell-string command builder to quote CLI values."""
    if "import shlex" not in text:
        text = text.replace("import shutil\n", "import shutil\nimport shlex\n", 1)
    if "import tempfile" not in text:
        if "import subprocess\n" in text:
            text = text.replace("import subprocess\n", "import subprocess\nimport tempfile\n", 1)
        else:
            text = text.replace("import shutil\n", "import shutil\nimport tempfile\n", 1)

    replacement = '''def option_to_append(cli_tag: str, value: Any) -> str:
    """
    Formats CLI options for appending to a shell command with safe quoting.
    """
    if value is None:
        return ""

    quoted_value = shlex.quote(str(value))
    if cli_tag == "":
        return quoted_value
    elif "=" in cli_tag:
        return f"{cli_tag}{quoted_value}"
    return f"{cli_tag} {quoted_value}"'''

    pattern = re.compile(
        r"def option_to_append\(cli_tag: str, value: Any\) -> str:.*?return f\"\{cli_tag\} \{value\}\"  # Append cli_tag and value",
        re.DOTALL,
    )
    patched, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not patch option_to_append in generated Gradio app.")
    return patched


def _replace_create_zip_function(text: str) -> str:
    replacement = '''def copy_files_for_gradio(output_files: list[str]) -> list[str]:
    """Copy returned output files to a Gradio-allowed temporary directory."""
    if not output_files:
        return []
    cache_dir = tempfile.mkdtemp(prefix="cideconvolve_files_")
    cached_files: list[str] = []
    for file_path in output_files:
        if os.path.isfile(file_path):
            cached_path = os.path.join(cache_dir, os.path.basename(file_path))
            shutil.copy2(file_path, cached_path)
            cached_files.append(cached_path)
    return cached_files


def create_zip_from_files(output_files: list[str], output_folder_name: Optional[str]) -> Optional[str]:
    """Create a Gradio-downloadable zip archive from output files."""
    archive_root = output_folder_name or os.getcwd()
    if not output_files and output_folder_name and os.path.isdir(output_folder_name):
        output_files = []
        for root, _, files in os.walk(output_folder_name):
            for filename in files:
                output_files.append(os.path.join(root, filename))
    if not output_files:
        return None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"outputs_{timestamp}.zip"
    cache_dir = tempfile.mkdtemp(prefix="cideconvolve_outputs_")
    zip_path = os.path.join(cache_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_files:
            if os.path.isfile(file_path):
                arcname = os.path.relpath(file_path, archive_root)
                zipf.write(file_path, arcname=arcname)
    return zip_path'''

    pattern = re.compile(
        r"def create_zip_from_files\(output_files: list\[str\], output_folder_name: Optional\[str\]\) -> Optional\[str\]:.*?return zip_path",
        re.DOTALL,
    )
    patched, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("Could not patch create_zip_from_files in generated Gradio app.")
    patched, return_count = re.subn(
        r"(?P<indent>\s+)zip_path = create_zip_from_files\(output_files, output_folder_name\)\n"
        r"\s+return output_files, zip_path",
        r"\g<indent>zip_path = create_zip_from_files(output_files, output_folder_name)\n"
        r"\g<indent>return copy_files_for_gradio(output_files), zip_path",
        patched,
        count=1,
    )
    if return_count != 1:
        raise RuntimeError("Could not patch Gradio output file return paths.")
    return patched


def generate_interface_artifact(config_path: Path, workdir: Path, interface: str, output_path: Path) -> None:
    if interface not in SUPPORTED_INTERFACES:
        raise ValueError(f"Unsupported interface: {interface}")

    config = load_yaml(config_path)
    interface_config = build_interface_config(config, interface)
    algorithm_folder_name = str(interface_config.get("algorithm_folder_name", "")).strip()
    if not algorithm_folder_name:
        raise ValueError("config.yaml is missing algorithm_folder_name")

    with tempfile.TemporaryDirectory(prefix=f"cideconvolve_{interface}_") as tmp:
        tmp_config = Path(tmp) / f"config.{interface}.yaml"
        tmp_config.write_text(
            yaml.safe_dump(interface_config, sort_keys=False),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "bilayers_cli",
                "generate",
                str(tmp_config),
                "--interface",
                interface,
            ],
            cwd=workdir,
            check=True,
        )

    generated_path = _find_generated_artifact(workdir, algorithm_folder_name, interface)
    if not generated_path.exists():
        found = sorted(str(path.relative_to(workdir)) for path in (workdir / "dist").rglob("*") if path.is_file())
        found_text = ", ".join(found) if found else "no files under dist"
        raise FileNotFoundError(
            f"Bilayers did not generate the expected {interface} artifact. "
            f"Looked for {_generated_artifact_name(interface)}; found: {found_text}"
        )
    if interface == "gradio":
        patch_gradio_app(generated_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(generated_path, output_path)


def _generated_artifact_path(workdir: Path, algorithm_folder_name: str, interface: str) -> Path:
    if interface == "gradio":
        return workdir / "dist" / "gradio" / algorithm_folder_name / "app.py"
    if interface == "jupyter":
        return workdir / "dist" / "jupyter" / algorithm_folder_name / "generated_notebook.ipynb"
    raise ValueError(f"Unsupported interface: {interface}")


def _generated_artifact_name(interface: str) -> str:
    if interface == "gradio":
        return "app.py"
    if interface == "jupyter":
        return "generated_notebook.ipynb"
    raise ValueError(f"Unsupported interface: {interface}")


def _find_generated_artifact(workdir: Path, algorithm_folder_name: str, interface: str) -> Path:
    expected = _generated_artifact_path(workdir, algorithm_folder_name, interface)
    if expected.exists():
        return expected

    artifact_name = _generated_artifact_name(interface)
    search_root = workdir / "dist" / interface
    matches = sorted(search_root.rglob(artifact_name)) if search_root.exists() else []
    if len(matches) == 1:
        return matches[0]
    return expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/app/config.yaml"))
    parser.add_argument("--workdir", type=Path, default=Path("/app"))
    parser.add_argument("--interface", choices=sorted(SUPPORTED_INTERFACES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    generate_interface_artifact(args.config, args.workdir, args.interface, args.output)
    print(f"Generated Docker {args.interface} artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
