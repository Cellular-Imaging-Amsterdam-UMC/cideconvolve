import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
HELPER_PATH = ROOT / "docker" / "prepare_bilayers_interface.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("prepare_bilayers_interface", HELPER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_gradio_config_adds_outfolder_parameter_without_mutating_config() -> None:
    helper = _load_helper()
    config = _load_config()

    assert all(param.get("cli_tag") != "--outfolder" for param in config["parameters"])

    gradio_config = helper.build_gradio_config(config)
    gradio_outfolders = [
        param for param in gradio_config["parameters"] if param.get("cli_tag") == "--outfolder"
    ]

    assert len(gradio_outfolders) == 1
    assert gradio_outfolders[0]["name"] == "outfolder"
    assert gradio_outfolders[0]["type"] == "textbox"
    assert gradio_outfolders[0]["default"] == "/data/out"
    assert gradio_outfolders[0]["output_dir_set"] is True
    assert all(param.get("cli_tag") != "--outfolder" for param in config["parameters"])
    assert config["exec_function"]["hidden_args"][0]["value"] == "true"
    assert gradio_config["exec_function"]["hidden_args"][0]["value"] is True


def test_jupyter_config_adds_outfolder_parameter_without_mutating_config() -> None:
    helper = _load_helper()
    config = _load_config()

    assert all(param.get("cli_tag") != "--outfolder" for param in config["parameters"])

    jupyter_config = helper.build_jupyter_config(config)
    jupyter_outfolders = [
        param for param in jupyter_config["parameters"] if param.get("cli_tag") == "--outfolder"
    ]

    assert len(jupyter_outfolders) == 1
    assert jupyter_outfolders[0]["name"] == "outfolder"
    assert jupyter_outfolders[0]["default"] == "/data/out"
    assert all(param.get("cli_tag") != "--outfolder" for param in config["parameters"])


def test_gradio_app_patch_disables_share_and_zips_directory_outputs(tmp_path: Path) -> None:
    helper = _load_helper()
    app_path = tmp_path / "app.py"
    app_path.write_text(
        '''
import datetime
import os
import shutil
import zipfile
from typing import Any
from typing import Optional

def option_to_append(cli_tag: str, value: Any) -> str:
    """
    Formats CLI options for appending to the command.
    """
    if value is None:
        return ""
    
    if cli_tag == "":
        return str(value)  # Append only the value
    elif "=" in cli_tag:
        return f"{cli_tag}{value}"
    return f"{cli_tag} {value}"  # Append cli_tag and value

def create_zip_from_files(output_files: list[str], output_folder_name: Optional[str]) -> Optional[str]:
    """ Creates a zip archive from the output files and returns the zip path """
    if not output_files:
        return None
    base_dir = output_folder_name or os.getcwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"outputs_{timestamp}.zip"
    zip_path = os.path.join(base_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_files:
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=os.path.basename(file_path))
    return zip_path

def on_submit():
    output_files = ["out.ome.tiff"]
    output_folder_name = "outputs"
    zip_path = create_zip_from_files(output_files, output_folder_name)
    return output_files, zip_path

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7878, share=True)
''',
        encoding="utf-8",
    )

    helper.patch_gradio_app(app_path)

    patched = app_path.read_text(encoding="utf-8")
    assert 'os.environ.get("GRADIO_SHARE", "false")' in patched
    assert 'os.walk(output_folder_name)' in patched
    assert "import shlex" in patched
    assert "import tempfile" in patched
    assert "shlex.quote(str(value))" in patched
    assert 'tempfile.mkdtemp(prefix="cideconvolve_files_")' in patched
    assert "return copy_files_for_gradio(output_files), zip_path" in patched
    assert 'tempfile.mkdtemp(prefix="cideconvolve_outputs_")' in patched
    assert 'zip_path = os.path.join(cache_dir, zip_name)' in patched
    assert "share=True" not in patched
