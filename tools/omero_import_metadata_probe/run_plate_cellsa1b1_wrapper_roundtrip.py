"""Rerunnable HCS plate OME-Zarr wrapper round-trip test.

The test uses ``localdata/cellsA1B1.ome.zarr`` as an OMERO/BIOMERO plate input,
runs the CIDeconvolve wrapper on the HCS plate Zarr, and imports the wrapper
output plate back into OMERO through both direct and BIOMERO paths.

This script does not start Docker Compose. It assumes the local NL-BIOMERO
containers are already running.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path(r"C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve\python.exe")
DEFAULT_SOURCE = ROOT / "localdata" / "cellsA1B1.ome.zarr"
DEFAULT_REPORT_ROOT = Path.home() / "Downloads" / "cideconvolve_omero_roundtrips"
DEFAULT_SHARED_SUBDIR = "cideconvolve_probe/cellsa1b1_plate_roundtrip"


def _run(cmd: list[str], *, cwd: Path = ROOT, log: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "$ " + " ".join(cmd) + "\n\n"
            + "STDOUT\n"
            + (proc.stdout or "")
            + "\nSTDERR\n"
            + (proc.stderr or ""),
            encoding="utf-8",
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n{proc.stderr}")
    return proc


def _python(args: argparse.Namespace) -> str:
    exe = Path(args.python)
    return str(exe if exe.exists() else sys.executable)


def _load_report(report_dir: Path) -> dict[str, Any]:
    with (report_dir / "report.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _plate_ids(report: dict[str, Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name, result in report.get("imports", {}).items():
        out[name] = [int(pid) for pid in result.get("plate_ids", [])]
    return out


def _first_imported_plate_id(report: dict[str, Any]) -> int:
    for result in report.get("imports", {}).values():
        plate_ids = result.get("plate_ids", [])
        if plate_ids:
            return int(plate_ids[0])
    raise RuntimeError("Probe report did not contain an imported plate ID.")


def _created_screen(args: argparse.Namespace, run_dir: Path) -> int:
    code = r"""
import os
from omero.gateway import BlitzGateway
from omero.model import ScreenI
from omero.rtypes import rstring

user = os.environ.get("OMERO_USER", "root")
password = os.environ.get("OMERO_PASSWORD", "omero")
host = os.environ.get("OMERO_HOST", "omeroserver")
port = int(os.environ.get("OMERO_PORT", "4064"))
group = os.environ.get("OMERO_GROUP", "system")
name = os.environ["PROBE_SCREEN_NAME"]
conn = BlitzGateway(user, password, host=host, port=port)
conn.connect()
try:
    conn.setGroupForSession(group)
except Exception:
    pass
screen = ScreenI()
screen.setName(rstring(name))
screen = conn.getUpdateService().saveAndReturnObject(screen)
print(screen.getId().getValue())
conn.close()
"""
    name = f"CIDeconvolve HCS plate wrapper roundtrip {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    proc = _run(
        [
            "docker",
            "exec",
            "-e",
            f"PROBE_SCREEN_NAME={name}",
            "-e",
            f"OMERO_GROUP={args.group}",
            args.importer_container,
            "python",
            "-c",
            code,
        ],
        log=run_dir / "logs" / "create_screen.log",
    )
    return int((proc.stdout or "").strip().splitlines()[-1])


def _shared_mount(importer_container: str) -> tuple[Path, str]:
    proc = _run([
        "docker",
        "inspect",
        importer_container,
        "--format",
        "{{json .Mounts}}",
    ])
    mounts = json.loads(proc.stdout)
    for mount in mounts:
        if mount.get("Type") != "bind":
            continue
        destination = str(mount.get("Destination") or "").replace("\\", "/")
        if destination != "/data":
            continue
        source = str(mount.get("Source") or "")
        if source.startswith("/run/desktop/mnt/host/"):
            tail = source[len("/run/desktop/mnt/host/"):]
            drive, rest = tail.split("/", 1)
            rest_windows = rest.replace("/", "\\")
            source = f"{drive.upper()}:\\{rest_windows}"
        return Path(source), destination
    raise RuntimeError(f"No /data bind mount found on {importer_container}.")


def _replace_copy(src: Path, dest: Path) -> Path:
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    return dest


def _copy_to_shared(src: Path, shared_root: Path, run_id: str, label: str) -> Path:
    return _replace_copy(src, shared_root / DEFAULT_SHARED_SUBDIR / run_id / label / src.name)


def _probe_cmd(args: argparse.Namespace, *extra: str) -> list[str]:
    return [
        _python(args),
        str(ROOT / "tools" / "omero_import_metadata_probe" / "omero_import_metadata_probe.py"),
        "--importer-container",
        args.importer_container,
        "--user",
        args.user,
        "--group",
        args.group,
        *extra,
    ]


def _run_probe(args: argparse.Namespace, report_dir: Path, *extra: str) -> dict[str, Any]:
    _run(
        _probe_cmd(args, *extra, "--out", str(report_dir)),
        log=report_dir / "probe_command.log",
    )
    return _load_report(report_dir)


def _validate_screen_target(target: str) -> str:
    text = str(target or "").strip()
    if not text.lower().startswith("screen:"):
        raise ValueError("Plate round-trip target must be a Screen, for example Screen:123.")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the cellsA1B1 HCS plate OMERO -> wrapper -> OMERO round-trip test.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source HCS OME-Zarr plate.")
    parser.add_argument("--existing-plate", type=int, help="Use an existing source OMERO Plate ID in the summary instead of importing --source first.")
    parser.add_argument("--target", help="Existing target Screen:ID. If omitted, a new Screen is created.")
    parser.add_argument("--source-import-mode", choices=["direct", "biomero", "both"], default="direct", help="How to import the source plate before the wrapper run.")
    parser.add_argument("--importer-container", default="deployment_scenarios-biomero-importer-1", help="Already-running biomero-importer container name.")
    parser.add_argument("--user", default="root")
    parser.add_argument("--group", default="system")
    parser.add_argument("--iterations", default="5", help="Wrapper iterations for the HCS plate run.")
    parser.add_argument("--method", default="ci_rl")
    parser.add_argument("--cleanup-imports", choices=["always", "success", "never"], default="never", help="Cleanup mode for wrapper output imports.")
    parser.add_argument("--cleanup-source", choices=["always", "success", "never"], default="never", help="Cleanup mode for optional source import.")
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.report_root) / f"cellsa1b1_plate_roundtrip_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    target = _validate_screen_target(args.target) if args.target else f"Screen:{_created_screen(args, run_dir)}"
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    shared_root, _container_root = _shared_mount(args.importer_container)
    source_shared = _copy_to_shared(source, shared_root, run_id, "source")

    source_plate_id = int(args.existing_plate) if args.existing_plate else None
    source_import_report = None
    if source_plate_id is None:
        source_import_report = _run_probe(
            args,
            run_dir / "01_source_plate_import",
            "--input",
            str(source_shared),
            "--target",
            target,
            "--mode",
            args.source_import_mode,
            "--cleanup",
            args.cleanup_source,
        )
        source_plate_id = _first_imported_plate_id(source_import_report)

    wrapper_root = run_dir / "02_wrapper"
    input_dir = wrapper_root / "input"
    output_dir = wrapper_root / "out_plate"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_source = _replace_copy(source, input_dir / source.name)

    wrapper_cmd = [
        _python(args),
        str(ROOT / "wrapper.py"),
        "--local",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--method",
        args.method,
        "--iterations",
        str(args.iterations),
        "--output_format",
        "ome-zarr",
        "--streaming",
        "never",
        "--device",
        "auto",
        "--projection",
        "none",
    ]
    _run(wrapper_cmd, log=run_dir / "logs" / "wrapper_plate.log")

    plate_outputs = sorted(output_dir.glob("*.ome.zarr"))
    if not plate_outputs:
        raise RuntimeError(f"No HCS OME-Zarr plate output found in {output_dir}")
    plate_output = plate_outputs[0]
    plate_output_shared = _copy_to_shared(plate_output, shared_root, run_id, "wrapper_output")

    output_import_report = _run_probe(
        args,
        run_dir / "03_output_plate_import",
        "--input",
        str(plate_output_shared),
        "--target",
        target,
        "--mode",
        "both",
        "--cleanup",
        args.cleanup_imports,
    )

    summary = {
        "run_id": run_id,
        "target": target,
        "source": str(source),
        "source_shared": str(source_shared),
        "source_plate_id": source_plate_id,
        "source_imported_plate_ids": _plate_ids(source_import_report) if source_import_report else None,
        "wrapper_input": str(staged_source),
        "wrapper_output": str(plate_output),
        "wrapper_output_shared": str(plate_output_shared),
        "output_imported_plate_ids": _plate_ids(output_import_report),
        "reports": {
            "source_import": str(run_dir / "01_source_plate_import") if source_import_report else None,
            "output_import": str(run_dir / "03_output_plate_import"),
        },
    }
    (run_dir / "roundtrip_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "roundtrip_summary.md").write_text(
        "# CIDeconvolve HCS Plate OMERO Wrapper Round Trip\n\n"
        f"- Target: `{target}`\n"
        f"- Source plate ID: `{source_plate_id}`\n"
        f"- Source shared path: `{source_shared}`\n"
        f"- Wrapper output: `{plate_output}`\n"
        f"- Wrapper output shared path: `{plate_output_shared}`\n"
        f"- Output imported plate IDs: `{summary['output_imported_plate_ids']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nPlate round-trip report: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
