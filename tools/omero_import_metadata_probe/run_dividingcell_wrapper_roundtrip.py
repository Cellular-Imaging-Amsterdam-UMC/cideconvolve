"""Rerunnable OMERO/BIOMERO wrapper round-trip test for CIDeconvolve.

This script intentionally does not start Docker Compose. It assumes the local
NL-BIOMERO stack is already running and uses the running biomero-importer
container to talk to OMERO, matching the manual metadata test.
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
DEFAULT_SOURCE = ROOT / "localdata" / "DividingCellcrop.ome.tiff"
DEFAULT_REPORT_ROOT = Path.home() / "Downloads" / "cideconvolve_omero_roundtrips"
DEFAULT_SHARED_SUBDIR = "cideconvolve_probe/dividing_roundtrip_rerun"


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


def _first_imported_image_id(report: dict[str, Any]) -> int:
    for result in report.get("imports", {}).values():
        images = result.get("objects", {}).get("images", [])
        if images:
            return int(images[0]["id"])
    raise RuntimeError("Probe report did not contain an imported image ID.")


def _image_ids(report: dict[str, Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for name, result in report.get("imports", {}).items():
        out[name] = [int(img["id"]) for img in result.get("objects", {}).get("images", [])]
    return out


def _created_dataset(args: argparse.Namespace, run_dir: Path) -> int:
    code = r"""
import os
from omero.gateway import BlitzGateway
from omero.model import DatasetI
from omero.rtypes import rstring

user = os.environ.get("OMERO_USER", "root")
password = os.environ.get("OMERO_PASSWORD", "omero")
host = os.environ.get("OMERO_HOST", "omeroserver")
port = int(os.environ.get("OMERO_PORT", "4064"))
group = os.environ.get("OMERO_GROUP", "system")
name = os.environ["PROBE_DATASET_NAME"]
conn = BlitzGateway(user, password, host=host, port=port)
conn.connect()
try:
    conn.setGroupForSession(group)
except Exception:
    pass
ds = DatasetI()
ds.setName(rstring(name))
ds = conn.getUpdateService().saveAndReturnObject(ds)
print(ds.getId().getValue())
conn.close()
"""
    name = f"CIDeconvolve wrapper OMERO roundtrip {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    cmd = [
        "docker",
        "exec",
        "-e",
        f"PROBE_DATASET_NAME={name}",
        "-e",
        f"OMERO_GROUP={args.group}",
        args.importer_container,
        "python",
        "-c",
        code,
    ]
    proc = _run(cmd, log=run_dir / "logs" / "create_dataset.log")
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


def _copy_output_to_shared(src: Path, shared_root: Path, run_id: str) -> Path:
    dest_dir = shared_root / DEFAULT_SHARED_SUBDIR / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    return dest


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DividingCell OMERO -> Slurm Zarr -> wrapper -> OMERO round-trip test.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source OME-TIFF to import when --existing-image is not given.")
    parser.add_argument("--existing-image", type=int, help="Use an existing OMERO Image ID instead of importing --source first.")
    parser.add_argument("--target", help="Existing target such as Dataset:152. If omitted, a new Dataset is created.")
    parser.add_argument("--importer-container", default="deployment_scenarios-biomero-importer-1", help="Already-running biomero-importer container name.")
    parser.add_argument("--user", default="root")
    parser.add_argument("--group", default="system")
    parser.add_argument("--iterations", default="5", help="Wrapper iterations for both stack and MIP runs.")
    parser.add_argument("--method", default="ci_rl")
    parser.add_argument("--cleanup-imports", choices=["always", "success", "never"], default="never", help="Cleanup mode for output imports.")
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.report_root) / f"dividingcell_roundtrip_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    target = args.target
    if not target:
        target = f"Dataset:{_created_dataset(args, run_dir)}"

    source_image_id = int(args.existing_image) if args.existing_image else None
    source_import_report = None
    if source_image_id is None:
        source_import_report = _run_probe(
            args,
            run_dir / "01_source_import",
            "--input",
            str(Path(args.source)),
            "--target",
            target,
            "--mode",
            "direct",
            "--cleanup",
            "never",
        )
        source_image_id = _first_imported_image_id(source_import_report)

    slurm_report = _run_probe(
        args,
        run_dir / "02_slurm_input_export",
        "--slurm-input-image",
        str(source_image_id),
    )
    slurm_zarr = Path(slurm_report["slurm_input_export"]["local_zarr_path"])

    wrapper_root = run_dir / "03_wrapper"
    input_dir = wrapper_root / "input"
    stack_out = wrapper_root / "out_stack"
    mip_out = wrapper_root / "out_mip"
    input_dir.mkdir(parents=True, exist_ok=True)
    stack_out.mkdir(parents=True, exist_ok=True)
    mip_out.mkdir(parents=True, exist_ok=True)
    staged_zarr = input_dir / slurm_zarr.name
    if staged_zarr.exists():
        shutil.rmtree(staged_zarr)
    shutil.copytree(slurm_zarr, staged_zarr)

    common_wrapper = [
        _python(args),
        str(ROOT / "wrapper.py"),
        "--local",
        "--input-dir",
        str(input_dir),
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
    ]
    _run(common_wrapper + ["--output-dir", str(stack_out), "--projection", "none"], log=run_dir / "logs" / "wrapper_stack.log")
    _run(common_wrapper + ["--output-dir", str(mip_out), "--projection", "mip"], log=run_dir / "logs" / "wrapper_mip.log")

    stack_outputs = sorted(stack_out.glob("*.ome.zarr"))
    mip_outputs = sorted(mip_out.glob("*.ome.zarr"))
    if not stack_outputs:
        raise RuntimeError(f"No stack OME-Zarr output found in {stack_out}")
    if not mip_outputs:
        raise RuntimeError(f"No MIP OME-Zarr output found in {mip_out}")

    shared_root, _container_root = _shared_mount(args.importer_container)
    stack_shared = _copy_output_to_shared(stack_outputs[0], shared_root, run_id)
    mip_shared = _copy_output_to_shared(mip_outputs[0], shared_root, run_id)

    stack_import_report = _run_probe(
        args,
        run_dir / "04_stack_output_import",
        "--input",
        str(stack_shared),
        "--target",
        target,
        "--mode",
        "both",
        "--cleanup",
        args.cleanup_imports,
    )
    mip_import_report = _run_probe(
        args,
        run_dir / "05_mip_output_import",
        "--input",
        str(mip_shared),
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
        "source": str(Path(args.source)),
        "source_image_id": source_image_id,
        "slurm_input_zarr": str(slurm_zarr),
        "stack_output": str(stack_outputs[0]),
        "mip_output": str(mip_outputs[0]),
        "stack_shared": str(stack_shared),
        "mip_shared": str(mip_shared),
        "stack_imported_image_ids": _image_ids(stack_import_report),
        "mip_imported_image_ids": _image_ids(mip_import_report),
        "reports": {
            "source_import": str(run_dir / "01_source_import") if source_import_report else None,
            "slurm_input_export": str(run_dir / "02_slurm_input_export"),
            "stack_import": str(run_dir / "04_stack_output_import"),
            "mip_import": str(run_dir / "05_mip_output_import"),
        },
    }
    (run_dir / "roundtrip_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "roundtrip_summary.md").write_text(
        "# CIDeconvolve OMERO Wrapper Round Trip\n\n"
        f"- Target: `{target}`\n"
        f"- Source OMERO image: `{source_image_id}`\n"
        f"- Slurm-input Zarr: `{slurm_zarr}`\n"
        f"- Stack output: `{stack_outputs[0]}`\n"
        f"- MIP output: `{mip_outputs[0]}`\n"
        f"- Stack imported image IDs: `{summary['stack_imported_image_ids']}`\n"
        f"- MIP imported image IDs: `{summary['mip_imported_image_ids']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"\nRound-trip report: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
