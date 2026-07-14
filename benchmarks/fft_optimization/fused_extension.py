from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _safe_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _toolkit_version(cuda_home: Path) -> str | None:
    nvcc = cuda_home / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    if not nvcc.is_file():
        return None
    output = subprocess.check_output([str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"release\s+(\d+\.\d+)", output)
    return match.group(1) if match else None


def load_fused_extension(*, verbose: bool = False) -> tuple[Any | None, str | None]:
    try:
        from core.optimized_cuda import load_optimized_extension

        return load_optimized_extension(required=True, verbose=verbose), None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, str(exc)


def lto_callback_capability() -> tuple[bool, str]:
    cuda_home = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "")))
    header = cuda_home / "include" / "cufftXt.h"
    if not header.is_file():
        return False, f"missing {header}"
    text = header.read_text(encoding="utf-8", errors="ignore")
    if "cufftXtSetJITCallback" not in text:
        return False, "cufftXtSetJITCallback is absent from the installed cufftXt.h"
    return True, "cufftXtSetJITCallback is available"
