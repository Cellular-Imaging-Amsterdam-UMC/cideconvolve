from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import torch


def main() -> int:
    cuda_home = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "")))
    nvcc = cuda_home / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    if not nvcc.is_file():
        print(f"ERROR: CUDA compiler not found at {nvcc}", file=sys.stderr)
        return 2
    output = subprocess.check_output([str(nvcc), "--version"], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"release\s+(\d+\.\d+)", output)
    toolkit = match.group(1) if match else None
    runtime = torch.version.cuda
    if toolkit != runtime:
        print(
            f"ERROR: CUDA Toolkit {toolkit or 'unknown'} does not match "
            f"PyTorch {torch.__version__} CUDA runtime {runtime or 'none'}.",
            file=sys.stderr,
        )
        return 3
    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot access a CUDA GPU", file=sys.stderr)
        return 4
    capability = ".".join(str(part) for part in torch.cuda.get_device_capability(0))
    requested = os.environ.get("TORCH_CUDA_ARCH_LIST", capability)
    print(f"PyTorch: {torch.__version__} (CUDA {runtime})")
    print(f"Toolkit: {toolkit} at {cuda_home}")
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{capability.replace('.', '')})")
    print(f"PyTorch wheel architectures: {' '.join(torch.cuda.get_arch_list())}")
    print(f"Extension architectures: {requested}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
