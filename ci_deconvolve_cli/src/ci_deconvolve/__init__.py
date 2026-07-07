"""Focused command-line package for CI-RL deconvolution."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_version() -> str:
    source_version = Path(__file__).resolve().parents[2] / "version.txt"
    if source_version.is_file():
        return source_version.read_text(encoding="utf-8").strip()
    try:
        return version("ci-deconvolve")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _read_version()
