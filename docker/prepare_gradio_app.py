"""Backward-compatible wrapper for generating the CIDeconvolve Gradio app."""

from __future__ import annotations

import argparse
from pathlib import Path

from prepare_bilayers_interface import (  # noqa: F401
    build_gradio_config,
    generate_interface_artifact,
    patch_gradio_app,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("/app/config.yaml"))
    parser.add_argument("--workdir", type=Path, default=Path("/app"))
    parser.add_argument("--output", type=Path, default=Path("/app/gradio_app.py"))
    args = parser.parse_args(argv)

    generate_interface_artifact(args.config, args.workdir, "gradio", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
