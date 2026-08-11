from __future__ import annotations

import argparse
from pathlib import Path
import re


SEMVER = re.compile(
    r"\Av?(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?\Z"
)
TAG_LINE = re.compile(
    r"^(?P<prefix>[ \t]+tag:[ \t]*)(?P<value>\S+)(?P<suffix>[ \t]*)$"
)


def sync_config_version(
    config_path: Path, version: str, *, check: bool = False
) -> bool:
    """Synchronize ``docker_image.tag`` and return whether a change is needed."""
    if not SEMVER.fullmatch(version):
        raise ValueError(f"Invalid release version: {version}")

    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    in_docker_image = False
    match_index: int | None = None
    match: re.Match[str] | None = None

    for index, line in enumerate(lines):
        content = line.rstrip("\r\n")
        if not in_docker_image:
            if content == "docker_image:":
                in_docker_image = True
            continue
        if content and not content[0].isspace():
            break
        candidate, _, _comment = content.partition("#")
        candidate_match = TAG_LINE.fullmatch(candidate)
        if candidate_match is None:
            continue
        if match_index is not None:
            raise ValueError("docker_image contains more than one tag")
        match_index = index
        match = candidate_match

    if match_index is None or match is None:
        raise ValueError("Could not find docker_image.tag in config")

    current = match.group("value").strip("'\"")
    if current == version:
        return False
    if check:
        return True

    original = lines[match_index]
    ending = (
        "\r\n"
        if original.endswith("\r\n")
        else "\n"
        if original.endswith("\n")
        else ""
    )
    content = original[: -len(ending)] if ending else original
    _, separator, comment = content.partition("#")
    comment_suffix = f"#{comment}" if separator else ""
    lines[match_index] = (
        f"{match.group('prefix')}{version}{match.group('suffix')}"
        f"{comment_suffix}{ending}"
    )
    with config_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("".join(lines))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize config.yaml docker_image.tag with a release version."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("version")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    try:
        changed = sync_config_version(args.config, args.version, check=args.check)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if not changed:
        print(f"config.yaml Docker tag is already {args.version}.")
        return 0
    if args.check:
        print(f"config.yaml Docker tag must be updated to {args.version}.")
        return 2
    print(f"Updated config.yaml Docker tag to {args.version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())