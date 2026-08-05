#!/usr/bin/env python3
"""
Pack an exported-conversations directory and an image directory into a single tar file.

The archive layout is always:
  - exported_conversations/
  - images/
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exported-conversations-dir", required=True, help="Directory containing exported JSON files.")
    parser.add_argument(
        "--image-dir",
        default=None,
        help=(
            "Directory containing the source image tree. If omitted, infer it from "
            "file:// refs in exported conversations."
        ),
    )
    parser.add_argument("--output-tar", required=True, help="Output tar or tar.gz path.")
    return parser.parse_args()


def validate_input_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def tar_mode_for_path(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "w:gz"
    if name.endswith(".tar"):
        return "w"
    raise ValueError(f"Unsupported output tar path {path}; expected .tar, .tar.gz, or .tgz")


def iter_export_paths(exported_conversations_dir: Path) -> list[Path]:
    paths = sorted(exported_conversations_dir.rglob("*.json"))
    if not paths:
        raise ValueError(f"No JSON files found under {exported_conversations_dir}")
    return paths


def local_file_paths_from_record(record: dict) -> list[Path]:
    paths: list[Path] = []
    image_references = record.get("image_references")
    if not isinstance(image_references, dict):
        return paths
    input_images = image_references.get("input_images")
    if not isinstance(input_images, list):
        return paths
    for ref in input_images:
        if not isinstance(ref, dict):
            continue
        value = ref.get("value")
        if not isinstance(value, str) or not value.startswith("file://"):
            continue
        parsed = urlparse(value)
        if parsed.scheme != "file":
            continue
        paths.append(Path(parsed.path).resolve())
    return paths


def infer_image_dir(exported_conversations_dir: Path) -> Path:
    all_paths: list[Path] = []
    for export_path in iter_export_paths(exported_conversations_dir):
        with export_path.open(encoding="utf-8") as f:
            record = json.load(f)
        all_paths.extend(local_file_paths_from_record(record))
    if not all_paths:
        raise ValueError(
            "Could not infer image dir: no local file:// refs found in exported conversations"
        )

    common_path = Path(os.path.commonpath([str(path) for path in all_paths]))
    if common_path.is_file():
        common_path = common_path.parent
    if not common_path.exists() or not common_path.is_dir():
        raise ValueError(f"Inferred image dir is not a directory: {common_path}")
    return common_path


def main() -> None:
    args = parse_args()
    exported_conversations_dir = validate_input_dir(
        Path(args.exported_conversations_dir), "exported conversations directory"
    )
    if args.image_dir is None:
        image_dir = infer_image_dir(exported_conversations_dir)
        print(f"Inferred image dir: {image_dir}")
    else:
        image_dir = validate_input_dir(Path(args.image_dir), "image directory")
    output_tar = Path(args.output_tar).expanduser().resolve()
    output_tar.parent.mkdir(parents=True, exist_ok=True)

    mode = tar_mode_for_path(output_tar)
    with tarfile.open(output_tar, mode) as tar:
        tar.add(exported_conversations_dir, arcname="exported_conversations")
        tar.add(image_dir, arcname="images")

    print(f"Packed {exported_conversations_dir} as exported_conversations/")
    print(f"Packed {image_dir} as images/")
    print(f"Wrote archive to {output_tar}")


if __name__ == "__main__":
    main()
