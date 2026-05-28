#!/usr/bin/env python3
"""Verify externally supplied manifest questions for unanswerability."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_unanswerable_core import (
    MANIFEST_MARKER_NAME as MARKER_NAME,
    MANIFEST_PROMPT_VERSION as PROMPT_VERSION,
    add_common_verification_args,
    build_manifest_accepted_manifest_row,
    build_manifest_prompt as build_prompt,
    load_manifest_rows,
    normalize_manifest_row,
    process_manifest_row,
    run_manifest_verification as run,
    select_verification_pages,
    validate_verification_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Input manifest.jsonl containing questions to verify.")
    add_common_verification_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
