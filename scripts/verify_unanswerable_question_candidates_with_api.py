#!/usr/bin/env python3
"""Verify synthetic unanswerable question candidates with broader document context."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_unanswerable_core import (
    CANDIDATE_MARKER_NAME as MARKER_NAME,
    CANDIDATE_PROMPT_VERSION as PROMPT_VERSION,
    add_common_verification_args,
    build_accepted_manifest_row,
    build_candidate_prompt as build_prompt,
    has_meaningful_answer_text,
    load_candidates,
    materialize_pdf_image_tree,
    process_candidate,
    query_verification_api,
    run_candidate_verification as run,
    select_verification_pages,
    validate_verification_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-jsonl", required=True)
    add_common_verification_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
