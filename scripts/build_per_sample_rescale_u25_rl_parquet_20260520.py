#!/usr/bin/env python3
"""Build a u25 RL parquet from the per-sample-rescale length-filtered row set.

This mirrors the existing new multiscale final parquet, but uses the row set
selected by each row's extra_info.initial_rescale rather than a uniform 0.5
initial_rescale estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SOURCE_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_"
    "per_sample_rescale_maxarea3500x3500_le11000-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_"
    "per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale-insight_qwen_agent.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--unanswerable-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extra_info_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        return extra_info.get(key, default)
    return default


def question_id(row: dict[str, Any]) -> str:
    return str(extra_info_value(row, "question_id", row.get("question_id", "")))


def is_unanswerable(row: dict[str, Any]) -> bool:
    data_source = str(row.get("data_source") or extra_info_value(row, "data_source", ""))
    if data_source == "insight_doc_rl_synthetic_unanswerable_vr2_wrong":
        return True
    if data_source == "insight_doc_rl_answerable":
        return False
    question_type = extra_info_value(row, "question_type", row.get("question_type"))
    return "not-answerable" in json.dumps(question_type, ensure_ascii=False).lower()


def stable_sample_key(row: dict[str, Any], seed: int) -> str:
    index = extra_info_value(row, "index", "")
    key = f"{seed}:{question_id(row)}:{index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def source_bucket(row: dict[str, Any]) -> str:
    subset = str(extra_info_value(row, "subset", row.get("subset", "")))
    document_id = str(extra_info_value(row, "document_id", row.get("document_id", "")))
    if subset in {"veqa", "mveqa"} or "arxiv" in document_id:
        return "arxiv"
    if subset == "bigpage_map":
        return "map"
    if subset == "bigpage_poster":
        return "poster"
    if subset == "bigpage_info":
        return "info"
    return subset or "<missing>"


def counter_by(rows: list[dict[str, Any]], fn) -> dict[str, int]:
    return dict(sorted(Counter(str(fn(row)) for row in rows).items()))


def first_prompt_role(row: dict[str, Any]) -> str | None:
    prompt = row.get("prompt")
    if prompt is None or len(prompt) == 0:
        return None
    first = prompt[0]
    if isinstance(first, dict):
        return first.get("role")
    return None


def main() -> int:
    args = parse_args()
    if not (0.0 < args.unanswerable_fraction < 1.0):
        raise ValueError("--unanswerable-fraction must be in (0, 1)")

    source_parquet = args.source_parquet.expanduser().resolve()
    output_parquet = args.output_parquet.expanduser().resolve()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source_parquet)
    rows = df.to_dict("records")
    answerable_rows = [row for row in rows if not is_unanswerable(row)]
    unanswerable_rows = [row for row in rows if is_unanswerable(row)]

    max_unanswerable = int(len(answerable_rows) * args.unanswerable_fraction / (1.0 - args.unanswerable_fraction))
    selected_unanswerable = sorted(unanswerable_rows, key=lambda row: stable_sample_key(row, args.seed))[
        :max_unanswerable
    ]
    selected_ids = {id(row) for row in answerable_rows}
    selected_ids.update(id(row) for row in selected_unanswerable)
    selected_rows = [row for row in rows if id(row) in selected_ids]
    dropped_unanswerable = [row for row in unanswerable_rows if id(row) not in selected_ids]

    selected_df = pd.DataFrame(selected_rows, columns=df.columns)
    selected_df.to_parquet(output_parquet, index=False)

    dropped_qids_path = output_parquet.with_suffix(".dropped_unanswerable_question_ids.txt")
    dropped_qids_path.write_text(
        "".join(f"{question_id(row)}\n" for row in dropped_unanswerable),
        encoding="utf-8",
    )

    system_rows = sum(1 for row in selected_rows if first_prompt_role(row) == "system")
    summary = {
        "source_parquet": str(source_parquet),
        "output_parquet": str(output_parquet),
        "dropped_unanswerable_question_ids": str(dropped_qids_path),
        "selection_note": (
            "Rows first filtered by extra_info.initial_rescale prompt-token estimate; "
            "then all answerable rows are kept and synthetic unanswerable rows are sampled to u25."
        ),
        "seed": args.seed,
        "target_unanswerable_fraction": args.unanswerable_fraction,
        "rows_source": len(rows),
        "source_answerable_rows": len(answerable_rows),
        "source_unanswerable_rows": len(unanswerable_rows),
        "rows": len(selected_rows),
        "answerable_rows": len(answerable_rows),
        "unanswerable_rows": len(selected_unanswerable),
        "unanswerable_fraction": len(selected_unanswerable) / len(selected_rows) if selected_rows else None,
        "dropped_unanswerable_rows": len(dropped_unanswerable),
        "system_rows": system_rows,
        "rows_with_initial_rescale": sum(
            1 for row in selected_rows if extra_info_value(row, "initial_rescale") is not None
        ),
        "data_source_counts": counter_by(
            selected_rows, lambda row: row.get("data_source") or extra_info_value(row, "data_source", "")
        ),
        "initial_rescale_counts": counter_by(selected_rows, lambda row: extra_info_value(row, "initial_rescale")),
        "initial_rescale_dpi_counts": counter_by(selected_rows, lambda row: extra_info_value(row, "initial_rescale_dpi")),
        "source_counts": counter_by(selected_rows, source_bucket),
        "answerable_source_counts": counter_by(answerable_rows, source_bucket),
        "unanswerable_source_counts": counter_by(selected_unanswerable, source_bucket),
    }
    summary_path = output_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
