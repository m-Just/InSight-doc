#!/usr/bin/env python3
"""Cap unanswerable rows to <=25% within each main source."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SOURCE_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_"
    "per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced"
    "-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_"
    "per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced_source_u25"
    "-insight_qwen_agent.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--unanswerable-denominator", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extra_info(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("extra_info")
    return value if isinstance(value, dict) else {}


def extra_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    return extra_info(row).get(key, row.get(key, default))


def question_id(row: dict[str, Any]) -> str:
    return str(extra_value(row, "question_id", ""))


def source_bucket(row: dict[str, Any]) -> str:
    subset = str(extra_value(row, "subset", ""))
    document_id = str(extra_value(row, "document_id", ""))
    if subset in {"veqa", "mveqa"} or "arxiv" in document_id:
        return "arxiv"
    if subset == "bigpage_map":
        return "map"
    if subset == "bigpage_poster":
        return "poster"
    if subset == "bigpage_info":
        return "info"
    return subset or "<missing>"


def is_unanswerable(row: dict[str, Any]) -> bool:
    data_source = str(row.get("data_source") or extra_value(row, "data_source", ""))
    if data_source == "insight_doc_rl_synthetic_unanswerable_vr2_wrong":
        return True
    if data_source == "insight_doc_rl_answerable":
        return False
    question_type = extra_value(row, "question_type")
    return "not-answerable" in json.dumps(question_type, ensure_ascii=False).lower()


def stable_key(row: dict[str, Any], seed: int) -> str:
    key = f"{seed}:{question_id(row)}:{extra_info(row).get('index', '')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def counter_by(rows: list[dict[str, Any]], fn) -> dict[str, int]:
    return dict(sorted(Counter(str(fn(row)) for row in rows).items()))


def main() -> int:
    args = parse_args()
    if args.unanswerable_denominator <= 0:
        raise ValueError("--unanswerable-denominator must be positive")

    source_parquet = args.source_parquet.expanduser().resolve()
    output_parquet = args.output_parquet.expanduser().resolve()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source_parquet)
    rows = df.to_dict("records")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[source_bucket(row)].append(row)

    selected_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    per_source_summary: dict[str, dict[str, Any]] = {}

    for source in sorted(by_source):
        source_rows = by_source[source]
        answerable = [row for row in source_rows if not is_unanswerable(row)]
        unanswerable = [row for row in source_rows if is_unanswerable(row)]
        max_unanswerable = len(answerable) // args.unanswerable_denominator
        selected_unanswerable = sorted(unanswerable, key=lambda row: stable_key(row, args.seed))[:max_unanswerable]
        selected_ids = {id(row) for row in answerable}
        selected_ids.update(id(row) for row in selected_unanswerable)
        selected_source_rows = [row for row in source_rows if id(row) in selected_ids]
        dropped_source_rows = [row for row in source_rows if id(row) not in selected_ids]
        selected_rows.extend(selected_source_rows)
        dropped_rows.extend(dropped_source_rows)
        per_source_summary[source] = {
            "before_total": len(source_rows),
            "before_answerable": len(answerable),
            "before_unanswerable": len(unanswerable),
            "selected_total": len(selected_source_rows),
            "selected_answerable": len(answerable),
            "selected_unanswerable": len(selected_unanswerable),
            "selected_unanswerable_fraction": (
                len(selected_unanswerable) / len(selected_source_rows) if selected_source_rows else None
            ),
            "dropped_unanswerable": len(dropped_source_rows),
        }

    selected_id_set = {id(row) for row in selected_rows}
    selected_rows_ordered = [row for row in rows if id(row) in selected_id_set]
    pd.DataFrame(selected_rows_ordered, columns=df.columns).to_parquet(output_parquet, index=False)

    dropped_path = output_parquet.with_suffix(".dropped_unanswerable_question_ids.txt")
    dropped_path.write_text("".join(f"{question_id(row)}\n" for row in dropped_rows), encoding="utf-8")

    summary = {
        "source_parquet": str(source_parquet),
        "output_parquet": str(output_parquet),
        "dropped_unanswerable_question_ids": str(dropped_path),
        "seed": args.seed,
        "unanswerable_denominator": args.unanswerable_denominator,
        "selection_note": "Keep all answerable rows; cap unanswerable rows per main source at floor(answerable / denominator).",
        "rows_source": len(rows),
        "rows": len(selected_rows_ordered),
        "rows_dropped": len(dropped_rows),
        "source_counts": counter_by(selected_rows_ordered, source_bucket),
        "data_source_counts": counter_by(
            selected_rows_ordered, lambda row: row.get("data_source") or extra_value(row, "data_source", "")
        ),
        "initial_rescale_counts": counter_by(selected_rows_ordered, lambda row: extra_info(row).get("initial_rescale")),
        "unanswerable_rows": sum(is_unanswerable(row) for row in selected_rows_ordered),
        "unanswerable_fraction": (
            sum(is_unanswerable(row) for row in selected_rows_ordered) / len(selected_rows_ordered)
            if selected_rows_ordered
            else None
        ),
        "per_source": per_source_summary,
    }
    summary_path = output_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
