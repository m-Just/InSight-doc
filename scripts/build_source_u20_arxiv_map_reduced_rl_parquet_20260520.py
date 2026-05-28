#!/usr/bin/env python3
"""Further reduce arxiv and map rows from the source-u20 RL parquet."""

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
    "per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced_source_u20"
    "-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_"
    "per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced_source_u20"
    "_arxiv_map_7of8-insight_qwen_agent.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--keep-fraction", type=float, default=7.0 / 8.0)
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


def stable_key(row: dict[str, Any], seed: int) -> str:
    key = f"{seed}:{source_bucket(row)}:{question_id(row)}:{extra_info(row).get('index', '')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def is_unanswerable(row: dict[str, Any]) -> bool:
    data_source = str(row.get("data_source") or extra_value(row, "data_source", ""))
    if data_source == "insight_doc_rl_synthetic_unanswerable_vr2_wrong":
        return True
    if data_source == "insight_doc_rl_answerable":
        return False
    question_type = extra_value(row, "question_type")
    return "not-answerable" in json.dumps(question_type, ensure_ascii=False).lower()


def counter_by(rows: list[dict[str, Any]], fn) -> dict[str, int]:
    return dict(sorted(Counter(str(fn(row)) for row in rows).items()))


def main() -> int:
    args = parse_args()
    if not 0.0 < args.keep_fraction <= 1.0:
        raise ValueError("--keep-fraction must be in (0, 1]")

    source_parquet = args.source_parquet.expanduser().resolve()
    output_parquet = args.output_parquet.expanduser().resolve()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source_parquet)
    rows = df.to_dict("records")
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(source_bucket(row), []).append(row)

    selected_ids: set[int] = set()
    dropped_rows: list[dict[str, Any]] = []
    per_source: dict[str, dict[str, Any]] = {}
    for source, source_rows in sorted(by_source.items()):
        if source in {"arxiv", "map"}:
            keep_n = round(len(source_rows) * args.keep_fraction)
            selected = sorted(source_rows, key=lambda row: stable_key(row, args.seed))[:keep_n]
        else:
            selected = source_rows
        selected_set = {id(row) for row in selected}
        selected_ids.update(selected_set)
        dropped = [row for row in source_rows if id(row) not in selected_set]
        dropped_rows.extend(dropped)
        per_source[source] = {
            "before": len(source_rows),
            "selected": len(selected),
            "dropped": len(dropped),
            "before_unanswerable": sum(is_unanswerable(row) for row in source_rows),
            "selected_unanswerable": sum(is_unanswerable(row) for row in selected),
        }

    selected_rows = [row for row in rows if id(row) in selected_ids]
    pd.DataFrame(selected_rows, columns=df.columns).to_parquet(output_parquet, index=False)

    dropped_path = output_parquet.with_suffix(".dropped_arxiv_map_question_ids.txt")
    dropped_path.write_text("".join(f"{source_bucket(row)}\t{question_id(row)}\n" for row in dropped_rows), encoding="utf-8")

    unanswerable_rows = sum(is_unanswerable(row) for row in selected_rows)
    summary = {
        "source_parquet": str(source_parquet),
        "output_parquet": str(output_parquet),
        "dropped_arxiv_map_question_ids": str(dropped_path),
        "seed": args.seed,
        "keep_fraction_for_arxiv_and_map": args.keep_fraction,
        "rows_source": len(rows),
        "rows": len(selected_rows),
        "rows_dropped": len(dropped_rows),
        "source_counts": counter_by(selected_rows, source_bucket),
        "source_fractions": {
            source: count / len(selected_rows)
            for source, count in counter_by(selected_rows, source_bucket).items()
        },
        "data_source_counts": counter_by(
            selected_rows, lambda row: row.get("data_source") or extra_value(row, "data_source", "")
        ),
        "initial_rescale_counts": counter_by(selected_rows, lambda row: extra_info(row).get("initial_rescale")),
        "unanswerable_rows": unanswerable_rows,
        "unanswerable_fraction": unanswerable_rows / len(selected_rows) if selected_rows else None,
        "per_source": per_source,
    }
    summary_path = output_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
