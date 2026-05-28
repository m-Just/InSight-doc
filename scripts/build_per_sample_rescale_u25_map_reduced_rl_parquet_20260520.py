#!/usr/bin/env python3
"""Build a map-reduced derivative of the per-sample-rescale u25 RL parquet."""

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
    "per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_PARQUET = Path(
    "/scratch/ywxzml3j/likaican/data/insight_doc/"
    "answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_"
    "per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced"
    "-insight_qwen_agent.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--travelmap-keep-fraction", type=float, default=0.5)
    parser.add_argument("--metromap-keep-fraction", type=float, default=0.75)
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


def map_kind(row: dict[str, Any]) -> str | None:
    if str(extra_value(row, "subset", "")) != "bigpage_map":
        return None
    text = f"{extra_value(row, 'document_id', '')} {question_id(row)}"
    if "travelmap" in text:
        return "travelmap"
    if "metromap" in text:
        return "metromap"
    return "other_map"


def stable_key(row: dict[str, Any], seed: int) -> str:
    key = f"{seed}:{question_id(row)}:{extra_info(row).get('index', '')}"
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
    source_parquet = args.source_parquet.expanduser().resolve()
    output_parquet = args.output_parquet.expanduser().resolve()
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(source_parquet)
    rows = df.to_dict("records")

    by_kind: dict[str | None, list[dict[str, Any]]] = {None: [], "travelmap": [], "metromap": [], "other_map": []}
    for row in rows:
        by_kind.setdefault(map_kind(row), []).append(row)

    keep_travelmap = round(len(by_kind["travelmap"]) * args.travelmap_keep_fraction)
    keep_metromap = round(len(by_kind["metromap"]) * args.metromap_keep_fraction)
    selected_travelmap = sorted(by_kind["travelmap"], key=lambda row: stable_key(row, args.seed))[:keep_travelmap]
    selected_metromap = sorted(by_kind["metromap"], key=lambda row: stable_key(row, args.seed))[:keep_metromap]

    selected_ids = {id(row) for row in by_kind[None]}
    selected_ids.update(id(row) for row in by_kind["other_map"])
    selected_ids.update(id(row) for row in selected_travelmap)
    selected_ids.update(id(row) for row in selected_metromap)
    selected_rows = [row for row in rows if id(row) in selected_ids]
    dropped_rows = [row for row in rows if id(row) not in selected_ids]

    pd.DataFrame(selected_rows, columns=df.columns).to_parquet(output_parquet, index=False)

    dropped_path = output_parquet.with_suffix(".dropped_map_question_ids.txt")
    dropped_path.write_text("".join(f"{question_id(row)}\n" for row in dropped_rows), encoding="utf-8")

    map_rows = [row for row in selected_rows if source_bucket(row) == "map"]
    summary = {
        "source_parquet": str(source_parquet),
        "output_parquet": str(output_parquet),
        "dropped_map_question_ids": str(dropped_path),
        "seed": args.seed,
        "travelmap_keep_fraction": args.travelmap_keep_fraction,
        "metromap_keep_fraction": args.metromap_keep_fraction,
        "rows_source": len(rows),
        "rows": len(selected_rows),
        "rows_dropped": len(dropped_rows),
        "source_map_rows_before": len(by_kind["travelmap"]) + len(by_kind["metromap"]) + len(by_kind["other_map"]),
        "map_rows": len(map_rows),
        "map_fraction": len(map_rows) / len(selected_rows) if selected_rows else None,
        "map_kind_counts_before": {
            "travelmap": len(by_kind["travelmap"]),
            "metromap": len(by_kind["metromap"]),
            "other_map": len(by_kind["other_map"]),
        },
        "map_kind_counts_after": counter_by(map_rows, map_kind),
        "source_counts": counter_by(selected_rows, source_bucket),
        "data_source_counts": counter_by(
            selected_rows, lambda row: row.get("data_source") or extra_value(row, "data_source", "")
        ),
        "initial_rescale_counts": counter_by(selected_rows, lambda row: extra_info(row).get("initial_rescale")),
        "unanswerable_rows": sum(is_unanswerable(row) for row in selected_rows),
        "unanswerable_fraction": (
            sum(is_unanswerable(row) for row in selected_rows) / len(selected_rows) if selected_rows else None
        ),
    }
    summary_path = output_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
