#!/usr/bin/env python3
"""Build old-marginal RL ablation parquets with capped replacement.

The output keeps the old final RL parquet's subset counts and per-subset
answerability counts exactly. Within each subset, up to 30% of rows are replaced
by rows from an easy/super-easy candidate pool. If a subset has fewer added
candidates than the cap, all available candidates are used.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from build_old_rl_easy_ablation_parquets_20260713 import (
    OUTPUT_DATA_ROOT,
    OLD_FINAL_PARQUET,
    add_stage_to_record,
    build_source_stage_map,
    parquet_qid,
    parquet_subset,
    parquet_unanswerable,
    row_extra_info,
    stable_key,
    summarize_parquet,
)


BUILD_ROOT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_rl_easy_super_easy_ablation_20260713"
)


VARIANTS: dict[str, dict[str, Any]] = {
    "plus_easy32b": {
        "source_label": "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b",
        "output_label": "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_replace30cap_oldmarginals",
        "stage_priority": ["easy32b"],
        "source_suffix": ".parquet",
    },
    "plus_easy32b_plus_super_easy8b": {
        "source_label": "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_plus_super_easy8b",
        "output_label": (
            "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_plus_super_easy8b"
            "_replace30cap_oldmarginals"
        ),
        "stage_priority": ["super_easy8b", "easy32b"],
        "source_suffix": ".parquet",
    },
    "plus_easy32b_tmp": {
        "source_label": "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b",
        "output_label": "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_replace20cap_oldmarginals_from_tmp",
        "stage_priority": ["easy32b"],
        "source_suffix": ".estprompt_le11000.tmp.parquet",
    },
    "plus_easy32b_plus_super_easy8b_tmp": {
        "source_label": "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_plus_super_easy8b",
        "output_label": (
            "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_plus_super_easy8b"
            "_replace20cap_oldmarginals_from_tmp"
        ),
        "stage_priority": ["super_easy8b", "easy32b"],
        "source_suffix": ".estprompt_le11000.tmp.parquet",
    },
}


def answerability_key(record: dict[str, Any]) -> str:
    return "unanswerable" if parquet_unanswerable(record) else "answerable"


def annotate_old_record(record: dict[str, Any], variant_key: str) -> dict[str, Any]:
    out = dict(record)
    extra = dict(row_extra_info(out))
    extra["rl_source_stage"] = "baseline_medium_hard"
    extra["rl_source_variant"] = variant_key
    out["extra_info"] = extra
    out["data_source"] = "insight_doc_rl"
    return out


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: stable_key(parquet_qid(record)))


def allocate_count(total: int, caps: dict[str, int], weights: dict[str, float]) -> dict[str, int]:
    """Allocate total across keys, respecting caps and approximately weights."""
    keys = sorted(caps)
    alloc = {key: 0 for key in keys}
    remaining = min(total, sum(caps.values()))
    active = [key for key in keys if caps[key] > 0 and weights.get(key, 0.0) > 0]
    if not active:
        return alloc

    weight_sum = sum(weights[key] for key in active)
    desired = {key: remaining * weights[key] / weight_sum for key in active}
    for key in active:
        take = min(caps[key], int(math.floor(desired[key])))
        alloc[key] = take
    leftover = remaining - sum(alloc.values())

    # Largest-remainder fill first, then any remaining capacity.
    remainder_order = sorted(
        active,
        key=lambda key: (desired[key] - math.floor(desired[key]), weights[key], key),
        reverse=True,
    )
    while leftover > 0:
        progressed = False
        for key in remainder_order:
            if leftover <= 0:
                break
            if alloc[key] < caps[key]:
                alloc[key] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def select_added_by_stage(
    records: list[dict[str, Any]],
    count: int,
    *,
    stage_priority: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if count <= 0:
        return [], {}

    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stage[str(row_extra_info(record).get("rl_source_stage", "unknown"))].append(record)
    for stage in list(by_stage):
        by_stage[stage] = sort_records(by_stage[stage])

    chosen: list[dict[str, Any]] = []
    remaining = count
    for stage in list(stage_priority) + sorted(stage for stage in by_stage if stage not in stage_priority):
        take = min(remaining, len(by_stage.get(stage, [])))
        if take > 0:
            chosen.extend(by_stage[stage][:take])
            remaining -= take
        if remaining <= 0:
            break
    return sort_records(chosen), dict(Counter(str(row_extra_info(record).get("rl_source_stage", "unknown")) for record in chosen))


def build_variant(variant_key: str, *, replacement_cap: float, overwrite: bool) -> dict[str, Any]:
    config = VARIANTS[variant_key]
    source_label = str(config["source_label"])
    output_label = str(config["output_label"])
    stage_priority = list(config["stage_priority"])

    source_parquet = OUTPUT_DATA_ROOT / f"{source_label}-insight_qwen_agent{config['source_suffix']}"
    output_parquet = OUTPUT_DATA_ROOT / f"{output_label}-insight_qwen_agent.parquet"
    raw_root = BUILD_ROOT / source_label / "raw_grouped_manifests"
    if output_parquet.exists() and not overwrite:
        raise FileExistsError(f"{output_parquet} exists; pass --overwrite")

    stage_by_qid = build_source_stage_map(raw_root)
    old_df = pd.read_parquet(OLD_FINAL_PARQUET)
    source_df = pd.read_parquet(source_parquet)

    old_records = [annotate_old_record(record, variant_key) for record in old_df.to_dict("records")]
    candidate_records = [
        add_stage_to_record(record, stage_by_qid, variant_key) for record in source_df.to_dict("records")
    ]

    old_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in old_records:
        old_by_bucket[(parquet_subset(record), answerability_key(record))].append(record)
    for bucket in list(old_by_bucket):
        old_by_bucket[bucket] = sort_records(old_by_bucket[bucket])

    added_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        stage = str(row_extra_info(record).get("rl_source_stage", "unknown"))
        if stage == "baseline_medium_hard":
            continue
        added_by_bucket[(parquet_subset(record), answerability_key(record))].append(record)
    for bucket in list(added_by_bucket):
        added_by_bucket[bucket] = sort_records(added_by_bucket[bucket])

    subsets = sorted({bucket[0] for bucket in old_by_bucket})
    selected: list[dict[str, Any]] = []
    replacement_summary: dict[str, Any] = {}
    selected_added_stage_counts: Counter[str] = Counter()

    for subset in subsets:
        subset_old_count = sum(len(rows) for (bucket_subset, _), rows in old_by_bucket.items() if bucket_subset == subset)
        subset_cap = int(math.floor(subset_old_count * replacement_cap))
        answerability_caps = {
            answerability: min(
                len(old_by_bucket.get((subset, answerability), [])),
                len(added_by_bucket.get((subset, answerability), [])),
            )
            for answerability in ("answerable", "unanswerable")
        }
        answerability_weights = {
            answerability: float(len(old_by_bucket.get((subset, answerability), [])))
            for answerability in ("answerable", "unanswerable")
        }
        replacement_counts = allocate_count(subset_cap, answerability_caps, answerability_weights)

        subset_stage_counts: Counter[str] = Counter()
        subset_selected_added = 0
        for answerability in ("answerable", "unanswerable"):
            bucket = (subset, answerability)
            replace_n = replacement_counts.get(answerability, 0)
            added_records, stage_counts = select_added_by_stage(
                added_by_bucket.get(bucket, []),
                replace_n,
                stage_priority=stage_priority,
            )
            selected.extend(added_records)
            selected.extend(old_by_bucket.get(bucket, [])[replace_n:])
            subset_selected_added += len(added_records)
            subset_stage_counts.update(stage_counts)
            selected_added_stage_counts.update(stage_counts)

        replacement_summary[subset] = {
            "old_rows": subset_old_count,
            "cap_rows": subset_cap,
            "replaced_rows": subset_selected_added,
            "replaced_ratio": subset_selected_added / subset_old_count if subset_old_count else None,
            "answerability_replacement_counts": {
                key: value for key, value in sorted(replacement_counts.items()) if value
            },
            "available_added_by_answerability": {
                key: value for key, value in sorted(answerability_caps.items()) if value
            },
            "selected_added_by_stage": dict(sorted(subset_stage_counts.items())),
        }

    selected = sort_records(selected)
    out_df = pd.DataFrame(selected, columns=old_df.columns)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_parquet, index=False)

    summary = summarize_parquet(output_parquet)
    summary.update(
        {
            "input_parquet": str(source_parquet),
            "output_parquet": str(output_parquet),
            "old_final_parquet": str(OLD_FINAL_PARQUET),
            "replacement_cap": replacement_cap,
            "replacement_policy": "per-subset capped replacement, preserving old subset and answerability counts",
            "stage_priority": stage_priority,
            "selected_added_stage_counts": dict(sorted(selected_added_stage_counts.items())),
            "replacement_summary": replacement_summary,
        }
    )
    output_parquet.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement-cap", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append")
    args = parser.parse_args()

    variants = args.variant or sorted(VARIANTS)
    summaries = [
        build_variant(variant, replacement_cap=args.replacement_cap, overwrite=args.overwrite)
        for variant in variants
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
