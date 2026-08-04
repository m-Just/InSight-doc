#!/usr/bin/env python3
"""Sample easy/super-easy rows for stage-4 VR2 SFT quality inspection."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/"
    "_rl_easy_super_easy_ablation_20260713/"
    "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_plus_super_easy8b/"
    "merged_for_parquet"
)
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.jsonl"
PDF_IMAGE_TARGET = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/"
    "_easy_supereasy_sft_quality_sample_20260715"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size-per-stage", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_root} exists and is not empty; pass --overwrite")
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(SOURCE_MANIFEST)
    pools: dict[str, list[dict[str, Any]]] = {
        "easy32b": [],
        "super_easy8b": [],
    }
    for row in rows:
        stage = row.get("rl_source_stage")
        if stage in pools:
            pools[stage].append(row)

    rng = random.Random(args.seed)
    sampled: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "source_manifest": str(SOURCE_MANIFEST),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "sample_size_per_stage": args.sample_size_per_stage,
        "source_pool_counts": {stage: len(stage_rows) for stage, stage_rows in pools.items()},
        "stages": {},
    }

    for stage in ("easy32b", "super_easy8b"):
        pool = sorted(pools[stage], key=lambda row: str(row["question_id"]))
        if len(pool) < args.sample_size_per_stage:
            raise ValueError(f"{stage} has only {len(pool)} rows, need {args.sample_size_per_stage}")
        chosen = rng.sample(pool, args.sample_size_per_stage)
        chosen = sorted(chosen, key=lambda row: str(row["question_id"]))
        sampled.extend(chosen)
        (args.output_root / f"question_ids_{stage}.txt").write_text(
            "\n".join(str(row["question_id"]) for row in chosen) + "\n",
            encoding="utf-8",
        )
        summary["stages"][stage] = {
            "rows": len(chosen),
            "subset_counts": dict(sorted(Counter(str(row.get("subset")) for row in chosen).items())),
            "question_ids_file": str(args.output_root / f"question_ids_{stage}.txt"),
        }

    sampled = sorted(sampled, key=lambda row: (str(row.get("rl_source_stage")), str(row["question_id"])))
    write_jsonl(args.output_root / "manifest.jsonl", sampled)
    (args.output_root / "question_ids.txt").write_text(
        "\n".join(str(row["question_id"]) for row in sampled) + "\n",
        encoding="utf-8",
    )

    pdf_image_link = args.output_root / "pdf_image"
    if pdf_image_link.exists() or pdf_image_link.is_symlink():
        if pdf_image_link.resolve() != PDF_IMAGE_TARGET.resolve():
            raise FileExistsError(f"{pdf_image_link} exists but does not point to {PDF_IMAGE_TARGET}")
    else:
        pdf_image_link.symlink_to(PDF_IMAGE_TARGET, target_is_directory=True)

    summary["total_rows"] = len(sampled)
    summary["combined_subset_counts"] = dict(sorted(Counter(str(row.get("subset")) for row in sampled).items()))
    summary["manifest"] = str(args.output_root / "manifest.jsonl")
    summary["question_ids_file"] = str(args.output_root / "question_ids.txt")
    summary["pdf_image"] = str(pdf_image_link)
    (args.output_root / "sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
