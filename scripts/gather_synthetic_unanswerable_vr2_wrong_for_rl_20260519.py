#!/usr/bin/env python3
"""Gather synthetic unanswerable examples that VR2 still answered incorrectly.

These are the medium/VR2 ``wrong_question_ids.txt`` rows from the first and
second synthetic-unanswerable batches, mapped back to the verified synthetic
manifest rows. The output is split by rescale/DPI for RL data construction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "notes/generated/synthetic_unanswerable_vr2_wrong_rl_20260519"


@dataclass(frozen=True)
class Lane:
    batch: str
    rescale: str
    dpi: int
    source_manifest: Path
    wrong_ids_path: Path
    sample_root: Path


def lanes() -> list[Lane]:
    first_root = REPO_ROOT / "artifacts/synthetic_unanswerable_pipeline/first_batch_sft_parquets_20260517"
    second_root = REPO_ROOT / "artifacts/synthetic_unanswerable_pipeline/second_batch_sft_parquets_20260518"
    seed42 = REPO_ROOT / "artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed42"
    seed43 = REPO_ROOT / "artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed43_disjoint_from_seed42"
    seed44 = REPO_ROOT / "artifacts/synthetic_unanswerable_pipeline/balanced_sample10k_seed44_disjoint_from_seed42_seed43"

    return [
        Lane(
            "first",
            "rescale025",
            50,
            REPO_ROOT
            / "artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed42_run1_az/"
            "verify_all_preview_c32/manifest.jsonl",
            first_root / "rescale025/medium/processed_gpt5_nano_rewrite/wrong_question_ids.txt",
            seed42,
        ),
        Lane(
            "first",
            "rescale035",
            70,
            REPO_ROOT
            / "artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed43_disjoint_from_seed42_run1_az/"
            "verify_all_rescale_mix_3to2_seed43/rescale0375/manifest.jsonl",
            first_root / "rescale035/medium/processed_gpt5_nano_rewrite/wrong_question_ids.txt",
            seed43,
        ),
        Lane(
            "first",
            "rescale05",
            100,
            REPO_ROOT
            / "artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed43_disjoint_from_seed42_run1_az/"
            "verify_all_rescale_mix_3to2_seed43/rescale05/manifest.jsonl",
            first_root / "rescale05/medium/processed_gpt5_nano_rewrite/wrong_question_ids.txt",
            seed43,
        ),
        Lane(
            "second",
            "rescale025",
            50,
            REPO_ROOT
            / "artifacts/synthetic_unanswerable_pipeline/balanced_sample10k_seed44_disjoint_from_seed42_seed43_run1_az_noproxy/"
            "verify_all_rescale_mix_5to3to2_seed44/rescale025/manifest.jsonl",
            second_root / "rescale025/medium/processed_gpt5_nano_rewrite/wrong_question_ids.txt",
            seed44,
        ),
        Lane(
            "second",
            "rescale035",
            70,
            REPO_ROOT
            / "artifacts/synthetic_unanswerable_pipeline/balanced_sample10k_seed44_disjoint_from_seed42_seed43_run1_az_noproxy/"
            "verify_all_rescale_mix_5to3to2_seed44/rescale035/manifest.jsonl",
            second_root / "rescale035/medium/processed_gpt5_nano_rewrite/wrong_question_ids.txt",
            seed44,
        ),
        Lane(
            "second",
            "rescale05",
            100,
            REPO_ROOT
            / "artifacts/synthetic_unanswerable_pipeline/balanced_sample10k_seed44_disjoint_from_seed42_seed43_run1_az_noproxy/"
            "verify_all_rescale_mix_5to3to2_seed44/rescale05/manifest.jsonl",
            second_root / "rescale05/medium/processed_gpt5_nano_rewrite/wrong_question_ids.txt",
            seed44,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_question_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"row {row_index} missing question_id in {path}")
            index[qid] = {"row_index": row_index, "row": row}
    return index


def load_source_question_map(sample_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for manifest_path in sorted(sample_root.glob("*/manifest.jsonl")):
        source_name = manifest_path.parent.name
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row.get("question_id")
                if isinstance(qid, str) and qid:
                    mapping[qid] = source_name
    return mapping


def source_name_for_row(row: dict[str, Any], source_map: dict[str, str]) -> str:
    metadata = row.get("synthetic_unanswerable_metadata") or {}
    if isinstance(metadata, dict):
        source_qid = metadata.get("source_question_id")
        if isinstance(source_qid, str):
            return source_map.get(source_qid, "<unmapped_source>")
    return "<missing_source_question_id>"


def mutation_type_for_row(row: dict[str, Any]) -> str:
    metadata = row.get("synthetic_unanswerable_metadata") or {}
    if isinstance(metadata, dict):
        value = metadata.get("mutation_type")
        if isinstance(value, str) and value:
            return value
    return "<missing>"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    all_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "output_root": str(output_root),
        "description": "Synthetic unanswerable examples whose medium/VR2 run was marked wrong.",
        "splits": {},
    }

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lane in lanes():
        if not lane.source_manifest.is_file():
            raise FileNotFoundError(f"missing source manifest: {lane.source_manifest}")
        if not lane.wrong_ids_path.is_file():
            raise FileNotFoundError(f"missing wrong ids: {lane.wrong_ids_path}")

        manifest_index = load_manifest(lane.source_manifest)
        source_map = load_source_question_map(lane.sample_root)
        question_ids = load_question_ids(lane.wrong_ids_path)
        duplicate_rows = sum(count - 1 for count in Counter(question_ids).values() if count > 1)
        missing: list[str] = []

        for input_line_index, qid in enumerate(question_ids, start=1):
            hit = manifest_index.get(qid)
            if hit is None:
                missing.append(qid)
                continue
            row = hit["row"]
            record = {
                "batch": lane.batch,
                "rescale": lane.rescale,
                "dpi": lane.dpi,
                "input_line_index": input_line_index,
                "question_id": qid,
                "source_manifest_path": str(lane.source_manifest),
                "source_row_index": hit["row_index"],
                "wrong_question_ids_path": str(lane.wrong_ids_path),
                "source_name": source_name_for_row(row, source_map),
                "subset": row.get("subset"),
                "mutation_type": mutation_type_for_row(row),
                "row": row,
            }
            all_records.append(record)
            by_split[f"{lane.rescale}_{lane.dpi}dpi"].append(record)

        summary["splits"].setdefault(f"{lane.rescale}_{lane.dpi}dpi", {"lanes": []})["lanes"].append(
            {
                "batch": lane.batch,
                "rescale": lane.rescale,
                "dpi": lane.dpi,
                "source_manifest": str(lane.source_manifest),
                "wrong_question_ids_path": str(lane.wrong_ids_path),
                "input_question_ids": len(question_ids),
                "duplicate_question_id_rows": duplicate_rows,
                "mapped_rows": len(question_ids) - len(missing),
                "missing_rows": len(missing),
            }
        )
        if missing:
            write_lines(output_root / "missing" / f"{lane.batch}_{lane.rescale}_{lane.dpi}dpi_missing.txt", missing)

    for split_name, records in sorted(by_split.items()):
        question_ids = [record["question_id"] for record in records]
        manifest_rows = [record["row"] for record in records]
        split_dir = output_root / split_name
        mapped_path = split_dir / "mapped.jsonl"
        question_ids_path = split_dir / "question_ids.txt"
        manifest_path = split_dir / "manifest.jsonl"
        write_jsonl(mapped_path, records)
        write_lines(question_ids_path, question_ids)
        write_jsonl(manifest_path, manifest_rows)

        source_counts = Counter(record["source_name"] for record in records)
        subset_counts = Counter(str(record["subset"]) for record in records)
        mutation_counts = Counter(record["mutation_type"] for record in records)
        summary["splits"][split_name].update(
            {
                "rows": len(records),
                "unique_question_ids": len(set(question_ids)),
                "mapped_jsonl": str(mapped_path),
                "question_ids": str(question_ids_path),
                "manifest_jsonl": str(manifest_path),
                "source_counts": dict(sorted(source_counts.items())),
                "subset_counts": dict(sorted(subset_counts.items())),
                "mutation_type_top20": dict(mutation_counts.most_common(20)),
            }
        )

    combined_dir = output_root / "combined"
    write_jsonl(combined_dir / "mapped.jsonl", all_records)
    write_lines(combined_dir / "question_ids.txt", [record["question_id"] for record in all_records])
    write_jsonl(combined_dir / "manifest.jsonl", [record["row"] for record in all_records])
    summary["combined"] = {
        "rows": len(all_records),
        "unique_question_ids": len({record["question_id"] for record in all_records}),
        "mapped_jsonl": str(combined_dir / "mapped.jsonl"),
        "question_ids": str(combined_dir / "question_ids.txt"),
        "manifest_jsonl": str(combined_dir / "manifest.jsonl"),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
