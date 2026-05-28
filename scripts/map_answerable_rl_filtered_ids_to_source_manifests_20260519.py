#!/usr/bin/env python3
"""Map filtered answerable RL question IDs back to source manifest rows.

The 2026-05-18 filter file covers the original O3/arxiv/spanning/dude source
parts used by the first RL data build. The 2026-05-19 filter file covers the
later source parts; some of those intentionally reuse earlier O3 source
manifests, so the output keeps both the logical source part and the original
manifest path.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "notes/generated/answerable_rl_afterfiltering_manifest_mapping_20260519"

FILTER_0518 = Path("/scratch/ywxzml3j/likaican/src/InSight-doc/data/answerable_RL_afterfiltering0518.txt")
FILTER_0519 = Path("/scratch/ywxzml3j/likaican/src/InSight-doc/data/answerable_RL_afterfiltering0519.txt")

O3_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424")
ARXIV_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess")
GROUPED_HINT_ROOTS = {
    "0518": Path(
        "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/"
        "_rl_wrong_question_manifests_medium_only_balanced_half_unanswerable_dude_reduced"
    ),
    "0519": Path(
        "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/"
        "_rl_wrong_question_manifests_medium_only_part3abcd_part45"
    ),
}


@dataclass(frozen=True)
class SourcePart:
    batch: str
    logical_part: str
    source_group: str
    manifest_path: Path


def source_parts() -> list[SourcePart]:
    arxiv_train = ARXIV_ROOT / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
    arxiv_additional = ARXIV_ROOT / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
    arxiv_spanning = ARXIV_ROOT / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"

    return [
        SourcePart(
            "0518",
            "O3_data_0424/train_part1",
            "O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part1/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0518",
            "O3_data_0424/train_part2a",
            "O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2a/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0518",
            "O3_data_0424/train_part2b",
            "O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2b/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0518",
            "O3_data_0424/train_part2c",
            "O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2c/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0518",
            "O3_data_0424/dude_poster_unanswerable",
            "O3_data_0424__dude_poster_unanswerable__dpi200_aug_noaug_maxp40",
            O3_ROOT / "dude_poster_unanswerable/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0518",
            "arxiv/train_part1",
            "arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40",
            arxiv_train / "dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0518",
            "arxiv/train_part2",
            "arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40",
            arxiv_train / "dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0518",
            "arxiv/train_part3",
            (
                "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__"
                "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            ),
            arxiv_additional / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0/manifest.jsonl",
        ),
        SourcePart(
            "0518",
            "arxiv/spanning_train_part1",
            "arxiv__spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning__dpi200_aug_noaug_maxp40",
            arxiv_spanning / "dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0519",
            "O3_data_0424/train_part3a",
            "O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2a/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0519",
            "O3_data_0424/train_part3b",
            "O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2b/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0519",
            "O3_data_0424/train_part3c",
            "O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part2c/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0519",
            "O3_data_0424/train_part3d",
            "O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40",
            O3_ROOT / "0426_selected_train_part1/dpi200_aug_noaug_maxp40/manifest.jsonl",
        ),
        SourcePart(
            "0519",
            "arxiv/train_part4",
            (
                "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__"
                "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            ),
            arxiv_additional / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0/manifest.jsonl",
        ),
        SourcePart(
            "0519",
            "arxiv/train_part5",
            (
                "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__"
                "dpi200_aug_noaug_maxp40_jitter_seed0"
            ),
            arxiv_additional / "dpi200_aug_noaug_maxp40_jitter_seed0/manifest.jsonl",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-0518", type=Path, default=FILTER_0518)
    parser.add_argument("--filter-0519", type=Path, default=FILTER_0519)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_question_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"manifest row {row_index} missing question_id in {path}")
            index[qid].append({"row_index": row_index, "row": row})
    return index


def build_manifest_indexes(parts: list[SourcePart]) -> dict[Path, dict[str, list[dict[str, Any]]]]:
    indexes: dict[Path, dict[str, list[dict[str, Any]]]] = {}
    for manifest_path in sorted({part.manifest_path for part in parts}):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing source manifest: {manifest_path}")
        indexes[manifest_path] = load_manifest(manifest_path)
    return indexes


def load_group_hints(root: Path) -> dict[str, list[str]]:
    hints: dict[str, list[str]] = defaultdict(list)
    if not root.is_dir():
        return hints
    for manifest_path in sorted(root.glob("*/manifest.jsonl")):
        group = manifest_path.parent.name
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row.get("question_id")
                if isinstance(qid, str) and qid:
                    hints[qid].append(group)
    return hints


def build_group_hints() -> dict[str, dict[str, list[str]]]:
    return {batch: load_group_hints(root) for batch, root in GROUPED_HINT_ROOTS.items()}


def jsonable_match(part: SourcePart, row_index: int) -> dict[str, Any]:
    return {
        "batch": part.batch,
        "logical_part": part.logical_part,
        "source_group": part.source_group,
        "manifest_path": str(part.manifest_path),
        "row_index": row_index,
    }


def map_ids(
    *,
    batch: str,
    input_file: Path,
    question_ids: list[str],
    parts: list[SourcePart],
    manifest_indexes: dict[Path, dict[str, list[dict[str, Any]]]],
    group_hints: dict[str, list[str]],
) -> list[dict[str, Any]]:
    batch_parts = [part for part in parts if part.batch == batch]
    records: list[dict[str, Any]] = []
    for line_index, qid in enumerate(question_ids, start=1):
        matches: list[dict[str, Any]] = []
        seen_row_keys: set[tuple[str, int]] = set()
        rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}

        for part in batch_parts:
            for row_hit in manifest_indexes[part.manifest_path].get(qid, []):
                row_index = int(row_hit["row_index"])
                key = (str(part.manifest_path), row_index)
                match = jsonable_match(part, row_index)
                matches.append(match)
                rows_by_key[key] = row_hit["row"]
                seen_row_keys.add(key)

        selected_match: dict[str, Any] | None = None
        selected_row: dict[str, Any] | None = None
        preferred_groups = group_hints.get(qid, [])
        for preferred_group in preferred_groups:
            selected_match = next(
                (match for match in matches if match["source_group"] == preferred_group),
                None,
            )
            if selected_match is not None:
                break
        if selected_match is None and matches:
            selected_match = matches[0]
        if selected_match is not None:
            selected_key = (selected_match["manifest_path"], int(selected_match["row_index"]))
            selected_row = rows_by_key[selected_key]

        records.append(
            {
                "batch": batch,
                "input_file": str(input_file),
                "input_line_index": line_index,
                "question_id": qid,
                "mapped": selected_row is not None,
                "preferred_source_groups": preferred_groups,
                "selected_match": selected_match,
                "matches": matches,
                "row": selected_row,
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_grouped_manifest_subsets(output_root: Path, batch: str, records: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if not record["mapped"]:
            continue
        selected = record["selected_match"]
        row = record["row"]
        if not isinstance(selected, dict) or not isinstance(row, dict):
            continue
        group = str(selected["source_group"])
        grouped[group][str(record["question_id"])] = row

    counts: dict[str, int] = {}
    for group, rows_by_qid in sorted(grouped.items()):
        rows = [rows_by_qid[qid] for qid in sorted(rows_by_qid)]
        path = output_root / "grouped_manifest_subsets" / batch / group / "manifest.jsonl"
        write_jsonl(path, rows)
        counts[group] = len(rows)
    return counts


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    mapped = [record for record in records if record["mapped"]]
    unmapped = [record for record in records if not record["mapped"]]
    selected_groups = Counter(
        record["selected_match"]["source_group"]
        for record in mapped
        if isinstance(record.get("selected_match"), dict)
    )
    selected_parts = Counter(
        record["selected_match"]["logical_part"]
        for record in mapped
        if isinstance(record.get("selected_match"), dict)
    )
    multi_match = sum(1 for record in mapped if len(record["matches"]) > 1)
    distinct_manifest_row_multi_match = 0
    for record in mapped:
        row_keys = {
            (match["manifest_path"], match["row_index"])
            for match in record["matches"]
            if isinstance(match, dict)
        }
        if len(row_keys) > 1:
            distinct_manifest_row_multi_match += 1

    return {
        "input_rows": len(records),
        "unique_question_ids": len({record["question_id"] for record in records}),
        "mapped_rows": len(mapped),
        "unmapped_rows": len(unmapped),
        "multi_match_rows": multi_match,
        "distinct_manifest_row_multi_match_rows": distinct_manifest_row_multi_match,
        "selected_group_counts": dict(sorted(selected_groups.items())),
        "selected_logical_part_counts": dict(sorted(selected_parts.items())),
    }


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    parts = source_parts()
    manifest_indexes = build_manifest_indexes(parts)
    group_hints_by_batch = build_group_hints()

    inputs = {
        "0518": args.filter_0518.expanduser().resolve(),
        "0519": args.filter_0519.expanduser().resolve(),
    }
    all_records: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "output_root": str(output_root),
        "input_files": {batch: str(path) for batch, path in inputs.items()},
        "source_parts": [
            {
                "batch": part.batch,
                "logical_part": part.logical_part,
                "source_group": part.source_group,
                "manifest_path": str(part.manifest_path),
            }
            for part in parts
        ],
        "batches": {},
    }

    for batch, input_path in inputs.items():
        question_ids = load_question_ids(input_path)
        duplicate_count = sum(count - 1 for count in Counter(question_ids).values() if count > 1)
        records = map_ids(
            batch=batch,
            input_file=input_path,
            question_ids=question_ids,
            parts=parts,
            manifest_indexes=manifest_indexes,
            group_hints=group_hints_by_batch.get(batch, {}),
        )
        all_records[batch] = records
        mapping_path = output_root / f"answerable_RL_afterfiltering{batch}_mapped.jsonl"
        write_jsonl(mapping_path, records)
        unmapped_ids = [record["question_id"] for record in records if not record["mapped"]]
        unmapped_path = output_root / f"answerable_RL_afterfiltering{batch}_unmapped.txt"
        unmapped_path.write_text("".join(f"{qid}\n" for qid in unmapped_ids), encoding="utf-8")
        grouped_counts = write_grouped_manifest_subsets(output_root, batch, records)

        batch_summary = summarize_records(records)
        batch_summary.update(
            {
                "input_file": str(input_path),
                "mapping_jsonl": str(mapping_path),
                "unmapped_question_ids": str(unmapped_path),
                "duplicate_question_id_rows": duplicate_count,
                "group_hint_root": str(GROUPED_HINT_ROOTS[batch]),
                "group_hint_question_ids": len(group_hints_by_batch.get(batch, {})),
                "grouped_manifest_subset_counts": grouped_counts,
            }
        )
        summary["batches"][batch] = batch_summary

    overlap = sorted(
        {record["question_id"] for record in all_records["0518"]}
        & {record["question_id"] for record in all_records["0519"]}
    )
    summary["cross_batch_overlap"] = {
        "count": len(overlap),
        "path": str(output_root / "question_ids_overlap_0518_0519.txt"),
    }
    (output_root / "question_ids_overlap_0518_0519.txt").write_text(
        "".join(f"{qid}\n" for qid in overlap),
        encoding="utf-8",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
