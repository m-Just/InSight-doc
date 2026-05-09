#!/usr/bin/env python3
"""Build RL manifest subsets from generated/processed wrong_question_ids.txt files.

This scans the reorganized generated tree, reads each processed/wrong_question_ids.txt
except excluded datasets such as val, maps each processed dataset back to the
manifest that originally produced it, and writes deduplicated manifest subsets
grouped by source image/manifest directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GENERATED_ROOT_DEFAULT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")
ARXIV_POSTPROCESS_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess"
)
O3_ROOT_DEFAULT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424")
OUTPUT_ROOT_DEFAULT = GENERATED_ROOT_DEFAULT / "_rl_wrong_question_manifests"


@dataclass(frozen=True)
class DatasetManifestSpec:
    manifest_path: Path
    group_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, default=GENERATED_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--arxiv-postprocess-root", type=Path, default=ARXIV_POSTPROCESS_ROOT_DEFAULT)
    parser.add_argument("--o3-root", type=Path, default=O3_ROOT_DEFAULT)
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium"),
        default=None,
        help="Only include processed wrong_question_ids from this difficulty.",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=["val_sample_102"],
        help="Skip processed roots whose relative path contains this substring. May be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned grouping and counts without writing output manifests.",
    )
    return parser.parse_args()


def dataset_specs(arxiv_postprocess_root: Path, o3_root: Path) -> dict[tuple[str, str], DatasetManifestSpec]:
    return {
        ("O3_data_0424", "train_part1"): DatasetManifestSpec(
            manifest_path=o3_root / "0426_selected_train_part1" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            group_name="O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40",
        ),
        ("O3_data_0424", "train_part2a"): DatasetManifestSpec(
            manifest_path=o3_root / "0426_selected_train_part2a" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            group_name="O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40",
        ),
        ("O3_data_0424", "train_part2b"): DatasetManifestSpec(
            manifest_path=o3_root / "0426_selected_train_part2b" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            group_name="O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40",
        ),
        ("O3_data_0424", "train_part2c"): DatasetManifestSpec(
            manifest_path=o3_root / "0426_selected_train_part2c" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            group_name="O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40",
        ),
        ("O3_data_0424", "dude_poster_unanswerable"): DatasetManifestSpec(
            manifest_path=o3_root / "dude_poster_unanswerable" / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
            group_name="O3_data_0424__dude_poster_unanswerable__dpi200_aug_noaug_maxp40",
        ),
        ("arxiv", "train_part1"): DatasetManifestSpec(
            manifest_path=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "manifest.jsonl",
            group_name="arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40",
        ),
        ("arxiv", "train_part2"): DatasetManifestSpec(
            manifest_path=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
            / "dpi200_aug_noaug_maxp40"
            / "manifest.jsonl",
            group_name="arxiv__veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train__dpi200_aug_noaug_maxp40",
        ),
        ("arxiv", "train_part3"): DatasetManifestSpec(
            manifest_path=arxiv_postprocess_root
            / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
            / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            / "manifest.jsonl",
            group_name=(
                "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__"
                "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            ),
        ),
        ("arxiv", "spanning_train_part1"): DatasetManifestSpec(
            manifest_path=arxiv_postprocess_root
            / "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
            / "dpi200_aug_noaug_maxp40"
            / "manifest.jsonl",
            group_name=(
                "arxiv__spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning__"
                "dpi200_aug_noaug_maxp40"
            ),
        ),
    }


def iter_processed_wrong_id_files(
    generated_root: Path, exclude_patterns: list[str], difficulty: str | None
) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(generated_root.glob("**/processed/wrong_question_ids.txt")):
        rel = str(path.relative_to(generated_root))
        if any(pattern and pattern in rel for pattern in exclude_patterns):
            continue
        rel_parts = path.relative_to(generated_root).parts
        if len(rel_parts) < 5:
            continue
        if difficulty is not None and rel_parts[2] != difficulty:
            continue
        paths.append(path)
    return paths


def load_wrong_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                ids.append(value)
    return ids


def question_id_from_manifest_row(row: dict[str, Any]) -> str:
    value = row.get("question_id")
    if isinstance(value, str):
        return value
    raise ValueError(f"manifest row missing string question_id: {row}")


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def group_output_path(output_root: Path, group_name: str) -> Path:
    return output_root / sanitize_filename(group_name) / "manifest.jsonl"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    generated_root = args.generated_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    specs = dataset_specs(
        arxiv_postprocess_root=args.arxiv_postprocess_root.expanduser().resolve(),
        o3_root=args.o3_root.expanduser().resolve(),
    )

    wrong_id_files = iter_processed_wrong_id_files(generated_root, args.exclude_pattern, args.difficulty)
    if not wrong_id_files:
        raise SystemExit(f"no processed/wrong_question_ids.txt found under {generated_root}")

    grouped_ids: dict[str, set[str]] = defaultdict(set)
    grouped_sources: dict[str, list[str]] = defaultdict(list)
    grouped_manifest_paths: dict[str, Path] = {}
    total_input_ids = 0

    for wrong_id_path in wrong_id_files:
        rel_parts = wrong_id_path.relative_to(generated_root).parts
        if len(rel_parts) < 5:
            raise SystemExit(f"unexpected processed path shape: {wrong_id_path}")
        dataset_key = (rel_parts[0], rel_parts[1])
        spec = specs.get(dataset_key)
        if spec is None:
            raise SystemExit(f"no manifest mapping for processed wrong ids: {wrong_id_path}")
        if not spec.manifest_path.is_file():
            raise SystemExit(f"mapped manifest does not exist: {spec.manifest_path}")
        ids = load_wrong_ids(wrong_id_path)
        total_input_ids += len(ids)
        grouped_ids[spec.group_name].update(ids)
        grouped_sources[spec.group_name].append(str(wrong_id_path))
        grouped_manifest_paths[spec.group_name] = spec.manifest_path

    summary_groups: list[dict[str, Any]] = []
    total_output_rows = 0
    total_missing_ids = 0

    for group_name in sorted(grouped_ids):
        manifest_path = grouped_manifest_paths[group_name]
        wanted_ids = grouped_ids[group_name]
        matched_rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = question_id_from_manifest_row(row)
                if qid in wanted_ids:
                    matched_rows.append(row)
                    seen_ids.add(qid)

        missing_ids = sorted(wanted_ids - seen_ids)
        total_output_rows += len(matched_rows)
        total_missing_ids += len(missing_ids)

        output_manifest = group_output_path(output_root, group_name)
        output_meta = output_manifest.parent / "meta.json"
        output_missing = output_manifest.parent / "missing_question_ids.txt"

        group_summary = {
            "group_name": group_name,
            "manifest_path": str(manifest_path),
            "output_manifest": str(output_manifest),
            "contributing_wrong_id_files": grouped_sources[group_name],
            "input_unique_question_ids": len(wanted_ids),
            "output_rows": len(matched_rows),
            "missing_question_ids": len(missing_ids),
        }
        summary_groups.append(group_summary)

        if args.dry_run:
            continue

        ensure_parent(output_manifest)
        with output_manifest.open("w", encoding="utf-8") as handle:
            for row in matched_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_meta.write_text(json.dumps(group_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if missing_ids:
            output_missing.write_text("\n".join(missing_ids) + "\n", encoding="utf-8")
        elif output_missing.exists():
            output_missing.unlink()

    final_summary = {
        "generated_root": str(generated_root),
        "output_root": str(output_root),
        "processed_wrong_id_files": [str(path) for path in wrong_id_files],
        "difficulty": args.difficulty,
        "exclude_patterns": args.exclude_pattern,
        "input_wrong_question_ids_total": total_input_ids,
        "groups": summary_groups,
        "output_rows_total": total_output_rows,
        "missing_question_ids_total": total_missing_ids,
    }

    print(json.dumps(final_summary, ensure_ascii=False, indent=2))

    if not args.dry_run:
        ensure_parent(output_root / "summary.json")
        (output_root / "summary.json").write_text(
            json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
