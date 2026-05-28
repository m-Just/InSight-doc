#!/usr/bin/env python3
"""Build grouped RL manifests from selected Stage-5 wrong_question_ids outputs.

This is the "output-literal" variant for the new batch:
- O3 train_part3a / 3b / 3c / 3d final Stage-5 wrong ids
- arxiv train_part4 / train_part5 final Stage-5 wrong ids

Rows are retrieved from the source manifests that actually backed those final
outputs. This means the O3 part3* lanes map back to the reused manifest roots
from part1 / part2a / part2b / part2c, matching how the data was created.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_ROOT_DEFAULT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/"
    "_rl_wrong_question_manifests_medium_only_part3abcd_part45"
)


@dataclass(frozen=True)
class SourceSpec:
    wrong_ids_path: Path
    manifest_path: Path
    group_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_specs() -> list[SourceSpec]:
    o3_root = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424")
    arxiv_root = Path(
        "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess"
    )
    stage5_root = Path(
        "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
        "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed"
    )

    return [
        SourceSpec(
            wrong_ids_path=stage5_root
            / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3a_resumable/wrong_question_ids.txt",
            manifest_path=o3_root / "0426_selected_train_part2a/dpi200_aug_noaug_maxp40/manifest.jsonl",
            group_name="O3_data_0424__0426_selected_train_part2a__dpi200_aug_noaug_maxp40",
        ),
        SourceSpec(
            wrong_ids_path=stage5_root
            / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3b_resumable/wrong_question_ids.txt",
            manifest_path=o3_root / "0426_selected_train_part2b/dpi200_aug_noaug_maxp40/manifest.jsonl",
            group_name="O3_data_0424__0426_selected_train_part2b__dpi200_aug_noaug_maxp40",
        ),
        SourceSpec(
            wrong_ids_path=stage5_root
            / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3c_resumable/wrong_question_ids.txt",
            manifest_path=o3_root / "0426_selected_train_part2c/dpi200_aug_noaug_maxp40/manifest.jsonl",
            group_name="O3_data_0424__0426_selected_train_part2c__dpi200_aug_noaug_maxp40",
        ),
        SourceSpec(
            wrong_ids_path=stage5_root
            / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3d_resumable/wrong_question_ids.txt",
            manifest_path=o3_root / "0426_selected_train_part1/dpi200_aug_noaug_maxp40/manifest.jsonl",
            group_name="O3_data_0424__0426_selected_train_part1__dpi200_aug_noaug_maxp40",
        ),
        SourceSpec(
            wrong_ids_path=stage5_root
            / (
                "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-"
                "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0-0426_train_part4_resumable/"
                "wrong_question_ids.txt"
            ),
            manifest_path=arxiv_root
            / (
                "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
                "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0/manifest.jsonl"
            ),
            group_name=(
                "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__"
                "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
            ),
        ),
        SourceSpec(
            wrong_ids_path=stage5_root
            / (
                "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-"
                "dpi200_aug_noaug_maxp40_jitter_seed0-0426_train_part5_resumable/"
                "wrong_question_ids.txt"
            ),
            manifest_path=arxiv_root
            / (
                "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
                "dpi200_aug_noaug_maxp40_jitter_seed0/manifest.jsonl"
            ),
            group_name=(
                "arxiv__veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional__"
                "dpi200_aug_noaug_maxp40_jitter_seed0"
            ),
        ),
    ]


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


def sanitize_group_name(value: str) -> str:
    return value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()

    grouped_ids: dict[str, set[str]] = {}
    grouped_manifest_paths: dict[str, Path] = {}
    grouped_sources: dict[str, list[str]] = {}
    input_wrong_question_ids_total = 0

    for spec in source_specs():
        if not spec.wrong_ids_path.is_file():
            raise FileNotFoundError(f"missing wrong ids file: {spec.wrong_ids_path}")
        if not spec.manifest_path.is_file():
            raise FileNotFoundError(f"missing source manifest: {spec.manifest_path}")
        ids = load_wrong_ids(spec.wrong_ids_path)
        input_wrong_question_ids_total += len(ids)
        grouped_ids.setdefault(spec.group_name, set()).update(ids)
        grouped_sources.setdefault(spec.group_name, []).append(str(spec.wrong_ids_path))
        grouped_manifest_paths[spec.group_name] = spec.manifest_path

    groups_summary: list[dict[str, Any]] = []
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

        out_dir = output_root / sanitize_group_name(group_name)
        output_manifest = out_dir / "manifest.jsonl"
        output_meta = out_dir / "meta.json"
        output_missing = out_dir / "missing_question_ids.txt"

        group_summary = {
            "group_name": group_name,
            "manifest_path": str(manifest_path),
            "output_manifest": str(output_manifest),
            "contributing_wrong_id_files": grouped_sources[group_name],
            "input_unique_question_ids": len(wanted_ids),
            "output_rows": len(matched_rows),
            "missing_question_ids": len(missing_ids),
        }
        groups_summary.append(group_summary)

        if not args.dry_run:
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
        "output_root": str(output_root),
        "input_wrong_question_ids_total": input_wrong_question_ids_total,
        "groups": groups_summary,
        "output_rows_total": total_output_rows,
        "missing_question_ids_total": total_missing_ids,
    }

    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "summary.json").write_text(
            json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
