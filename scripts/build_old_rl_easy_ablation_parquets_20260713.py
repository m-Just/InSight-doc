#!/usr/bin/env python3
"""Build old-RL-data ablation parquets with easy/super-easy source rows.

This reconstructs the old `insight_doc_rl_balanced_dude_reduced_merged_u25`
pipeline, but changes the raw source manifest pool:

1. old medium/hard RL source rows + 32B-easy rows
2. old medium/hard RL source rows + 32B-easy rows + 8B 20-DPI super-easy rows

The downstream construction intentionally follows the old note:
raw grouped manifests -> balanced/dude-reduced manifests -> merged manifest ->
u25 reduction -> InSightDocRL parquet -> estimated-prompt filter.

After prompt filtering, a final selection pass keeps the old per-subset and
overall unanswerable ratios as close as integer counts permit and annotates
`extra_info.rl_source_stage` for reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INSIGHT_DOC_ROOT = Path("/scratch/ywxzml3j/likaican/src/InSight-doc")
GENERATED_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated")
O3_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424")
ARXIV_POSTPROCESS_ROOT = Path(
    "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess"
)
OUTPUT_DATA_ROOT = Path("/scratch/ywxzml3j/likaican/data/insight_doc")

OLD_RAW_ROOT = GENERATED_ROOT / "_rl_wrong_question_manifests_medium_only"
OLD_FINAL_PARQUET = OUTPUT_DATA_ROOT / "insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.parquet"

WORK_ROOT = GENERATED_ROOT / "_rl_easy_super_easy_ablation_20260713"
SUPER_EASY_8B_ROOT = Path(
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
    "insight_qwen_agent_initial_0_1_zoom_factor2_default_sys_0426_resumable/"
    "qwen3-vl-8b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40"
)


@dataclass(frozen=True)
class SourceSpec:
    dataset_group: str
    split: str
    group_name: str
    manifest_path: Path
    question_ids_path: Path | None = None
    super_easy_pass_root: Path | None = None


def o3_spec(split: str, manifest_dir_name: str, *, include_super_easy: bool = False) -> SourceSpec:
    manifest_root = O3_ROOT / manifest_dir_name / "dpi200_aug_noaug_maxp40"
    return SourceSpec(
        dataset_group="O3_data_0424",
        split=split,
        group_name=f"O3_data_0424__{manifest_dir_name}__dpi200_aug_noaug_maxp40",
        manifest_path=manifest_root / "manifest.jsonl",
        question_ids_path=manifest_root / "question_ids.txt" if include_super_easy else None,
        super_easy_pass_root=SUPER_EASY_8B_ROOT / split if include_super_easy else None,
    )


def arxiv_spec(split: str, postprocess_name: str, group_suffix: str | None = None) -> SourceSpec:
    suffix = group_suffix or postprocess_name
    return SourceSpec(
        dataset_group="arxiv",
        split=split,
        group_name=f"arxiv__{suffix}__dpi200_aug_noaug_maxp40",
        manifest_path=ARXIV_POSTPROCESS_ROOT / postprocess_name / "dpi200_aug_noaug_maxp40" / "manifest.jsonl",
    )


def arxiv_part3_spec() -> SourceSpec:
    name = "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
    suffix = f"{name}__dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
    return SourceSpec(
        dataset_group="arxiv",
        split="train_part3",
        group_name=f"arxiv__{suffix}",
        manifest_path=ARXIV_POSTPROCESS_ROOT
        / name
        / "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
        / "manifest.jsonl",
    )


SOURCE_SPECS: list[SourceSpec] = [
    o3_spec("train_part1", "0426_selected_train_part1", include_super_easy=True),
    o3_spec("train_part2a", "0426_selected_train_part2a", include_super_easy=True),
    o3_spec("train_part2b", "0426_selected_train_part2b", include_super_easy=True),
    o3_spec("train_part2c", "0426_selected_train_part2c", include_super_easy=True),
    o3_spec("dude_poster_unanswerable", "dude_poster_unanswerable", include_super_easy=False),
    arxiv_spec(
        "spanning_train_part1",
        "spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning",
    ),
    arxiv_spec(
        "train_part1",
        "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train",
        group_suffix="veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train",
    ),
    arxiv_spec(
        "train_part2",
        "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train",
        group_suffix="veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train",
    ),
    arxiv_part3_spec(),
]


VARIANTS = {
    "plus_easy32b": {
        "label": "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b",
        "include_easy32b": True,
        "include_super_easy8b": False,
    },
    "plus_easy32b_plus_super_easy8b": {
        "label": "insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_plus_super_easy8b",
        "include_easy32b": True,
        "include_super_easy8b": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        help="Variant(s) to build. Defaults to both.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-create-parquet", action="store_true")
    parser.add_argument("--num-workers", type=int, default=32)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def qid(row: dict[str, Any]) -> str:
    value = row.get("question_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"row missing question_id: {row}")
    return value


def is_unanswerable_manifest_row(row: dict[str, Any]) -> bool:
    return "not-answerable" in json.dumps(row.get("question_type"), ensure_ascii=False).lower()


def stable_key(value: str, seed: int = 0) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def load_manifest_index(specs: list[SourceSpec]) -> dict[str, dict[str, dict[str, Any]]]:
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    loaded_paths: dict[Path, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        manifest_path = spec.manifest_path.resolve()
        if manifest_path not in loaded_paths:
            rows = read_jsonl(manifest_path)
            loaded_paths[manifest_path] = {qid(row): row for row in rows}
        by_group.setdefault(spec.group_name, {}).update(loaded_paths[manifest_path])
    return by_group


def load_degenerate_bad_paths() -> set[str]:
    bad: set[str] = set()
    report_root = GENERATED_ROOT / "_quality_reports"
    for path in report_root.glob("easy*/*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("bad_example"):
                    raw_path = record.get("path")
                    if raw_path:
                        bad.add(str(Path(raw_path).resolve()))
    return bad


def raw_accuracy_reward(record: dict[str, Any]) -> float | None:
    score = (record.get("reward") or {}).get("score") or {}
    value = score.get("accuracy_reward")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_easy32b_qids(specs: list[SourceSpec]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Recover the exact processed 32B-easy qids.

    `processed_drop_degenerate/wrong_question_ids.txt` is written by the
    converter for every non-converted row when `--only-correct-answers` is used:
    wrong answers, degenerate conversations, and conversion failures such as
    missing rewritten image paths. Therefore the processed easy set is the raw
    exported qids minus that wrong/filtered-id file. This matches the SFT parquet
    row count more exactly than checking `accuracy_reward == 1` ourselves.
    """
    bad_paths = load_degenerate_bad_paths()
    by_group: dict[str, set[str]] = defaultdict(set)
    split_summaries: list[dict[str, Any]] = []
    total_raw_json = 0
    total_correct = 0
    total_bad_correct = 0
    total_converter_filtered_ids = 0

    for spec in specs:
        raw_dir = GENERATED_ROOT / spec.dataset_group / spec.split / "easy" / "raw"
        if not raw_dir.exists():
            split_summaries.append(
                {
                    "dataset_group": spec.dataset_group,
                    "split": spec.split,
                    "group_name": spec.group_name,
                    "raw_dir": str(raw_dir),
                    "exists": False,
                }
            )
            continue

        raw_json = 0
        correct = 0
        bad_correct = 0
        raw_qids: set[str] = set()
        kept = 0
        for path in sorted(raw_dir.glob("*.json")):
            raw_json += 1
            total_raw_json += 1
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            extra_info = record.get("extra_info") or {}
            question_id = extra_info.get("question_id")
            if isinstance(question_id, str) and question_id:
                raw_qids.add(question_id)
            if raw_accuracy_reward(record) != 1.0:
                continue
            correct += 1
            total_correct += 1
            if str(path.resolve()) in bad_paths:
                bad_correct += 1
                total_bad_correct += 1
                continue

        sft_parquet = GENERATED_ROOT / spec.dataset_group / spec.split / "easy" / "processed_drop_degenerate" / "sft_data.parquet"
        wrong_ids_path = (
            GENERATED_ROOT
            / spec.dataset_group
            / spec.split
            / "easy"
            / "processed_drop_degenerate"
            / "wrong_question_ids.txt"
        )
        converter_filtered_ids = read_ids(wrong_ids_path) if wrong_ids_path.exists() else set()
        total_converter_filtered_ids += len(converter_filtered_ids)
        kept_qids = raw_qids - converter_filtered_ids
        by_group[spec.group_name].update(kept_qids)
        kept = len(kept_qids)
        sft_rows = None
        if sft_parquet.exists():
            try:
                sft_rows = int(len(pd.read_parquet(sft_parquet, columns=["messages"])))
            except Exception:
                sft_rows = None
        split_summaries.append(
            {
                "dataset_group": spec.dataset_group,
                "split": spec.split,
                "group_name": spec.group_name,
                "raw_dir": str(raw_dir),
                "exists": True,
                "raw_json": raw_json,
                "raw_qids": len(raw_qids),
                "correct_reward_rows": correct,
                "degenerate_correct_rows": bad_correct,
                "converter_filtered_ids": len(converter_filtered_ids),
                "kept_easy32b_qids": kept,
                "processed_sft_rows": sft_rows,
                "matches_processed_sft_rows": (sft_rows == kept) if sft_rows is not None else None,
            }
        )

    summary = {
        "bad_degenerate_paths": len(bad_paths),
        "raw_json_total": total_raw_json,
        "correct_reward_rows_total": total_correct,
        "degenerate_correct_rows_total": total_bad_correct,
        "converter_filtered_ids_total": total_converter_filtered_ids,
        "kept_easy32b_qids_total": sum(len(v) for v in by_group.values()),
        "splits": split_summaries,
    }
    return dict(by_group), summary


def latest_passed_ids_path(pass_root: Path) -> Path:
    paths = sorted(pass_root.glob("pass*/converted_sft/passed_question_ids.txt"))
    if not paths:
        raise FileNotFoundError(f"no passed_question_ids.txt under {pass_root}")
    return paths[-1]


def collect_super_easy8b_qids(specs: list[SourceSpec]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    by_group: dict[str, set[str]] = defaultdict(set)
    split_summaries: list[dict[str, Any]] = []
    for spec in specs:
        if spec.question_ids_path is None or spec.super_easy_pass_root is None:
            continue
        original = read_ids(spec.question_ids_path)
        passed_path = latest_passed_ids_path(spec.super_easy_pass_root)
        passed = read_ids(passed_path)
        super_easy = original - passed
        by_group[spec.group_name].update(super_easy)
        split_summaries.append(
            {
                "dataset_group": spec.dataset_group,
                "split": spec.split,
                "group_name": spec.group_name,
                "question_ids": len(original),
                "passed_question_ids": len(passed),
                "super_easy8b_qids": len(super_easy),
                "passed_question_ids_path": str(passed_path),
            }
        )
    return dict(by_group), {
        "kept_super_easy8b_qids_total": sum(len(v) for v in by_group.values()),
        "splits": split_summaries,
    }


def annotate_manifest_row(row: dict[str, Any], source_stage: str, variant: str) -> dict[str, Any]:
    out = dict(row)
    out["rl_source_stage"] = source_stage
    out["rl_source_variant"] = variant
    return out


def build_augmented_raw_root(
    *,
    variant_key: str,
    variant_label: str,
    include_easy32b: bool,
    include_super_easy8b: bool,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    raw_root = WORK_ROOT / variant_label / "raw_grouped_manifests"
    if raw_root.exists():
        if overwrite:
            shutil.rmtree(raw_root)
        else:
            raise FileExistsError(f"raw root exists, pass --overwrite: {raw_root}")

    manifest_index = load_manifest_index(SOURCE_SPECS)
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_by_group: dict[str, set[str]] = defaultdict(set)
    missing_added: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for group_dir in sorted(OLD_RAW_ROOT.glob("*")):
        manifest = group_dir / "manifest.jsonl"
        if not manifest.is_file():
            continue
        group_name = group_dir.name
        for row in read_jsonl(manifest):
            grouped_rows[group_name].append(annotate_manifest_row(row, "baseline_medium_hard", variant_key))
            seen_by_group[group_name].add(qid(row))

    easy_summary: dict[str, Any] | None = None
    super_easy_summary: dict[str, Any] | None = None

    if include_easy32b:
        easy_qids, easy_summary = collect_easy32b_qids(SOURCE_SPECS)
        for group_name, qids in easy_qids.items():
            source_rows = manifest_index.get(group_name, {})
            for question_id in sorted(qids):
                if question_id in seen_by_group[group_name]:
                    continue
                row = source_rows.get(question_id)
                if row is None:
                    missing_added[group_name]["easy32b"] += 1
                    continue
                grouped_rows[group_name].append(annotate_manifest_row(row, "easy32b", variant_key))
                seen_by_group[group_name].add(question_id)

    if include_super_easy8b:
        super_qids, super_easy_summary = collect_super_easy8b_qids(SOURCE_SPECS)
        for group_name, qids in super_qids.items():
            source_rows = manifest_index.get(group_name, {})
            for question_id in sorted(qids):
                if question_id in seen_by_group[group_name]:
                    continue
                row = source_rows.get(question_id)
                if row is None:
                    missing_added[group_name]["super_easy8b"] += 1
                    continue
                grouped_rows[group_name].append(annotate_manifest_row(row, "super_easy8b", variant_key))
                seen_by_group[group_name].add(question_id)

    group_summaries: list[dict[str, Any]] = []
    for group_name in sorted(grouped_rows):
        rows = grouped_rows[group_name]
        output_manifest = raw_root / group_name / "manifest.jsonl"
        write_jsonl(output_manifest, rows)
        stage_counts = Counter(str(row.get("rl_source_stage", "unknown")) for row in rows)
        subset_counts = Counter(str(row.get("subset", "<missing>")) for row in rows)
        unans_counts = Counter(str(row.get("subset", "<missing>")) for row in rows if is_unanswerable_manifest_row(row))
        meta = {
            "group_name": group_name,
            "output_manifest": str(output_manifest),
            "rows": len(rows),
            "stage_counts": dict(sorted(stage_counts.items())),
            "subset_counts": dict(sorted(subset_counts.items())),
            "subset_unanswerable_counts": dict(sorted(unans_counts.items())),
        }
        (output_manifest.parent / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        group_summaries.append(meta)

    overall_stage = Counter()
    overall_subset = Counter()
    overall_unans = Counter()
    for rows in grouped_rows.values():
        overall_stage.update(str(row.get("rl_source_stage", "unknown")) for row in rows)
        overall_subset.update(str(row.get("subset", "<missing>")) for row in rows)
        overall_unans.update(str(row.get("subset", "<missing>")) for row in rows if is_unanswerable_manifest_row(row))

    summary = {
        "variant_key": variant_key,
        "variant_label": variant_label,
        "raw_root": str(raw_root),
        "old_raw_root": str(OLD_RAW_ROOT),
        "include_easy32b": include_easy32b,
        "include_super_easy8b": include_super_easy8b,
        "rows": sum(len(rows) for rows in grouped_rows.values()),
        "stage_counts": dict(sorted(overall_stage.items())),
        "subset_counts": dict(sorted(overall_subset.items())),
        "subset_unanswerable_counts": dict(sorted(overall_unans.items())),
        "missing_added_counts": {k: dict(v) for k, v in sorted(missing_added.items())},
        "easy32b_summary": easy_summary,
        "super_easy8b_summary": super_easy_summary,
        "groups": group_summaries,
    }
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw_root, summary


def run_cmd(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_old_pipeline(raw_root: Path, variant_label: str, *, num_workers: int) -> dict[str, Path]:
    variant_root = WORK_ROOT / variant_label
    balanced_root = variant_root / "balanced_half_unanswerable_dude_reduced"
    merged_root = variant_root / "merged_for_parquet"
    u25_root = variant_root / "merged_for_parquet_u25"
    prefilter_parquet = OUTPUT_DATA_ROOT / f"{variant_label}-insight_qwen_agent.prefilter.parquet"
    filtered_tmp_parquet = OUTPUT_DATA_ROOT / f"{variant_label}-insight_qwen_agent.estprompt_le11000.tmp.parquet"

    run_cmd(
        [
            "python",
            str(REPO_ROOT / "scripts/build_balanced_rl_subset_from_manifests.py"),
            "--input-root",
            str(raw_root),
            "--output-root",
            str(balanced_root),
            "--cap-mode",
            "manual",
            "--cap-count",
            "1182",
            "--balanced-half-subset",
            "dude",
            "--balanced-half-subset",
            "mveqa",
            "--balanced-half-subset",
            "veqa",
            "--subset-count-override",
            "dude=486",
            "--seed",
            "0",
        ]
    )
    run_cmd(
        [
            "python",
            str(REPO_ROOT / "scripts/merge_rl_manifests_for_create_parquet.py"),
            "--input-root",
            str(balanced_root),
            "--output-root",
            str(merged_root),
        ]
    )
    run_cmd(
        [
            "python",
            str(REPO_ROOT / "scripts/reduce_unanswerable_ratio_in_merged_rl_manifest.py"),
            "--input-root",
            str(merged_root),
            "--output-root",
            str(u25_root),
            "--unanswerable-denominator",
            "3",
            "--seed",
            "0",
        ]
    )
    run_cmd(
        [
            "python",
            str(INSIGHT_DOC_ROOT / "verl/recipe/vsearch/create_parquet_dataset.py"),
            "--dataset",
            "InSightDocRL",
            "--data_root",
            str(u25_root),
            "--split",
            "all",
            "--prompt",
            "insight_qwen_agent",
            "--output_path",
            str(prefilter_parquet),
            "--agent_name",
            "insight_qwen_agent",
            "--num_workers",
            str(num_workers),
            "--extra_options",
            '{"manifest_file": "manifest.jsonl"}',
        ],
        cwd=INSIGHT_DOC_ROOT,
    )
    run_cmd(
        [
            "python",
            str(REPO_ROOT / "scripts/build_rl_parquet_uniform_rescale_filtered_20260519.py"),
            "--source-parquet",
            str(prefilter_parquet),
            "--output-parquet",
            str(filtered_tmp_parquet),
            "--uniform-initial-rescale",
            "0.25",
            "--gpt-image-max-area",
            str(3500 * 3500),
            "--max-prompt-tokens",
            "11000",
        ]
    )
    return {
        "balanced_root": balanced_root,
        "merged_root": merged_root,
        "u25_root": u25_root,
        "prefilter_parquet": prefilter_parquet,
        "filtered_tmp_parquet": filtered_tmp_parquet,
    }


def row_extra_info(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    value = row["extra_info"]
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return json.loads(value)
    return dict(value or {})


def parquet_qid(row: pd.Series | dict[str, Any]) -> str:
    return str(row_extra_info(row).get("question_id"))


def parquet_subset(row: pd.Series | dict[str, Any]) -> str:
    return str(row_extra_info(row).get("subset"))


def parquet_unanswerable(row: pd.Series | dict[str, Any]) -> bool:
    return "not-answerable" in json.dumps(row_extra_info(row).get("question_type"), ensure_ascii=False).lower()


def old_targets() -> tuple[dict[str, dict[str, int]], set[str], float]:
    df = pd.read_parquet(OLD_FINAL_PARQUET)
    targets: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "unanswerable": 0})
    old_qids: set[str] = set()
    for record in df.to_dict("records"):
        subset = parquet_subset(record)
        targets[subset]["total"] += 1
        if parquet_unanswerable(record):
            targets[subset]["unanswerable"] += 1
        old_qids.add(parquet_qid(record))
    overall_ratio = sum(v["unanswerable"] for v in targets.values()) / float(sum(v["total"] for v in targets.values()))
    return dict(targets), old_qids, overall_ratio


def build_source_stage_map(raw_root: Path) -> dict[str, str]:
    stage_by_qid: dict[str, str] = {}
    priority = {"baseline_medium_hard": 0, "easy32b": 1, "super_easy8b": 2}
    for manifest in sorted(raw_root.glob("*/manifest.jsonl")):
        for row in read_jsonl(manifest):
            question_id = qid(row)
            stage = str(row.get("rl_source_stage", "unknown"))
            old = stage_by_qid.get(question_id)
            if old is None or priority.get(stage, 99) < priority.get(old, 99):
                stage_by_qid[question_id] = stage
    return stage_by_qid


def priority_sort_records(records: list[dict[str, Any]], old_qids: set[str], stage_by_qid: dict[str, str]) -> list[dict[str, Any]]:
    stage_rank = {"baseline_medium_hard": 0, "easy32b": 1, "super_easy8b": 2, "unknown": 3}

    def key(record: dict[str, Any]) -> tuple[int, int, str]:
        question_id = parquet_qid(record)
        stage = stage_by_qid.get(question_id, "unknown")
        return (0 if question_id in old_qids else 1, stage_rank.get(stage, 9), stable_key(question_id))

    return sorted(records, key=key)


def choose_ratio_count(total_available: int, positive_available: int, target_ratio: float) -> tuple[int, int]:
    if target_ratio <= 0:
        return total_available, 0
    negative_available = total_available - positive_available
    max_by_positive = math.floor(positive_available / target_ratio)
    max_by_negative = math.floor(negative_available / (1.0 - target_ratio))
    total = max(0, min(total_available, max_by_positive, max_by_negative))
    positive = int(round(total * target_ratio))
    positive = min(positive, positive_available)
    negative = min(total - positive, negative_available)
    total = positive + negative
    return total, positive


def add_stage_to_record(record: dict[str, Any], stage_by_qid: dict[str, str], variant_key: str) -> dict[str, Any]:
    out = dict(record)
    extra = row_extra_info(out)
    question_id = str(extra.get("question_id"))
    extra["rl_source_stage"] = stage_by_qid.get(question_id, "unknown")
    extra["rl_source_variant"] = variant_key
    out["extra_info"] = extra
    # Preserve old top-level routing for reward/training compatibility.
    out["data_source"] = "insight_doc_rl"
    return out


def ratio_match_final_parquet(
    *,
    input_parquet: Path,
    output_parquet: Path,
    raw_root: Path,
    variant_key: str,
) -> dict[str, Any]:
    targets, old_qids, old_overall_ratio = old_targets()
    stage_by_qid = build_source_stage_map(raw_root)
    df = pd.read_parquet(input_parquet)
    records = [add_stage_to_record(record, stage_by_qid, variant_key) for record in df.to_dict("records")]

    by_subset: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"answerable": [], "unanswerable": []})
    for record in records:
        subset = parquet_subset(record)
        key = "unanswerable" if parquet_unanswerable(record) else "answerable"
        by_subset[subset][key].append(record)

    selected_by_subset: dict[str, list[dict[str, Any]]] = {}
    zero_subset_pool: list[dict[str, Any]] = []
    nonzero_selected_total = 0
    nonzero_selected_unans = 0
    subset_selection: dict[str, Any] = {}

    for subset in sorted(by_subset):
        ans = priority_sort_records(by_subset[subset]["answerable"], old_qids, stage_by_qid)
        unans = priority_sort_records(by_subset[subset]["unanswerable"], old_qids, stage_by_qid)
        target = targets.get(subset, {"total": 0, "unanswerable": 0})
        target_ratio = (target["unanswerable"] / target["total"]) if target["total"] else 0.0
        if target_ratio <= 0:
            zero_subset_pool.extend(ans)
            selected_by_subset[subset] = []
            subset_selection[subset] = {
                "target_ratio": target_ratio,
                "available_answerable": len(ans),
                "available_unanswerable": len(unans),
                "initial_selected_answerable": len(ans),
                "initial_selected_unanswerable": 0,
                "zero_ratio_subset": True,
            }
            continue
        total_target, unans_target = choose_ratio_count(len(ans) + len(unans), len(unans), target_ratio)
        ans_target = total_target - unans_target
        chosen = ans[:ans_target] + unans[:unans_target]
        selected_by_subset[subset] = chosen
        nonzero_selected_total += len(chosen)
        nonzero_selected_unans += unans_target
        subset_selection[subset] = {
            "target_ratio": target_ratio,
            "available_answerable": len(ans),
            "available_unanswerable": len(unans),
            "selected_answerable": ans_target,
            "selected_unanswerable": unans_target,
            "zero_ratio_subset": False,
        }

    zero_subset_pool = priority_sort_records(zero_subset_pool, old_qids, stage_by_qid)
    if old_overall_ratio > 0 and nonzero_selected_unans > 0:
        max_total_for_overall = int(round(nonzero_selected_unans / old_overall_ratio))
        zero_allowed = max(0, min(len(zero_subset_pool), max_total_for_overall - nonzero_selected_total))
    else:
        zero_allowed = len(zero_subset_pool)
    selected_zero = zero_subset_pool[:zero_allowed]
    for record in selected_zero:
        selected_by_subset[parquet_subset(record)].append(record)

    selected: list[dict[str, Any]] = []
    for subset in sorted(selected_by_subset):
        selected.extend(selected_by_subset[subset])
    selected.sort(key=lambda record: stable_key(parquet_qid(record)))

    out_df = pd.DataFrame(selected, columns=df.columns)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_parquet, index=False)

    summary = summarize_parquet(output_parquet)
    summary.update(
        {
            "input_parquet": str(input_parquet),
            "output_parquet": str(output_parquet),
            "old_final_parquet": str(OLD_FINAL_PARQUET),
            "old_overall_unanswerable_ratio": old_overall_ratio,
            "zero_subset_candidates": len(zero_subset_pool),
            "zero_subset_selected": len(selected_zero),
            "subset_selection": subset_selection,
        }
    )
    summary_path = output_parquet.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def summarize_parquet(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    subset_counts: Counter[str] = Counter()
    subset_unans: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    stage_subset_counts: Counter[str] = Counter()
    old_targets_dict, old_qids, _ = old_targets()
    old_qid_count = 0
    rows = df.to_dict("records")
    for record in rows:
        extra = row_extra_info(record)
        subset = str(extra.get("subset"))
        question_id = str(extra.get("question_id"))
        stage = str(extra.get("rl_source_stage", "unknown"))
        subset_counts[subset] += 1
        stage_counts[stage] += 1
        stage_subset_counts[f"{subset}/{stage}"] += 1
        if parquet_unanswerable(record):
            subset_unans[subset] += 1
        if question_id in old_qids:
            old_qid_count += 1

    total_unans = sum(subset_unans.values())
    added_rows = len(rows) - stage_counts.get("baseline_medium_hard", 0)
    return {
        "parquet": str(path),
        "rows": len(rows),
        "old_qid_rows": old_qid_count,
        "old_qid_ratio": old_qid_count / len(rows) if rows else None,
        "added_rows_by_stage": {
            key: value for key, value in sorted(stage_counts.items()) if key != "baseline_medium_hard"
        },
        "added_rows_total": added_rows,
        "added_rows_ratio": added_rows / len(rows) if rows else None,
        "stage_counts": dict(sorted(stage_counts.items())),
        "stage_subset_counts": dict(sorted(stage_subset_counts.items())),
        "subset_counts": dict(sorted(subset_counts.items())),
        "subset_unanswerable_counts": dict(sorted(subset_unans.items())),
        "overall_unanswerable": total_unans,
        "overall_unanswerable_ratio": total_unans / len(rows) if rows else None,
        "subset_unanswerable_ratio": {
            subset: (subset_unans.get(subset, 0) / count if count else None)
            for subset, count in sorted(subset_counts.items())
        },
        "old_target_subset_counts": old_targets_dict,
    }


def build_variant(variant_key: str, *, overwrite: bool, num_workers: int, skip_create_parquet: bool) -> dict[str, Any]:
    config = VARIANTS[variant_key]
    variant_label = str(config["label"])
    raw_root, raw_summary = build_augmented_raw_root(
        variant_key=variant_key,
        variant_label=variant_label,
        include_easy32b=bool(config["include_easy32b"]),
        include_super_easy8b=bool(config["include_super_easy8b"]),
        overwrite=overwrite,
    )
    if skip_create_parquet:
        return {"raw_summary": raw_summary}

    paths = run_old_pipeline(raw_root, variant_label, num_workers=num_workers)
    final_parquet = OUTPUT_DATA_ROOT / f"{variant_label}-insight_qwen_agent.parquet"
    if final_parquet.exists() and overwrite:
        final_parquet.unlink()
    final_summary = ratio_match_final_parquet(
        input_parquet=paths["filtered_tmp_parquet"],
        output_parquet=final_parquet,
        raw_root=raw_root,
        variant_key=variant_key,
    )
    combined = {
        "variant_key": variant_key,
        "variant_label": variant_label,
        "raw_summary": raw_summary,
        "paths": {key: str(value) for key, value in paths.items()},
        "final_summary": final_summary,
    }
    combined_path = WORK_ROOT / variant_label / "build_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return combined


def main() -> int:
    args = parse_args()
    variants = args.variant or sorted(VARIANTS)
    outputs = []
    for variant_key in variants:
        outputs.append(
            build_variant(
                variant_key,
                overwrite=args.overwrite,
                num_workers=args.num_workers,
                skip_create_parquet=args.skip_create_parquet,
            )
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
