#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "answerable_vr2_inputs" / "all_hard_retry_vr2_input_20260518"


@dataclass(frozen=True)
class Lane:
    name: str
    source_root: Path
    manifest_file: str
    wrong_ids_path: Path
    dataset_basename: str


LANES: list[Lane] = [
    Lane(
        name="o3_part1_hard_retry",
        source_root=Path(
            "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424/"
            "0426_selected_train_part1/dpi200_aug_noaug_maxp40"
        ),
        manifest_file="manifest.jsonl",
        wrong_ids_path=Path(
            "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3d_resumable/wrong_question_ids.txt"
        ),
        dataset_basename="answerable_o3_part1_hard_retry_vr2",
    ),
    Lane(
        name="o3_part2a_hard_retry",
        source_root=Path(
            "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424/"
            "0426_selected_train_part2a/dpi200_aug_noaug_maxp40"
        ),
        manifest_file="manifest.jsonl",
        wrong_ids_path=Path(
            "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3a_resumable/wrong_question_ids.txt"
        ),
        dataset_basename="answerable_o3_part2a_hard_retry_vr2",
    ),
    Lane(
        name="o3_part2b_hard_retry",
        source_root=Path(
            "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424/"
            "0426_selected_train_part2b/dpi200_aug_noaug_maxp40"
        ),
        manifest_file="manifest.jsonl",
        wrong_ids_path=Path(
            "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3b_resumable/wrong_question_ids.txt"
        ),
        dataset_basename="answerable_o3_part2b_hard_retry_vr2",
    ),
    Lane(
        name="o3_part2c_hard_retry",
        source_root=Path(
            "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424/"
            "0426_selected_train_part2c/dpi200_aug_noaug_maxp40"
        ),
        manifest_file="manifest.jsonl",
        wrong_ids_path=Path(
            "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3c_resumable/wrong_question_ids.txt"
        ),
        dataset_basename="answerable_o3_part2c_hard_retry_vr2",
    ),
    Lane(
        name="arxiv_part1_hard_retry",
        source_root=Path(
            "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/"
            "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train/dpi200_aug_noaug_maxp40"
        ),
        manifest_file="manifest.jsonl",
        wrong_ids_path=Path(
            "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/"
            "wrong_question_ids.txt"
        ),
        dataset_basename="answerable_arxiv_part1_hard_retry_vr2",
    ),
    Lane(
        name="arxiv_part2_hard_retry",
        source_root=Path(
            "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/"
            "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train/dpi200_aug_noaug_maxp40"
        ),
        manifest_file="manifest.jsonl",
        wrong_ids_path=Path(
            "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part2_resumable/"
            "wrong_question_ids.txt"
        ),
        dataset_basename="answerable_arxiv_part2_hard_retry_vr2",
    ),
    Lane(
        name="arxiv_part3_hard_retry",
        source_root=Path(
            "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess/"
            "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
            "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0"
        ),
        manifest_file="manifest.jsonl",
        wrong_ids_path=Path(
            "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
            "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/"
            "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-"
            "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0-0426_train_part3_resumable/"
            "wrong_question_ids.txt"
        ),
        dataset_basename="answerable_arxiv_part3_hard_retry_vr2",
    ),
]


def load_question_ids(path: Path) -> list[str]:
    question_ids = sorted({line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()})
    if not question_ids:
        raise ValueError(f"no question_ids found in {path}")
    return question_ids


def write_question_ids(path: Path, question_ids: list[str]) -> None:
    path.write_text("".join(f"{qid}\n" for qid in question_ids), encoding="utf-8")


def load_answerable_question_id_set(source_root: Path, manifest_file: str) -> set[str]:
    manifest_path = source_root / manifest_file
    answerable: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            continue
        question_type = row.get("question_type")
        if question_type == "not-answerable":
            continue
        answerable.add(question_id)
    if not answerable:
        raise ValueError(f"no answerable question_ids found in {manifest_path}")
    return answerable


def run_create_parquet(lane: Lane, question_ids_path: Path, output_parquet: Path) -> None:
    cmd = [
        "python",
        str(REPO_ROOT / "recipe/vsearch/create_parquet_dataset.py"),
        "--dataset",
        "InSightDocMixedWithArxiv",
        "--data_root",
        str(lane.source_root),
        "--split",
        "all",
        "--prompt",
        "vreasoner",
        "--output_path",
        str(output_parquet),
        "--agent_name",
        "vreasoner_v2",
        "--num_workers",
        "1",
        "--question_id_file",
        str(question_ids_path),
        "--extra_options",
        json.dumps({"manifest_file": lane.manifest_file}),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> int:
    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    lane_dir = output_root / "lanes"
    lane_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    summary: dict[str, object] = {
        "output_root": str(output_root),
        "final_parquet": str(output_root / "sft_data.vreasoner_v2_hard_retry_merged.parquet"),
        "lanes": [],
    }

    all_qids: list[str] = []
    for lane in LANES:
        raw_qids = load_question_ids(lane.wrong_ids_path)
        answerable_qids = load_answerable_question_id_set(lane.source_root, lane.manifest_file)
        qids = sorted(qid for qid in raw_qids if qid in answerable_qids)
        if not qids:
            raise ValueError(f"no answerable hard-retry question_ids remain for {lane.name}")
        all_qids.extend(qids)

        lane_output_dir = lane_dir / lane.name
        lane_output_dir.mkdir(parents=True, exist_ok=True)
        qid_path = lane_output_dir / "question_ids.txt"
        parquet_path = lane_output_dir / f"{lane.dataset_basename}.parquet"

        write_question_ids(qid_path, qids)
        run_create_parquet(lane, qid_path, parquet_path)

        df = pd.read_parquet(parquet_path)
        all_frames.append(df)
        summary["lanes"].append(
            {
                "name": lane.name,
                "source_root": str(lane.source_root),
                "manifest_file": lane.manifest_file,
                "wrong_ids_path": str(lane.wrong_ids_path),
                "question_ids_file": str(qid_path),
                "parquet": str(parquet_path),
                "raw_selected_question_ids": len(raw_qids),
                "answerable_selected_question_ids": len(qids),
                "rows": len(df),
            }
        )

    unique_qids = sorted(set(all_qids))
    if len(unique_qids) != len(all_qids):
        raise ValueError(f"duplicate question_ids across lanes: {len(all_qids)} vs {len(unique_qids)}")

    merged = pd.concat(all_frames, ignore_index=True)
    merged_parquet = output_root / "sft_data.vreasoner_v2_hard_retry_merged.parquet"
    merged.to_parquet(merged_parquet)

    write_question_ids(output_root / "question_ids.all_hard_retry.txt", unique_qids)
    summary["total_question_ids"] = len(unique_qids)
    summary["total_rows"] = len(merged)

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
