#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "answerable_vr2_inputs" / "by_rescale_20260518"


@dataclass(frozen=True)
class EasyLane:
    name: str
    rescale: str
    source_root: Path
    manifest_file: str
    export_dir: Path
    wrong_ids_path: Path
    dataset_basename: str


@dataclass(frozen=True)
class HardLane:
    name: str
    rescale: str
    source_root: Path
    manifest_file: str
    wrong_ids_path: Path
    dataset_basename: str


O3_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424")
ARXIV_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess")
OUTPUTS_ROOT = Path("/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs")
MMS_CONVERTED_ROOT = Path(
    "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/"
    "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed"
)
LEGACY_CONVERTED_ROOT = Path(
    "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/converted_sft/multi_agent_vsearch/"
    "arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed"
)


EASY_LANES: list[EasyLane] = [
    EasyLane(
        name="o3_part1_easy",
        rescale="025",
        source_root=O3_ROOT / "0426_selected_train_part1/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part1/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part1/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part1_easy_vr2",
    ),
    EasyLane(
        name="o3_part2a_easy",
        rescale="025",
        source_root=O3_ROOT / "0426_selected_train_part2a/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part2a/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part2a/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part2a_easy_vr2",
    ),
    EasyLane(
        name="o3_part2b_easy",
        rescale="025",
        source_root=O3_ROOT / "0426_selected_train_part2b/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part2b/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part2b/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part2b_easy_vr2",
    ),
    EasyLane(
        name="o3_part2c_easy",
        rescale="025",
        source_root=O3_ROOT / "0426_selected_train_part2c/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part2c/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part2c/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part2c_easy_vr2",
    ),
    EasyLane(
        name="o3_part3b_easy",
        rescale="035",
        source_root=O3_ROOT / "0426_selected_train_part2b/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area1800_rescale035_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part3b/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area1800_rescale035_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part3b/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part3b_easy_vr2",
    ),
    EasyLane(
        name="o3_part3d_easy",
        rescale="035",
        source_root=O3_ROOT / "0426_selected_train_part1/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area1800_rescale035_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part3d/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area1800_rescale035_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part3d/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part3d_easy_vr2",
    ),
    EasyLane(
        name="o3_part3a_easy",
        rescale="05",
        source_root=O3_ROOT / "0426_selected_train_part2a/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area2500_rescale05_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part3a/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area2500_rescale05_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part3a/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part3a_easy_vr2",
    ),
    EasyLane(
        name="o3_part3c_easy",
        rescale="05",
        source_root=O3_ROOT / "0426_selected_train_part2c/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area2500_rescale05_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part3c/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area2500_rescale05_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "O3_data_0424-dpi200_aug_noaug_maxp40/train_part3c/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part3c_easy_vr2",
    ),
    EasyLane(
        name="arxiv_part1_easy",
        rescale="025",
        source_root=ARXIV_ROOT / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part1/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part1/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part1_easy_vr2",
    ),
    EasyLane(
        name="arxiv_part2_easy",
        rescale="025",
        source_root=ARXIV_ROOT / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part2/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part2/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part2_easy_vr2",
    ),
    EasyLane(
        name="arxiv_part3_easy",
        rescale="025",
        source_root=ARXIV_ROOT
        / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
        "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0/train_part3/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0/train_part3/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part3_easy_vr2",
    ),
    EasyLane(
        name="arxiv_part4_easy",
        rescale="035",
        source_root=ARXIV_ROOT
        / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
        "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area1800_rescale035_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0/train_part4/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area1800_rescale035_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0/train_part4/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part4_easy_vr2",
    ),
    EasyLane(
        name="arxiv_part5_easy",
        rescale="05",
        source_root=ARXIV_ROOT
        / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
        "dpi200_aug_noaug_maxp40_jitter_seed0",
        manifest_file="manifest.jsonl",
        export_dir=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area2500_rescale05_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0/train_part5/exported_conversations",
        wrong_ids_path=OUTPUTS_ROOT
        / "insight_qwen_agent_zoom_factor2_area2500_rescale05_default_sys_0426_resumable/qwen3-vl-32b-instruct/"
        "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0/train_part5/converted_sft/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part5_easy_vr2",
    ),
]


HARD_LANES: list[HardLane] = [
    HardLane(
        name="o3_part1_hard_retry",
        rescale="025",
        source_root=O3_ROOT / "0426_selected_train_part1/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part1_hard_retry_vr2",
    ),
    HardLane(
        name="o3_part2a_hard_retry",
        rescale="025",
        source_root=O3_ROOT / "0426_selected_train_part2a/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=LEGACY_CONVERTED_ROOT / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2a_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part2a_hard_retry_vr2",
    ),
    HardLane(
        name="o3_part2b_hard_retry",
        rescale="025",
        source_root=O3_ROOT / "0426_selected_train_part2b/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2b_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part2b_hard_retry_vr2",
    ),
    HardLane(
        name="o3_part2c_hard_retry",
        rescale="025",
        source_root=O3_ROOT / "0426_selected_train_part2c/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2c_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part2c_hard_retry_vr2",
    ),
    HardLane(
        name="o3_part3b_hard_retry",
        rescale="035",
        source_root=O3_ROOT / "0426_selected_train_part2b/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3b_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part3b_hard_retry_vr2",
    ),
    HardLane(
        name="o3_part3d_hard_retry",
        rescale="035",
        source_root=O3_ROOT / "0426_selected_train_part1/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3d_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part3d_hard_retry_vr2",
    ),
    HardLane(
        name="o3_part3a_hard_retry",
        rescale="05",
        source_root=O3_ROOT / "0426_selected_train_part2a/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3a_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part3a_hard_retry_vr2",
    ),
    HardLane(
        name="o3_part3c_hard_retry",
        rescale="05",
        source_root=O3_ROOT / "0426_selected_train_part2c/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3c_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_o3_part3c_hard_retry_vr2",
    ),
    HardLane(
        name="arxiv_part1_hard_retry",
        rescale="025",
        source_root=ARXIV_ROOT / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part1_hard_retry_vr2",
    ),
    HardLane(
        name="arxiv_part2_hard_retry",
        rescale="025",
        source_root=ARXIV_ROOT / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train/dpi200_aug_noaug_maxp40",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part2_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part2_hard_retry_vr2",
    ),
    HardLane(
        name="arxiv_part3_hard_retry",
        rescale="025",
        source_root=ARXIV_ROOT
        / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
        "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0-0426_train_part3_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part3_hard_retry_vr2",
    ),
    HardLane(
        name="arxiv_part4_hard_retry",
        rescale="035",
        source_root=ARXIV_ROOT
        / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
        "dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0_pagedrop0.5_irrel0.3_seed0-0426_train_part4_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part4_hard_retry_vr2",
    ),
    HardLane(
        name="arxiv_part5_hard_retry",
        rescale="05",
        source_root=ARXIV_ROOT
        / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional/"
        "dpi200_aug_noaug_maxp40_jitter_seed0",
        manifest_file="manifest.jsonl",
        wrong_ids_path=MMS_CONVERTED_ROOT / "veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional-dpi200_aug_noaug_maxp40_jitter_seed0-0426_train_part5_resumable/wrong_question_ids.txt",
        dataset_basename="answerable_arxiv_part5_hard_retry_vr2",
    ),
]


def load_answerable_question_id_set(source_root: Path, manifest_file: str) -> set[str]:
    manifest_path = source_root / manifest_file
    answerable: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid:
            continue
        if row.get("question_type") == "not-answerable":
            continue
        answerable.add(qid)
    if not answerable:
        raise ValueError(f"no answerable question_ids found in {manifest_path}")
    return answerable


def load_export_question_ids(export_dir: Path) -> set[str]:
    qids: set[str] = set()
    for path in sorted(export_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        qid = (obj.get("extra_info") or {}).get("question_id")
        if isinstance(qid, str) and qid:
            qids.add(qid)
    if not qids:
        raise ValueError(f"no exported question_ids found in {export_dir}")
    return qids


def load_question_ids(path: Path) -> set[str]:
    qids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not qids:
        raise ValueError(f"no question_ids found in {path}")
    return qids


def write_question_ids(path: Path, question_ids: list[str]) -> None:
    path.write_text("".join(f"{qid}\n" for qid in question_ids), encoding="utf-8")


def run_create_parquet(source_root: Path, manifest_file: str, question_ids_path: Path, output_parquet: Path) -> None:
    cmd = [
        "python",
        str(REPO_ROOT / "recipe/vsearch/create_parquet_dataset.py"),
        "--dataset",
        "InSightDocMixedWithArxiv",
        "--data_root",
        str(source_root),
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
        json.dumps({"manifest_file": manifest_file}),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def build_easy() -> dict[str, object]:
    root = OUTPUT_ROOT / "easy"
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"output_root": str(root), "rescale_groups": {}}

    for rescale in ("025", "035", "05"):
        lanes = [lane for lane in EASY_LANES if lane.rescale == rescale]
        group_dir = root / f"rescale{rescale}"
        lane_dir = group_dir / "lanes"
        lane_dir.mkdir(parents=True, exist_ok=True)

        group_frames: list[pd.DataFrame] = []
        group_qids: list[str] = []
        group_info: list[dict[str, object]] = []

        for lane in lanes:
            answerable_qids = load_answerable_question_id_set(lane.source_root, lane.manifest_file)
            exported_qids = load_export_question_ids(lane.export_dir)
            wrong_qids = load_question_ids(lane.wrong_ids_path)
            qids = sorted(qid for qid in exported_qids if qid in answerable_qids and qid not in wrong_qids)
            if not qids:
                raise ValueError(f"no easy question_ids remain for {lane.name}")

            lane_output_dir = lane_dir / lane.name
            lane_output_dir.mkdir(parents=True, exist_ok=True)
            qid_path = lane_output_dir / "question_ids.txt"
            parquet_path = lane_output_dir / f"{lane.dataset_basename}.parquet"
            write_question_ids(qid_path, qids)
            run_create_parquet(lane.source_root, lane.manifest_file, qid_path, parquet_path)
            df = pd.read_parquet(parquet_path)
            group_frames.append(df)
            group_qids.extend(qids)
            group_info.append(
                {
                    "name": lane.name,
                    "source_root": str(lane.source_root),
                    "manifest_file": lane.manifest_file,
                    "export_dir": str(lane.export_dir),
                    "wrong_ids_path": str(lane.wrong_ids_path),
                    "question_ids_file": str(qid_path),
                    "parquet": str(parquet_path),
                    "rows": len(df),
                }
            )

        unique_qids = sorted(set(group_qids))
        if len(unique_qids) != len(group_qids):
            raise ValueError(f"duplicate easy question_ids within rescale{rescale}: {len(group_qids)} vs {len(unique_qids)}")

        merged = pd.concat(group_frames, ignore_index=True)
        merged_parquet = group_dir / f"sft_data.vreasoner_v2_easy_rescale{rescale}.parquet"
        merged.to_parquet(merged_parquet)
        write_question_ids(group_dir / "question_ids.txt", unique_qids)
        group_summary = {
            "rescale": rescale,
            "parquet": str(merged_parquet),
            "question_ids_file": str(group_dir / "question_ids.txt"),
            "rows": len(merged),
            "question_ids": len(unique_qids),
            "lanes": group_info,
        }
        summary["rescale_groups"][f"rescale{rescale}"] = group_summary

    return summary


def build_hard() -> dict[str, object]:
    root = OUTPUT_ROOT / "hard"
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"output_root": str(root), "rescale_groups": {}}

    for rescale in ("025", "035", "05"):
        lanes = [lane for lane in HARD_LANES if lane.rescale == rescale]
        group_dir = root / f"rescale{rescale}"
        lane_dir = group_dir / "lanes"
        lane_dir.mkdir(parents=True, exist_ok=True)

        group_frames: list[pd.DataFrame] = []
        group_qids: list[str] = []
        group_info: list[dict[str, object]] = []

        for lane in lanes:
            answerable_qids = load_answerable_question_id_set(lane.source_root, lane.manifest_file)
            raw_qids = load_question_ids(lane.wrong_ids_path)
            qids = sorted(qid for qid in raw_qids if qid in answerable_qids)
            if not qids:
                raise ValueError(f"no hard question_ids remain for {lane.name}")

            lane_output_dir = lane_dir / lane.name
            lane_output_dir.mkdir(parents=True, exist_ok=True)
            qid_path = lane_output_dir / "question_ids.txt"
            parquet_path = lane_output_dir / f"{lane.dataset_basename}.parquet"
            write_question_ids(qid_path, qids)
            run_create_parquet(lane.source_root, lane.manifest_file, qid_path, parquet_path)
            df = pd.read_parquet(parquet_path)
            group_frames.append(df)
            group_qids.extend(qids)
            group_info.append(
                {
                    "name": lane.name,
                    "source_root": str(lane.source_root),
                    "manifest_file": lane.manifest_file,
                    "wrong_ids_path": str(lane.wrong_ids_path),
                    "question_ids_file": str(qid_path),
                    "parquet": str(parquet_path),
                    "rows": len(df),
                }
            )

        unique_qids = sorted(set(group_qids))
        if len(unique_qids) != len(group_qids):
            raise ValueError(f"duplicate hard question_ids within rescale{rescale}: {len(group_qids)} vs {len(unique_qids)}")

        merged = pd.concat(group_frames, ignore_index=True)
        merged_parquet = group_dir / f"sft_data.vreasoner_v2_hard_retry_rescale{rescale}.parquet"
        merged.to_parquet(merged_parquet)
        write_question_ids(group_dir / "question_ids.txt", unique_qids)
        group_summary = {
            "rescale": rescale,
            "parquet": str(merged_parquet),
            "question_ids_file": str(group_dir / "question_ids.txt"),
            "rows": len(merged),
            "question_ids": len(unique_qids),
            "lanes": group_info,
        }
        summary["rescale_groups"][f"rescale{rescale}"] = group_summary

    return summary


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    easy_summary = build_easy()
    hard_summary = build_hard()
    summary = {"output_root": str(OUTPUT_ROOT), "easy": easy_summary, "hard": hard_summary}
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
