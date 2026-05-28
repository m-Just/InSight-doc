#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "synthetic_unanswerable_pipeline" / "all_easy_vr2_input_20260518"
)


@dataclass(frozen=True)
class Lane:
    name: str
    source_root: Path
    export_dir: Path
    dataset_basename: str


LANES: list[Lane] = [
    Lane(
        name="first_batch_rescale025_easy",
        source_root=REPO_ROOT
        / "artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed42_run1_az/verify_all_preview_c32",
        export_dir=Path(
            "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
            "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
            "qwen3-vl-32b-instruct/"
            "balanced_sample5k_seed42_run1_az_verify_all_preview_c32_rescale025_gpu01234567_seq/"
            "exported_conversations"
        ),
        dataset_basename="balanced_sample5k_seed42_run1_az_verify_all_preview_c32_rescale025_easy_vr2",
    ),
    Lane(
        name="first_batch_rescale035_easy",
        source_root=REPO_ROOT
        / "artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed43_disjoint_from_seed42_run1_az/"
        "verify_all_rescale_mix_3to2_seed43/rescale0375",
        export_dir=Path(
            "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
            "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
            "qwen3-vl-32b-instruct/"
            "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale035_gpu01234567_seq/"
            "exported_conversations"
        ),
        dataset_basename="balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale035_easy_vr2",
    ),
    Lane(
        name="first_batch_rescale05_easy",
        source_root=REPO_ROOT
        / "artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed43_disjoint_from_seed42_run1_az/"
        "verify_all_rescale_mix_3to2_seed43/rescale05",
        export_dir=Path(
            "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
            "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
            "qwen3-vl-32b-instruct/"
            "balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale05_gpu01234567_seq/"
            "exported_conversations"
        ),
        dataset_basename="balanced_sample5k_seed43_disjoint_from_seed42_run1_az_rescale05_easy_vr2",
    ),
    Lane(
        name="second_batch_rescale025_easy",
        source_root=REPO_ROOT
        / "artifacts/synthetic_unanswerable_pipeline/balanced_sample10k_seed44_disjoint_from_seed42_seed43_run1_az_noproxy/"
        "verify_all_rescale_mix_5to3to2_seed44/rescale025",
        export_dir=Path(
            "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
            "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
            "qwen3-vl-32b-instruct/"
            "balanced_sample10k_seed44_run1_az_verify_all_rescale025_gpu01234567_seq/"
            "exported_conversations"
        ),
        dataset_basename="balanced_sample10k_seed44_run1_az_verify_all_rescale025_easy_vr2",
    ),
    Lane(
        name="second_batch_rescale035_easy",
        source_root=REPO_ROOT
        / "artifacts/synthetic_unanswerable_pipeline/balanced_sample10k_seed44_disjoint_from_seed42_seed43_run1_az_noproxy/"
        "verify_all_rescale_mix_5to3to2_seed44/rescale035",
        export_dir=Path(
            "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
            "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
            "qwen3-vl-32b-instruct/"
            "balanced_sample10k_seed44_run1_az_verify_all_rescale035_gpu01234567_seq/"
            "exported_conversations"
        ),
        dataset_basename="balanced_sample10k_seed44_run1_az_verify_all_rescale035_easy_vr2",
    ),
    Lane(
        name="second_batch_rescale05_easy",
        source_root=REPO_ROOT
        / "artifacts/synthetic_unanswerable_pipeline/balanced_sample10k_seed44_disjoint_from_seed42_seed43_run1_az_noproxy/"
        "verify_all_rescale_mix_5to3to2_seed44/rescale05",
        export_dir=Path(
            "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/"
            "insight_qwen_agent_zoom_factor2_synthetic_unanswerable_resumable/"
            "qwen3-vl-32b-instruct/"
            "balanced_sample10k_seed44_run1_az_verify_all_rescale05_gpu01234567_seq/"
            "exported_conversations"
        ),
        dataset_basename="balanced_sample10k_seed44_run1_az_verify_all_rescale05_easy_vr2",
    ),
]


def load_easy_question_ids(export_dir: Path) -> list[str]:
    question_ids: list[str] = []
    for path in sorted(export_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        extra_info = obj.get("extra_info") or {}
        question_id = extra_info.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"missing question_id in {path}")
        question_ids.append(question_id)
    unique = sorted(set(question_ids))
    if len(unique) != len(question_ids):
        raise ValueError(f"duplicate question_ids in {export_dir}: {len(question_ids)} vs {len(unique)}")
    return unique


def write_question_ids(path: Path, question_ids: list[str]) -> None:
    path.write_text("".join(f"{qid}\n" for qid in question_ids), encoding="utf-8")


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
        json.dumps({"manifest_file": "manifest.normalized.jsonl"}),
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
        "final_parquet": str(output_root / "sft_data.vreasoner_v2_easy_merged.parquet"),
        "lanes": [],
    }

    all_qids: list[str] = []
    for lane in LANES:
        qids = load_easy_question_ids(lane.export_dir)
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
                "export_dir": str(lane.export_dir),
                "question_ids_file": str(qid_path),
                "parquet": str(parquet_path),
                "rows": len(df),
            }
        )

    unique_qids = sorted(set(all_qids))
    if len(unique_qids) != len(all_qids):
        raise ValueError(f"duplicate question_ids across lanes: {len(all_qids)} vs {len(unique_qids)}")

    merged = pd.concat(all_frames, ignore_index=True)
    merged_parquet = output_root / "sft_data.vreasoner_v2_easy_merged.parquet"
    merged.to_parquet(merged_parquet)

    write_question_ids(output_root / "question_ids.all_easy.txt", unique_qids)
    summary["total_question_ids"] = len(unique_qids)
    summary["total_rows"] = len(merged)

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
