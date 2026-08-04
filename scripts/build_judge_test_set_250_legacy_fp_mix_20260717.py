#!/usr/bin/env python3
"""Build a 250-row judge set with both legacy FN and FP stress cases."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_judge_test_set_200_20260717 import compact_conversation_text, loads_line, to_text
from scripts.build_judge_test_set_with_legacy_fn_supplement_20260717 import hydrate_trajectory_text

SRC_DIR = REPO / "notes/generated/judge_test_set_250_hard_evalmix_legacy_fn_20260717"
SRC_PARQUET = SRC_DIR / "judge_test_set_250.parquet"
CANDIDATE_PARQUET = SRC_DIR / "legacy_fp_candidate_mining/paired_legacy1_v2_0_candidates.parquet"
OUT_DIR = REPO / "notes/generated/judge_test_set_250_legacy_fp_mix_20260717"

# These were manually inspected from the paired legacy=1 / single_call_v2=0
# candidate pool. Labels are intentionally strict: refusal/uncertainty on an
# answerable question, or selecting an option inconsistent with the GT, is
# treated as incorrect even if the answer mentions the GT in passing.
SELECTED_FP_UIDS = [
    "Reasoning/Autonomous_Driving/Attention_TrafficSignal/0015",
    "Reasoning/Autonomous_Driving/Prediction_Intention_Ego/0118",
    "Perception/Autonomous_Driving/Attribute_Motion_Vehicle/0031",
    "Perception/Autonomous_Driving/Attribute_Motion_Vehicle/0093",
    "reason/monitoring/property/0451",
    "Perception/Autonomous_Driving/Attribute_Motion_Vehicle/0033",
    "longdocurl_free_gpt4o_4177240_35_56_8",
    "longdocurl_free_gpt4o_4182532_1_30_12",
    "Perception/Autonomous_Driving/Attribute_Motion_MultiPedestrians/0064",
    "Perception/Autonomous_Driving/Attribute_Motion_MultiPedestrians/0203",
    "Perception/Autonomous_Driving/Objects_Identify/0071",
    "Perception/Autonomous_Driving/Objects_Identify/0200",
    "Perception/Autonomous_Driving/Objects_Identify/0351",
    "Perception/Autonomous_Driving/Objects_Identify/0454",
    "Perception/Autonomous_Driving/Objects_Identify/1095",
    "Reasoning/Autonomous_Driving/Attention_TrafficSignal/0138",
    "Reasoning/Autonomous_Driving/Attention_TrafficSignal/0161",
    "Reasoning/Autonomous_Driving/Attention_TrafficSignal/0191",
    "Reasoning/Autonomous_Driving/Prediction_Intention_Ego/0006",
    "Reasoning/Autonomous_Driving/Prediction_Intention_Ego/0210",
]

# Remove 20 rows from the previously added legacy-FN supplement to keep the
# artifact size fixed. Prefer removing unanswerable FN rows so the remaining FN
# supplement still includes all long-GT answerable cases.
REMOVE_FN_BY_SOURCE = {
    "dude": 10,
    "mmlongbench": 7,
    "longdocurl": 3,
}


def make_id(*parts: Any) -> str:
    raw = "||".join(to_text(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_score_sample(row: pd.Series) -> dict[str, Any]:
    rel_path = Path(to_text(row["v2_path"]))
    path = rel_path if rel_path.is_absolute() else REPO / rel_path
    key = (to_text(row["uid"]), to_text(row.get("output_index", 0)))
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            sample = loads_line(line)
            sample_key = (to_text(sample.get("uid")), to_text(sample.get("output_index", 0)))
            if sample_key == key:
                return sample
    raise KeyError(f"missing score sample for {key} in {path}")


def selected_fp_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["legacy_extracted_text"] = candidates["legacy_extracted"].fillna("").astype(str)
    candidates["extracted_len"] = candidates["legacy_extracted_text"].str.len()

    # Prefer rows with the strongest explicit FP flags and concise accepted
    # answers. Drop duplicate UIDs after sorting.
    def priority(flags: list[str]) -> int:
        flags_set = set(flags)
        value = 0
        if "mc_wrong_final_option_scored_correct" in flags_set:
            value += 100
        if "answerable_refusal_scored_correct" in flags_set:
            value += 90
        if "answerable_absence_claim_scored_correct" in flags_set:
            value += 80
        return value

    candidates["priority"] = candidates["flags"].apply(priority)
    candidates = candidates[candidates["uid"].astype(str).isin(SELECTED_FP_UIDS)]
    candidates = candidates.sort_values(
        ["priority", "data_source", "uid", "extracted_len"],
        ascending=[False, True, True, True],
    )
    selected = candidates.drop_duplicates(subset=["uid"], keep="first").copy()
    missing = sorted(set(SELECTED_FP_UIDS) - set(selected["uid"].astype(str)))
    if missing:
        raise RuntimeError(f"missing selected FP candidates: {missing}")
    return selected


def fp_candidate_to_row(row: pd.Series) -> dict[str, Any]:
    sample = load_score_sample(row)
    extra = sample.get("extra_info") or {}
    conv_path = to_text(row.get("conversation_export_json_path") or extra.get("conversation_export_json_path"))
    if conv_path and not Path(conv_path).is_absolute():
        conv_path = str((REPO / conv_path).resolve())
    final_answer = to_text(row.get("legacy_extracted") or row.get("v2_extracted") or row.get("answer"))
    raw_flags = row.get("flags")
    flags = list(raw_flags) if isinstance(raw_flags, (list, tuple)) or hasattr(raw_flags, "tolist") else []
    if hasattr(raw_flags, "tolist"):
        flags = list(raw_flags.tolist())
    flags.extend(["legacy_false_positive_mined", "paired_legacy1_single_call_v2_0"])
    return {
        "id": make_id("legacy_fp_supplement", row["v2_path"], row["uid"], row.get("output_index"), final_answer),
        "qa_id": to_text(row["uid"]),
        "source_split": "eval",
        "model_name": to_text(row["model_name"]),
        "benchmark_or_data_source": to_text(row["data_source"]),
        "answerability": to_text(row["answerability"]),
        "question_type": to_text(row["question_type"]),
        "question": to_text(row["question"]),
        "ground_truth": to_text(row["ground_truth"]),
        "mc_options_json": json.dumps(extra.get("mc_options"), ensure_ascii=False),
        "correct_option": None if pd.isna(row.get("correct_option")) else to_text(row.get("correct_option")),
        "final_option": None if pd.isna(row.get("final_option")) else to_text(row.get("final_option")),
        "final_answer": final_answer,
        "trajectory_path": conv_path,
        "trajectory_text": hydrate_trajectory_text(conv_path, final_answer),
        "old_judge_name": "paired_legacy1_single_call_v2_0",
        "old_judge_score": float(row["legacy_score"]),
        "heuristic_flags_json": json.dumps(sorted(set(flags)), ensure_ascii=False),
        "human_label": 0,
        "failure_mode_json": json.dumps(["legacy_false_positive"], ensure_ascii=False),
        "notes": "added_20260717; final_manual_pass_accepted; paired legacy=1 and single_call_v2=0",
        "manual_pass_status": "accepted",
        "manual_pass_reviewer": "codex_20260717",
        "manual_pass_notes": "Question/GT/legacy-accepted answer manually inspected; retained as a legacy false-positive negative.",
        "rescale": to_text(row.get("rescale")),
        "source_path": to_text(row["v2_path"]),
        "row_origin": "paired_score_file_legacy1_v2_0",
        "response_truncated": bool(row.get("response_truncated")),
        "critical_failure": bool(row.get("critical_failure")),
        "n_tool_calls": None if pd.isna(row.get("n_tool_calls")) else row.get("n_tool_calls"),
        "global_step": None,
        "legacy_source_path": to_text(row["legacy_path"]),
        "single_call_v2_source_path": to_text(row["v2_path"]),
        "legacy_extracted_answer": to_text(row.get("legacy_extracted")),
        "single_call_v2_score": float(row["v2_score"]),
    }


def remove_fn_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    remove_indices: list[int] = []
    supplement = df[
        (df["row_origin"].astype(str) == "paired_score_file_legacy0_v2_1")
        & (df["answerability"].astype(str) == "unanswerable")
    ]
    for source, count in REMOVE_FN_BY_SOURCE.items():
        part = supplement[supplement["benchmark_or_data_source"].astype(str) == source]
        if len(part) < count:
            raise RuntimeError(f"not enough removable FN supplement rows for {source}: {len(part)} < {count}")
        remove_indices.extend(part.index[:count].tolist())
    removed = df.loc[remove_indices].copy()
    kept = df.drop(index=remove_indices).copy()
    return kept, removed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(SRC_PARQUET)
    candidates = pd.read_parquet(CANDIDATE_PARQUET)
    fp_candidates = selected_fp_candidates(candidates)
    fp_rows = [fp_candidate_to_row(row) for _, row in fp_candidates.iterrows()]

    kept, removed = remove_fn_rows(base)
    out_rows = kept.to_dict(orient="records") + fp_rows
    out_df = pd.DataFrame(out_rows)
    if len(out_df) != 250:
        raise RuntimeError(f"expected 250 rows, got {len(out_df)}")

    out_df.to_parquet(OUT_DIR / "judge_test_set_250.parquet", index=False)
    write_jsonl(OUT_DIR / "judge_test_set_250.jsonl", out_rows)

    annotation_cols = [
        "id",
        "source_split",
        "model_name",
        "benchmark_or_data_source",
        "answerability",
        "question_type",
        "question",
        "ground_truth",
        "final_answer",
        "old_judge_name",
        "old_judge_score",
        "single_call_v2_score",
        "legacy_extracted_answer",
        "heuristic_flags_json",
        "human_label",
        "failure_mode_json",
        "notes",
        "manual_pass_status",
        "manual_pass_reviewer",
        "manual_pass_notes",
        "trajectory_path",
    ]
    out_df.reindex(columns=annotation_cols).to_csv(OUT_DIR / "judge_test_set_250_annotation.csv", index=False)
    removed.to_csv(OUT_DIR / "removed_legacy_fn_rows.csv", index=False)
    fp_candidates.to_csv(OUT_DIR / "added_legacy_fp_candidates.csv", index=False)

    fp_df = pd.DataFrame(fp_rows)
    summary_lines = [
        "# Judge Test Set 250 Legacy FP Mix",
        "",
        f"Source artifact: `{SRC_DIR.name}`.",
        "",
        f"Rows: {len(out_df)}",
        f"Removed legacy-FN supplement rows: {len(removed)}",
        f"Added legacy-FP supplement rows: {len(fp_rows)}",
        "",
        "Composition:",
        "",
        out_df["human_label"].value_counts().sort_index().rename(index={0: "label_0", 1: "label_1"}).to_markdown(),
        "",
        "Added legacy-FP rows by source:",
        "",
        fp_df["benchmark_or_data_source"].value_counts().to_markdown(),
        "",
        "Removed legacy-FN rows by source:",
        "",
        removed["benchmark_or_data_source"].value_counts().to_markdown(),
        "",
        "Selection notes:",
        "",
        "- Added rows come from paired score files where legacy `/scores/` marked the row correct but `scores_single_call_v2` marked the same rollout incorrect.",
        "- Added rows were manually inspected from question/GT/legacy-accepted answer text.",
        "- Labels are strict: answerable refusal/uncertainty and MC option mismatch are labeled incorrect.",
        "- Removed rows are unanswerable legacy-FN supplement rows only; all long-GT answerable FN supplement rows are retained.",
        "",
        "Outputs:",
        "",
        "- `judge_test_set_250.parquet`",
        "- `judge_test_set_250.jsonl`",
        "- `judge_test_set_250_annotation.csv`",
        "- `added_legacy_fp_candidates.csv`",
        "- `removed_legacy_fn_rows.csv`",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(out_df)} rows to {OUT_DIR}")
    print("labels")
    print(out_df["human_label"].value_counts().sort_index().to_string())
    print("added FP by source")
    print(fp_df["benchmark_or_data_source"].value_counts().to_string())
    print("removed FN by source")
    print(removed["benchmark_or_data_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
