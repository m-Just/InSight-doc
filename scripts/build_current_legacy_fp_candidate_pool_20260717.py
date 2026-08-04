#!/usr/bin/env python3
"""Build a strict-negative pool for mining current legacy judge false positives."""

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

from scripts.build_judge_test_set_200_20260717 import to_text
from scripts.build_judge_test_set_with_legacy_fn_supplement_20260717 import hydrate_trajectory_text

SRC_DIR = REPO / "notes/generated/judge_test_set_250_hard_evalmix_legacy_fn_20260717"
CANDIDATE_PARQUET = SRC_DIR / "legacy_fp_candidate_mining/paired_legacy1_v2_0_candidates.parquet"
OUT_DIR = REPO / "notes/generated/current_legacy_fp_mining_20260717"


def make_id(*parts: Any) -> str:
    raw = "||".join(to_text(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def flag_set(value: Any) -> set[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {to_text(item) for item in value}


def selected_pool(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["flag_set"] = candidates["flags"].apply(flag_set)

    def is_strict_negative(flags: set[str]) -> bool:
        return bool(
            "mc_wrong_final_option_scored_correct" in flags
            or "answerable_refusal_scored_correct" in flags
            or "answerable_absence_claim_scored_correct" in flags
        )

    candidates = candidates[candidates["flag_set"].apply(is_strict_negative)].copy()

    def priority(flags: set[str]) -> int:
        value = 0
        if "mc_wrong_final_option_scored_correct" in flags:
            value += 100
        if "answerable_refusal_scored_correct" in flags:
            value += 90
        if "answerable_absence_claim_scored_correct" in flags:
            value += 80
        return value

    candidates["priority"] = candidates["flag_set"].apply(priority)
    candidates["answer_len"] = candidates["legacy_extracted"].fillna("").astype(str).str.len()
    candidates = candidates.sort_values(
        ["priority", "data_source", "uid", "answer_len"],
        ascending=[False, True, True, True],
    )
    # Keep one rollout per question first so the mining pool is not dominated
    # by duplicate mmlite questions from multiple runs.
    return candidates.drop_duplicates(subset=["uid"], keep="first").reset_index(drop=True)


def candidate_to_row(row: pd.Series) -> dict[str, Any]:
    final_answer = to_text(row.get("legacy_extracted") or row.get("v2_extracted") or row.get("answer"))
    conv_path = to_text(row.get("conversation_export_json_path"))
    if conv_path and not Path(conv_path).is_absolute():
        conv_path = str((REPO / conv_path).resolve())
    flags = sorted(flag_set(row.get("flags")) | {"strict_negative_candidate", "current_legacy_fp_mining"})
    return {
        "id": make_id("current_legacy_fp_mining", row.get("v2_path"), row.get("uid"), row.get("output_index"), final_answer),
        "qa_id": to_text(row.get("uid")),
        "source_split": "eval",
        "model_name": to_text(row.get("model_name")),
        "benchmark_or_data_source": to_text(row.get("data_source")),
        "answerability": to_text(row.get("answerability")),
        "question_type": to_text(row.get("question_type")),
        "question": to_text(row.get("question")),
        "ground_truth": to_text(row.get("ground_truth")),
        "mc_options_json": json.dumps(None),
        "correct_option": None if pd.isna(row.get("correct_option")) else to_text(row.get("correct_option")),
        "final_option": None if pd.isna(row.get("final_option")) else to_text(row.get("final_option")),
        "final_answer": final_answer,
        "trajectory_path": conv_path,
        "trajectory_text": hydrate_trajectory_text(conv_path, final_answer),
        "old_judge_name": "paired_legacy1_single_call_v2_0_source_score",
        "old_judge_score": float(row.get("legacy_score", 1.0)),
        "heuristic_flags_json": json.dumps(flags, ensure_ascii=False),
        "human_label": 0,
        "failure_mode_json": json.dumps(["strict_negative_candidate"], ensure_ascii=False),
        "notes": "strict negative candidate for current legacy FP mining; label is deterministic/strict, not inferred from current legacy",
        "manual_pass_status": "",
        "manual_pass_reviewer": "",
        "manual_pass_notes": "",
        "rescale": to_text(row.get("rescale")),
        "source_path": to_text(row.get("v2_path")),
        "row_origin": "current_legacy_fp_mining_candidate",
        "response_truncated": bool(row.get("response_truncated")),
        "critical_failure": bool(row.get("critical_failure")),
        "n_tool_calls": None if pd.isna(row.get("n_tool_calls")) else row.get("n_tool_calls"),
        "global_step": None,
        "legacy_source_path": to_text(row.get("legacy_path")),
        "single_call_v2_source_path": to_text(row.get("v2_path")),
        "legacy_extracted_answer": to_text(row.get("legacy_extracted")),
        "single_call_v2_score": float(row.get("v2_score", 0.0)),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(CANDIDATE_PARQUET)
    selected = selected_pool(candidates)
    rows = [candidate_to_row(row) for _, row in selected.iterrows()]
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT_DIR / "strict_negative_candidates.parquet", index=False)
    write_jsonl(OUT_DIR / "strict_negative_candidates.jsonl", rows)
    out_df.to_csv(OUT_DIR / "strict_negative_candidates.csv", index=False)

    summary = [
        "# Current Legacy FP Mining Candidate Pool",
        "",
        f"Rows: {len(out_df)}",
        "",
        "Selection:",
        "",
        "- Starts from paired source-score disagreements where a prior `/scores/` value was 1 and `scores_single_call_v2` was 0.",
        "- Keeps strict-negative heuristic patterns only: wrong MC final option, answerable refusal, or answerable absence/uncertainty claim.",
        "- De-duplicates by `uid` before current-legacy scoring.",
        "- All rows are labeled `human_label=0` for mining; legacy-positive hits still require manual review before adding to the final test set.",
        "",
        "By source:",
        "",
        out_df["benchmark_or_data_source"].value_counts().to_markdown(),
        "",
        "By question type:",
        "",
        out_df["question_type"].value_counts().to_markdown(),
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"wrote {len(out_df)} rows to {OUT_DIR}")
    print(out_df["benchmark_or_data_source"].value_counts().to_string())
    print(out_df["question_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
