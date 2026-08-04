#!/usr/bin/env python3
"""Append mined legacy false-negative examples to the 200-row judge test set."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.build_judge_test_set_200_20260717 import compact_conversation_text, loads_line, to_text


BASE_DIR = REPO / "notes/generated/judge_test_set_200_hard_evalmix_20260717"
BASE_PARQUET = BASE_DIR / "judge_test_set_200.parquet"
CANDIDATE_PARQUET = BASE_DIR / "legacy_fn_candidate_mining/paired_legacy0_v2_1_candidates.parquet"
OUT_DIR = REPO / "notes/generated/judge_test_set_250_hard_evalmix_legacy_fn_20260717"

UNANSWERABLE_QUOTAS = {
    "dude": 20,
    "longdocurl": 3,
    "mmlongbench": 12,
}

LONG_GT_UIDS = [
    "mmlongbench_938",
    "mmlongbench_717",
    "mmlongbench_189",
    "mmlongbench_123",
    "mpdocvqa_62330",
    "mpdocvqa_45134",
    "mpdocvqa_52274",
    "mpdocvqa_45142",
    "mpdocvqa_50418",
    "mpdocvqa_50295",
    "mpdocvqa_61778",
    "mpdocvqa_33153",
    "mpdocvqa_63494",
    "mpdocvqa_59881",
    "dude_6fda75189bd01109cc6d7eb41e8d57de_f75089da828fc43e3fff7881bb25572b",
]

EXCLUDED_UIDS = {
    # GT is "Not answerable", but the candidate gives a substantive definition
    # rather than a clean absence/refusal answer.
    "mmlongbench_69",
}


def make_id(*parts: Any) -> str:
    raw = "||".join(to_text(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def score_value(score_obj: Any) -> float | None:
    if isinstance(score_obj, dict):
        value = score_obj.get("accuracy_reward", score_obj.get("score"))
    else:
        value = score_obj
    try:
        return float(value)
    except Exception:
        return None


def sample_key(row: pd.Series) -> tuple[str, str]:
    return str(row["uid"]), str(row.get("output_index", 0))


def load_score_file(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    score_path = Path(path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with score_path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            sample = loads_line(line)
            out[(str(sample.get("uid")), str(sample.get("output_index", 0)))] = sample
    return out


def model_name_from_path(path: str) -> str:
    parts = Path(path).parts
    for marker in ("full5_tool", "full5_no_tool_no_system"):
        if marker in parts:
            idx = parts.index(marker)
            if idx > 0:
                return parts[idx - 1]
    return "unknown"


def rescale_from_path(path: str) -> str | None:
    text = path.lower()
    if "rescale025" in text:
        return "0.25"
    if "rescale035" in text:
        return "0.35"
    if "rescale05" in text:
        return "0.5"
    return None


def infer_answerability(row: pd.Series) -> str:
    if bool(row.get("is_unanswerable")):
        return "unanswerable"
    return "answerable"


def infer_question_type(question: str, ground_truth: Any) -> str:
    gt = to_text(ground_truth).strip().upper()
    if len(gt) == 1 and gt in set("ABCDE"):
        return "mcq"
    if len(re.findall(r"\([A-E]\)", question)) >= 2:
        return "mcq"
    return "non_mcq"


def hydrate_trajectory_text(path: str | None, fallback: str) -> str:
    if not path:
        return fallback[:6000]
    conv_path = Path(path)
    if not conv_path.is_absolute():
        conv_path = REPO / conv_path
    if not conv_path.exists():
        return fallback[:6000]
    try:
        obj = loads_line(conv_path.read_bytes())
    except Exception:
        return fallback[:6000]
    if isinstance(obj, dict) and "conversation" in obj:
        return compact_conversation_text(obj.get("conversation") or [])
    return fallback[:6000]


def existing_signatures(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["qa_id"].astype(str), df["final_answer"].astype(str)))


def select_unanswerable(candidates: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["final_answer"] = candidates["v2_extracted"].fillna(candidates["answer"]).astype(str)
    mask = (
        candidates["is_unanswerable"].astype(bool)
        & candidates["final_answer"].str.contains(
            r"blank|redacted|not included|not provide|not visible|not shown|not found|not mention|"
            r"cannot|no .*shown|no .*provided|no .*visible|no .*listed|unanswerable|does not",
            case=False,
            regex=True,
            na=False,
        )
    )
    pool = candidates[mask].drop_duplicates(subset=["uid", "question"]).copy()
    pool = pool[~pool["uid"].astype(str).isin(EXCLUDED_UIDS)]
    pool["final_len"] = pool["final_answer"].str.len()
    seen = existing_signatures(base)
    selected: list[pd.DataFrame] = []
    for source, quota in UNANSWERABLE_QUOTAS.items():
        part = pool[pool["data_source"].eq(source)].sort_values(["final_len", "uid"])
        rows = []
        for _, row in part.iterrows():
            sig = (str(row["uid"]), str(row["final_answer"]))
            if sig in seen:
                continue
            rows.append(row)
            seen.add(sig)
            if len(rows) >= quota:
                break
        selected.append(pd.DataFrame(rows))
    return pd.concat(selected, ignore_index=True)


def select_long_gt(candidates: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    candidates["final_answer"] = candidates["v2_extracted"].fillna(candidates["answer"]).astype(str)
    pool = candidates[
        (~candidates["is_unanswerable"].astype(bool))
        & (candidates["gt_chars"] > 180)
        & (~candidates["final_answer"].str.contains(r"cannot determine|not possible|no mention|does not", case=False, regex=True, na=False))
    ].drop_duplicates(subset=["uid", "question"]).copy()
    seen = existing_signatures(base)
    rows = []
    for uid in LONG_GT_UIDS:
        matches = pool[pool["uid"].astype(str).str.startswith(uid)]
        if matches.empty:
            continue
        row = matches.sort_values(["gt_chars"], ascending=False).iloc[0]
        sig = (str(row["uid"]), str(row["final_answer"]))
        if sig in seen:
            continue
        rows.append(row)
        seen.add(sig)
    return pd.DataFrame(rows)


def candidate_to_row(row: pd.Series, v2_sample: dict[str, Any]) -> dict[str, Any]:
    final_answer = to_text(row.get("final_answer") or row.get("v2_extracted") or row.get("answer"))
    question = to_text(row["question"])
    ground_truth = to_text(row["ground_truth"])
    extra = v2_sample.get("extra_info") or {}
    conv_path = to_text(v2_sample.get("conversation_export_json_path"))
    if conv_path and not Path(conv_path).is_absolute():
        conv_path = str((REPO / conv_path).resolve())
    answerability = infer_answerability(row)
    flags = [
        "legacy_false_negative_mined",
        "paired_legacy0_single_call_v2_1",
    ]
    if answerability == "unanswerable":
        flags.append("unanswerable_correct_refusal")
    if int(row.get("gt_chars", 0)) > 100:
        flags.append("long_gt")
    return {
        "id": make_id("legacy_fn_supplement", row["v2_path"], row["uid"], row.get("output_index"), final_answer),
        "qa_id": to_text(row["uid"]),
        "source_split": "eval",
        "model_name": model_name_from_path(to_text(row["v2_path"])),
        "benchmark_or_data_source": to_text(row["data_source"]),
        "answerability": answerability,
        "question_type": infer_question_type(question, ground_truth),
        "question": question,
        "ground_truth": ground_truth,
        "mc_options_json": json.dumps(extra.get("mc_options"), ensure_ascii=False),
        "correct_option": None,
        "final_option": None,
        "final_answer": final_answer,
        "trajectory_path": conv_path,
        "trajectory_text": hydrate_trajectory_text(conv_path, final_answer),
        "old_judge_name": "paired_legacy0_single_call_v2_1",
        "old_judge_score": float(row["legacy_score"]),
        "heuristic_flags_json": json.dumps(flags, ensure_ascii=False),
        "human_label": 1,
        "failure_mode_json": json.dumps(["legacy_false_negative"], ensure_ascii=False),
        "notes": "added_20260717; final_manual_pass_accepted; paired legacy=0 and single_call_v2=1",
        "manual_pass_status": "accepted",
        "manual_pass_reviewer": "codex_20260717",
        "manual_pass_notes": "Question/GT/final answer manually inspected; retained as a legacy false-negative positive.",
        "rescale": rescale_from_path(to_text(row["v2_path"])),
        "source_path": to_text(row["v2_path"]),
        "row_origin": "paired_score_file_legacy0_v2_1",
        "response_truncated": bool(v2_sample.get("response_truncated")),
        "critical_failure": bool(v2_sample.get("critical_failure")),
        "n_tool_calls": v2_sample.get("n_tool_calls"),
        "global_step": None,
        "legacy_source_path": to_text(row["legacy_path"]),
        "single_call_v2_source_path": to_text(row["v2_path"]),
        "legacy_extracted_answer": to_text(row.get("legacy_extracted")),
        "single_call_v2_score": float(row["v2_score"]),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(BASE_PARQUET)
    candidates = pd.read_parquet(CANDIDATE_PARQUET)
    selected = pd.concat(
        [
            select_unanswerable(candidates, base),
            select_long_gt(candidates, base),
        ],
        ignore_index=True,
    )
    if len(selected) != 50:
        raise RuntimeError(f"expected 50 selected supplement rows, got {len(selected)}")

    score_cache: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    supplement_rows = []
    for _, row in selected.iterrows():
        path = to_text(row["v2_path"])
        if path not in score_cache:
            score_cache[path] = load_score_file(path)
        sample = score_cache[path][sample_key(row)]
        supplement_rows.append(candidate_to_row(row, sample))

    base_rows = base.to_dict(orient="records")
    all_rows = base_rows + supplement_rows
    out_df = pd.DataFrame(all_rows)
    out_df.to_parquet(OUT_DIR / "judge_test_set_250.parquet", index=False)
    write_jsonl(OUT_DIR / "judge_test_set_250.jsonl", all_rows)

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

    summary_lines = [
        "# Judge Test Set 250 With Legacy-FN Supplement",
        "",
        "Base artifact: `judge_test_set_200_hard_evalmix_20260717`.",
        "",
        f"Rows: {len(out_df)}",
        f"Supplement rows: {len(supplement_rows)}",
        "",
        "Supplement selection:",
        "",
        "- Paired score files where normal legacy `/scores/` marked the row incorrect and `scores_single_call_v2` marked the same rollout correct.",
        "- 35 high-confidence unanswerable/refusal rows: GT is unanswerable and final answer explicitly says the requested item is absent, blank, redacted, unsupported, or not visible.",
        "- 15 long-GT answerable rows: final answer clearly contains the required long phrase/list while legacy extracted a short fragment.",
        "- Supplement rows are marked `human_label=1` and `failure_mode_json=[\"legacy_false_negative\"]` after manual triage from question/GT/final-answer text.",
        "- Final manual pass: all 50 supplement rows accepted on 2026-07-17 after inspecting question, GT, and final answer. One weak candidate (`mmlongbench_69`) was excluded before this final artifact.",
        "",
        "Supplement by source:",
        "",
        out_df.tail(len(supplement_rows))["benchmark_or_data_source"].value_counts().to_markdown(),
        "",
        "Supplement by answerability:",
        "",
        out_df.tail(len(supplement_rows))["answerability"].value_counts().to_markdown(),
        "",
        "Outputs:",
        "",
        "- `judge_test_set_250.parquet`",
        "- `judge_test_set_250.jsonl`",
        "- `judge_test_set_250_annotation.csv`",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(out_df)} rows to {OUT_DIR}")
    print(out_df.tail(len(supplement_rows))["benchmark_or_data_source"].value_counts().to_string())
    print(out_df.tail(len(supplement_rows))["answerability"].value_counts().to_string())


if __name__ == "__main__":
    main()
