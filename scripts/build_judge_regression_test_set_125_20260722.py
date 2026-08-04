#!/usr/bin/env python3
"""Build a compact human-labeled judge regression set.

The set is intentionally small enough for repeated API judge runs while still
covering the main cases that matter for the Insight-Qwen judge:

- 50 diverse legacy-correct regression rows.
- 25 multi-target rows.
- 25 medium/long-answer rows.
- 25 unanswerable rows.

Most rows come from already manually labeled judge/audit artifacts. A small
multi-target slice uses deterministic arXiv structural QAs with synthetic
correct/partial answers so the set covers the newly added arXiv-style training
cases without depending on unapproved LLM rewrites.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notes/generated/judge_regression_test_set_125_20260722"

JUDGE_275 = ROOT / "notes/generated/judge_test_set_275_answerable_refusal_supplement_20260718/judge_test_set_275.parquet"
JUDGE_275_LEGACY = (
    ROOT
    / "notes/generated/judge_test_set_275_answerable_refusal_supplement_20260718"
    / "judge_eval_20260718_globalai/predictions_legacy.jsonl"
)
ARXIV_STRUCT_1K = (
    ROOT
    / "notes/generated/arxiv_struct_longdocurl_like_1k_uniform_multi_final_v5_pagechecked_20260722"
    / "arxiv_struct_longdocurl_like_deterministic_1000-insight_qwen_agent.parquet"
)
FAIRNESS_AUDITS = [
    ROOT / "notes/generated/legacy_judge_fairness_pair_audit_20260718/manual_labels.jsonl",
    ROOT / "notes/generated/legacy_judge_fairness_pair_audit_rescale025_20260718/manual_labels.jsonl",
]


UNANSWERABLE_GTS = {
    "not answerable",
    "[the information provided in the document cannot answer this question]",
}


REFUSAL_PAT = re.compile(
    r"\b("
    r"cannot answer|can not answer|cannot determine|can not determine|cannot be determined|"
    r"can not be determined|not enough information|unanswerable|not provided|not visible|"
    r"not shown|not stated|not available|does not provide|no information"
    r")\b",
    flags=re.IGNORECASE,
)


def stable_id(*parts: Any) -> str:
    key = "\n".join(str(part) for part in parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def parse_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if value is None:
        return None
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def word_count(value: Any) -> int:
    return len(re.findall(r"\w+", str(value or "")))


def answerability_from_gt(value: Any, fallback: Any = None) -> str:
    if isinstance(fallback, str) and fallback:
        return fallback
    text = str(value or "").strip().lower()
    return "unanswerable" if text in UNANSWERABLE_GTS else "answerable"


def looks_like_multi_target(question: Any, ground_truth: Any) -> bool:
    items = parse_list(ground_truth)
    if not items or len(items) <= 1:
        return False
    question_norm = str(question or "").lower()
    if re.search(
        r"\b("
        r"list|all|what are|which sections|which tables|which titles|which figures|"
        r"which .*names|enumerate|features|applications|services|components|top 3|route"
        r")\b",
        question_norm,
    ):
        return True
    # Some datasets encode alternative aliases as lists. Treat those as
    # single-target unless the question itself asks for multiple targets.
    return False


def is_refusal_like(value: Any) -> bool:
    return bool(REFUSAL_PAT.search(str(value or "")))


def load_legacy_predictions(path: Path) -> dict[str, dict[str, Any]]:
    preds: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            preds[str(row["id"])] = row
    return preds


def normalize_275_rows() -> pd.DataFrame:
    df = pd.read_parquet(JUDGE_275).copy()
    legacy = load_legacy_predictions(JUDGE_275_LEGACY)
    df["legacy_pred"] = df["id"].map(lambda x: int(legacy[str(x)]["judge_pred"]))
    df["legacy_score_current"] = df["id"].map(lambda x: float(legacy[str(x)]["judge_score"]))
    df["legacy_correct"] = df["legacy_pred"].astype(int) == df["human_label"].astype(int)
    df["source_pool"] = "judge_test_set_275"
    df["source_row_id"] = df["id"].astype(str)
    df["benchmark_or_data_source"] = df["benchmark_or_data_source"].fillna("unknown")
    df["answerability"] = [
        answerability_from_gt(gt, ans)
        for gt, ans in zip(df["ground_truth"], df.get("answerability", pd.Series([None] * len(df))))
    ]
    return df


def normalize_fairness_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in FAIRNESS_AUDITS:
        if not path.exists():
            continue
        audit_name = path.parent.name
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                for model_key in ("base", "rl"):
                    final_answer = item.get(f"{model_key}_answer")
                    human_label = int(bool(item.get(f"manual_{model_key}_correct")))
                    legacy_score = float(item.get(f"{model_key}_score") or 0.0)
                    source_path = item.get(f"{model_key}_conversation_export_json_path")
                    row_id = stable_id(audit_name, item.get("audit_id"), model_key, final_answer)
                    rows.append(
                        {
                            "id": row_id,
                            "qa_id": item.get("question_id"),
                            "source_split": "manual_fairness_audit",
                            "model_name": model_key,
                            "benchmark_or_data_source": item.get("data_source") or "unknown",
                            "answerability": answerability_from_gt(
                                item.get("ground_truth"),
                                "unanswerable" if item.get("is_not_answerable") else "answerable",
                            ),
                            "question_type": "non_mcq",
                            "question": item.get("question"),
                            "ground_truth": item.get("ground_truth"),
                            "mc_options_json": None,
                            "correct_option": None,
                            "final_option": None,
                            "final_answer": final_answer,
                            "trajectory_path": source_path,
                            "trajectory_text": None,
                            "old_judge_name": "legacy",
                            "old_judge_score": legacy_score,
                            "heuristic_flags_json": None,
                            "human_label": human_label,
                            "failure_mode_json": None,
                            "notes": f"manual fairness audit; issue={item.get('manual_issue') or ''}; note={item.get('manual_note') or ''}",
                            "rescale": "0.5" if "rescale025" not in audit_name else "0.25",
                            "source_path": source_path,
                            "row_origin": audit_name,
                            "response_truncated": False,
                            "critical_failure": False,
                            "n_tool_calls": None,
                            "global_step": None,
                            "manual_pass_status": "accepted_from_prior_manual_audit",
                            "manual_pass_reviewer": "prior_audit",
                            "manual_pass_notes": item.get("manual_note"),
                            "legacy_source_path": source_path,
                            "single_call_v2_source_path": None,
                            "legacy_extracted_answer": None,
                            "single_call_v2_score": None,
                            "gt_chars": len(str(item.get("ground_truth") or "")),
                            "answer_chars": len(str(final_answer or "")),
                            "gt_item_count": len(parse_list(item.get("ground_truth")) or []),
                            "current_legacy_score_json": None,
                            "legacy_pred": int(legacy_score > 0.0),
                            "legacy_score_current": legacy_score,
                            "legacy_correct": int(legacy_score > 0.0) == human_label,
                            "source_pool": audit_name,
                            "source_row_id": f"{audit_name}:{item.get('audit_id')}:{model_key}",
                        }
                    )
    return pd.DataFrame(rows)


def extract_question_from_prompt(prompt: Any) -> str | None:
    if not isinstance(prompt, (list, tuple)):
        try:
            prompt = list(prompt)
        except TypeError:
            return None
    for message in reversed(prompt):
        if isinstance(message, dict) and message.get("role") == "user":
            text = str(message.get("content") or "")
            return re.sub(r"^(?:<image>)+", "", text).strip()
    return None


def normalize_answer_items(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    parsed = parse_list(value)
    if parsed:
        return [str(item) for item in parsed if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def build_synthetic_final_answer(items: list[str], *, correct: bool) -> str:
    shown_items = items if correct or len(items) <= 1 else items[:-1]
    prefix = "The matching items are" if len(shown_items) > 1 else "The matching item is"
    return f"{prefix}: " + "; ".join(shown_items)


def normalize_arxiv_struct_rows() -> pd.DataFrame:
    if not ARXIV_STRUCT_1K.exists():
        return pd.DataFrame()

    src = pd.read_parquet(ARXIV_STRUCT_1K)
    rows: list[dict[str, Any]] = []
    per_subset_counts: Counter[str] = Counter()
    for idx, item in src.iterrows():
        extra_info = item.get("extra_info") if isinstance(item.get("extra_info"), dict) else {}
        reward_model = item.get("reward_model") if isinstance(item.get("reward_model"), dict) else {}
        answer_items = normalize_answer_items(extra_info.get("answer_items"))
        if len(answer_items) <= 1:
            continue
        subset = str(extra_info.get("subset") or extra_info.get("question_type") or item.get("data_source") or "unknown")
        if per_subset_counts[subset] >= 12:
            continue
        question = extra_info.get("question") or extract_question_from_prompt(item.get("prompt"))
        ground_truth = reward_model.get("ground_truth")
        if not question or not ground_truth:
            continue
        per_subset_counts[subset] += 1

        for correct in (True, False):
            row_id = stable_id("arxiv_struct_pagechecked_synthetic_answer", idx, correct, question, ground_truth)
            rows.append(
                {
                    "id": row_id,
                    "qa_id": extra_info.get("question_id"),
                    "source_split": "arxiv_struct_pagechecked_synthetic",
                    "model_name": "synthetic_correct_answer" if correct else "synthetic_partial_answer",
                    "benchmark_or_data_source": item.get("data_source") or f"arxiv_struct_{subset}_answerable",
                    "answerability": "answerable",
                    "question_type": "non_mcq",
                    "question": question,
                    "ground_truth": ground_truth,
                    "mc_options_json": None,
                    "correct_option": None,
                    "final_option": None,
                    "final_answer": build_synthetic_final_answer(answer_items, correct=correct),
                    "trajectory_path": None,
                    "trajectory_text": None,
                    "old_judge_name": None,
                    "old_judge_score": None,
                    "heuristic_flags_json": json.dumps(
                        {
                            "synthetic_arxiv_struct": True,
                            "synthetic_answer_correct": correct,
                            "source_answer_items": answer_items,
                        },
                        ensure_ascii=False,
                    ),
                    "human_label": int(correct),
                    "failure_mode_json": None,
                    "notes": "Deterministic arXiv structural QA with synthetic correct/partial answer for judge regression.",
                    "rescale": str(extra_info.get("initial_rescale") or ""),
                    "source_path": str(ARXIV_STRUCT_1K),
                    "row_origin": "arxiv_struct_pagechecked_synthetic_answer",
                    "response_truncated": False,
                    "critical_failure": False,
                    "n_tool_calls": None,
                    "global_step": None,
                    "manual_pass_status": "pending",
                    "manual_pass_reviewer": None,
                    "manual_pass_notes": None,
                    "legacy_source_path": None,
                    "single_call_v2_source_path": None,
                    "legacy_extracted_answer": None,
                    "single_call_v2_score": None,
                    "current_legacy_score_json": None,
                    "legacy_pred": -1,
                    "legacy_score_current": -1.0,
                    "legacy_correct": False,
                    "source_pool": "arxiv_struct_pagechecked_synthetic_answer",
                    "source_row_id": f"{ARXIV_STRUCT_1K}:{idx}:{'correct' if correct else 'partial'}",
                }
            )
    return pd.DataFrame(rows)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in (
        "qa_id",
        "source_split",
        "model_name",
        "benchmark_or_data_source",
        "answerability",
        "question_type",
        "question",
        "ground_truth",
        "mc_options_json",
        "correct_option",
        "final_option",
        "final_answer",
        "trajectory_path",
        "trajectory_text",
        "old_judge_name",
        "old_judge_score",
        "heuristic_flags_json",
        "human_label",
        "failure_mode_json",
        "notes",
        "rescale",
        "source_path",
        "row_origin",
        "response_truncated",
        "critical_failure",
        "n_tool_calls",
        "global_step",
        "manual_pass_status",
        "manual_pass_reviewer",
        "manual_pass_notes",
        "legacy_source_path",
        "single_call_v2_source_path",
        "legacy_extracted_answer",
        "single_call_v2_score",
        "gt_chars",
        "answer_chars",
        "gt_item_count",
        "current_legacy_score_json",
    ):
        if col not in df.columns:
            df[col] = None
    df["id"] = df["id"].astype(str)
    df["human_label"] = df["human_label"].astype(int)
    df["question_type"] = df["question_type"].fillna("non_mcq")
    df["answerability"] = [answerability_from_gt(gt, ans) for gt, ans in zip(df["ground_truth"], df["answerability"])]
    df["gt_words"] = df["ground_truth"].map(word_count)
    df["answer_words"] = df["final_answer"].map(word_count)
    df["gt_item_count"] = df["ground_truth"].map(lambda x: len(parse_list(x) or []))
    df["is_multi_target"] = [looks_like_multi_target(q, gt) for q, gt in zip(df["question"], df["ground_truth"])]
    df["is_refusal_like_answer"] = df["final_answer"].map(is_refusal_like)
    df["legacy_correct"] = df["legacy_correct"].astype(bool)
    df["legacy_pred"] = df["legacy_pred"].astype(int)
    df["legacy_score_current"] = df["legacy_score_current"].astype(float)
    return df


def round_robin_select(
    df: pd.DataFrame,
    n: int,
    *,
    already_selected: set[str],
    bucket: str,
    reason: str,
    label_targets: dict[int, int] | None = None,
) -> pd.DataFrame:
    pool = df[~df["id"].isin(already_selected)].copy()
    if pool.empty:
        raise ValueError(f"empty candidate pool for {bucket}")

    selected: list[pd.Series] = []
    by_label_counts: Counter[int] = Counter()

    # Stable ordering favors diversity first, then shorter/non-refusal rows for
    # less-borderline examples, with id as deterministic tiebreaker.
    pool["_source_key"] = (
        pool["benchmark_or_data_source"].astype(str)
        + "|"
        + pool["row_origin"].astype(str)
        + "|"
        + pool["human_label"].astype(str)
    )
    groups = {
        key: group.sort_values(
            by=["is_refusal_like_answer", "gt_words", "answer_words", "id"],
            ascending=[True, True, True, True],
        ).to_dict("records")
        for key, group in pool.groupby("_source_key", sort=True)
    }
    keys = sorted(groups)

    while len(selected) < n:
        made_progress = False
        for key in keys:
            if len(selected) >= n:
                break
            rows = groups[key]
            while rows:
                candidate = rows.pop(0)
                label = int(candidate["human_label"])
                if label_targets and by_label_counts[label] >= label_targets.get(label, n):
                    continue
                selected.append(pd.Series(candidate))
                already_selected.add(str(candidate["id"]))
                by_label_counts[label] += 1
                made_progress = True
                break
        if not made_progress:
            break

    if len(selected) < n and label_targets:
        # Fill any remainder without strict label targets.
        rest = pool[~pool["id"].isin(already_selected)].sort_values(
            by=["benchmark_or_data_source", "row_origin", "is_refusal_like_answer", "gt_words", "id"]
        )
        for _, row in rest.iterrows():
            if len(selected) >= n:
                break
            selected.append(row)
            already_selected.add(str(row["id"]))

    if len(selected) < n:
        raise ValueError(f"only selected {len(selected)}/{n} rows for {bucket}")

    out = pd.DataFrame(selected).head(n).copy()
    out["test_bucket"] = bucket
    out["selection_reason"] = reason
    return out


def main() -> None:
    df = pd.concat(
        [normalize_275_rows(), normalize_fairness_rows(), normalize_arxiv_struct_rows()],
        ignore_index=True,
        sort=False,
    )
    df = enrich(df)
    df.loc[df["source_pool"].eq("arxiv_struct_pagechecked_synthetic_answer"), "is_multi_target"] = True

    # Deduplicate exact same judged case; keep richer 275 metadata first.
    df["_dedup_key"] = [
        stable_id(q, gt, ans)
        for q, gt, ans in zip(df["question"], df["ground_truth"], df["final_answer"])
    ]
    df["_pool_rank"] = df["source_pool"].map(lambda x: 0 if x == "judge_test_set_275" else 1)
    df = df.sort_values(["_pool_rank", "id"]).drop_duplicates("_dedup_key", keep="first")

    selected_ids: set[str] = set()
    buckets: list[pd.DataFrame] = []

    arxiv_struct_multi_pool = df[
        df["source_pool"].eq("arxiv_struct_pagechecked_synthetic_answer")
        & df["is_multi_target"]
        & df["answerability"].eq("answerable")
        & df["question_type"].eq("non_mcq")
    ].copy()
    buckets.append(
        round_robin_select(
            arxiv_struct_multi_pool,
            10,
            already_selected=selected_ids,
            bucket="multi_target",
            reason="Deterministic arXiv structural multi-target QA with synthetic correct/partial answer.",
            label_targets={1: 6, 0: 4},
        )
    )

    multi_pool = df[
        df["is_multi_target"]
        & ~df["source_pool"].eq("arxiv_struct_pagechecked_synthetic_answer")
        & df["answerability"].eq("answerable")
        & df["question_type"].eq("non_mcq")
        & ~df["critical_failure"].fillna(False).astype(bool)
        & ~df["response_truncated"].fillna(False).astype(bool)
    ].copy()
    buckets.append(
        round_robin_select(
            multi_pool,
            15,
            already_selected=selected_ids,
            bucket="multi_target",
            reason="GT has multiple requested targets and the question asks for a list/selection.",
            label_targets={1: 10, 0: 5},
        )
    )

    long_pool = df[
        ~df["is_multi_target"]
        & df["answerability"].eq("answerable")
        & df["question_type"].eq("non_mcq")
        & df["gt_words"].ge(10)
        & ~df["critical_failure"].fillna(False).astype(bool)
        & ~df["response_truncated"].fillna(False).astype(bool)
    ].copy()
    buckets.append(
        round_robin_select(
            long_pool,
            25,
            already_selected=selected_ids,
            bucket="long_answer",
            reason="Non-multi-target answerable question with medium/long GT answer.",
            label_targets={1: 15, 0: 10},
        )
    )

    unans_pool = df[
        df["answerability"].eq("unanswerable")
        & df["question_type"].eq("non_mcq")
        & ~df["critical_failure"].fillna(False).astype(bool)
        & ~df["response_truncated"].fillna(False).astype(bool)
    ].copy()
    buckets.append(
        round_robin_select(
            unans_pool,
            25,
            already_selected=selected_ids,
            bucket="unanswerable",
            reason="Question/GT is unanswerable; tests refusal correctness.",
            label_targets={1: 18, 0: 7},
        )
    )

    regression_pool = df[
        df["legacy_correct"]
        & ~df["id"].isin(selected_ids)
        & ~df["critical_failure"].fillna(False).astype(bool)
        & ~df["response_truncated"].fillna(False).astype(bool)
    ].copy()
    buckets.append(
        round_robin_select(
            regression_pool,
            50,
            already_selected=selected_ids,
            bucket="legacy_correct_regression",
            reason="Diverse row that legacy already got right; guards against regressions.",
            label_targets={1: 25, 0: 25},
        )
    )

    out = pd.concat(buckets, ignore_index=True, sort=False)
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")
    out["manual_inspection_status"] = "pending"
    out["manual_inspection_notes"] = ""

    required_cols = [
        "id",
        "test_bucket",
        "selection_reason",
        "source_pool",
        "source_row_id",
        "qa_id",
        "benchmark_or_data_source",
        "answerability",
        "question_type",
        "question",
        "ground_truth",
        "final_answer",
        "human_label",
        "legacy_pred",
        "legacy_correct",
        "legacy_score_current",
        "old_judge_score",
        "row_origin",
        "rescale",
        "gt_words",
        "answer_words",
        "gt_item_count",
        "is_multi_target",
        "is_refusal_like_answer",
        "trajectory_path",
        "source_path",
        "notes",
        "manual_inspection_status",
        "manual_inspection_notes",
    ]
    remaining = [col for col in out.columns if col not in required_cols]
    out = out[required_cols + remaining]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / "judge_regression_test_set_125.parquet"
    jsonl_path = OUT_DIR / "judge_regression_test_set_125.jsonl"
    csv_path = OUT_DIR / "judge_regression_test_set_125_annotation.csv"
    review_path = OUT_DIR / "manual_review.md"
    summary_path = OUT_DIR / "README.md"

    out.to_parquet(parquet_path, index=False)
    out.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    out.to_csv(csv_path, index=False)

    summary: dict[str, Any] = {
        "rows": len(out),
        "bucket_counts": out["test_bucket"].value_counts().sort_index().to_dict(),
        "label_counts": out["human_label"].value_counts().sort_index().to_dict(),
        "legacy_correct_counts": out["legacy_correct"].value_counts().sort_index().to_dict(),
        "answerability_counts": out["answerability"].value_counts().sort_index().to_dict(),
        "benchmark_counts": out["benchmark_or_data_source"].value_counts().sort_index().to_dict(),
        "bucket_label_counts": {
            bucket: group["human_label"].value_counts().sort_index().to_dict()
            for bucket, group in out.groupby("test_bucket")
        },
        "bucket_benchmark_counts": {
            bucket: group["benchmark_or_data_source"].value_counts().sort_index().to_dict()
            for bucket, group in out.groupby("test_bucket")
        },
    }

    lines = [
        "# Judge Regression Test Set 125",
        "",
        "This 125-row set is built mostly from previously manually labeled artifacts, plus a small deterministic arXiv structural synthetic-answer slice:",
        "",
        f"- `{JUDGE_275.relative_to(ROOT)}`",
        "- `notes/generated/legacy_judge_fairness_pair_audit_20260718/manual_labels.jsonl`",
        "- `notes/generated/legacy_judge_fairness_pair_audit_rescale025_20260718/manual_labels.jsonl`",
        f"- `{ARXIV_STRUCT_1K.relative_to(ROOT)}`",
        "",
        "Buckets:",
        "",
        "| bucket | rows | label=1 | label=0 | legacy-correct |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket, group in out.groupby("test_bucket", sort=True):
        lines.append(
            f"| `{bucket}` | {len(group)} | {int((group['human_label'] == 1).sum())} | "
            f"{int((group['human_label'] == 0).sum())} | {int(group['legacy_correct'].sum())} |"
        )
    lines.extend(["", "Benchmark/source distribution:", "", "| source | rows |", "|---|---:|"])
    for source, count in out["benchmark_or_data_source"].value_counts().sort_index().items():
        lines.append(f"| `{source}` | {count} |")
    lines.extend(["", "Files:", "", "- `judge_regression_test_set_125.parquet`", "- `judge_regression_test_set_125.jsonl`", "- `judge_regression_test_set_125_annotation.csv`", "- `manual_review.md`"])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    review_lines = ["# Manual Review Sheet", ""]
    for bucket, group in out.groupby("test_bucket", sort=True):
        review_lines.extend([f"## {bucket}", ""])
        for idx, row in group.reset_index(drop=True).iterrows():
            review_lines.extend(
                [
                    f"### {bucket} #{idx + 1}: `{row['id']}`",
                    "",
                    f"- source: `{row['benchmark_or_data_source']}` / `{row['row_origin']}` / `{row['source_pool']}`",
                    f"- label: `{row['human_label']}`, legacy_pred: `{row['legacy_pred']}`, legacy_correct: `{row['legacy_correct']}`",
                    f"- answerability: `{row['answerability']}`, gt_words: `{row['gt_words']}`, gt_item_count: `{row['gt_item_count']}`",
                    f"- question: {str(row['question']).replace(chr(10), ' ')}",
                    f"- GT: {str(row['ground_truth']).replace(chr(10), ' ')}",
                    f"- final_answer: {str(row['final_answer']).replace(chr(10), ' ')}",
                    "- manual inspection: pending",
                    "",
                ]
            )
    review_path.write_text("\n".join(review_lines), encoding="utf-8")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
