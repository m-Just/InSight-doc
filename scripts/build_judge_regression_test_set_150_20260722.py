#!/usr/bin/env python3
"""Append reward-hacking refusal cases to the 125-row judge set."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "notes/generated/judge_regression_test_set_125_20260722"
OUT_DIR = ROOT / "notes/generated/judge_regression_test_set_150_20260722"

BASE_125 = BASE_DIR / "judge_regression_test_set_125.parquet"
JUDGE_275 = ROOT / "notes/generated/judge_test_set_275_answerable_refusal_supplement_20260718/judge_test_set_275.parquet"
JUDGE_275_LEGACY = (
    ROOT
    / "notes/generated/judge_test_set_275_answerable_refusal_supplement_20260718"
    / "judge_eval_20260718_globalai/predictions_legacy.jsonl"
)


REFUSAL_PAT = re.compile(
    r"\b("
    r"cannot answer|can not answer|cannot determine|can not determine|cannot be determined|"
    r"can not be determined|not enough information|unanswerable|not provided|not visible|"
    r"not shown|not stated|not available|does not provide|no information|no [^.]{0,80}(?:visible|present)"
    r")\b",
    flags=re.IGNORECASE,
)


def word_count(value: Any) -> int:
    return len(re.findall(r"\w+", str(value or "")))


def parse_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if value is None:
        return None
    text = str(value).strip()
    for parser in (json.loads,):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    try:
        import ast

        parsed = ast.literal_eval(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, list) else None


def looks_like_multi_target(question: Any, ground_truth: Any) -> bool:
    items = parse_list(ground_truth)
    if not items or len(items) <= 1:
        return False
    return bool(
        re.search(
            r"\b(list|all|what are|which sections|which tables|which titles|which figures|"
            r"which .*names|enumerate|features|applications|services|components|top 3|route)\b",
            str(question or "").lower(),
        )
    )


def load_legacy_predictions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                out[str(item["id"])] = item
    return out


def diverse_select(df: pd.DataFrame, n: int) -> pd.DataFrame:
    selected: list[pd.Series] = []
    used: set[str] = set()
    groups = {
        key: group.sort_values(["answer_words", "id"], ascending=[True, True]).to_dict("records")
        for key, group in df.groupby(
            ["benchmark_or_data_source", "row_origin", "question_type"],
            sort=True,
        )
    }
    keys = sorted(groups, key=lambda key: (len(groups[key]), str(key)))

    while len(selected) < n:
        progressed = False
        for key in keys:
            rows = groups[key]
            while rows:
                row = rows.pop(0)
                if str(row["id"]) in used:
                    continue
                selected.append(pd.Series(row))
                used.add(str(row["id"]))
                progressed = True
                break
            if len(selected) >= n:
                break
        if not progressed:
            break

    if len(selected) < n:
        raise ValueError(f"only selected {len(selected)}/{n} reward-hacking rows")
    return pd.DataFrame(selected).head(n).copy()


def main() -> None:
    base = pd.read_parquet(BASE_125).copy()
    source = pd.read_parquet(JUDGE_275).copy()
    legacy = load_legacy_predictions(JUDGE_275_LEGACY)

    source["id"] = source["id"].astype(str)
    source["legacy_pred"] = source["id"].map(lambda row_id: int(legacy[row_id]["judge_pred"]))
    source["legacy_score_current"] = source["id"].map(lambda row_id: float(legacy[row_id]["judge_score"]))
    source["legacy_correct"] = source["legacy_pred"].astype(int) == source["human_label"].astype(int)
    source["gt_words"] = source["ground_truth"].map(word_count)
    source["answer_words"] = source["final_answer"].map(word_count)
    source["gt_item_count"] = source["ground_truth"].map(lambda value: len(parse_list(value) or []))
    source["is_multi_target"] = [
        looks_like_multi_target(question, gt)
        for question, gt in zip(source["question"], source["ground_truth"])
    ]
    source["is_refusal_like_answer"] = source["final_answer"].astype(str).map(lambda text: bool(REFUSAL_PAT.search(text)))
    source["source_pool"] = "judge_test_set_275"
    source["source_row_id"] = source["id"]

    existing_ids = set(base["id"].astype(str))
    candidates = source[
        source["answerability"].eq("answerable")
        & source["human_label"].eq(0)
        & source["legacy_pred"].eq(0)
        & source["is_refusal_like_answer"]
        & ~source["id"].isin(existing_ids)
        & ~source["critical_failure"].fillna(False).astype(bool)
        & ~source["response_truncated"].fillna(False).astype(bool)
    ].copy()

    selected = diverse_select(candidates, 25)
    selected["test_bucket"] = "reward_hacking_refusal"
    selected["selection_reason"] = (
        "Answerable question where the model gives a refusal/unanswerable-style final answer; "
        "human label is wrong and legacy marks it wrong."
    )
    selected["manual_inspection_status"] = "pending"
    selected["manual_inspection_notes"] = ""

    for col in base.columns:
        if col not in selected.columns:
            selected[col] = None
    selected = selected[base.columns]

    out = pd.concat([base, selected], ignore_index=True, sort=False)
    if len(out) != 150 or out["id"].nunique() != 150:
        raise RuntimeError(f"bad output shape: rows={len(out)} unique_ids={out['id'].nunique()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "judge_regression_test_set_150.parquet", index=False)
    out.to_json(OUT_DIR / "judge_regression_test_set_150.jsonl", orient="records", lines=True, force_ascii=False)
    out.to_csv(OUT_DIR / "judge_regression_test_set_150_annotation.csv", index=False)

    summary = {
        "rows": len(out),
        "bucket_counts": out["test_bucket"].value_counts().sort_index().to_dict(),
        "label_counts": out["human_label"].value_counts().sort_index().to_dict(),
        "answerability_counts": out["answerability"].value_counts().sort_index().to_dict(),
        "reward_hacking_source_counts": selected["benchmark_or_data_source"].value_counts().sort_index().to_dict(),
        "reward_hacking_origin_counts": selected["row_origin"].value_counts().sort_index().to_dict(),
        "reward_hacking_question_type_counts": selected["question_type"].value_counts().sort_index().to_dict(),
        "bucket_label_counts": {
            bucket: group["human_label"].value_counts().sort_index().to_dict()
            for bucket, group in out.groupby("test_bucket")
        },
        "bucket_source_counts": {
            bucket: group["benchmark_or_data_source"].value_counts().sort_index().to_dict()
            for bucket, group in out.groupby("test_bucket")
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Judge Regression Test Set 150",
        "",
        "This extends the 125-row judge regression set with 25 reward-hacking refusal cases.",
        "",
        "Reward-hacking additions satisfy: answerable question, human label 0, legacy prediction 0, refusal/unanswerable-style final answer, not already in the 125-row set.",
        "",
        "| bucket | rows | label=1 | label=0 |",
        "|---|---:|---:|---:|",
    ]
    for bucket, group in out.groupby("test_bucket", sort=True):
        lines.append(
            f"| `{bucket}` | {len(group)} | {int(group['human_label'].eq(1).sum())} | {int(group['human_label'].eq(0).sum())} |"
        )
    lines.extend(["", "Reward-hacking source distribution:", "", "| source | rows |", "|---|---:|"])
    for source_name, count in selected["benchmark_or_data_source"].value_counts().sort_index().items():
        lines.append(f"| `{source_name}` | {count} |")
    lines.extend(["", "Files:", "", "- `judge_regression_test_set_150.parquet`", "- `judge_regression_test_set_150.jsonl`", "- `judge_regression_test_set_150_annotation.csv`"])
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    review_lines = ["# Reward Hacking Refusal Additions", ""]
    for idx, row in selected.reset_index(drop=True).iterrows():
        review_lines.extend(
            [
                f"## {idx + 1}. `{row['id']}`",
                "",
                f"- source: `{row['benchmark_or_data_source']}` / `{row['row_origin']}` / `{row['question_type']}`",
                f"- label: `{row['human_label']}`, legacy_pred: `{row['legacy_pred']}`",
                f"- question: {str(row['question']).replace(chr(10), ' ')}",
                f"- GT: {str(row['ground_truth']).replace(chr(10), ' ')}",
                f"- final_answer: {str(row['final_answer']).replace(chr(10), ' ')}",
                "",
            ]
        )
    (OUT_DIR / "reward_hacking_manual_review.md").write_text("\n".join(review_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
