#!/usr/bin/env python3
"""Filter low-quality rows from the MC false-E calibration parquet.

This is intentionally rule-based and does not call any model. It keeps the raw
generated parquet intact and writes a filtered parquet plus a JSON report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PARQUET = (
    REPO_ROOT
    / "notes/generated/rl_16k_prompt24k_r05_to_r035_simple_datasource_mc_false_e_20260705/"
    "insight_doc_rl_16k_prompt24k_r05_to_r035_mc_false_e-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_PARQUET = (
    DEFAULT_INPUT_PARQUET.parent
    / "insight_doc_rl_16k_prompt24k_r05_to_r035_mc_false_e_filtered-insight_qwen_agent.parquet"
)
DEFAULT_REPORT_JSON = DEFAULT_INPUT_PARQUET.parent / "filter_report.json"

UNANSWERABLE_MARKERS = (
    "cannot answer",
    "can not answer",
    "not answerable",
    "unanswerable",
    "not provided",
    "not stated",
    "not shown",
    "not available",
    "not enough information",
    "insufficient information",
    "unknown",
)
STYLE_PREFIXES = (
    "station:",
    "stations:",
    "adjacent stations:",
    "answer:",
    "final answer:",
)
YESNO_STARTERS = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "has ",
    "have ",
    "had ",
    "should ",
    "would ",
    "will ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", default=str(DEFAULT_INPUT_PARQUET))
    parser.add_argument("--output-parquet", default=str(DEFAULT_OUTPUT_PARQUET))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-option-chars", type=int, default=240)
    return parser.parse_args()


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def lower(value: Any) -> str:
    return norm(value).lower()


def alnum(value: Any) -> str:
    return re.sub(r"[^\w]+", "", lower(value))


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"\w+", lower(value)))


def jaccard(a: Any, b: Any) -> float:
    aa = token_set(a)
    bb = token_set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def has_unanswerable_semantics(value: Any) -> bool:
    text = lower(value)
    return any(marker in text for marker in UNANSWERABLE_MARKERS)


def has_source_artifact(value: Any) -> bool:
    text = norm(value)
    ltext = text.lower()
    return (
        text.startswith("[")
        or text.endswith("]")
        or "' or '" in text
        or '" or "' in text
        or " or " in ltext
        or text.startswith("(green bounding box indicates")
        or "Â" in text
    )


def is_numeric(value: Any) -> bool:
    return bool(re.fullmatch(r"[$€£]?[-+]?\d[\d,]*(?:\.\d+)?%?", norm(value)))


def is_email(value: Any) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", norm(value)))


def is_date_like(value: Any) -> bool:
    text = lower(value)
    return bool(
        re.fullmatch(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", text)
        or re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", text)
    )


def is_route(value: Any) -> bool:
    text = norm(value)
    return "-" in text or "→" in text or "(transfer)" in lower(text)


def is_code(value: Any) -> bool:
    text = norm(value)
    return bool(re.fullmatch(r"[A-Za-z]?\d{2,}|[A-Z]{1,6}\d{0,6}|[A-Za-z]'{1,2}", text))


def answer_type(value: Any) -> str:
    if is_email(value):
        return "email"
    if is_numeric(value):
        return "number"
    if is_date_like(value):
        return "date"
    if is_route(value):
        return "route"
    if is_code(value):
        return "code"
    if len(norm(value).split()) <= 3:
        return "short_text"
    return "long_text"


def is_yesno(value: Any) -> bool:
    text = lower(value)
    return text in {"yes", "no"} or text.startswith("yes,") or text.startswith("no,")


def is_yesno_question(question: Any) -> bool:
    text = lower(question)
    return text.startswith(YESNO_STARTERS)


def shared_style_prefix(value: Any) -> str | None:
    text = lower(value)
    for prefix in STYLE_PREFIXES:
        if text.startswith(prefix):
            return prefix
    return None


def should_drop(row: pd.Series, max_option_chars: int) -> list[str]:
    reasons: list[str] = []
    extra = row.get("extra_info")
    reward_model = row.get("reward_model")
    if not isinstance(extra, dict) or not isinstance(reward_model, dict):
        return ["bad_metadata"]
    opts = extra.get("mc_options")
    gt = reward_model.get("ground_truth")
    if not isinstance(opts, dict) or set(opts) != set("ABCDE") or gt not in {"A", "B", "C", "D"}:
        return ["bad_options_or_label"]

    correct = opts[gt]
    distractors = [opts[letter] for letter in "ABCD" if letter != gt]
    question = extra.get("original_question") or extra.get("question") or ""

    values = [opts[letter] for letter in "ABCDE"]
    if any(not norm(value) for value in values):
        reasons.append("empty_option")
    if any(len(norm(value)) > max_option_chars for value in values):
        reasons.append("option_too_long")
    if len({lower(value) for value in values}) < 5 or len({alnum(value) for value in values}) < 5:
        reasons.append("duplicate_option")
    if any(has_unanswerable_semantics(value) for value in distractors):
        reasons.append("unanswerable_distractor")

    if has_source_artifact(correct):
        reasons.append("correct_answer_artifact")
    if any(has_source_artifact(value) for value in distractors):
        reasons.append("distractor_artifact")

    if is_yesno(correct) and not is_yesno_question(question):
        reasons.append("yesno_answer_for_non_yesno_question")

    correct_type = answer_type(correct)
    if correct_type in {"number", "email", "date", "code", "route"}:
        matching = sum(answer_type(value) == correct_type for value in distractors)
        if matching < 2:
            reasons.append(f"type_mismatch_{correct_type}")

    style_prefix = shared_style_prefix(correct)
    if style_prefix is not None:
        matching_prefix = sum(lower(value).startswith(style_prefix) for value in distractors)
        if matching_prefix < 2:
            reasons.append("style_prefix_leak")

    # Substring overlaps are often useful hard negatives for route/path answers.
    if correct_type != "route":
        correct_lower = lower(correct)
        for value in distractors:
            value_lower = lower(value)
            if (
                correct_lower != value_lower
                and min(len(correct_lower), len(value_lower)) >= 4
                and (correct_lower in value_lower or value_lower in correct_lower)
            ):
                reasons.append("substring_overlap_non_route")
                break
            if jaccard(correct, value) >= 0.9:
                reasons.append("high_overlap_non_route")
                break

    # Very short code-like answers are easy to make ambiguous unless the question
    # explicitly asks for a code/number/letter/ID.
    if len(alnum(correct)) <= 2 and correct_type in {"number", "code"}:
        q = lower(question)
        if not any(marker in q for marker in ("number", "code", "id", "letter", "symbol", "label")):
            reasons.append("short_code_answer_ambiguous")

    return sorted(set(reasons))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_parquet).expanduser().resolve()
    output_path = Path(args.output_parquet).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")

    df = pd.read_parquet(input_path)
    keep_indices: list[int] = []
    drops: list[dict[str, Any]] = []
    reason_counts: Counter = Counter()
    source_counts: Counter = Counter()
    for idx, row in df.iterrows():
        reasons = should_drop(row, args.max_option_chars)
        if reasons:
            extra = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else {}
            drops.append(
                {
                    "row_index": int(idx),
                    "data_source": row.get("data_source"),
                    "reasons": reasons,
                    "question": extra.get("original_question"),
                    "correct_answer": extra.get("mc_correct_answer"),
                    "options": extra.get("mc_options"),
                }
            )
            reason_counts.update(reasons)
            source_counts.update([row.get("data_source")])
        else:
            keep_indices.append(idx)

    filtered = df.loc[keep_indices].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path, index=False)
    report = {
        "input_parquet": str(input_path),
        "output_parquet": str(output_path),
        "input_rows": int(len(df)),
        "output_rows": int(len(filtered)),
        "dropped_rows": int(len(drops)),
        "reason_counts": dict(reason_counts),
        "drop_source_counts": dict(source_counts),
        "drops": drops,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "drops"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
