#!/usr/bin/env python3
"""Rewrite arXiv structural-QA rows into LongDocURL-like questions.

The LLM is used only to rewrite the user-facing question. The answer, document
grounding, images, and metadata remain unchanged and are written to review
artifacts for manual inspection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import httpx


DEFAULT_INPUT_PARQUET = Path(
    "notes/generated/arxiv_struct_longdocurl_like_1k_uniform_multi_final_v5_pagechecked_20260722/"
    "arxiv_struct_longdocurl_like_deterministic_1000-insight_qwen_agent.parquet"
)
DEFAULT_LONGDOCURL_PARQUET = Path(
    "notes/generated/testcase_0504_full_parquets/longdocurl_full-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_DIR = Path("notes/generated/arxiv_struct_rewrite_pilot_20_20260722")

TASK_SUFFIX = {
    "topic2title": "Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
    "summary2title": "Select titles from the doc that best answer the question, do not alter or analyze the titles themselves.",
    "summary2tab": "Select table names from the doc that best answer the question, do not alter or analyze the table names themselves.",
    "extract_fig2tab": "",
}

FORBIDDEN_STYLE_PATTERNS = [
    r"\bthese descriptions\b",
    r"\bthese pieces of content\b",
    r"\bpieces of evidence\b",
    r"\bthese topics\b",
    r"\bfollowing content\b",
    r"\bcorresponding to this description\b",
    r"\bcorresponding to these descriptions\b",
    r"\bsection title that corresponds\b",
    r"\bFind the section title\b",
    r"\bFind the section titles\b",
]

RAW_SOURCE_PATTERNS = [
    r"\bIn this section\b",
    r"\bwe introduce\b",
    r"\bwe propose\b",
    r"\bwe present\b",
    r"\bwe apply\b",
    r"\bwe evaluated\b",
    r"\bwe acknowledge\b",
    r"\bas described in section\b",
    r"\breaders may\b",
]


def conv(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [conv(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): conv(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [conv(x) for x in obj]
    return obj


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def normalize_type(question_type: str) -> str:
    if question_type in {
        "extract_figure2table",
        "extract_table_other_tables",
        "extract_table2figure",
        "extract_figure_other_figures",
    }:
        return "extract_fig2tab"
    return question_type


def get_question(row: pd.Series) -> str:
    extra = conv(row["extra_info"])
    if extra.get("question"):
        return str(extra["question"])
    prompt = conv(row["prompt"])
    for msg in reversed(prompt):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", "")).replace("<image>", "")
    return ""


def set_question_in_prompt(row: pd.Series, new_question: str) -> list[dict[str, Any]]:
    prompt = conv(row["prompt"])
    for msg in reversed(prompt):
        if isinstance(msg, dict) and msg.get("role") == "user":
            old = str(msg.get("content", ""))
            prefix = "<image>" * old.count("<image>")
            msg["content"] = prefix + new_question
            break
    return prompt


def reward_ground_truth(row: pd.Series) -> str:
    reward = conv(row["reward_model"])
    if isinstance(reward, dict):
        return str(reward.get("ground_truth", ""))
    return str(reward)


def answer_items(row: pd.Series) -> list[str]:
    extra = conv(row["extra_info"])
    items = extra.get("answer_items")
    if isinstance(items, list):
        return [str(x) for x in items]
    gt = reward_ground_truth(row)
    try:
        parsed = json.loads(gt)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [gt]


def extract_evidence_from_original(question: str) -> str:
    q = normalize_ws(question)
    q = re.sub(r"Select (?:table names|titles).*", "", q).strip()
    q = re.sub(r"^.*?:\s*", "", q).strip()
    q = q.strip(" .")
    return q


def strip_number_prefix(text: str) -> str:
    text = normalize_ws(text)
    text = re.sub(r"^\s*(?:section\s+)?(?:[A-Z]?\d+(?:\.\d+)*|[IVXLCDM]+)\s*[\).:-]?\s+", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:figure|fig\.?|table)\s+[A-Z]?\d+(?:\.\d+)*\s*[:.-]?\s*", "", text, flags=re.I)
    return normalize_ws(text)


def normalize_for_match(text: str) -> str:
    text = strip_number_prefix(text).lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return normalize_ws(text)


def answer_leak_flags(question: str, items: list[str], task_type: str) -> list[str]:
    if task_type == "extract_fig2tab":
        # The anchor visual title is intentionally in the question. Only flag
        # target answer title leakage, which should still be avoided.
        pass
    q_norm = normalize_for_match(question)
    flags = []
    for item in items:
        item_norm = normalize_for_match(item)
        if len(item_norm) >= 10 and item_norm in q_norm:
            flags.append(f"answer_title_leak:{item[:80]}")
    return flags


def answer_quality_flags(items: list[str]) -> list[str]:
    flags: list[str] = []
    for item in items:
        text = normalize_ws(item)
        low = text.lower()
        if len(text) < 4:
            flags.append(f"answer_too_short:{text[:80]}")
        if len(text) > 260:
            flags.append(f"answer_too_long:{text[:80]}")
        if re.search(r"\b[A-Za-z]{2,}-\s+[A-Za-z]{2,}\b", text):
            flags.append(f"answer_ocr_hyphen_break:{text[:80]}")
        if re.search(r"\b(?:performace|molecualr|availabl|conditonal|concenration|rel\s+a\s+ted|contra-\s+ceptives)\b", low):
            flags.append(f"answer_common_ocr_typo:{text[:80]}")
        if text.count("$") >= 6 or text.count("\\") >= 6:
            flags.append(f"answer_math_heavy:{text[:80]}")
        alpha = sum(ch.isalpha() for ch in text)
        if alpha < max(4, len(text) * 0.35):
            flags.append(f"answer_low_alpha_ratio:{text[:80]}")
    return flags


def style_flags(question: str, items: list[str], task_type: str) -> list[str]:
    flags: list[str] = []
    q = normalize_ws(question)
    lead = re.split(r"\bSelect\b", q, maxsplit=1)[0]
    for pat in FORBIDDEN_STYLE_PATTERNS:
        if re.search(pat, q, flags=re.I):
            flags.append(f"forbidden_style:{pat}")
    for pat in RAW_SOURCE_PATTERNS:
        if re.search(pat, lead, flags=re.I):
            flags.append(f"raw_source_phrase:{pat}")
    if task_type in {"topic2title", "summary2tab"} and ";" in lead:
        flags.append("semicolon_enumeration")
    if task_type != "summary2title" and re.search(r"</?description>", q, flags=re.I):
        flags.append("description_xml_outside_summary2title")
    if re.search(r'"[^"]{50,}"', lead):
        flags.append("long_raw_quote")
    if re.search(r"\b[A-Za-z]{2,}-\s+[A-Za-z]{2,}\b", q):
        flags.append("ocr_hyphen_break")
    if len(re.findall(r"\w+", lead)) > 45 and task_type != "summary2title":
        flags.append("lead_too_long")
    flags.extend(answer_leak_flags(q, items, task_type))
    return flags


def risk_score(question: str, items: list[str], task_type: str) -> int:
    flags = style_flags(question, items, task_type)
    score = 5 * len(flags)
    score += len(re.findall(r"\w+", question)) // 20
    return score


def longdoc_type(row: pd.Series) -> str | None:
    qid = str(conv(row["extra_info"]).get("question_id", ""))
    for task_type in ["topic2title", "summary2title", "summary2tab", "extract_fig2tab"]:
        if f"_{task_type}_" in qid or qid.startswith(f"longdocurl_{task_type}_"):
            return task_type
    return None


def load_longdoc_examples(path: Path, per_type: int) -> dict[str, list[dict[str, str]]]:
    df = pd.read_parquet(path)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for _, row in df.iterrows():
        task_type = longdoc_type(row)
        if not task_type or len(examples[task_type]) >= per_type:
            continue
        examples[task_type].append(
            {
                "question": get_question(row),
                "answer": reward_ground_truth(row),
            }
        )
    return examples


def choose_pilot_rows(df: pd.DataFrame, count: int, include_indices: list[int], require_clean_answers: bool) -> list[int]:
    quotas = {
        "topic2title": count // 4,
        "summary2tab": count // 4,
        "summary2title": count // 4,
        "extract_fig2tab": count - 3 * (count // 4),
    }
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for idx, row in df.iterrows():
        extra = conv(row["extra_info"])
        task_type = normalize_type(str(extra.get("question_type") or extra.get("subset")))
        if task_type not in quotas:
            continue
        q = get_question(row)
        if require_clean_answers and answer_quality_flags(answer_items(row)):
            continue
        grouped[task_type].append((risk_score(q, answer_items(row), task_type), int(idx)))

    selected: list[int] = []
    for idx in include_indices:
        if 0 <= idx < len(df):
            selected.append(idx)

    selected_set = set(selected)
    for task_type, quota in quotas.items():
        already = sum(
            normalize_type(str(conv(df.iloc[idx]["extra_info"]).get("question_type"))) == task_type
            for idx in selected
        )
        need = max(0, quota - already)
        for _, idx in sorted(grouped[task_type], reverse=True):
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            need -= 1
            if need <= 0:
                break
    return selected[:count]


def build_prompt(record: dict[str, Any], examples: list[dict[str, str]]) -> list[dict[str, str]]:
    task_type = record["task_type"]
    task_notes = {
        "topic2title": (
            "Ask for section titles using a natural intent-style question. "
            "Do not paste raw source snippets; synthesize the common topic."
        ),
        "summary2title": (
            "Ask for the section title from a concise natural description. "
            "A description-style question is acceptable, but clean OCR and remove boilerplate."
        ),
        "summary2tab": (
            "Ask for table names using a natural intent-style question. "
            "Do not list evidence snippets; synthesize what the tables report or compare."
        ),
        "extract_fig2tab": (
            "This is a same-page visual-title lookup. Keep the anchor figure/table title if needed, "
            "but make the wording clean and unambiguous."
        ),
    }[task_type]
    suffix = TASK_SUFFIX[task_type]
    style_examples = "\n".join(
        f"- Q: {ex['question']}\n  A: {ex['answer']}" for ex in examples[:3]
    )
    answer_json = json.dumps(record["answer_items"], ensure_ascii=False)
    content = f"""Rewrite one synthetic document QA question so it looks like a natural LongDocURL benchmark question.

The correct answer is fixed and must remain exactly the answer listed below. You are not choosing the answer; you are only rewriting the question.

Task type: {task_type}
Task guidance: {task_notes}
Document id: {record['document_id']}
Fixed correct answer titles/names: {answer_json}
Anchor, if any: {record.get('anchor') or ''}
Original synthetic question: {record['old_question']}
Evidence/context extracted from the old question: {record['evidence']}

LongDocURL style examples for this task:
{style_examples}

Requirements:
- Produce a question that is answerable uniquely by the fixed answer(s), not a generic or ambiguous question.
- Keep similar difficulty to LongDocURL; slightly harder is acceptable.
- Do not copy the fixed answer title(s)/name(s) into the question.
- Do not use raw-snippet wording like "these descriptions", "pieces of evidence", "these topics", or semicolon-separated source snippets.
- Do not expose OCR artifacts such as broken hyphenated words.
- Do not mention that this is a rewrite or that an answer was provided.
- Output JSON only with keys "question" and "rationale".
"""
    if suffix:
        content += (
            "\nThe final question should naturally include this instruction sentence, "
            f"unless doing so would make it awkward: {suffix}\n"
        )
    return [
        {
            "role": "system",
            "content": "You write concise, natural benchmark questions for document visual QA. Output valid JSON only.",
        },
        {"role": "user", "content": content},
    ]


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict) or "question" not in obj:
        raise ValueError(f"unexpected response JSON: {obj}")
    return obj


def call_openai_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int,
    proxy: str | None,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(proxy=proxy, timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"]


def rewrite_one(
    record: dict[str, Any],
    examples: dict[str, list[dict[str, str]]],
    args: argparse.Namespace,
    proxy: str | None,
) -> dict[str, Any]:
    messages = build_prompt(record, examples.get(record["task_type"], []))
    last_error = ""
    for attempt in range(args.max_retries + 1):
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if not api_key or not base_url:
                raise RuntimeError("OPENAI_API_KEY and OPENAI_BASE_URL must be set")
            raw = call_openai_chat(
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                timeout=args.timeout,
                proxy=proxy,
            )
            obj = parse_json_response(raw)
            question = normalize_ws(obj["question"])
            flags = style_flags(question, record["answer_items"], record["task_type"])
            result = dict(record)
            result.update(
                {
                    "new_question": question,
                    "rewrite_rationale": normalize_ws(obj.get("rationale", "")),
                    "validation_flags": flags,
                    "rewrite_attempts": attempt + 1,
                    "raw_response": raw,
                }
            )
            if not flags or attempt == args.max_retries:
                return result
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Revise the question to fix these validation issues: "
                    + json.dumps(flags, ensure_ascii=False)
                    + ". Output JSON only.",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch moving and record errors.
            last_error = repr(exc)
            if attempt < args.max_retries:
                time.sleep(2 + attempt)
                continue
    result = dict(record)
    result.update(
        {
            "new_question": "",
            "rewrite_rationale": "",
            "validation_flags": ["rewrite_failed"],
            "rewrite_attempts": args.max_retries + 1,
            "error": last_error,
        }
    )
    return result


def row_record(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = df.iloc[idx]
    extra = conv(row["extra_info"])
    task_type = normalize_type(str(extra.get("question_type") or extra.get("subset")))
    old_question = get_question(row)
    items = answer_items(row)
    return {
        "index": int(idx),
        "question_id": extra.get("question_id"),
        "task_type": task_type,
        "raw_question_type": extra.get("question_type") or extra.get("subset"),
        "document_id": extra.get("document_id"),
        "page_ids": extra.get("question_involved_visuals") or extra.get("page_ids"),
        "source_block_ids": extra.get("source_block_ids"),
        "anchor": extra.get("anchor"),
        "answer_items": items,
        "answer_quality_flags": answer_quality_flags(items),
        "ground_truth": reward_ground_truth(row),
        "old_question": old_question,
        "old_validation_flags": style_flags(old_question, items, task_type),
        "evidence": extract_evidence_from_original(old_question),
    }


def write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# arXiv Structural QA Rewrite Pilot",
        "",
        f"Generated at: {dt.datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
        "This pilot rewrites only the user-facing question. Answers and grounding are unchanged.",
        "",
        "## Summary",
        "",
    ]
    by_type = Counter(r["task_type"] for r in results)
    flagged = sum(bool(r.get("validation_flags")) for r in results)
    failed = sum("rewrite_failed" in r.get("validation_flags", []) for r in results)
    lines.append(f"- Rows: {len(results)}")
    lines.append(f"- By task type: {dict(by_type)}")
    lines.append(f"- Rows with validation flags after rewrite: {flagged}")
    lines.append(f"- Rewrite failures: {failed}")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    for r in sorted(results, key=lambda x: x["index"]):
        lines.extend(
            [
                f"### Row {r['index']} - `{r['task_type']}`",
                "",
                f"- Question id: `{r.get('question_id')}`",
                f"- Document: `{r.get('document_id')}`",
                f"- Pages: `{r.get('page_ids')}`",
                f"- Answer: `{json.dumps(r.get('answer_items'), ensure_ascii=False)}`",
                f"- Answer-quality flags: `{json.dumps(r.get('answer_quality_flags'), ensure_ascii=False)}`",
                f"- Old flags: `{json.dumps(r.get('old_validation_flags'), ensure_ascii=False)}`",
                f"- New flags: `{json.dumps(r.get('validation_flags'), ensure_ascii=False)}`",
                "",
                "**Old question**",
                "",
                r.get("old_question", ""),
                "",
                "**New question**",
                "",
                r.get("new_question", ""),
                "",
                "**Rewrite rationale**",
                "",
                r.get("rewrite_rationale", ""),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rewritten_parquet(input_df: pd.DataFrame, results: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for result in results:
        if not result.get("new_question"):
            continue
        row = input_df.iloc[result["index"]].copy()
        extra = conv(row["extra_info"])
        extra["original_question_before_llm_rewrite"] = extra.get("question")
        extra["question"] = result["new_question"]
        extra["llm_rewrite_model"] = result.get("model")
        extra["llm_rewrite_validation_flags"] = result.get("validation_flags", [])
        row["extra_info"] = extra
        row["prompt"] = set_question_in_prompt(row, result["new_question"])
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_parquet(path, index=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda x: int(x["index"])):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, default=DEFAULT_INPUT_PARQUET)
    parser.add_argument("--longdocurl-parquet", type=Path, default=DEFAULT_LONGDOCURL_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--include-index", type=int, action="append", default=[993])
    parser.add_argument("--all-rows", action="store_true", help="Rewrite every input row instead of selecting a pilot subset.")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--longdoc-examples-per-type", type=int, default=4)
    parser.add_argument("--allow-unclean-answers", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed rows from rewritten_sample.jsonl in output-dir.")
    args = parser.parse_args()

    input_df = pd.read_parquet(args.input_parquet)
    examples = load_longdoc_examples(args.longdocurl_parquet, args.longdoc_examples_per_type)
    if args.all_rows:
        selected = list(range(len(input_df)))
    else:
        selected = choose_pilot_rows(
            input_df,
            args.count,
            args.include_index,
            require_clean_answers=not args.allow_unclean_answers,
        )
    records = [row_record(input_df, idx) for idx in selected]
    for record in records:
        record["model"] = args.model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_rows.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    proxy = None
    http_proxy = os.environ.get("API_HTTP_PROXY")
    https_proxy = os.environ.get("API_HTTPS_PROXY")
    if https_proxy:
        proxy = https_proxy
    elif http_proxy:
        proxy = http_proxy

    jsonl_path = args.output_dir / "rewritten_sample.jsonl"
    existing_results = read_jsonl(jsonl_path) if args.resume else []
    result_by_idx: dict[int, dict[str, Any]] = {
        int(result["index"]): result
        for result in existing_results
        if result.get("new_question") and "rewrite_failed" not in result.get("validation_flags", [])
    }
    pending_records = [record for record in records if int(record["index"]) not in result_by_idx]
    if result_by_idx:
        print(
            f"Resuming with {len(result_by_idx)}/{len(records)} completed rows; "
            f"{len(pending_records)} pending.",
            flush=True,
        )

    record_by_idx = {record["index"]: record for record in records}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_idx = {
            pool.submit(rewrite_one, record, examples, args, proxy): record["index"]
            for record in pending_records
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = dict(record_by_idx.get(idx, {"index": idx}))
                result.update(
                    {
                    "index": idx,
                    "validation_flags": ["rewrite_failed"],
                    "error": repr(exc),
                    }
                )
            result_by_idx[int(idx)] = result
            write_jsonl_atomic(list(result_by_idx.values()), jsonl_path)
            print(
                f"[{len(result_by_idx):04d}/{len(records):04d}] row={idx} "
                f"flags={result.get('validation_flags')} question={result.get('new_question', '')[:120]}",
                flush=True,
            )

    results = [result_by_idx[idx] for idx in selected if idx in result_by_idx]
    write_jsonl_atomic(results, jsonl_path)
    write_markdown(results, args.output_dir / "review.md")
    write_rewritten_parquet(input_df, results, args.output_dir / "rewritten_sample.parquet")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {args.output_dir / 'review.md'}")


if __name__ == "__main__":
    main()
