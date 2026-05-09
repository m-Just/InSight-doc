#!/usr/bin/env python3
"""Rewrite wrong unanswerable exported conversations into safe refusals.

This script targets page-dropped/not-answerable conversations where the agent
gave a concrete answer and the reward marked it wrong. It maps each dropped
question back to the original answerable QA, filters out cases where the final
answer appears to recover the original answer, then asks an API model to decide
whether rewriting the final assistant answer is coherent. Safe rewrites are
materialized as exported-conversation JSONs in a new output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rewrite_exported_convos_final_answers_with_api import (  # noqa: E402
    accuracy_reward,
    atomic_write_json,
    find_final_answer_message,
    find_initial_question,
    load_api_helpers,
    source_fingerprint,
)


PROMPT_VERSION = "unanswerable_wrong_rewrite_v3_question_specific"
METADATA_KEY = "unanswerable_final_answer_rewrite"
MARKER_NAME = "scripts/rewrite_unanswerable_wrong_exported_convos_with_api.py"
UNANSWERABLE_ANSWER = "the information provided in the document cannot answer this question"
CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "between",
    "both",
    "by",
    "compared",
    "comparing",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "specific",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing exported conversation JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write rewritten exported JSON files.")
    parser.add_argument(
        "--postprocess-dir",
        required=True,
        help="Postprocess directory containing valid_veqa_qas.json and valid_multivisual_group_qas.json.",
    )
    parser.add_argument("--model", default="gpt-5-nano", help="API model used for safe rewrite decisions.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--cache-jsonl", default=None, help="Defaults to OUTPUT_DIR/rewrite_cache.jsonl.")
    parser.add_argument("--status-jsonl", default=None, help="Defaults to OUTPUT_DIR/rewrite_status.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N selected candidates.")
    parser.add_argument(
        "--sample-per-tool-count",
        type=int,
        default=None,
        help="Select up to N candidates per tool-call count after filtering; useful for smoke tests.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--min-tool-calls", type=int, default=0)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument(
        "--include-strong-original-matches",
        action="store_true",
        help="Do not filter final answers that strongly match the original reference answer.",
    )
    parser.add_argument(
        "--original-match-filter-mode",
        choices=("precision", "balanced"),
        default="precision",
        help=(
            "How aggressively to filter candidates whose final answer may match the original reference. "
            "'precision' is conservative and prefers missing rewrite candidates over rewriting plausible/correct answers; "
            "'balanced' uses the older, looser strong-match thresholds."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run deterministic selection/filtering only. Does not call API or write rewritten JSONs.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use existing cache entries only; report cache_miss instead of calling the API.",
    )
    parser.add_argument(
        "--write-unsafe-copies",
        action="store_true",
        help="Copy non-rewritten eligible input records to output with metadata. Mainly for debugging.",
    )
    parser.add_argument(
        "--copy-unmodified",
        action="store_true",
        help=(
            "After rewriting, copy every input JSON that does not already have a rewritten output. "
            "This produces a full exported-conversation directory for the SFT converter."
        ),
    )
    parser.add_argument(
        "--max-api-failure-ratio",
        type=float,
        default=0.02,
        help="Exit nonzero if API/validation failures exceed this fraction of API-eligible rows.",
    )
    parser.add_argument("--max-api-failures", type=int, default=None)
    parser.add_argument(
        "--insight-doc-root",
        default=os.environ.get("INSIGHT_DOC_ROOT", str(REPO_ROOT.parent / "InSight-doc")),
    )
    parser.add_argument(
        "--api-logger-save-dir",
        default=os.environ.get("API_LOGGER_SAVE_DIR", str(Path.home() / ".dumps/api_requests")),
    )
    parser.add_argument("--api-logger-project-name", default="unanswerable_wrong_rewrite_gpt5_nano")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = item.get("key")
            result = item.get("result")
            if isinstance(key, str) and isinstance(result, dict):
                cache[key] = result
    return cache


def _normalize_reference_answer(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " | ".join(parts)
    if value is None:
        return ""
    return str(value)


def _iter_qa_rows_from_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("qas", [])
        return rows if isinstance(rows, list) else []
    if isinstance(data, list):
        return data
    return []


def load_original_qas(postprocess_dir: Path) -> dict[str, dict[str, str]]:
    original: dict[str, dict[str, str]] = {}
    for name in ("valid_veqa_qas.json", "valid_multivisual_group_qas.json"):
        path = postprocess_dir / name
        if not path.exists():
            continue
        for item in _iter_qa_rows_from_json(path):
            qa_id = item.get("qa_id")
            qa = item.get("qa") or {}
            if not isinstance(qa_id, str):
                continue
            original[qa_id] = {
                "question": str(qa.get("question") or ""),
                "reference_answer": _normalize_reference_answer(qa.get("reference_answer")),
                "source_file": name,
            }

    manifest_path = postprocess_dir / "manifest.jsonl"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qa_id = item.get("question_id")
                if not isinstance(qa_id, str):
                    continue
                original[qa_id] = {
                    "question": str(item.get("question") or ""),
                    "reference_answer": _normalize_reference_answer(item.get("answer")),
                    "source_file": "manifest.jsonl",
                }
    if not original:
        raise ValueError(f"No original QAs loaded from {postprocess_dir}")
    return original


def base_question_id(question_id: str) -> str:
    return re.sub(r"__drop_relevant_.*_unanswerable$", "", question_id)


def question_id(record: dict[str, Any]) -> str:
    extra_info = record.get("extra_info")
    if isinstance(extra_info, dict) and extra_info.get("question_id") is not None:
        return str(extra_info["question_id"])
    return ""


def is_unanswerable_record(record: dict[str, Any]) -> bool:
    qid = question_id(record).lower()
    reward = record.get("reward") if isinstance(record.get("reward"), dict) else {}
    ground_truth = str(reward.get("ground_truth") or "").lower()
    return "unanswerable" in qid or "cannot answer" in ground_truth or "not answer" in ground_truth


def assistant_tool_call_count(record: dict[str, Any]) -> int:
    conversation = record.get("conversation")
    if not isinstance(conversation, list):
        return 0
    return sum(
        1
        for message in conversation
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), dict)
            and isinstance(message["content"].get("tool_call"), dict)
        )
    )


def tool_call_summaries(record: dict[str, Any], *, max_items: int = 12) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    conversation = record.get("conversation")
    if not isinstance(conversation, list):
        return summaries
    for message in conversation:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        call = content.get("tool_call")
        call_dict = call if isinstance(call, dict) else {}
        summaries.append(
            {
                "think": str(content.get("think") or "")[:500],
                "img_idx": str(call_dict.get("img_idx", "")),
                "region_description": str(call_dict.get("region_description") or call_dict.get("label") or "")[:500],
            }
        )
        if len(summaries) >= max_items:
            break
    return summaries


def extracted_or_final_answer(record: dict[str, Any]) -> str:
    reward = record.get("reward")
    if isinstance(reward, dict) and isinstance(reward.get("extracted_answer"), str):
        return reward["extracted_answer"].strip()
    final = find_final_answer_message(record)
    if final is None:
        return ""
    _, _, think, answer = final
    return (answer or think).strip()


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[α-ωΑ-ΩµμνΩ]+", text.lower())


def token_f1(a: str, b: str) -> float:
    ca = Counter(normalized_tokens(a))
    cb = Counter(normalized_tokens(b))
    if not ca or not cb:
        return 0.0
    common = sum((ca & cb).values())
    precision = common / max(1, sum(ca.values()))
    recall = common / max(1, sum(cb.values()))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def sequence_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, " ".join(normalized_tokens(a)), " ".join(normalized_tokens(b))).ratio()


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?", text.lower()))


def content_token_set(text: str) -> set[str]:
    return {token for token in normalized_tokens(text) if len(token) > 1 and token not in CONTENT_STOPWORDS}


def strong_original_answer_match(reference_answer: str, final_answer: str) -> tuple[bool, dict[str, Any]]:
    ref_tokens = normalized_tokens(reference_answer)
    ans_tokens = normalized_tokens(final_answer)
    ref_norm = " ".join(ref_tokens)
    ans_norm = " ".join(ans_tokens)
    f1 = token_f1(reference_answer, final_answer)
    ratio = sequence_ratio(reference_answer, final_answer)
    ref_nums = numeric_tokens(reference_answer)
    ans_nums = numeric_tokens(final_answer)
    ref_content_tokens = content_token_set(reference_answer)
    ans_content_tokens = content_token_set(final_answer)
    numeric_overlap = (len(ref_nums & ans_nums) / len(ref_nums)) if ref_nums else 0.0
    reference_content_coverage = (
        len(ref_content_tokens & ans_content_tokens) / len(ref_content_tokens) if ref_content_tokens else 0.0
    )
    exact_or_substring = bool(
        ref_norm
        and (
            ref_norm == ans_norm
            or (len(ref_norm) >= 8 and ref_norm in ans_norm)
            or (len(ans_norm) >= 8 and ans_norm in ref_norm and len(ref_tokens) <= 8)
        )
    )
    short_answer_containment = bool(len(ref_tokens) <= 5 and set(ref_tokens).issubset(set(ans_tokens)))
    strong = (
        exact_or_substring
        or short_answer_containment
        or f1 >= 0.60
        or ratio >= 0.70
        or (bool(ref_nums) and numeric_overlap >= 0.70)
    )
    return strong, {
        "token_f1": round(f1, 6),
        "sequence_ratio": round(ratio, 6),
        "numeric_overlap": round(numeric_overlap, 6),
        "reference_content_coverage": round(reference_content_coverage, 6),
        "exact_or_substring": exact_or_substring,
        "short_answer_containment": short_answer_containment,
        "reference_numbers": sorted(ref_nums),
        "answer_numbers": sorted(ans_nums),
        "overlap_content_tokens": sorted(ref_content_tokens & ans_content_tokens)[:40],
    }


def original_answer_match_for_filter(
    reference_answer: str,
    final_answer: str,
    *,
    mode: str,
) -> tuple[bool, dict[str, Any]]:
    strong, metrics = strong_original_answer_match(reference_answer, final_answer)
    if mode == "balanced":
        metrics["match_filter_mode"] = mode
        metrics["precision_first_match"] = False
        return strong, metrics

    # Precision-first mode is intentionally conservative: if the model answer
    # looks like a plausible partial/qualitative match to the original answer,
    # do not rewrite it. Missing good rewrite candidates is preferable to
    # turning a correct/mostly-correct answer into a false missing-evidence one.
    precision_first_match = (
        strong
        or metrics["token_f1"] >= 0.40
        or metrics["sequence_ratio"] >= 0.50
        or metrics["reference_content_coverage"] >= 0.42
        or (
            metrics["reference_content_coverage"] >= 0.28
            and bool(metrics["reference_numbers"])
            and metrics["numeric_overlap"] >= 0.30
        )
        or (bool(metrics["reference_numbers"]) and metrics["numeric_overlap"] >= 0.50 and metrics["token_f1"] >= 0.25)
    )
    metrics["match_filter_mode"] = mode
    metrics["precision_first_match"] = precision_first_match and not strong
    return precision_first_match, metrics


def cache_key(payload: dict[str, Any], model: str) -> str:
    encoded = json.dumps(
        {"model": model, "prompt_version": PROMPT_VERSION, **payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are cleaning exported visual-QA conversations for supervised fine-tuning.\n"
        "\n"
        "The current example is a page-dropped/not-answerable version of an originally answerable question. "
        "The model's final answer was scored wrong. Your job is to decide whether it is coherent to rewrite "
        "only the final assistant answer into a missing-evidence answer.\n"
        "\n"
        "Rewrite-safe means:\n"
        "- the final answer is actually wrong compared with the original reference answer, and\n"
        "- the conversation/search history does not look like it already found reliable evidence for the final answer, and\n"
        "- replacing only the final answer with a concise missing-evidence response would not contradict the preceding turns.\n"
        "Prefer false negatives over false positives: if the final answer is a plausible partial answer, "
        "qualitative paraphrase, or less-detailed version of the original reference answer, mark it unsafe.\n"
        "\n"
        "Unsafe means:\n"
        "- the final answer substantially matches the original reference answer, or\n"
        "- the final answer partially or qualitatively matches the original reference answer, even if it omits exact values, or\n"
        "- the previous turns appear to have found strong direct evidence, or\n"
        "- the final answer is only partially wrong/ambiguous enough that rewriting would be misleading.\n"
        "\n"
        "If safe, write a natural final answer that says the needed evidence is not available in the provided "
        "images/pages. The rewrite must be specific to the question: name the missing attribute, comparison, "
        "entity, metric, or relationship being asked about. Do not return a generic standalone template such as "
        "'The needed evidence is not available.' Keep it to one concise sentence unless the question has multiple "
        "parts. Do not reveal or mention the original reference answer. Do not mention page dropping, hidden "
        "ground truth, labels, reward, or dataset construction. Do not add concrete answer values.\n"
        "\n"
        "Return strict JSON only, with this schema:\n"
        "{\"rewrite_safe\": boolean, \"reason\": string, \"rewrite\": string}\n"
        "If rewrite_safe is false, set rewrite to an empty string.\n"
        "\n"
        f"Question:\n{payload['question']}\n\n"
        f"Original reference answer for safety checking only; do not reveal it:\n{payload['reference_answer']}\n\n"
        f"Model final answer scored wrong:\n{payload['final_answer']}\n\n"
        f"Assistant tool-call search summaries:\n{json.dumps(payload['tool_calls'], ensure_ascii=False, indent=2)}\n"
    )


def parse_json_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("API response JSON is not an object")
    return data


def validate_api_result(result: dict[str, Any], reference_answer: str, question: str) -> str | None:
    if not isinstance(result.get("rewrite_safe"), bool):
        return "rewrite_safe_not_boolean"
    if not isinstance(result.get("reason"), str):
        return "reason_not_string"
    if not isinstance(result.get("rewrite"), str):
        return "rewrite_not_string"
    rewrite = result["rewrite"].strip()
    if not result["rewrite_safe"]:
        return None
    if not rewrite:
        return "safe_but_empty_rewrite"
    lowered = rewrite.lower()
    if any(tag in lowered for tag in ("<think", "</think", "<answer", "</answer", "<tool_call", "</tool_call")):
        return "rewrite_contains_disallowed_tag"
    if "page drop" in lowered or "ground truth" in lowered or "dataset" in lowered or "reward" in lowered:
        return "rewrite_mentions_dataset_internals"
    if not any(
        phrase in lowered
        for phrase in (
            "cannot determine",
            "can't determine",
            "not enough evidence",
            "not contain enough evidence",
            "does not contain enough evidence",
            "not enough information",
            "do not provide enough information",
            "does not provide enough information",
            "don't provide enough information",
            "doesn't provide enough information",
            "necessary evidence",
            "insufficient",
            "not visible",
            "not visible here",
            "not shown",
            "do not show",
            "does not show",
            "not available",
            "not provided",
            "aren't visible",
            "isn't visible",
            "not legible",
            "not enough data",
            "unable to determine",
            "do not include",
            "does not include",
            "do not appear",
            "does not appear",
        )
    ):
        return "rewrite_lacks_missing_evidence_language"
    question_tokens = content_token_set(question)
    rewrite_tokens = content_token_set(rewrite)
    if len(question_tokens) >= 5 and len(question_tokens & rewrite_tokens) < 2:
        return "rewrite_too_generic"
    strong_match, _ = strong_original_answer_match(reference_answer, rewrite)
    if strong_match:
        return "rewrite_leaks_original_reference_answer"
    if len(rewrite.split()) > 160:
        return "rewrite_too_long"
    return None


def completed_output_matches(output_path: Path, fingerprint: dict[str, Any], cache_key_value: str) -> bool:
    if not output_path.exists():
        return False
    try:
        record = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    metadata = record.get(METADATA_KEY)
    if not isinstance(metadata, dict) or not metadata.get("rewrite_safe"):
        return False
    return (
        metadata.get("source_size") == fingerprint.get("source_size")
        and metadata.get("source_mtime_ns") == fingerprint.get("source_mtime_ns")
        and metadata.get("source_sha256") == fingerprint.get("source_sha256")
        and metadata.get("rewrite_cache_key") == cache_key_value
    )


async def query_api_json(
    *,
    prompt: str,
    model: str,
    timeout: float,
    max_retries: int,
    max_completion_tokens: int,
    insight_doc_root: Path,
) -> dict[str, Any]:
    create_async_openai_client, query_model_with_retry = load_api_helpers(insight_doc_root, ensure_api_logger=True)
    client = create_async_openai_client(timeout=timeout)
    try:
        call = await asyncio.wait_for(
            query_model_with_retry(
                query=prompt,
                model=model,
                client=client,
                context=[
                    {
                        "role": "system",
                        "content": "You classify and rewrite visual-QA final answers. Return strict JSON only.",
                    }
                ],
                max_attempts=max_retries + 1,
                retry_initial_delay_sec=1.0,
                max_completion_tokens=max_completion_tokens,
            ),
            timeout=max(timeout * (max_retries + 1) + 10.0, timeout + 10.0),
        )
    finally:
        try:
            await asyncio.wait_for(client.close(), timeout=5.0)
        except Exception:
            pass
    if not call.success or call.response is None:
        raise RuntimeError(call.error or "API call failed without an error message")
    content = call.response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    if not isinstance(content, str):
        raise RuntimeError("API response did not contain string content")
    return parse_json_response(content)


def write_record_with_metadata(
    *,
    record: dict[str, Any],
    output_path: Path,
    source_path: Path,
    fingerprint: dict[str, Any],
    message_index: int | None,
    model: str,
    cache_key_value: str,
    result: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    record = json.loads(json.dumps(record, ensure_ascii=False))
    if result.get("rewrite_safe") and message_index is not None:
        content = dict(record["conversation"][message_index]["content"])
        content["think"] = ""
        content["answer"] = result["rewrite"].strip()
        record["conversation"][message_index]["content"] = content
        reward = record.get("reward")
        if isinstance(reward, dict):
            reward["ground_truth"] = UNANSWERABLE_ANSWER
            reward["extracted_answer"] = result["rewrite"].strip()
            reward["reward"] = 1.0
            score = reward.get("score")
            if isinstance(score, dict):
                score["accuracy_reward"] = 1.0
                score["score"] = 1.0
                score["extracted_answer"] = result["rewrite"].strip()
    record[METADATA_KEY] = {
        "schema_version": "unanswerable_final_answer_rewrite_v1",
        "marked_by": MARKER_NAME,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "source_path": str(source_path),
        **fingerprint,
        "message_index": message_index,
        "rewrite_cache_key": cache_key_value,
        "rewrite_safe": bool(result.get("rewrite_safe")),
        "reason": result.get("reason", ""),
        "selection": selection,
        "rewritten_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_path, record)


async def process_candidate(
    *,
    path: Path,
    input_dir: Path,
    output_dir: Path,
    original_qas: dict[str, dict[str, str]],
    args: argparse.Namespace,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    cache_lock: asyncio.Lock,
) -> dict[str, Any]:
    relative_path = path.relative_to(input_dir)
    output_path = output_dir / relative_path
    fingerprint = source_fingerprint(path)
    status_base = {
        "path": str(path),
        "relative_path": str(relative_path),
        "output_path": str(output_path),
        **fingerprint,
    }
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {**status_base, "status": "error", "reason": f"json_load_failed:{type(exc).__name__}:{exc}"}

    qid = question_id(record)
    base_qid = base_question_id(qid)
    original = original_qas.get(base_qid)
    if original is None:
        return {**status_base, "status": "filtered_missing_original_reference", "question_id": qid}

    final = find_final_answer_message(record)
    if final is None:
        return {**status_base, "status": "filtered_no_final_answer", "question_id": qid}
    message_index, _, _, _ = final
    final_answer = extracted_or_final_answer(record)
    tool_count = assistant_tool_call_count(record)
    strong_match, match_metrics = original_answer_match_for_filter(
        original["reference_answer"],
        final_answer,
        mode=args.original_match_filter_mode,
    )
    selection = {
        "question_id": qid,
        "base_question_id": base_qid,
        "tool_call_count": tool_count,
        "strong_original_answer_match": strong_match,
        "match_metrics": match_metrics,
    }
    if strong_match and not args.include_strong_original_matches:
        return {**status_base, "status": "filtered_strong_original_answer_match", **selection}

    payload = {
        "question": original["question"] or find_initial_question(record),
        "reference_answer": original["reference_answer"],
        "final_answer": final_answer,
        "tool_calls": tool_call_summaries(record),
    }
    key = cache_key(payload, args.model)
    selection["rewrite_cache_key"] = key

    if args.dry_run:
        return {**status_base, "status": "dry_run_candidate", **selection}

    if completed_output_matches(output_path, fingerprint, key):
        return {**status_base, "status": "skipped_existing_output", **selection}

    async with cache_lock:
        result = cache.get(key)
    used_cache = result is not None
    if result is None:
        if args.cache_only:
            return {**status_base, "status": "cache_miss", "reason": "cache_only_without_cached_result", **selection}
        try:
            result = await query_api_json(
                prompt=build_prompt(payload),
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
                insight_doc_root=Path(args.insight_doc_root).expanduser().resolve(),
            )
        except Exception as exc:
            return {**status_base, "status": "api_failure", "reason": f"{type(exc).__name__}: {exc}", **selection}
        invalid = validate_api_result(result, original["reference_answer"], payload["question"])
        if invalid is not None:
            return {**status_base, "status": "validation_failure", "reason": invalid, "api_result": result, **selection}
        async with cache_lock:
            cached_after_api = cache.get(key)
            if cached_after_api is not None:
                result = cached_after_api
                used_cache = True
            else:
                cache[key] = result
                append_jsonl(cache_path, {"key": key, "result": result})
    else:
        invalid = validate_api_result(result, original["reference_answer"], payload["question"])
        if invalid is not None:
            return {
                **status_base,
                "status": "validation_failure",
                "reason": f"cached:{invalid}",
                "api_result": result,
                **selection,
            }

    if result.get("rewrite_safe") or args.write_unsafe_copies:
        write_record_with_metadata(
            record=record,
            output_path=output_path,
            source_path=path,
            fingerprint=fingerprint,
            message_index=message_index,
            model=args.model,
            cache_key_value=key,
            result=result,
            selection=selection,
        )
    return {
        **status_base,
        "status": "rewritten_cached" if used_cache and result.get("rewrite_safe") else (
            "unsafe_cached" if used_cache else ("rewritten_api" if result.get("rewrite_safe") else "unsafe_api")
        ),
        "api_result": result,
        **selection,
    }


def select_candidates(paths: list[Path], original_qas: dict[str, dict[str, str]], args: argparse.Namespace) -> tuple[list[Path], Counter[str]]:
    selected: list[tuple[int, Path]] = []
    counts: Counter[str] = Counter()
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            counts["json_error"] += 1
            continue
        if not is_unanswerable_record(record):
            counts["filtered_not_unanswerable"] += 1
            continue
        if accuracy_reward(record) != 0.0:
            counts["filtered_not_wrong"] += 1
            continue
        qid = question_id(record)
        if base_question_id(qid) not in original_qas:
            counts["filtered_missing_original_reference"] += 1
            continue
        tool_count = assistant_tool_call_count(record)
        if tool_count < args.min_tool_calls:
            counts["filtered_below_min_tool_calls"] += 1
            continue
        if args.max_tool_calls is not None and tool_count > args.max_tool_calls:
            counts["filtered_above_max_tool_calls"] += 1
            continue
        selected.append((tool_count, path))
        counts["selected_before_sampling"] += 1
        counts[f"selected_tool_calls:{tool_count}"] += 1

    if args.sample_per_tool_count is not None:
        import random

        rng = random.Random(args.sample_seed)
        by_count: dict[int, list[Path]] = {}
        for tool_count, path in selected:
            by_count.setdefault(tool_count, []).append(path)
        sampled: list[Path] = []
        for tool_count in sorted(by_count):
            group = by_count[tool_count]
            if len(group) > args.sample_per_tool_count:
                group = rng.sample(group, args.sample_per_tool_count)
            sampled.extend(sorted(group))
        selected_paths = sampled
        counts["selected_after_sampling"] = len(selected_paths)
    else:
        selected_paths = [path for _, path in selected]

    if args.limit is not None:
        selected_paths = selected_paths[: args.limit]
        counts["selected_after_limit"] = len(selected_paths)
    return selected_paths, counts


async def run_async(args: argparse.Namespace) -> tuple[Counter[str], int, int]:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    postprocess_dir = Path(args.postprocess_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input dir does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("API_LOGGER_SAVE_DIR", str(Path(args.api_logger_save_dir).expanduser()))
    os.environ.setdefault("API_LOGGER_PROJECT_NAME", args.api_logger_project_name)
    os.environ["ENSURE_API_LOGGER"] = "1"

    original_qas = load_original_qas(postprocess_dir)
    paths = sorted(input_dir.glob("*.json"))
    selected_paths, selection_counts = select_candidates(paths, original_qas, args)
    cache_path = Path(args.cache_jsonl).expanduser().resolve() if args.cache_jsonl else output_dir / "rewrite_cache.jsonl"
    status_path = Path(args.status_jsonl).expanduser().resolve() if args.status_jsonl else output_dir / "rewrite_status.jsonl"
    cache = load_jsonl_cache(cache_path)
    print(f"input_files={len(paths)} original_qas={len(original_qas)} selected={len(selected_paths)}")
    print("selection_summary:")
    for key, value in selection_counts.most_common():
        print(f"  {value}\t{key}")
    print(f"cache_entries={len(cache)}")

    status_counts: Counter[str] = Counter()
    eligible_for_api = 0
    failures = 0
    completed = 0
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    cache_lock = asyncio.Lock()

    async def run_one(path: Path) -> dict[str, Any]:
        async with semaphore:
            try:
                return await process_candidate(
                    path=path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    original_qas=original_qas,
                    args=args,
                    cache=cache,
                    cache_path=cache_path,
                    cache_lock=cache_lock,
                )
            except Exception as exc:
                return {
                    "path": str(path),
                    "relative_path": str(path.relative_to(input_dir)),
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

    tasks = [asyncio.create_task(run_one(path)) for path in selected_paths]
    for task in asyncio.as_completed(tasks):
        status = await task
        append_jsonl(status_path, status)
        status_name = status["status"]
        status_counts[status_name] += 1
        if status_name in {
            "dry_run_candidate",
            "rewritten_api",
            "rewritten_cached",
            "unsafe_api",
            "unsafe_cached",
            "api_failure",
            "validation_failure",
            "cache_miss",
        }:
            eligible_for_api += 1
        if status_name in {"api_failure", "validation_failure", "cache_miss"}:
            failures += 1
        if status_name in {"api_failure", "validation_failure", "cache_miss", "error"}:
            print(f"{status_name.upper()} {status.get('path')}: {status.get('reason')}", file=sys.stderr)
        completed += 1
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == len(selected_paths)):
            print(f"progress {completed}/{len(selected_paths)} {dict(status_counts)}", flush=True)
    if args.copy_unmodified:
        copied = 0
        skipped = 0
        for path in paths:
            output_path = output_dir / path.relative_to(input_dir)
            if output_path.exists():
                skipped += 1
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output_path)
            copied += 1
        print(f"copy_unmodified copied={copied} skipped_existing={skipped}", flush=True)
    return status_counts, eligible_for_api, failures


def main() -> int:
    args = parse_args()
    status_counts, eligible, failures = asyncio.run(run_async(args))
    print("rewrite summary:")
    for key, value in status_counts.most_common():
        print(f"  {value}\t{key}")
    failure_ratio = failures / eligible if eligible else 0.0
    print(f"eligible_for_api={eligible}")
    print(f"api_or_validation_failures={failures}")
    print(f"failure_ratio={failure_ratio:.6f}")
    too_many = failure_ratio > args.max_api_failure_ratio
    if args.max_api_failures is not None and failures > args.max_api_failures:
        too_many = True
    return 1 if too_many else 0


if __name__ == "__main__":
    raise SystemExit(main())
