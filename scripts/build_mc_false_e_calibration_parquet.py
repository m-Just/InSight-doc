#!/usr/bin/env python3
"""Build multiple-choice false-E calibration rows from an existing RL parquet.

The output parquet keeps the original image inputs and agent/system prompt, but
rewrites answerable short-answer questions into A-E multiple-choice questions.
Option E is always an "insufficient information" option and is intentionally
wrong for every generated row. The correct answer is placed in A-D.

The script is resumable: successful API-generated distractors are cached in
JSONL, and the final parquet is written only after all selected rows have
usable options.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_PARQUET = (
    REPO_ROOT
    / "notes/generated/rl_16k_prompt24k_r05_to_r035_simple_datasource_20260617/"
    "insight_doc_rl_16k_prompt24k_r05_to_r035_simple_datasource-insight_qwen_agent.parquet"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "notes/generated/rl_16k_prompt24k_r05_to_r035_simple_datasource_mc_false_e_20260705"
)
DEFAULT_E_OPTION = "The provided images do not contain enough information to answer."
PROMPT_VERSION = "mc_false_e_distractors_v1"
BUILDER_NAME = "scripts/build_mc_false_e_calibration_parquet.py"
PART_MARKER_RE = re.compile(r"(?:(?<=^)|(?<=[\s;]))\([a-z]\)(?=\s)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", default=str(DEFAULT_INPUT_PARQUET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-parquet", default=None)
    parser.add_argument("--cache-jsonl", default=None)
    parser.add_argument("--status-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--target-rows", type=int, default=2600)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-answer-chars", type=int, default=120)
    parser.add_argument("--max-question-chars", type=int, default=1000)
    parser.add_argument("--e-option", default=DEFAULT_E_OPTION)
    parser.add_argument(
        "--distractor-mode",
        choices=("api", "sampled"),
        default="api",
        help="Use gpt-5-nano/API or deterministic sampled answers for distractors.",
    )
    parser.add_argument(
        "--include-data-source",
        action="append",
        default=[],
        help="Optional data_source allow-list. May be repeated.",
    )
    parser.add_argument(
        "--exclude-data-source",
        action="append",
        default=[],
        help="Optional data_source deny-list. May be repeated.",
    )
    parser.add_argument("--allow-multipart", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--insight-doc-root",
        default=os.environ.get("INSIGHT_DOC_ROOT", str(REPO_ROOT.parent / "InSight-doc")),
        help="Used only to import the InSight-doc API helper/logger when available.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("key")
            value = record.get("result")
            if isinstance(key, str) and isinstance(value, dict):
                out[key] = value
    return out


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return " ".join(str(value).strip().split())


def normalize_for_match(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def has_unanswerable_semantics(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "cannot answer",
        "can not answer",
        "not answerable",
        "unanswerable",
        "not provided",
        "not stated",
        "does not state",
        "do not state",
        "not shown",
        "not available",
        "not enough information",
    )
    return any(marker in lowered for marker in markers)


def strip_answer_prefix(text: str) -> str:
    stripped = text.strip()
    for prefix in ("answer:", "final answer:"):
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def parse_single_answer(value: Any) -> str | None:
    text = strip_answer_prefix(normalize_text(value))
    if not text:
        return None
    if PART_MARKER_RE.search(text):
        return None

    # Accept a singleton literal list/string, but reject multi-answer lists.
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, str):
            return normalize_text(parsed)
        if isinstance(parsed, (list, tuple)):
            if len(parsed) != 1:
                return None
            return normalize_text(parsed[0])
        return normalize_text(parsed)
    return text


def get_question(row: pd.Series) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        question = extra.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    prompt = row.get("prompt")
    if prompt is not None and len(prompt) >= 2:
        content = prompt[1].get("content", "") if isinstance(prompt[1], dict) else ""
        if isinstance(content, str):
            return re.sub(r"^(?:<image>)+", "", content).strip()
    return ""


def leading_image_prefix(row: pd.Series) -> str:
    prompt = row.get("prompt")
    if prompt is not None and len(prompt) >= 2 and isinstance(prompt[1], dict):
        content = prompt[1].get("content", "")
        if isinstance(content, str):
            match = re.match(r"^(?:<image>)+", content)
            if match:
                return match.group(0)
    images = row.get("images")
    try:
        return "<image>" * len(images)
    except Exception:
        return ""


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def cache_key(row_index: int, question: str, answer: str, model: str) -> str:
    payload = {
        "version": PROMPT_VERSION,
        "row_index": row_index,
        "question": question,
        "answer": answer,
        "model": model,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def build_generation_prompt(question: str, answer: str, data_source: str) -> str:
    return (
        "Create three plausible but incorrect multiple-choice distractors for a visual/document QA item.\n"
        "\n"
        "Requirements:\n"
        "- The correct answer is provided below; do not repeat it or paraphrase it as a distractor.\n"
        "- Distractors should be concise and similar in type/format to the correct answer.\n"
        "- Do not include an unanswerable/unknown/insufficient-information option; that will be added separately.\n"
        "- Do not mention option letters.\n"
        "- Return only strict JSON: {\"distractors\": [\"...\", \"...\", \"...\"]}\n"
        "\n"
        f"Data source: {data_source}\n"
        f"Question: {question}\n"
        f"Correct answer: {answer}\n"
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate_distractors(answer: str, distractors: list[Any]) -> tuple[list[str], str | None]:
    if len(distractors) < 3:
        return [], "too_few_distractors"
    answer_norm = normalize_for_match(answer)
    accepted: list[str] = []
    seen: set[str] = {answer_norm}
    for item in distractors:
        text = normalize_text(item)
        norm = normalize_for_match(text)
        if not text:
            continue
        if len(text) > 180:
            continue
        if has_unanswerable_semantics(text):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        accepted.append(text)
        if len(accepted) == 3:
            break
    if len(accepted) < 3:
        return accepted, "not_enough_valid_unique_distractors"
    return accepted, None


def maybe_import_insight_api(insight_doc_root: Path):
    if str(insight_doc_root) not in sys.path:
        sys.path.insert(0, str(insight_doc_root))
    try:
        from utils.api import create_async_openai_client, complete_chat_and_maybe_log
    except Exception:
        return None, None
    return create_async_openai_client, complete_chat_and_maybe_log


def create_api_client(insight_doc_root: Path, timeout: float):
    os.environ.setdefault("ENSURE_API_LOGGER", "1")
    create_client, complete_chat = maybe_import_insight_api(insight_doc_root)
    if create_client is not None and complete_chat is not None:
        return create_client(timeout=timeout), complete_chat
    return (
        AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
        ),
        None,
    )


async def call_api_for_distractors(
    *,
    client: AsyncOpenAI,
    complete_chat: Any | None,
    prompt: str,
    model: str,
    max_retries: int,
    max_completion_tokens: int,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You generate concise multiple-choice distractors as strict JSON.",
        },
        {"role": "user", "content": prompt},
    ]
    response = None
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if complete_chat is not None:
                response = await complete_chat(
                    messages=messages,
                    model=model,
                    client=client,
                    max_completion_tokens=max_completion_tokens,
                    response_format={"type": "json_object"},
                )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    response_format={"type": "json_object"},
                )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                raise
            await asyncio.sleep(min(20.0, 1.5 * (2**attempt)))
    if response is None:
        raise RuntimeError(f"API call failed: {last_error}")
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if not isinstance(content, str):
        raise RuntimeError("API response content is not text")
    return content.strip()


def sampled_distractors(
    *,
    row_index: int,
    answer: str,
    data_source: str,
    pools_by_source: dict[str, list[str]],
    global_pool: list[str],
    seed: int,
) -> list[str]:
    rng = random.Random(f"{seed}:{row_index}:{answer}")
    answer_norm = normalize_for_match(answer)
    candidates = list(pools_by_source.get(data_source, [])) + list(global_pool)
    rng.shuffle(candidates)
    out: list[str] = []
    seen = {answer_norm}
    for candidate in candidates:
        norm = normalize_for_match(candidate)
        if not norm or norm in seen:
            continue
        if len(candidate) > 180 or has_unanswerable_semantics(candidate):
            continue
        seen.add(norm)
        out.append(candidate)
        if len(out) == 3:
            return out
    return out


def build_mc_question(question: str, options: dict[str, str]) -> str:
    option_text = "\n".join(f"({letter}) {options[letter]}" for letter in ["A", "B", "C", "D", "E"])
    return (
        f"{question.strip()}\n\n"
        f"{option_text}\n\n"
        "Choose the letter name in front of the right option from A, B, C, D, E."
    )


def build_output_row(
    *,
    source_row: pd.Series,
    row_index: int,
    question: str,
    answer: str,
    distractors: list[str],
    e_option: str,
    seed: int,
    generation_key: str,
    model: str,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}:options:{row_index}:{answer}")
    letters = ["A", "B", "C", "D"]
    shuffled = [answer, *distractors[:3]]
    rng.shuffle(shuffled)
    options = dict(zip(letters, shuffled, strict=True))
    options["E"] = e_option
    correct_letter = next(letter for letter, value in options.items() if value == answer)
    mc_question = build_mc_question(question, options)

    prompt = copy.deepcopy(list(source_row["prompt"]))
    prefix = leading_image_prefix(source_row)
    prompt[1] = dict(prompt[1])
    prompt[1]["content"] = f"{prefix}{mc_question}"

    reward_model = copy.deepcopy(source_row["reward_model"])
    if not isinstance(reward_model, dict):
        reward_model = {}
    reward_model["ground_truth"] = correct_letter
    reward_model["style"] = reward_model.get("style", "rule")

    extra_info = copy.deepcopy(source_row["extra_info"])
    if not isinstance(extra_info, dict):
        extra_info = {}
    original_question = extra_info.get("question", question)
    extra_info.update(
        {
            "question": mc_question,
            "original_question": original_question,
            "original_ground_truth": copy.deepcopy(source_row["reward_model"].get("ground_truth")),
            "mc_correct_answer": answer,
            "mc_correct_letter": correct_letter,
            "mc_options": options,
            "mc_e_option": e_option,
            "mc_generation_key": generation_key,
            "mc_generation_model": model,
            "mc_builder": BUILDER_NAME,
            "prompt_style": f"{extra_info.get('prompt_style', 'insight_qwen_agent')}_mc_false_e",
        }
    )

    data_source = f"{source_row['data_source']}_mc_false_e"
    return {
        "images": copy.deepcopy(source_row["images"]),
        "data_source": data_source,
        "prompt": prompt,
        "reward_model": reward_model,
        "extra_info": extra_info,
        "agent_name": source_row.get("agent_name", "insight_qwen_agent"),
    }


def candidate_records(df: pd.DataFrame, args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    records: list[dict[str, Any]] = []
    include = set(args.include_data_source)
    exclude = set(args.exclude_data_source)
    for row_index, row in df.iterrows():
        data_source = str(row.get("data_source", ""))
        if include and data_source not in include:
            stats["skip_not_in_include_data_source"] += 1
            continue
        if data_source in exclude:
            stats["skip_excluded_data_source"] += 1
            continue
        lowered_ds = data_source.lower()
        if "answerable" not in lowered_ds or "unanswerable" in lowered_ds:
            stats["skip_not_answerable_source"] += 1
            continue
        question = get_question(row)
        if not question:
            stats["skip_missing_question"] += 1
            continue
        if len(question) > args.max_question_chars:
            stats["skip_question_too_long"] += 1
            continue
        reward_model = row.get("reward_model")
        if not isinstance(reward_model, dict):
            stats["skip_bad_reward_model"] += 1
            continue
        answer = parse_single_answer(reward_model.get("ground_truth"))
        if not answer:
            stats["skip_non_single_answer"] += 1
            continue
        if len(answer) > args.max_answer_chars:
            stats["skip_answer_too_long"] += 1
            continue
        if has_unanswerable_semantics(answer):
            stats["skip_answer_unanswerable_semantics"] += 1
            continue
        if not args.allow_multipart and (PART_MARKER_RE.search(question) or PART_MARKER_RE.search(answer)):
            stats["skip_multipart"] += 1
            continue
        records.append(
            {
                "row_index": int(row_index),
                "data_source": data_source,
                "question": question,
                "answer": answer,
            }
        )
        stats["candidate"] += 1
    return records, stats


def build_answer_pools(candidates: list[dict[str, Any]]) -> tuple[dict[str, list[str]], list[str]]:
    by_source: dict[str, list[str]] = defaultdict(list)
    global_pool: list[str] = []
    seen_global: set[str] = set()
    for record in candidates:
        answer = normalize_text(record["answer"])
        norm = normalize_for_match(answer)
        if not answer or norm in seen_global:
            continue
        seen_global.add(norm)
        by_source[record["data_source"]].append(answer)
        global_pool.append(answer)
    return by_source, global_pool


async def generate_all_distractors(
    *,
    selected: list[dict[str, Any]],
    args: argparse.Namespace,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    status_path: Path,
    pools_by_source: dict[str, list[str]],
    global_pool: list[str],
) -> dict[str, dict[str, Any]]:
    result_by_key: dict[str, dict[str, Any]] = {}
    cache_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    insight_doc_root = Path(args.insight_doc_root).expanduser().resolve()
    api_client = None
    complete_chat = None
    if args.distractor_mode == "api":
        api_client, complete_chat = create_api_client(insight_doc_root, args.timeout)

    async def process_one(record: dict[str, Any]) -> None:
        key = cache_key(record["row_index"], record["question"], record["answer"], args.model)
        record["generation_key"] = key
        if key in cache:
            result = cache[key]
            distractors, reason = validate_distractors(record["answer"], result.get("distractors", []))
            if reason is None:
                result_by_key[key] = {**result, "distractors": distractors}
                append_jsonl(
                    status_path,
                    {"time": now_iso(), "key": key, "row_index": record["row_index"], "status": "cache_hit"},
                )
                return

        if args.distractor_mode == "sampled":
            distractors = sampled_distractors(
                row_index=record["row_index"],
                answer=record["answer"],
                data_source=record["data_source"],
                pools_by_source=pools_by_source,
                global_pool=global_pool,
                seed=args.seed,
            )
            distractors, reason = validate_distractors(record["answer"], distractors)
            if reason is not None:
                append_jsonl(
                    status_path,
                    {
                        "time": now_iso(),
                        "key": key,
                        "row_index": record["row_index"],
                        "status": "failed",
                        "reason": reason,
                    },
                )
                return
            result = {"distractors": distractors, "mode": "sampled", "model": None, "raw": None}
            result_by_key[key] = result
            async with cache_lock:
                append_jsonl(cache_path, {"key": key, "result": result})
            append_jsonl(
                status_path,
                {"time": now_iso(), "key": key, "row_index": record["row_index"], "status": "sampled"},
            )
            return

        prompt = build_generation_prompt(record["question"], record["answer"], record["data_source"])
        async with semaphore:
            raw = ""
            distractors: list[str] = []
            last_error: Exception | None = None
            try:
                if api_client is None:
                    raise RuntimeError("API client was not initialized")
                for attempt in range(args.max_retries + 1):
                    try:
                        raw = await call_api_for_distractors(
                            client=api_client,
                            complete_chat=complete_chat,
                            prompt=prompt,
                            model=args.model,
                            max_retries=0,
                            max_completion_tokens=args.max_completion_tokens,
                        )
                        parsed = extract_json_object(raw)
                        if parsed is None:
                            raise ValueError("response_not_json")
                        distractors, reason = validate_distractors(record["answer"], parsed.get("distractors", []))
                        if reason is not None:
                            raise ValueError(reason)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        if attempt >= args.max_retries:
                            raise
                        await asyncio.sleep(min(20.0, 1.5 * (2**attempt)))
            except Exception as exc:  # noqa: BLE001
                append_jsonl(
                    status_path,
                    {
                        "time": now_iso(),
                        "key": key,
                        "row_index": record["row_index"],
                        "status": "failed",
                        "reason": repr(exc),
                        "last_error": repr(last_error) if last_error is not None else None,
                        "raw_preview": raw[:500] if raw else "",
                    },
                )
                return
            result = {
                "distractors": distractors,
                "mode": "api",
                "model": args.model,
                "raw": raw,
                "prompt_version": PROMPT_VERSION,
            }
            result_by_key[key] = result
            async with cache_lock:
                append_jsonl(cache_path, {"key": key, "result": result})
            append_jsonl(
                status_path,
                {"time": now_iso(), "key": key, "row_index": record["row_index"], "status": "generated"},
            )

    try:
        tasks = [asyncio.create_task(process_one(record)) for record in selected]
        completed = 0
        for task in asyncio.as_completed(tasks):
            await task
            completed += 1
            if args.progress_every > 0 and completed % args.progress_every == 0:
                print(f"[{now_iso()}] distractors processed {completed}/{len(selected)}", flush=True)
        return result_by_key
    finally:
        if api_client is not None:
            await api_client.close()


def main() -> None:
    args = parse_args()
    input_parquet = Path(args.input_parquet).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = (
        Path(args.output_parquet).expanduser().resolve()
        if args.output_parquet
        else output_dir / "insight_doc_rl_16k_prompt24k_r05_to_r035_mc_false_e-insight_qwen_agent.parquet"
    )
    cache_path = Path(args.cache_jsonl).expanduser().resolve() if args.cache_jsonl else output_dir / "distractor_cache.jsonl"
    status_path = Path(args.status_jsonl).expanduser().resolve() if args.status_jsonl else output_dir / "distractor_status.jsonl"
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_dir / "summary.json"

    if output_parquet.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output parquet already exists: {output_parquet}. Use --overwrite to replace it.")
    if args.distractor_mode == "api" and not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set for --distractor-mode api")

    df = pd.read_parquet(input_parquet)
    candidates, filter_stats = candidate_records(df, args)
    rng = random.Random(args.seed)
    selected = list(candidates)
    rng.shuffle(selected)
    if args.target_rows > 0:
        selected = selected[: args.target_rows]
    selected = sorted(selected, key=lambda item: item["row_index"])

    pools_by_source, global_pool = build_answer_pools(candidates)
    summary: dict[str, Any] = {
        "builder": BUILDER_NAME,
        "created_at": now_iso(),
        "input_parquet": str(input_parquet),
        "output_parquet": str(output_parquet),
        "target_rows": args.target_rows,
        "selected_rows": len(selected),
        "seed": args.seed,
        "distractor_mode": args.distractor_mode,
        "model": args.model if args.distractor_mode == "api" else None,
        "filter_stats": dict(filter_stats),
        "candidate_data_source_counts": dict(Counter(item["data_source"] for item in candidates)),
        "selected_data_source_counts": dict(Counter(item["data_source"] for item in selected)),
        "e_option": args.e_option,
    }

    if args.dry_run:
        write_json(summary_path, {**summary, "dry_run": True})
        print(json.dumps({**summary, "dry_run": True}, ensure_ascii=False, indent=2))
        return

    cache = load_cache(cache_path)
    started = time.time()
    distractors_by_key = asyncio.run(
        generate_all_distractors(
            selected=selected,
            args=args,
            cache=cache,
            cache_path=cache_path,
            status_path=status_path,
            pools_by_source=pools_by_source,
            global_pool=global_pool,
        )
    )
    output_rows: list[dict[str, Any]] = []
    build_stats: Counter = Counter()
    for record in selected:
        key = record["generation_key"]
        result = distractors_by_key.get(key)
        if result is None:
            build_stats["skip_missing_distractors"] += 1
            continue
        source_row = df.iloc[record["row_index"]]
        output_rows.append(
            build_output_row(
                source_row=source_row,
                row_index=record["row_index"],
                question=record["question"],
                answer=record["answer"],
                distractors=result["distractors"],
                e_option=args.e_option,
                seed=args.seed,
                generation_key=key,
                model=args.model if args.distractor_mode == "api" else "sampled",
            )
        )
        build_stats["built"] += 1

    if not output_rows:
        raise RuntimeError("No output rows were built.")

    out_df = pd.DataFrame(output_rows, columns=list(df.columns))
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_parquet, index=False)
    elapsed = time.time() - started
    summary.update(
        {
            "output_rows": len(out_df),
            "build_stats": dict(build_stats),
            "elapsed_s": elapsed,
            "cache_jsonl": str(cache_path),
            "status_jsonl": str(status_path),
        }
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
