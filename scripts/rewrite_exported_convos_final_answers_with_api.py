#!/usr/bin/env python3
"""Rewrite exported conversation final answers with an API model.

This is intentionally separate from SFT parquet conversion. It creates a new
exported-conversation directory where successfully rewritten rows are updated
and ineligible rows can be passed through unchanged so downstream conversion
still sees the original dataset accounting.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_exported_convos_to_qwen3_vl_zoom_in_sft import (  # noqa: E402
    build_degenerate_thresholds,
    degenerate_drop_reason,
)


PROMPT_VERSION = "qwen32b_style_v1"
REWRITE_METADATA_KEY = "final_answer_rewrite"
MARKER_NAME = "scripts/rewrite_exported_convos_final_answers_with_api.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing exported conversation JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write rewritten exported JSON files.")
    parser.add_argument("--model", default="gpt-5-nano", help="API model used for rewriting.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request API timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=4, help="Retry count for transient API failures.")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="Completion token budget, including model reasoning tokens.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent API rewrite requests.",
    )
    parser.add_argument(
        "--cache-jsonl",
        default=None,
        help="JSONL cache of successful rewrites. Defaults to OUTPUT_DIR/rewrite_cache.jsonl.",
    )
    parser.add_argument(
        "--status-jsonl",
        default=None,
        help="Append one status record per input. Defaults to OUTPUT_DIR/rewrite_status.jsonl.",
    )
    parser.add_argument(
        "--only-correct-answers",
        action="store_true",
        help=(
            "Rewrite only conversations whose accuracy_reward is exactly 1.0. "
            "Rows with other rewards are copied through unchanged so downstream "
            "conversion can still account for them."
        ),
    )
    parser.add_argument(
        "--drop-degenerate-conversations",
        action="store_true",
        help="Skip conversations with existing or computed degenerate assistant text markers.",
    )
    parser.add_argument("--degenerate-max-assistant-chars", type=int, default=50_000)
    parser.add_argument("--degenerate-max-assistant-words", type=int, default=8_000)
    parser.add_argument("--degenerate-min-unique-word-ratio", type=float, default=0.20)
    parser.add_argument("--degenerate-min-words-for-unique-ratio", type=int, default=1_000)
    parser.add_argument("--degenerate-max-same-word-run", type=int, default=10)
    parser.add_argument("--degenerate-ngram-size", type=int, default=8)
    parser.add_argument("--degenerate-max-ngram-repeats", type=int, default=50)
    parser.add_argument("--degenerate-min-words-for-ngram", type=int, default=1_000)
    parser.add_argument("--degenerate-preview-chars", type=int, default=240)
    parser.add_argument(
        "--max-api-failure-ratio",
        type=float,
        default=0.005,
        help=(
            "Exit nonzero if API/validation failures exceed this fraction of otherwise eligible rows. "
            "Successful rewrites remain on disk and are skipped on rerun."
        ),
    )
    parser.add_argument(
        "--max-api-failures",
        type=int,
        default=None,
        help="Optional absolute cap for API/validation failures before the run exits nonzero.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many input JSON files, useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run filtering/resume checks without calling the API or writing rewritten JSON files.",
    )
    parser.add_argument(
        "--ensure-api-logger",
        action="store_true",
        default=True,
        help="Require InSight-doc API logger import by setting ENSURE_API_LOGGER=1 before API helper import.",
    )
    parser.add_argument(
        "--no-ensure-api-logger",
        dest="ensure_api_logger",
        action="store_false",
        help="Do not force ENSURE_API_LOGGER=1.",
    )
    parser.add_argument(
        "--insight-doc-root",
        default=os.environ.get("INSIGHT_DOC_ROOT", str(REPO_ROOT.parent / "InSight-doc")),
        help="Path to the InSight-doc repo containing insight_doc.utils.api.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N input files.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def extract_part_labels(text: str) -> list[str]:
    import re

    labels = re.findall(r"\(([a-z])\)", text, flags=re.IGNORECASE)
    expected_ord = ord("a")
    part_labels: list[str] = []
    for label in labels:
        normalized = label.lower()
        if normalized == chr(expected_ord):
            part_labels.append(normalized)
            expected_ord += 1
    return part_labels if len(part_labels) >= 2 else []


def has_part_label(text: str, label: str) -> bool:
    import re

    escaped = re.escape(label)
    return (
        re.search(rf"(?<![A-Za-z0-9])\(?{escaped}\)?[\).:,]", text, flags=re.IGNORECASE) is not None
        or re.search(rf"\b(?:part|question|item)\s*\(?{escaped}\)?\b", text, flags=re.IGNORECASE) is not None
    )


def text_from_query_content(content: Any) -> str:
    if isinstance(content, dict):
        question = content.get("question")
        if isinstance(question, str):
            return question
        text = content.get("text")
        if isinstance(text, str):
            return text
    if isinstance(content, str):
        return content
    return ""


def source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": digest,
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_name = handle.name
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp_name, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_rewrite_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = item.get("key")
            output = item.get("output")
            if isinstance(key, str) and isinstance(output, str):
                cache[key] = output
    return cache


def rewrite_cache_key(*, question: str, original_message: str, reference_answer: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "question": question,
            "original_message": original_message,
            "reference_answer": reference_answer,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def completed_output_metadata(output_path: Path, fingerprint: dict[str, Any], model: str) -> dict[str, Any] | None:
    if not output_path.exists():
        return None
    try:
        record = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metadata = record.get(REWRITE_METADATA_KEY)
    if not isinstance(metadata, dict):
        return None
    matches = (
        metadata.get("marked_by") == MARKER_NAME
        and metadata.get("model") == model
        and metadata.get("source_sha256") == fingerprint["source_sha256"]
        and metadata.get("source_size") == fingerprint["source_size"]
        and metadata.get("source_mtime_ns") == fingerprint["source_mtime_ns"]
        and isinstance(metadata.get("rewrite_cache_key"), str)
    )
    return metadata if matches else None


def accuracy_reward(record: dict[str, Any]) -> float | None:
    reward = record.get("reward")
    if not isinstance(reward, dict):
        return None
    score = reward.get("score")
    if isinstance(score, dict) and score.get("accuracy_reward") is not None:
        return float(score["accuracy_reward"])
    if reward.get("accuracy_reward") is not None:
        return float(reward["accuracy_reward"])
    return None


def find_initial_question(record: dict[str, Any]) -> str:
    for message in record.get("conversation", []):
        if message.get("role") == "user" and message.get("type") == "query":
            return text_from_query_content(message.get("content"))
    return ""


def find_final_answer_message(record: dict[str, Any]) -> tuple[int, dict[str, Any], str, str] | None:
    conversation = record.get("conversation")
    if not isinstance(conversation, list):
        return None
    for index in range(len(conversation) - 1, -1, -1):
        message = conversation[index]
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant" or message.get("type") != "answer":
            continue
        content = message.get("content")
        if not isinstance(content, dict):
            return None
        think = normalize_text(content.get("think"))
        answer = normalize_text(content.get("answer"))
        if not answer:
            return None
        return index, message, think, answer
    return None


def build_original_plain_message(think: str, answer: str) -> str:
    if think and answer:
        return f"{think}\n\n{answer}"
    return think or answer


def build_prompt(question: str, original_message: str, reference_answer: str) -> str:
    return (
        "Rewrite this final visual-QA assistant response so it reads like a natural plain-text "
        "Qwen3-VL style SFT target.\n"
        "\n"
        "Style target, condensed from Qwen3-VL-32B final answers in this dataset:\n"
        "- Give a direct answer with brief visual/document evidence folded into normal prose.\n"
        "- Prefer one coherent response over a reasoning paragraph followed by a bare answer line.\n"
        "- Keep concrete values, names, dates, labels, routes, and measurements exact.\n"
        "- It is fine to start with phrases like \"Based on the document/image...\" when useful.\n"
        "- Keep the response concise, but do not remove necessary context for multi-part questions.\n"
        "- For multi-part questions, preserve the user's part labels such as (a), (b), (c), and keep each "
        "part's answer aligned with its label.\n"
        "- Use light formatting only when it genuinely clarifies a multi-part or structured answer.\n"
        "- If the original already reads naturally, return it unchanged.\n"
        "\n"
        "Hard constraints:\n"
        "- Use only the information in the original response and optional reference answer.\n"
        "- Do not add new facts, new uncertainty, citations, XML tags, or tool-call text.\n"
        "- Do not use a detached final line that merely repeats the answer.\n"
        "- Output only the rewritten assistant message.\n"
        "\n"
        f"Question:\n{question.strip()}\n\n"
        f"Original final assistant response:\n{original_message.strip()}\n\n"
        f"Reference final answer, if present:\n{reference_answer.strip()}\n"
    )


def validate_rewrite(output: str, original_message: str, reference_answer: str, question: str) -> str | None:
    if not output.strip():
        return "empty_output"
    lowered = output.lower()
    if any(tag in lowered for tag in ("<think", "</think", "<answer", "</answer", "<tool_call", "</tool_call")):
        return "contains_disallowed_xml_or_tool_tag"
    original_words = max(1, len(original_message.split()))
    if len(output.split()) > max(80, int(original_words * 1.6)):
        return "output_too_long"
    for label in extract_part_labels(question):
        if not has_part_label(output, label):
            return f"missing_part_label:({label})"
    return None


def load_api_helpers(insight_doc_root: Path, ensure_api_logger: bool):
    if ensure_api_logger:
        os.environ["ENSURE_API_LOGGER"] = "1"
    if str(insight_doc_root) not in sys.path:
        sys.path.insert(0, str(insight_doc_root))
    from insight_doc.utils.api import create_async_openai_client, query_model_with_retry

    return create_async_openai_client, query_model_with_retry


async def query_rewrite_api(
    *,
    prompt: str,
    model: str,
    timeout: float,
    max_retries: int,
    max_completion_tokens: int,
    insight_doc_root: Path,
    ensure_api_logger: bool,
) -> str:
    create_async_openai_client, query_model_with_retry = load_api_helpers(insight_doc_root, ensure_api_logger)
    client = create_async_openai_client(timeout=timeout)
    try:
        call = await query_model_with_retry(
            query=prompt,
            model=model,
            client=client,
            context=[
                {
                    "role": "system",
                    "content": "You rewrite existing assistant answers for supervised fine-tuning without changing facts.",
                }
            ],
            max_attempts=max_retries + 1,
            retry_initial_delay_sec=1.0,
            max_completion_tokens=max_completion_tokens,
        )
    finally:
        await client.close()
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
    return content.strip()


def write_rewritten_record(
    *,
    record: dict[str, Any],
    output_path: Path,
    fingerprint: dict[str, Any],
    source_path: Path,
    message_index: int,
    rewrite_key: str,
    model: str,
    rewritten_answer: str,
) -> None:
    conversation = record["conversation"]
    message = conversation[message_index]
    content = dict(message["content"])
    content["think"] = ""
    content["answer"] = rewritten_answer
    message["content"] = content
    record[REWRITE_METADATA_KEY] = {
        "schema_version": "final_answer_rewrite_v1",
        "marked_by": MARKER_NAME,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "source_path": str(source_path),
        "source_size": fingerprint["source_size"],
        "source_mtime_ns": fingerprint["source_mtime_ns"],
        "source_sha256": fingerprint["source_sha256"],
        "message_index": message_index,
        "rewrite_cache_key": rewrite_key,
        "rewritten_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_path, record)


def write_passthrough_record(
    *,
    record: dict[str, Any],
    output_path: Path,
    fingerprint: dict[str, Any],
    source_path: Path,
    model: str,
    reason: str,
) -> None:
    record = json.loads(json.dumps(record, ensure_ascii=False))
    record[REWRITE_METADATA_KEY] = {
        "schema_version": "final_answer_rewrite_v1",
        "marked_by": MARKER_NAME,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "source_path": str(source_path),
        "source_size": fingerprint["source_size"],
        "source_mtime_ns": fingerprint["source_mtime_ns"],
        "source_sha256": fingerprint["source_sha256"],
        "rewrite_cache_key": "",
        "passthrough_only_correct": True,
        "passthrough_reason": reason,
        "rewritten_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_path, record)


async def process_one(
    *,
    path: Path,
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    rewrite_cache: dict[str, str],
    cache_path: Path,
    degenerate_thresholds,
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
    output_metadata = completed_output_metadata(output_path, fingerprint, args.model)
    if output_metadata is not None:
        if output_metadata.get("passthrough_only_correct"):
            return {**status_base, "status": "skipped_existing_passthrough_only_correct"}
        return {**status_base, "status": "skipped_existing_output"}

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {**status_base, "status": "error", "reason": f"json_load_failed:{type(exc).__name__}:{exc}"}

    if args.only_correct_answers:
        reward = accuracy_reward(record)
        if reward != 1.0:
            write_passthrough_record(
                record=record,
                output_path=output_path,
                fingerprint=fingerprint,
                source_path=path,
                model=args.model,
                reason="only_correct_answers_passthrough",
            )
            return {**status_base, "status": "passthrough_only_correct", "accuracy_reward": reward}

    if args.drop_degenerate_conversations:
        reason = degenerate_drop_reason(record, degenerate_thresholds, args.degenerate_preview_chars)
        if reason is not None:
            return {**status_base, "status": "filtered_degenerate", "reason": reason}

    final_answer = find_final_answer_message(record)
    if final_answer is None:
        return {**status_base, "status": "filtered_no_final_answer"}
    message_index, _, think, answer = final_answer
    question = find_initial_question(record)
    original_message = build_original_plain_message(think, answer)
    rewrite_key = rewrite_cache_key(
        question=question,
        original_message=original_message,
        reference_answer=answer,
        model=args.model,
    )

    if args.dry_run:
        return {**status_base, "status": "dry_run_candidate", "rewrite_cache_key": rewrite_key}

    async with cache_lock:
        rewritten = rewrite_cache.get(rewrite_key)
    used_cache = rewritten is not None
    if rewritten is None:
        prompt = build_prompt(question, original_message, answer)
        try:
            rewritten = await query_rewrite_api(
                prompt=prompt,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_completion_tokens=args.max_completion_tokens,
                insight_doc_root=Path(args.insight_doc_root).expanduser().resolve(),
                ensure_api_logger=args.ensure_api_logger,
            )
        except Exception as exc:
            return {
                **status_base,
                "status": "api_failure",
                "reason": f"{type(exc).__name__}: {exc}",
                "rewrite_cache_key": rewrite_key,
            }
        invalid_reason = validate_rewrite(rewritten, original_message, answer, question)
        if invalid_reason is not None:
            return {
                **status_base,
                "status": "validation_failure",
                "reason": invalid_reason,
                "rewrite_cache_key": rewrite_key,
            }
        async with cache_lock:
            cached_after_api = rewrite_cache.get(rewrite_key)
            if cached_after_api is not None:
                rewritten = cached_after_api
                used_cache = True
            else:
                rewrite_cache[rewrite_key] = rewritten
                append_jsonl(cache_path, {"key": rewrite_key, "output": rewritten})

    write_rewritten_record(
        record=record,
        output_path=output_path,
        fingerprint=fingerprint,
        source_path=path,
        message_index=message_index,
        rewrite_key=rewrite_key,
        model=args.model,
        rewritten_answer=rewritten,
    )
    return {
        **status_base,
        "status": "rewritten_cached" if used_cache else "rewritten_api",
        "rewrite_cache_key": rewrite_key,
        "message_index": message_index,
        "original_words": len(original_message.split()),
        "rewritten_words": len(rewritten.split()),
    }


async def process_paths(
    *,
    paths: list[Path],
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    rewrite_cache: dict[str, str],
    cache_path: Path,
    status_path: Path,
    degenerate_thresholds,
) -> tuple[Counter[str], int, int]:
    status_counts: Counter[str] = Counter()
    eligible = 0
    failures = 0
    completed = 0
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    cache_lock = asyncio.Lock()

    async def run_one(path: Path) -> dict[str, Any]:
        async with semaphore:
            try:
                return await process_one(
                    path=path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    args=args,
                    rewrite_cache=rewrite_cache,
                    cache_path=cache_path,
                    degenerate_thresholds=degenerate_thresholds,
                    cache_lock=cache_lock,
                )
            except Exception as exc:
                return {
                    "path": str(path),
                    "relative_path": str(path.relative_to(input_dir)),
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }

    tasks = [asyncio.create_task(run_one(path)) for path in paths]
    for task in asyncio.as_completed(tasks):
        status = await task
        append_jsonl(status_path, status)
        status_name = status["status"]
        status_counts[status_name] += 1
        if status_name in {"rewritten_api", "rewritten_cached", "skipped_existing_output", "dry_run_candidate"}:
            eligible += 1
        if status_name in {"api_failure", "validation_failure"}:
            eligible += 1
            failures += 1
        if status_name in {"api_failure", "validation_failure", "error"}:
            print(f"{status_name.upper()} {status.get('path')}: {status.get('reason')}", file=sys.stderr)
        completed += 1
        if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == len(paths)):
            print(f"progress {completed}/{len(paths)} {dict(status_counts)}", flush=True)

    return status_counts, eligible, failures


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input dir does not exist or is not a directory: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        Path(args.cache_jsonl).expanduser().resolve()
        if args.cache_jsonl
        else output_dir / "rewrite_cache.jsonl"
    )
    status_path = (
        Path(args.status_jsonl).expanduser().resolve()
        if args.status_jsonl
        else output_dir / "rewrite_status.jsonl"
    )
    degenerate_thresholds = build_degenerate_thresholds(args)
    rewrite_cache = load_rewrite_cache(cache_path)
    paths = sorted(input_dir.glob("*.json"))
    if args.limit is not None:
        paths = paths[: args.limit]

    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"model={args.model}")
    print(f"concurrency={args.concurrency}")
    print(f"cache_jsonl={cache_path}")
    print(f"status_jsonl={status_path}")
    print(f"existing_cache_entries={len(rewrite_cache)}")
    print(f"files={len(paths)}")

    status_counts, eligible, failures = asyncio.run(
        process_paths(
            paths=paths,
            input_dir=input_dir,
            output_dir=output_dir,
            args=args,
            rewrite_cache=rewrite_cache,
            cache_path=cache_path,
            status_path=status_path,
            degenerate_thresholds=degenerate_thresholds,
        )
    )

    failure_ratio = failures / eligible if eligible else 0.0
    summary = {
        "status": "summary",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "counts": dict(status_counts),
        "eligible": eligible,
        "api_or_validation_failures": failures,
        "failure_ratio": failure_ratio,
        "thresholds": {
            "max_api_failure_ratio": args.max_api_failure_ratio,
            "max_api_failures": args.max_api_failures,
            "degenerate": asdict(degenerate_thresholds),
        },
    }
    append_jsonl(status_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    too_many_failures = failure_ratio > args.max_api_failure_ratio
    if args.max_api_failures is not None and failures > args.max_api_failures:
        too_many_failures = True
    return 2 if too_many_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
