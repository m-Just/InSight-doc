#!/usr/bin/env python3
"""Mark degenerate exported conversation JSON files.

This script scans exported conversation JSON files using text-only assistant
message statistics. It is intended to catch pathological generations such as
very long repeated phrases before converting exports to SFT parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


QUALITY_FIELD = "quality_flags"
MARKER_SCHEMA_VERSION = "conversation_quality_v1"
MARKER_NAME = "scripts/mark_bad_exported_conversations.py"
WORD_RE = re.compile(r"\S+")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Thresholds:
    max_assistant_chars: int
    max_assistant_words: int
    min_unique_word_ratio: float
    min_words_for_unique_ratio: int
    max_same_word_run: int
    ngram_size: int
    max_ngram_repeats: int
    min_words_for_ngram: int


@dataclass
class TextStats:
    assistant_chars: int
    assistant_words: int
    unique_words: int
    unique_word_ratio: float
    max_same_word_run: int
    ngram_size: int
    max_ngram_repeats: int
    top_repeated_ngram: str | None
    preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan exported conversation JSON files and mark examples with "
            "degenerate assistant text."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing exported conversation JSON files. Scanned recursively.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write quality markers back into flagged JSON files. Default is report-only.",
    )
    parser.add_argument(
        "--mark-clean",
        action="store_true",
        help="With --write, also write a clean quality marker to non-flagged JSON files.",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Remove existing markers written by this script before applying new results.",
    )
    parser.add_argument(
        "--output-jsonl",
        help="Optional path to write one JSON summary record per scanned file.",
    )
    parser.add_argument(
        "--fail-on-bad",
        action="store_true",
        help="Exit with status 2 if any bad examples are found.",
    )
    parser.add_argument("--max-assistant-chars", type=int, default=50_000)
    parser.add_argument("--max-assistant-words", type=int, default=8_000)
    parser.add_argument("--min-unique-word-ratio", type=float, default=0.20)
    parser.add_argument("--min-words-for-unique-ratio", type=int, default=1_000)
    parser.add_argument("--max-same-word-run", type=int, default=10)
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--max-ngram-repeats", type=int, default=50)
    parser.add_argument("--min-words-for-ngram", type=int, default=1_000)
    parser.add_argument("--preview-chars", type=int, default=240)
    return parser.parse_args()


def normalize_text(text: str, max_chars: int | None = None) -> str:
    text = WHITESPACE_RE.sub(" ", text).strip()
    if max_chars is not None:
        return text[:max_chars]
    return text


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        return "".join(chunks)
    if isinstance(content, dict):
        chunks = []
        for key in ("think", "answer", "text", "tool_call"):
            value = content.get(key)
            if isinstance(value, str):
                chunks.append(value)
            elif value is not None and key == "tool_call":
                chunks.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        extracted_tags = content.get("extracted_tags")
        if isinstance(extracted_tags, dict):
            for value in extracted_tags.values():
                if isinstance(value, str):
                    chunks.append(value)
                elif value is not None:
                    chunks.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return "\n".join(chunks)
    return str(content)


def assistant_message_text(message: dict[str, Any]) -> str:
    return text_from_content(message.get("content"))


def compute_text_stats(text: str, thresholds: Thresholds, preview_chars: int) -> TextStats:
    words = WORD_RE.findall(text.lower())
    unique_words = len(set(words))
    unique_word_ratio = unique_words / len(words) if words else 1.0

    max_same_word_run = 0
    current_run = 0
    previous_word = None
    for word in words:
        if word == previous_word:
            current_run += 1
        else:
            current_run = 1
            previous_word = word
        max_same_word_run = max(max_same_word_run, current_run)

    max_ngram_repeats = 0
    top_repeated_ngram = None
    ngram_size = thresholds.ngram_size
    if ngram_size > 0 and len(words) >= max(thresholds.min_words_for_ngram, ngram_size):
        ngrams = Counter(tuple(words[index : index + ngram_size]) for index in range(len(words) - ngram_size + 1))
        if ngrams:
            top_ngram, max_ngram_repeats = ngrams.most_common(1)[0]
            top_repeated_ngram = " ".join(top_ngram)

    return TextStats(
        assistant_chars=len(text),
        assistant_words=len(words),
        unique_words=unique_words,
        unique_word_ratio=round(unique_word_ratio, 6),
        max_same_word_run=max_same_word_run,
        ngram_size=ngram_size,
        max_ngram_repeats=max_ngram_repeats,
        top_repeated_ngram=top_repeated_ngram,
        preview=normalize_text(text, preview_chars),
    )


def reasons_for_stats(stats: TextStats, thresholds: Thresholds) -> list[str]:
    reasons: list[str] = []
    if stats.assistant_chars > thresholds.max_assistant_chars:
        reasons.append(
            f"assistant_chars>{thresholds.max_assistant_chars} ({stats.assistant_chars})"
        )
    if stats.assistant_words > thresholds.max_assistant_words:
        reasons.append(
            f"assistant_words>{thresholds.max_assistant_words} ({stats.assistant_words})"
        )
    if (
        stats.assistant_words >= thresholds.min_words_for_unique_ratio
        and stats.unique_word_ratio < thresholds.min_unique_word_ratio
    ):
        reasons.append(
            "unique_word_ratio"
            f"<{thresholds.min_unique_word_ratio} ({stats.unique_word_ratio})"
        )
    if stats.max_same_word_run >= thresholds.max_same_word_run:
        reasons.append(
            f"max_same_word_run>={thresholds.max_same_word_run} ({stats.max_same_word_run})"
        )
    if (
        stats.assistant_words >= thresholds.min_words_for_ngram
        and stats.max_ngram_repeats >= thresholds.max_ngram_repeats
    ):
        reasons.append(
            f"{thresholds.ngram_size}gram_repeats>={thresholds.max_ngram_repeats} "
            f"({stats.max_ngram_repeats})"
        )
    return reasons


def iter_assistant_messages(record: dict[str, Any]):
    for index, message in enumerate(record.get("conversation", [])):
        if isinstance(message, dict) and message.get("role") == "assistant":
            yield index, message


def clear_existing_markers(record: dict[str, Any]) -> None:
    marker = record.get(QUALITY_FIELD)
    if isinstance(marker, dict) and marker.get("marked_by") == MARKER_NAME:
        record.pop(QUALITY_FIELD, None)
    for _, message in iter_assistant_messages(record):
        marker = message.get(QUALITY_FIELD)
        if isinstance(marker, dict) and marker.get("marked_by") == MARKER_NAME:
            message.pop(QUALITY_FIELD, None)


def build_marker(
    *,
    bad_example: bool,
    reasons: list[str],
    stats: TextStats,
    thresholds: Thresholds,
    message_indices: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": MARKER_SCHEMA_VERSION,
        "marked_by": MARKER_NAME,
        "bad_example": bad_example,
        "reasons": reasons,
        "assistant_message_indices": message_indices,
        "assistant_text_stats": asdict(stats),
        "thresholds": asdict(thresholds),
    }


def scan_record(
    record: dict[str, Any],
    thresholds: Thresholds,
    preview_chars: int,
) -> tuple[bool, dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    message_results: list[tuple[int, dict[str, Any]]] = []
    all_texts: list[str] = []
    bad_reasons: list[str] = []
    bad_message_indices: list[int] = []

    for message_index, message in iter_assistant_messages(record):
        text = assistant_message_text(message)
        all_texts.append(text)
        stats = compute_text_stats(text, thresholds, preview_chars)
        reasons = reasons_for_stats(stats, thresholds)
        if reasons:
            bad_message_indices.append(message_index)
            bad_reasons.extend(f"message[{message_index}]: {reason}" for reason in reasons)
        message_results.append(
            (
                message_index,
                build_marker(
                    bad_example=bool(reasons),
                    reasons=reasons,
                    stats=stats,
                    thresholds=thresholds,
                    message_indices=[message_index],
                ),
            )
        )

    combined_text = "\n".join(all_texts)
    combined_stats = compute_text_stats(combined_text, thresholds, preview_chars)
    combined_reasons = reasons_for_stats(combined_stats, thresholds)
    if combined_reasons:
        bad_reasons.extend(f"combined: {reason}" for reason in combined_reasons)
    bad_example = bool(bad_reasons)

    top_level_marker = build_marker(
        bad_example=bad_example,
        reasons=bad_reasons,
        stats=combined_stats,
        thresholds=thresholds,
        message_indices=bad_message_indices,
    )
    return bad_example, top_level_marker, message_results


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
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


def scan_path(
    path: Path,
    thresholds: Thresholds,
    preview_chars: int,
    *,
    write: bool,
    mark_clean: bool,
    clear_existing: bool,
) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if clear_existing:
        clear_existing_markers(record)

    bad_example, top_level_marker, message_results = scan_record(record, thresholds, preview_chars)

    should_write = write and (bad_example or mark_clean or clear_existing)
    if should_write:
        if bad_example or mark_clean:
            record[QUALITY_FIELD] = top_level_marker
            bad_message_indices = set(top_level_marker["assistant_message_indices"])
            for message_index, message_marker in message_results:
                if message_index in bad_message_indices or mark_clean:
                    record["conversation"][message_index][QUALITY_FIELD] = message_marker
        atomic_write_json(path, record)

    return {
        "path": str(path),
        "bad_example": bad_example,
        "reasons": top_level_marker["reasons"],
        "assistant_message_indices": top_level_marker["assistant_message_indices"],
        "assistant_text_stats": top_level_marker["assistant_text_stats"],
        "written": bool(should_write),
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input dir does not exist or is not a directory: {input_dir}")

    thresholds = Thresholds(
        max_assistant_chars=args.max_assistant_chars,
        max_assistant_words=args.max_assistant_words,
        min_unique_word_ratio=args.min_unique_word_ratio,
        min_words_for_unique_ratio=args.min_words_for_unique_ratio,
        max_same_word_run=args.max_same_word_run,
        ngram_size=args.ngram_size,
        max_ngram_repeats=args.max_ngram_repeats,
        min_words_for_ngram=args.min_words_for_ngram,
    )

    json_paths = sorted(input_dir.rglob("*.json"))
    summaries: list[dict[str, Any]] = []
    num_bad = 0
    num_written = 0

    for path in json_paths:
        try:
            summary = scan_path(
                path,
                thresholds,
                args.preview_chars,
                write=args.write,
                mark_clean=args.mark_clean,
                clear_existing=args.clear_existing,
            )
        except Exception as exc:
            summary = {
                "path": str(path),
                "bad_example": False,
                "error": str(exc),
                "written": False,
            }
        summaries.append(summary)
        if summary.get("bad_example"):
            num_bad += 1
            stats = summary.get("assistant_text_stats", {})
            print(
                "BAD "
                f"{summary['path']} "
                f"chars={stats.get('assistant_chars')} "
                f"words={stats.get('assistant_words')} "
                f"uniq={stats.get('unique_word_ratio')} "
                f"max_ngram={stats.get('max_ngram_repeats')}"
            )
            for reason in summary.get("reasons", []):
                print(f"  - {reason}")
        if summary.get("written"):
            num_written += 1

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for summary in summaries:
                handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")

    num_errors = sum(1 for summary in summaries if "error" in summary)
    print(
        f"Scanned {len(json_paths)} JSON files; bad={num_bad}; "
        f"written={num_written}; errors={num_errors}"
    )
    if args.output_jsonl:
        print(f"Wrote summary JSONL: {args.output_jsonl}")
    if num_errors:
        return 1
    if args.fail_on_bad and num_bad:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
