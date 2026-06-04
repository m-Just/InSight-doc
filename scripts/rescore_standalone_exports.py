#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import (  # noqa: E402
    build_reward_kwargs,
    build_summary_metrics,
    ground_truth_is_not_answerable,
    json_safe,
    parse_list_arg,
    question_type_contains_not_answerable,
)
from verl.utils.reward_score.vsearch_batch import compute_score_batch  # noqa: E402


def _last_answer_text(conversation: list[dict[str, Any]]) -> str:
    for message in reversed(conversation):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, dict):
            answer = content.get("answer")
            if answer:
                return str(answer)
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _assistant_message_to_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, dict):
        return str(content or "")

    parts: list[str] = []
    think = content.get("think")
    if think is not None:
        parts.append(f"<think>{think}</think>")

    if message.get("type") == "tool_call" and content.get("tool_call") is not None:
        parts.append(f"<tool_call>{json.dumps(content['tool_call'], ensure_ascii=False)}</tool_call>")
    elif content.get("answer") is not None:
        parts.append(f"<answer>{content['answer']}</answer>")
    return "\n".join(parts)


def conversation_to_solution_str(conversation: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    saw_assistant = False
    for message in conversation:
        role = message.get("role")
        if role == "assistant":
            prefix = "assistant\n" if not saw_assistant else "\nassistant\n"
            chunks.append(prefix + _assistant_message_to_text(message))
            saw_assistant = True
        elif role == "user" and message.get("type") == "tool_result":
            chunks.append("\nuser\n<tool_response>Image returned.</tool_response>")
    return "".join(chunks).strip()


def load_rows_by_question_id(val_files: list[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for parquet_path in val_files:
        df = pd.read_parquet(parquet_path, columns=["data_source", "reward_model", "extra_info"])
        for _, row in df.iterrows():
            extra_info = dict(row["extra_info"] or {})
            question_id = extra_info.get("question_id")
            if not question_id:
                continue
            rows[str(question_id)] = {
                "data_source": row["data_source"],
                "ground_truth": (row["reward_model"] or {}).get("ground_truth"),
                "extra_info": extra_info,
            }
    return rows


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def build_samples(output_dir: Path, rows_by_question_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    export_paths = sorted((output_dir / "exported_conversations").glob("*.json"))
    for export_path in export_paths:
        record = json.loads(export_path.read_text(encoding="utf-8"))
        export_extra_info = dict(record.get("extra_info") or {})
        question_id = export_extra_info.get("question_id")
        if question_id not in rows_by_question_id:
            raise KeyError(f"could not find source row for question_id={question_id!r} export={export_path}")
        row = rows_by_question_id[question_id]

        extra_info = {**row["extra_info"], **export_extra_info}
        extra_info["conversation_export_json_path"] = str(export_path)
        extra_info["failure_reasons"] = (record.get("status") or {}).get("final_failure_reasons")

        loop_params = (record.get("parameters") or {}).get("loop") or {}
        timing = loop_params.get("timing") or {}
        lengths = loop_params.get("lengths") or {}
        conversation = record.get("conversation") or []
        assistant_messages = [message for message in conversation if message.get("role") == "assistant"]

        sample = {
            "sample_index": (record.get("job") or {}).get("trajectory_sample_index"),
            "trial_idx": (record.get("job") or {}).get("rollout_n", 0),
            "uid": str(question_id),
            "data_source": row["data_source"],
            "ground_truth": row["ground_truth"],
            "solution_str": conversation_to_solution_str(conversation),
            "final_answer_text": _last_answer_text(conversation),
            "extra_info": extra_info,
            "conversation_export_json_path": str(export_path),
            "response_truncated": False,
            "critical_failure": (record.get("status") or {}).get("critical_failure"),
            "failure_reasons": (record.get("status") or {}).get("final_failure_reasons"),
            "num_turns": len(assistant_messages),
            "wall_time_s": finite_float(timing.get("conversation_wall_time")),
            "core_inference_time": finite_float(timing.get("core_inference_time")),
            "generate_sequences": finite_float(timing.get("generate_sequences")),
            "tool_parsing": finite_float(timing.get("tool_parsing")),
            "tool_calls": finite_float(timing.get("tool_calls")),
            "conversation_wall_time": finite_float(timing.get("conversation_wall_time")),
            "prompt_tokens": finite_float(lengths.get("prompt_tokens")),
            "response_tokens_total": finite_float(lengths.get("response_tokens_total")),
            "response_tokens_generated": finite_float(lengths.get("response_tokens_generated")),
            "response_tokens_tool": finite_float(lengths.get("response_tokens_tool")),
            "is_not_answerable": question_type_contains_not_answerable(extra_info.get("question_type"))
            or ground_truth_is_not_answerable(row["ground_truth"]),
        }
        samples.append(sample)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score exported conversations from a standalone eval run.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-files", required=True)
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--fallback-judge-model", default=None)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--judge-task-timeout", type=int, default=60)
    parser.add_argument("--judge-min-success-rate", type=float, default=0.99)
    parser.add_argument("--judge-max-retries", type=int, default=10)
    parser.add_argument("--judge-retry-interval", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows_by_question_id = load_rows_by_question_id(parse_list_arg(args.val_files))
    samples = build_samples(output_dir, rows_by_question_id)
    if not samples:
        raise RuntimeError(f"no exported conversations found under {output_dir / 'exported_conversations'}")

    reward_kwargs = build_reward_kwargs(args)
    score_dicts = compute_score_batch(
        data_sources=[sample["data_source"] for sample in samples],
        solution_strs=[sample["solution_str"] for sample in samples],
        ground_truths=[sample["ground_truth"] for sample in samples],
        extra_infos=[sample["extra_info"] for sample in samples],
        **reward_kwargs,
    )
    for sample, score in zip(samples, score_dicts, strict=True):
        sample["score"] = score

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(json_safe(sample), ensure_ascii=False) + "\n")

    summary = build_summary_metrics(samples)
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "rescore_from_exports": True,
        "val_files": parse_list_arg(args.val_files),
        "reward_kwargs": reward_kwargs,
        "num_samples": len(samples),
    }
    (output_dir / "rescore_manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"rescored {len(samples)} samples")
    print(f"metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
