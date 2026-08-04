#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from verl.utils.reward_score.vsearch_batch import compute_score_batch


DEFAULT_EXPORT_ROOT = Path(
    "/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/"
    "insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_new_sft_2ep_new_data_sp4_rl16k_prompt24k_weighted_refill_simple_source_unans014_mc_false_e05_judge_single_call_v1_from_sft/"
    "dude200_mmlongbench200_o3bench0502_insight_qwen_agent"
)


def assistant_message_to_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, dict):
        return str(content or "")

    parts: list[str] = []
    think = content.get("think")
    if think:
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
            chunks.append(prefix + assistant_message_to_text(message))
            saw_assistant = True
        elif role == "user" and message.get("type") == "tool_result":
            parts = message.get("parts") or []
            labels = [
                part.get("label")
                for part in parts
                if isinstance(part, dict) and part.get("kind") == "image_ref"
            ]
            label = labels[0] if labels else "Image returned"
            chunks.append(f"\nuser\n<tool_response>{label}</tool_response>")
    return "".join(chunks).strip()


def load_all_index_records(export_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    index_root = export_root / "index"
    for step_dir in sorted(index_root.glob("global_step_*"), key=lambda path: int(path.name.rsplit("_", 1)[1])):
        step = int(step_dir.name.rsplit("_", 1)[1])
        train_dir = step_dir / "train"
        for index_file in sorted(train_dir.glob("*.jsonl")):
            with index_file.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record["_index_file"] = str(index_file)
                    record["_index_line_no"] = line_no
                    record["_step"] = step
                    records.append(record)
    return records


def load_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload_path = Path(record["path"])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    reward = payload.get("reward") or {}
    extra_info = dict(payload.get("extra_info") or {})
    if "agent_name" not in extra_info:
        extra_info["agent_name"] = payload.get("agent_name") or reward.get("agent_name") or "insight_qwen_agent"

    conversation = payload.get("conversation") or []
    step = record.get("_step")
    score = reward.get("score") or {}
    status = payload.get("status") or {}
    return {
        "sample_key": f"step{step}_{record.get('export_id') or payload_path.stem}",
        "index_record": record,
        "payload_path": str(payload_path),
        "data_source": reward.get("data_source")
        or extra_info.get("data_source")
        or extra_info.get("subset")
        or "unknown",
        "ground_truth": reward.get("ground_truth"),
        "question": extra_info.get("question"),
        "question_id": extra_info.get("question_id") or record.get("question_id"),
        "document_id": extra_info.get("document_id") or record.get("document_id"),
        "subset": extra_info.get("subset"),
        "global_step": (payload.get("job") or {}).get("global_step") or record.get("global_step") or step,
        "conversation_len": len(conversation),
        "n_tool_calls": sum(
            1 for message in conversation if message.get("role") == "assistant" and message.get("type") == "tool_call"
        ),
        "stored_reward": reward.get("reward"),
        "stored_score": score.get("accuracy_reward"),
        "stored_extracted_answer": reward.get("extracted_answer") or score.get("extracted_answer"),
        "stored_judge_model_used": score.get("judge_model_used"),
        "stored_judge_fallback_used": score.get("judge_fallback_used"),
        "solution_str": conversation_to_solution_str(conversation),
        "extra_info": extra_info,
        "critical_failure": bool(status.get("critical_failure")),
        "failure_reasons": status.get("final_failure_reasons"),
    }


def write_outputs(out_dir: Path, samples: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "sampled_scored_full.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with (out_dir / "sampled_scored_compact.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            score = sample.get("current_score") or {}
            compact = {
                key: sample.get(key)
                for key in [
                    "sample_key",
                    "payload_path",
                    "global_step",
                    "data_source",
                    "subset",
                    "question_id",
                    "document_id",
                    "question",
                    "ground_truth",
                    "stored_score",
                    "stored_extracted_answer",
                    "stored_judge_model_used",
                    "stored_judge_fallback_used",
                    "conversation_len",
                    "n_tool_calls",
                    "critical_failure",
                    "failure_reasons",
                ]
            }
            compact.update(
                {
                    "current_accuracy_reward": score.get("accuracy_reward"),
                    "current_extracted_answer": score.get("extracted_answer"),
                    "current_judge_model_used": score.get("judge_model_used"),
                    "current_judge_fallback_used": score.get("judge_fallback_used"),
                    "current_compute_score_success": score.get("compute_score_success"),
                }
            )
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly audit judge behavior over training exported conversations.")
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--judge-workers", type=int, default=12)
    args = parser.parse_args()

    records = load_all_index_records(args.export_root)
    rng = random.Random(args.seed)
    selected = rng.sample(records, min(args.sample_size, len(records)))
    samples = [load_payload(record) for record in selected]
    valid_for_score = [
        sample for sample in samples if not sample["critical_failure"] and sample["ground_truth"] is not None
    ]

    print(
        f"records={len(records)} sampled={len(samples)} valid_for_score={len(valid_for_score)} seed={args.seed}",
        flush=True,
    )
    started_at = time.time()
    scores = compute_score_batch(
        data_sources=[sample["data_source"] for sample in valid_for_score],
        solution_strs=[sample["solution_str"] for sample in valid_for_score],
        ground_truths=[sample["ground_truth"] for sample in valid_for_score],
        extra_infos=[sample["extra_info"] for sample in valid_for_score],
        reward_type="conditioned_on_tool_reward",
        reward_weights={"format": 0.2, "accuracy": 0.8, "iou": 0.8, "tool": 1.0},
        format_reward={"must_have_answer": True, "simple": False},
        iou_reward={"iou_low": 0.25, "iou_high": 1.0, "pseudo_iou_reward_type": "caller_feedback"},
        tool_reward={"max_consecutive_iou": 0.6},
        judge_model=args.judge_model,
        fallback_judge_model=None,
        num_workers=args.judge_workers,
        task_timeout=600,
        min_success_rate=1.0,
        max_retries=5,
        retry_interval=10,
        insight_qwen_judge_mode="single_call_v2",
    )
    for sample, score in zip(valid_for_score, scores, strict=True):
        sample["current_score"] = score

    stored_available = [sample for sample in valid_for_score if sample.get("stored_score") is not None]
    disagreements = [
        sample
        for sample in stored_available
        if float(sample.get("stored_score"))
        != float((sample.get("current_score") or {}).get("accuracy_reward", -999))
    ]
    by_data_source = Counter(sample["data_source"] for sample in valid_for_score)
    by_step_bucket = Counter((int(sample["global_step"]) // 50) * 50 for sample in valid_for_score)
    current_correct = sum(
        1 for sample in valid_for_score if (sample.get("current_score") or {}).get("accuracy_reward") == 1.0
    )
    stored_correct = sum(1 for sample in stored_available if float(sample.get("stored_score")) == 1.0)
    summary = {
        "seed": args.seed,
        "sample_size": len(samples),
        "valid_for_score": len(valid_for_score),
        "elapsed_s": time.time() - started_at,
        "records_total": len(records),
        "global_step_min": min(sample["global_step"] for sample in valid_for_score),
        "global_step_max": max(sample["global_step"] for sample in valid_for_score),
        "data_source_counts": dict(by_data_source),
        "step_bucket_counts": dict(sorted(by_step_bucket.items())),
        "current_correct": current_correct,
        "current_accuracy": current_correct / len(valid_for_score) if valid_for_score else None,
        "stored_correct": stored_correct,
        "stored_accuracy": stored_correct / len(stored_available) if stored_available else None,
        "stored_available": len(stored_available),
        "num_disagreements": len(disagreements),
        "disagreement_keys": [sample["sample_key"] for sample in disagreements],
        "output_dir": str(args.out_dir),
    }
    write_outputs(args.out_dir, samples, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
