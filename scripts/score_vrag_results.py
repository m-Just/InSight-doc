#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = Path(os.environ.get("VERL_ROOT", REPO_ROOT / "verl")).resolve()
for extra_path in (REPO_ROOT, VERL_ROOT):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from evals.core.export import (  # noqa: E402
    ground_truth_is_not_answerable,
    question_type_contains_not_answerable,
)
from evals.core.metrics import build_summary_metrics  # noqa: E402
from evals.core.utils import json_safe  # noqa: E402
from evals.judge import (  # noqa: E402
    DEFAULT_FALLBACK_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_REWARD_KWARGS,
    write_eval_summary_outputs,
)
from verl.utils.reward_score.vsearch_batch import compute_score_batch  # noqa: E402


ZERO_SCORE = {
    "format_reward": 0.0,
    "accuracy_reward": 0.0,
    "n_valid_tool_calls": 0.0,
    "score": 0.0,
    "compute_score_success": True,
    "extracted_answer": None,
}


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSON in {path}:{line_no}: {exc}") from exc
    return rows


def select_rows(
    rows: list[dict[str, Any]],
    *,
    sample_per_dataset: int | None,
    datasets: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    wanted = set(datasets)
    by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in datasets}
    for row in rows:
        dataset = str(row.get("dataset") or row.get("data_source") or "")
        if dataset in wanted:
            by_dataset[dataset].append(row)

    for dataset in datasets:
        group = by_dataset[dataset]
        if not group:
            raise ValueError(f"no rows found for dataset={dataset!r}")
        if sample_per_dataset is None or sample_per_dataset >= len(group):
            chosen = list(group)
        else:
            chosen = rng.sample(group, sample_per_dataset)
            chosen.sort(key=lambda row: int(row.get("row_index", 0) or 0))
        selected.extend(chosen)
    return selected


def trace_to_solution_str(row: dict[str, Any]) -> str:
    chunks: list[str] = []
    trace = row.get("trace") or []
    for idx, step in enumerate(trace):
        assistant = str(step.get("assistant") or "").strip()
        if assistant:
            prefix = "assistant\n" if not chunks else "\nassistant\n"
            chunks.append(prefix + assistant)
        action = step.get("action")
        if idx + 1 < len(trace) and action in {"search", "bbox"}:
            chunks.append("\nuser\n<tool_response>Tool returned.</tool_response>")
    return "".join(chunks).strip()


def build_sample(row: dict[str, Any], sample_index: int) -> dict[str, Any]:
    dataset = str(row.get("dataset") or row.get("data_source") or "unknown")
    question_id = str(row.get("question_id") or f"sample-{sample_index}")
    subset = row.get("subset")
    status = row.get("status")
    trace = row.get("trace") or []
    n_tool_calls = sum(1 for step in trace if step.get("action") in {"search", "bbox"})
    solution_str = trace_to_solution_str(row) if status == "answered" else ""
    extra_info = {
        "agent_name": "insight_qwen_agent",
        "question": row.get("question") or "",
        "question_id": question_id,
        "question_type": subset,
        "subset": subset,
        "vrag_status": status,
        "document_id": row.get("document_id"),
        "row_index": row.get("row_index"),
        "source_parquet": row.get("source_parquet"),
    }
    ground_truth = row.get("ground_truth")
    return {
        "sample_index": sample_index,
        "trial_idx": 0,
        "uid": question_id,
        "data_source": dataset,
        "ground_truth": ground_truth,
        "solution_str": solution_str,
        "final_answer_text": row.get("prediction") or "",
        "extra_info": extra_info,
        "response_truncated": False,
        "critical_failure": False,
        "failure_reasons": None if status == "answered" else [f"vrag_status={status}"],
        "num_turns": len(trace),
        "n_tool_calls": n_tool_calls,
        "wall_time_s": None,
        "core_inference_time": None,
        "conversation_wall_time": None,
        "is_not_answerable": question_type_contains_not_answerable(subset)
        or ground_truth_is_not_answerable(ground_truth),
        "vrag_prediction": row.get("prediction"),
        "vrag_status": status,
    }


def build_reward_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    reward_kwargs = json.loads(json.dumps(DEFAULT_REWARD_KWARGS))
    reward_kwargs.update(
        {
            "judge_model": args.judge_model,
            "fallback_judge_model": args.fallback_judge_model or None,
            "num_workers": args.judge_workers,
            "task_timeout": args.judge_task_timeout,
            "min_success_rate": args.judge_min_success_rate,
            "max_retries": args.judge_max_retries,
            "retry_interval": args.judge_retry_interval,
            "insight_qwen_judge_mode": args.insight_qwen_judge_mode,
        }
    )
    return reward_kwargs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    tmp.replace(path)


def normalize_scored_sample_keys(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_keys: set[str] = set()
    for sample in samples:
        score = sample.get("score")
        if isinstance(score, dict):
            score_keys.update(score)
    if not score_keys:
        return samples

    normalized: list[dict[str, Any]] = []
    for sample in samples:
        current = dict(sample)
        score = current.get("score")
        if isinstance(score, dict):
            current["score"] = {key: score.get(key) for key in sorted(score_keys)}
        normalized.append(current)
    return normalized


def attach_zero_scores(samples: list[dict[str, Any]]) -> None:
    for sample in samples:
        if sample.get("vrag_status") != "answered":
            sample["score"] = dict(ZERO_SCORE)


def restore_existing_scores(samples: list[dict[str, Any]], output_dir: Path) -> int:
    samples_path = output_dir / "samples.jsonl"
    if not samples_path.exists():
        return 0
    restored = 0
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(samples_path):
        score = row.get("score")
        if not isinstance(score, dict):
            continue
        key = (str(row.get("data_source")), str(row.get("uid")))
        existing[key] = score
    for sample in samples:
        key = (str(sample.get("data_source")), str(sample.get("uid")))
        score = existing.get(key)
        if score is not None:
            sample["score"] = score
            restored += 1
    return restored


def sample_needs_api_score(sample: dict[str, Any]) -> bool:
    return sample.get("score") is None and sample.get("vrag_status") == "answered"


def score_samples(
    samples: list[dict[str, Any]],
    *,
    reward_kwargs: dict[str, Any],
    batch_size: int,
    output_dir: Path,
) -> None:
    pending = [sample for sample in samples if sample_needs_api_score(sample)]
    total = len(pending)
    started = time.perf_counter()
    for batch_start in range(0, total, batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        score_dicts = compute_score_batch(
            data_sources=[sample["data_source"] for sample in batch],
            solution_strs=[sample["solution_str"] for sample in batch],
            ground_truths=[sample["ground_truth"] for sample in batch],
            extra_infos=[sample["extra_info"] for sample in batch],
            **reward_kwargs,
        )
        for sample, score in zip(batch, score_dicts, strict=True):
            sample["score"] = score
        write_outputs(
            output_dir=output_dir,
            samples=samples,
            reward_kwargs=reward_kwargs,
            wall_time_s=time.perf_counter() - started,
            done=False,
        )
        scored = min(batch_start + len(batch), total)
        print(f"scored {scored}/{total} answered rows", flush=True)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted({str(sample["data_source"]) for sample in samples}):
        group = [sample for sample in samples if str(sample["data_source"]) == dataset]
        accs = [
            finite_float((sample.get("score") or {}).get("accuracy_reward"))
            for sample in group
        ]
        acc_values = [value for value in accs if value is not None]
        by_dataset[dataset] = {
            "n": len(group),
            "num_scored": len(acc_values),
            "accuracy_mean": statistics.fmean(acc_values) if acc_values else None,
            "num_correct": sum(1 for value in acc_values if value == 1.0),
            "num_max_steps_or_nonanswered": sum(1 for sample in group if sample.get("vrag_status") != "answered"),
        }
    all_accs = [
        finite_float((sample.get("score") or {}).get("accuracy_reward"))
        for sample in samples
    ]
    all_acc_values = [value for value in all_accs if value is not None]
    return {
        "n": len(samples),
        "num_scored": len(all_acc_values),
        "accuracy_mean": statistics.fmean(all_acc_values) if all_acc_values else None,
        "num_correct": sum(1 for value in all_acc_values if value == 1.0),
        "by_dataset": by_dataset,
    }


def write_outputs(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    reward_kwargs: dict[str, Any],
    wall_time_s: float,
    done: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_samples = normalize_scored_sample_keys(samples)
    write_jsonl(output_dir / "samples.jsonl", normalized_samples)
    scored_samples = [sample for sample in normalized_samples if sample.get("score") is not None]
    metrics = build_summary_metrics(scored_samples) if scored_samples else {}
    metrics["vrag_summary"] = summarize_samples(normalized_samples)
    metrics["judge_progress"] = {
        "num_samples": len(normalized_samples),
        "num_scored_samples": len(scored_samples),
        "done": done,
    }
    metrics["wall_times"] = {"score_wall_time_s": wall_time_s}
    metrics["eval_summary_outputs"] = write_eval_summary_outputs(output_dir, normalized_samples)
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "reward_kwargs": reward_kwargs,
        "num_samples": len(normalized_samples),
        "num_scored_samples": len(scored_samples),
        "done": done,
        "wall_time_s": wall_time_s,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if done:
        (output_dir / "done").touch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score VRAG eval JSONL with the standalone InSight scorer.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-per-dataset", type=int, default=None)
    parser.add_argument("--datasets", default="mmlongbench,longdocurl")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--fallback-judge-model", default=DEFAULT_FALLBACK_JUDGE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--judge-task-timeout", type=int, default=600)
    parser.add_argument("--judge-min-success-rate", type=float, default=0.99)
    parser.add_argument("--judge-max-retries", type=int, default=10)
    parser.add_argument("--judge-retry-interval", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--insight-qwen-judge-mode",
        choices=["legacy_prompt_v2"],
        default="legacy_prompt_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    rows = load_jsonl(args.input_jsonl)
    selected_rows = select_rows(
        rows,
        sample_per_dataset=args.sample_per_dataset,
        datasets=datasets,
        seed=args.seed,
    )
    samples = [build_sample(row, idx) for idx, row in enumerate(selected_rows)]
    attach_zero_scores(samples)
    restored = restore_existing_scores(samples, args.output_dir) if args.resume else 0
    if restored:
        print(f"restored {restored} existing scores from {args.output_dir / 'samples.jsonl'}", flush=True)
    reward_kwargs = build_reward_kwargs(args)
    started = time.perf_counter()
    write_outputs(
        output_dir=args.output_dir,
        samples=samples,
        reward_kwargs=reward_kwargs,
        wall_time_s=0.0,
        done=False,
    )
    score_samples(
        samples,
        reward_kwargs=reward_kwargs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    wall_time_s = time.perf_counter() - started
    write_outputs(
        output_dir=args.output_dir,
        samples=samples,
        reward_kwargs=reward_kwargs,
        wall_time_s=wall_time_s,
        done=True,
    )
    print(json.dumps(json_safe(summarize_samples(samples)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
