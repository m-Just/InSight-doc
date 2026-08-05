#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = Path(os.environ.get("VERL_ROOT", REPO_ROOT / "verl")).resolve()
for extra_path in (REPO_ROOT, VERL_ROOT):
    if extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from evals.core.metrics import build_summary_metrics
from evals.core.resume import iter_resume_record_samples, sample_has_score, write_samples_jsonl_atomic
from evals.core.utils import json_safe, progress_bar
from verl.utils.reward_score.vsearch_batch import compute_score_batch


POLL_INTERVAL_SECONDS = float(os.environ.get("JUDGE_POLL_INTERVAL_SECONDS", "30"))
DEFAULT_JUDGE_MODEL = "gpt-5-nano"
DEFAULT_FALLBACK_JUDGE_MODEL = ""
DEFAULT_JUDGE_TASK_TIMEOUT_SECONDS = 60
JUDGE_MIN_SUCCESS_RATE = float(os.environ.get("JUDGE_MIN_SUCCESS_RATE", "0.99"))
JUDGE_MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "10"))
JUDGE_RETRY_INTERVAL_SECONDS = int(os.environ.get("JUDGE_RETRY_INTERVAL_SECONDS", "30"))
JUDGE_BATCH_SIZE = int(os.environ.get("JUDGE_BATCH_SIZE", "0"))


DEFAULT_REWARD_KWARGS = {
    "reward_type": "conditioned_on_tool_reward",
    "reward_weights": {
        "format": 0.2,
        "accuracy": 0.8,
        "iou": 0.8,
        "tool": 1.0,
    },
    "format_reward": {
        "must_have_answer": True,
        "simple": False,
    },
    "iou_reward": {
        "iou_low": 0.25,
        "iou_high": 1.0,
        "pseudo_iou_reward_type": "caller_feedback",
    },
    "tool_reward": {
        "max_consecutive_iou": 0.6,
    },
}


def resolve_judge_task_timeout_seconds(args: argparse.Namespace) -> int:
    if "JUDGE_TASK_TIMEOUT_SECONDS" in os.environ:
        return int(os.environ["JUDGE_TASK_TIMEOUT_SECONDS"])
    return DEFAULT_JUDGE_TASK_TIMEOUT_SECONDS


def build_reward_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    task_timeout_seconds = resolve_judge_task_timeout_seconds(args)
    reward_kwargs = json.loads(json.dumps(DEFAULT_REWARD_KWARGS))
    reward_kwargs.update(
        {
            "judge_model": args.judge_model,
            "fallback_judge_model": args.fallback_judge_model or None,
            "num_workers": args.judge_workers,
            "task_timeout": task_timeout_seconds,
            "min_success_rate": JUDGE_MIN_SUCCESS_RATE,
            "max_retries": JUDGE_MAX_RETRIES,
            "retry_interval": JUDGE_RETRY_INTERVAL_SECONDS,
            "insight_qwen_judge_mode": args.insight_qwen_judge_mode,
        }
    )
    return reward_kwargs


def load_samples_from_jsonl(path: Path) -> dict[int, tuple[float, dict[str, Any]]]:
    samples: dict[int, tuple[float, dict[str, Any]]] = {}
    for job_idx, sample, timestamp in iter_resume_record_samples(path):
        samples[job_idx] = (timestamp, sample)
    return samples


def load_rollout_samples(rollout_dir: Path) -> dict[int, dict[str, Any]]:
    candidates: dict[int, tuple[float, int, dict[str, Any]]] = {}
    sequence = 0
    checkpoint_dir = rollout_dir / "checkpoints"
    paths = []
    if checkpoint_dir.exists():
        paths.extend(sorted(checkpoint_dir.glob("*.jsonl")))
    for path in (rollout_dir / "samples.jsonl",):
        if path.exists():
            paths.append(path)
    for path in paths:
        for job_idx, sample, timestamp in iter_resume_record_samples(path):
            sequence += 1
            previous = candidates.get(job_idx)
            item = (float(timestamp), sequence, sample)
            if previous is None or item[:2] >= previous[:2]:
                candidates[job_idx] = item
    return {job_idx: sample for job_idx, (_, _, sample) in candidates.items()}


def merge_samples(
    rollout_samples: dict[int, dict[str, Any]],
    judged_samples: dict[int, dict[str, Any]],
    *,
    rescore_existing: bool,
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for job_idx, sample in rollout_samples.items():
        sample = dict(sample)
        judged = judged_samples.get(job_idx)
        if judged is not None and not rescore_existing and sample_has_score(judged):
            sample["score"] = judged["score"]
        merged[job_idx] = sample
    for job_idx, sample in judged_samples.items():
        if job_idx not in merged:
            merged[job_idx] = dict(sample)
    return [merged[job_idx] for job_idx in sorted(merged)]


def load_judged_samples(output_dir: Path) -> dict[int, dict[str, Any]]:
    judged: dict[int, dict[str, Any]] = {}
    for path in (output_dir / "samples.jsonl", output_dir / "scored_samples.jsonl"):
        if not path.exists():
            continue
        for job_idx, (_, sample) in load_samples_from_jsonl(path).items():
            judged[job_idx] = sample
    return judged


def load_default_agent_name(rollout_dir: Path) -> str | None:
    basic_config_path = rollout_dir / "basic_config.json"
    if not basic_config_path.exists():
        return None
    try:
        basic_config = json.loads(basic_config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    agent = basic_config.get("agent") if isinstance(basic_config, dict) else None
    if not isinstance(agent, dict):
        return None
    name = agent.get("name") or (agent.get("settings") or {}).get("name")
    return str(name) if name else None


def normalize_sample_for_judge(sample: dict[str, Any], *, default_agent_name: str | None) -> dict[str, Any]:
    sample = dict(sample)
    extra_info = dict(sample.get("extra_info") or {})
    if default_agent_name and "agent_name" not in extra_info:
        extra_info["agent_name"] = default_agent_name
    sample["extra_info"] = extra_info
    return sample


def sample_is_terminal_for_judge(sample: dict[str, Any]) -> bool:
    return sample_has_score(sample) or bool(sample.get("critical_failure"))


def sample_needs_judge_score(sample: dict[str, Any], *, rescore_existing: bool) -> bool:
    if sample.get("critical_failure"):
        return False
    return bool(rescore_existing or not sample_has_score(sample))


def build_failure_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    by_data_source: dict[str, dict[str, Any]] = {}
    for sample in samples:
        data_source = str(sample.get("data_source") or "unknown")
        ds_summary = by_data_source.setdefault(
            data_source,
            {
                "n": 0,
                "num_scored": 0,
                "num_critical_failure": 0,
                "num_unscored_noncritical": 0,
                "failure_reasons": {},
            },
        )
        ds_summary["n"] += 1
        if sample_has_score(sample):
            ds_summary["num_scored"] += 1
            continue
        if sample.get("critical_failure"):
            ds_summary["num_critical_failure"] += 1
            reasons = sample.get("failure_reasons") or ["critical_failure"]
            for reason in reasons:
                reason_text = str(reason).split("\n")[0]
                by_reason[reason_text] = by_reason.get(reason_text, 0) + 1
                ds_summary["failure_reasons"][reason_text] = ds_summary["failure_reasons"].get(reason_text, 0) + 1
        else:
            ds_summary["num_unscored_noncritical"] += 1
    return {
        "num_samples": len(samples),
        "num_scored": sum(1 for sample in samples if sample_has_score(sample)),
        "num_critical_failure": sum(1 for sample in samples if sample.get("critical_failure")),
        "num_unscored_noncritical": sum(
            1 for sample in samples if not sample_has_score(sample) and not sample.get("critical_failure")
        ),
        "failure_reasons": by_reason,
        "by_data_source": by_data_source,
    }


SUMMARY_NUMERIC_KEYS = [
    "score",
    "accuracy_reward",
    "core_inference_time",
    "core_inference_time_raw",
    "generate_sequences",
    "tool_parsing",
    "tool_calls",
    "conversation_wall_time",
    "prompt_tokens",
    "total_tokens",
    "sequence_tokens",
    "response_tokens_total",
    "response_tokens_generated",
    "response_tokens_tool",
    "n_tool_calls",
    "n_valid_tool_calls",
]


def _numeric_sample_value(sample: dict[str, Any], key: str) -> float | None:
    score = sample.get("score") or {}
    value = score.get(key) if isinstance(score, dict) and key in score else sample.get(key)
    if value is None and key == "score" and isinstance(score, dict):
        value = score.get("score")
    if value is None and key in {"total_tokens", "sequence_tokens"}:
        prompt_tokens = _numeric_sample_value(sample, "prompt_tokens")
        response_tokens = _numeric_sample_value(sample, "response_tokens_total")
        if prompt_tokens is not None and response_tokens is not None:
            value = prompt_tokens + response_tokens
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _summarize_sample_group(samples: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "n": len(samples),
        "num_scored": sum(1 for sample in samples if sample_has_score(sample)),
        "num_critical_failure": sum(1 for sample in samples if sample.get("critical_failure")),
    }
    row["num_unscored_noncritical"] = row["n"] - row["num_scored"] - row["num_critical_failure"]
    row["valid_score_ratio"] = (row["num_scored"] / row["n"]) if row["n"] else None
    for key in SUMMARY_NUMERIC_KEYS:
        values = [value for sample in samples if (value := _numeric_sample_value(sample, key)) is not None]
        mean = _mean_or_none(values)
        if mean is not None:
            row[f"{key}_mean"] = mean
    return row


def build_eval_summary_tables(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_trial: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sample in samples:
        data_source = str(sample.get("data_source") or "unknown")
        trial_idx = int(sample.get("trial_idx", 0) or 0)
        by_trial.setdefault((data_source, trial_idx), []).append(sample)

    trial_rows: list[dict[str, Any]] = []
    for (data_source, trial_idx), group_samples in sorted(by_trial.items()):
        row = {
            "data_source": data_source,
            "trial_idx": trial_idx,
            **_summarize_sample_group(group_samples),
        }
        trial_rows.append(row)

    by_data_source: dict[str, list[dict[str, Any]]] = {}
    for row in trial_rows:
        by_data_source.setdefault(str(row["data_source"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    base_keys = ["n", "num_scored", "num_critical_failure", "num_unscored_noncritical", "valid_score_ratio"]
    metric_keys = [f"{key}_mean" for key in SUMMARY_NUMERIC_KEYS]
    for data_source, rows in sorted(by_data_source.items()):
        summary: dict[str, Any] = {
            "data_source": data_source,
            "num_trials": len(rows),
        }
        for key in [*base_keys, *metric_keys]:
            values = []
            for row in rows:
                value = row.get(key)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isnan(value):
                    values.append(value)
            mean = _mean_or_none(values)
            if mean is not None:
                summary[f"{key}_trial_mean"] = mean
                summary[f"{key}_trial_std"] = _std_or_none(values)
        summary_rows.append(summary)
    return trial_rows, summary_rows


def build_failure_reason_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for sample in samples:
        if sample_has_score(sample) or not sample.get("critical_failure"):
            continue
        data_source = str(sample.get("data_source") or "unknown")
        reasons = sample.get("failure_reasons") or ["critical_failure"]
        for reason in reasons:
            reason_text = str(reason).split("\n")[0]
            key = (data_source, reason_text)
            counts[key] = counts.get(key, 0) + 1
    return [
        {"data_source": data_source, "failure_reason": reason, "count": count}
        for (data_source, reason), count in sorted(counts.items())
    ]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_eval_summary_outputs(output_dir: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    trial_rows, summary_rows = build_eval_summary_tables(samples)
    failure_rows = build_failure_reason_rows(samples)
    write_tsv(output_dir / "eval_summary_by_trial.tsv", trial_rows)
    write_tsv(output_dir / "eval_summary.tsv", summary_rows)
    write_tsv(output_dir / "eval_failure_summary.tsv", failure_rows)
    (output_dir / "eval_summary_by_trial.json").write_text(
        json.dumps(json_safe(trial_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "eval_summary.json").write_text(
        json.dumps(json_safe(summary_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "eval_failure_summary.json").write_text(
        json.dumps(json_safe(failure_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "by_trial_tsv": str(output_dir / "eval_summary_by_trial.tsv"),
        "summary_tsv": str(output_dir / "eval_summary.tsv"),
        "failure_summary_tsv": str(output_dir / "eval_failure_summary.tsv"),
        "num_trial_rows": len(trial_rows),
        "num_summary_rows": len(summary_rows),
        "num_failure_rows": len(failure_rows),
    }


async def score_pending_samples(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    reward_kwargs: dict[str, Any],
    *,
    output_dir: Path | None = None,
    started_at: float | None = None,
    loop_t0: float | None = None,
    total_scored_before: int = 0,
) -> int:
    samples_to_score = [
        sample for sample in samples if sample_needs_judge_score(sample, rescore_existing=args.rescore_existing)
    ]
    if not samples_to_score:
        return 0
    batch_size = JUDGE_BATCH_SIZE if JUDGE_BATCH_SIZE > 0 else len(samples_to_score)
    with progress_bar(total=len(samples_to_score), desc="Judge scoring") as pbar:
        scored = 0
        for batch_start in range(0, len(samples_to_score), batch_size):
            batch = samples_to_score[batch_start : batch_start + batch_size]
            score_dicts = await asyncio.to_thread(
                compute_score_batch,
                data_sources=[sample["data_source"] for sample in batch],
                solution_strs=[sample["solution_str"] for sample in batch],
                ground_truths=[sample["ground_truth"] for sample in batch],
                extra_infos=[sample["extra_info"] for sample in batch],
                **reward_kwargs,
            )
            for sample, score in zip(batch, score_dicts, strict=True):
                sample["score"] = score
            scored += len(batch)
            pbar.update(len(batch))
            if output_dir is not None and started_at is not None and loop_t0 is not None:
                wall_times = {
                    "judge_wall_time_s": time.perf_counter() - started_at,
                    "last_loop_wall_time_s": time.perf_counter() - loop_t0,
                    "total_scored_this_process": total_scored_before + scored,
                }
                write_judge_outputs(
                    output_dir=output_dir,
                    samples=samples,
                    reward_kwargs=reward_kwargs,
                    args=args,
                    wall_times=wall_times,
                )
                pending_remaining = len(samples_to_score) - scored
                print(
                    f"judge partial write: batch_size={len(batch)} "
                    f"scored_this_process={total_scored_before + scored} "
                    f"pending_remaining={pending_remaining}",
                    flush=True,
                )
    return scored


def write_judge_outputs(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    reward_kwargs: dict[str, Any],
    args: argparse.Namespace,
    wall_times: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_samples_jsonl_atomic(output_dir / "samples.jsonl", samples)
    scored_samples = [sample for sample in samples if sample_has_score(sample)]
    summary = build_summary_metrics(scored_samples) if scored_samples else {}
    summary["judge_progress"] = {
        "num_generated_samples": len(samples),
        "num_scored_samples": len(scored_samples),
        "valid_score_ratio": (len(scored_samples) / len(samples)) if samples else None,
    }
    summary["failure_summary"] = build_failure_summary(samples)
    summary["eval_summary_outputs"] = write_eval_summary_outputs(output_dir, samples)
    summary["wall_times"] = wall_times
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "run_id": uuid.uuid4().hex,
        "rollout_dir": str(args.rollout_dir.resolve()),
        "reward_kwargs": reward_kwargs,
        "mode": "follow_until_rollout_done",
        "poll_interval": POLL_INTERVAL_SECONDS,
        "rescore_existing": args.rescore_existing,
        "wall_times": wall_times,
        "num_generated_samples": len(samples),
        "num_scored_samples": len(scored_samples),
    }
    (output_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


async def run_judge(args: argparse.Namespace) -> None:
    reward_kwargs = build_reward_kwargs(args)
    output_dir = args.rollout_dir / args.scores_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    default_agent_name = load_default_agent_name(args.rollout_dir)
    started_at = time.perf_counter()
    total_scored = 0
    print(
        "standalone judge settings: "
        f"judge_model={args.judge_model} "
        f"fallback_judge_model={args.fallback_judge_model or None} "
        f"workers={args.judge_workers} "
        f"mode={args.insight_qwen_judge_mode} "
        f"task_timeout={reward_kwargs['task_timeout']} "
        f"batch_size={JUDGE_BATCH_SIZE}",
        flush=True,
    )

    while True:
        loop_t0 = time.perf_counter()
        rollout_samples = load_rollout_samples(args.rollout_dir)
        judged_samples = load_judged_samples(output_dir)
        samples = merge_samples(rollout_samples, judged_samples, rescore_existing=args.rescore_existing)
        samples = [normalize_sample_for_judge(sample, default_agent_name=default_agent_name) for sample in samples]
        scored_this_loop = await score_pending_samples(
            samples,
            args,
            reward_kwargs,
            output_dir=output_dir,
            started_at=started_at,
            loop_t0=loop_t0,
            total_scored_before=total_scored,
        )
        total_scored += scored_this_loop
        wall_times = {
            "judge_wall_time_s": time.perf_counter() - started_at,
            "last_loop_wall_time_s": time.perf_counter() - loop_t0,
            "total_scored_this_process": total_scored,
        }
        write_judge_outputs(
            output_dir=output_dir,
            samples=samples,
            reward_kwargs=reward_kwargs,
            args=args,
            wall_times=wall_times,
        )

        rollout_done = (args.rollout_dir / "done").exists()
        all_loaded_samples_scored = all(sample_is_terminal_for_judge(sample) for sample in samples)
        if rollout_done:
            # Rollout can finish while a long judge batch is still running. In
            # that case `samples` is a stale checkpoint snapshot, so reload once
            # before deciding the judge is complete.
            fresh_rollout_samples = load_rollout_samples(args.rollout_dir)
            fresh_judged_samples = load_judged_samples(output_dir)
            fresh_samples = merge_samples(
                fresh_rollout_samples,
                fresh_judged_samples,
                rescore_existing=args.rescore_existing,
            )
            fresh_samples = [
                normalize_sample_for_judge(sample, default_agent_name=default_agent_name) for sample in fresh_samples
            ]
            fresh_all_scored = all(sample_is_terminal_for_judge(sample) for sample in fresh_samples)
            if len(fresh_samples) == len(samples) and fresh_all_scored and all_loaded_samples_scored:
                break
            print(
                f"judge follow: rollout_done=True but fresh reload has "
                f"{sum(1 for sample in fresh_samples if sample_is_terminal_for_judge(sample))}/"
                f"{len(fresh_samples)} terminal; continuing without sleep",
                flush=True,
            )
            continue
        print(
            f"judge follow: terminal={sum(1 for sample in samples if sample_is_terminal_for_judge(sample))}/"
            f"{len(samples)} rollout_done={rollout_done}; sleeping {POLL_INTERVAL_SECONDS}s",
            flush=True,
        )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    (output_dir / "done").touch()
    print(f"standalone judge complete: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score rollout-only standalone eval outputs.")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--scores-subdir", default="scores")
    parser.add_argument("--rescore-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--fallback-judge-model", default=DEFAULT_FALLBACK_JUDGE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument(
        "--insight-qwen-judge-mode",
        choices=["legacy_prompt_v2"],
        default="legacy_prompt_v2",
        help="Release judge mode for insight_qwen_agent.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_judge(parse_args()))


if __name__ == "__main__":
    main()
