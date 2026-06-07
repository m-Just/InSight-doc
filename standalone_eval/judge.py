#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standalone_eval.core.metrics import build_summary_metrics
from standalone_eval.core.resume import iter_resume_record_samples, sample_has_score, write_samples_jsonl_atomic
from standalone_eval.core.utils import json_safe, progress_bar
from verl.utils.reward_score.vsearch_batch import compute_score_batch


POLL_INTERVAL_SECONDS = 30.0
FALLBACK_JUDGE_MODEL = None
JUDGE_TASK_TIMEOUT_SECONDS = 60
JUDGE_MIN_SUCCESS_RATE = 0.99
JUDGE_MAX_RETRIES = 10
JUDGE_RETRY_INTERVAL_SECONDS = 30


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


def build_reward_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    reward_kwargs = json.loads(json.dumps(DEFAULT_REWARD_KWARGS))
    reward_kwargs.update(
        {
            "judge_model": args.judge_model,
            "fallback_judge_model": FALLBACK_JUDGE_MODEL,
            "num_workers": args.judge_workers,
            "task_timeout": JUDGE_TASK_TIMEOUT_SECONDS,
            "min_success_rate": JUDGE_MIN_SUCCESS_RATE,
            "max_retries": JUDGE_MAX_RETRIES,
            "retry_interval": JUDGE_RETRY_INTERVAL_SECONDS,
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


async def score_pending_samples(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    reward_kwargs: dict[str, Any],
) -> int:
    samples_to_score = [sample for sample in samples if args.rescore_existing or not sample_has_score(sample)]
    if not samples_to_score:
        return 0
    with progress_bar(total=len(samples_to_score), desc="Judge scoring") as pbar:
        batches = [samples_to_score]
        scored = 0
        for batch in batches:
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
    output_dir = args.rollout_dir / "scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    total_scored = 0

    while True:
        loop_t0 = time.perf_counter()
        rollout_samples = load_rollout_samples(args.rollout_dir)
        judged_samples = load_judged_samples(output_dir)
        samples = merge_samples(rollout_samples, judged_samples, rescore_existing=args.rescore_existing)
        scored_this_loop = await score_pending_samples(samples, args, reward_kwargs)
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
        all_loaded_samples_scored = all(sample_has_score(sample) for sample in samples)
        if rollout_done and all_loaded_samples_scored:
            break
        print(
            f"judge follow: scored={sum(1 for sample in samples if sample_has_score(sample))}/"
            f"{len(samples)} rollout_done={rollout_done}; sleeping {POLL_INTERVAL_SECONDS}s",
            flush=True,
        )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    (output_dir / "done").touch()
    print(f"standalone judge complete: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score rollout-only standalone eval outputs.")
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--rescore-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--judge-workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_judge(parse_args()))


if __name__ == "__main__":
    main()
