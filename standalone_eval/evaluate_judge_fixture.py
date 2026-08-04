#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from standalone_eval.judge import DEFAULT_REWARD_KWARGS
from verl.utils.reward_score.vsearch_batch import compute_score_batch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_reward_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    reward_kwargs = json.loads(json.dumps(DEFAULT_REWARD_KWARGS))
    reward_kwargs.update(
        {
            "judge_model": args.judge_model,
            "fallback_judge_model": None,
            "num_workers": args.judge_workers,
            "task_timeout": args.task_timeout,
            "min_success_rate": args.min_success_rate,
            "max_retries": args.max_retries,
            "retry_interval": args.retry_interval,
            "insight_qwen_judge_mode": args.insight_qwen_judge_mode,
        }
    )
    return reward_kwargs


def sample_question_id(sample: dict[str, Any]) -> str | None:
    extra_info = sample.get("extra_info") or {}
    return extra_info.get("question_id") or sample.get("uid")


def build_fixture_batch(
    *,
    expectations: list[dict[str, Any]],
    rollout_samples: list[dict[str, Any]],
    roles: set[str],
    assertable_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_by_qid = {}
    for sample in rollout_samples:
        qid = sample_question_id(sample)
        if qid and qid not in sample_by_qid:
            sample_by_qid[qid] = sample

    selected_expectations = []
    selected_samples = []
    missing = []
    for exp in expectations:
        if exp["test_role"] not in roles:
            continue
        if assertable_only and not exp.get("assertable"):
            continue
        qid = exp["question_id"]
        sample = sample_by_qid.get(qid)
        if sample is None:
            missing.append(qid)
            continue

        sample_copy = copy.deepcopy(sample)
        extra_info = dict(sample_copy.get("extra_info") or {})
        extra_info["agent_name"] = "insight_qwen_agent"
        # Keep fixture scoring isolated: compute_score_batch appends reward info
        # to exported conversations when this path is present.
        extra_info.pop("conversation_export_json_path", None)
        sample_copy["extra_info"] = extra_info
        selected_expectations.append(exp)
        selected_samples.append(sample_copy)

    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Missing {len(missing)} fixture rows in rollout samples. First missing: {preview}")
    return selected_expectations, selected_samples


def summarize_results(expectations: list[dict[str, Any]], scores: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    rows = []
    by_role = defaultdict(list)
    for exp, score in zip(expectations, scores, strict=True):
        expected = exp.get("expected_accuracy_after_fix")
        actual = float(score.get("accuracy_reward", 0.0))
        matched = bool(expected is None or actual == float(expected))
        row = {
            "data_source": exp["data_source"],
            "sample_index_within_benchmark": exp["sample_index_within_benchmark"],
            "question_id": exp["question_id"],
            "test_role": exp["test_role"],
            "assertable": exp["assertable"],
            "expected_accuracy_after_fix": expected,
            "actual_accuracy_reward": actual,
            "matched_expectation": matched,
            "compute_score_success": score.get("compute_score_success"),
            "extracted_answer_before": exp.get("current_extracted_answer"),
            "extracted_answer_after": score.get("extracted_answer"),
            "ground_truth": exp.get("ground_truth"),
            "question": exp.get("question"),
            "note": exp.get("note"),
        }
        rows.append(row)
        by_role[exp["test_role"]].append(row)

    role_summary = {}
    for role, role_rows in sorted(by_role.items()):
        role_summary[role] = {
            "n": len(role_rows),
            "accuracy_mean": sum(r["actual_accuracy_reward"] for r in role_rows) / len(role_rows),
            "expectation_match_rate": sum(r["matched_expectation"] for r in role_rows) / len(role_rows),
            "num_expectation_mismatches": sum(not r["matched_expectation"] for r in role_rows),
            "num_compute_failures": sum(not r["compute_score_success"] for r in role_rows),
        }

    return {
        "n": len(rows),
        "elapsed_s": elapsed_s,
        "role_counts": dict(Counter(r["test_role"] for r in rows)),
        "role_summary": role_summary,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the judge on audited F2P/P2P fixture rows.")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--rollout-samples", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["F2P", "P2P", "F2F_NEGATIVE"],
        help="Fixture roles to score.",
    )
    parser.add_argument("--assertable-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--insight-qwen-judge-mode",
        choices=["legacy", "raw_final_answer_fallback_v2", "single_call_v1", "single_call_v2"],
        default="legacy",
    )
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--task-timeout", type=int, default=60)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--retry-interval", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expectations = load_jsonl(args.fixture)
    rollout_samples = load_jsonl(args.rollout_samples)
    selected_expectations, selected_samples = build_fixture_batch(
        expectations=expectations,
        rollout_samples=rollout_samples,
        roles=set(args.roles),
        assertable_only=args.assertable_only,
    )
    if not selected_samples:
        raise RuntimeError("No fixture samples selected.")

    reward_kwargs = build_reward_kwargs(args)
    t0 = time.perf_counter()
    scores = compute_score_batch(
        data_sources=[sample["data_source"] for sample in selected_samples],
        solution_strs=[sample["solution_str"] for sample in selected_samples],
        ground_truths=[sample["ground_truth"] for sample in selected_samples],
        extra_infos=[sample["extra_info"] for sample in selected_samples],
        **reward_kwargs,
    )
    elapsed_s = time.perf_counter() - t0

    summary = summarize_results(selected_expectations, scores, elapsed_s)
    summary["judge_model"] = args.judge_model
    summary["insight_qwen_judge_mode"] = args.insight_qwen_judge_mode
    summary["reward_kwargs"] = reward_kwargs
    summary["fixture"] = str(args.fixture)
    summary["rollout_samples"] = str(args.rollout_samples)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["n", "elapsed_s", "role_counts", "role_summary"]}, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
