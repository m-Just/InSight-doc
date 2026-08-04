#!/usr/bin/env python3
"""Evaluate Insight-Qwen judge modes on the curated human-labeled judge test set."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from openai import AsyncOpenAI

from verl.utils.reward_score.vsearch_batch import compute_score_single_insight_qwen_agent


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="notes/generated/judge_test_set_200_hard_evalmix_20260717/judge_test_set_200.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="notes/generated/judge_test_set_200_hard_evalmix_20260717/judge_eval_20260717_globalai",
    )
    parser.add_argument("--modes", default="legacy,single_call_v1,single_call_v2")
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-wait", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--api-proxy",
        default=None,
        help="Explicit HTTP(S) proxy URL for judge API calls. If omitted, API_HTTPS_PROXY/API_HTTP_PROXY is used when set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clear_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    required = {"id", "question", "ground_truth", "final_answer", "human_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.to_dict(orient="records")


def parse_options(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def build_solution_str(row: dict[str, Any]) -> str:
    answer = row.get("final_answer")
    if answer is None or (isinstance(answer, float) and pd.isna(answer)):
        answer = ""
    return f"assistant\n{str(answer)}"


def build_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = {
        "question": str(row["question"]),
        "agent_name": "insight_qwen_agent",
        "id": row.get("id"),
    }
    options = parse_options(row.get("mc_options_json"))
    if options:
        extra_info["options"] = options
    return extra_info


def score_to_dict(score: Any) -> dict[str, Any]:
    if is_dataclass(score):
        return asdict(score)
    if hasattr(score, "__dict__"):
        return dict(score.__dict__)
    return {"score": score}


async def score_one(
    *,
    row: dict[str, Any],
    mode: str,
    client: AsyncOpenAI,
    judge_model: str,
    max_retries: int,
    retry_wait: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    reward_kwargs: dict[str, Any] = {
        "judge_client": client,
        "judge_model": judge_model,
        "insight_qwen_judge_mode": mode,
    }
    if mode == "legacy":
        reward_kwargs["insight_qwen_judge_mode"] = "legacy"

    async with semaphore:
        start = time.time()
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                score = await compute_score_single_insight_qwen_agent(
                    data_source=str(row.get("benchmark_or_data_source") or row.get("source_split") or "judge_test"),
                    solution_str=build_solution_str(row),
                    ground_truth=str(row["ground_truth"]),
                    extra_info=build_extra_info(row),
                    **reward_kwargs,
                )
                score_dict = score_to_dict(score)
                judge_score = float(score_dict.get("accuracy_reward", score_dict.get("score", 0.0)) or 0.0)
                return {
                    "id": row["id"],
                    "mode": mode,
                    "human_label": int(row["human_label"]),
                    "judge_score": judge_score,
                    "judge_pred": int(judge_score > 0.0),
                    "latency_s": time.time() - start,
                    "attempts": attempt,
                    "error": None,
                    "score": score_dict,
                    "question": row.get("question"),
                    "ground_truth": row.get("ground_truth"),
                    "final_answer": row.get("final_answer"),
                    "benchmark_or_data_source": row.get("benchmark_or_data_source"),
                    "answerability": row.get("answerability"),
                    "question_type": row.get("question_type"),
                    "test_bucket": row.get("test_bucket"),
                    "source_pool": row.get("source_pool"),
                    "row_origin": row.get("row_origin"),
                    "old_judge_score": row.get("old_judge_score"),
                    "heuristic_flags_json": row.get("heuristic_flags_json"),
                    "failure_mode_json": row.get("failure_mode_json"),
                }
            except Exception as exc:  # noqa: BLE001 - preserve full row-level failure.
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    await asyncio.sleep(retry_wait * attempt)
        return {
            "id": row["id"],
            "mode": mode,
            "human_label": int(row["human_label"]),
            "judge_score": None,
            "judge_pred": None,
            "latency_s": time.time() - start,
            "attempts": max_retries,
            "error": last_error,
            "question": row.get("question"),
            "ground_truth": row.get("ground_truth"),
            "final_answer": row.get("final_answer"),
            "benchmark_or_data_source": row.get("benchmark_or_data_source"),
            "answerability": row.get("answerability"),
            "question_type": row.get("question_type"),
            "test_bucket": row.get("test_bucket"),
            "source_pool": row.get("source_pool"),
            "row_origin": row.get("row_origin"),
            "old_judge_score": row.get("old_judge_score"),
            "heuristic_flags_json": row.get("heuristic_flags_json"),
            "failure_mode_json": row.get("failure_mode_json"),
        }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("judge_pred") is None:
                continue
            out[str(item["id"])] = item
    return out


async def score_mode(
    *,
    rows: list[dict[str, Any]],
    mode: str,
    output_path: Path,
    client: AsyncOpenAI,
    judge_model: str,
    concurrency: int,
    max_retries: int,
    retry_wait: float,
    overwrite: bool,
) -> list[dict[str, Any]]:
    existing = {} if overwrite else load_existing(output_path)
    pending = [row for row in rows if str(row["id"]) not in existing]
    print(f"[{mode}] loaded={len(existing)} pending={len(pending)} output={output_path}", flush=True)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            score_one(
                row=row,
                mode=mode,
                client=client,
                judge_model=judge_model,
                max_retries=max_retries,
                retry_wait=retry_wait,
                semaphore=semaphore,
            )
        )
        for row in pending
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode_results = list(existing.values())
    completed = 0
    with output_path.open("w" if overwrite else "a", encoding="utf-8") as f:
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            mode_results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if completed % 10 == 0 or completed == len(tasks):
                errors = sum(1 for r in mode_results if r.get("error"))
                print(f"[{mode}] completed_new={completed}/{len(tasks)} total={len(mode_results)} errors={errors}", flush=True)

    if not overwrite and existing:
        # Rewrite in input order after resume so downstream diffs are deterministic.
        by_id = {str(item["id"]): item for item in mode_results}
        ordered = [by_id[str(row["id"])] for row in rows if str(row["id"]) in by_id]
        with output_path.open("w", encoding="utf-8") as f:
            for item in ordered:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        mode_results = ordered

    return mode_results


def confusion_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in results if r.get("judge_pred") is not None]
    tp = sum(1 for r in valid if r["human_label"] == 1 and r["judge_pred"] == 1)
    tn = sum(1 for r in valid if r["human_label"] == 0 and r["judge_pred"] == 0)
    fp = sum(1 for r in valid if r["human_label"] == 0 and r["judge_pred"] == 1)
    fn = sum(1 for r in valid if r["human_label"] == 1 and r["judge_pred"] == 0)
    n = len(valid)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "errors": len(results) - n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_latency_s": sum(float(r.get("latency_s") or 0.0) for r in valid) / n if n else 0.0,
    }


def grouped_metrics(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        groups.setdefault(str(result.get(key) or "missing"), []).append(result)
    return {name: confusion_metrics(items) for name, items in sorted(groups.items())}


def write_summary(output_dir: Path, all_results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode, results in all_results.items():
        summary[mode] = {
            "overall": confusion_metrics(results),
            "by_answerability": grouped_metrics(results, "answerability"),
            "by_question_type": grouped_metrics(results, "question_type"),
            "by_test_bucket": grouped_metrics(results, "test_bucket"),
            "by_source_pool": grouped_metrics(results, "source_pool"),
            "by_row_origin": grouped_metrics(results, "row_origin"),
        }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = ["# Judge Test Set Evaluation", ""]
    lines.append("| mode | n | errors | acc | precision | recall | f1 | tp | tn | fp | fn | mean_latency_s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode, metrics in summary.items():
        overall = metrics["overall"]
        lines.append(
            "| {mode} | {n} | {errors} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | "
            "{tp} | {tn} | {fp} | {fn} | {mean_latency_s:.2f} |".format(mode=mode, **overall)
        )
    lines.extend(["", "## Slices", ""])
    for mode, metrics in summary.items():
        lines.append(f"### {mode}")
        for group_name in ("by_answerability", "by_question_type", "by_test_bucket", "by_source_pool", "by_row_origin"):
            lines.append("")
            lines.append(f"**{group_name}**")
            lines.append("")
            lines.append("| slice | n | errors | acc | tp | tn | fp | fn |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for slice_name, item in metrics[group_name].items():
                lines.append(
                    "| {slice_name} | {n} | {errors} | {accuracy:.4f} | {tp} | {tn} | {fp} | {fn} |".format(
                        slice_name=slice_name,
                        **item,
                    )
                )
            lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


async def main() -> None:
    args = parse_args()
    api_proxy = args.api_proxy or os.getenv("API_HTTPS_PROXY") or os.getenv("API_HTTP_PROXY")
    clear_proxy_env()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    rows = load_rows(input_path, args.limit)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not os.getenv("OPENAI_BASE_URL"):
        raise RuntimeError("OPENAI_BASE_URL is not set")

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout),
        limits=httpx.Limits(max_connections=max(args.concurrency, 1), max_keepalive_connections=0),
        proxy=api_proxy,
        trust_env=False,
    )
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        http_client=http_client,
        max_retries=0,
    )

    try:
        all_results: dict[str, list[dict[str, Any]]] = {}
        for mode in modes:
            output_path = output_dir / f"predictions_{mode}.jsonl"
            all_results[mode] = await score_mode(
                rows=rows,
                mode=mode,
                output_path=output_path,
                client=client,
                judge_model=args.judge_model,
                concurrency=args.concurrency,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
                overwrite=args.overwrite,
            )
        summary = write_summary(output_dir, all_results)
        print(json.dumps({mode: item["overall"] for mode, item in summary.items()}, indent=2), flush=True)
        print(f"wrote {output_dir}", flush=True)
    finally:
        await http_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
