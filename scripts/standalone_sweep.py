#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON list, got: {value}")
        return [str(item) for item in parsed]
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_benchmark_specs(specs: list[str], benchmark_json: str | None) -> dict[str, str]:
    benchmarks: dict[str, str] = {}
    if benchmark_json:
        parsed = json.loads(Path(benchmark_json).read_text(encoding="utf-8") if Path(benchmark_json).exists() else benchmark_json)
        if not isinstance(parsed, dict):
            raise ValueError("--benchmark-json must be a JSON object or a path to one")
        benchmarks.update({str(k): str(v) for k, v in parsed.items()})
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--benchmark expects NAME=PARQUET, got: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"invalid benchmark spec: {spec}")
        benchmarks[name] = path
    if not benchmarks:
        raise ValueError("provide at least one --benchmark NAME=PARQUET or --benchmark-json")
    return benchmarks


def rescale_tag(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("empty rescale value")
    if "e" in text.lower():
        text = f"{float(text):.8f}".rstrip("0").rstrip(".")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text.startswith("0."):
        return "0" + text[2:]
    if text.startswith("."):
        return "0" + text[1:]
    return text.replace(".", "")


def format_rescale_for_name(value: str) -> str:
    return str(value).strip().replace(".", "p")


def resolve_agent_config(template: str, rescale: str) -> str:
    tag = rescale_tag(rescale)
    path = template.format(rescale=rescale, tag=tag, rescale_tag=tag)
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"agent config not found for rescale={rescale}: {resolved}")
    return str(resolved)


def safe_name(value: str) -> str:
    out = []
    for char in str(value):
        if char.isalnum() or char in "._-":
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("._") or "unknown"


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def stat_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stat_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_samples(output_dir: Path) -> list[dict[str, Any]]:
    samples_path = output_dir / "samples.jsonl"
    if not samples_path.exists():
        return []
    samples: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def sample_score(sample: dict[str, Any]) -> dict[str, Any]:
    score = sample.get("score")
    return score if isinstance(score, dict) else {}


def sample_accuracy(sample: dict[str, Any]) -> float | None:
    score = sample_score(sample)
    for key in ("accuracy_reward", "score", "reward"):
        value = to_float(score.get(key))
        if value is not None:
            return value
    return None


def response_tokens(sample: dict[str, Any]) -> float | None:
    total = to_float(sample.get("response_tokens_total"))
    if total is not None:
        return total
    generated = to_float(sample.get("response_tokens_generated"))
    tool = to_float(sample.get("response_tokens_tool"))
    if generated is not None or tool is not None:
        return (generated or 0.0) + (tool or 0.0)
    return None


def seq_len(sample: dict[str, Any]) -> float | None:
    prompt = to_float(sample.get("prompt_tokens"))
    response = response_tokens(sample)
    if prompt is None or response is None:
        return None
    return prompt + response


def failure_reasons(sample: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    raw = sample.get("failure_reasons")
    if isinstance(raw, str) and raw:
        reasons.append(raw)
    elif isinstance(raw, list):
        reasons.extend(str(item) for item in raw if item)
    if sample.get("critical_failure") and not reasons:
        reasons.append("critical_failure")
    score = sample_score(sample)
    if score.get("compute_score_success") is False:
        reasons.append(str(score.get("failure_reason") or score.get("fail_reason") or "score_failed"))
    for key in ("fail_reason", "failure_reason", "error_type"):
        value = score.get(key)
        if value:
            reasons.append(str(value))
    return reasons


def normalize_failure_reason(reason: str) -> str:
    text = str(reason).strip()
    lower = text.lower()
    if not text:
        return "unknown_failure"
    if "timeouterror" in lower or "timeout" in lower:
        return "timeout"
    if "rate limit" in lower or "429" in lower or "too many requests" in lower:
        return "rate_limit"
    if "context" in lower or "max_model_len" in lower or "prompt is too long" in lower:
        return "context_overflow"
    if "empty" in lower and "assistant" in lower:
        return "empty_assistant_message"
    if "score" in lower or "judge" in lower:
        return text.split(":", 1)[0]
    if ":" in text:
        parts = [part.strip() for part in text.split(":") if part.strip()]
        return ":".join(parts[:2])
    return text[:96]


def is_valid_sample(sample: dict[str, Any]) -> bool:
    if failure_reasons(sample):
        return False
    score = sample_score(sample)
    if score.get("compute_score_success") is False:
        return False
    return sample_accuracy(sample) is not None


def expected_counts(samples: list[dict[str, Any]], output_dir: Path) -> tuple[int, int]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            total = int(manifest.get("num_expected_samples") or manifest.get("num_samples") or 0)
            num_trials = int(manifest.get("num_trials") or 1)
            if total > 0 and num_trials > 0:
                return total, num_trials
        except Exception:
            pass
    if not samples:
        return 0, 1
    trials = sorted({int(sample.get("trial_idx", 0)) for sample in samples})
    num_trials = max(trials) + 1 if trials else 1
    per_trial_counts = Counter(int(sample.get("trial_idx", 0)) for sample in samples)
    expected_per_trial = max(per_trial_counts.values()) if per_trial_counts else 0
    return expected_per_trial * num_trials, num_trials


def summarize_cell(
    *,
    benchmark: str,
    rescale: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = read_samples(output_dir)
    expected_total, num_trials = expected_counts(samples, output_dir)
    expected_per_trial = expected_total // num_trials if num_trials else 0
    by_trial: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_trial[int(sample.get("trial_idx", 0))].append(sample)

    per_trial_rows: list[dict[str, Any]] = []
    failure_counter: Counter[str] = Counter()
    total_invalid_or_missing = 0
    for trial_idx in range(num_trials):
        trial_samples = by_trial.get(trial_idx, [])
        seen_indices = {int(sample.get("sample_index", -1)) for sample in trial_samples}
        missing = max(0, expected_per_trial - len(seen_indices))
        if missing:
            failure_counter["missing_sample"] += missing
            total_invalid_or_missing += missing

        valid_samples = [sample for sample in trial_samples if is_valid_sample(sample)]
        invalid_samples = [sample for sample in trial_samples if not is_valid_sample(sample)]
        total_invalid_or_missing += len(invalid_samples)
        for sample in invalid_samples:
            reasons = failure_reasons(sample)
            failure_counter[normalize_failure_reason(reasons[0] if reasons else "invalid_or_unscored")] += 1

        acc_values = [sample_accuracy(sample) for sample in valid_samples]
        acc_values = [value for value in acc_values if value is not None]
        seq_values = [seq_len(sample) for sample in valid_samples]
        seq_values = [value for value in seq_values if value is not None]
        core_values = [to_float(sample.get("core_inference_time")) for sample in valid_samples]
        core_values = [value for value in core_values if value is not None]
        all_accuracy_num = sum(acc_values)
        denom = expected_per_trial or len(trial_samples) or 1
        per_trial_rows.append(
            {
                "benchmark": benchmark,
                "rescale": rescale,
                "trial_idx": trial_idx,
                "n_expected": expected_per_trial,
                "n_exported": len(trial_samples),
                "n_valid": len(valid_samples),
                "valid_ratio": len(valid_samples) / denom,
                "accuracy_valid": stat_mean(acc_values),
                "accuracy_all": all_accuracy_num / denom,
                "seq_len_mean": stat_mean(seq_values),
                "core_time_s_mean": stat_mean(core_values),
                "n_fail_or_missing": denom - len(valid_samples),
            }
        )

    def collect_metric(name: str) -> list[float]:
        values = [to_float(row.get(name)) for row in per_trial_rows]
        return [value for value in values if value is not None]

    failure_breakdown = dict(sorted(failure_counter.items(), key=lambda item: (-item[1], item[0])))
    summary = {
        "benchmark": benchmark,
        "rescale": rescale,
        "output_dir": str(output_dir),
        "num_trials": num_trials,
        "n_expected_per_trial": expected_per_trial,
        "n_expected_total": expected_total,
        "n_exported_total": len(samples),
        "valid_ratio_mean": stat_mean(collect_metric("valid_ratio")),
        "valid_ratio_std": stat_std(collect_metric("valid_ratio")),
        "accuracy_valid_mean": stat_mean(collect_metric("accuracy_valid")),
        "accuracy_valid_std": stat_std(collect_metric("accuracy_valid")),
        "accuracy_all_mean": stat_mean(collect_metric("accuracy_all")),
        "accuracy_all_std": stat_std(collect_metric("accuracy_all")),
        "seq_len_mean": stat_mean(collect_metric("seq_len_mean")),
        "seq_len_std": stat_std(collect_metric("seq_len_mean")),
        "core_time_s_mean": stat_mean(collect_metric("core_time_s_mean")),
        "core_time_s_std": stat_std(collect_metric("core_time_s_mean")),
        "n_fail_or_missing_total": total_invalid_or_missing,
        "failure_breakdown": failure_breakdown,
    }
    return per_trial_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def fmt(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.4f}"


def write_markdown(path: Path, summaries: list[dict[str, Any]], command_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Standalone Sweep Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "| benchmark | rescale | trials | valid ratio | acc valid | acc all | seq len | core time s | failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        failures = "; ".join(f"{key}={value}" for key, value in (row.get("failure_breakdown") or {}).items())
        lines.append(
            "| {benchmark} | {rescale} | {trials} | {vr} +/- {vrs} | {acc} +/- {accs} | {acca} +/- {accas} | {seq} +/- {seqs} | {core} +/- {cores} | {failures} |".format(
                benchmark=row["benchmark"],
                rescale=row["rescale"],
                trials=row["num_trials"],
                vr=fmt(row["valid_ratio_mean"]),
                vrs=fmt(row["valid_ratio_std"]),
                acc=fmt(row["accuracy_valid_mean"]),
                accs=fmt(row["accuracy_valid_std"]),
                acca=fmt(row["accuracy_all_mean"]),
                accas=fmt(row["accuracy_all_std"]),
                seq=fmt(row["seq_len_mean"]),
                seqs=fmt(row["seq_len_std"]),
                core=fmt(row["core_time_s_mean"]),
                cores=fmt(row["core_time_s_std"]),
                failures=failures,
            )
        )
    lines.extend(["", "## Commands", ""])
    for row in command_rows:
        status = row.get("status", "planned")
        lines.append(f"- `{row['benchmark']}` rescale `{row['rescale']}`: `{status}` -> `{row['output_dir']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_eval_command(
    *,
    args: argparse.Namespace,
    benchmark: str,
    parquet: str,
    rescale: str,
    output_dir: Path,
    passthrough_args: list[str],
) -> list[str]:
    agent_config = resolve_agent_config(args.agent_config_template, rescale)
    cmd = [
        args.python,
        str(REPO_ROOT / "standalone_eval" / "rollout.py"),
        "--model-config",
        args.model_config,
        "--val-files",
        json.dumps([parquet]),
        "--output-dir",
        str(output_dir),
        "--agent-config",
        agent_config,
        "--num-trials",
        str(args.num_trials),
        "--agent-worker-processes",
        str(args.agent_worker_processes),
        "--worker-concurrency",
        str(args.worker_concurrency),
    ]
    if args.ray_server_manifest:
        cmd.extend(["--ray-server-manifest", args.ray_server_manifest])
    cmd.extend(passthrough_args)
    return cmd


def run_command(cmd: list[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        return 0
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n===== command =====\n")
        log_f.write(" ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run and summarize standalone rollout.py sweeps over benchmarks, rescale ratios, and trials."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--ray-server-manifest")
    parser.add_argument("--benchmark", action="append", default=[], help="Benchmark spec NAME=PARQUET. Can be repeated.")
    parser.add_argument("--benchmark-json", help="JSON object or path mapping benchmark names to parquet paths.")
    parser.add_argument("--rescale-ratios", required=True, help="Comma-separated or JSON list, e.g. 0.25,0.35,0.5")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--agent-worker-processes", type=int, default=1)
    parser.add_argument("--worker-concurrency", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--agent-config-template",
        default="recipe/vsearch/config/agent_insight_qwen_agent_core_zoom_factor2_area3500_rescale{tag}.yaml",
        help="Format string with {rescale}, {tag}, or {rescale_tag}.",
    )
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args, passthrough_args = parser.parse_known_args()
    if args.num_trials < 1:
        parser.error("--num-trials must be >= 1")
    if args.agent_worker_processes < 1:
        parser.error("--agent-worker-processes must be >= 1")
    if args.worker_concurrency < 1:
        parser.error("--worker-concurrency must be >= 1")
    return args, passthrough_args


def main() -> None:
    args, passthrough_args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    benchmarks = parse_benchmark_specs(args.benchmark, args.benchmark_json)
    rescales = parse_list(args.rescale_ratios)
    command_rows: list[dict[str, Any]] = []

    for rescale in rescales:
        for benchmark, parquet in benchmarks.items():
            cell_name = f"{safe_name(benchmark)}__rescale{rescale_tag(rescale)}"
            output_dir = output_root / cell_name
            log_path = output_root / "logs" / f"{cell_name}.log"
            cmd = build_eval_command(
                args=args,
                benchmark=benchmark,
                parquet=parquet,
                rescale=rescale,
                output_dir=output_dir,
                passthrough_args=passthrough_args,
            )
            row = {
                "benchmark": benchmark,
                "rescale": rescale,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "command": cmd,
                "status": "skipped_summarize_only" if args.summarize_only else "planned",
            }
            command_rows.append(row)
            if args.summarize_only:
                continue
            print(f"running benchmark={benchmark} rescale={rescale} output={output_dir}", flush=True)
            rc = run_command(cmd, log_path, args.dry_run)
            row["returncode"] = rc
            row["status"] = "dry_run" if args.dry_run else ("ok" if rc == 0 else "failed")
            if rc != 0 and not args.continue_on_error:
                write_markdown(output_root / "sweep_summary.md", [], command_rows)
                raise SystemExit(rc)

    per_trial_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for row in command_rows:
        trial_rows, summary = summarize_cell(
            benchmark=str(row["benchmark"]),
            rescale=str(row["rescale"]),
            output_dir=Path(str(row["output_dir"])),
        )
        per_trial_rows.extend(trial_rows)
        summaries.append(summary)

    write_csv(output_root / "sweep_per_trial.csv", per_trial_rows)
    write_csv(output_root / "sweep_summary.csv", summaries)
    (output_root / "sweep_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "sweep_commands.json").write_text(json.dumps(command_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_root / "sweep_summary.md", summaries, command_rows)
    print(f"summary written to {output_root / 'sweep_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
