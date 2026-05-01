#!/usr/bin/env python3
import argparse
import glob
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


def load_events(profile_dir: Path):
    for path in sorted(glob.glob(str(profile_dir / "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q / 100)))
    return values[idx]


def fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def describe(values):
    if not values:
        return "count=0"
    return (
        f"count={len(values)} mean={fmt(mean(values))} p50={fmt(pct(values, 50))} "
        f"p90={fmt(pct(values, 90))} p95={fmt(pct(values, 95))} max={fmt(max(values))}"
    )


def summarize(profile_dir: Path, top: int):
    events = list(load_events(profile_dir))
    by_event = Counter(e.get("event") for e in events)
    print(f"profile_dir: {profile_dir}")
    print(f"updated_at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"events: {sum(by_event.values())}")
    for event, count in sorted(by_event.items()):
        print(f"  {event}: {count}")

    print("\nValidation batches")
    validation_totals = []
    for e in events:
        if e.get("event") == "validation_batch":
            timing = e.get("timing_s", {})
            validation_totals.append(timing.get("total_batch", 0.0))
            print(
                f"  batch={e.get('batch_idx')} raw={e.get('raw_batch_size')} "
                f"reorganized={e.get('reorganized_batch_size')} "
                f"generate={fmt(timing.get('generate'))} reward={fmt(timing.get('reward_compute'))} "
                f"total={fmt(timing.get('total_batch'))}"
            )
    if validation_totals:
        print(f"  total summary: {describe(validation_totals)}")

    print("\nAgent manager ray_get")
    ray_gets = [e.get("timing_s", {}).get("ray_get", 0.0) for e in events if e.get("event") == "agent_manager_generate_sequences"]
    print(f"  {describe(ray_gets)}")

    print("\nVReasoner samples")
    sample_rows = []
    component_totals = defaultdict(float)
    for e in events:
        if e.get("event") != "vreasoner_v2_sample":
            continue
        timing = e.get("timing_s", {})
        total = timing.get("total", 0.0)
        sample_rows.append((total, e))
        for key in ("api_calls", "child_vsearcher", "message_replay_tokenize", "conversation_export", "postprocess_total"):
            component_totals[key] += timing.get(key, 0.0)
    sample_totals = [row[0] for row in sample_rows]
    print(f"  total: {describe(sample_totals)}")
    if component_totals:
        print("  component sums:")
        for key, value in sorted(component_totals.items(), key=lambda item: item[1], reverse=True):
            print(f"    {key}: {fmt(value)}")
    print(f"  slowest {min(top, len(sample_rows))}:")
    for total, e in sorted(sample_rows, reverse=True)[: top]:
        timing = e.get("timing_s", {})
        print(
            f"    total={fmt(total)} api={fmt(timing.get('api_calls'))} "
            f"child={fmt(timing.get('child_vsearcher'))} tools={e.get('n_tool_calls')} "
            f"api_rounds={e.get('api_round_count')} children={e.get('child_vsearcher_count')} "
            f"job={e.get('job_id')} export={e.get('conversation_export_json_path')}"
        )

    print("\nAPI rounds")
    api_times = [e.get("timing_s", {}).get("api_round", 0.0) for e in events if e.get("event") == "vreasoner_v2_api_round"]
    print(f"  {describe(api_times)}")
    api_failures = Counter()
    for e in events:
        if e.get("event") == "vreasoner_v2_api_round":
            for reason in e.get("failure_reasons") or []:
                api_failures[reason] += 1
    if api_failures:
        print("  failure reasons:")
        for reason, count in api_failures.most_common(top):
            print(f"    {reason}: {count}")

    print("\nChild vsearcher")
    child_times = [e.get("timing_s", {}).get("child_vsearcher", 0.0) for e in events if e.get("event") == "vreasoner_v2_child_vsearcher"]
    print(f"  {describe(child_times)}")

    print("\nReward")
    reward_batch_times = [e.get("timing_s", {}).get("total", 0.0) for e in events if e.get("event") == "reward_batch"]
    print(f"  batches: {describe(reward_batch_times)}")
    reward_task_times = [e.get("timing_s", {}).get("task", 0.0) for e in events if e.get("event") == "reward_task"]
    print(f"  tasks: {describe(reward_task_times)}")
    reward_errors = Counter(e.get("error_type") for e in events if e.get("event") == "reward_task" and not e.get("success"))
    if reward_errors:
        print("  task errors:")
        for error_type, count in reward_errors.most_common(top):
            print(f"    {error_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Summarize VSearch validation profiling JSONL files.")
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh summary every N seconds until interrupted.")
    args = parser.parse_args()

    if args.watch and args.watch > 0:
        while True:
            print("\033[2J\033[H", end="")
            summarize(args.profile_dir, args.top)
            time.sleep(args.watch)
    else:
        summarize(args.profile_dir, args.top)


if __name__ == "__main__":
    main()
