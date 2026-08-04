#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/ywxzml3j/ywxzml3juser40/.conda/envs/vllm-latest/bin/python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-$REPO_ROOT/workspace/standalone_full_eval_goal_tests_${RUN_ID}}"
MAX_RERUNS="${MAX_RERUNS:-2}"

SWEEP_SCRIPT="${SWEEP_SCRIPT:-$REPO_ROOT/scripts/queue_standalone_full_eval_sweep.sh}"
COMMON_ENV=(
  TELEGRAM_NOTIFY_ON_FINISH="${TELEGRAM_NOTIFY_ON_FINISH:-1}"
  STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
)

summary_has_failures() {
  local output_root="$1"
  local summary_prefix="${2:-}"
  "$PYTHON_BIN" - "$output_root" "$summary_prefix" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
summary_path = root / (f"{prefix}_sweep_summary.tsv" if prefix else "sweep_summary.tsv")
failure_path = root / (f"{prefix}_sweep_failure_summary.tsv" if prefix else "sweep_failure_summary.tsv")
if not summary_path.exists() or summary_path.stat().st_size == 0:
    print(f"missing_or_empty_summary={summary_path}")
    raise SystemExit(2)

bad = []
with summary_path.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        def as_float(name, default=0.0):
            value = row.get(name)
            if value in (None, ""):
                return default
            return float(value)
        if as_float("num_critical_failure_trial_mean") > 0:
            bad.append((row.get("model_slug"), row.get("val_slug"), row.get("rescale"), "critical_failure"))
        if as_float("num_unscored_noncritical_trial_mean") > 0:
            bad.append((row.get("model_slug"), row.get("val_slug"), row.get("rescale"), "unscored_noncritical"))
        if as_float("valid_score_ratio_trial_mean", 1.0) < 1.0:
            bad.append((row.get("model_slug"), row.get("val_slug"), row.get("rescale"), "valid_score_ratio"))

failure_rows = 0
if failure_path.exists() and failure_path.stat().st_size > 0:
    with failure_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                count = int(float(row.get("count") or 0))
            except ValueError:
                count = 0
            if count > 0:
                failure_rows += count

if bad or failure_rows:
    print(f"bad_summary_rows={len(bad)} failure_rows={failure_rows}")
    for item in bad[:20]:
        print("bad", *item)
    raise SystemExit(1)
print(f"summary_ok={summary_path}")
raise SystemExit(0)
PY
}

run_stage_with_retries() {
  local label="$1"
  local output_root="$2"
  local summary_prefix="$3"
  shift 3
  local attempt status
  mkdir -p "$output_root/logs"
  for attempt in $(seq 0 "$MAX_RERUNS"); do
    echo "[$(date -u +'%F %T')] stage=$label attempt=$attempt output_root=$output_root"
    status=0
    env "${COMMON_ENV[@]}" OUTPUT_ROOT="$output_root" "$@" "$SWEEP_SCRIPT" \
      > "$output_root/logs/${label}_attempt${attempt}.log" 2>&1 || status=$?
    echo "[$(date -u +'%F %T')] stage=$label attempt=$attempt sweep_exit=$status"
    if summary_has_failures "$output_root" "$summary_prefix"; then
      echo "[$(date -u +'%F %T')] stage=$label complete"
      return 0
    fi
    echo "[$(date -u +'%F %T')] stage=$label still has failures; rerun budget remaining=$((MAX_RERUNS - attempt))"
  done
  echo "[$(date -u +'%F %T')] stage=$label failed after $((MAX_RERUNS + 1)) attempts" >&2
  return 1
}

echo "base_output_root=$BASE_OUTPUT_ROOT"
echo "api/proxy settings are inherited from the caller; this script does not set or modify them"
echo "judge_model=gpt-5-nano"

run_stage_with_retries \
  "max32_legacy" \
  "$BASE_OUTPUT_ROOT/max32_legacy" \
  "" \
  MAX_SAMPLES=32 \
  JUDGE_MODEL=gpt-5-nano \
  INSIGHT_QWEN_JUDGE_MODE=legacy \
  SCORES_SUBDIR=scores \
  SUMMARY_PREFIX=

run_stage_with_retries \
  "max1000_legacy" \
  "$BASE_OUTPUT_ROOT/max1000" \
  "" \
  MAX_SAMPLES=1000 \
  JUDGE_MODEL=gpt-5-nano \
  INSIGHT_QWEN_JUDGE_MODE=legacy \
  SCORES_SUBDIR=scores \
  SUMMARY_PREFIX=

run_stage_with_retries \
  "max1000_single_call_v1" \
  "$BASE_OUTPUT_ROOT/max1000" \
  "single_call_v1" \
  MAX_SAMPLES=1000 \
  JUDGE_MODEL=gpt-5-nano \
  INSIGHT_QWEN_JUDGE_MODE=single_call_v1 \
  SCORES_SUBDIR=scores_single_call_v1 \
  SUMMARY_PREFIX=single_call_v1 \
  JUDGE_RESCORE_EXISTING=1

echo "[$(date -u +'%F %T')] staged eval tests complete: $BASE_OUTPUT_ROOT"
