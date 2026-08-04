#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set}"
: "${API_HTTP_PROXY:?API_HTTP_PROXY must be set}"
: "${API_HTTPS_PROXY:?API_HTTPS_PROXY must be set}"

PYTHON_BIN="${PYTHON_BIN:-/home/ywxzml3j/ywxzml3juser40/.conda/envs/vllm-latest/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export ENSURE_API_LOGGER="${ENSURE_API_LOGGER:-1}"
export API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME:-standalone_gemini_3_1_flash_lite_broad_eval_retry_failed}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/workspace/api_models_broad_standalone_https_rescale025_05_20260610_gemini_png_retry/gemini_3_1_flash_lite/rescale05}"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO_ROOT/workspace/api_models_broad_standalone_https_rescale025_05_20260610_gemini_png_retry/gemini_3_1_flash_lite/model_config.yaml}"
AGENT_CONFIG="${AGENT_CONFIG:-standalone_eval/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-32}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"
MAX_EXTRA_ATTEMPTS="${MAX_EXTRA_ATTEMPTS:-6}"
HTTPS_TIMEOUT="${HTTPS_TIMEOUT:-}"
HTTPS_MAX_RETRIES="${HTTPS_MAX_RETRIES:-}"
RESCALE="${RESCALE:-0.5}"
VAL_FILES="${VAL_FILES:-[\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/dude200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/longdocurl200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlite200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlongbench200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mpdocvqa200-insight_qwen_agent_no_tool_no_system.parquet\"]}"

mkdir -p "$OUTPUT_DIR/logs"

with_api_proxy() {
  CUDA_VISIBLE_DEVICES="" \
  HTTP_PROXY="$API_HTTP_PROXY" \
  HTTPS_PROXY="$API_HTTPS_PROXY" \
  http_proxy="$API_HTTP_PROXY" \
  https_proxy="$API_HTTPS_PROXY" \
  "$@"
}

count_rollout_failures() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "samples.jsonl"
if not path.exists():
    print("1000000000")
    raise SystemExit
failures = 0
for line in path.open():
    if not line.strip():
        continue
    obj = json.loads(line)
    sample = obj.get("sample", obj)
    failures += int(bool(sample.get("critical_failure")))
print(failures)
PY
}

count_scored_rows() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "scores" / "samples.jsonl"
count = 0
if path.exists():
    for line in path.open():
        if line.strip():
            count += 1
print(count)
PY
}

echo "[$(date -u +'%F %T')] retry_failed_rows output=$OUTPUT_DIR"
for attempt in $(seq 1 "$MAX_EXTRA_ATTEMPTS"); do
  before="$(count_rollout_failures)"
  echo "[$(date -u +'%F %T')] extra_rollout_attempt=$attempt failures_before=$before"
  if [[ "$before" == "0" ]]; then
    break
  fi
  rollout_extra_args=()
  if [[ -n "$HTTPS_TIMEOUT" ]]; then
    rollout_extra_args+=(--https-timeout-override "$HTTPS_TIMEOUT")
  fi
  if [[ -n "$HTTPS_MAX_RETRIES" ]]; then
    rollout_extra_args+=(--https-max-retries-override "$HTTPS_MAX_RETRIES")
  fi
  with_api_proxy "$PYTHON_BIN" -u standalone_eval/rollout.py \
    --model-config "$MODEL_CONFIG" \
    --val-files "$VAL_FILES" \
    --output-dir "$OUTPUT_DIR" \
    --agent-config "$AGENT_CONFIG" \
    --agent-config-override "tools.qwen_tool_list=[]" \
    --agent-config-override "images.initial_rescale=$RESCALE" \
    --agent-worker-processes 1 \
    --worker-concurrency "$WORKER_CONCURRENCY" \
    "${rollout_extra_args[@]}" \
    > "$OUTPUT_DIR/logs/rerun_failed_rollout_attempt${attempt}.log" 2>&1 || true
  after="$(count_rollout_failures)"
  echo "[$(date -u +'%F %T')] extra_rollout_attempt=$attempt failures_after=$after"
  if [[ "$after" == "0" ]]; then
    break
  fi
done

remaining="$(count_rollout_failures)"
echo "[$(date -u +'%F %T')] rollout_retries_done remaining_failures=$remaining"

with_api_proxy "$PYTHON_BIN" -u standalone_eval/judge.py \
  --rollout-dir "$OUTPUT_DIR" \
  --judge-model "$JUDGE_MODEL" \
  --judge-workers "$JUDGE_WORKERS" \
  > "$OUTPUT_DIR/logs/rerun_failed_judge.log" 2>&1

echo "[$(date -u +'%F %T')] judge_done scored_rows=$(count_scored_rows) output=$OUTPUT_DIR/scores"
