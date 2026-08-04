#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"

if [[ "${TELEGRAM_NOTIFY_ON_FINISH:-1}" == "1" && "${TELEGRAM_WRAPPED:-0}" != "1" && -x "$SCRIPT_DIR/run_with_telegram_notification.sh" ]]; then
  export TELEGRAM_WRAPPED=1
  exec "$SCRIPT_DIR/run_with_telegram_notification.sh" \
    --label "${TELEGRAM_NOTIFY_LABEL:-gpt5mini_gemini_broad_standalone_https}" \
    -- "$0" "$@"
fi

cd "$REPO_ROOT"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for HTTPS generation/judging}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set for HTTPS generation/judging}"
: "${API_HTTP_PROXY:?API_HTTP_PROXY must be set for API proxy}"
: "${API_HTTPS_PROXY:?API_HTTPS_PROXY must be set for API proxy}"

PYTHON_BIN="${PYTHON_BIN:-/home/ywxzml3j/ywxzml3juser40/.conda/envs/vllm-latest/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export ENSURE_API_LOGGER="${ENSURE_API_LOGGER:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/workspace/api_models_broad_standalone_https_rescale025_05_${RUN_ID}}"
AGENT_CONFIG="${AGENT_CONFIG:-standalone_eval/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-32}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"
MAX_RERUNS="${MAX_RERUNS:-2}"
RESCALES="${RESCALES:-0.25 0.5}"
MODELS="${MODELS:-gpt-5-mini gemini-3.1-flash-lite}"
IMAGE_FORMAT="${IMAGE_FORMAT:-png}"

VAL_FILES="${VAL_FILES:-[\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/dude200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/longdocurl200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlite200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlongbench200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mpdocvqa200-insight_qwen_agent_no_tool_no_system.parquet\"]}"

mkdir -p "$OUTPUT_ROOT/logs"

rescale_slug() {
  local value="$1"
  printf '%s\n' "${value/./}"
}

model_slug() {
  local value="$1"
  printf '%s\n' "$value" | tr '.-' '__'
}

with_api_proxy() {
  CUDA_VISIBLE_DEVICES="" \
  HTTP_PROXY="$API_HTTP_PROXY" \
  HTTPS_PROXY="$API_HTTPS_PROXY" \
  http_proxy="$API_HTTP_PROXY" \
  https_proxy="$API_HTTPS_PROXY" \
  "$@"
}

write_model_config() {
  local model="$1"
  local path="$2"
  local reasoning_effort="high"
  if [[ "$model" == gemini-* ]]; then
    reasoning_effort="null"
  fi
  cat > "$path" <<EOF
model: $model
backend: https_openai_chat

generation:
  max_tokens_after_initial_prompt: 16384

https_openai_chat:
  base_url: null
  api_key_env: OPENAI_API_KEY
  timeout: default
  max_retries: default
  image_format: $IMAGE_FORMAT
  image_detail: high
  reasoning_effort: $reasoning_effort
EOF
}

count_rollout_failures() {
  local output_dir="$1"
  "$PYTHON_BIN" - "$output_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
path = root / "samples.jsonl"
if not path.exists():
    print("1000000000")
    raise SystemExit
failures = 0
for line in path.open():
    if not line.strip():
        continue
    obj = json.loads(line)
    sample = obj.get("sample", obj)
    if sample.get("critical_failure"):
        failures += 1
print(failures)
PY
}

count_judge_pending() {
  local output_dir="$1"
  "$PYTHON_BIN" - "$output_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rollout_path = root / "samples.jsonl"
scores_path = root / "scores" / "samples.jsonl"
if not rollout_path.exists():
    print("1000000000")
    raise SystemExit
rollout_noncritical = 0
for line in rollout_path.open():
    if not line.strip():
        continue
    obj = json.loads(line)
    sample = obj.get("sample", obj)
    if not sample.get("critical_failure"):
        rollout_noncritical += 1
scored = 0
if scores_path.exists():
    for line in scores_path.open():
        if not line.strip():
            continue
        obj = json.loads(line)
        sample = obj.get("sample", obj)
        if sample.get("score"):
            scored += 1
print(max(0, rollout_noncritical - scored))
PY
}

run_rollout_with_retries() {
  local model="$1"
  local model_config="$2"
  local rescale="$3"
  local output_dir="$4"
  local attempt failures
  for attempt in $(seq 0 "$MAX_RERUNS"); do
    echo "[$(date -u +'%F %T')] rollout attempt=$attempt model=$model rescale=$rescale output=$output_dir"
    if ! with_api_proxy "$PYTHON_BIN" -u standalone_eval/rollout.py \
      --model-config "$model_config" \
      --val-files "$VAL_FILES" \
      --output-dir "$output_dir" \
      --agent-config "$AGENT_CONFIG" \
      --agent-config-override "tools.qwen_tool_list=[]" \
      --agent-config-override "images.initial_rescale=$rescale" \
      --agent-worker-processes 1 \
      --worker-concurrency "$WORKER_CONCURRENCY" \
      > "$output_dir/logs/rollout_attempt${attempt}.log" 2>&1; then
      echo "[$(date -u +'%F %T')] rollout attempt=$attempt exited nonzero; will inspect output and retry if possible"
    fi
    failures="$(count_rollout_failures "$output_dir")"
    echo "[$(date -u +'%F %T')] rollout attempt=$attempt failures=$failures"
    if [[ "$failures" == "0" ]]; then
      return 0
    fi
  done
}

run_judge_with_retries() {
  local model="$1"
  local rescale="$2"
  local output_dir="$3"
  local attempt pending
  for attempt in $(seq 0 "$MAX_RERUNS"); do
    echo "[$(date -u +'%F %T')] judge attempt=$attempt model=$model rescale=$rescale output=$output_dir"
    if ! with_api_proxy "$PYTHON_BIN" -u standalone_eval/judge.py \
      --rollout-dir "$output_dir" \
      --judge-model "$JUDGE_MODEL" \
      --judge-workers "$JUDGE_WORKERS" \
      > "$output_dir/logs/judge_attempt${attempt}.log" 2>&1; then
      echo "[$(date -u +'%F %T')] judge attempt=$attempt exited nonzero; will inspect output and retry if possible"
    fi
    pending="$(count_judge_pending "$output_dir")"
    echo "[$(date -u +'%F %T')] judge attempt=$attempt pending_noncritical_scores=$pending"
    if [[ "$pending" == "0" ]]; then
      return 0
    fi
  done
}

run_one() {
  local model="$1"
  local rescale="$2"
  local mslug slug output_dir model_config
  mslug="$(model_slug "$model")"
  slug="$(rescale_slug "$rescale")"
  output_dir="$OUTPUT_ROOT/${mslug}/rescale${slug}"
  model_config="$OUTPUT_ROOT/${mslug}/model_config.yaml"
  mkdir -p "$output_dir/logs"
  mkdir -p "$(dirname "$model_config")"
  write_model_config "$model" "$model_config"

  {
    echo "started_utc=$(date -u +'%F %T')"
    echo "output_dir=$output_dir"
    echo "model=$model"
    echo "model_config=$model_config"
    echo "openai_base_url=$OPENAI_BASE_URL"
    echo "api_proxy_configured=1"
    echo "worker_concurrency=$WORKER_CONCURRENCY"
    echo "judge_model=$JUDGE_MODEL"
    echo "judge_workers=$JUDGE_WORKERS"
    echo "max_reruns=$MAX_RERUNS"
  } > "$output_dir/logs/run.env"

  run_rollout_with_retries "$model" "$model_config" "$rescale" "$output_dir"
  run_judge_with_retries "$model" "$rescale" "$output_dir"
  echo "[$(date -u +'%F %T')] done model=$model rescale=$rescale output=$output_dir"
}

echo "output_root=$OUTPUT_ROOT"
echo "models=$MODELS"
echo "rescales=$RESCALES"
echo "val_files=$VAL_FILES"
echo "api_logger_dir=~/.dumps/api_requests"

for model in $MODELS; do
  export API_LOGGER_PROJECT_NAME="standalone_${model//[^A-Za-z0-9_]/_}_broad_eval"
  for rescale in $RESCALES; do
    run_one "$model" "$rescale"
  done
done

echo "[$(date -u +'%F %T')] all done output_root=$OUTPUT_ROOT"
