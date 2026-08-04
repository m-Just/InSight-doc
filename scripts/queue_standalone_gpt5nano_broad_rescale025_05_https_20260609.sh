#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"

if [[ "${TELEGRAM_NOTIFY_ON_FINISH:-1}" == "1" && "${TELEGRAM_WRAPPED:-0}" != "1" && -x "$SCRIPT_DIR/run_with_telegram_notification.sh" ]]; then
  export TELEGRAM_WRAPPED=1
  exec "$SCRIPT_DIR/run_with_telegram_notification.sh" \
    --label "${TELEGRAM_NOTIFY_LABEL:-gpt5nano_broad_standalone_https}" \
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
export API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME:-standalone_gpt5nano_broad_eval}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/workspace/gpt5nano_broad_standalone_https_rescale025_05_${RUN_ID}}"
MODEL_CONFIG="${MODEL_CONFIG:-standalone_eval/model_configs/gpt_5_nano.yaml}"
AGENT_CONFIG="${AGENT_CONFIG:-standalone_eval/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-32}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"

VAL_FILES="${VAL_FILES:-[\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/dude200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/longdocurl200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlite200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlongbench200-insight_qwen_agent_no_tool_no_system.parquet\",\"$REPO_ROOT/notes/generated/eval_data_0502_parquets_no_tool_no_system/mpdocvqa200-insight_qwen_agent_no_tool_no_system.parquet\"]}"

mkdir -p "$OUTPUT_ROOT/logs"

rescale_slug() {
  local value="$1"
  printf '%s\n' "${value/./}"
}

with_api_proxy() {
  CUDA_VISIBLE_DEVICES="" \
  HTTP_PROXY="$API_HTTP_PROXY" \
  HTTPS_PROXY="$API_HTTPS_PROXY" \
  http_proxy="$API_HTTP_PROXY" \
  https_proxy="$API_HTTPS_PROXY" \
  "$@"
}

run_one() {
  local rescale="$1"
  local slug
  slug="$(rescale_slug "$rescale")"
  local output_dir="$OUTPUT_ROOT/rescale${slug}"
  mkdir -p "$output_dir/logs"

  {
    echo "started_utc=$(date -u +'%F %T')"
    echo "output_dir=$output_dir"
    echo "model_config=$MODEL_CONFIG"
    echo "openai_base_url=$OPENAI_BASE_URL"
    echo "api_proxy_configured=1"
    echo "worker_concurrency=$WORKER_CONCURRENCY"
    echo "judge_model=$JUDGE_MODEL"
    echo "judge_workers=$JUDGE_WORKERS"
  } > "$output_dir/logs/run.env"

  echo "[$(date -u +'%F %T')] start gpt-5-nano broad https rescale=$rescale output=$output_dir"

  with_api_proxy "$PYTHON_BIN" -u standalone_eval/rollout.py \
    --model-config "$MODEL_CONFIG" \
    --val-files "$VAL_FILES" \
    --output-dir "$output_dir" \
    --agent-config "$AGENT_CONFIG" \
    --agent-config-override "tools.qwen_tool_list=[]" \
    --agent-config-override "images.initial_rescale=$rescale" \
    --agent-worker-processes 1 \
    --worker-concurrency "$WORKER_CONCURRENCY" \
    > "$output_dir/logs/rollout.log" 2>&1

  with_api_proxy "$PYTHON_BIN" -u standalone_eval/judge.py \
    --rollout-dir "$output_dir" \
    --judge-model "$JUDGE_MODEL" \
    --judge-workers "$JUDGE_WORKERS" \
    > "$output_dir/logs/judge.log" 2>&1

  echo "[$(date -u +'%F %T')] done gpt-5-nano broad https rescale=$rescale"
}

echo "output_root=$OUTPUT_ROOT"
echo "val_files=$VAL_FILES"
echo "api_logger_dir=~/.dumps/api_requests"
echo "api_logger_project=$API_LOGGER_PROJECT_NAME"

run_one 0.25
run_one 0.5

echo "[$(date -u +'%F %T')] all done output_root=$OUTPUT_ROOT"
