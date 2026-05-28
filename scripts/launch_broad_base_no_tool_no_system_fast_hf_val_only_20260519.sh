#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
cd "$REPO_ROOT"

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF

if [[ -f /home/ywxzml3j/ywxzml3juser40/.bashrc ]]; then
  OPENAI_EXPORTS="$(grep -E '^export OPENAI_(API_KEY|BASE_URL)=' /home/ywxzml3j/ywxzml3juser40/.bashrc || true)"
  if [[ -n "$OPENAI_EXPORTS" ]]; then
    eval "$OPENAI_EXPORTS"
  fi
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"

GPUS="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_NAME="${RUN_NAME:-base_no_tool_no_system_fast_hf_val_only_broad_eval_0502_256k_zoom2_area3500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs}"
WORK_DIR="${WORK_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
VERIFY_TIMEOUT_SECONDS="${VERIFY_TIMEOUT_SECONDS:-1800}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"

VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/dude200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/longdocurl200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlite200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mmlongbench200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/mpdocvqa200-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/eval_data_0502_parquets_no_tool_no_system/o3bench0502-insight_qwen_agent_no_tool_no_system.parquet"]'

mkdir -p "$WORK_DIR"
LAUNCH_LOG="$WORK_DIR/${RUN_NAME}.launch.log"

echo "[$(date '+%F %T')] starting $RUN_NAME on GPUs=$GPUS"
echo "[$(date '+%F %T')] work_dir=$WORK_DIR"

EVAL_CUDA_VISIBLE_DEVICES="$GPUS" \
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct" \
LOAD_FORMAT="auto" \
WORK_DIR="$WORK_DIR" \
EXP_NAME="$RUN_NAME" \
WANDB_NAME="$RUN_NAME" \
VAL_FILES="$VAL_FILES" \
AGENT_LOOP_CONFIG_PATH="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500.yaml" \
QWEN_TOOL_LIST="[]" \
LOGGER="['console']" \
MAX_RESPONSE_LENGTH="15360" \
VAL_BATCH_SIZE="32" \
TOOL_MAX_USER_TURNS="10" \
TOOL_MAX_ASSISTANT_TURNS="11" \
DATA_MAX_PROMPT_LENGTH="262144" \
DATA_VALIDATION_MAX_PROMPT_LENGTH="262144" \
ROLLOUT_MAX_MODEL_LEN="262144" \
VLLM_MAX_MODEL_LEN="262144" \
VLLM_GPU_MEMORY_UTILIZATION="0.9" \
FALLBACK_JUDGE_MODEL="$FALLBACK_JUDGE_MODEL" \
TELEGRAM_NOTIFY_LABEL="$RUN_NAME" \
bash scripts/run_iq_base_eval_default_sampling_insight_rl15360.sh \
  trainer.val_only_hf_model_rollout=true \
  trainer.resume_mode=disable &
pid=$!

waited=0
while kill -0 "$pid" 2>/dev/null && (( waited < VERIFY_TIMEOUT_SECONDS )); do
  if [[ -f "$LAUNCH_LOG" ]]; then
    if grep -q "'val_only_hf_model_rollout': True" "$LAUNCH_LOG"; then
      echo "[$(date '+%F %T')] verified fast path in $LAUNCH_LOG"
      wait "$pid"
      exit $?
    fi
    if grep -q "'val_only_hf_model_rollout': False" "$LAUNCH_LOG"; then
      echo "[$(date '+%F %T')] ERROR: fast path override did not take effect; killing pid=$pid" >&2
      kill "$pid" 2>/dev/null || true
      wait "$pid" || true
      exit 1
    fi
  fi
  sleep 5
  waited=$((waited + 5))
done

if kill -0 "$pid" 2>/dev/null; then
  echo "[$(date '+%F %T')] ERROR: timed out waiting for fast-path config; killing pid=$pid" >&2
  kill "$pid" 2>/dev/null || true
  wait "$pid" || true
fi
exit 1
