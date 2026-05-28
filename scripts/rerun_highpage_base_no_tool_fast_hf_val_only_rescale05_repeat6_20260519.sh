#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
cd "$REPO_ROOT"

# Keep evals direct. Proxy env has caused judge/network failures before.
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY
unset https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF

if [[ -f /home/ywxzml3j/ywxzml3juser40/.bashrc ]]; then
  OPENAI_EXPORTS="$(grep -E '^export OPENAI_(API_KEY|BASE_URL)=' /home/ywxzml3j/ywxzml3juser40/.bashrc || true)"
  if [[ -n "$OPENAI_EXPORTS" ]]; then
    eval "$OPENAI_EXPORTS"
  fi
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_fast_hf_val_only_rescale05_07_repeats_20260519}"
RUN_NAME="${RUN_NAME:-base_no_tool_no_system_fast_hf_val_only_repeat6_highpage_0507_rescale05}"
GPUS="${GPUS:-4,5,6,7}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"
VERIFY_TIMEOUT_SECONDS="${VERIFY_TIMEOUT_SECONDS:-1800}"

VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets_no_tool_no_system/longdocurl_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets_no_tool_no_system/mmlongbench_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet"]'

WORK_DIR="$OUTPUT_ROOT/$RUN_NAME"
LAUNCH_LOG="$WORK_DIR/${RUN_NAME}.launch.log"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
mkdir -p "$WORK_DIR"

verify_fast_path_or_kill() {
  local launch_log="$1"
  local pid="$2"
  local waited=0

  while kill -0 "$pid" 2>/dev/null && (( waited < VERIFY_TIMEOUT_SECONDS )); do
    if [[ -f "$launch_log" ]]; then
      if grep -q "'val_only_hf_model_rollout': True" "$launch_log"; then
        echo "[$(date '+%F %T')] verified fast path in ${launch_log}"
        return 0
      fi
      if grep -q "'val_only_hf_model_rollout': False" "$launch_log"; then
        echo "[$(date '+%F %T')] ERROR: fast path override did not take effect; killing pid=${pid}"
        kill "$pid" 2>/dev/null || true
        return 1
      fi
    fi
    sleep 5
    waited=$((waited + 5))
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: timed out waiting for effective fast-path config; killing pid=${pid}"
    kill "$pid" 2>/dev/null || true
  fi
  return 1
}

echo "[$(date '+%F %T')] starting ${RUN_NAME} gpus=${GPUS}"

set +e
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u NO_PROXY \
    -u https_proxy -u http_proxy -u all_proxy -u no_proxy \
    -u PYTORCH_CUDA_ALLOC_CONF \
EVAL_CUDA_VISIBLE_DEVICES="$GPUS" \
MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct" \
LOAD_FORMAT="auto" \
WORK_DIR="$WORK_DIR" \
EXP_NAME="$RUN_NAME" \
WANDB_NAME="$RUN_NAME" \
VAL_FILES="$VAL_FILES" \
AGENT_LOOP_CONFIG_PATH="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml" \
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

verify_fast_path_or_kill "$LAUNCH_LOG" "$pid"
verify_status=$?
wait "$pid"
exit_code=$?
set -e

status="success"
if (( verify_status != 0 || exit_code != 0 )); then
  status="failed"
fi

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "manual" "6" "base_no_tool_no_system_fast_hf_val_only" "05" "0.5" "$status" "$exit_code" "$WORK_DIR" >> "$STATUS_TSV"

echo "[$(date '+%F %T')] finished ${RUN_NAME} status=${status} exit_code=${exit_code}"
exit "$exit_code"
