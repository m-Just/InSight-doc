#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
cd "$REPO_ROOT"

if [[ -f /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-vllm-latest}"
fi
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"

GPUS="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_base_no_tool_no_system_current_lora_setting_20260515}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"

VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets_no_tool_no_system/longdocurl_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets_no_tool_no_system/mmlongbench_highpage_0507-insight_qwen_agent_no_tool_no_system.parquet"]'

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
if [[ "${RESET_STATUS:-0}" == "1" || ! -s "$STATUS_TSV" ]]; then
  printf "model\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"
fi

log() {
  echo "[$(date '+%F %T')] $*" >&2
}

record_status() {
  local model="$1"
  local scale_id="$2"
  local scale_value="$3"
  local status="$4"
  local exit_code="$5"
  local path="$6"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$model" "$scale_id" "$scale_value" "$status" "$exit_code" "$path" >> "$STATUS_TSV"
}

run_eval_one() {
  local scale_id="$1"
  local scale_value="$2"
  local agent_cfg="$3"
  local model_label="base_no_tool_no_system_current_lora_setting"
  local run_name="${model_label}_highpage_0507_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local status="success"
  local exit_code=0

  log "starting ${run_name} initial_rescale=${scale_value}"

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="$GPUS" \
  MODEL_PATH="$MODEL_PATH" \
  LOAD_FORMAT="${LOAD_FORMAT:-auto}" \
  WORK_DIR="$work_dir" \
  EXP_NAME="$run_name" \
  WANDB_NAME="$run_name" \
  VAL_FILES="$VAL_FILES" \
  AGENT_LOOP_CONFIG_PATH="$agent_cfg" \
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
  TELEGRAM_NOTIFY_LABEL="$run_name" \
  bash scripts/run_iq_base_eval_default_sampling_insight_rl15360.sh \
    trainer.val_only_hf_model_rollout=true \
    trainer.resume_mode=disable
  exit_code=$?
  set -e

  if (( exit_code != 0 )); then
    status="failed"
  fi
  record_status "$model_label" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir"
  log "finished ${run_name} status=${status} exit_code=${exit_code}"
}

log "output_root=${OUTPUT_ROOT}"
log "model_path=${MODEL_PATH}"
log "GPUS=${GPUS}"
log "val_files=${VAL_FILES}"
log "qwen_tool_list=[]"

run_eval_one 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml
run_eval_one 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
run_eval_one 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml

log "all base no-tool no-system highpage eval jobs finished"
