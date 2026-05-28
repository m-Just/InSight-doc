#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
cd "$REPO_ROOT"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"

MODEL_PATH="${MODEL_PATH:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_medium_data_ablation_40k_4gpu_lanes_wandb_noactoff_allowover_continue_20260513_210801/lora_basic_lr2e-4_cosine_minlr2e-5_len40960_bs32_rank32_alpha64_freeze_vt_medium_only/sft_checkpoints/global_step_792/huggingface_merged_lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_40k_freeze_vt_medium_only_20260514}"
GPUS="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet"]'

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
printf "scale_id\tinitial_rescale\trun_name\tstatus\texit_code\twork_dir\n" > "$STATUS_TSV"

run_one() {
  local scale_id="$1"
  local scale_value="$2"
  local agent_cfg="$3"
  local run_name="lora_basic_40k_freeze_vt_medium_only_highpage_0507_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local status="success"
  local exit_code=0

  echo "[$(date '+%F %T')] starting ${run_name} initial_rescale=${scale_value}"

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="$GPUS" \
  MODEL_PATH="$MODEL_PATH" \
  LOAD_FORMAT="safetensors" \
  WORK_DIR="$work_dir" \
  EXP_NAME="$run_name" \
  WANDB_NAME="$run_name" \
  VAL_FILES="$VAL_FILES" \
  AGENT_LOOP_CONFIG_PATH="$agent_cfg" \
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
  TELEGRAM_NOTIFY_LABEL="$run_name" \
  bash scripts/run_iq_ft_eval_default_sampling_rl15360.sh \
    trainer.val_only_hf_model_rollout=true \
    trainer.resume_mode=disable
  exit_code=$?
  set -e

  if (( exit_code != 0 )); then
    status="failed"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$scale_id" "$scale_value" "$run_name" "$status" "$exit_code" "$work_dir" >> "$STATUS_TSV"
  echo "[$(date '+%F %T')] finished ${run_name} status=${status} exit_code=${exit_code}"
}

run_one 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml
run_one 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml
run_one 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml

echo "[$(date '+%F %T')] all highpage LoRA eval jobs finished"
echo "output_root=${OUTPUT_ROOT}"
