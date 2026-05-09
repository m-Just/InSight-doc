#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script}"

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_rl_ckpt425_0507}"
export LOGGER="${LOGGER:-['console']}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export TOOL_MAX_USER_TURNS="${TOOL_MAX_USER_TURNS:-10}"
export TOOL_MAX_ASSISTANT_TURNS="${TOOL_MAX_ASSISTANT_TURNS:-11}"
export DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-262144}"
export DATA_VALIDATION_MAX_PROMPT_LENGTH="${DATA_VALIDATION_MAX_PROMPT_LENGTH:-262144}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-262144}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
export MODEL_PATH="${MODEL_PATH:-/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams/global_step_425/actor_merged_hf}"
export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

VAL_FILES_JSON='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets/longdocurl200_uncapped-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets/mmlongbench200_uncapped-insight_qwen_agent.parquet"]'
mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
SUMMARY_TXT="$OUTPUT_ROOT/summary.txt"
printf "gpus\tscale_id\tinitial_rescale\trun_name\tstatus\texit_code\twork_dir\n" > "$STATUS_TSV"

declare -A SCALE_VALUES=(
  [0175]="0.175"
  [025]="0.25"
  [035]="0.35"
  [05]="0.5"
  [07]="0.7"
)

declare -A SCALE_CONFIGS=(
  [0175]="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale0175.yaml"
  [025]="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml"
  [035]="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml"
  [05]="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml"
  [07]="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale07.yaml"
)

scale_ids=(0175 025 035 05 07)

run_job() {
  local gpus="$1"
  local scale_id="$2"

  local scale_value="${SCALE_VALUES[$scale_id]}"
  local agent_cfg="${SCALE_CONFIGS[$scale_id]}"
  local run_name="rl_ckpt425_longdocurl200_uncapped_mmlongbench200_uncapped_rescale${scale_id}"
  local work_dir="${OUTPUT_ROOT}/${run_name}"
  local launch_script="$REPO_ROOT/scripts/run_iq_ft_eval_default_sampling_rl15360.sh"
  local exit_code=0
  local status="success"

  echo "[$(date '+%F %T')] gpus=${gpus} starting ${run_name} initial_rescale=${scale_value}"

  set +e
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
  MODEL_PATH="$MODEL_PATH" \
  WORK_DIR="$work_dir" \
  EXP_NAME="$run_name" \
  WANDB_NAME="$run_name" \
  VAL_FILES="$VAL_FILES_JSON" \
  AGENT_LOOP_CONFIG_PATH="$agent_cfg" \
  LOGGER="$LOGGER" \
  MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
  VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
  TOOL_MAX_USER_TURNS="$TOOL_MAX_USER_TURNS" \
  TOOL_MAX_ASSISTANT_TURNS="$TOOL_MAX_ASSISTANT_TURNS" \
  DATA_MAX_PROMPT_LENGTH="$DATA_MAX_PROMPT_LENGTH" \
  DATA_VALIDATION_MAX_PROMPT_LENGTH="$DATA_VALIDATION_MAX_PROMPT_LENGTH" \
  ROLLOUT_MAX_MODEL_LEN="$ROLLOUT_MAX_MODEL_LEN" \
  VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
  VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
  bash "$launch_script"
  exit_code=$?
  set -e

  if (( exit_code != 0 )); then
    status="failed"
    echo "[$(date '+%F %T')] gpus=${gpus} failed ${run_name} exit_code=${exit_code}"
  else
    echo "[$(date '+%F %T')] gpus=${gpus} finished ${run_name}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$gpus" "$scale_id" "$scale_value" "$run_name" "$status" "$exit_code" "$work_dir" \
    >> "$STATUS_TSV"
}

echo "Started queue:"
echo "  gpus=${EVAL_CUDA_VISIBLE_DEVICES}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  benchmarks=longdocurl200_uncapped,mmlongbench200_uncapped"
echo "  model_path=${MODEL_PATH}"
echo "  initial_rescales=0.175,0.25,0.35,0.5,0.7"

for scale_id in "${scale_ids[@]}"; do
  run_job "$EVAL_CUDA_VISIBLE_DEVICES" "$scale_id"
done

success_count="$(awk -F '\t' 'NR > 1 && $5 == "success" {count++} END {print count + 0}' "$STATUS_TSV")"
failure_count="$(awk -F '\t' 'NR > 1 && $5 == "failed" {count++} END {print count + 0}' "$STATUS_TSV")"
total_count="$(awk 'END {print NR - 1}' "$STATUS_TSV")"

{
  echo "output_root=${OUTPUT_ROOT}"
  echo "total_jobs=${total_count}"
  echo "successes=${success_count}"
  echo "failures=${failure_count}"
  echo "status_tsv=${STATUS_TSV}"
} > "$SUMMARY_TXT"

cat "$SUMMARY_TXT"

if (( failure_count > 0 )); then
  exit 1
fi
