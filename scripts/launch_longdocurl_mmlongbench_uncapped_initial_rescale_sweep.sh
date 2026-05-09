#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script}"

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_0506}"
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

VAL_FILES_JSON='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets/longdocurl200_uncapped-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets/mmlongbench200_uncapped-insight_qwen_agent.parquet"]'
mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
SUMMARY_TXT="$OUTPUT_ROOT/summary.txt"
printf "lane\tgpus\tmodel_id\tscale_id\tinitial_rescale\trun_name\tstatus\texit_code\twork_dir\n" > "$STATUS_TSV"

declare -A MODEL_PATHS=(
  [base]="Qwen/Qwen3-VL-8B-Instruct"
  [freeze_vt_bs32_tool_arg_order_medium_only_epoch]="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt_tool_arg_order_medium_only/sft_checkpoints/global_step_792/huggingface"
)

declare -A MODEL_KIND=(
  [base]="base"
  [freeze_vt_bs32_tool_arg_order_medium_only_epoch]="ft"
)

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

model_ids=(
  base
  freeze_vt_bs32_tool_arg_order_medium_only_epoch
)
scale_ids=(0175 025 035 05 07)

jobs=()
for model_id in "${model_ids[@]}"; do
  for scale_id in "${scale_ids[@]}"; do
    jobs+=("${model_id}|${scale_id}")
  done
done

lane0_jobs=()
lane1_jobs=()
for idx in "${!jobs[@]}"; do
  if (( idx % 2 == 0 )); then
    lane0_jobs+=("${jobs[$idx]}")
  else
    lane1_jobs+=("${jobs[$idx]}")
  fi
done

run_job() {
  local lane_name="$1"
  local gpus="$2"
  local job="$3"

  IFS='|' read -r model_id scale_id <<< "$job"
  local model_path="${MODEL_PATHS[$model_id]}"
  local model_kind="${MODEL_KIND[$model_id]}"
  local scale_value="${SCALE_VALUES[$scale_id]}"
  local agent_cfg="${SCALE_CONFIGS[$scale_id]}"
  local run_name="${model_id}_longdocurl200_uncapped_mmlongbench200_uncapped_rescale${scale_id}"
  local work_dir="${OUTPUT_ROOT}/${run_name}"
  local launch_script
  local exit_code=0
  local status="success"

  if [[ "$model_kind" == "base" ]]; then
    launch_script="$REPO_ROOT/scripts/run_iq_base_eval_default_sampling_insight_rl15360.sh"
  else
    launch_script="$REPO_ROOT/scripts/run_iq_ft_eval_default_sampling_rl15360.sh"
  fi

  echo "[$(date '+%F %T')] lane=${lane_name} gpus=${gpus} starting ${run_name} model=${model_path} initial_rescale=${scale_value}"

  set +e
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
  MODEL_PATH="$model_path" \
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
    echo "[$(date '+%F %T')] lane=${lane_name} gpus=${gpus} failed ${run_name} exit_code=${exit_code}"
  else
    echo "[$(date '+%F %T')] lane=${lane_name} gpus=${gpus} finished ${run_name}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$lane_name" "$gpus" "$model_id" "$scale_id" "$scale_value" "$run_name" "$status" "$exit_code" "$work_dir" \
    >> "$STATUS_TSV"
}

run_lane() {
  local lane_name="$1"
  local gpus="$2"
  shift 2
  local lane_jobs=("$@")

  for job in "${lane_jobs[@]}"; do
    run_job "$lane_name" "$gpus" "$job"
  done
}

(
  run_lane lane0 "0,1,2,3" "${lane0_jobs[@]}"
) > >(tee -a "$OUTPUT_ROOT/lane0.queue.log") 2>&1 &
lane0_pid=$!

(
  run_lane lane1 "4,5,6,7" "${lane1_jobs[@]}"
) > >(tee -a "$OUTPUT_ROOT/lane1.queue.log") 2>&1 &
lane1_pid=$!

echo "Started queue lanes:"
echo "  lane0 pid=${lane0_pid} gpus=0,1,2,3"
echo "  lane1 pid=${lane1_pid} gpus=4,5,6,7"
echo "  output_root=${OUTPUT_ROOT}"
echo "  benchmarks=longdocurl200_uncapped,mmlongbench200_uncapped"
echo "  models=base,freeze_vt_bs32_tool_arg_order_medium_only_epoch"
echo "  initial_rescales=0.175,0.25,0.35,0.5,0.7"

wait "$lane0_pid"
wait "$lane1_pid"

success_count="$(awk -F '\t' 'NR > 1 && $7 == "success" {count++} END {print count + 0}' "$STATUS_TSV")"
failure_count="$(awk -F '\t' 'NR > 1 && $7 == "failed" {count++} END {print count + 0}' "$STATUS_TSV")"
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
