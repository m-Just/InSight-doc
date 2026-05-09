#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script}"

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/longdocurl_mmlongbench_uncapped_initial_rescale_sweep_base_no_tool_no_system_0507}"
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
export QWEN_TOOL_LIST="${QWEN_TOOL_LIST:-[]}"
export WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-30}"
export WAIT_IDLE_STABLE_SECONDS="${WAIT_IDLE_STABLE_SECONDS:-120}"

VAL_FILES_NO_SYSTEM_JSON='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets_no_tool_no_system/longdocurl200_uncapped-insight_qwen_agent_no_tool_no_system.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets_no_tool_no_system/mmlongbench200_uncapped-insight_qwen_agent_no_tool_no_system.parquet"]'
VAL_FILES_NO_SYSTEM_ANSWER_ONLY_JSON='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets_no_tool_no_system_answer_only/longdocurl200_uncapped-insight_qwen_agent_no_tool_no_system_answer_only.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/uncapped_eval_0430_public_parquets_no_tool_no_system_answer_only/mmlongbench200_uncapped-insight_qwen_agent_no_tool_no_system_answer_only.parquet"]'

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
SUMMARY_TXT="$OUTPUT_ROOT/summary.txt"
printf "lane\tgpus\tvariant\tscale_id\tinitial_rescale\trun_name\tstatus\texit_code\twork_dir\n" > "$STATUS_TSV"

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

wait_for_gpus_idle() {
  local gpus_csv="$1"
  local poll_seconds="$2"
  local stable_seconds="$3"
  IFS=',' read -r -a gpu_ids <<< "$gpus_csv"
  local idle_since=""

  while true; do
    local target_uuids=()
    while IFS=',' read -r idx uuid; do
      idx="${idx//[[:space:]]/}"
      uuid="${uuid//[[:space:]]/}"
      for gpu_id in "${gpu_ids[@]}"; do
        if [[ "$idx" == "$gpu_id" ]]; then
          target_uuids+=("$uuid")
          break
        fi
      done
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)

    local busy=0
    while IFS=',' read -r pid gpu_uuid used_gpu_memory; do
      gpu_uuid="${gpu_uuid//[[:space:]]/}"
      for target_uuid in "${target_uuids[@]}"; do
        if [[ -n "$gpu_uuid" && "$gpu_uuid" == "$target_uuid" ]]; then
          busy=1
          break 2
        fi
      done
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader 2>/dev/null || true)

    local now
    now="$(date +%s)"

    if (( busy == 0 )); then
      if [[ -z "$idle_since" ]]; then
        idle_since="$now"
        echo "[$(date '+%F %T')] gpus=${gpus_csv} first observed with no compute processes; waiting ${stable_seconds}s for stability"
      fi

      local idle_elapsed=$(( now - idle_since ))
      if (( idle_elapsed >= stable_seconds )); then
        echo "[$(date '+%F %T')] gpus=${gpus_csv} have had no compute processes for ${idle_elapsed}s; proceeding"
        return 0
      fi

      echo "[$(date '+%F %T')] gpus=${gpus_csv} still idle for ${idle_elapsed}s / ${stable_seconds}s"
    else
      if [[ -n "$idle_since" ]]; then
        echo "[$(date '+%F %T')] gpus=${gpus_csv} became busy again; resetting idle stability timer"
      else
        echo "[$(date '+%F %T')] waiting for gpus=${gpus_csv} to have no compute processes"
      fi
      idle_since=""
    fi
    sleep "$poll_seconds"
  done
}

run_job() {
  local lane_name="$1"
  local gpus="$2"
  local variant="$3"
  local scale_id="$4"

  local scale_value="${SCALE_VALUES[$scale_id]}"
  local agent_cfg="${SCALE_CONFIGS[$scale_id]}"
  local val_files_json
  local run_name
  if [[ "$variant" == "no_system" ]]; then
    val_files_json="$VAL_FILES_NO_SYSTEM_JSON"
    run_name="base_no_tool_no_system_longdocurl200_uncapped_mmlongbench200_uncapped_rescale${scale_id}"
  elif [[ "$variant" == "no_system_answer_only" ]]; then
    val_files_json="$VAL_FILES_NO_SYSTEM_ANSWER_ONLY_JSON"
    run_name="base_no_tool_no_system_answer_only_longdocurl200_uncapped_mmlongbench200_uncapped_rescale${scale_id}"
  else
    echo "Unsupported variant=$variant" >&2
    return 2
  fi

  local work_dir="${OUTPUT_ROOT}/${run_name}"
  local launch_script="$REPO_ROOT/scripts/run_iq_base_eval_default_sampling_insight_rl15360.sh"
  local exit_code=0
  local status="success"

  echo "[$(date '+%F %T')] lane=${lane_name} gpus=${gpus} starting ${run_name} variant=${variant} initial_rescale=${scale_value}"

  set +e
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
  MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct" \
  WORK_DIR="$work_dir" \
  EXP_NAME="$run_name" \
  WANDB_NAME="$run_name" \
  VAL_FILES="$val_files_json" \
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
  QWEN_TOOL_LIST="$QWEN_TOOL_LIST" \
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
    "$lane_name" "$gpus" "$variant" "$scale_id" "$scale_value" "$run_name" "$status" "$exit_code" "$work_dir" \
    >> "$STATUS_TSV"
}

run_lane() {
  local lane_name="$1"
  local gpus="$2"
  local variant="$3"
  wait_for_gpus_idle "$gpus" "$WAIT_POLL_SECONDS" "$WAIT_IDLE_STABLE_SECONDS"
  for scale_id in "${scale_ids[@]}"; do
    run_job "$lane_name" "$gpus" "$variant" "$scale_id"
  done
}

(
  run_lane lane0 "0,1,2,3" "no_system"
) > >(tee -a "$OUTPUT_ROOT/lane0.queue.log") 2>&1 &
lane0_pid=$!

(
  run_lane lane1 "4,5,6,7" "no_system_answer_only"
) > >(tee -a "$OUTPUT_ROOT/lane1.queue.log") 2>&1 &
lane1_pid=$!

echo "Started queue lanes:"
echo "  lane0 pid=${lane0_pid} gpus=0,1,2,3 variant=no_system"
echo "  lane1 pid=${lane1_pid} gpus=4,5,6,7 variant=no_system_answer_only"
echo "  output_root=${OUTPUT_ROOT}"
echo "  benchmarks=longdocurl200_uncapped,mmlongbench200_uncapped"
echo "  model=base_no_tool_no_system"
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
