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
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/broad_tool_models_rescale035_05_20260519}"
IDLE_SECONDS="${IDLE_SECONDS:-300}"
POLL_SECONDS="${POLL_SECONDS:-30}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"

VAL_FILES='["/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet"]'

RL_CKPT425_MODEL="/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams/global_step_425__actor_merged_hf"
FREEZE_VT_MEDIUM_MODEL="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt_tool_arg_order_medium_only/sft_checkpoints/global_step_792/huggingface"

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
if [[ ! -s "$STATUS_TSV" ]]; then
  printf "run_name\tmodel_label\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"
fi

log() {
  echo "[$(date '+%F %T')] $*"
}

gpu_has_processes() {
  local wanted_csv=",$GPUS,"
  local idx pid _rest

  while read -r idx pid _rest; do
    [[ -z "${idx:-}" || "$idx" == "#" || "$idx" == "Idx" ]] && continue
    [[ "$pid" == "-" || -z "${pid:-}" ]] && continue
    if [[ "$wanted_csv" == *",$idx,"* ]]; then
      return 0
    fi
  done < <(nvidia-smi pmon -c 1 2>/dev/null || true)

  return 1
}

wait_for_gpu_idle() {
  local idle_for=0
  log "waiting for GPUs $GPUS to have no compute processes for ${IDLE_SECONDS}s"
  while (( idle_for < IDLE_SECONDS )); do
    if gpu_has_processes; then
      idle_for=0
      log "GPUs $GPUS still busy; polling again in ${POLL_SECONDS}s"
    else
      idle_for=$((idle_for + POLL_SECONDS))
      log "GPUs $GPUS idle for ${idle_for}/${IDLE_SECONDS}s"
    fi
    sleep "$POLL_SECONDS"
  done
}

agent_config_for_scale() {
  case "$1" in
    035) echo "recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml" ;;
    05) echo "recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml" ;;
    *) echo "unknown scale_id: $1" >&2; return 2 ;;
  esac
}

run_eval_one() {
  local model_label="$1"
  local model_path="$2"
  local scale_id="$3"
  local scale_value="$4"
  local agent_cfg
  agent_cfg="$(agent_config_for_scale "$scale_id")"

  local run_name="${model_label}_broad_eval_0502_256k_zoom2_area3500_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local status="success"
  local exit_code=0

  log "starting $run_name model=$model_path initial_rescale=$scale_value GPUs=$GPUS"
  mkdir -p "$work_dir"

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="$GPUS" \
  MODEL_PATH="$model_path" \
  LOAD_FORMAT="auto" \
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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$run_name" "$model_label" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir" >> "$STATUS_TSV"
  log "finished $run_name status=$status exit_code=$exit_code"
}

if [[ ! -d "$RL_CKPT425_MODEL" ]]; then
  echo "missing RL model path: $RL_CKPT425_MODEL" >&2
  exit 2
fi
if [[ ! -d "$FREEZE_VT_MEDIUM_MODEL" ]]; then
  echo "missing freeze-vt medium model path: $FREEZE_VT_MEDIUM_MODEL" >&2
  exit 2
fi

wait_for_gpu_idle

run_eval_one "rl_ckpt425_actor_merged_hf" "$RL_CKPT425_MODEL" 035 0.35
run_eval_one "rl_ckpt425_actor_merged_hf" "$RL_CKPT425_MODEL" 05 0.5
run_eval_one "freeze_vt_bs32_tool_arg_order_medium_only_epoch" "$FREEZE_VT_MEDIUM_MODEL" 035 0.35
run_eval_one "freeze_vt_bs32_tool_arg_order_medium_only_epoch" "$FREEZE_VT_MEDIUM_MODEL" 05 0.5

log "broad tool model rescale queue finished"
