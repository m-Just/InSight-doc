#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]] && [[ -f /home/ywxzml3j/ywxzml3juser40/.bashrc ]]; then
  OPENAI_EXPORTS="$(grep -E '^export OPENAI_(API_KEY|BASE_URL)=' /home/ywxzml3j/ywxzml3juser40/.bashrc || true)"
  if [[ -n "$OPENAI_EXPORTS" ]]; then
    eval "$OPENAI_EXPORTS"
  fi
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set}"

MODEL_PATH="${MODEL_PATH:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt/sft_checkpoints/global_step_1052/huggingface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/freeze_vt_bs32_epoch2_broad_fast_rescale035_05_repeats_20260525}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"
TELEGRAM_NOTIFY_ON_FINISH="${TELEGRAM_NOTIFY_ON_FINISH:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"

TRIAL_A_GPUS="${TRIAL_A_GPUS:-0,1,2,3}"
TRIAL_B_GPUS="${TRIAL_B_GPUS:-4,5,6,7}"
TRIAL_A_WAIT_LOG="${TRIAL_A_WAIT_LOG:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/rl_ckpt700_broad_highpage_fast_rescale025_035_05_20260525/broad_gpu0123/queue.log}"
TRIAL_B_WAIT_LOG="${TRIAL_B_WAIT_LOG:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/rl_ckpt700_broad_highpage_fast_rescale025_035_05_20260525/highpage_gpu4567/queue.log}"

BROAD_VAL_FILES='["/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet"]'

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
MASTER_LOG="$OUTPUT_ROOT/queue.log"
if [[ ! -s "$STATUS_TSV" ]]; then
  printf "time\ttrial\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"
fi

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

agent_config_for_scale() {
  case "$1" in
    035) echo "recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml" ;;
    05) echo "recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml" ;;
    *) echo "unknown scale_id: $1" >&2; return 2 ;;
  esac
}

scale_value_for_id() {
  case "$1" in
    035) echo "0.35" ;;
    05) echo "0.5" ;;
    *) echo "unknown scale_id: $1" >&2; return 2 ;;
  esac
}

wait_for_queue_finished() {
  local trial="$1"
  local wait_log="$2"
  log "${trial}: waiting for dependency queue: $wait_log"
  while true; do
    if [[ -f "$wait_log" ]] && grep -q "queue finished status=" "$wait_log"; then
      log "${trial}: dependency queue finished"
      return 0
    fi
    log "${trial}: dependency still running; polling again in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
}

run_eval_one() {
  local trial="$1"
  local gpus="$2"
  local scale_id="$3"
  local scale_value
  scale_value="$(scale_value_for_id "$scale_id")"
  local agent_cfg
  agent_cfg="$(agent_config_for_scale "$scale_id")"
  local model_label="freeze_vt_bs32_epoch2_${trial}"
  local run_name="${model_label}_broad_fast_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local ray_tmp="/tmp/fvt32_${trial}_${scale_id}_20260525"
  local launch_log="$work_dir/${run_name}.launch.log"
  local status="success"
  local exit_code=0

  mkdir -p "$work_dir" "$ray_tmp"
  log "${trial}: starting ${run_name} initial_rescale=${scale_value} GPUs=${gpus}"

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
  MODEL_PATH="$MODEL_PATH" \
  LOAD_FORMAT="safetensors" \
  WORK_DIR="$work_dir" \
  EXP_NAME="$run_name" \
  WANDB_NAME="$run_name" \
  VAL_FILES="$BROAD_VAL_FILES" \
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
  RAY_TMPDIR="$ray_tmp" \
  FALLBACK_JUDGE_MODEL="$FALLBACK_JUDGE_MODEL" \
  TELEGRAM_NOTIFY_ON_FINISH="$TELEGRAM_NOTIFY_ON_FINISH" \
  TELEGRAM_NOTIFY_LABEL="$run_name" \
  bash scripts/run_iq_base_eval_default_sampling_insight_rl15360.sh \
    trainer.val_only_hf_model_rollout=true \
    trainer.resume_mode=disable
  exit_code=$?
  set -e

  if [[ -f "$launch_log" ]] && grep -q "'val_only_hf_model_rollout': True\\|val_only_hf_model_rollout: True" "$launch_log"; then
    log "${trial}: verified fast path for ${run_name}"
  else
    log "${trial}: WARNING fast-path flag not found for ${run_name}"
  fi

  if (( exit_code != 0 )); then
    status="failed"
  fi
  printf "%s\t%s\tfreeze_vt_bs32_epoch2\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$trial" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir" >> "$STATUS_TSV"
  log "${trial}: finished ${run_name} status=${status} exit_code=${exit_code}"
  return "$exit_code"
}

run_trial() {
  local trial="$1"
  local gpus="$2"
  local wait_log="$3"
  local status=0
  wait_for_queue_finished "$trial" "$wait_log"
  for scale_id in 035 05; do
    run_eval_one "$trial" "$gpus" "$scale_id" || status=1
  done
  log "${trial}: lane finished status=${status}"
  return "$status"
}

[[ -d "$MODEL_PATH" ]] || { echo "missing model path: $MODEL_PATH" >&2; exit 2; }
for cfg in \
  recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml \
  recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml; do
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 2; }
done

log "queue started output_root=${OUTPUT_ROOT} model=${MODEL_PATH}"

status=0
run_trial "trialA" "$TRIAL_A_GPUS" "$TRIAL_A_WAIT_LOG" &
pid_a=$!
run_trial "trialB" "$TRIAL_B_GPUS" "$TRIAL_B_WAIT_LOG" &
pid_b=$!

wait "$pid_a" || status=1
wait "$pid_b" || status=1

log "queue finished status=${status}"
exit "$status"
