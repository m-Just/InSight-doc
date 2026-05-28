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

MODEL_ROOT="${MODEL_ROOT:-/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_new_data}"
STEP="${STEP:-$(tr -dc '0-9' < "$MODEL_ROOT/latest_checkpointed_iteration.txt")}"
ACTOR_DIR="${ACTOR_DIR:-$MODEL_ROOT/global_step_${STEP}/actor}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/global_step_${STEP}__actor_merged_hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/new_data_rl_fast_broad_highpage_rescale025_035_05_20260524}"
MODEL_LABEL="${MODEL_LABEL:-rl_new_data_step${STEP}_actor_merged_hf}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}"
TELEGRAM_NOTIFY_ON_FINISH="${TELEGRAM_NOTIFY_ON_FINISH:-1}"
IDLE_SECONDS="${IDLE_SECONDS:-300}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RUN_BROAD="${RUN_BROAD:-1}"
RUN_HIGHPAGE="${RUN_HIGHPAGE:-1}"

BROAD_VAL_FILES='["/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet"]'
HIGHPAGE_VAL_FILES='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet"]'

mkdir -p "$OUTPUT_ROOT"
STATUS_TSV="$OUTPUT_ROOT/status.tsv"
MASTER_LOG="$OUTPUT_ROOT/queue.log"
if [[ ! -s "$STATUS_TSV" ]]; then
  printf "time\tlane\tmodel\tscale_id\tinitial_rescale\tstatus\texit_code\tpath\n" > "$STATUS_TSV"
fi

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

has_hf_weights() {
  [[ -f "$1/model.safetensors.index.json" || -f "$1/model.safetensors" || -n "$(find "$1" -maxdepth 1 -name '*.safetensors' -print -quit 2>/dev/null)" ]]
}

ensure_merged_hf() {
  [[ -d "$ACTOR_DIR" ]] || { echo "missing actor dir: $ACTOR_DIR" >&2; exit 2; }
  if has_hf_weights "$MODEL_PATH"; then
    log "using existing merged HF model: $MODEL_PATH"
    return
  fi

  log "merging FSDP actor checkpoint: $ACTOR_DIR -> $MODEL_PATH"
  source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
  conda activate vllm-latest
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$MODEL_PATH" \
    2>&1 | tee -a "$MASTER_LOG"

  if ! has_hf_weights "$MODEL_PATH"; then
    echo "merge did not produce HF weights: $MODEL_PATH" >&2
    exit 2
  fi
}

agent_config_for_scale() {
  case "$1" in
    025) echo "recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml" ;;
    035) echo "recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml" ;;
    05) echo "recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml" ;;
    *) echo "unknown scale_id: $1" >&2; return 2 ;;
  esac
}

scale_value_for_id() {
  case "$1" in
    025) echo "0.25" ;;
    035) echo "0.35" ;;
    05) echo "0.5" ;;
    *) echo "unknown scale_id: $1" >&2; return 2 ;;
  esac
}

gpu_has_processes() {
  local gpus="$1"
  local wanted_csv=",$gpus,"
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
  local lane="$1"
  local gpus="$2"
  local idle_for=0
  log "${lane}: waiting for GPUs ${gpus} to have no compute processes for ${IDLE_SECONDS}s"
  while (( idle_for < IDLE_SECONDS )); do
    if gpu_has_processes "$gpus"; then
      idle_for=0
      log "${lane}: GPUs ${gpus} busy; polling again in ${POLL_SECONDS}s"
    else
      idle_for=$((idle_for + POLL_SECONDS))
      log "${lane}: GPUs ${gpus} idle for ${idle_for}/${IDLE_SECONDS}s"
    fi
    sleep "$POLL_SECONDS"
  done
}

run_eval_one() {
  local lane="$1"
  local gpus="$2"
  local val_files="$3"
  local scale_id="$4"
  local scale_value
  scale_value="$(scale_value_for_id "$scale_id")"
  local agent_cfg
  agent_cfg="$(agent_config_for_scale "$scale_id")"
  local run_name="${MODEL_LABEL}_${lane}_fast_rescale${scale_id}"
  local work_dir="$OUTPUT_ROOT/$run_name"
  local ray_tmp="/tmp/ndr_${lane}_${scale_id}"
  local status="success"
  local exit_code=0
  local launch_log="$work_dir/${run_name}.launch.log"

  mkdir -p "$work_dir" "$ray_tmp"
  log "${lane}: starting ${run_name} initial_rescale=${scale_value} GPUs=${gpus}"

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="$gpus" \
  MODEL_PATH="$MODEL_PATH" \
  LOAD_FORMAT="$LOAD_FORMAT" \
  WORK_DIR="$work_dir" \
  EXP_NAME="$run_name" \
  WANDB_NAME="$run_name" \
  VAL_FILES="$val_files" \
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

  if [[ -f "$launch_log" ]]; then
    if grep -q "'val_only_hf_model_rollout': True\\|val_only_hf_model_rollout: True" "$launch_log"; then
      log "${lane}: verified fast path for ${run_name}"
    else
      log "${lane}: WARNING fast-path flag not found for ${run_name}"
    fi
  fi

  if (( exit_code != 0 )); then
    status="failed"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "$lane" "$MODEL_LABEL" "$scale_id" "$scale_value" "$status" "$exit_code" "$work_dir" >> "$STATUS_TSV"
  log "${lane}: finished ${run_name} status=${status} exit_code=${exit_code}"
  return "$exit_code"
}

run_lane() {
  local lane="$1"
  local gpus="$2"
  local val_files="$3"
  local lane_status=0
  wait_for_gpu_idle "$lane" "$gpus"
  for scale_id in 025 035 05; do
    run_eval_one "$lane" "$gpus" "$val_files" "$scale_id" || lane_status=1
  done
  log "${lane}: lane finished status=${lane_status}"
  return "$lane_status"
}

for cfg in \
  recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml \
  recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml \
  recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml; do
  [[ -f "$cfg" ]] || { echo "missing config: $cfg" >&2; exit 2; }
done

log "queue started output_root=${OUTPUT_ROOT} model=${MODEL_PATH} actor=${ACTOR_DIR}"
ensure_merged_hf

status=0
pids=()
if [[ "$RUN_BROAD" == "1" ]]; then
  run_lane "broad" "${BROAD_GPUS:-4,5,6,7}" "$BROAD_VAL_FILES" &
  pids+=("$!")
fi
if [[ "$RUN_HIGHPAGE" == "1" ]]; then
  run_lane "highpage" "${HIGHPAGE_GPUS:-0,1,2,3}" "$HIGHPAGE_VAL_FILES" &
  pids+=("$!")
fi
if (( ${#pids[@]} == 0 )); then
  log "queue has no enabled lanes; set RUN_BROAD=1 and/or RUN_HIGHPAGE=1"
  exit 2
fi

for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
log "queue finished status=${status}"
exit "$status"
