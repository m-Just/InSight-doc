#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

MODEL_ROOT="${MODEL_ROOT:-/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams}"
ACTOR_DIR="${ACTOR_DIR:-$MODEL_ROOT/global_step_700/actor}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/global_step_700__actor_merged_hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/rl_ckpt700_broad_highpage_fast_rescale025_035_05_20260525}"
QUEUE_SCRIPT="${QUEUE_SCRIPT:-scripts/queue_new_data_rl_broad_highpage_fast_rescale025_035_05_20260524.sh}"

BROAD_WAIT_LOG="${BROAD_WAIT_LOG:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/rl_ckpt425_broad_fast_rescale025_035_05_repeat_trialA_20260525/queue.log}"
HIGHPAGE_WAIT_LOG="${HIGHPAGE_WAIT_LOG:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/rl_ckpt425_broad_fast_rescale025_035_05_repeat_trialB_20260525/queue.log}"

BROAD_GPUS="${BROAD_GPUS:-0,1,2,3}"
HIGHPAGE_GPUS="${HIGHPAGE_GPUS:-4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$OUTPUT_ROOT"
MASTER_LOG="$OUTPUT_ROOT/queue_wrapper.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

has_hf_weights() {
  [[ -f "$1/model.safetensors.index.json" || -f "$1/model.safetensors" || -n "$(find "$1" -maxdepth 1 -name '*.safetensors' -print -quit 2>/dev/null)" ]]
}

wait_for_queue_finished() {
  local label="$1"
  local wait_log="$2"
  log "$label: waiting for dependency queue: $wait_log"
  while true; do
    if [[ -f "$wait_log" ]] && grep -q "queue finished status=" "$wait_log"; then
      log "$label: dependency queue finished"
      return 0
    fi
    log "$label: dependency still running; polling again in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
}

ensure_merged_hf_once() {
  if has_hf_weights "$MODEL_PATH"; then
    log "using existing merged HF model: $MODEL_PATH"
    return 0
  fi

  [[ -d "$ACTOR_DIR" ]] || { echo "missing actor dir: $ACTOR_DIR" >&2; exit 2; }
  mkdir -p "$(dirname "$MODEL_PATH")"

  (
    flock 9
    if has_hf_weights "$MODEL_PATH"; then
      log "merged HF model appeared while waiting for lock: $MODEL_PATH"
      exit 0
    fi
    log "merging FSDP actor checkpoint: $ACTOR_DIR -> $MODEL_PATH"
    source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
    conda activate vllm-latest
    python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$ACTOR_DIR" \
      --target_dir "$MODEL_PATH" \
      2>&1 | tee -a "$MASTER_LOG"
  ) 9>/tmp/rl_ckpt700_actor_merge_20260525.lock

  if ! has_hf_weights "$MODEL_PATH"; then
    echo "merge did not produce HF weights: $MODEL_PATH" >&2
    exit 2
  fi
}

run_broad_lane() {
  wait_for_queue_finished "broad" "$BROAD_WAIT_LOG"
  ensure_merged_hf_once
  log "broad: launching rl_ckpt700 broad queue on GPUs $BROAD_GPUS"
  RUN_BROAD=1 \
  RUN_HIGHPAGE=0 \
  BROAD_GPUS="$BROAD_GPUS" \
  MODEL_LABEL="rl_ckpt700_actor_merged_hf" \
  MODEL_PATH="$MODEL_PATH" \
  ACTOR_DIR="$MODEL_PATH" \
  OUTPUT_ROOT="$OUTPUT_ROOT/broad_gpu${BROAD_GPUS//,/}" \
  LOAD_FORMAT=auto \
  bash "$QUEUE_SCRIPT"
}

run_highpage_lane() {
  wait_for_queue_finished "highpage" "$HIGHPAGE_WAIT_LOG"
  ensure_merged_hf_once
  log "highpage: launching rl_ckpt700 highpage queue on GPUs $HIGHPAGE_GPUS"
  RUN_BROAD=0 \
  RUN_HIGHPAGE=1 \
  HIGHPAGE_GPUS="$HIGHPAGE_GPUS" \
  MODEL_LABEL="rl_ckpt700_actor_merged_hf" \
  MODEL_PATH="$MODEL_PATH" \
  ACTOR_DIR="$MODEL_PATH" \
  OUTPUT_ROOT="$OUTPUT_ROOT/highpage_gpu${HIGHPAGE_GPUS//,/}" \
  LOAD_FORMAT=auto \
  bash "$QUEUE_SCRIPT"
}

log "wrapper started output_root=$OUTPUT_ROOT model_path=$MODEL_PATH actor_dir=$ACTOR_DIR"

status=0
run_broad_lane &
broad_pid=$!
run_highpage_lane &
highpage_pid=$!

wait "$broad_pid" || status=1
wait "$highpage_pid" || status=1

log "wrapper finished status=$status"
exit "$status"
