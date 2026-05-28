#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
cd "$REPO_ROOT"

RL_QUEUE_ROOT="${RL_QUEUE_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/new_data_rl_fast_broad_highpage_rescale025_035_05_retry_no_o3_20260524}"
RL_QUEUE_LOG="${RL_QUEUE_ROOT}/queue.log"
POLL_SECONDS="${POLL_SECONDS:-30}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/base_tool_highpage035_fast_retry_after_rl_new_data_broad_20260524}"
HIGHPAGE_GPUS="${HIGHPAGE_GPUS:-4,5,6,7}"

log() {
  echo "[$(date '+%F %T')] $*"
}

log "waiting for rl_new_data broad lane to finish: ${RL_QUEUE_LOG}"
while true; do
  if [[ -f "$RL_QUEUE_LOG" ]] && grep -q 'broad: lane finished' "$RL_QUEUE_LOG"; then
    log "rl_new_data broad lane finished; launching base tool-use highpage rescale=0.35"
    break
  fi
  log "rl_new_data broad lane still running or waiting; polling again in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

RUN_BROAD=0 \
RUN_HIGHPAGE=1 \
WAIT_FOR_PIDS='' \
OUTPUT_ROOT="$BASE_OUTPUT_ROOT" \
HIGHPAGE_GPUS="$HIGHPAGE_GPUS" \
bash -x scripts/queue_base_tool_corrected_broad035_05_highpage035_fast_20260524.sh
