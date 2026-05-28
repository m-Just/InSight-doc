#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
PREV_EVAL_LOG="${PREV_EVAL_LOG:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_64k_sp2_bs32_8gpu_20260518_wandb_save2_rerun2/eval_queue.log}"
POLL_SECONDS="${POLL_SECONDS:-300}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_both_higher_dpi_unans02503505_64k_sp2_bs32_8gpu_20260518_wandb_save2_epoch2_after_rerun2_eval}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_64k_sp2_bs32_8gpu_20260518_wandb_save2_epoch2_after_rerun2_eval}"
EXP_ID="${EXP_ID:-full_sft_both_higher_dpi_unans02503505_len65536_sp2_bs32_freeze_vt_epoch2_after_rerun2_eval}"
MODEL_LABEL="${MODEL_LABEL:-full_sft_both_higher_dpi_unans02503505_len65536_sp2_bs32_epoch2_after_rerun2_eval}"

MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/chain.log}"
mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

log() {
  echo "[$(date '+%F %T')] $*" >&2
}

wait_for_previous_eval() {
  while true; do
    if [[ -f "${PREV_EVAL_LOG}" ]] && grep -q "all evals finished" "${PREV_EVAL_LOG}"; then
      log "previous eval completed according to ${PREV_EVAL_LOG}"
      return 0
    fi
    log "waiting for previous eval to finish; next check in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
  done
}

main() {
  log "chain started"
  log "prev_eval_log=${PREV_EVAL_LOG}"
  log "output_root=${OUTPUT_ROOT}"
  log "eval_output_root=${EVAL_OUTPUT_ROOT}"
  wait_for_previous_eval

  cd "${REPO_ROOT}"
  log "launching 2-epoch full-SFT + eval"
  env \
    HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= NO_PROXY= \
    https_proxy= http_proxy= all_proxy= no_proxy= \
    RUN_EVAL=1 \
    TOTAL_EPOCHS=2 \
    CHECKPOINTS_PER_EPOCH=2 \
    DATALOADER_NUM_WORKERS=4 \
    TRAINER_LOGGERS="['console','wandb']" \
    ENGINE_MODEL_DTYPE=bf16 \
    RDZV_PORT=29894 \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
    EXP_ID="${EXP_ID}" \
    MODEL_LABEL="${MODEL_LABEL}" \
    bash scripts/queue_full_sft_both_higher_dpi_unans02503505_64k_sp2_8gpu_20260518.sh
  log "chain finished"
}

main "$@"
