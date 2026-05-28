#!/usr/bin/env bash
set -euo pipefail

CURRENT_PID="${CURRENT_PID:-2142207}"
CURRENT_STATUS="${CURRENT_STATUS:-/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_scratch_20260519/status.tsv}"
CURRENT_LOG="${CURRENT_LOG:-/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_scratch_20260519/full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch1_fp32_scratch/train.log}"
QUEUE_LOG="${QUEUE_LOG:-/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_queue_20260519/queue.log}"

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/scripts/queue_full_sft_both_higher_dpi_unans02503505_64k_sp2_8gpu_20260518.sh}"

mkdir -p "$(dirname "${QUEUE_LOG}")"

{
  echo "[queue2] watcher started=$(date -Is) current_pid=${CURRENT_PID}"
  while kill -0 "${CURRENT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "[queue2] current pid exited=$(date -Is)"

  if [[ ! -f "${CURRENT_STATUS}" ]]; then
    echo "[queue2] abort: missing current status file ${CURRENT_STATUS}"
    exit 1
  fi
  if ! grep -q "\\[queue\\] finished=" "${CURRENT_LOG}"; then
    echo "[queue2] abort: current queue did not print finished marker"
    tail -n 80 "${CURRENT_LOG}" || true
    exit 1
  fi
  awk -F'\t' '
    NR > 1 {
      if (($2 == "train" || $2 == "hf_model") && $4 != "0") bad = 1
      if ($2 == "eval" && $4 != "success") bad = 1
    }
    END { exit bad }
  ' "${CURRENT_STATUS}"

  echo "[queue2] current status successful; launching epoch2=$(date -Is)"
  cd "${REPO_ROOT}"
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy NO_PROXY no_proxy

  RUN_EVAL=1 \
  TOTAL_EPOCHS=2 \
  ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
  REFRESH_FREQ=50 \
  CHECKPOINTS_PER_EPOCH=2 \
  DATALOADER_NUM_WORKERS=4 \
  TRAINER_LOGGERS="['console','wandb']" \
  ENGINE_MODEL_DTYPE=fp32 \
  RDZV_PORT=29916 \
  RESUME_MODE=disable \
  OUTPUT_ROOT=/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260519 \
  EXP_ID=full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch2_fp32_scratch \
  MODEL_LABEL=full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_epoch2_fp32_scratch \
  bash "${LAUNCHER}"

  echo "[queue2] epoch2 finished=$(date -Is)"
} >> "${QUEUE_LOG}" 2>&1
