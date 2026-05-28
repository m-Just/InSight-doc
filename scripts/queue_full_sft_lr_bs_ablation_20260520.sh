#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
LAUNCHER="${LAUNCHER:-${REPO_ROOT}/scripts/queue_full_sft_both_higher_dpi_unans02503505_64k_sp2_8gpu_20260518.sh}"
RUN_ROOT="${RUN_ROOT:-/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_lr_bs_ablation_20260520}"
EVAL_ROOT="${EVAL_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_lr_bs_ablation_20260520}"
MASTER_LOG="${MASTER_LOG:-${RUN_ROOT}/queue.log}"
MASTER_STATUS="${MASTER_STATUS:-${RUN_ROOT}/status.tsv}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]] && [[ -f /home/ywxzml3j/ywxzml3juser40/.bashrc ]]; then
  OPENAI_EXPORTS="$(grep -E '^export OPENAI_(API_KEY|BASE_URL)=' /home/ywxzml3j/ywxzml3juser40/.bashrc || true)"
  if [[ -n "${OPENAI_EXPORTS}" ]]; then
    eval "${OPENAI_EXPORTS}"
  fi
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for eval judging}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set for eval judging}"

mkdir -p "${RUN_ROOT}" "${EVAL_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

printf "time\tstage\texp_id\tstatus\tpath\n" > "${MASTER_STATUS}"

record_master_status() {
  printf "%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$1" "$2" "$3" "$4" >> "${MASTER_STATUS}"
}

run_one() {
  local tag="$1"
  local lr="$2"
  local min_lr="$3"
  local bs="$4"
  local port="$5"

  local exp_id="full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs${bs}_lr${tag}_freeze_vt_epoch1_fp32"
  local model_label="full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs${bs}_lr${tag}_epoch1_fp32"
  local output_root="${RUN_ROOT}/${exp_id}"
  local eval_output_root="${EVAL_ROOT}/${exp_id}"
  local launcher_log="${RUN_ROOT}/${exp_id}.launcher.log"
  local launcher_status="${RUN_ROOT}/${exp_id}.status.tsv"

  echo "[ablation:${exp_id}] start=$(date -Is) lr=${lr} min_lr=${min_lr} batch=${bs} port=${port}"
  record_master_status start "${exp_id}" running "${output_root}"

  unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
  unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

  CUDA_DEVICES="${CUDA_DEVICES}" \
  EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  RDZV_PORT="${port}" \
  RUN_EVAL=1 \
  ALLOW_EVAL_FAILURES=1 \
  ALLOW_POST_SAVE_TRAIN_CRASH=1 \
  SKIP_TRAIN_IF_HF_EXISTS=1 \
  TOTAL_EPOCHS=1 \
  MAX_LENGTH=65536 \
  ULYSSES_SEQUENCE_PARALLEL_SIZE=4 \
  MAX_TOKEN_LEN_PER_GPU=32768 \
  TRAIN_BATCH_SIZE="${bs}" \
  MICRO_BATCH_SIZE_PER_GPU=1 \
  LEARNING_RATE="${lr}" \
  MIN_LR="${min_lr}" \
  ENGINE_MODEL_DTYPE=fp32 \
  REFRESH_FREQ=50 \
  CHECKPOINTS_PER_EPOCH=2 \
  TESTS_PER_EPOCH=4 \
  DATALOADER_NUM_WORKERS=4 \
  TRAINER_LOGGERS="['console','wandb']" \
  RESUME_MODE=disable \
  FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-gemini-3.1-flash-lite-preview}" \
  TELEGRAM_NOTIFY_ON_FINISH="${TELEGRAM_NOTIFY_ON_FINISH:-1}" \
  OUTPUT_ROOT="${output_root}" \
  EVAL_OUTPUT_ROOT="${eval_output_root}" \
  MASTER_LOG="${launcher_log}" \
  STATUS_TSV="${launcher_status}" \
  EXP_ID="${exp_id}" \
  MODEL_LABEL="${model_label}" \
  bash "${LAUNCHER}"

  record_master_status finish "${exp_id}" success "${output_root}"
  echo "[ablation:${exp_id}] finish=$(date -Is)"
}

echo "[ablation] queue started=$(date -Is)"
echo "[ablation] run_root=${RUN_ROOT}"
echo "[ablation] eval_root=${EVAL_ROOT}"
echo "[ablation] cuda_devices=${CUDA_DEVICES} eval_cuda_devices=${EVAL_CUDA_DEVICES} nproc=${NPROC_PER_NODE}"
echo "[ablation] no proxy env is used by this wrapper or child evals"

run_one "2e-6" "2e-6" "2e-7" 32 29931
run_one "2e-6" "2e-6" "2e-7" 16 29932
run_one "1e-6" "1e-6" "1e-7" 32 29933

echo "[ablation] queue finished=$(date -Is)"
