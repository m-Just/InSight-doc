#!/usr/bin/env bash
set -euo pipefail

LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/launch_insight_qwen_agent_full_sft_all_convos_0426.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_lora_sweep}"
MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/sweep.log}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-${MAX_LENGTH}}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-1}"
ENABLE_SFT_BATCH_SNAPSHOT="${ENABLE_SFT_BATCH_SNAPSHOT:-0}"
SFT_BATCH_SNAPSHOT_STEPS="${SFT_BATCH_SNAPSHOT_STEPS:-1}"
SFT_BATCH_SNAPSHOT_RANKS="${SFT_BATCH_SNAPSHOT_RANKS:-0}"
SFT_BATCH_SNAPSHOT_MAX_SAMPLES="${SFT_BATCH_SNAPSHOT_MAX_SAMPLES:-4}"
SFT_BATCH_SNAPSHOT_MAX_TOKENS_PER_SAMPLE="${SFT_BATCH_SNAPSHOT_MAX_TOKENS_PER_SAMPLE:-4096}"
SFT_BATCH_SNAPSHOT_INCLUDE_TOKEN_TABLE="${SFT_BATCH_SNAPSHOT_INCLUDE_TOKEN_TABLE:-True}"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

run_one() {
  local exp_id="$1"
  local lr="$2"
  local min_lr="$3"
  local train_batch_size="$4"
  local lora_rank="$5"
  local lora_alpha="$6"

  local min_lr_ratio
  min_lr_ratio="$(LR="${lr}" MIN_LR="${min_lr}" python - <<'PY'
import os
lr = float(os.environ["LR"])
min_lr = float(os.environ["MIN_LR"])
print(f"{min_lr / lr:.12g}")
PY
)"

  local exp_name="insight_qwen_agent_full_sft_all_convos_0426_${exp_id}"
  local work_dir="${OUTPUT_ROOT}/${exp_name}"
  local log_path="${work_dir}/train.log"
  local snapshot_dir=""
  local extra_sft_args=""
  local start_ts end_ts elapsed status

  mkdir -p "${work_dir}"
  if [[ "${ENABLE_SFT_BATCH_SNAPSHOT}" == "1" || "${ENABLE_SFT_BATCH_SNAPSHOT}" == "true" || "${ENABLE_SFT_BATCH_SNAPSHOT}" == "yes" ]]; then
    snapshot_dir="${work_dir}/sft_batch_snapshots"
  fi

  extra_sft_args="optim.lr_scheduler_type=cosine optim.lr_warmup_steps_ratio=0.05 optim.min_lr_ratio=${min_lr_ratio}"
  extra_sft_args="${extra_sft_args} model.lora_rank=${lora_rank} model.lora_alpha=${lora_alpha} model.target_modules=${TARGET_MODULES}"
  if [[ "${FREEZE_VISION_TOWER}" == "1" || "${FREEZE_VISION_TOWER}" == "true" || "${FREEZE_VISION_TOWER}" == "yes" ]]; then
    extra_sft_args="${extra_sft_args} model.freeze_vision_tower=True"
  fi
  if [[ -n "${snapshot_dir}" ]]; then
    extra_sft_args="${extra_sft_args} trainer.batch_snapshot_dir=${snapshot_dir}"
    extra_sft_args="${extra_sft_args} trainer.batch_snapshot_steps=\"${SFT_BATCH_SNAPSHOT_STEPS}\""
    extra_sft_args="${extra_sft_args} trainer.batch_snapshot_max_samples=${SFT_BATCH_SNAPSHOT_MAX_SAMPLES}"
    extra_sft_args="${extra_sft_args} trainer.batch_snapshot_max_tokens_per_sample=${SFT_BATCH_SNAPSHOT_MAX_TOKENS_PER_SAMPLE}"
    extra_sft_args="${extra_sft_args} trainer.batch_snapshot_include_token_table=${SFT_BATCH_SNAPSHOT_INCLUDE_TOKEN_TABLE}"
    if [[ -n "${SFT_BATCH_SNAPSHOT_RANKS:-}" ]]; then
      extra_sft_args="${extra_sft_args} trainer.batch_snapshot_ranks=\"${SFT_BATCH_SNAPSHOT_RANKS}\""
    fi
  fi

  echo
  echo "================================================================"
  echo "[${exp_id}] start: $(date -Is)"
  echo "[${exp_id}] lr=${lr} scheduler=cosine warmup=0.05 min_lr=${min_lr} min_lr_ratio=${min_lr_ratio} max_length=${MAX_LENGTH} max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} train_batch_size=${train_batch_size} total_epochs=2"
  echo "[${exp_id}] lora_rank=${lora_rank} lora_alpha=${lora_alpha} target_modules=${TARGET_MODULES} freeze_vision_tower=${FREEZE_VISION_TOWER}"
  echo "[${exp_id}] nproc_per_node=${NPROC_PER_NODE:-8} rdzv_port=${RDZV_PORT:-29521}"
  echo "[${exp_id}] work_dir=${work_dir}"
  echo "[${exp_id}] log_path=${log_path}"
  echo "[${exp_id}] sft_batch_snapshot_dir=${snapshot_dir:-disabled} steps=${SFT_BATCH_SNAPSHOT_STEPS} ranks=${SFT_BATCH_SNAPSHOT_RANKS}"
  echo "================================================================"

  start_ts="$(date +%s)"
  set +e
  EXP_NAME="${exp_name}" \
  WORK_DIR="${work_dir}" \
  LOG_PATH="${log_path}" \
  LEARNING_RATE="${lr}" \
  TOTAL_EPOCHS=2 \
  MAX_LENGTH="${MAX_LENGTH}" \
  MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU}" \
  TRAIN_BATCH_SIZE="${train_batch_size}" \
  TESTS_PER_EPOCH=4 \
  EXTRA_SFT_ARGS="${extra_sft_args}" \
    bash "${LAUNCH_SCRIPT}"
  status=$?
  set -e
  end_ts="$(date +%s)"
  elapsed=$(( end_ts - start_ts ))

  echo "================================================================"
  echo "[${exp_id}] end: $(date -Is)"
  echo "[${exp_id}] status=${status} elapsed_seconds=${elapsed}"
  printf '[%s] elapsed_hms=%02d:%02d:%02d\n' "${exp_id}" "$(( elapsed / 3600 ))" "$(( elapsed % 3600 / 60 ))" "$(( elapsed % 60 ))"
  echo "================================================================"

  return "${status}"
}

main() {
  local failures=0

  # Top full-finetune references on arxiv val:
  # - exp2_lr5e-6_cosine_minlr5e-7_len32768_bs32: 0.38235
  # - exp4_lr2e-6_cosine_minlr2e-7_len32768_bs8: 0.38235
  # - exp1_lr5e-6_cosine_minlr5e-7_len32768_bs16: 0.37255
  #
  # First-pass LoRA sweep:
  # - fix batch size at 32 to isolate the learning-rate effect
  # - keep LoRA shape fixed at rank 32 / alpha 64
  # - use a wider LR range than full finetuning would typically tolerate

  run_one "lora_exp1_lr2e-4_cosine_minlr2e-5_len32768_bs32_rank32_alpha64" "2e-4" "2e-5" "32" "32" "64" || {
    failures=$(( failures + 1 ))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  }

  run_one "lora_exp2_lr1e-4_cosine_minlr1e-5_len32768_bs32_rank32_alpha64" "1e-4" "1e-5" "32" "32" "64" || {
    failures=$(( failures + 1 ))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  }

  run_one "lora_exp3_lr5e-5_cosine_minlr5e-6_len32768_bs32_rank32_alpha64" "5e-5" "5e-6" "32" "32" "64" || {
    failures=$(( failures + 1 ))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  }

  echo
  echo "[sweep] finished: $(date -Is), failures=${failures}, master_log=${MASTER_LOG}"
  return "${failures}"
}

main "$@"
