#!/usr/bin/env bash
set -euo pipefail

LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-/scratch/ywxzml3j/likaican/src/InSight-doc/verl/scripts/launch_insight_qwen_agent_full_sft_all_convos_0426.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_lr_sweep}"
MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/sweep.log}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

run_one() {
  local exp_id="$1"
  local lr="$2"
  local min_lr="$3"
  local max_length="$4"
  local train_batch_size="$5"

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
  local start_ts end_ts elapsed status

  mkdir -p "${work_dir}"

  echo
  echo "================================================================"
  echo "[${exp_id}] start: $(date -Is)"
  echo "[${exp_id}] lr=${lr} scheduler=cosine warmup=0.05 min_lr=${min_lr} min_lr_ratio=${min_lr_ratio} max_length=${max_length} train_batch_size=${train_batch_size} total_epochs=3"
  echo "[${exp_id}] work_dir=${work_dir}"
  echo "[${exp_id}] log_path=${log_path}"
  echo "================================================================"

  start_ts="$(date +%s)"
  set +e
  EXP_NAME="${exp_name}" \
  WORK_DIR="${work_dir}" \
  LOG_PATH="${log_path}" \
  LEARNING_RATE="${lr}" \
  TOTAL_EPOCHS=3 \
  MAX_LENGTH="${max_length}" \
  MAX_TOKEN_LEN_PER_GPU="${max_length}" \
  TRAIN_BATCH_SIZE="${train_batch_size}" \
  TESTS_PER_EPOCH=4 \
  EXTRA_SFT_ARGS="optim.lr_scheduler_type=cosine optim.lr_warmup_steps_ratio=0.05 optim.min_lr_ratio=${min_lr_ratio}" \
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

  # run_one "exp1_lr5e-6_cosine_minlr5e-7_len32768_bs16" "5e-6" "5e-7" "32768" "16" || {
  #   failures=$(( failures + 1 ))
  #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  # }

  run_one "exp2_lr3e-6_cosine_minlr3e-7_len32768_bs16" "3e-6" "3e-7" "32768" "16" || {
    failures=$(( failures + 1 ))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  }

  run_one "exp3_lr8e-6_cosine_minlr8e-7_len32768_bs16" "8e-6" "8e-7" "32768" "16" || {
    failures=$(( failures + 1 ))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  }

  echo
  echo "[sweep] finished: $(date -Is), failures=${failures}, master_log=${MASTER_LOG}"
  return "${failures}"
}

main "$@"
