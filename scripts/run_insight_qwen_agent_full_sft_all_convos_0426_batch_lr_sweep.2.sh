#!/usr/bin/env bash
set -euo pipefail

LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-/scratch/ywxzml3j/likaican/src/InSight-doc/verl/scripts/launch_insight_qwen_agent_full_sft_all_convos_0426.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep}"
MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/sweep.log}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-${MAX_LENGTH}}"
ENABLE_SFT_BATCH_SNAPSHOT="${ENABLE_SFT_BATCH_SNAPSHOT:-1}"
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
  local start_ts end_ts elapsed status

  mkdir -p "${work_dir}"
  if [[ "${ENABLE_SFT_BATCH_SNAPSHOT}" == "1" || "${ENABLE_SFT_BATCH_SNAPSHOT}" == "true" || "${ENABLE_SFT_BATCH_SNAPSHOT}" == "yes" ]]; then
    snapshot_dir="${work_dir}/sft_batch_snapshots"
  fi

  echo
  echo "================================================================"
  echo "[${exp_id}] start: $(date -Is)"
  echo "[${exp_id}] lr=${lr} scheduler=cosine warmup=0.05 min_lr=${min_lr} min_lr_ratio=${min_lr_ratio} max_length=${MAX_LENGTH} max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} train_batch_size=${train_batch_size} total_epochs=3"
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
  TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU}" \
  TRAIN_BATCH_SIZE="${train_batch_size}" \
  TESTS_PER_EPOCH=4 \
  SFT_BATCH_SNAPSHOT_DIR="${snapshot_dir}" \
  SFT_BATCH_SNAPSHOT_STEPS="${SFT_BATCH_SNAPSHOT_STEPS}" \
  SFT_BATCH_SNAPSHOT_RANKS="${SFT_BATCH_SNAPSHOT_RANKS}" \
  SFT_BATCH_SNAPSHOT_MAX_SAMPLES="${SFT_BATCH_SNAPSHOT_MAX_SAMPLES}" \
  SFT_BATCH_SNAPSHOT_MAX_TOKENS_PER_SAMPLE="${SFT_BATCH_SNAPSHOT_MAX_TOKENS_PER_SAMPLE}" \
  SFT_BATCH_SNAPSHOT_INCLUDE_TOKEN_TABLE="${SFT_BATCH_SNAPSHOT_INCLUDE_TOKEN_TABLE}" \
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

  # - exp2_lr5e-6_cosine_minlr5e-7_len32768_bs32: 0.38235
  # - exp4_lr2e-6_cosine_minlr2e-7_len32768_bs8: 0.38235
  # - exp1_lr5e-6_cosine_minlr5e-7_len32768_bs16: 0.37255
  # - exp3_lr8e-6_cosine_minlr8e-7_len32768_bs16: 0.35294
  # - exp1_lr3e-6_cosine_minlr3e-7_len32768_bs32_clean_data: 0.34314
  # - exp1_lr3e-6_cosine_minlr3e-7_len32768_bs32: 0.32353
  # - exp2_lr3e-6_cosine_minlr3e-7_len32768_bs16: 0.32353
  # - exp3_lr3e-6_cosine_minlr3e-7_len32768_bs8: 0.29412
  # - exp5_lr1e-6_cosine_minlr1e-7_len32768_bs32: 0.26471

  if [ "${USE_VSEARCHER_SYSTEM_FOR_EASY:-0}" == "1" ]; then

    run_one "lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_vs_sys4easy" "5e-6" "5e-7" "32" || {
      failures=$(( failures + 1 ))
      [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    }

    run_one "lr2e-6_cosine_minlr2e-7_len32768_bs8_full_clean_data_vs_sys4easy" "2e-6" "2e-7" "8" || {
      failures=$(( failures + 1 ))
      [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    }

    run_one "lr5e-6_cosine_minlr5e-7_len32768_bs16_full_clean_data_vs_sys4easy" "5e-6" "5e-7" "16" || {
      failures=$(( failures + 1 ))
      [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    }

  else

    export FREEZE_VISION_TOWER=1
    # run_one "lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt" "5e-6" "5e-7" "32" || {
    #   failures=$(( failures + 1 ))
    #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    # }

    export USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM=1
    export TRAIN_MEDIUM_ONLY=1
    run_one "lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt_tool_arg_order_medium_only" "5e-6" "5e-7" "32" || {
      failures=$(( failures + 1 ))
      [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    }
    # export TRAIN_MEDIUM_ONLY=0
    # run_one "lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt_tool_arg_order" "5e-6" "5e-7" "32" || {
    #   failures=$(( failures + 1 ))
    #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    # }
    # export USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM=0

    # export TOTAL_EPOCHS=5
    # run_one "lr2e-6_cosine_minlr2e-7_len32768_bs8_full_clean_data_freeze_vt" "2e-6" "2e-7" "8" || {
    #   failures=$(( failures + 1 ))
    #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    # }

    # export FREEZE_VISION_TOWER=0
    # export TOTAL_EPOCHS=3
    # run_one "lr5e-6_cosine_minlr5e-7_len32768_bs8_full_clean_data_freeze_vt" "5e-6" "5e-7" "8" || {
    #   failures=$(( failures + 1 ))
    #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    # }

    # run_one "lr2e-6_cosine_minlr2e-7_len32768_bs8_full_clean_data" "2e-6" "2e-7" "8" || {
    #   failures=$(( failures + 1 ))
    #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    # }

    # run_one "lr5e-6_cosine_minlr5e-7_len32768_bs16_full_clean_data" "5e-6" "5e-7" "16" || {
    #   failures=$(( failures + 1 ))
    #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    # }
  fi

  # ================================

  # run_one "exp1_lr3e-6_cosine_minlr3e-7_len32768_bs32_clean_data" "3e-6" "3e-7" "32" || {
  #   failures=$(( failures + 1 ))
  #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  # }

  # ================================

  # run_one "exp1_lr3e-6_cosine_minlr3e-7_len32768_bs32" "3e-6" "3e-7" "32" || {
  #   failures=$(( failures + 1 ))
  #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  # }

  

  # run_one "exp2_lr5e-6_cosine_minlr5e-7_len32768_bs32" "5e-6" "5e-7" "32" || {
  #   failures=$(( failures + 1 ))
  #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  # }

  # run_one "exp3_lr3e-6_cosine_minlr3e-7_len32768_bs8" "3e-6" "3e-7" "8" || {
  #   failures=$(( failures + 1 ))
  #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  # }

  # run_one "exp4_lr2e-6_cosine_minlr2e-7_len32768_bs8" "2e-6" "2e-7" "8" || {
  #   failures=$(( failures + 1 ))
  #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  # }

  # run_one "exp5_lr1e-6_cosine_minlr1e-7_len32768_bs32" "1e-6" "1e-7" "32" || {
  #   failures=$(( failures + 1 ))
  #   [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  # }

  echo
  echo "[sweep] finished: $(date -Is), failures=${failures}, master_log=${MASTER_LOG}"
  return "${failures}"
}

main "$@"
