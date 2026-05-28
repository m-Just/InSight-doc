#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
GENERATED_BASE_DIR="${GENERATED_BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"

CURRENT_OUTPUT_ROOT="${CURRENT_OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_both_higher_dpi_medium_only_epoch1_sp2_64k_20260517}"
CURRENT_EXP_ID="${CURRENT_EXP_ID:-lora_both_w_higher_dpi_len65536_sp2_bs32_rank32_alpha64_freeze_vt_medium_only_epoch1}"
CURRENT_CKPT_ROOT="${CURRENT_CKPT_ROOT:-${CURRENT_OUTPUT_ROOT}/${CURRENT_EXP_ID}/sft_checkpoints}"
CURRENT_STATUS_TSV="${CURRENT_STATUS_TSV:-${CURRENT_OUTPUT_ROOT}/status.tsv}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_both_higher_dpi_easy500_rs03505_20260517}"
MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/queue.log}"
STATUS_TSV="${STATUS_TSV:-${OUTPUT_ROOT}/status.tsv}"
EASY_SAMPLE_ROOT="${EASY_SAMPLE_ROOT:-${REPO_ROOT}/notes/generated/highpage_extra_easy500_rescale_samples_20260517}"
EASY_ROWS_PER_RESCALE="${EASY_ROWS_PER_RESCALE:-500}"
ADDITIONAL_TRAIN_FILES_JSON="${ADDITIONAL_TRAIN_FILES_JSON:-[]}"
CURRENT_REQUIRED_STAGE="${CURRENT_REQUIRED_STAGE:-}"

BASELINE_EVAL_OUTPUT_ROOT="${BASELINE_EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_epoch1_sp2_64k_20260517}"
EXTRA_EVAL_OUTPUT_ROOT="${EXTRA_EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_epoch1_plus_easy500_rs03505_20260517}"
EXTRA_EVAL_LABEL="${EXTRA_EVAL_LABEL:-lora_both_higher_dpi_epoch1_plus_easy500_rs03505}"
SKIP_BASELINE_EVAL="${SKIP_BASELINE_EVAL:-0}"
SKIP_EXTRA_EVAL="${SKIP_EXTRA_EVAL:-0}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VAL_FILE="${VAL_FILE:-${GENERATED_BASE_DIR}/arxiv/val_sample_102/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://az.gptplus5.com/v1}"

CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
RDZV_PORT="${RDZV_PORT:-29831}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TESTS_PER_EPOCH="${TESTS_PER_EPOCH:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MIN_LR="${MIN_LR:-2e-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console','wandb']}"
RESUME_MODE="${RESUME_MODE:-disable}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-False}"
ALLOW_OVERLENGTH="${ALLOW_OVERLENGTH:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DEBUG_SKIP_TRAIN_STEPS="${DEBUG_SKIP_TRAIN_STEPS:-0}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

EXTRA_EXP_ID="${EXTRA_EXP_ID:-lora_both_w_higher_dpi_easy500_rs035_05_len${MAX_LENGTH}_sp${ULYSSES_SEQUENCE_PARALLEL_SIZE}_bs${TRAIN_BATCH_SIZE}_rank${LORA_RANK}_alpha${LORA_ALPHA}_freeze_vt_epoch${TOTAL_EPOCHS}}"
EXTRA_WORK_DIR="${EXTRA_WORK_DIR:-${OUTPUT_ROOT}/${EXTRA_EXP_ID}}"
EXTRA_LOG_PATH="${EXTRA_LOG_PATH:-${EXTRA_WORK_DIR}/train.log}"

mkdir -p "${OUTPUT_ROOT}" "${EXTRA_WORK_DIR}/sft_checkpoints"
exec > >(tee -a "${MASTER_LOG}") 2>&1

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

export OPENAI_API_KEY OPENAI_BASE_URL
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-${REPO_ROOT}}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

json_from_lines() {
  python -c 'import json, sys; print(json.dumps([line.rstrip("\n") for line in sys.stdin if line.strip()]))'
}

min_lr_ratio() {
  LR="${LEARNING_RATE}" MIN_LR="${MIN_LR}" python - <<'PY'
import os
print(f"{float(os.environ['MIN_LR']) / float(os.environ['LR']):.12g}")
PY
}

latest_ckpt() {
  local root="$1"
  local marker="${root}/latest_checkpointed_iteration.txt"
  local step=""
  if [[ -s "${marker}" ]]; then
    step="$(tr -dc '0-9' < "${marker}")"
  else
    step="$(find "${root}" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' 2>/dev/null | sed 's/global_step_//' | sort -n | tail -1)"
  fi
  if [[ -z "${step}" ]]; then
    return 1
  fi
  printf '%s/global_step_%s\n' "${root}" "${step}"
}

wait_for_current_training() {
  echo "[queue] waiting for current training status: ${CURRENT_STATUS_TSV}"
  while true; do
    if [[ -s "${CURRENT_STATUS_TSV}" ]] && (( "$(wc -l < "${CURRENT_STATUS_TSV}")" > 1 )); then
      local stage
      stage="$(tail -1 "${CURRENT_STATUS_TSV}" | awk -F'\t' '{print $2}')"
      local status
      status="$(tail -1 "${CURRENT_STATUS_TSV}" | awk -F'\t' '{print $3}')"
      if [[ "${status}" == "0" ]]; then
        if [[ -z "${CURRENT_REQUIRED_STAGE}" || "${stage}" == "${CURRENT_REQUIRED_STAGE}" ]]; then
          echo "[queue] current queue reached required successful stage=${stage}: $(tail -1 "${CURRENT_STATUS_TSV}")"
          return 0
        fi
        echo "[queue] current queue latest successful stage=${stage}; waiting for required stage=${CURRENT_REQUIRED_STAGE}"
      else
        echo "[queue] current training finished with nonzero status=${status}; stopping"
        return 1
      fi
    fi
    local ckpt
    ckpt="$(latest_ckpt "${CURRENT_CKPT_ROOT}" 2>/dev/null || true)"
    echo "[queue] current training not done yet; latest_ckpt=${ckpt:-none}; sleeping 120s ($(date -Is))"
    sleep 120
  done
}

run_highpage_eval() {
  local ckpt_root="$1"
  local model_label="$2"
  local eval_output_root="$3"

  echo "[queue] highpage eval start model=${model_label} ckpt_root=${ckpt_root} output=${eval_output_root} $(date -Is)"
  EVAL_CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  OUTPUT_ROOT="${eval_output_root}" \
  MODEL_LABEL="${model_label}" \
  BOTH_ROOT="${ckpt_root}" \
  RESET_STATUS=1 \
  bash "${REPO_ROOT}/scripts/launch_highpage_lora_both_higher_dpi_20260515.sh"
  echo "[queue] highpage eval done model=${model_label} $(date -Is)"
}

medium_file() {
  local part="$1"
  case "${part}" in
    O3_data_0424/train_part3a|O3_data_0424/train_part3b|O3_data_0424/train_part3c|O3_data_0424/train_part3d)
      local aspect_filtered="${GENERATED_BASE_DIR}/${part}/medium/processed_gpt5_nano_rewrite_aspect_drop/sft_data_base_model_tool_argument_order.parquet"
      if [[ -e "${aspect_filtered}" ]]; then
        printf '%s\n' "${aspect_filtered}"
        return 0
      fi
      ;;
  esac
  printf '%s\n' "${GENERATED_BASE_DIR}/${part}/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet"
}

medium_parts() {
  printf '%s\n' \
    "O3_data_0424/train_part1" \
    "O3_data_0424/train_part2a" \
    "O3_data_0424/train_part2b" \
    "O3_data_0424/train_part2c" \
    "O3_data_0424/dude_poster_unanswerable" \
    "arxiv/train_part1" \
    "arxiv/train_part2" \
    "arxiv/train_part3" \
    "arxiv/spanning_train_part1" \
    "arxiv/train_part4" \
    "arxiv/train_part5" \
    "O3_data_0424/train_part3a" \
    "O3_data_0424/train_part3b" \
    "O3_data_0424/train_part3c" \
    "O3_data_0424/train_part3d"
}

build_train_files() {
  local part
  while IFS= read -r part; do
    medium_file "${part}"
  done < <(medium_parts)
  printf '%s\n' "${EASY_SAMPLE_ROOT}/easy_random${EASY_ROWS_PER_RESCALE}_rescale035_with_vsearcher_system.parquet"
  printf '%s\n' "${EASY_SAMPLE_ROOT}/easy_random${EASY_ROWS_PER_RESCALE}_rescale05_with_vsearcher_system.parquet"
  ADDITIONAL_TRAIN_FILES_JSON="${ADDITIONAL_TRAIN_FILES_JSON}" python - <<'PY'
import json
import os

for path in json.loads(os.environ["ADDITIONAL_TRAIN_FILES_JSON"]):
    print(path)
PY
}

run_extra_sft() {
  local train_files_json train_rows val_rows steps_per_epoch test_freq save_freq min_lr_ratio_value
  local total_training_steps_args=()
  local status start_ts end_ts elapsed

  python "${REPO_ROOT}/scripts/build_highpage_extra_easy_rescale_samples_20260517.py" \
    --out-root "${EASY_SAMPLE_ROOT}" \
    --rows-per-rescale "${EASY_ROWS_PER_RESCALE}" \
    --overwrite

  train_files_json="$(build_train_files | json_from_lines)"
  min_lr_ratio_value="$(min_lr_ratio)"

  read -r train_rows val_rows <<EOF
$(TRAIN_FILES_JSON="${train_files_json}" VAL_FILE="${VAL_FILE}" python - <<'PY'
import json
import os
import pyarrow.parquet as pq

train_files = json.loads(os.environ["TRAIN_FILES_JSON"])
val_file = os.environ["VAL_FILE"]
missing = [path for path in [*train_files, val_file] if not os.path.exists(path)]
if missing:
    raise SystemExit("Missing parquet files:\n" + "\n".join(missing))
print(sum(pq.read_metadata(path).num_rows for path in train_files), pq.read_metadata(val_file).num_rows)
PY
)
EOF

  steps_per_epoch=$(( train_rows / TRAIN_BATCH_SIZE ))
  if (( steps_per_epoch < 1 )); then
    steps_per_epoch=1
  fi
  save_freq="${SAVE_FREQ:-${steps_per_epoch}}"
  test_freq="${TEST_FREQ:-$(( steps_per_epoch / TESTS_PER_EPOCH ))}"
  if (( test_freq < 1 )); then
    test_freq=1
  fi
  if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    total_training_steps_args=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
  fi

  echo "[queue:${EXTRA_EXP_ID}] train_rows=${train_rows} val_rows=${val_rows}"
  echo "[queue:${EXTRA_EXP_ID}] dataloader_num_workers=${DATALOADER_NUM_WORKERS} debug_skip_train_steps=${DEBUG_SKIP_TRAIN_STEPS} total_training_steps=${TOTAL_TRAINING_STEPS:-auto}"
  echo "[queue:${EXTRA_EXP_ID}] train_files=${train_files_json}"
  echo "[queue:${EXTRA_EXP_ID}] work_dir=${EXTRA_WORK_DIR}"
  echo "[queue:${EXTRA_EXP_ID}] log_path=${EXTRA_LOG_PATH}"

  cd "${REPO_ROOT}"
  start_ts="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc-per-node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:${RDZV_PORT}" \
    -m verl.trainer.sft_trainer \
    "data.train_files=${train_files_json}" \
    "data.val_files=${VAL_FILE}" \
    "data.train_max_samples=${train_rows}" \
    "data.val_max_samples=${val_rows}" \
    "data.messages_key=messages" \
    "data.tools_key=tools" \
    "+data.message_loss_mask_key=message_loss_mask" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}" \
    "data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}" \
    "data.use_dynamic_bsz=${USE_DYNAMIC_BSZ}" \
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}" \
    "data.max_length=${MAX_LENGTH}" \
    "data.allow_overlength=${ALLOW_OVERLENGTH}" \
    "data.pad_mode=no_padding" \
    "data.truncation=error" \
    "model.path=${BASE_MODEL}" \
    "model.use_remove_padding=True" \
    "engine.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    "model.freeze_vision_tower=True" \
    "model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD}" \
    "model.lora_rank=${LORA_RANK}" \
    "model.lora_alpha=${LORA_ALPHA}" \
    "model.target_modules=${TARGET_MODULES}" \
    "optim.lr=${LEARNING_RATE}" \
    "optim.lr_scheduler_type=cosine" \
    "optim.lr_warmup_steps_ratio=0.05" \
    "optim.min_lr_ratio=${min_lr_ratio_value}" \
    "engine=fsdp" \
    "optim=fsdp" \
    "trainer.logger=${TRAINER_LOGGERS}" \
    "trainer.project_name=insight_doc" \
    "trainer.experiment_name=${EXTRA_EXP_ID}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "${total_training_steps_args[@]}" \
    "trainer.debug_skip_train_steps=${DEBUG_SKIP_TRAIN_STEPS}" \
    "trainer.test_freq=${test_freq}" \
    "trainer.save_freq=${save_freq}" \
    "trainer.default_local_dir=${EXTRA_WORK_DIR}/sft_checkpoints" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]" \
    > >(tee -a "${EXTRA_LOG_PATH}") 2>&1
  status=$?
  set -e
  end_ts="$(date +%s)"
  elapsed=$(( end_ts - start_ts ))
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "extra_sft" "${status}" "${train_rows}" "${EXTRA_WORK_DIR}" "${EXTRA_LOG_PATH}" >> "${STATUS_TSV}"
  echo "[queue:${EXTRA_EXP_ID}] finished status=${status} elapsed_seconds=${elapsed}"
  return "${status}"
}

main() {
  echo -e "time\tstage\tstatus\trows\twork_dir\tlog_path" > "${STATUS_TSV}"
  echo "[queue] started=$(date -Is)"
  echo "[queue] cuda_devices=${CUDA_DEVICES}"
  echo "[queue] current_ckpt_root=${CURRENT_CKPT_ROOT}"
  echo "[queue] output_root=${OUTPUT_ROOT}"

  wait_for_current_training
  if [[ "${SKIP_BASELINE_EVAL}" == "1" ]]; then
    echo "[queue] skipping baseline eval; using existing baseline output=${BASELINE_EVAL_OUTPUT_ROOT}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "baseline_eval_skipped" "0" "-" "${BASELINE_EVAL_OUTPUT_ROOT}" "-" >> "${STATUS_TSV}"
  else
    run_highpage_eval "${CURRENT_CKPT_ROOT}" "lora_both_higher_dpi_epoch1_sp2_64k" "${BASELINE_EVAL_OUTPUT_ROOT}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "baseline_eval" "0" "-" "${BASELINE_EVAL_OUTPUT_ROOT}" "-" >> "${STATUS_TSV}"
  fi

  run_extra_sft
  if [[ "${SKIP_EXTRA_EVAL}" == "1" ]]; then
    echo "[queue] skipping extra eval"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "extra_eval_skipped" "0" "-" "${EXTRA_EVAL_OUTPUT_ROOT}" "-" >> "${STATUS_TSV}"
  else
    run_highpage_eval "${EXTRA_WORK_DIR}/sft_checkpoints" "${EXTRA_EVAL_LABEL}" "${EXTRA_EVAL_OUTPUT_ROOT}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "extra_eval" "0" "-" "${EXTRA_EVAL_OUTPUT_ROOT}" "-" >> "${STATUS_TSV}"
  fi

  echo "[queue] finished=$(date -Is)"
}

main "$@"
