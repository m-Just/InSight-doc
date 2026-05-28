#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
GENERATED_BASE_DIR="${GENERATED_BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
UNANSWERABLE_ROOT="${UNANSWERABLE_ROOT:-${REPO_ROOT}/artifacts/synthetic_unanswerable_pipeline/first_batch_sft_parquets_20260517}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_unanswerable_medium_lane0_20260517}"
MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/queue.log}"
STATUS_TSV="${STATUS_TSV:-${OUTPUT_ROOT}/status.tsv}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VAL_FILE="${VAL_FILE:-${GENERATED_BASE_DIR}/arxiv/val_sample_102/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

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
RUN_BASIC="${RUN_BASIC:-1}"
RUN_BOTH="${RUN_BOTH:-1}"
BASIC_EXP_ID="${BASIC_EXP_ID:-}"
BOTH_EXP_ID="${BOTH_EXP_ID:-}"
BASIC_TOTAL_TRAINING_STEPS="${BASIC_TOTAL_TRAINING_STEPS:-}"
BOTH_TOTAL_TRAINING_STEPS="${BOTH_TOTAL_TRAINING_STEPS:-}"
BASIC_EVAL_OUTPUT_ROOT="${BASIC_EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_basic_unanswerable025_20260517}"
BOTH_EVAL_OUTPUT_ROOT="${BOTH_EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_lora_both_higher_dpi_unanswerable02503505_20260517}"
BASIC_EVAL_LABEL="${BASIC_EVAL_LABEL:-lora_basic_unanswerable025_len32768_sp1_epoch1}"
BOTH_EVAL_LABEL="${BOTH_EVAL_LABEL:-lora_both_higher_dpi_unanswerable02503505_len65536_sp2_epoch1}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

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

unanswerable_medium_base_order_file() {
  local scale="$1"
  printf '%s\n' "${UNANSWERABLE_ROOT}/${scale}/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet"
}

basic_parts() {
  printf '%s\n' \
    "O3_data_0424/train_part1" \
    "O3_data_0424/train_part2a" \
    "O3_data_0424/train_part2b" \
    "O3_data_0424/train_part2c" \
    "O3_data_0424/dude_poster_unanswerable" \
    "arxiv/train_part1" \
    "arxiv/train_part2" \
    "arxiv/train_part3" \
    "arxiv/spanning_train_part1"
}

both_higher_dpi_extra_parts() {
  printf '%s\n' \
    "arxiv/train_part4" \
    "arxiv/train_part5" \
    "O3_data_0424/train_part3a" \
    "O3_data_0424/train_part3b" \
    "O3_data_0424/train_part3c" \
    "O3_data_0424/train_part3d"
}

build_train_files() {
  local dataset="$1"
  local part
  while IFS= read -r part; do
    medium_file "${part}"
  done < <(basic_parts)
  if [[ "${dataset}" == "both_higher_dpi_unans02503505" ]]; then
    while IFS= read -r part; do
      medium_file "${part}"
    done < <(both_higher_dpi_extra_parts)
    unanswerable_medium_base_order_file "rescale025"
    unanswerable_medium_base_order_file "rescale035"
    unanswerable_medium_base_order_file "rescale05"
  elif [[ "${dataset}" == "basic_unans025" ]]; then
    unanswerable_medium_base_order_file "rescale025"
  else
    echo "unknown dataset: ${dataset}" >&2
    return 1
  fi
}

run_sft() {
  local dataset="$1"
  local exp_id="$2"
  local max_length="$3"
  local sp="$4"
  local max_token_len_per_gpu="$5"
  local rdzv_port="$6"
  local total_training_steps="${7:-}"

  local work_dir="${OUTPUT_ROOT}/${exp_id}"
  local log_path="${work_dir}/train.log"
  local train_files_json train_rows val_rows steps_per_epoch freq_steps test_freq save_freq min_lr_ratio_value
  local total_training_steps_args=()
  local status start_ts end_ts elapsed

  mkdir -p "${work_dir}/sft_checkpoints"
  train_files_json="$(build_train_files "${dataset}" | json_from_lines)"
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
  freq_steps="${steps_per_epoch}"
  if [[ -n "${total_training_steps}" ]]; then
    freq_steps="${total_training_steps}"
    total_training_steps_args=("trainer.total_training_steps=${total_training_steps}")
  fi
  save_freq="${SAVE_FREQ:-${freq_steps}}"
  test_freq="${TEST_FREQ:-$(( freq_steps / TESTS_PER_EPOCH ))}"
  if (( test_freq < 1 )); then
    test_freq=1
  fi

  echo
  echo "================================================================"
  echo "[train:${exp_id}] start=$(date -Is)"
  echo "[train:${exp_id}] cuda_devices=${CUDA_DEVICES} rdzv_port=${rdzv_port} dataset=${dataset} train_rows=${train_rows} val_rows=${val_rows}"
  echo "[train:${exp_id}] max_length=${max_length} max_token_len_per_gpu=${max_token_len_per_gpu} ulysses_sp=${sp}"
  echo "[train:${exp_id}] train_batch_size=${TRAIN_BATCH_SIZE} total_epochs=${TOTAL_EPOCHS} total_training_steps=${total_training_steps:-auto} lr=${LEARNING_RATE} min_lr=${MIN_LR} min_lr_ratio=${min_lr_ratio_value}"
  echo "[train:${exp_id}] lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA} target_modules=${TARGET_MODULES} freeze_vt=True"
  echo "[train:${exp_id}] train_files=${train_files_json}"
  echo "[train:${exp_id}] work_dir=${work_dir}"
  echo "[train:${exp_id}] log_path=${log_path}"
  echo "================================================================"

  cd "${REPO_ROOT}"
  start_ts="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:${rdzv_port}" \
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
    "data.use_dynamic_bsz=${USE_DYNAMIC_BSZ}" \
    "data.max_token_len_per_gpu=${max_token_len_per_gpu}" \
    "data.max_length=${max_length}" \
    "data.allow_overlength=${ALLOW_OVERLENGTH}" \
    "data.pad_mode=no_padding" \
    "data.truncation=error" \
    "model.path=${BASE_MODEL}" \
    "model.use_remove_padding=True" \
    "engine.ulysses_sequence_parallel_size=${sp}" \
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
    "trainer.experiment_name=${exp_id}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "${total_training_steps_args[@]}" \
    "trainer.test_freq=${test_freq}" \
    "trainer.save_freq=${save_freq}" \
    "trainer.default_local_dir=${work_dir}/sft_checkpoints" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]" \
    > >(tee -a "${log_path}") 2>&1
  status=$?
  set -e
  end_ts="$(date +%s)"
  elapsed=$(( end_ts - start_ts ))

  echo "[train:${exp_id}] finished=$(date -Is) status=${status} elapsed_seconds=${elapsed}"
  printf '%s\ttrain\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${exp_id}" "${status}" "${train_rows}" "${work_dir}" "${log_path}" >> "${STATUS_TSV}"
  return "${status}"
}

run_eval() {
  local exp_id="$1"
  local model_label="$2"
  local output_root="$3"
  local sft_root="${OUTPUT_ROOT}/${exp_id}/sft_checkpoints"

  echo
  echo "================================================================"
  echo "[eval:${model_label}] start=$(date -Is)"
  echo "[eval:${model_label}] sft_root=${sft_root}"
  echo "[eval:${model_label}] output_root=${output_root}"
  echo "================================================================"

  unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy
  RESET_STATUS=1 \
  EVAL_CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  OUTPUT_ROOT="${output_root}" \
  MODEL_LABEL="${model_label}" \
  BOTH_ROOT="${sft_root}" \
  bash "${REPO_ROOT}/scripts/launch_highpage_lora_both_higher_dpi_20260515.sh"

  printf '%s\teval\t%s\t0\t-\t%s\t%s\n' \
    "$(date -Is)" "${model_label}" "${output_root}" "${output_root}/status.tsv" >> "${STATUS_TSV}"
}

main() {
  echo -e "time\tstage\tname\tstatus\ttrain_rows\twork_dir\tlog_path" > "${STATUS_TSV}"
  echo "[queue] started=$(date -Is)"
  echo "[queue] cuda_devices=${CUDA_DEVICES}"
  echo "[queue] output_root=${OUTPUT_ROOT}"
  echo "[queue] master_log=${MASTER_LOG}"
  echo "[queue] unanswerable_root=${UNANSWERABLE_ROOT}"
  echo "[queue] no_proxy=1"
  echo "[queue] run_basic=${RUN_BASIC} run_both=${RUN_BOTH}"

  local basic_exp="${BASIC_EXP_ID:-lora_basic_unanswerable025_len32768_sp1_bs32_rank32_alpha64_freeze_vt_medium_only_epoch1}"
  if [[ "${RUN_BASIC}" == "1" ]]; then
    run_sft "basic_unans025" "${basic_exp}" 32768 1 32768 29841 "${BASIC_TOTAL_TRAINING_STEPS}"
    run_eval "${basic_exp}" \
      "${BASIC_EVAL_LABEL}" \
      "${BASIC_EVAL_OUTPUT_ROOT}"
  fi

  local both_exp="${BOTH_EXP_ID:-lora_both_higher_dpi_unanswerable02503505_len65536_sp2_bs32_rank32_alpha64_freeze_vt_medium_only_epoch1}"
  if [[ "${RUN_BOTH}" == "1" ]]; then
    run_sft "both_higher_dpi_unans02503505" "${both_exp}" 65536 2 32768 29851 "${BOTH_TOTAL_TRAINING_STEPS}"
    run_eval "${both_exp}" \
      "${BOTH_EVAL_LABEL}" \
      "${BOTH_EVAL_OUTPUT_ROOT}"
  fi

  echo "[queue] finished=$(date -Is)"
}

main "$@"
