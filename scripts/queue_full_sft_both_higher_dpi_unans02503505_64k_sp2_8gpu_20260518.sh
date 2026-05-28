#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
GENERATED_BASE_DIR="${GENERATED_BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
UNANSWERABLE_ROOT="${UNANSWERABLE_ROOT:-${REPO_ROOT}/artifacts/synthetic_unanswerable_pipeline/first_batch_sft_parquets_20260517}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_both_higher_dpi_unans02503505_64k_sp2_bs32_8gpu_20260518}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/highpage_full_sft_both_higher_dpi_unans02503505_64k_sp2_bs32_8gpu_20260518}"
MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/queue.log}"
STATUS_TSV="${STATUS_TSV:-${OUTPUT_ROOT}/status.tsv}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VAL_FILE="${VAL_FILE:-${GENERATED_BASE_DIR}/arxiv/val_sample_102/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-0,1,2,3}"
RDZV_PORT="${RDZV_PORT:-29861}"

MAX_LENGTH="${MAX_LENGTH:-65536}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TESTS_PER_EPOCH="${TESTS_PER_EPOCH:-4}"
CHECKPOINTS_PER_EPOCH="${CHECKPOINTS_PER_EPOCH:-1}"
REFRESH_FREQ="${REFRESH_FREQ:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MIN_LR="${MIN_LR:-5e-7}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console','wandb']}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-False}"
ALLOW_OVERLENGTH="${ALLOW_OVERLENGTH:-False}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
RUN_EVAL="${RUN_EVAL:-1}"
ALLOW_EVAL_FAILURES="${ALLOW_EVAL_FAILURES:-0}"
ALLOW_POST_SAVE_TRAIN_CRASH="${ALLOW_POST_SAVE_TRAIN_CRASH:-0}"
SKIP_TRAIN_IF_HF_EXISTS="${SKIP_TRAIN_IF_HF_EXISTS:-0}"
ENGINE_MODEL_DTYPE="${ENGINE_MODEL_DTYPE:-bf16}"
TRAINER_DEBUG_SKIP_TRAIN_STEPS="${TRAINER_DEBUG_SKIP_TRAIN_STEPS:-0}"
TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-null}"

EXP_ID="${EXP_ID:-full_sft_both_higher_dpi_unans02503505_len${MAX_LENGTH}_sp${ULYSSES_SEQUENCE_PARALLEL_SIZE}_bs${TRAIN_BATCH_SIZE}_freeze_vt_epoch${TOTAL_EPOCHS}}"
MODEL_LABEL="${MODEL_LABEL:-full_sft_both_higher_dpi_unans02503505_len65536_sp2_bs32_epoch${TOTAL_EPOCHS}}"
WORK_DIR="${WORK_DIR:-${OUTPUT_ROOT}/${EXP_ID}}"
LOG_PATH="${LOG_PATH:-${WORK_DIR}/train.log}"

mkdir -p "${WORK_DIR}/sft_checkpoints"
exec > >(tee -a "${MASTER_LOG}") 2>&1

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-${REPO_ROOT}}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VAL_FILES_HIGH_PAGE='["/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/longdocurl_highpage_0507-insight_qwen_agent.parquet","/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/testcase_0507_parquets/mmlongbench_highpage_0507-insight_qwen_agent.parquet"]'

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

train_parts() {
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
  done < <(train_parts)
  unanswerable_medium_base_order_file "rescale025"
  unanswerable_medium_base_order_file "rescale035"
  unanswerable_medium_base_order_file "rescale05"
}

latest_hf_model() {
  local root="$1"
  local marker="${root}/latest_checkpointed_iteration.txt"
  local step
  if [[ -s "${marker}" ]]; then
    step="$(tr -dc '0-9' < "${marker}")"
  else
    step="$(find "${root}" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n | tail -1)"
  fi
  local hf_dir="${root}/global_step_${step}/huggingface"
  if [[ -z "${step}" || "${step}" -le 0 || ! -f "${hf_dir}/model.safetensors.index.json" ]]; then
    echo "ERROR: no usable HF checkpoint under ${root}, latest=${step:-missing}" >&2
    return 1
  fi
  printf '%s\n' "${hf_dir}"
}

record_status() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" "$4" >> "${STATUS_TSV}"
}

run_train() {
  local train_files_json train_rows val_rows steps_per_epoch freq_steps test_freq save_freq min_lr_ratio_value
  local status start_ts end_ts elapsed

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
  freq_steps="${steps_per_epoch}"
  save_freq="${SAVE_FREQ:-$(( (freq_steps + CHECKPOINTS_PER_EPOCH - 1) / CHECKPOINTS_PER_EPOCH ))}"
  test_freq="${TEST_FREQ:-$(( freq_steps / TESTS_PER_EPOCH ))}"
  if (( test_freq < 1 )); then
    test_freq=1
  fi
  if (( save_freq < 1 )); then
    save_freq=1
  fi

  echo -e "time\tstage\tname\tstatus\tpath" > "${STATUS_TSV}"
  echo "[queue] started=$(date -Is)"
  echo "[queue] cuda_devices=${CUDA_DEVICES} nproc=${NPROC_PER_NODE} eval_cuda_devices=${EVAL_CUDA_DEVICES}"
  echo "[queue] output_root=${OUTPUT_ROOT}"
  echo "[queue] eval_output_root=${EVAL_OUTPUT_ROOT}"
  echo "[queue] run_eval=${RUN_EVAL}"
  echo "[queue] allow_eval_failures=${ALLOW_EVAL_FAILURES}"
  echo "[queue] allow_post_save_train_crash=${ALLOW_POST_SAVE_TRAIN_CRASH}"
  echo "[queue] skip_train_if_hf_exists=${SKIP_TRAIN_IF_HF_EXISTS}"
  echo "[train:${EXP_ID}] train_rows=${train_rows} val_rows=${val_rows}"
  echo "[train:${EXP_ID}] max_length=${MAX_LENGTH} max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} ulysses_sp=${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
  echo "[train:${EXP_ID}] test_freq=${test_freq} save_freq=${save_freq} refresh_freq=${REFRESH_FREQ} checkpoints_per_epoch=${CHECKPOINTS_PER_EPOCH} dataloader_num_workers=${DATALOADER_NUM_WORKERS}"
  echo "[train:${EXP_ID}] full_weight=True freeze_vt=True lora_rank=0 engine_model_dtype=${ENGINE_MODEL_DTYPE}"
  echo "[train:${EXP_ID}] train_batch_size=${TRAIN_BATCH_SIZE} total_epochs=${TOTAL_EPOCHS} lr=${LEARNING_RATE} min_lr=${MIN_LR} min_lr_ratio=${min_lr_ratio_value}"
  echo "[train:${EXP_ID}] debug_skip_train_steps=${TRAINER_DEBUG_SKIP_TRAIN_STEPS} total_training_steps=${TRAINER_TOTAL_TRAINING_STEPS}"
  echo "[train:${EXP_ID}] resume_mode=${RESUME_MODE} resume_from_path=${RESUME_FROM_PATH}"
  echo "[train:${EXP_ID}] train_files=${train_files_json}"
  echo "[train:${EXP_ID}] work_dir=${WORK_DIR}"
  echo "[train:${EXP_ID}] log_path=${LOG_PATH}"

  cd "${REPO_ROOT}"
  start_ts="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc-per-node="${NPROC_PER_NODE}" \
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
    "model.freeze_vision_tower=True" \
    "model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD}" \
    "model.lora_rank=0" \
    "optim.lr=${LEARNING_RATE}" \
    "optim.lr_scheduler_type=cosine" \
    "optim.lr_warmup_steps_ratio=0.05" \
    "optim.min_lr_ratio=${min_lr_ratio_value}" \
    "engine=fsdp" \
    "optim=fsdp" \
    "engine.model_dtype=${ENGINE_MODEL_DTYPE}" \
    "engine.ulysses_sequence_parallel_size=${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    "trainer.logger=${TRAINER_LOGGERS}" \
    "trainer.project_name=insight_doc" \
    "trainer.experiment_name=${EXP_ID}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "trainer.total_training_steps=${TRAINER_TOTAL_TRAINING_STEPS}" \
    "trainer.debug_skip_train_steps=${TRAINER_DEBUG_SKIP_TRAIN_STEPS}" \
    "trainer.test_freq=${test_freq}" \
    "trainer.save_freq=${save_freq}" \
    "+trainer.refresh_freq=${REFRESH_FREQ}" \
    "trainer.default_local_dir=${WORK_DIR}/sft_checkpoints" \
    "trainer.resume_mode=${RESUME_MODE}" \
    "trainer.resume_from_path=${RESUME_FROM_PATH}" \
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]" \
    > >(tee -a "${LOG_PATH}") 2>&1
  status=$?
  set -e
  end_ts="$(date +%s)"
  elapsed=$(( end_ts - start_ts ))

  echo "[train:${EXP_ID}] finished=$(date -Is) status=${status} elapsed_seconds=${elapsed}"
  record_status train "${EXP_ID}" "${status}" "${WORK_DIR}"
  return "${status}"
}

run_eval_one() {
  local model_path="$1"
  local scale_id="$2"
  local scale_value="$3"
  local agent_cfg="$4"
  local run_name="${MODEL_LABEL}_highpage_0507_rescale${scale_id}"
  local work_dir="${EVAL_OUTPUT_ROOT}/${run_name}"
  local status="success"
  local exit_code=0

  echo "[eval:${run_name}] start=$(date -Is) model_path=${model_path}"

  unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
  unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

  set +e
  EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICES}" \
  MODEL_PATH="${model_path}" \
  LOAD_FORMAT="safetensors" \
  WORK_DIR="${work_dir}" \
  EXP_NAME="${run_name}" \
  WANDB_NAME="${run_name}" \
  VAL_FILES="${VAL_FILES_HIGH_PAGE}" \
  AGENT_LOOP_CONFIG_PATH="${agent_cfg}" \
  LOGGER="['console']" \
  MAX_RESPONSE_LENGTH="15360" \
  VAL_BATCH_SIZE="32" \
  TOOL_MAX_USER_TURNS="10" \
  TOOL_MAX_ASSISTANT_TURNS="11" \
  DATA_MAX_PROMPT_LENGTH="262144" \
  DATA_VALIDATION_MAX_PROMPT_LENGTH="262144" \
  ROLLOUT_MAX_MODEL_LEN="262144" \
  VLLM_MAX_MODEL_LEN="262144" \
  VLLM_GPU_MEMORY_UTILIZATION="0.9" \
  TELEGRAM_NOTIFY_LABEL="${run_name}" \
  bash scripts/run_iq_ft_eval_default_sampling_rl15360.sh \
    trainer.val_only_hf_model_rollout=true \
    trainer.resume_mode=disable
  exit_code=$?
  set -e

  if (( exit_code != 0 )); then
    status="failed"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" eval "${run_name}" "${status}" "${scale_value}" "${exit_code}" "${work_dir}" >> "${STATUS_TSV}"
  echo "[eval:${run_name}] finished=$(date -Is) status=${status} exit_code=${exit_code}"
}

run_eval() {
  local model_path="$1"
  local failed=0
  mkdir -p "${EVAL_OUTPUT_ROOT}"
  run_eval_one "${model_path}" 025 0.25 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale025.yaml || failed=1
  run_eval_one "${model_path}" 035 0.35 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale035.yaml || failed=1
  run_eval_one "${model_path}" 05 0.5 recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500_rescale05.yaml || failed=1
  if (( failed != 0 )); then
    if [[ "${ALLOW_EVAL_FAILURES}" == "1" || "${ALLOW_EVAL_FAILURES}" == "true" || "${ALLOW_EVAL_FAILURES}" == "yes" ]]; then
      echo "[eval] one or more evals failed; continuing because ALLOW_EVAL_FAILURES=${ALLOW_EVAL_FAILURES}"
      return 0
    fi
    return 1
  fi
}

main() {
  local hf_model
  if [[ "${SKIP_TRAIN_IF_HF_EXISTS}" == "1" || "${SKIP_TRAIN_IF_HF_EXISTS}" == "true" || "${SKIP_TRAIN_IF_HF_EXISTS}" == "yes" ]]; then
    if hf_model="$(latest_hf_model "${WORK_DIR}/sft_checkpoints")"; then
      echo "[queue] reusing existing HF checkpoint because SKIP_TRAIN_IF_HF_EXISTS=${SKIP_TRAIN_IF_HF_EXISTS}: ${hf_model}"
      record_status train "${EXP_ID}" skipped_existing_hf "${WORK_DIR}"
    else
      echo "[queue] no existing HF checkpoint found; training will run"
      run_train || {
        local train_status=$?
        if [[ "${ALLOW_POST_SAVE_TRAIN_CRASH}" == "1" || "${ALLOW_POST_SAVE_TRAIN_CRASH}" == "true" || "${ALLOW_POST_SAVE_TRAIN_CRASH}" == "yes" ]] && hf_model="$(latest_hf_model "${WORK_DIR}/sft_checkpoints")"; then
          echo "[queue] training exited with status=${train_status}, but usable HF checkpoint exists; continuing: ${hf_model}"
          record_status train "${EXP_ID}" "accepted_post_save_exit_${train_status}" "${WORK_DIR}"
        else
          return "${train_status}"
        fi
      }
    fi
  else
    run_train || {
      local train_status=$?
      if [[ "${ALLOW_POST_SAVE_TRAIN_CRASH}" == "1" || "${ALLOW_POST_SAVE_TRAIN_CRASH}" == "true" || "${ALLOW_POST_SAVE_TRAIN_CRASH}" == "yes" ]] && hf_model="$(latest_hf_model "${WORK_DIR}/sft_checkpoints")"; then
        echo "[queue] training exited with status=${train_status}, but usable HF checkpoint exists; continuing: ${hf_model}"
        record_status train "${EXP_ID}" "accepted_post_save_exit_${train_status}" "${WORK_DIR}"
      else
        return "${train_status}"
      fi
    }
  fi
  hf_model="$(latest_hf_model "${WORK_DIR}/sft_checkpoints")"
  record_status hf_model "${MODEL_LABEL}" 0 "${hf_model}"
  if [[ "${RUN_EVAL}" == "1" ]]; then
    run_eval "${hf_model}"
  else
    echo "[queue] skipping eval because RUN_EVAL=${RUN_EVAL}"
  fi
  echo "[queue] finished=$(date -Is)"
}

main "$@"
