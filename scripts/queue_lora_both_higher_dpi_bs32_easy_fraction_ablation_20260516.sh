#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
GENERATED_BASE_DIR="${GENERATED_BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
EASY_SAMPLE_ROOT="${EASY_SAMPLE_ROOT:-${REPO_ROOT}/notes/generated/easy_fraction_samples_20260516}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_both_higher_dpi_bs32_easy_fraction_ablation_20260516}"
MASTER_LOG="${MASTER_LOG:-${OUTPUT_ROOT}/queue.log}"
STATUS_TSV="${STATUS_TSV:-${OUTPUT_ROOT}/status.tsv}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VAL_FILE="${VAL_FILE:-${GENERATED_BASE_DIR}/arxiv/val_sample_102/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet}"
OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://az.gptplus5.com/v1}"

MAX_LENGTH="${MAX_LENGTH:-65536}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TESTS_PER_EPOCH="${TESTS_PER_EPOCH:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MIN_LR="${MIN_LR:-2e-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console','wandb']}"
RESUME_MODE="${RESUME_MODE:-disable}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-False}"
ALLOW_OVERLENGTH="${ALLOW_OVERLENGTH:-False}"

WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"
WAIT_GPUS="${WAIT_GPUS:-0,1,2,3,4,5,6,7}"
WAIT_MEM_THRESHOLD_MIB="${WAIT_MEM_THRESHOLD_MIB:-2048}"
WAIT_IDLE_STABLE_SECONDS="${WAIT_IDLE_STABLE_SECONDS:-300}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

mkdir -p "${OUTPUT_ROOT}"
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

wait_for_idle_gpus() {
  [[ "${WAIT_FOR_IDLE}" == "1" || "${WAIT_FOR_IDLE}" == "true" || "${WAIT_FOR_IDLE}" == "yes" ]] || return 0
  local stable=0
  local wait_list=",${WAIT_GPUS},"
  echo "[queue] waiting for GPUs ${WAIT_GPUS} to stay under ${WAIT_MEM_THRESHOLD_MIB} MiB for ${WAIT_IDLE_STABLE_SECONDS}s"
  while true; do
    local busy=0
    local line idx mem
    while IFS=, read -r idx mem; do
      idx="${idx//[[:space:]]/}"
      mem="${mem// MiB/}"
      mem="${mem//[[:space:]]/}"
      if [[ "${wait_list}" == *",${idx},"* && "${mem}" =~ ^[0-9]+$ && "${mem}" -gt "${WAIT_MEM_THRESHOLD_MIB}" ]]; then
        busy=1
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    if (( busy == 0 )); then
      stable=$(( stable + WAIT_POLL_SECONDS ))
      echo "[queue] GPUs idle for ${stable}/${WAIT_IDLE_STABLE_SECONDS}s ($(date -Is))"
      if (( stable >= WAIT_IDLE_STABLE_SECONDS )); then
        return 0
      fi
    else
      stable=0
      echo "[queue] GPUs still busy; sleeping ${WAIT_POLL_SECONDS}s ($(date -Is))"
    fi
    sleep "${WAIT_POLL_SECONDS}"
  done
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

build_medium_files() {
  local parts=(
    "O3_data_0424/train_part1"
    "O3_data_0424/train_part2a"
    "O3_data_0424/train_part2b"
    "O3_data_0424/train_part2c"
    "O3_data_0424/dude_poster_unanswerable"
    "arxiv/train_part1"
    "arxiv/train_part2"
    "arxiv/train_part3"
    "arxiv/spanning_train_part1"
    "arxiv/train_part4"
    "arxiv/train_part5"
    "O3_data_0424/train_part3a"
    "O3_data_0424/train_part3b"
    "O3_data_0424/train_part3c"
    "O3_data_0424/train_part3d"
  )
  local part
  for part in "${parts[@]}"; do
    medium_file "${part}"
  done
}

build_easy_files() {
  local fraction_label="$1"
  MANIFEST="${EASY_SAMPLE_ROOT}/manifest.tsv" FRACTION_LABEL="${fraction_label}" python - <<'PY'
import csv
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
label = os.environ["FRACTION_LABEL"]
if not manifest.exists():
    raise SystemExit(f"missing manifest: {manifest}")
with manifest.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["fraction_label"] == label:
            print(row["output"])
PY
}

build_train_files() {
  local fraction_label="$1"
  build_medium_files
  build_easy_files "${fraction_label}"
}

run_one() {
  local fraction_label="$1"
  local fraction_name="$2"
  local lane="$3"
  local cuda_devices="$4"
  local rdzv_port="$5"

  local exp_id="lora_both_w_higher_dpi_easy${fraction_name}_lr2e-4_cosine_minlr2e-5_len${MAX_LENGTH}_bs${TRAIN_BATCH_SIZE}_rank${LORA_RANK}_alpha${LORA_ALPHA}_freeze_vt"
  local work_dir="${OUTPUT_ROOT}/${exp_id}"
  local log_path="${work_dir}/train.log"
  local train_files_json train_rows val_rows steps_per_epoch test_freq save_freq min_lr_ratio_value
  local status start_ts end_ts elapsed

  mkdir -p "${work_dir}/sft_checkpoints"
  train_files_json="$(build_train_files "${fraction_label}" | json_from_lines)"
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

  echo
  echo "================================================================"
  echo "[${lane}:${exp_id}] start: $(date -Is)"
  echo "[${lane}:${exp_id}] cuda_devices=${cuda_devices} rdzv_port=${rdzv_port} train_rows=${train_rows} val_rows=${val_rows}"
  echo "[${lane}:${exp_id}] medium=both_w_higher_dpi easy_fraction=${fraction_name} max_length=${MAX_LENGTH} max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} ulysses_sp=${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
  echo "[${lane}:${exp_id}] train_batch_size=${TRAIN_BATCH_SIZE} total_epochs=${TOTAL_EPOCHS} lr=${LEARNING_RATE} min_lr=${MIN_LR} min_lr_ratio=${min_lr_ratio_value}"
  echo "[${lane}:${exp_id}] lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA} target_modules=${TARGET_MODULES} freeze_vt=True"
  echo "[${lane}:${exp_id}] work_dir=${work_dir}"
  echo "[${lane}:${exp_id}] log_path=${log_path}"
  echo "================================================================"

  cd "${REPO_ROOT}"
  start_ts="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${cuda_devices}" torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc-per-node=4 \
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
    "trainer.experiment_name=${exp_id}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
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

  echo "================================================================"
  echo "[${lane}:${exp_id}] end: $(date -Is)"
  echo "[${lane}:${exp_id}] status=${status} elapsed_seconds=${elapsed}"
  printf '[%s:%s] elapsed_hms=%02d:%02d:%02d\n' "${lane}" "${exp_id}" "$(( elapsed / 3600 ))" "$(( elapsed % 3600 / 60 ))" "$(( elapsed % 60 ))"
  echo "================================================================"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${lane}" "${fraction_name}" "${status}" "${train_rows}" "${work_dir}" "${log_path}" >> "${STATUS_TSV}"

  return "${status}"
}

main() {
  echo "[queue] output_root=${OUTPUT_ROOT}"
  echo "[queue] master_log=${MASTER_LOG}"
  echo "[queue] easy_sample_root=${EASY_SAMPLE_ROOT}"
  echo "[queue] started=$(date -Is)"
  echo -e "time\tlane\teasy_fraction\tstatus\ttrain_rows\twork_dir\tlog_path" > "${STATUS_TSV}"

  python "${REPO_ROOT}/scripts/build_easy_fraction_sft_parquets_20260516.py" --out-root "${EASY_SAMPLE_ROOT}"
  wait_for_idle_gpus

  run_one "500" "50pct" lane0 "0,1,2,3" "${LANE0_RDZV_PORT:-29761}" &
  local lane0_pid=$!
  run_one "250" "25pct" lane1 "4,5,6,7" "${LANE1_RDZV_PORT:-29771}" &
  local lane1_pid=$!

  local status0=0
  local status1=0
  wait "${lane0_pid}" || status0=$?
  wait "${lane1_pid}" || status1=$?

  echo "[queue] finished=$(date -Is) lane0_status=${status0} lane1_status=${status1}"
  if (( status0 != 0 || status1 != 0 )); then
    return 1
  fi
}

main "$@"
