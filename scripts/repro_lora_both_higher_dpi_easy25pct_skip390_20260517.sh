#!/usr/bin/env bash
set -euo pipefail

unset HTTPS_PROXY HTTP_PROXY ALL_PROXY NO_PROXY https_proxy http_proxy all_proxy no_proxy

REPO_ROOT="${REPO_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl}"
GENERATED_BASE_DIR="${GENERATED_BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
EASY_SAMPLE_ROOT="${EASY_SAMPLE_ROOT:-${REPO_ROOT}/notes/generated/easy_fraction_samples_20260516}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_easy25pct_skip390_repro_20260517}"
LOG_PATH="${OUTPUT_ROOT}/train.log"
STATUS_TSV="${OUTPUT_ROOT}/status.tsv"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
VAL_FILE="${VAL_FILE:-${GENERATED_BASE_DIR}/arxiv/val_sample_102/medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet}"

MAX_LENGTH="${MAX_LENGTH:-65536}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MIN_LR="${MIN_LR:-2e-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-False}"
ALLOW_OVERLENGTH="${ALLOW_OVERLENGTH:-False}"
DEBUG_SKIP_TRAIN_STEPS="${DEBUG_SKIP_TRAIN_STEPS:-390}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
RDZV_PORT="${RDZV_PORT:-29871}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console']}"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-${REPO_ROOT}}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SFT_BATCH_SNAPSHOT_DIR="${SFT_BATCH_SNAPSHOT_DIR:-${OUTPUT_ROOT}/batch_snapshots}"
export SFT_BATCH_SNAPSHOT_STEPS="${SFT_BATCH_SNAPSHOT_STEPS:-391,400,407,408}"
export SFT_BATCH_SNAPSHOT_RANKS="${SFT_BATCH_SNAPSHOT_RANKS:-all}"

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
  MANIFEST="${EASY_SAMPLE_ROOT}/manifest.tsv" python - <<'PY'
import csv
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
with manifest.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["fraction_label"] == "250":
            print(row["output"])
PY
}

build_train_files() {
  build_medium_files
  build_easy_files
}

main() {
  cd "${REPO_ROOT}"
  local train_files_json train_rows val_rows steps_per_epoch test_freq save_freq min_lr_ratio_value start_ts end_ts status
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
  test_freq=$(( steps_per_epoch / 4 ))
  save_freq="${steps_per_epoch}"
  local exp_id="repro_easy25pct_skip${DEBUG_SKIP_TRAIN_STEPS}_lora_both_w_higher_dpi_len${MAX_LENGTH}_bs${TRAIN_BATCH_SIZE}"

  echo "time	status	train_rows	log_path" > "${STATUS_TSV}"
  echo "[repro] start=$(date -Is)"
  echo "[repro] cuda_devices=${CUDA_DEVICES} rdzv_port=${RDZV_PORT}"
  echo "[repro] train_rows=${train_rows} val_rows=${val_rows} steps_per_epoch=${steps_per_epoch}"
  echo "[repro] debug_skip_train_steps=${DEBUG_SKIP_TRAIN_STEPS}; previous kill was near step 407"
  echo "[repro] snapshots=${SFT_BATCH_SNAPSHOT_DIR} steps=${SFT_BATCH_SNAPSHOT_STEPS}"
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
    "trainer.default_local_dir=${OUTPUT_ROOT}/sft_checkpoints" \
    "trainer.resume_mode=disable" \
    "trainer.debug_skip_train_steps=${DEBUG_SKIP_TRAIN_STEPS}" \
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]"
  status=$?
  set -e
  end_ts="$(date +%s)"
  echo "[repro] end=$(date -Is) status=${status} elapsed_seconds=$((end_ts - start_ts))"
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${status}" "${train_rows}" "${LOG_PATH}" >> "${STATUS_TSV}"
  return "${status}"
}

main "$@"
