#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

: "${TRAIN_FILES:?Set TRAIN_FILES to a JSON list or comma-separated list of SFT parquet files.}"
: "${VAL_FILES:?Set VAL_FILES to a JSON list or comma-separated list of validation parquet files.}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/sft}"
EXP_NAME="${EXP_NAME:-insight_doc_sft_qwen3vl8b}"
WORK_DIR="${WORK_DIR:-$OUTPUT_ROOT/$EXP_NAME}"
LOG_PATH="${LOG_PATH:-$WORK_DIR/train.log}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RDZV_PORT="${RDZV_PORT:-29500}"

MAX_LENGTH="${MAX_LENGTH:-65536}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MIN_LR="${MIN_LR:-5e-7}"
ENGINE_MODEL_DTYPE="${ENGINE_MODEL_DTYPE:-fp32}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console']}"
TEST_FREQ="${TEST_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
REFRESH_FREQ="${REFRESH_FREQ:-50}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
ALLOW_OVERLENGTH="${ALLOW_OVERLENGTH:-False}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-False}"
MESSAGE_LOSS_MASK_KEY="${MESSAGE_LOSS_MASK_KEY:-message_loss_mask}"

mkdir -p "$WORK_DIR/sft_checkpoints"

min_lr_ratio="$($PYTHON_BIN - <<PY_INNER
print(float("$MIN_LR") / float("$LEARNING_RATE"))
PY_INNER
)"

hydra_args=(
  "data.train_files=$TRAIN_FILES"
  "data.val_files=$VAL_FILES"
  "data.messages_key=messages"
  "data.tools_key=tools"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU"
  "data.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
  "data.use_dynamic_bsz=$USE_DYNAMIC_BSZ"
  "data.max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU"
  "data.max_length=$MAX_LENGTH"
  "data.allow_overlength=$ALLOW_OVERLENGTH"
  "data.pad_mode=no_padding"
  "data.truncation=error"
  "model.path=$BASE_MODEL"
  "model.use_remove_padding=True"
  "model.freeze_vision_tower=True"
  "model.enable_activation_offload=$ENABLE_ACTIVATION_OFFLOAD"
  "model.lora_rank=0"
  "optim.lr=$LEARNING_RATE"
  "optim.lr_scheduler_type=cosine"
  "optim.lr_warmup_steps_ratio=0.05"
  "optim.min_lr_ratio=$min_lr_ratio"
  "engine=fsdp"
  "optim=fsdp"
  "engine.model_dtype=$ENGINE_MODEL_DTYPE"
  "engine.ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE"
  "trainer.logger=$TRAINER_LOGGERS"
  "trainer.project_name=insight_doc"
  "trainer.experiment_name=$EXP_NAME"
  "trainer.total_epochs=$TOTAL_EPOCHS"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.save_freq=$SAVE_FREQ"
  "+trainer.refresh_freq=$REFRESH_FREQ"
  "trainer.default_local_dir=$WORK_DIR/sft_checkpoints"
  "trainer.resume_mode=$RESUME_MODE"
  "trainer.resume_from_path=$RESUME_FROM_PATH"
  "checkpoint.save_contents=[model,optimizer,extra,hf_model]"
)

if [[ -n "$MESSAGE_LOSS_MASK_KEY" ]]; then
  hydra_args+=("+data.message_loss_mask_key=$MESSAGE_LOSS_MASK_KEY")
fi

echo "[sft] repo=$REPO_ROOT"
echo "[sft] work_dir=$WORK_DIR"
echo "[sft] train_files=$TRAIN_FILES"
echo "[sft] val_files=$VAL_FILES"
echo "[sft] max_length=$MAX_LENGTH sp=$ULYSSES_SEQUENCE_PARALLEL_SIZE batch=$TRAIN_BATCH_SIZE lr=$LEARNING_RATE min_lr=$MIN_LR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc-per-node="$NPROC_PER_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="127.0.0.1:$RDZV_PORT" \
  -m verl.trainer.sft_trainer \
  "${hydra_args[@]}" \
  "$@" 2>&1 | tee -a "$LOG_PATH"
