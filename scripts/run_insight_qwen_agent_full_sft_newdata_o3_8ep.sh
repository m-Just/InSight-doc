#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

TRAIN_FILE_1="${TRAIN_FILE_1:-/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet}"
TRAIN_FILE_2="${TRAIN_FILE_2:-/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet}"
TRAIN_FILE_3="${TRAIN_FILE_3:-/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet}"
VAL_FILE="${VAL_FILE:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_eval_resume48_pathimg_mp/converted_val/train.parquet}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
WORK_DIR="${WORK_DIR:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_newdata_o3_8ep}"
EXP_NAME="${EXP_NAME:-insight_qwen_agent_full_sft_newdata_o3_8ep}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-65536}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
RDZV_PORT="${RDZV_PORT:-29501}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console','wandb']}"
LOG_PATH="${LOG_PATH:-${WORK_DIR}/train.log}"

mkdir -p "${WORK_DIR}/sft_checkpoints"
mkdir -p "$(dirname "${LOG_PATH}")"

exec > >(tee -a "${LOG_PATH}") 2>&1

read -r TRAIN_ROWS VAL_ROWS <<EOF
$(TRAIN_FILE_1="${TRAIN_FILE_1}" TRAIN_FILE_2="${TRAIN_FILE_2}" TRAIN_FILE_3="${TRAIN_FILE_3}" VAL_FILE="${VAL_FILE}" python - <<'PY'
import os
import pyarrow.parquet as pq

train_files = [
    os.environ["TRAIN_FILE_1"],
    os.environ["TRAIN_FILE_2"],
    os.environ["TRAIN_FILE_3"],
]
val_file = os.environ["VAL_FILE"]
print(sum(pq.read_metadata(path).num_rows for path in train_files), pq.read_metadata(val_file).num_rows)
PY
)
EOF

if (( TRAIN_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

STEPS_PER_EPOCH=$(( TRAIN_ROWS / TRAIN_BATCH_SIZE ))
if (( STEPS_PER_EPOCH < 1 )); then
  STEPS_PER_EPOCH=1
fi

cd "${REPO_ROOT}"

CMD=(
  torchrun
  --nnodes=1
  --node_rank=0
  --nproc-per-node="${NPROC_PER_NODE}"
  --rdzv_backend=c10d
  --rdzv_endpoint="127.0.0.1:${RDZV_PORT}"
  -m
  verl.trainer.sft_trainer
  "data.train_files=['${TRAIN_FILE_1}','${TRAIN_FILE_2}','${TRAIN_FILE_3}']"
  "data.val_files=${VAL_FILE}"
  "data.train_max_samples=${TRAIN_ROWS}"
  "data.val_max_samples=${VAL_ROWS}"
  "data.messages_key=messages"
  "data.tools_key=tools"
  "+data.message_loss_mask_key=message_loss_mask"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}"
  "data.use_dynamic_bsz=False"
  "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
  "data.max_length=${MAX_LENGTH}"
  "data.pad_mode=no_padding"
  "data.truncation=error"
  "model.path=${BASE_MODEL}"
  "model.use_remove_padding=True"
  "optim.lr=${LEARNING_RATE}"
  "engine=fsdp"
  "optim=fsdp"
  "trainer.logger=${TRAINER_LOGGERS}"
  "trainer.project_name=insight_doc"
  "trainer.experiment_name=${EXP_NAME}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.test_freq=${STEPS_PER_EPOCH}"
  "trainer.save_freq=${STEPS_PER_EPOCH}"
  "trainer.default_local_dir=${WORK_DIR}/sft_checkpoints"
  "trainer.resume_mode=${RESUME_MODE}"
  "checkpoint.save_contents=[model,optimizer,extra,hf_model]"
)

if [[ -n "${RESUME_FROM_PATH}" ]]; then
  CMD+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

printf '\n[run] '
printf '%q ' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"
