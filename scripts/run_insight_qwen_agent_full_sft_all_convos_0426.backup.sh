#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

TRAIN_FILES=(
  "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet"
  "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part2b/converted_sft/sft_data.parquet"
  "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2b_resumable/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part2c/converted_sft/sft_data.parquet"
  "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2c_resumable/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/dude_poster_unanswerable/converted_sft/sft_data.parquet"
  "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-dude_poster_unanswerable_resumable/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet"
  "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part2/converted_sft/sft_data.parquet"
  "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part2_resumable/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet"
  "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet"
)
VAL_FILE="${VAL_FILE:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_eval_resume48_pathimg_mp/converted_val/train.parquet}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
WORK_DIR="${WORK_DIR:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426}"
EXP_NAME="${EXP_NAME:-insight_qwen_agent_full_sft_all_convos_0426}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-65536}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
RDZV_PORT="${RDZV_PORT:-29501}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console','wandb']}"
LOG_PATH="${LOG_PATH:-${WORK_DIR}/train.log}"
EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS:-}"

mkdir -p "${WORK_DIR}/sft_checkpoints"
mkdir -p "$(dirname "${LOG_PATH}")"

exec > >(tee -a "${LOG_PATH}") 2>&1

TRAIN_FILES_JSON="$(TRAIN_FILES_JOINED="$(printf '%s\n' "${TRAIN_FILES[@]}")" python - <<'PY'
import json
import os

print(json.dumps(os.environ["TRAIN_FILES_JOINED"].splitlines()))
PY
)"

read -r TRAIN_ROWS VAL_ROWS <<EOF
$(TRAIN_FILES_JSON="${TRAIN_FILES_JSON}" VAL_FILE="${VAL_FILE}" python - <<'PY'
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

if (( TRAIN_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 1
fi

STEPS_PER_EPOCH=$(( TRAIN_ROWS / TRAIN_BATCH_SIZE ))
if (( STEPS_PER_EPOCH < 1 )); then
  STEPS_PER_EPOCH=1
fi

SAVE_FREQ="${SAVE_FREQ:-${STEPS_PER_EPOCH}}"
if [[ -n "${TESTS_PER_EPOCH:-}" ]]; then
  if (( TESTS_PER_EPOCH < 1 )); then
    echo "TESTS_PER_EPOCH=${TESTS_PER_EPOCH} must be >= 1" >&2
    exit 1
  fi
  TEST_FREQ="${TEST_FREQ:-$(( STEPS_PER_EPOCH / TESTS_PER_EPOCH ))}"
  if (( TEST_FREQ < 1 )); then
    TEST_FREQ=1
  fi
else
  TEST_FREQ="${TEST_FREQ:-${STEPS_PER_EPOCH}}"
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
  "data.train_files=${TRAIN_FILES_JSON}"
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
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.default_local_dir=${WORK_DIR}/sft_checkpoints"
  "trainer.resume_mode=${RESUME_MODE}"
  "checkpoint.save_contents=[model,optimizer,extra,hf_model]"
)

if [[ -n "${RESUME_FROM_PATH}" ]]; then
  CMD+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
fi
if [[ -n "${EXTRA_SFT_ARGS}" ]]; then
  read -r -a EXTRA_SFT_ARGS_ARRAY <<< "${EXTRA_SFT_ARGS}"
  CMD+=("${EXTRA_SFT_ARGS_ARRAY[@]}")
fi

printf '\n[run] '
printf '%q ' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"
