#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

TRAIN_FILES=(
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part1/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part1/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2a/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2a/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2b/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2b/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2c/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2c/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/dude_poster_unanswerable/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/dude_poster_unanswerable/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part1/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part1/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part2/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part2/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part3/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part3/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part4/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part4/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part5/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part5/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/spanning_train_part1/easy/processed_drop_degenerate/sft_data.parquet"
  "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/spanning_train_part1/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
)

# "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3a/easy/processed_drop_degenerate/sft_data.parquet"
# "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3a/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
# "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3b/easy/processed_drop_degenerate/sft_data.parquet"
# "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3b/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
# "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3c/easy/processed_drop_degenerate/sft_data.parquet"
# "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3c/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
# "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3d/easy/processed_drop_degenerate/sft_data.parquet"
# "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part3d/medium/processed_gpt5_nano_rewrite/sft_data.parquet"

USE_VSEARCHER_SYSTEM_FOR_EASY="${USE_VSEARCHER_SYSTEM_FOR_EASY:-0}"
if [[ "${USE_VSEARCHER_SYSTEM_FOR_EASY}" == "1" || "${USE_VSEARCHER_SYSTEM_FOR_EASY}" == "true" || "${USE_VSEARCHER_SYSTEM_FOR_EASY}" == "yes" ]]; then
  for idx in "${!TRAIN_FILES[@]}"; do
    if [[ "${TRAIN_FILES[$idx]}" == */easy/processed_drop_degenerate/sft_data.parquet ]]; then
      TRAIN_FILES[$idx]="${TRAIN_FILES[$idx]%/sft_data.parquet}/sft_data_with_vsearcher_system.parquet"
    fi
  done
fi

USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM="${USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM:-0}"
if [[ "${USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM}" == "1" || "${USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM}" == "true" || "${USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM}" == "yes" ]]; then
  for idx in "${!TRAIN_FILES[@]}"; do
    if [[ "${TRAIN_FILES[$idx]}" == */medium/processed_gpt5_nano_rewrite/sft_data.parquet ]]; then
      TRAIN_FILES[$idx]="${TRAIN_FILES[$idx]%/sft_data.parquet}/sft_data_base_model_tool_argument_order.parquet"
    fi
  done
fi

TRAIN_MEDIUM_ONLY="${TRAIN_MEDIUM_ONLY:-0}"
if [[ "${TRAIN_MEDIUM_ONLY}" == "1" || "${TRAIN_MEDIUM_ONLY}" == "true" || "${TRAIN_MEDIUM_ONLY}" == "yes" ]]; then
  MEDIUM_TRAIN_FILES=()
  for path in "${TRAIN_FILES[@]}"; do
    if [[ "${path}" == */medium/* ]]; then
      MEDIUM_TRAIN_FILES+=("${path}")
    fi
  done
  TRAIN_FILES=("${MEDIUM_TRAIN_FILES[@]}")
fi

# TRAIN_FILES=(
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part1/easy/processed_drop_degenerate/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part1/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2b/easy/processed_drop_degenerate/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2b/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2c/easy/processed_drop_degenerate/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/train_part2c/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/dude_poster_unanswerable/easy/processed_drop_degenerate/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/O3_data_0424/dude_poster_unanswerable/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part1/easy/processed_drop_degenerate/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part1/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part2/easy/processed_drop_degenerate/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/train_part2/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/spanning_train_part1/easy/processed_drop_degenerate/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/spanning_train_part1/medium/processed_gpt5_nano_rewrite/sft_data.parquet"
# )

# Old TRAIN_FILES before generated/drop-degenerate + rewrite wiring. Kept for reference.
# TRAIN_FILES=(
#   "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet"
#   "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part2b/converted_sft/sft_data.parquet"
#   "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2b_resumable/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part2c/converted_sft/sft_data.parquet"
#   "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part2c_resumable/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_resumable/qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/dude_poster_unanswerable/converted_sft/sft_data.parquet"
#   "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/O3_data_0424-dpi200_aug_noaug_maxp40-dude_poster_unanswerable_resumable/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet"
#   "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40/train_part2/converted_sft/sft_data.parquet"
#   "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train-dpi200_aug_noaug_maxp40-0426_train_part2_resumable/sft_data.parquet"
#   "/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs/insight_qwen_agent_zoom_factor2_default_sys_0426_resumable/qwen3-vl-32b-instruct/spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning-dpi200_aug_noaug_maxp40/train_part1/converted_sft/sft_data.parquet"
#   "/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed/spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning-dpi200_aug_noaug_maxp40-0426_train_part1_resumable/sft_data.parquet"
# )

echo TRAIN_FILES: "${TRAIN_FILES[@]}"

# VAL_FILE="${VAL_FILE:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_eval_resume48_pathimg_mp/converted_val/train.parquet}"
VAL_FILE="${VAL_FILE:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/arxiv/val_sample_102/medium/processed_gpt5_nano_rewrite/sft_data.parquet}"
if [[ "${USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM}" == "1" || "${USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM}" == "true" || "${USE_BASE_MODEL_TOOL_ARGUMENT_ORDER_FOR_MEDIUM}" == "yes" ]]; then
  if [[ "${VAL_FILE}" == */medium/processed_gpt5_nano_rewrite/sft_data.parquet ]]; then
    VAL_FILE="${VAL_FILE%/sft_data.parquet}/sft_data_base_model_tool_argument_order.parquet"
  fi
fi

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
TRAINER_LOGGERS="${TRAINER_LOGGERS:-['console']}"
LOG_PATH="${LOG_PATH:-${WORK_DIR}/train.log}"
EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS:-}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-0}"
if [[ "${FREEZE_VISION_TOWER}" == "1" || "${FREEZE_VISION_TOWER}" == "true" || "${FREEZE_VISION_TOWER}" == "yes" ]]; then
  EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS} model.freeze_vision_tower=True"
fi
SFT_BATCH_SNAPSHOT_DIR="${SFT_BATCH_SNAPSHOT_DIR:-}"
if [[ -n "${SFT_BATCH_SNAPSHOT_DIR}" ]]; then
  SFT_BATCH_SNAPSHOT_STEPS="${SFT_BATCH_SNAPSHOT_STEPS:-1}"
  SFT_BATCH_SNAPSHOT_MAX_SAMPLES="${SFT_BATCH_SNAPSHOT_MAX_SAMPLES:-4}"
  SFT_BATCH_SNAPSHOT_MAX_TOKENS_PER_SAMPLE="${SFT_BATCH_SNAPSHOT_MAX_TOKENS_PER_SAMPLE:-4096}"
  SFT_BATCH_SNAPSHOT_INCLUDE_TOKEN_TABLE="${SFT_BATCH_SNAPSHOT_INCLUDE_TOKEN_TABLE:-True}"
  EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS} trainer.batch_snapshot_dir=${SFT_BATCH_SNAPSHOT_DIR}"
  EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS} trainer.batch_snapshot_steps=\"${SFT_BATCH_SNAPSHOT_STEPS}\""
  EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS} trainer.batch_snapshot_max_samples=${SFT_BATCH_SNAPSHOT_MAX_SAMPLES}"
  EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS} trainer.batch_snapshot_max_tokens_per_sample=${SFT_BATCH_SNAPSHOT_MAX_TOKENS_PER_SAMPLE}"
  EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS} trainer.batch_snapshot_include_token_table=${SFT_BATCH_SNAPSHOT_INCLUDE_TOKEN_TABLE}"
  if [[ -n "${SFT_BATCH_SNAPSHOT_RANKS:-}" ]]; then
    EXTRA_SFT_ARGS="${EXTRA_SFT_ARGS} trainer.batch_snapshot_ranks=\"${SFT_BATCH_SNAPSHOT_RANKS}\""
  fi
fi

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
