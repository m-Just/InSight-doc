#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-raw_unanswerable_wrong_gpt5_nano_rewrite}"
MODEL="${MODEL:-gpt-5-nano}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://globalai.vip/v1}"
OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT:-120}"
API_LOGGER_SAVE_DIR="${API_LOGGER_SAVE_DIR:-${HOME}/.dumps/api_requests}"
API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME:-generated_medium_unanswerable_wrong_rewrite}"
LOG_ROOT="${LOG_ROOT:-${BASE_DIR}/_rewrite_logs/unanswerable_wrong_$(date +%Y%m%d_%H%M%S)}"

CONCURRENCY="${CONCURRENCY:-4}"
MAX_RETRIES="${MAX_RETRIES:-6}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"
MAX_API_FAILURE_RATIO="${MAX_API_FAILURE_RATIO:-0.10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

ARXIV_POSTPROCESS_ROOT="${ARXIV_POSTPROCESS_ROOT:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess}"
O3_ROOT="${O3_ROOT:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"

DATASETS=(
  "O3_data_0424/train_part1/medium|${BASE_DIR}/O3_data_0424/train_part1/medium/raw|${O3_ROOT}/0426_selected_train_part1/dpi200_aug_noaug_maxp40"
  "O3_data_0424/train_part2a/medium|${BASE_DIR}/O3_data_0424/train_part2a/medium/raw|${O3_ROOT}/0426_selected_train_part2a/dpi200_aug_noaug_maxp40"
  "O3_data_0424/train_part2b/medium|${BASE_DIR}/O3_data_0424/train_part2b/medium/raw|${O3_ROOT}/0426_selected_train_part2b/dpi200_aug_noaug_maxp40"
  "O3_data_0424/train_part2c/medium|${BASE_DIR}/O3_data_0424/train_part2c/medium/raw|${O3_ROOT}/0426_selected_train_part2c/dpi200_aug_noaug_maxp40"
  "O3_data_0424/dude_poster_unanswerable/medium|${BASE_DIR}/O3_data_0424/dude_poster_unanswerable/medium/raw|${O3_ROOT}/dude_poster_unanswerable/dpi200_aug_noaug_maxp40"
  "arxiv/train_part1/medium|${BASE_DIR}/arxiv/train_part1/medium/raw|${ARXIV_POSTPROCESS_ROOT}/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
  "arxiv/train_part2/medium|${BASE_DIR}/arxiv/train_part2/medium/raw|${ARXIV_POSTPROCESS_ROOT}/veqa_batch_0350_r2_train_mveqa_batch_0352_r2_train"
  "arxiv/train_part3/medium|${BASE_DIR}/arxiv/train_part3/medium/raw|${ARXIV_POSTPROCESS_ROOT}/veqa_batch_0350_r2_train_6508_additional_mveqa_batch_0352_r2_train_6508_additional"
  "arxiv/spanning_train_part1/medium|${BASE_DIR}/arxiv/spanning_train_part1/medium/raw|${ARXIV_POSTPROCESS_ROOT}/spanning_veqa_demo_0014_compound_from_spanning_demo_0014_compound_spanning"
  "arxiv/val_sample_102/medium|${BASE_DIR}/arxiv/val_sample_102/medium/raw|${ARXIV_POSTPROCESS_ROOT}/veqa_batch_0350_mveqa_batch_0352"
)

echo "BASE_DIR=${BASE_DIR}"
echo "OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME}"
echo "MODEL=${MODEL}"
echo "CONCURRENCY=${CONCURRENCY}"
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "API_LOGGER_SAVE_DIR=${API_LOGGER_SAVE_DIR}"
echo "API_LOGGER_PROJECT_NAME=${API_LOGGER_PROJECT_NAME}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "datasets=${#DATASETS[@]}"

for spec in "${DATASETS[@]}"; do
  IFS="|" read -r rel raw_dir reference_dir <<<"${spec}"
  output_dir="$(dirname "${raw_dir}")/${OUTPUT_DIR_NAME}"
  safe="${rel//\//__}"
  log_file="${LOG_ROOT}/${safe}.log"

  if [[ ! -d "${raw_dir}" ]]; then
    echo "missing raw dir: ${raw_dir}" | tee "${log_file}"
    exit 1
  fi
  if [[ ! -d "${reference_dir}" ]]; then
    echo "missing reference dir: ${reference_dir}" | tee "${log_file}"
    exit 1
  fi
  if [[ "${ALLOW_OVERWRITE}" == "1" ]]; then
    rm -rf "${output_dir}"
  fi

  echo "=== ${rel} ===" | tee "${log_file}"
  (
    set -x
    OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
    OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT}" \
    ENSURE_API_LOGGER=1 \
    API_LOGGER_SAVE_DIR="${API_LOGGER_SAVE_DIR}" \
    API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME}" \
    python /scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/rewrite_unanswerable_wrong_exported_convos_with_api.py \
      --input-dir "${raw_dir}" \
      --output-dir "${output_dir}" \
      --postprocess-dir "${reference_dir}" \
      --model "${MODEL}" \
      --timeout "${OPENAI_CLIENT_TIMEOUT}" \
      --max-retries "${MAX_RETRIES}" \
      --max-completion-tokens "${MAX_COMPLETION_TOKENS}" \
      --concurrency "${CONCURRENCY}" \
      --copy-unmodified \
      --original-match-filter-mode precision \
      --max-api-failure-ratio "${MAX_API_FAILURE_RATIO}" \
      --api-logger-save-dir "${API_LOGGER_SAVE_DIR}" \
      --api-logger-project-name "${API_LOGGER_PROJECT_NAME}" \
      --progress-every "${PROGRESS_EVERY}"
  ) 2>&1 | tee -a "${log_file}"
done

echo "done"
echo "LOG_ROOT=${LOG_ROOT}"
