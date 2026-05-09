#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-processed_gpt5_nano_rewrite}"
REWRITE_DIR_NAME="${REWRITE_DIR_NAME:-raw_gpt5_nano_rewrite}"
MODEL="${MODEL:-gpt-5-nano}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://az.gptplus5.com/v1}"
OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT:-120}"
API_LOGGER_SAVE_DIR="${API_LOGGER_SAVE_DIR:-${HOME}/.dumps/api_requests}"
API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME:-medium_final_answer_rewrite_gpt5_nano}"
INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT:-/scratch/ywxzml3j/likaican/src/InSight-doc}"
LOG_ROOT="${LOG_ROOT:-${BASE_DIR}/_conversion_logs/medium_gpt5_nano_rewrite_$(date +%Y%m%d_%H%M%S)}"

REWRITE_CONCURRENCY="${REWRITE_CONCURRENCY:-8}"
CONVERT_NUM_WORKERS="${CONVERT_NUM_WORKERS:-8}"
MAX_RETRIES="${MAX_RETRIES:-4}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
MAX_FAILURE_RATIO="${MAX_FAILURE_RATIO:-0.005}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

PART2A_OLD_IMAGE_ROOT="${PART2A_OLD_IMAGE_ROOT:-/root/likaican/data/insight_doc/O3_data_0424/0426_selected_train_part2a/dpi200_aug_noaug_maxp40/pdf_image}"
PART2A_NEW_IMAGE_ROOT="${PART2A_NEW_IMAGE_ROOT:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424/0426_selected_train_part2a/dpi200_aug_noaug_maxp40/pdf_image}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"

mapfile -t RAW_DIRS < <(find "${BASE_DIR}" -mindepth 4 -maxdepth 4 -type l -path "*/medium/raw" | sort)
if [[ "${#RAW_DIRS[@]}" -eq 0 ]]; then
  echo "No medium/raw symlink leaves found under ${BASE_DIR}" >&2
  exit 1
fi

echo "BASE_DIR=${BASE_DIR}"
echo "OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME}"
echo "REWRITE_DIR_NAME=${REWRITE_DIR_NAME}"
echo "MODEL=${MODEL}"
echo "REWRITE_CONCURRENCY=${REWRITE_CONCURRENCY}"
echo "CONVERT_NUM_WORKERS=${CONVERT_NUM_WORKERS}"
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "API_LOGGER_SAVE_DIR=${API_LOGGER_SAVE_DIR}"
echo "API_LOGGER_PROJECT_NAME=${API_LOGGER_PROJECT_NAME}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "datasets=${#RAW_DIRS[@]}"

for raw_dir in "${RAW_DIRS[@]}"; do
  leaf_dir="$(dirname "${raw_dir}")"
  rel="${leaf_dir#${BASE_DIR}/}"
  output_dir="${leaf_dir}/${OUTPUT_DIR_NAME}"
  rewrite_dir="${leaf_dir}/${REWRITE_DIR_NAME}"
  safe="${rel//\//__}"
  log_file="${LOG_ROOT}/${safe}.log"

  if [[ -e "${output_dir}/sft_data.parquet" && "${ALLOW_OVERWRITE}" != "1" ]]; then
    echo "Skipping existing ${output_dir}/sft_data.parquet; set ALLOW_OVERWRITE=1 to regenerate" | tee "${log_file}"
    continue
  fi

  cmd=(
    python /scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py
    --input-dir "${raw_dir}"
    --output-dir "${output_dir}"
    --val-ratio 0.0
    --output-parquet-name sft_data.parquet
    --stitch-runtime-hints
    --only-correct-answers
    --drop-degenerate-conversations
    --final-answer-rewrite-mode api
    --final-answer-rewrite-output-dir "${rewrite_dir}"
    --final-answer-rewrite-model "${MODEL}"
    --final-answer-rewrite-concurrency "${REWRITE_CONCURRENCY}"
    --final-answer-rewrite-timeout "${OPENAI_CLIENT_TIMEOUT}"
    --final-answer-rewrite-max-retries "${MAX_RETRIES}"
    --final-answer-rewrite-max-completion-tokens "${MAX_COMPLETION_TOKENS}"
    --final-answer-rewrite-max-failure-ratio "${MAX_FAILURE_RATIO}"
    --final-answer-rewrite-progress-every "${PROGRESS_EVERY}"
    --final-answer-rewrite-openai-base-url "${OPENAI_BASE_URL}"
    --api-logger-save-dir "${API_LOGGER_SAVE_DIR}"
    --api-logger-project-name "${API_LOGGER_PROJECT_NAME}"
    --insight-doc-root "${INSIGHT_DOC_ROOT}"
    --num-workers "${CONVERT_NUM_WORKERS}"
  )

  if [[ "${leaf_dir}" == *"/O3_data_0424/train_part2a/medium" ]]; then
    cmd+=(--rewrite-file-uri-prefix "${PART2A_OLD_IMAGE_ROOT}=${PART2A_NEW_IMAGE_ROOT}")
  fi

  echo "=== ${rel} ===" | tee "${log_file}"
  (
    set -x
    OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
    OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT}" \
    ENSURE_API_LOGGER=1 \
    API_LOGGER_SAVE_DIR="${API_LOGGER_SAVE_DIR}" \
    API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME}" \
    INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT}" \
    "${cmd[@]}"
  ) 2>&1 | tee -a "${log_file}"
done

echo "done"
echo "LOG_ROOT=${LOG_ROOT}"
