#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-raw_unanswerable_wrong_gpt5_nano_rewrite}"
MODEL="${MODEL:-gpt-5-nano}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://globalai.vip/v1}"
OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT:-120}"
API_LOGGER_SAVE_DIR="${API_LOGGER_SAVE_DIR:-${HOME}/.dumps/api_requests}"
API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME:-generated_medium_unanswerable_wrong_retry}"
LOG_ROOT="${LOG_ROOT:-${BASE_DIR}/_rewrite_logs/unanswerable_wrong_api_retry_$(date +%Y%m%d_%H%M%S)}"
TMP_ROOT="${TMP_ROOT:-/tmp/generated_medium_unanswerable_wrong_api_retry}"

CONCURRENCY="${CONCURRENCY:-2}"
MAX_RETRIES="${MAX_RETRIES:-6}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"
MAX_API_FAILURE_RATIO="${MAX_API_FAILURE_RATIO:-1.0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"

ARXIV_POSTPROCESS_ROOT="${ARXIV_POSTPROCESS_ROOT:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/arxiv_0307_sample/qa_gen/postprocess}"
O3_ROOT="${O3_ROOT:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}" "${TMP_ROOT}"

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
echo "TMP_ROOT=${TMP_ROOT}"
echo "datasets=${#DATASETS[@]}"

for spec in "${DATASETS[@]}"; do
  IFS="|" read -r rel raw_dir reference_dir <<<"${spec}"
  output_dir="$(dirname "${raw_dir}")/${OUTPUT_DIR_NAME}"
  status_path="${output_dir}/rewrite_status.jsonl"
  cache_path="${output_dir}/rewrite_cache.jsonl"
  retry_status_path="${output_dir}/api_failure_retry_status.jsonl"
  safe="${rel//\//__}"
  retry_input_dir="${TMP_ROOT}/${safe}"
  log_file="${LOG_ROOT}/${safe}.log"

  if [[ ! -d "${output_dir}" || ! -f "${status_path}" ]]; then
    echo "Skipping ${rel}: missing output dir or rewrite_status.jsonl" | tee "${log_file}"
    continue
  fi

  rm -rf "${retry_input_dir}"
  mkdir -p "${retry_input_dir}"
  rm -f "${retry_status_path}"

  python - "${status_path}" "${retry_input_dir}" <<'PY'
import json
import os
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
retry_input_dir = Path(sys.argv[2])
count = 0
seen = set()
with status_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        obj = json.loads(line)
        if obj.get("status") != "api_failure":
            continue
        src = obj.get("path")
        rel = obj.get("relative_path")
        if not isinstance(src, str) or not isinstance(rel, str):
            continue
        if rel in seen:
            continue
        seen.add(rel)
        target = retry_input_dir / rel
        target.symlink_to(Path(src))
        count += 1
print(count)
PY

  retry_count="$(find "${retry_input_dir}" -maxdepth 1 -type l | wc -l | tr -d ' ')"
  echo "=== ${rel} ===" | tee "${log_file}"
  echo "api_failure_retry_count=${retry_count}" | tee -a "${log_file}"
  if [[ "${retry_count}" == "0" ]]; then
    continue
  fi

  (
    set -x
    OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
    OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT}" \
    ENSURE_API_LOGGER=1 \
    API_LOGGER_SAVE_DIR="${API_LOGGER_SAVE_DIR}" \
    API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME}" \
    python /scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/rewrite_unanswerable_wrong_exported_convos_with_api.py \
      --input-dir "${retry_input_dir}" \
      --output-dir "${output_dir}" \
      --postprocess-dir "${reference_dir}" \
      --model "${MODEL}" \
      --timeout "${OPENAI_CLIENT_TIMEOUT}" \
      --max-retries "${MAX_RETRIES}" \
      --max-completion-tokens "${MAX_COMPLETION_TOKENS}" \
      --concurrency "${CONCURRENCY}" \
      --original-match-filter-mode precision \
      --cache-jsonl "${cache_path}" \
      --status-jsonl "${retry_status_path}" \
      --max-api-failure-ratio "${MAX_API_FAILURE_RATIO}" \
      --api-logger-save-dir "${API_LOGGER_SAVE_DIR}" \
      --api-logger-project-name "${API_LOGGER_PROJECT_NAME}" \
      --progress-every "${PROGRESS_EVERY}"
  ) 2>&1 | tee -a "${log_file}"
done

echo "done"
echo "LOG_ROOT=${LOG_ROOT}"
