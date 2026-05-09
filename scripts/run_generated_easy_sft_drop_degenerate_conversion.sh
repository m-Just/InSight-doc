#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-processed_drop_degenerate}"
LOG_ROOT="${LOG_ROOT:-${BASE_DIR}/_conversion_logs/easy_drop_degenerate_$(date +%Y%m%d_%H%M%S)}"
REPORT_ROOT="${REPORT_ROOT:-${BASE_DIR}/_quality_reports/easy_drop_degenerate_$(date +%Y%m%d_%H%M%S)}"

CONVERT_NUM_WORKERS="${CONVERT_NUM_WORKERS:-8}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
MARK_ONLY="${MARK_ONLY:-0}"
CONVERT_ONLY="${CONVERT_ONLY:-0}"
LIMIT_DATASETS="${LIMIT_DATASETS:-0}"
DATASET_FILTER="${DATASET_FILTER:-}"

DEGENERATE_MAX_ASSISTANT_CHARS="${DEGENERATE_MAX_ASSISTANT_CHARS:-50000}"
DEGENERATE_MAX_ASSISTANT_WORDS="${DEGENERATE_MAX_ASSISTANT_WORDS:-8000}"
DEGENERATE_MIN_UNIQUE_WORD_RATIO="${DEGENERATE_MIN_UNIQUE_WORD_RATIO:-0.20}"
DEGENERATE_MIN_WORDS_FOR_UNIQUE_RATIO="${DEGENERATE_MIN_WORDS_FOR_UNIQUE_RATIO:-1000}"
DEGENERATE_MAX_SAME_WORD_RUN="${DEGENERATE_MAX_SAME_WORD_RUN:-10}"
DEGENERATE_NGRAM_SIZE="${DEGENERATE_NGRAM_SIZE:-8}"
DEGENERATE_MAX_NGRAM_REPEATS="${DEGENERATE_MAX_NGRAM_REPEATS:-50}"
DEGENERATE_MIN_WORDS_FOR_NGRAM="${DEGENERATE_MIN_WORDS_FOR_NGRAM:-1000}"
DEGENERATE_PREVIEW_CHARS="${DEGENERATE_PREVIEW_CHARS:-240}"

PART2A_OLD_IMAGE_ROOT="${PART2A_OLD_IMAGE_ROOT:-/root/likaican/data/insight_doc/O3_data_0424/0426_selected_train_part2a/dpi200_aug_noaug_maxp40/pdf_image}"
PART2A_NEW_IMAGE_ROOT="${PART2A_NEW_IMAGE_ROOT:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/O3_data_0424/0426_selected_train_part2a/dpi200_aug_noaug_maxp40/pdf_image}"

CONVERTER="${CONVERTER:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py}"
MARKER="${MARKER:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/scripts/mark_bad_exported_conversations.py}"

mapfile -t RAW_DIRS < <(find "${BASE_DIR}" -mindepth 4 -maxdepth 4 \( -type l -o -type d \) -path "*/easy/raw" | sort)
if [[ "${#RAW_DIRS[@]}" -eq 0 ]]; then
  echo "No easy/raw leaves found under ${BASE_DIR}" >&2
  exit 1
fi

if [[ -n "${DATASET_FILTER}" ]]; then
  mapfile -t RAW_DIRS < <(printf '%s\n' "${RAW_DIRS[@]}" | grep -F "${DATASET_FILTER}" || true)
fi

if [[ "${LIMIT_DATASETS}" != "0" ]]; then
  mapfile -t RAW_DIRS < <(printf '%s\n' "${RAW_DIRS[@]}" | head -n "${LIMIT_DATASETS}")
fi

if [[ "${#RAW_DIRS[@]}" -eq 0 ]]; then
  echo "No datasets selected after filters" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"

echo "BASE_DIR=${BASE_DIR}"
echo "OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "CONVERT_NUM_WORKERS=${CONVERT_NUM_WORKERS}"
echo "ALLOW_OVERWRITE=${ALLOW_OVERWRITE}"
echo "DRY_RUN=${DRY_RUN}"
echo "MARK_ONLY=${MARK_ONLY}"
echo "CONVERT_ONLY=${CONVERT_ONLY}"
echo "DATASET_FILTER=${DATASET_FILTER}"
echo "LIMIT_DATASETS=${LIMIT_DATASETS}"
echo "datasets=${#RAW_DIRS[@]}"

for raw_dir in "${RAW_DIRS[@]}"; do
  leaf_dir="$(dirname "${raw_dir}")"
  rel="${leaf_dir#${BASE_DIR}/}"
  safe="${rel//\//__}"
  output_dir="${leaf_dir}/${OUTPUT_DIR_NAME}"
  log_file="${LOG_ROOT}/${safe}.log"
  report_jsonl="${REPORT_ROOT}/${safe}.jsonl"

  echo "=== ${rel} ===" | tee "${log_file}"
  echo "raw_dir=${raw_dir}" | tee -a "${log_file}"
  echo "output_dir=${output_dir}" | tee -a "${log_file}"
  echo "report_jsonl=${report_jsonl}" | tee -a "${log_file}"

  marker_cmd=(
    python "${MARKER}"
    --input-dir "${raw_dir}"
    --output-jsonl "${report_jsonl}"
    --max-assistant-chars "${DEGENERATE_MAX_ASSISTANT_CHARS}"
    --max-assistant-words "${DEGENERATE_MAX_ASSISTANT_WORDS}"
    --min-unique-word-ratio "${DEGENERATE_MIN_UNIQUE_WORD_RATIO}"
    --min-words-for-unique-ratio "${DEGENERATE_MIN_WORDS_FOR_UNIQUE_RATIO}"
    --max-same-word-run "${DEGENERATE_MAX_SAME_WORD_RUN}"
    --ngram-size "${DEGENERATE_NGRAM_SIZE}"
    --max-ngram-repeats "${DEGENERATE_MAX_NGRAM_REPEATS}"
    --min-words-for-ngram "${DEGENERATE_MIN_WORDS_FOR_NGRAM}"
    --preview-chars "${DEGENERATE_PREVIEW_CHARS}"
  )

  convert_cmd=(
    python "${CONVERTER}"
    --input-dir "${raw_dir}"
    --output-dir "${output_dir}"
    --val-ratio 0.0
    --output-parquet-name sft_data.parquet
    --stitch-runtime-hints
    --only-correct-answers
    --drop-degenerate-conversations
    --num-workers "${CONVERT_NUM_WORKERS}"
    --degenerate-max-assistant-chars "${DEGENERATE_MAX_ASSISTANT_CHARS}"
    --degenerate-max-assistant-words "${DEGENERATE_MAX_ASSISTANT_WORDS}"
    --degenerate-min-unique-word-ratio "${DEGENERATE_MIN_UNIQUE_WORD_RATIO}"
    --degenerate-min-words-for-unique-ratio "${DEGENERATE_MIN_WORDS_FOR_UNIQUE_RATIO}"
    --degenerate-max-same-word-run "${DEGENERATE_MAX_SAME_WORD_RUN}"
    --degenerate-ngram-size "${DEGENERATE_NGRAM_SIZE}"
    --degenerate-max-ngram-repeats "${DEGENERATE_MAX_NGRAM_REPEATS}"
    --degenerate-min-words-for-ngram "${DEGENERATE_MIN_WORDS_FOR_NGRAM}"
    --degenerate-preview-chars "${DEGENERATE_PREVIEW_CHARS}"
  )

  if [[ "${leaf_dir}" == *"/O3_data_0424/train_part2a/easy" ]]; then
    convert_cmd+=(--rewrite-file-uri-prefix "${PART2A_OLD_IMAGE_ROOT}=${PART2A_NEW_IMAGE_ROOT}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf 'MARK:'
      printf ' %q' "${marker_cmd[@]}"
      printf '\n'
      printf 'CONVERT:'
      printf ' %q' "${convert_cmd[@]}"
      printf '\n'
    } | tee -a "${log_file}"
    continue
  fi

  if [[ "${CONVERT_ONLY}" != "1" ]]; then
    (
      set -x
      "${marker_cmd[@]}"
    ) 2>&1 | tee -a "${log_file}"
  fi

  if [[ "${MARK_ONLY}" == "1" ]]; then
    continue
  fi

  if [[ -e "${output_dir}/sft_data.parquet" && "${ALLOW_OVERWRITE}" != "1" ]]; then
    echo "Skipping existing ${output_dir}/sft_data.parquet; set ALLOW_OVERWRITE=1 to regenerate" | tee -a "${log_file}"
    continue
  fi

  (
    set -x
    "${convert_cmd[@]}"
  ) 2>&1 | tee -a "${log_file}"
done

echo "done"
echo "LOG_ROOT=${LOG_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
