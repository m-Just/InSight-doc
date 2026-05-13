#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/run_generated_easy_sft_drop_degenerate_for_sources.sh PART=EXPORTED_DIR [...]

Examples:
  scripts/run_generated_easy_sft_drop_degenerate_for_sources.sh \
    train_part3b=/home/.../qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part3b/exported_conversations \
    train_part3c=/home/.../qwen3-vl-32b-instruct/O3_data_0424-dpi200_aug_noaug_maxp40/train_part3c/exported_conversations

  DATASET_GROUP=O3_data_0507 scripts/run_generated_easy_sft_drop_degenerate_for_sources.sh \
    train_part4=/home/.../exported_conversations

  scripts/run_generated_easy_sft_drop_degenerate_for_sources.sh \
    O3_data_0507/train_part4=/home/.../exported_conversations

Each PART=EXPORTED_DIR creates/updates:
  ${GENERATED_BASE_DIR}/${DATASET_GROUP_OR_ARG}/PART/easy/raw -> EXPORTED_DIR
  easy/processed_drop_degenerate/sft_data.parquet
  easy/processed_drop_degenerate/wrong_question_ids.txt

Environment:
  GENERATED_BASE_DIR       default: /home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated
  DATASET_GROUP            default: O3_data_0424; used when PART has no slash
  OUTPUT_DIR_NAME          default: processed_drop_degenerate
  CONVERT_NUM_WORKERS      default: 8
  ALLOW_OVERWRITE          default: 0; set 1 to regenerate existing parquet
  MARK_ONLY                default: 0
  CONVERT_ONLY             default: 0
  REWRITE_FILE_URI_PREFIXES optional; comma-separated OLD=NEW mappings passed to conversion

Degenerate-filter thresholds can be overridden with:
  DEGENERATE_MAX_ASSISTANT_CHARS, DEGENERATE_MAX_ASSISTANT_WORDS,
  DEGENERATE_MIN_UNIQUE_WORD_RATIO, DEGENERATE_MIN_WORDS_FOR_UNIQUE_RATIO,
  DEGENERATE_MAX_SAME_WORD_RUN, DEGENERATE_NGRAM_SIZE,
  DEGENERATE_MAX_NGRAM_REPEATS, DEGENERATE_MIN_WORDS_FOR_NGRAM,
  DEGENERATE_PREVIEW_CHARS
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONVERTER="${CONVERTER:-${REPO_ROOT}/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py}"
MARKER="${MARKER:-${REPO_ROOT}/scripts/mark_bad_exported_conversations.py}"

GENERATED_BASE_DIR="${GENERATED_BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
DATASET_GROUP="${DATASET_GROUP:-O3_data_0424}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-processed_drop_degenerate}"
LOG_ROOT="${LOG_ROOT:-${GENERATED_BASE_DIR}/_conversion_logs/easy_drop_degenerate_sources_$(date +%Y%m%d_%H%M%S)}"
REPORT_ROOT="${REPORT_ROOT:-${GENERATED_BASE_DIR}/_quality_reports/easy_drop_degenerate_sources_$(date +%Y%m%d_%H%M%S)}"

CONVERT_NUM_WORKERS="${CONVERT_NUM_WORKERS:-8}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
MARK_ONLY="${MARK_ONLY:-0}"
CONVERT_ONLY="${CONVERT_ONLY:-0}"
REWRITE_FILE_URI_PREFIXES="${REWRITE_FILE_URI_PREFIXES:-}"

DEGENERATE_MAX_ASSISTANT_CHARS="${DEGENERATE_MAX_ASSISTANT_CHARS:-50000}"
DEGENERATE_MAX_ASSISTANT_WORDS="${DEGENERATE_MAX_ASSISTANT_WORDS:-8000}"
DEGENERATE_MIN_UNIQUE_WORD_RATIO="${DEGENERATE_MIN_UNIQUE_WORD_RATIO:-0.20}"
DEGENERATE_MIN_WORDS_FOR_UNIQUE_RATIO="${DEGENERATE_MIN_WORDS_FOR_UNIQUE_RATIO:-1000}"
DEGENERATE_MAX_SAME_WORD_RUN="${DEGENERATE_MAX_SAME_WORD_RUN:-10}"
DEGENERATE_NGRAM_SIZE="${DEGENERATE_NGRAM_SIZE:-8}"
DEGENERATE_MAX_NGRAM_REPEATS="${DEGENERATE_MAX_NGRAM_REPEATS:-50}"
DEGENERATE_MIN_WORDS_FOR_NGRAM="${DEGENERATE_MIN_WORDS_FOR_NGRAM:-1000}"
DEGENERATE_PREVIEW_CHARS="${DEGENERATE_PREVIEW_CHARS:-240}"

mkdir -p "${LOG_ROOT}" "${REPORT_ROOT}"

echo "GENERATED_BASE_DIR=${GENERATED_BASE_DIR}"
echo "DATASET_GROUP=${DATASET_GROUP}"
echo "OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "REPORT_ROOT=${REPORT_ROOT}"
echo "CONVERT_NUM_WORKERS=${CONVERT_NUM_WORKERS}"
echo "ALLOW_OVERWRITE=${ALLOW_OVERWRITE}"
echo "MARK_ONLY=${MARK_ONLY}"
echo "CONVERT_ONLY=${CONVERT_ONLY}"

rewrite_prefix_args=()
if [[ -n "${REWRITE_FILE_URI_PREFIXES}" ]]; then
  IFS=',' read -r -a prefix_mappings <<< "${REWRITE_FILE_URI_PREFIXES}"
  for mapping in "${prefix_mappings[@]}"; do
    if [[ -n "${mapping}" ]]; then
      rewrite_prefix_args+=(--rewrite-file-uri-prefix "${mapping}")
    fi
  done
fi

for spec in "$@"; do
  if [[ "${spec}" != *=* ]]; then
    echo "Invalid source spec, expected PART=EXPORTED_DIR: ${spec}" >&2
    exit 2
  fi

  rel="${spec%%=*}"
  source_dir="${spec#*=}"
  if [[ -z "${rel}" || -z "${source_dir}" ]]; then
    echo "Invalid empty source spec: ${spec}" >&2
    exit 2
  fi
  if [[ "${rel}" != */* ]]; then
    rel="${DATASET_GROUP}/${rel}"
  fi
  if [[ ! -d "${source_dir}" ]]; then
    echo "Source directory does not exist: ${source_dir}" >&2
    exit 1
  fi

  leaf_dir="${GENERATED_BASE_DIR}/${rel}/easy"
  raw_dir="${leaf_dir}/raw"
  output_dir="${leaf_dir}/${OUTPUT_DIR_NAME}"
  safe="${rel//\//__}"
  log_file="${LOG_ROOT}/${safe}.log"
  report_jsonl="${REPORT_ROOT}/${safe}.jsonl"

  mkdir -p "${leaf_dir}"
  ln -sfn "${source_dir}" "${raw_dir}"
  echo "Linked ${raw_dir} -> ${source_dir}" | tee "${log_file}"
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
    "${rewrite_prefix_args[@]}"
  )

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
