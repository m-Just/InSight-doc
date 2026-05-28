#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  OPENAI_API_KEY=... scripts/run_generated_medium_sft_gpt5_nano_rewrite_for_sources.sh PART=EXPORTED_DIR [...]

Examples:
  scripts/run_generated_medium_sft_gpt5_nano_rewrite_for_sources.sh \
    train_part3b=/scratch/.../O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3b_resumable \
    train_part3c=/scratch/.../O3_data_0424-dpi200_aug_noaug_maxp40-0426_train_part3c_resumable

  DATASET_GROUP=O3_data_0507 scripts/run_generated_medium_sft_gpt5_nano_rewrite_for_sources.sh \
    train_part4=/scratch/.../exported_conversations_part4

  scripts/run_generated_medium_sft_gpt5_nano_rewrite_for_sources.sh \
    O3_data_0507/train_part4=/scratch/.../exported_conversations_part4

Each PART=EXPORTED_DIR creates/updates:
  ${GENERATED_BASE_DIR}/${DATASET_GROUP_OR_ARG}/PART/medium/raw -> EXPORTED_DIR
  medium/processed_gpt5_nano_rewrite/sft_data.parquet
  medium/processed_gpt5_nano_rewrite/sft_data_base_model_tool_argument_order.parquet

Environment:
  GENERATED_BASE_DIR       default: /home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated
  DATASET_GROUP            default: O3_data_0424; used when PART has no slash
  MODEL                    default: gpt-5-nano
  OPENAI_BASE_URL          default: https://az.gptplus5.com/v1
  OPENAI_CLIENT_TIMEOUT    default: 120
  REWRITE_CONCURRENCY      default: 8
  CONVERT_NUM_WORKERS      default: 8
  MAX_RETRIES              default: 4
  MAX_COMPLETION_TOKENS    default: 4096
  MAX_FAILURE_RATIO        default: 0.005
  PROGRESS_EVERY           default: 50
  ALLOW_OVERWRITE          default: 0; set 1 to regenerate existing parquets
  REWRITE_FILE_URI_PREFIXES optional; comma-separated OLD=NEW mappings passed to conversion
  INVALID_IMAGE_ASPECT_POLICY default: drop; one of error,pad,drop
  MAX_IMAGE_ASPECT_RATIO   default: 200
  IMAGE_ASPECT_PAD_TARGET_RATIO default: 198
  CONVERSION_PROGRESS_EVERY default: 100
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

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONVERTER="${REPO_ROOT}/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py"

GENERATED_BASE_DIR="${GENERATED_BASE_DIR:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated}"
DATASET_GROUP="${DATASET_GROUP:-O3_data_0424}"
OUTPUT_DIR_NAME="${OUTPUT_DIR_NAME:-processed_gpt5_nano_rewrite}"
REWRITE_DIR_NAME="${REWRITE_DIR_NAME:-raw_gpt5_nano_rewrite}"
MODEL="${MODEL:-gpt-5-nano}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://az.gptplus5.com/v1}"
OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT:-120}"
API_LOGGER_SAVE_DIR="${API_LOGGER_SAVE_DIR:-${HOME}/.dumps/api_requests}"
API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME:-medium_final_answer_rewrite_gpt5_nano}"
INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT:-/scratch/ywxzml3j/likaican/src/InSight-doc}"
LOG_ROOT="${LOG_ROOT:-${GENERATED_BASE_DIR}/_conversion_logs/medium_gpt5_nano_rewrite_sources_$(date +%Y%m%d_%H%M%S)}"

REWRITE_CONCURRENCY="${REWRITE_CONCURRENCY:-8}"
CONVERT_NUM_WORKERS="${CONVERT_NUM_WORKERS:-8}"
MAX_RETRIES="${MAX_RETRIES:-4}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
MAX_FAILURE_RATIO="${MAX_FAILURE_RATIO:-0.005}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
REWRITE_FILE_URI_PREFIXES="${REWRITE_FILE_URI_PREFIXES:-}"
INVALID_IMAGE_ASPECT_POLICY="${INVALID_IMAGE_ASPECT_POLICY:-drop}"
MAX_IMAGE_ASPECT_RATIO="${MAX_IMAGE_ASPECT_RATIO:-200}"
IMAGE_ASPECT_PAD_TARGET_RATIO="${IMAGE_ASPECT_PAD_TARGET_RATIO:-198}"
CONVERSION_PROGRESS_EVERY="${CONVERSION_PROGRESS_EVERY:-100}"

mkdir -p "${LOG_ROOT}"

echo "GENERATED_BASE_DIR=${GENERATED_BASE_DIR}"
echo "DATASET_GROUP=${DATASET_GROUP}"
echo "OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME}"
echo "REWRITE_DIR_NAME=${REWRITE_DIR_NAME}"
echo "MODEL=${MODEL}"
echo "REWRITE_CONCURRENCY=${REWRITE_CONCURRENCY}"
echo "CONVERT_NUM_WORKERS=${CONVERT_NUM_WORKERS}"
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}"
echo "API_LOGGER_SAVE_DIR=${API_LOGGER_SAVE_DIR}"
echo "API_LOGGER_PROJECT_NAME=${API_LOGGER_PROJECT_NAME}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "INVALID_IMAGE_ASPECT_POLICY=${INVALID_IMAGE_ASPECT_POLICY}"
echo "MAX_IMAGE_ASPECT_RATIO=${MAX_IMAGE_ASPECT_RATIO}"
echo "IMAGE_ASPECT_PAD_TARGET_RATIO=${IMAGE_ASPECT_PAD_TARGET_RATIO}"
echo "CONVERSION_PROGRESS_EVERY=${CONVERSION_PROGRESS_EVERY}"

rewrite_prefix_args=()
if [[ -n "${REWRITE_FILE_URI_PREFIXES}" ]]; then
  IFS=',' read -r -a prefix_mappings <<< "${REWRITE_FILE_URI_PREFIXES}"
  for mapping in "${prefix_mappings[@]}"; do
    if [[ -n "${mapping}" ]]; then
      rewrite_prefix_args+=(--rewrite-file-uri-prefix "${mapping}")
    fi
  done
fi

image_aspect_args=(
  --invalid-image-aspect-policy "${INVALID_IMAGE_ASPECT_POLICY}"
  --max-image-aspect-ratio "${MAX_IMAGE_ASPECT_RATIO}"
  --image-aspect-pad-target-ratio "${IMAGE_ASPECT_PAD_TARGET_RATIO}"
)

run_convert() {
  local log_file="$1"
  shift
  (
    set -x
    OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
    OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT}" \
    ENSURE_API_LOGGER=1 \
    API_LOGGER_SAVE_DIR="${API_LOGGER_SAVE_DIR}" \
    API_LOGGER_PROJECT_NAME="${API_LOGGER_PROJECT_NAME}" \
    INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT}" \
    "$@"
  ) 2>&1 | tee -a "${log_file}"
}

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

  leaf_dir="${GENERATED_BASE_DIR}/${rel}/medium"
  raw_dir="${leaf_dir}/raw"
  output_dir="${leaf_dir}/${OUTPUT_DIR_NAME}"
  rewrite_dir="${leaf_dir}/${REWRITE_DIR_NAME}"
  safe="${rel//\//__}"

  mkdir -p "${leaf_dir}"
  ln -sfn "${source_dir}" "${raw_dir}"
  echo "Linked ${raw_dir} -> ${source_dir}"

  main_log="${LOG_ROOT}/${safe}.rewrite.log"
  if [[ -e "${output_dir}/sft_data.parquet" && "${ALLOW_OVERWRITE}" != "1" ]]; then
    echo "Skipping existing ${output_dir}/sft_data.parquet; set ALLOW_OVERWRITE=1 to regenerate" | tee "${main_log}"
  else
    echo "=== ${rel} rewrite+conversion ===" | tee "${main_log}"
    run_convert "${main_log}" \
      python "${CONVERTER}" \
        --input-dir "${raw_dir}" \
        --output-dir "${output_dir}" \
        --val-ratio 0.0 \
        --output-parquet-name sft_data.parquet \
        --stitch-runtime-hints \
        --only-correct-answers \
        --drop-degenerate-conversations \
        --final-answer-rewrite-mode api \
        --final-answer-rewrite-output-dir "${rewrite_dir}" \
        --final-answer-rewrite-model "${MODEL}" \
        --final-answer-rewrite-concurrency "${REWRITE_CONCURRENCY}" \
        --final-answer-rewrite-timeout "${OPENAI_CLIENT_TIMEOUT}" \
        --final-answer-rewrite-max-retries "${MAX_RETRIES}" \
        --final-answer-rewrite-max-completion-tokens "${MAX_COMPLETION_TOKENS}" \
        --final-answer-rewrite-max-failure-ratio "${MAX_FAILURE_RATIO}" \
        --final-answer-rewrite-progress-every "${PROGRESS_EVERY}" \
        --final-answer-rewrite-openai-base-url "${OPENAI_BASE_URL}" \
        --api-logger-save-dir "${API_LOGGER_SAVE_DIR}" \
        --api-logger-project-name "${API_LOGGER_PROJECT_NAME}" \
        --insight-doc-root "${INSIGHT_DOC_ROOT}" \
        --num-workers "${CONVERT_NUM_WORKERS}" \
        --conversion-progress-every "${CONVERSION_PROGRESS_EVERY}" \
        "${image_aspect_args[@]}" \
        "${rewrite_prefix_args[@]}"
  fi

  if [[ ! -d "${rewrite_dir}" ]]; then
    echo "Missing rewrite directory, cannot build base-model tool-argument-order parquet: ${rewrite_dir}" >&2
    exit 1
  fi

  tool_order_log="${LOG_ROOT}/${safe}.tool_arg_order.log"
  if [[ -e "${output_dir}/sft_data_base_model_tool_argument_order.parquet" && "${ALLOW_OVERWRITE}" != "1" ]]; then
    echo "Skipping existing ${output_dir}/sft_data_base_model_tool_argument_order.parquet; set ALLOW_OVERWRITE=1 to regenerate" | tee "${tool_order_log}"
  else
    echo "=== ${rel} base-model tool argument order conversion ===" | tee "${tool_order_log}"
    run_convert "${tool_order_log}" \
      python "${CONVERTER}" \
        --input-dir "${rewrite_dir}" \
        --output-dir "${output_dir}" \
        --val-ratio 0.0 \
        --output-parquet-name sft_data_base_model_tool_argument_order.parquet \
        --stitch-runtime-hints \
        --only-correct-answers \
        --drop-degenerate-conversations \
        --tool-argument-order base_model \
        --num-workers "${CONVERT_NUM_WORKERS}" \
        --conversion-progress-every "${CONVERSION_PROGRESS_EVERY}" \
        "${image_aspect_args[@]}" \
        "${rewrite_prefix_args[@]}"
  fi
done

echo "done"
echo "LOG_ROOT=${LOG_ROOT}"
