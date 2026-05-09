#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT:-/scratch/ywxzml3j/likaican/src/InSight-doc}"

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$VERL_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Input dataset: merged medium wrong-question manifest built from
# generated/*/medium/processed/wrong_question_ids.txt.
# ------------------------------------------------------------------

MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_medium_processed_wrong_question_manifests_merged_for_parquet}"
MANIFEST_FILE="${MANIFEST_FILE:-manifest.jsonl}"
DATASET_CLASS="${DATASET_CLASS:-InSightDocMixed}"
DATASET_NAME="${DATASET_NAME:-medium_wrong_merged}"
IMG_RENDER_SETTING_NAME="${IMG_RENDER_SETTING_NAME:-dpi100_rescale05_proxy}"

if [[ ! -f "${MANIFEST_ROOT}/${MANIFEST_FILE}" ]]; then
    echo "Missing manifest: ${MANIFEST_ROOT}/${MANIFEST_FILE}" >&2
    exit 1
fi

if [[ ! -L "${MANIFEST_ROOT}/pdf_image" && ! -d "${MANIFEST_ROOT}/pdf_image" ]]; then
    echo "Missing pdf_image under ${MANIFEST_ROOT}" >&2
    exit 1
fi

# ------------------------------------------------------------------
# Shared parquet / prompt settings
# ------------------------------------------------------------------

PARQUET_OUTPUT_ROOT="${PARQUET_OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/medium_wrong_manifest_100dpi_pipeline}"

PARQUET_AGENT_NAME_32B="${PARQUET_AGENT_NAME_32B:-insight_qwen_agent}"
PROMPT_32B="${PROMPT_32B:-insight_qwen_agent}"
PARQUET_AGENT_NAME_VR2="${PARQUET_AGENT_NAME_VR2:-vreasoner_v2}"
PROMPT_VR2="${PROMPT_VR2:-vreasoner}"
QWEN_TOOL_LIST="${QWEN_TOOL_LIST:-[]}"
NUM_WORKERS_PARQUET="${NUM_WORKERS_PARQUET:-32}"
NUM_WORKERS_CONVERT="${NUM_WORKERS_CONVERT:-8}"
DROP_DEGENERATE_CONVERSATIONS="${DROP_DEGENERATE_CONVERSATIONS:-0}"
FINAL_ANSWER_REWRITE_MODE="${FINAL_ANSWER_REWRITE_MODE:-none}"
FINAL_ANSWER_REWRITE_MODEL="${FINAL_ANSWER_REWRITE_MODEL:-gpt-5-nano}"
FINAL_ANSWER_REWRITE_CONCURRENCY="${FINAL_ANSWER_REWRITE_CONCURRENCY:-8}"
FINAL_ANSWER_REWRITE_TIMEOUT="${FINAL_ANSWER_REWRITE_TIMEOUT:-120}"
FINAL_ANSWER_REWRITE_MAX_RETRIES="${FINAL_ANSWER_REWRITE_MAX_RETRIES:-4}"
FINAL_ANSWER_REWRITE_MAX_COMPLETION_TOKENS="${FINAL_ANSWER_REWRITE_MAX_COMPLETION_TOKENS:-4096}"
FINAL_ANSWER_REWRITE_MAX_FAILURE_RATIO="${FINAL_ANSWER_REWRITE_MAX_FAILURE_RATIO:-0.005}"
FINAL_ANSWER_REWRITE_MAX_FAILURES="${FINAL_ANSWER_REWRITE_MAX_FAILURES:-}"
FINAL_ANSWER_REWRITE_RETRY_ROUNDS="${FINAL_ANSWER_REWRITE_RETRY_ROUNDS:-2}"
FINAL_ANSWER_REWRITE_RETRY_SLEEP="${FINAL_ANSWER_REWRITE_RETRY_SLEEP:-30}"
FINAL_ANSWER_REWRITE_PROGRESS_EVERY="${FINAL_ANSWER_REWRITE_PROGRESS_EVERY:-50}"
FINAL_ANSWER_REWRITE_OUTPUT_DIR_VR2="${FINAL_ANSWER_REWRITE_OUTPUT_DIR_VR2:-}"

DATASET_BASENAME="$(basename "${MANIFEST_ROOT}")"
PARQUET_OUTPUT_PATH_32B="${PARQUET_OUTPUT_PATH_32B:-${PARQUET_OUTPUT_ROOT}/${DATASET_BASENAME}-${PARQUET_AGENT_NAME_32B}.pre32b.parquet}"
PARQUET_OUTPUT_PATH_VR2="${PARQUET_OUTPUT_PATH_VR2:-${PARQUET_OUTPUT_ROOT}/${DATASET_BASENAME}-${PARQUET_AGENT_NAME_VR2}.prevr2.parquet}"

# ------------------------------------------------------------------
# Stage 1: 32B easy filtering with initial_rescale=0.5
# ------------------------------------------------------------------

MODEL_PATH_32B="${MODEL_PATH_32B:-Qwen/Qwen3-VL-32B-Instruct}"
AGENT_NAME_32B="${AGENT_NAME_32B:-insight_qwen_agent_initial_0_5}"
AGENT_LOOP_CONFIG_PATH_32B="${AGENT_LOOP_CONFIG_PATH_32B:-recipe/vsearch/config/agent_insight_qwen_agent_initial_0_5.yaml}"
TENSOR_MODEL_PARALLEL_SIZE_32B="${TENSOR_MODEL_PARALLEL_SIZE_32B:-4}"
AGENT_NUM_WORKERS_32B="${AGENT_NUM_WORKERS_32B:-2}"
MAX_RESPONSE_LENGTH_32B="${MAX_RESPONSE_LENGTH_32B:-1024}"

OUTPUT_ROOT_32B="${OUTPUT_ROOT_32B:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs}"
WORK_DIR_32B="${WORK_DIR_32B:-${OUTPUT_ROOT_32B}/${AGENT_NAME_32B}_default_sys_100dpi_resumable/qwen3-vl-32b-instruct/${DATASET_BASENAME}}"
EXP_NAME_32B="${EXP_NAME_32B:-${AGENT_NAME_32B}_default_sys_100dpi_qwen3-vl-32b-instruct_${DATASET_BASENAME}}"
if [[ -z "${CONVERTED_SFT_DIR_32B:-}" ]]; then
    if [[ "${DROP_DEGENERATE_CONVERSATIONS}" == "1" ]]; then
        CONVERTED_SFT_DIR_32B="${WORK_DIR_32B}/converted_sft_drop_degenerate"
    else
        CONVERTED_SFT_DIR_32B="${WORK_DIR_32B}/converted_sft"
    fi
fi

# ------------------------------------------------------------------
# Stage 2/3: vreasoner_v2 CoT generation with initial_rescale=0.5
# ------------------------------------------------------------------

VR2_MODEL_PATH="${VR2_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
VR2_EXP_NAME="${VR2_EXP_NAME:-arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed}"
VR2_EVAL_NAME="${VR2_EVAL_NAME:-${DATASET_BASENAME}-100dpi_resumable}"
AGENT_LOOP_CONFIG_PATH_VR2="${AGENT_LOOP_CONFIG_PATH_VR2:-recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_initial_0_5.yaml}"
CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${VR2_EXP_NAME}/${VR2_EVAL_NAME}}"
if [[ -z "${SFT_OUTPUT_DIR_VR2:-}" ]]; then
    SFT_OUTPUT_DIR_VR2="/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/${VR2_EXP_NAME}/${VR2_EVAL_NAME}"
fi
if [[ "${FINAL_ANSWER_REWRITE_MODE}" != "none" ]]; then
    SFT_OUTPUT_DIR_VR2_REWRITE="${SFT_OUTPUT_DIR_VR2_REWRITE:-${SFT_OUTPUT_DIR_VR2}_gpt5_nano_rewrite}"
else
    SFT_OUTPUT_DIR_VR2_REWRITE="${SFT_OUTPUT_DIR_VR2_REWRITE:-}"
fi

# ------------------------------------------------------------------
# Step toggles
# ------------------------------------------------------------------

RUN_CREATE_PARQUET_32B="${RUN_CREATE_PARQUET_32B:-1}"
RUN_EVAL_32B="${RUN_EVAL_32B:-1}"
RUN_CONVERT_32B="${RUN_CONVERT_32B:-1}"
RUN_CREATE_PARQUET_VR2="${RUN_CREATE_PARQUET_VR2:-1}"
RUN_EVAL_VR2="${RUN_EVAL_VR2:-1}"
RUN_CONVERT_VR2="${RUN_CONVERT_VR2:-1}"

if [[ "${RUN_EVAL_VR2}" == "1" || ( "${RUN_CONVERT_VR2}" == "1" && "${FINAL_ANSWER_REWRITE_MODE}" != "none" ) ]]; then
    : "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
    : "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set}"
fi

CONVERT_FLAGS_32B=()
if [[ "${DROP_DEGENERATE_CONVERSATIONS}" == "1" ]]; then
    CONVERT_FLAGS_32B+=(--drop-degenerate-conversations)
fi

if [[ "${RUN_CREATE_PARQUET_32B}" == "1" ]]; then
    cd "${INSIGHT_DOC_ROOT}"
    mkdir -p "$(dirname "${PARQUET_OUTPUT_PATH_32B}")"
    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset "${DATASET_CLASS}" \
        --data_root "${MANIFEST_ROOT}" \
        --split all \
        --prompt "${PROMPT_32B}" \
        --output_path "${PARQUET_OUTPUT_PATH_32B}" \
        --agent_name "${PARQUET_AGENT_NAME_32B}" \
        --num_workers "${NUM_WORKERS_PARQUET}" \
        --extra_options "{\"manifest_file\": \"${MANIFEST_FILE}\"}"
fi

if [[ "${RUN_EVAL_32B}" == "1" ]]; then
    cd "${VERL_ROOT}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH_32B}" \
    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
    MODEL_PATH="${MODEL_PATH_32B}" \
    QWEN_TOOL_LIST="${QWEN_TOOL_LIST}" \
    AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH_32B}" \
    TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE_32B}" \
    AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS_32B}" \
    OUTPUT_ROOT="${OUTPUT_ROOT_32B}" \
    WORK_DIR="${WORK_DIR_32B}" \
    EXP_NAME="${EXP_NAME_32B}" \
    VAL_FILES="[${PARQUET_OUTPUT_PATH_32B}]" \
    LOGGER="['console']" \
    bash "${VERL_ROOT}/scripts/run_iq_base_eval_default_sampling_insight_rl15360_ckpt0422.sh"
fi

if [[ "${RUN_CONVERT_32B}" == "1" ]]; then
    cd "${VERL_ROOT}"
    python "${VERL_ROOT}/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py" \
        --input-dir "${WORK_DIR_32B}/exported_conversations" \
        --output-dir "${CONVERTED_SFT_DIR_32B}" \
        --val-ratio 0.0 \
        --output-parquet-name "sft_data.parquet" \
        --stitch-runtime-hints \
        --only-correct-answers \
        --num-workers "${NUM_WORKERS_CONVERT}" \
        "${CONVERT_FLAGS_32B[@]}"
fi

if [[ ( "${RUN_CREATE_PARQUET_VR2}" == "1" || "${RUN_EVAL_VR2}" == "1" || "${RUN_CONVERT_VR2}" == "1" ) && ! -f "${CONVERTED_SFT_DIR_32B}/wrong_question_ids.txt" ]]; then
    echo "Missing 32B wrong ids: ${CONVERTED_SFT_DIR_32B}/wrong_question_ids.txt" >&2
    exit 1
fi

if [[ "${RUN_CREATE_PARQUET_VR2}" == "1" ]]; then
    cd "${INSIGHT_DOC_ROOT}"
    mkdir -p "$(dirname "${PARQUET_OUTPUT_PATH_VR2}")"
    python verl/recipe/vsearch/create_parquet_dataset.py \
        --dataset "${DATASET_CLASS}" \
        --data_root "${MANIFEST_ROOT}" \
        --split all \
        --prompt "${PROMPT_VR2}" \
        --output_path "${PARQUET_OUTPUT_PATH_VR2}" \
        --agent_name "${PARQUET_AGENT_NAME_VR2}" \
        --num_workers "${NUM_WORKERS_PARQUET}" \
        --extra_options "{\"manifest_file\": \"${MANIFEST_FILE}\"}" \
        --question_id_file "${CONVERTED_SFT_DIR_32B}/wrong_question_ids.txt"
fi

if [[ "${RUN_EVAL_VR2}" == "1" ]]; then
    cd "${INSIGHT_DOC_ROOT}"
    EXP_NAME="${VR2_EXP_NAME}" \
    EVAL_NAME="${VR2_EVAL_NAME}" \
    MODEL_PATH="${VR2_MODEL_PATH}" \
    VAL_FILES="[${PARQUET_OUTPUT_PATH_VR2}]" \
    AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH_VR2}" \
    CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR}" \
    bash "${INSIGHT_DOC_ROOT}/verl/recipe/vsearch/run.vr2/verl_latest.vr2.insight_doc_region_loc.sh"
fi

if [[ "${RUN_CONVERT_VR2}" == "1" ]]; then
    cd "${VERL_ROOT}"
    python "${VERL_ROOT}/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py" \
        --input-dir "${CONVERSATION_EXPORT_DIR}" \
        --output-dir "${SFT_OUTPUT_DIR_VR2}" \
        --val-ratio 0.0 \
        --output-parquet-name "sft_data.parquet" \
        --stitch-runtime-hints \
        --only-correct-answers \
        --num-workers "${NUM_WORKERS_CONVERT}"

    if [[ "${FINAL_ANSWER_REWRITE_MODE}" != "none" ]]; then
        VR2_CONVERT_FLAGS=()
        VR2_CONVERT_FLAGS+=(
            --final-answer-rewrite-mode "${FINAL_ANSWER_REWRITE_MODE}"
            --final-answer-rewrite-model "${FINAL_ANSWER_REWRITE_MODEL}"
            --final-answer-rewrite-concurrency "${FINAL_ANSWER_REWRITE_CONCURRENCY}"
            --final-answer-rewrite-timeout "${FINAL_ANSWER_REWRITE_TIMEOUT}"
            --final-answer-rewrite-max-retries "${FINAL_ANSWER_REWRITE_MAX_RETRIES}"
            --final-answer-rewrite-max-completion-tokens "${FINAL_ANSWER_REWRITE_MAX_COMPLETION_TOKENS}"
            --final-answer-rewrite-max-failure-ratio "${FINAL_ANSWER_REWRITE_MAX_FAILURE_RATIO}"
            --final-answer-rewrite-retry-rounds "${FINAL_ANSWER_REWRITE_RETRY_ROUNDS}"
            --final-answer-rewrite-retry-sleep "${FINAL_ANSWER_REWRITE_RETRY_SLEEP}"
            --final-answer-rewrite-progress-every "${FINAL_ANSWER_REWRITE_PROGRESS_EVERY}"
            --final-answer-rewrite-openai-base-url "${OPENAI_BASE_URL}"
        )
        if [[ -n "${FINAL_ANSWER_REWRITE_MAX_FAILURES}" ]]; then
            VR2_CONVERT_FLAGS+=(--final-answer-rewrite-max-failures "${FINAL_ANSWER_REWRITE_MAX_FAILURES}")
        fi
        if [[ -n "${FINAL_ANSWER_REWRITE_OUTPUT_DIR_VR2}" ]]; then
            VR2_CONVERT_FLAGS+=(--final-answer-rewrite-output-dir "${FINAL_ANSWER_REWRITE_OUTPUT_DIR_VR2}")
        fi
        python "${VERL_ROOT}/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py" \
            --input-dir "${CONVERSATION_EXPORT_DIR}" \
            --output-dir "${SFT_OUTPUT_DIR_VR2_REWRITE}" \
            --val-ratio 0.0 \
            --output-parquet-name "sft_data.parquet" \
            --stitch-runtime-hints \
            --only-correct-answers \
            --num-workers "${NUM_WORKERS_CONVERT}" \
            "${VR2_CONVERT_FLAGS[@]}"
    fi
fi

echo "Input manifest root: ${MANIFEST_ROOT}"
echo "32B parquet: ${PARQUET_OUTPUT_PATH_32B}"
echo "32B exported conversations: ${WORK_DIR_32B}/exported_conversations"
echo "32B converted_sft: ${CONVERTED_SFT_DIR_32B}"
echo "vreasoner parquet: ${PARQUET_OUTPUT_PATH_VR2}"
echo "vreasoner exports: ${CONVERSATION_EXPORT_DIR}"
echo "vreasoner converted_sft: ${SFT_OUTPUT_DIR_VR2}"
if [[ -n "${SFT_OUTPUT_DIR_VR2_REWRITE:-}" ]]; then
    echo "vreasoner converted_sft rewrite: ${SFT_OUTPUT_DIR_VR2_REWRITE}"
fi
