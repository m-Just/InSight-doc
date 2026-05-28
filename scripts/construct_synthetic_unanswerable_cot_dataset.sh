#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT:-/scratch/ywxzml3j/likaican/src/InSight-doc}"

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$VERL_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

cd "$VERL_ROOT"

# ==============================================================
# Input dataset: verified synthetic unanswerable manifest.
#   This script starts from the verification outputs and runs:
#     1. 32B filtering
#     2. vreasoner_v2 CoT generation
#     3. final CoT conversion
#   The 8B filtering prepass is intentionally skipped.
# ==============================================================

VERIFIED_DATA_ROOT="${VERIFIED_DATA_ROOT:-/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/artifacts/synthetic_unanswerable_pipeline/balanced_sample5k_seed42_run1_az/verify_all_preview_c32}"
MANIFEST_FILE="${MANIFEST_FILE:-manifest.jsonl}"
NORMALIZED_MANIFEST_FILE="${NORMALIZED_MANIFEST_FILE:-manifest.normalized.jsonl}"
DATASET_CLASS="${DATASET_CLASS:-InSightDocMixedWithArxiv}"
VERIFICATION_RECORDS_JSONL="${VERIFICATION_RECORDS_JSONL:-${VERIFIED_DATA_ROOT}/verified_candidates.jsonl}"

if [[ ! -f "${VERIFIED_DATA_ROOT}/${MANIFEST_FILE}" ]]; then
    echo "Missing manifest: ${VERIFIED_DATA_ROOT}/${MANIFEST_FILE}" >&2
    exit 1
fi

if [[ ! -L "${VERIFIED_DATA_ROOT}/pdf_image" && ! -d "${VERIFIED_DATA_ROOT}/pdf_image" ]]; then
    echo "Missing pdf_image under ${VERIFIED_DATA_ROOT}" >&2
    exit 1
fi

DATASET_BASENAME="${DATASET_BASENAME:-$(basename "${VERIFIED_DATA_ROOT}")}"
PARQUET_OUTPUT_ROOT="${PARQUET_OUTPUT_ROOT:-${VERIFIED_DATA_ROOT}/parquets}"
mkdir -p "${PARQUET_OUTPUT_ROOT}"

python - "${VERIFIED_DATA_ROOT}" "${VERIFICATION_RECORDS_JSONL}" <<'PY'
import json
import os
import sys
from pathlib import Path

verified_root = Path(sys.argv[1])
verified_jsonl = Path(sys.argv[2])
target_pdf_root = verified_root / "pdf_image"

if not verified_jsonl.exists():
    print(f"[repair-pdf-image] skip: verification records not found: {verified_jsonl}")
    raise SystemExit(0)

accepted_labels = {"document_mismatch", "missing_evidence"}
repaired = 0
missing_source = 0
checked = 0

with verified_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        verification = record.get("verification") or {}
        if verification.get("label") not in accepted_labels:
            continue
        candidate = record.get("candidate") or {}
        source_manifest_path = candidate.get("source_manifest_path")
        if not source_manifest_path:
            continue
        source_pdf_root = Path(source_manifest_path).resolve().parent / "pdf_image"
        selected_images = record.get("verification_selected_images") or []
        for rel in selected_images:
            checked += 1
            rel_path = Path(rel)
            dst = target_pdf_root / rel_path
            if dst.exists():
                continue
            src = source_pdf_root / rel_path
            if not src.exists():
                missing_source += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
            repaired += 1

print(
    f"[repair-pdf-image] checked={checked} repaired={repaired} missing_source={missing_source}"
)
PY

python - "${VERIFIED_DATA_ROOT}/${MANIFEST_FILE}" "${VERIFIED_DATA_ROOT}/${NORMALIZED_MANIFEST_FILE}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

CORE_LIST_FIELDS = {"images"}


def normalize_scalarish(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is not None and not isinstance(value, str):
        return str(value)
    return value


with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for key, value in list(row.items()):
            if key in CORE_LIST_FIELDS:
                continue
            if key == "synthetic_unanswerable_metadata" and isinstance(value, dict):
                normalized_meta = {k: normalize_scalarish(v) for k, v in value.items()}
                row[key] = json.dumps(normalized_meta, ensure_ascii=False)
                continue
            row[key] = normalize_scalarish(value)
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

# ==============================================================
# Shared parquet / conversion settings
# ==============================================================

QWEN_TOOL_LIST="${QWEN_TOOL_LIST:-[]}"
NUM_WORKERS_PARQUET="${NUM_WORKERS_PARQUET:-32}"
NUM_WORKERS_CONVERT="${NUM_WORKERS_CONVERT:-8}"

# ==============================================================
# Stage 1: 32B filtering with initial_rescale=0.25
# ==============================================================

PARQUET_AGENT_NAME_32B="${PARQUET_AGENT_NAME_32B:-insight_qwen_agent}"
PROMPT_32B="${PROMPT_32B:-default}"
PARQUET_OUTPUT_PATH_32B="${PARQUET_OUTPUT_PATH_32B:-${PARQUET_OUTPUT_ROOT}/${DATASET_BASENAME}-${PARQUET_AGENT_NAME_32B}.pre32b.parquet}"

MODEL_PATH_32B="${MODEL_PATH_32B:-Qwen/Qwen3-VL-32B-Instruct}"
AGENT_NAME_32B="${AGENT_NAME_32B:-insight_qwen_agent_zoom_factor2}"
AGENT_LOOP_CONFIG_PATH_32B="${AGENT_LOOP_CONFIG_PATH_32B:-recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2.yaml}"
TENSOR_MODEL_PARALLEL_SIZE_32B="${TENSOR_MODEL_PARALLEL_SIZE_32B:-4}"
AGENT_NUM_WORKERS_32B="${AGENT_NUM_WORKERS_32B:-2}"
MAX_RESPONSE_LENGTH_32B="${MAX_RESPONSE_LENGTH_32B:-1024}"
RAY_NOSET_VISIBLE_DEVICES_32B="${RAY_NOSET_VISIBLE_DEVICES_32B:-0}"

OUTPUT_ROOT_32B="${OUTPUT_ROOT_32B:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs}"
WORK_DIR_32B="${WORK_DIR_32B:-${OUTPUT_ROOT_32B}/${AGENT_NAME_32B}_synthetic_unanswerable_resumable/qwen3-vl-32b-instruct/${DATASET_BASENAME}}"
EXP_NAME_32B="${EXP_NAME_32B:-${AGENT_NAME_32B}_synthetic_unanswerable_qwen3-vl-32b-instruct_${DATASET_BASENAME}}"
CONVERTED_SFT_DIR_32B="${CONVERTED_SFT_DIR_32B:-${WORK_DIR_32B}/converted_sft}"

# ==============================================================
# Stage 2/3: vreasoner_v2 CoT generation with initial_rescale=0.25
# ==============================================================

PARQUET_AGENT_NAME_VR2="${PARQUET_AGENT_NAME_VR2:-vreasoner_v2}"
PROMPT_VR2="${PROMPT_VR2:-vreasoner}"
PARQUET_OUTPUT_PATH_VR2="${PARQUET_OUTPUT_PATH_VR2:-${PARQUET_OUTPUT_ROOT}/${DATASET_BASENAME}-${PARQUET_AGENT_NAME_VR2}.parquet}"

VR2_MODEL_PATH="${VR2_MODEL_PATH:-}"
VR2_EXP_NAME="${VR2_EXP_NAME:-synthetic_unanswerable_qwen3_region_loc}"
VR2_EVAL_NAME="${VR2_EVAL_NAME:-${DATASET_BASENAME}_resumable}"
AGENT_LOOP_CONFIG_PATH_VR2="${AGENT_LOOP_CONFIG_PATH_VR2:-recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_initial_0_25_max_calls10_unanswerable_aware_verify_explicit_evidence_linked.yaml}"
CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${VR2_EXP_NAME}/${VR2_EVAL_NAME}}"
SFT_OUTPUT_DIR_VR2="${SFT_OUTPUT_DIR_VR2:-/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/${VR2_EXP_NAME}/${VR2_EVAL_NAME}}"

# VR2 eval launch mode:
#   default         -> standard launcher behavior
#   hf_local_rollout -> adds:
#                      trainer.val_only_hf_model_rollout=true
#                      actor_rollout_ref.rollout.load_format=auto
#                      trainer.resume_mode=disable
#                      trainer.n_gpus_per_node=${VR2_N_GPUS_PER_NODE}
VR2_LAUNCH_MODE="${VR2_LAUNCH_MODE:-default}"
VR2_N_GPUS_PER_NODE="${VR2_N_GPUS_PER_NODE:-${N_GPUS_PER_NODE:-1}}"

# ==============================================================
# Step toggles
# ==============================================================

RUN_CREATE_PARQUET_32B="${RUN_CREATE_PARQUET_32B:-1}"
RUN_EVAL_32B="${RUN_EVAL_32B:-1}"
RUN_CONVERT_32B="${RUN_CONVERT_32B:-1}"
RUN_CREATE_PARQUET_VR2="${RUN_CREATE_PARQUET_VR2:-1}"
RUN_EVAL_VR2="${RUN_EVAL_VR2:-1}"
RUN_CONVERT_VR2="${RUN_CONVERT_VR2:-1}"
EXTRA_EVAL_ARGS_VR2="${EXTRA_EVAL_ARGS_VR2:-}"

MODE_EVAL_ARGS_VR2=""
case "${VR2_LAUNCH_MODE}" in
    default)
        ;;
    hf_local_rollout)
        MODE_EVAL_ARGS_VR2="trainer.n_gpus_per_node=${VR2_N_GPUS_PER_NODE} trainer.val_only_hf_model_rollout=true actor_rollout_ref.rollout.load_format=auto trainer.resume_mode=disable"
        ;;
    *)
        echo "Unsupported VR2_LAUNCH_MODE: ${VR2_LAUNCH_MODE}" >&2
        echo "Expected one of: default, hf_local_rollout" >&2
        exit 1
        ;;
esac

if [[ -n "${MODE_EVAL_ARGS_VR2}" && -n "${EXTRA_EVAL_ARGS_VR2}" ]]; then
    EXTRA_EVAL_ARGS_VR2="${MODE_EVAL_ARGS_VR2} ${EXTRA_EVAL_ARGS_VR2}"
elif [[ -n "${MODE_EVAL_ARGS_VR2}" ]]; then
    EXTRA_EVAL_ARGS_VR2="${MODE_EVAL_ARGS_VR2}"
fi

echo "VERIFIED_DATA_ROOT: ${VERIFIED_DATA_ROOT}"
echo "DATASET_BASENAME: ${DATASET_BASENAME}"
echo "DATASET_CLASS: ${DATASET_CLASS}"
echo "PARQUET_OUTPUT_PATH_32B: ${PARQUET_OUTPUT_PATH_32B}"
echo "WORK_DIR_32B: ${WORK_DIR_32B}"
echo "PARQUET_OUTPUT_PATH_VR2: ${PARQUET_OUTPUT_PATH_VR2}"
echo "CONVERSATION_EXPORT_DIR: ${CONVERSATION_EXPORT_DIR}"
echo "SFT_OUTPUT_DIR_VR2: ${SFT_OUTPUT_DIR_VR2}"
echo "VR2_LAUNCH_MODE: ${VR2_LAUNCH_MODE}"
echo "EXTRA_EVAL_ARGS_VR2: ${EXTRA_EVAL_ARGS_VR2}"

if [[ "${RUN_CREATE_PARQUET_32B}" == "1" ]]; then
    python "${INSIGHT_DOC_ROOT}/verl/recipe/vsearch/create_parquet_dataset.py" \
        --dataset "${DATASET_CLASS}" \
        --data_root "${VERIFIED_DATA_ROOT}" \
        --split all \
        --prompt "${PROMPT_32B}" \
        --output_path "${PARQUET_OUTPUT_PATH_32B}" \
        --agent_name "${PARQUET_AGENT_NAME_32B}" \
        --num_workers "${NUM_WORKERS_PARQUET}" \
        --extra_options "{\"manifest_file\": \"${NORMALIZED_MANIFEST_FILE}\"}"
fi

if [[ "${RUN_EVAL_32B}" == "1" ]]; then
    if [[ "${RAY_NOSET_VISIBLE_DEVICES_32B}" == "1" ]]; then
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
    else
        MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH_32B}" \
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
fi

if [[ "${RUN_CONVERT_32B}" == "1" ]]; then
    python "${VERL_ROOT}/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py" \
        --input-dir "${WORK_DIR_32B}/exported_conversations" \
        --output-dir "${CONVERTED_SFT_DIR_32B}" \
        --val-ratio 0.0 \
        --output-parquet-name "sft_data.parquet" \
        --stitch-runtime-hints \
        --only-correct-answers \
        --num-workers "${NUM_WORKERS_CONVERT}"
fi

if [[ ( "${RUN_CREATE_PARQUET_VR2}" == "1" || "${RUN_EVAL_VR2}" == "1" || "${RUN_CONVERT_VR2}" == "1" ) && ! -f "${CONVERTED_SFT_DIR_32B}/wrong_question_ids.txt" ]]; then
    echo "Missing 32B wrong ids: ${CONVERTED_SFT_DIR_32B}/wrong_question_ids.txt" >&2
    exit 1
fi

if [[ "${RUN_CREATE_PARQUET_VR2}" == "1" ]]; then
    python "${INSIGHT_DOC_ROOT}/verl/recipe/vsearch/create_parquet_dataset.py" \
        --dataset "${DATASET_CLASS}" \
        --data_root "${VERIFIED_DATA_ROOT}" \
        --split all \
        --prompt "${PROMPT_VR2}" \
        --output_path "${PARQUET_OUTPUT_PATH_VR2}" \
        --agent_name "${PARQUET_AGENT_NAME_VR2}" \
        --num_workers "${NUM_WORKERS_PARQUET}" \
        --extra_options "{\"manifest_file\": \"${NORMALIZED_MANIFEST_FILE}\"}" \
        --question_id_file "${CONVERTED_SFT_DIR_32B}/wrong_question_ids.txt"
fi

if [[ "${RUN_EVAL_VR2}" == "1" ]]; then
    if [[ -n "${EXTRA_EVAL_ARGS_VR2}" ]]; then
        # shellcheck disable=SC2086
        MODEL_PATH="${VR2_MODEL_PATH}" \
        EXP_NAME="${VR2_EXP_NAME}" \
        EVAL_NAME="${VR2_EVAL_NAME}" \
        VAL_FILES="[${PARQUET_OUTPUT_PATH_VR2}]" \
        AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH_VR2}" \
        CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR}" \
        bash "${VERL_ROOT}/recipe/vsearch/run.vr2/verl_latest.vr2.insight_doc_region_loc.sh" ${EXTRA_EVAL_ARGS_VR2}
    else
        MODEL_PATH="${VR2_MODEL_PATH}" \
        EXP_NAME="${VR2_EXP_NAME}" \
        EVAL_NAME="${VR2_EVAL_NAME}" \
        VAL_FILES="[${PARQUET_OUTPUT_PATH_VR2}]" \
        AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH_VR2}" \
        CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR}" \
        bash "${VERL_ROOT}/recipe/vsearch/run.vr2/verl_latest.vr2.insight_doc_region_loc.sh"
    fi
fi

if [[ "${RUN_CONVERT_VR2}" == "1" ]]; then
    python "${VERL_ROOT}/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py" \
        --input-dir "${CONVERSATION_EXPORT_DIR}" \
        --output-dir "${SFT_OUTPUT_DIR_VR2}" \
        --val-ratio 0.0 \
        --output-parquet-name "sft_data.parquet" \
        --stitch-runtime-hints \
        --only-correct-answers \
        --num-workers "${NUM_WORKERS_CONVERT}"
fi

echo "Done."
