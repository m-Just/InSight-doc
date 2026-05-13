#!/bin/bash
set -euo pipefail

# Stage 4/5 wrapper for verified synthetic unanswerable manifests.
#
# Expected input layout:
#   <VERIFIED_DATA_ROOT>/
#     manifest.jsonl
#     pdf_image/
#
# This script:
#   1. creates a vreasoner_v2 parquet from the verified manifest
#   2. runs vreasoner_v2 CoT generation
#   3. converts exported conversations into final SFT parquet

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set"
  exit 1
fi

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "OPENAI_BASE_URL is not set"
  exit 1
fi

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT:-/scratch/ywxzml3j/likaican/src/InSight-doc}"
VERIFY_ROOT_DEFAULT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/artifacts/synthetic_unanswerable_pipeline/low_multipart_sample40/verify_all"

VERIFIED_DATA_ROOT="${VERIFIED_DATA_ROOT:-$VERIFY_ROOT_DEFAULT}"
MANIFEST_FILE="${MANIFEST_FILE:-manifest.jsonl}"
NORMALIZED_MANIFEST_FILE="${NORMALIZED_MANIFEST_FILE:-manifest.normalized.jsonl}"
DATASET_NAME="${DATASET_NAME:-InSightDoc0352}"
PARQUET_AGENT_NAME="${PARQUET_AGENT_NAME:-vreasoner_v2}"
PROMPT="${PROMPT:-vreasoner}"

DATASET_BASENAME="${DATASET_BASENAME:-$(basename "$VERIFIED_DATA_ROOT")}"
PARQUET_OUTPUT_DIR="${PARQUET_OUTPUT_DIR:-$VERIFIED_DATA_ROOT/parquets}"
PARQUET_OUTPUT_PATH="${PARQUET_OUTPUT_PATH:-$PARQUET_OUTPUT_DIR/${DATASET_BASENAME}-${PARQUET_AGENT_NAME}.parquet}"

EXP_NAME="${EXP_NAME:-synthetic_unanswerable_qwen3_region_loc}"
EVAL_NAME="${EVAL_NAME:-${DATASET_BASENAME}_resumable}"
VAL_FILES="${VAL_FILES:-[${PARQUET_OUTPUT_PATH}]}"
AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_initial_0_35_max_calls10_unanswerable_aware.yaml}"
CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${EXP_NAME}/${EVAL_NAME}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/${EXP_NAME}/${EVAL_NAME}}"

RUN_CREATE_PARQUET="${RUN_CREATE_PARQUET:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

if [[ ! -f "$VERIFIED_DATA_ROOT/$MANIFEST_FILE" ]]; then
  echo "manifest file not found: $VERIFIED_DATA_ROOT/$MANIFEST_FILE" >&2
  exit 1
fi

if [[ ! -d "$VERIFIED_DATA_ROOT/pdf_image" ]]; then
  echo "pdf_image dir not found: $VERIFIED_DATA_ROOT/pdf_image" >&2
  exit 1
fi

mkdir -p "$PARQUET_OUTPUT_DIR"

python - "$VERIFIED_DATA_ROOT/$MANIFEST_FILE" "$VERIFIED_DATA_ROOT/$NORMALIZED_MANIFEST_FILE" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qpn = row.get("question_page_num")
        if isinstance(qpn, (list, dict)):
            row["question_page_num"] = json.dumps(qpn, ensure_ascii=False)
        elif qpn is not None and not isinstance(qpn, str):
            row["question_page_num"] = str(qpn)
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

echo "VERIFIED_DATA_ROOT: $VERIFIED_DATA_ROOT"
echo "MANIFEST_FILE: $MANIFEST_FILE"
echo "NORMALIZED_MANIFEST_FILE: $NORMALIZED_MANIFEST_FILE"
echo "PARQUET_OUTPUT_PATH: $PARQUET_OUTPUT_PATH"
echo "CONVERSATION_EXPORT_DIR: $CONVERSATION_EXPORT_DIR"
echo "SFT_OUTPUT_DIR: $SFT_OUTPUT_DIR"

if [[ "$RUN_CREATE_PARQUET" == "1" ]]; then
  python "$INSIGHT_DOC_ROOT/verl/recipe/vsearch/create_parquet_dataset.py" \
    --dataset "$DATASET_NAME" \
    --data_root "$VERIFIED_DATA_ROOT" \
    --split all \
    --prompt "$PROMPT" \
    --output_path "$PARQUET_OUTPUT_PATH" \
    --agent_name "$PARQUET_AGENT_NAME" \
    --num_workers 32 \
    --extra_options "{\"manifest_file\": \"${NORMALIZED_MANIFEST_FILE}\"}"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ -n "$EXTRA_EVAL_ARGS" ]]; then
    # shellcheck disable=SC2086
    EXP_NAME="$EXP_NAME" \
    EVAL_NAME="$EVAL_NAME" \
    VAL_FILES="$VAL_FILES" \
    AGENT_LOOP_CONFIG_PATH="$AGENT_LOOP_CONFIG_PATH" \
    CONVERSATION_EXPORT_DIR="$CONVERSATION_EXPORT_DIR" \
    bash ./recipe/vsearch/run.vr2/verl_latest.vr2.insight_doc_region_loc.sh $EXTRA_EVAL_ARGS
  else
    EXP_NAME="$EXP_NAME" \
    EVAL_NAME="$EVAL_NAME" \
    VAL_FILES="$VAL_FILES" \
    AGENT_LOOP_CONFIG_PATH="$AGENT_LOOP_CONFIG_PATH" \
    CONVERSATION_EXPORT_DIR="$CONVERSATION_EXPORT_DIR" \
    bash ./recipe/vsearch/run.vr2/verl_latest.vr2.insight_doc_region_loc.sh
  fi
fi

if [[ "$RUN_CONVERT" == "1" ]]; then
  python ./scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py \
    --input-dir "$CONVERSATION_EXPORT_DIR" \
    --output-dir "$SFT_OUTPUT_DIR" \
    --val-ratio 0.0 \
    --output-parquet-name "sft_data.parquet" \
    --stitch-runtime-hints \
    --only-correct-answers \
    --num-workers 8
fi

echo "Done."
