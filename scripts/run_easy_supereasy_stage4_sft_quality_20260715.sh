#!/bin/bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"
INSIGHT_DOC_ROOT="/scratch/ywxzml3j/likaican/src/InSight-doc"
SAMPLE_ROOT="/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_easy_supereasy_sft_quality_sample_20260715"
PARQUET_OUTPUT="/scratch/ywxzml3j/likaican/temp/easy_supereasy_sft_quality_sample_200_20260715-vreasoner_v2.parquet"
EXP_NAME="${EXP_NAME:-arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed}"
EVAL_NAME="${EVAL_NAME:-easy_supereasy_sft_quality_sample_200_20260715}"
AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_max_call10.yaml}"
CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${EXP_NAME}/${EVAL_NAME}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/${EXP_NAME}/${EVAL_NAME}}"
MODEL_PATH="${MODEL_PATH:-${STAGE_4_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}}"
VAL_ONLY_HF_MODEL_ROLLOUT="${VAL_ONLY_HF_MODEL_ROLLOUT:-true}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
RESUME_MODE="${RESUME_MODE:-disable}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

cd "$REPO_ROOT"
python scripts/build_easy_supereasy_stage4_sample_20260715.py --overwrite

cd "$INSIGHT_DOC_ROOT"
python verl/recipe/vsearch/create_parquet_dataset.py \
    --dataset InSightDocMixed \
    --data_root "$SAMPLE_ROOT" \
    --split all \
    --prompt vreasoner \
    --output_path "$PARQUET_OUTPUT" \
    --agent_name vreasoner_v2 \
    --num_workers "${PARQUET_NUM_WORKERS:-16}" \
    --extra_options '{"manifest_file": "manifest.jsonl"}'

cd "$REPO_ROOT"
echo "Stage 4 MODEL_PATH: $MODEL_PATH"
echo "Stage 4 VAL_ONLY_HF_MODEL_ROLLOUT: $VAL_ONLY_HF_MODEL_ROLLOUT"
echo "Stage 4 LOAD_FORMAT: $LOAD_FORMAT"
echo "Stage 4 RESUME_MODE: $RESUME_MODE"
echo "Stage 4 N_GPUS_PER_NODE: $N_GPUS_PER_NODE"
VAL_FILES="[${PARQUET_OUTPUT}]" \
EXP_NAME="$EXP_NAME" \
EVAL_NAME="$EVAL_NAME" \
AGENT_LOOP_CONFIG_PATH="$AGENT_LOOP_CONFIG_PATH" \
CONVERSATION_EXPORT_DIR="$CONVERSATION_EXPORT_DIR" \
MODEL_PATH="$MODEL_PATH" \
VAL_ONLY_HF_MODEL_ROLLOUT="$VAL_ONLY_HF_MODEL_ROLLOUT" \
LOAD_FORMAT="$LOAD_FORMAT" \
RESUME_MODE="$RESUME_MODE" \
N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
bash "$INSIGHT_DOC_ROOT/verl/recipe/vsearch/run.vr2/verl_latest.vr2.insight_doc_region_loc.sh"

python scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py \
    --input-dir "$CONVERSATION_EXPORT_DIR" \
    --output-dir "$SFT_OUTPUT_DIR" \
    --val-ratio 0.0 \
    --output-parquet-name sft_data.parquet \
    --stitch-runtime-hints \
    --only-correct-answers \
    --num-workers "${SFT_CONVERT_NUM_WORKERS:-8}"

echo "Sample root: $SAMPLE_ROOT"
echo "Eval parquet: $PARQUET_OUTPUT"
echo "Conversation export: $CONVERSATION_EXPORT_DIR"
echo "SFT output: $SFT_OUTPUT_DIR"
