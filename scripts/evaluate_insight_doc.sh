#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

: "${MODEL_PATH:?Set MODEL_PATH to a local HF checkpoint directory.}"
: "${VAL_FILES:?Set VAL_FILES to one or more eval parquet paths.}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for judging.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL for the OpenAI-compatible judge endpoint.}"

export REPO_ROOT
VERL_ROOT="${VERL_ROOT:-$REPO_ROOT/verl}"
export VERL_ROOT
export PYTHONPATH="$REPO_ROOT:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_CONFIGS="${MODEL_CONFIGS:-evals/model_configs/release_ray_vllm.yaml}"
export AGENT_CONFIG="${AGENT_CONFIG:-evals/agent_configs/insight_qwen_agent_core_zoom_factor2_area3500_rescale025.yaml}"
export RESCALES="${RESCALES:-0.25 0.35 0.5}"
export NUM_TRIALS="${NUM_TRIALS:-1}"
export GROUP_VAL_FILES="${GROUP_VAL_FILES:-1}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"
export INSIGHT_QWEN_JUDGE_MODE="${INSIGHT_QWEN_JUDGE_MODE:-legacy_prompt_v2}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/eval/$(date -u +%Y%m%d_%H%M%S)}"

echo "[eval] repo=$REPO_ROOT"
echo "[eval] model_path=$MODEL_PATH"
echo "[eval] val_files=$VAL_FILES"
echo "[eval] output_root=$OUTPUT_ROOT"

exec "$REPO_ROOT/scripts/queue_eval_full_sweep.sh" "$@"
