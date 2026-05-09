#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://az.gptplus5.com/v1}"
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
export WORK_DIR="${WORK_DIR:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426}"
export EXP_NAME="${EXP_NAME:-insight_qwen_agent_full_sft_all_convos_0426}"
export RDZV_PORT="${RDZV_PORT:-29521}"

cd "$REPO_ROOT"
exec scripts/run_insight_qwen_agent_full_sft_all_convos_0426.sh
