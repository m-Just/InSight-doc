#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ywxzml3j/ywxzml3juser40/data/insight_doc/generated/_medium_processed_wrong_question_manifests_merged_for_parquet_100dpi_part40}"
export AGENT_NAME_32B="${AGENT_NAME_32B:-insight_qwen_agent_initial_0_5}"
export AGENT_LOOP_CONFIG_PATH_32B="${AGENT_LOOP_CONFIG_PATH_32B:-recipe/vsearch/config/agent_insight_qwen_agent_initial_0_5.yaml}"
export AGENT_LOOP_CONFIG_PATH_VR2="${AGENT_LOOP_CONFIG_PATH_VR2:-recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_initial_0_5_max_calls10.yaml}"
export OUTPUT_ROOT_32B="${OUTPUT_ROOT_32B:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs}"
export PARQUET_OUTPUT_ROOT="${PARQUET_OUTPUT_ROOT:-/scratch/ywxzml3j/likaican/temp/medium_wrong_manifest_100dpi_pipeline}"
export DROP_DEGENERATE_CONVERSATIONS="${DROP_DEGENERATE_CONVERSATIONS:-1}"
export FINAL_ANSWER_REWRITE_MODE="${FINAL_ANSWER_REWRITE_MODE:-api}"

exec "${SCRIPT_DIR}/construct_medium_wrong_manifest_from_32b_100dpi.sh"
