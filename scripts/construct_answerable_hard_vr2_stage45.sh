#!/bin/bash
set -euo pipefail

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSIGHT_DOC_ROOT="${INSIGHT_DOC_ROOT:-/scratch/ywxzml3j/likaican/src/InSight-doc}"

cd "$VERL_ROOT"
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy NO_PROXY no_proxy

INPUT_ROOT="${INPUT_ROOT:-${VERL_ROOT}/artifacts/answerable_vr2_inputs/by_rescale_20260518/hard}"
RUN_TAG="${RUN_TAG:-retry_hard_by_rescale_20260518}"
EXP_NAME_BASE="${EXP_NAME_BASE:-answerable_qwen3_region_loc_hard_retry_by_rescale}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

run_one() {
  local rescale="$1"
  local val_parquet="$2"
  local agent_loop_config_path="$3"
  local eval_name="rescale${rescale}_${RUN_TAG}"
  local conversation_export_dir="/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${EXP_NAME_BASE}/${eval_name}"
  local sft_output_dir="/scratch/ywxzml3j/likaican/mms1_rl/converted_sft/multi_agent_vsearch/${EXP_NAME_BASE}/${eval_name}"

  if [[ ! -f "${val_parquet}" ]]; then
    echo "Missing VAL_PARQUET: ${val_parquet}" >&2
    exit 1
  fi

  if [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
    if [[ -e "${conversation_export_dir}" || -e "${sft_output_dir}" ]]; then
      echo "Refusing to overwrite existing outputs for rescale${rescale}." >&2
      echo "  CONVERSATION_EXPORT_DIR=${conversation_export_dir}" >&2
      echo "  SFT_OUTPUT_DIR=${sft_output_dir}" >&2
      echo "Set ALLOW_OVERWRITE=1 to reuse these paths." >&2
      exit 1
    fi
  fi

  echo "==> answerable hard rescale${rescale}"
  echo "VAL_PARQUET: ${val_parquet}"
  echo "AGENT_LOOP_CONFIG_PATH: ${agent_loop_config_path}"
  echo "CONVERSATION_EXPORT_DIR: ${conversation_export_dir}"
  echo "SFT_OUTPUT_DIR: ${sft_output_dir}"

  local val_files="[${val_parquet}]"
  EXP_NAME="${EXP_NAME_BASE}" \
  EVAL_NAME="${eval_name}" \
  VAL_FILES="${val_files}" \
  AGENT_LOOP_CONFIG_PATH="${agent_loop_config_path}" \
  CONVERSATION_EXPORT_DIR="${conversation_export_dir}" \
  bash "${INSIGHT_DOC_ROOT}/verl/recipe/vsearch/run.vr2/verl_latest.vr2.insight_doc_region_loc.sh"

  python "${VERL_ROOT}/scripts/convert_exported_convos_to_qwen3_vl_zoom_in_sft.py" \
    --input-dir "${conversation_export_dir}" \
    --output-dir "${sft_output_dir}" \
    --val-ratio 0.0 \
    --output-parquet-name "sft_data.parquet" \
    --stitch-runtime-hints \
    --only-correct-answers \
    --num-workers "${NUM_WORKERS}"
}

run_one "025" \
  "${INPUT_ROOT}/rescale025/sft_data.vreasoner_v2_hard_retry_rescale025.parquet" \
  "recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_max_call10.yaml"

run_one "035" \
  "${INPUT_ROOT}/rescale035/sft_data.vreasoner_v2_hard_retry_rescale035.parquet" \
  "recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_initial_0_35_max_calls10.yaml"

run_one "05" \
  "${INPUT_ROOT}/rescale05/sft_data.vreasoner_v2_hard_retry_rescale05.parquet" \
  "recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_initial_0_5_max_calls10.yaml"

echo "Done: answerable hard rescale 0.25 / 0.35 / 0.5"
