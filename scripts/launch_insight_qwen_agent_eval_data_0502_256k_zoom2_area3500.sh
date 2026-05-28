#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/ywxzml3j/likaican/src/verl-qwen3-vl"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script}"

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
DEFAULT_TRAINED_MODEL_PATH="/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_exp1_lr3e-6_cosine_minlr3e-7_len32768_bs32/sft_checkpoints/global_step_1086/huggingface"
export MODEL_PATH="${MODEL_PATH:-$DEFAULT_TRAINED_MODEL_PATH}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs}"
if [[ "$MODEL_PATH" == "Qwen/Qwen3-VL-8B-Instruct" ]]; then
  DEFAULT_RUN_STEM="qwen3_vl_8b_instruct_eval_data_0502_256k_zoom2_area3500"
elif [[ "$MODEL_PATH" == "$DEFAULT_TRAINED_MODEL_PATH" ]]; then
  DEFAULT_RUN_STEM="insight_qwen_agent_full_sft_all_convos_0426_exp1_lr3e-6_cosine_minlr3e-7_len32768_bs32_eval_data_0502_256k_zoom2_area3500"
else
  DEFAULT_RUN_STEM="$(basename "$MODEL_PATH")_eval_data_0502_256k_zoom2_area3500"
fi
export WORK_DIR="${WORK_DIR:-$OUTPUT_ROOT/$DEFAULT_RUN_STEM}"
export EXP_NAME="${EXP_NAME:-$DEFAULT_RUN_STEM}"
export EVAL_NAME="${EVAL_NAME:-heldout}"
export LOGGER="${LOGGER:-['console','wandb']}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-$WORK_DIR/exported_conversations}"

export AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500.yaml}"
export TOOL_MAX_USER_TURNS="${TOOL_MAX_USER_TURNS:-10}"
export TOOL_MAX_ASSISTANT_TURNS="${TOOL_MAX_ASSISTANT_TURNS:-11}"
export DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-262144}"
export DATA_VALIDATION_MAX_PROMPT_LENGTH="${DATA_VALIDATION_MAX_PROMPT_LENGTH:-262144}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-262144}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

VAL_FILES_JSON="$(python - <<'PY'
import json
paths = [
    "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet",
    "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet",
    "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet",
    "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet",
    "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mpdocvqa200-insight_qwen_agent.parquet",
    "/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet",
]
print(json.dumps(paths))
PY
)"
export VAL_FILES="${VAL_FILES:-${VAL_FILES_JSON}}"

cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/run_iq_ft_eval_default_sampling_rl15360.sh" "$@"
