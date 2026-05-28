#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TELEGRAM_WRAPPER="$SCRIPT_DIR/run_with_telegram_notification.sh"

source /home/ywxzml3j/ywxzml3juser40/miniconda3/etc/profile.d/conda.sh
conda activate vllm-latest

export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:/scratch/ywxzml3j/likaican/src/Qwen-Agent:${PYTHONPATH:-}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set}"

export WANDB_PROJECT="${WANDB_PROJECT:-insight_doc}"
export WANDB_ENTITY="${WANDB_ENTITY:-mjust-lkc-hkust2}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ywxzml3j/ywxzml3juser40/insight_doc/outputs}"
export WORK_DIR="${WORK_DIR:-$OUTPUT_ROOT/iq_base_eval_default_sampling_insight_rl15360}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export PROJECT_NAME="${PROJECT_NAME:-insight_doc}"
export EXP_NAME="${EXP_NAME:-iq_base_eval_default_sampling_insight_rl15360_r7}"
export EVAL_NAME="${EVAL_NAME:-heldout}"
export VAL_ONLY="True"
export TRAIN_FILES="${TRAIN_FILES:-[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet]}"
export VAL_FILES="${VAL_FILES:-[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet]}"
export NUM_VAL_TRIALS="${NUM_VAL_TRIALS:-1}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"
export MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE="${MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE:-0}"
export LOGGER="${LOGGER:-['console']}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-$WORK_DIR/exported_conversations}"
export AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-recipe/vsearch/config/agent_insight_qwen_agent.yaml}"
export TOOL_MAX_USER_TURNS="${TOOL_MAX_USER_TURNS:-6}"
export TOOL_MAX_ASSISTANT_TURNS="${TOOL_MAX_ASSISTANT_TURNS:-7}"
export QWEN_TOOL_LIST="${QWEN_TOOL_LIST:-[image_zoom_in_tool_qwen3vl]}"
export DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-17408}"
export DATA_VALIDATION_MAX_PROMPT_LENGTH="${DATA_VALIDATION_MAX_PROMPT_LENGTH:-17408}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-32768}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
export LOAD_FORMAT="${LOAD_FORMAT:-}"
export LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
export LORA_RANK="${LORA_RANK:-0}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
export VAL_TOP_P="${VAL_TOP_P:-0.8}"
export VAL_TOP_K="${VAL_TOP_K:-20}"
export VAL_PRESENCE_PENALTY="${VAL_PRESENCE_PENALTY:-1.5}"
export VAL_REPETITION_PENALTY="${VAL_REPETITION_PENALTY:-1.0}"
export TMPDIR="${TMPDIR:-/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rbi7}"
export LAUNCH_LOG_PATH="${LAUNCH_LOG_PATH:-$WORK_DIR/${EXP_NAME}.launch.log}"
export TELEGRAM_NOTIFY_ON_FINISH="${TELEGRAM_NOTIFY_ON_FINISH:-1}"
export TELEGRAM_NOTIFY_LABEL="${TELEGRAM_NOTIFY_LABEL:-$EXP_NAME}"

mkdir -p "$WORK_DIR" "$RAY_TMPDIR" "$CONVERSATION_EXPORT_DIR" "$(dirname "$LAUNCH_LOG_PATH")"

if [[ "${TELEGRAM_NOTIFY_ON_FINISH}" == "1" && "${TELEGRAM_WRAPPED:-0}" != "1" && -x "$TELEGRAM_WRAPPER" ]]; then
  export TELEGRAM_WRAPPED=1
  exec "$TELEGRAM_WRAPPER" --label "$TELEGRAM_NOTIFY_LABEL" -- "$0" "$@"
fi

exec > >(tee -a "$LAUNCH_LOG_PATH") 2>&1

cd "$REPO_ROOT"
source recipe/vsearch/_base.sh

hydra_args=(
  trainer.n_gpus_per_node=4
  data.max_prompt_length="$DATA_MAX_PROMPT_LENGTH"
  data.validation_max_prompt_length="$DATA_VALIDATION_MAX_PROMPT_LENGTH"
  actor_rollout_ref.model.custom_chat_template=null
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN"
  actor_rollout_ref.rollout.agent.default_agent_loop=insight_qwen_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH"
  +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="$CONVERSATION_EXPORT_DIR"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$TOOL_MAX_USER_TURNS"
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$TOOL_MAX_ASSISTANT_TURNS"
  actor_rollout_ref.rollout.multi_turn.qwen_tool_list="$QWEN_TOOL_LIST"
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE"
  actor_rollout_ref.rollout.val_kwargs.top_p="$VAL_TOP_P"
  actor_rollout_ref.rollout.val_kwargs.top_k="$VAL_TOP_K"
  actor_rollout_ref.rollout.val_kwargs.presence_penalty="$VAL_PRESENCE_PENALTY"
  actor_rollout_ref.rollout.val_kwargs.repetition_penalty="$VAL_REPETITION_PENALTY"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len="$VLLM_MAX_MODEL_LEN"
  +actor_rollout_ref.rollout.engine_kwargs.vllm.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION"
)

if [[ -n "$LORA_ADAPTER_PATH" ]]; then
  if [[ "$LORA_RANK" == "0" ]]; then
    echo "LORA_RANK must be set to a positive value when LORA_ADAPTER_PATH is set" >&2
    exit 2
  fi
  hydra_args+=(
    actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH"
    actor_rollout_ref.model.lora_rank="$LORA_RANK"
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
    actor_rollout_ref.rollout.load_format="${LOAD_FORMAT:-safetensors}"
    trainer.val_only_hf_model_rollout=false
  )
elif [[ -n "$LOAD_FORMAT" ]]; then
  hydra_args+=(actor_rollout_ref.rollout.load_format="$LOAD_FORMAT")
fi

if (($# > 0)); then
  hydra_args+=("$@")
fi

run_experiment "${hydra_args[@]}"
