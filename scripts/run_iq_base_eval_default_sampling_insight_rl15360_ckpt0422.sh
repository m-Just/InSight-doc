#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
export LOGGER="${LOGGER:-['console','wandb']}"
export WANDB_NAME="${WANDB_NAME:-$EXP_NAME}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-15360}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-$WORK_DIR/exported_conversations}"
export TMPDIR="${TMPDIR:-/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rbi7}"

mkdir -p "$WORK_DIR" "$RAY_TMPDIR" "$CONVERSATION_EXPORT_DIR"

cd "$REPO_ROOT"
source recipe/vsearch/_base.sh

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

run_experiment \
  trainer.n_gpus_per_node=8 \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS:-8}" \
  data.max_prompt_length=49152 \
  data.validation_max_prompt_length=49152 \
  actor_rollout_ref.model.custom_chat_template=null \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.agent.default_agent_loop=insight_qwen_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG_PATH:-recipe/vsearch/config/agent_insight_qwen_agent.yaml}" \
  +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="$CONVERSATION_EXPORT_DIR" \
  actor_rollout_ref.rollout.multi_turn.qwen_tool_list="${QWEN_TOOL_LIST:-[image_zoom_in_tool_qwen3vl]}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
  actor_rollout_ref.rollout.val_kwargs.top_k=20 \
  actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
  actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1.0 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=65536 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}"

