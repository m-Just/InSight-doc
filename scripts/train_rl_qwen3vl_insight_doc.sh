#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

: "${MODEL_PATH:?Set MODEL_PATH to the SFT HF checkpoint used to initialize RL.}"
: "${TRAIN_FILES:?Set TRAIN_FILES to the released RL parquet path/list.}"
: "${VAL_FILES:?Set VAL_FILES to one or more validation parquet paths.}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the judge/reward pipeline.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL for the OpenAI-compatible judge endpoint.}"

source "$REPO_ROOT/recipe/vsearch/_base.sh"

export VERL_ROOT="${VERL_ROOT:-$REPO_ROOT/verl}"
export VERL_PROJ_DIR="${VERL_PROJ_DIR:-$REPO_ROOT}"
export VERL_CONFIG_DIR="${VERL_CONFIG_DIR:-$REPO_ROOT/recipe/vsearch/config}"
export PYTHONPATH="$REPO_ROOT:$VERL_ROOT${INSIGHT_O3_ROOT:+:$INSIGHT_O3_ROOT}${QWEN_AGENT_ROOT:+:$QWEN_AGENT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export WORK_DIR="${WORK_DIR:-$REPO_ROOT/outputs/rl}"
export PROJECT_NAME="${PROJECT_NAME:-insight_doc}"
export EXP_NAME="${EXP_NAME:-insight_doc_rl_qwen3vl8b}"
export LOGGER="${LOGGER:-['console']}"
export EVAL_NAME="${EVAL_NAME:-insight_doc_eval}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"
export FALLBACK_JUDGE_MODEL="${FALLBACK_JUDGE_MODEL:-}"
export OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT:-60}"

export NUM_VAL_TRIALS="${NUM_VAL_TRIALS:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE="${MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE:-0}"
export MAX_IMG_TOKENS_TRAIN="${MAX_IMG_TOKENS_TRAIN:-16K}"
export MAX_IMG_TOKENS_VAL="${MAX_IMG_TOKENS_VAL:-16K}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"

WEIGHTS_FILE="${WEIGHTS_FILE:-$REPO_ROOT/recipe/vsearch/config/insight_doc_rl_sampling_weights_release.yaml}"
AGENT_CONFIG="${AGENT_CONFIG:-recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500.yaml}"
EXPORT_DIR="${EXPORT_DIR:-$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME/$EVAL_NAME}"

echo "[rl] repo=$REPO_ROOT"
echo "[rl] model_path=$MODEL_PATH"
echo "[rl] train_files=$TRAIN_FILES"
echo "[rl] val_files=$VAL_FILES"
echo "[rl] weights_file=$WEIGHTS_FILE"
echo "[rl] work_dir=$WORK_DIR project=$PROJECT_NAME exp=$EXP_NAME"

run_experiment \
  +custom_reward_function.reward_kwargs.insight_qwen_judge_mode="${INSIGHT_QWEN_JUDGE_MODE:-legacy_prompt_v2}" \
  data.batch_sampler.enabled=True \
  data.batch_sampler.class_name=VSearchWeightedRandomRefillBatchSampler \
  data.batch_sampler.weights_file="$WEIGHTS_FILE" \
  data.batch_sampler.stop_after=max_source_exhaustion \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-8192}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-24576}" \
  data.validation_max_prompt_length="${VALIDATION_MAX_PROMPT_LENGTH:-24576}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}" \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${REF_ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=insight_qwen_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_CONFIG" \
  actor_rollout_ref.rollout.multi_turn.qwen_tool_list=[image_zoom_in_tool_qwen3vl] \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len="${VLLM_MAX_MODEL_LEN:-32768}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${TOOL_MAX_USER_TURNS:-10}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${TOOL_MAX_ASSISTANT_TURNS:-11}" \
  actor_rollout_ref.model.custom_chat_template=null \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.8}" \
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K:-20}" \
  actor_rollout_ref.rollout.presence_penalty="${ROLLOUT_PRESENCE_PENALTY:-1.5}" \
  actor_rollout_ref.rollout.repetition_penalty="${ROLLOUT_REPETITION_PENALTY:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-0.8}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:-20}" \
  actor_rollout_ref.rollout.val_kwargs.presence_penalty="${VAL_PRESENCE_PENALTY:-1.5}" \
  actor_rollout_ref.rollout.val_kwargs.repetition_penalty="${VAL_REPETITION_PENALTY:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  algorithm.subagent_advantage_estimator=null \
  trainer.test_freq="${TEST_FREQ:-50}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-2000}" \
  custom_reward_function.reward_kwargs.max_retries="${REWARD_MAX_RETRIES:-15}" \
  custom_reward_function.reward_kwargs.retry_interval="${REWARD_RETRY_INTERVAL:-90}" \
  +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="$EXPORT_DIR" \
  "$@"
