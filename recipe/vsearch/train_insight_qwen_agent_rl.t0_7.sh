#!/bin/bash

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

source ./recipe/vsearch/_base.sh

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:$PYTHONPATH"
export VERL_PROJ_DIR="$PWD"
export MODEL_PATH='Qwen/Qwen3-VL-8B-Instruct'
# export WORK_DIR=/scratch/ywxzml3j/likaican/mms1_rl
export WORK_DIR=/home/ywxzml3j/ywxzml3juser40/mms1_rl
export PROJECT_NAME="insight_doc"
export EXP_NAME="arxiv_0307_sample_qwen3_insight_qwen_agent_rl_t0_7_def_sparams"
export EVAL_NAME="arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40_insight_qwen_agent"
export JUDGE_MODEL=gpt-5-nano
export OPENAI_CLIENT_TIMEOUT=60

export TRAIN_FILES='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.train.parquet]'
export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet]'
export NUM_VAL_TRIALS=1

USE_DYNAMIC_BSZ=True
# If use dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=4K
MAX_IMG_TOKENS_VAL=4K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=0

VAL_BEFORE_TRAIN=True
VAL_BATCH_SIZE=32

run_experiment \
    +data.batch_sampler.weights.insight_doc_0352=1.0 \
    data.max_response_length=12288 \
    data.max_prompt_length=12288 \
    data.validation_max_prompt_length=12288 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.agent.default_agent_loop=insight_qwen_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_insight_qwen_agent.yaml" \
    actor_rollout_ref.rollout.multi_turn.qwen_tool_list=[image_zoom_in_tool_qwen3vl] \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 24 * 1024 )) \
    actor_rollout_ref.model.custom_chat_template=null \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.presence_penalty=1.5 \
    algorithm.subagent_advantage_estimator=null \
    trainer.test_freq=15 \
    trainer.total_training_steps=500 \
    "$@"
