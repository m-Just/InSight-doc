#!/bin/bash

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

source ./recipe/vsearch/_base.sh

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:$PYTHONPATH"
export VERL_PROJ_DIR="$PWD"
export MODEL_PATH='/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_full_sft_all_convos_0426_batch_lr_sweep/insight_qwen_agent_full_sft_all_convos_0426_lr5e-6_cosine_minlr5e-7_len32768_bs32_full_clean_data_freeze_vt/sft_checkpoints/global_step_1052/huggingface'
# export WORK_DIR=/scratch/ywxzml3j/likaican/mms1_rl
export WORK_DIR=/home/ywxzml3j/ywxzml3juser40/mms1_rl
export PROJECT_NAME="insight_doc"
# export EXP_NAME="insight_doc_rl_balanced_dude_reduced_qwen3_insight_qwen_agent_rl_t0_7_def_sparams"
export EXP_NAME="insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_with_no_zoom_needed_data_and_supereasy20pct"
export EVAL_NAME="dude200_mmlongbench200_o3bench0502_insight_qwen_agent"
export JUDGE_MODEL=gpt-5-nano
export OPENAI_CLIENT_TIMEOUT=60

# export TRAIN_FILES='[/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent.parquet]'
# export TRAIN_FILES='[/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent_estprompt_le11000.parquet]'
# export TRAIN_FILES='[/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.parquet]'  # this is only le11000 filtered data
export TRAIN_FILES='[/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25_plus_easy32b_plus_super_easy8b_replace20cap_oldmarginals_from_tmp-insight_qwen_agent.parquet]'
# export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet]'
export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet]'
export NUM_VAL_TRIALS=1

USE_DYNAMIC_BSZ=True
# If use dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=4K  # roughly 2000 * 2000 pixels per image
MAX_IMG_TOKENS_VAL=4K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=0

VAL_BEFORE_TRAIN=False
VAL_BATCH_SIZE=32

run_experiment \
    +data.batch_sampler.weights.insight_doc_rl=1.0 \
    data.max_response_length=12288 \
    data.max_prompt_length=12288 \
    data.validation_max_prompt_length=12288 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.agent.default_agent_loop=insight_qwen_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500.yaml" \
    actor_rollout_ref.rollout.multi_turn.qwen_tool_list=[image_zoom_in_tool_qwen3vl] \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 24 * 1024 )) \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=11 \
    actor_rollout_ref.model.custom_chat_template=null \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.presence_penalty=1.5 \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
    actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.subagent_advantage_estimator=null \
    trainer.test_freq=50 \
    trainer.total_training_steps=150 \
    custom_reward_function.reward_kwargs.max_retries=15 \
    custom_reward_function.reward_kwargs.retry_interval=90 \
    +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${EXP_NAME}/${EVAL_NAME}" \
    "$@"
