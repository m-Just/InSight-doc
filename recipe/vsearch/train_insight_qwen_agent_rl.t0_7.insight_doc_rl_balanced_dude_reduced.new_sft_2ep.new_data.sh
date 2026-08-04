#!/bin/bash

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

source ./recipe/vsearch/_base.sh

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:$PYTHONPATH"
export VERL_PROJ_DIR="$PWD"
export MODEL_PATH='/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260519/full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch2_fp32_scratch/sft_checkpoints/global_step_1118/huggingface'
# export WORK_DIR=/scratch/ywxzml3j/likaican/mms1_rl
export WORK_DIR=/home/ywxzml3j/ywxzml3juser40/mms1_rl
export PROJECT_NAME="insight_doc"
# export EXP_NAME="insight_doc_rl_balanced_dude_reduced_qwen3_insight_qwen_agent_rl_t0_7_def_sparams"
export EXP_NAME="insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_new_sft_2ep_new_data"
export EVAL_NAME="dude200_mmlongbench200_o3bench0502_insight_qwen_agent"
export JUDGE_MODEL=gpt-5-nano
export OPENAI_CLIENT_TIMEOUT=60

# export TRAIN_FILES='[/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent.parquet]'
# export TRAIN_FILES='[/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/insight_doc_rl_balanced_dude_reduced_merged-insight_qwen_agent_estprompt_le11000.parquet]'
export TRAIN_FILES='[/scratch/ywxzml3j/likaican/data/insight_doc/answerable_plus_synthetic_unanswerable_vr2_wrong_rl_20260519_per_sample_rescale_maxarea3500x3500_le11000_u25_with_initial_rescale_map_reduced_source_u20_arxiv_map_4of5-insight_qwen_agent.parquet]'  # this is only le11000 filtered data
export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet]'
export NUM_VAL_TRIALS=1

USE_DYNAMIC_BSZ=True
# If use dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=16K  # roughly 3580 * 3580 pixels per image
MAX_IMG_TOKENS_VAL=16K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=0

VAL_BEFORE_TRAIN=False
VAL_BATCH_SIZE=32

    # +data.batch_sampler.weights.insight_doc_rl=1.0 \
run_experiment \
    data.batch_sampler.enabled=False \
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
    trainer.total_training_steps=2000 \
    custom_reward_function.reward_kwargs.max_retries=15 \
    custom_reward_function.reward_kwargs.retry_interval=90 \
    +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${EXP_NAME}/${EVAL_NAME}" \
    "$@"
