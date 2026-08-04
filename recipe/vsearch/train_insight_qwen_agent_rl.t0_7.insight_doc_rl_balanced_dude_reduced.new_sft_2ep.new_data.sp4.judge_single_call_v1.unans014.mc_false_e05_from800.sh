#!/bin/bash

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

source ./recipe/vsearch/_base.sh

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:$PYTHONPATH"
export VERL_PROJ_DIR="$PWD"
export MODEL_PATH='/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260519/full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch2_fp32_scratch/sft_checkpoints/global_step_1118/huggingface'
# export WORK_DIR=/scratch/ywxzml3j/likaican/mms1_rl
export WORK_DIR=/home/ywxzml3j/ywxzml3juser40/mms1_rl
export PROJECT_NAME="insight_doc"
export EXP_NAME="insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_new_sft_2ep_new_data_sp4_rl16k_prompt24k_weighted_refill_simple_source_unans014_mc_false_e05_judge_single_call_v1_from800"
export EVAL_NAME="dude200_mmlongbench200_o3bench0502_insight_qwen_agent"
export JUDGE_MODEL=gpt-5-nano
export OPENAI_CLIENT_TIMEOUT=60

# Forked from:
# insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_new_sft_2ep_new_data_sp4_rl16k_prompt24k_weighted_refill_simple_source_unans014_judge_single_call_v1
# at global_step_800. Override RESUME_FROM_STEP only if intentionally resuming
# from another checkpoint in this fork.
export RESUME_FROM_STEP="${RESUME_FROM_STEP:-800}"

export TRAIN_FILES='[/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e05_20260705/insight_doc_rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e_filtered-insight_qwen_agent.parquet]'
export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlite200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/longdocurl200-insight_qwen_agent.parquet]'
export NUM_VAL_TRIALS=1

USE_DYNAMIC_BSZ=True
# If use dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=16K  # roughly 3580 * 3580 pixels per image
MAX_IMG_TOKENS_VAL=16K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=0

VAL_BEFORE_TRAIN=False
VAL_BATCH_SIZE=32
# Sampling weights are stored in:
# recipe/vsearch/config/insight_doc_rl_category_answerability_sampling_weights_prompt24k_simple_source_custom_unans014_mc_false_e05_20260705.yaml.
# The train parquet uses simplified data_source keys:
#   - original rows: {category}_{answerability}
#   - MC false-E calibration rows: {category}_answerable_mc_false_e
# This variant keeps total unanswerable sampling mass at 14%, assigns 5% mass
# to filtered MC false-E rows, and leaves the remaining 81% on original
# answerable rows. The VSearchWeightedRandomRefillBatchSampler samples sources
# by these probabilities and refills each source-specific row pool after it exhausts.
run_experiment \
    +custom_reward_function.reward_kwargs.insight_qwen_judge_mode=single_call_v1 \
    data.batch_sampler.enabled=True \
    data.batch_sampler.class_name=VSearchWeightedRandomRefillBatchSampler \
    data.batch_sampler.weights_file=/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/recipe/vsearch/config/insight_doc_rl_category_answerability_sampling_weights_prompt24k_simple_source_custom_unans014_mc_false_e05_20260705.yaml \
    data.batch_sampler.stop_after=max_source_exhaustion \
    data.max_response_length=8192 \
    data.max_prompt_length=24576 \
    data.validation_max_prompt_length=24576 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.agent.default_agent_loop=insight_qwen_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_insight_qwen_agent_zoom_factor2_area3500.yaml" \
    actor_rollout_ref.rollout.multi_turn.qwen_tool_list=[image_zoom_in_tool_qwen3vl] \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 32 * 1024 )) \
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
