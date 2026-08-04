#!/bin/bash

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

source ./recipe/vsearch/_base.sh

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:$PYTHONPATH"
export VERL_PROJ_DIR="$PWD"
export MODEL_PATH='/scratch/ywxzml3j/likaican/temp/full_sft_both_higher_dpi_unans02503505_fp32_sp4_epoch2_scratch_20260519/full_sft_both_higher_dpi_unans02503505_len65536_sp4_bs32_freeze_vt_epoch2_fp32_scratch/sft_checkpoints/global_step_1118/huggingface'
# export WORK_DIR=/scratch/ywxzml3j/likaican/mms1_rl
export WORK_DIR=/home/ywxzml3j/ywxzml3juser40/mms1_rl
export PROJECT_NAME="insight_doc"
export EXP_NAME="insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_new_sft_2ep_new_data_sp4_rl16k_prompt24k_weighted_refill_simple_source_unans014_mc_false_e05_arxiv_struct1k_llm_judge_legacy_from450"
export EVAL_NAME="dude200_mmlongbench200_o3bench0502_insight_qwen_agent"
export JUDGE_MODEL=gpt-5-nano
export OPENAI_CLIENT_TIMEOUT=60

unset RESUME_FROM_STEP

export TRAIN_FILES='[/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/notes/generated/rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e05_plus_arxiv_struct1k_llm_20260722/insight_doc_rl_16k_prompt24k_r05_to_r035_simple_datasource_plus_mc_false_e_arxiv_struct1k_llm-insight_qwen_agent.parquet]'
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

SOURCE_EXP_NAME="insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_def_sparams_new_sft_2ep_new_data_sp4_rl16k_prompt24k_weighted_refill_simple_source_unans014_mc_false_e05_judge_legacy"
SOURCE_CKPT_STEP_DIR="/home/ywxzml3j/ywxzml3juser40/mms1_rl/ckpts/insight_doc/${SOURCE_EXP_NAME}/global_step_450"
FORK_CKPT_STEP_DIR="${WORK_DIR}/ckpts/${PROJECT_NAME}/${EXP_NAME}/global_step_450"

prepare_fork_checkpoint() {
    if [ ! -d "${SOURCE_CKPT_STEP_DIR}/actor" ]; then
        echo "Missing source actor checkpoint: ${SOURCE_CKPT_STEP_DIR}/actor" >&2
        exit 1
    fi
    if [ -e "${SOURCE_CKPT_STEP_DIR}/data.pt" ]; then
        echo "Source data.pt exists but will intentionally not be linked: ${SOURCE_CKPT_STEP_DIR}/data.pt"
        echo "This lets the fork use the new train parquet/sampler state instead of the old dataloader state."
    fi

    mkdir -p "${FORK_CKPT_STEP_DIR}"
    if [ -e "${FORK_CKPT_STEP_DIR}/data.pt" ]; then
        echo "Refusing to use fork checkpoint because data.pt exists: ${FORK_CKPT_STEP_DIR}/data.pt" >&2
        echo "Remove it manually if you intentionally want a fresh-dataloader fork." >&2
        exit 1
    fi

    if [ -L "${FORK_CKPT_STEP_DIR}/actor" ]; then
        linked_target="$(readlink "${FORK_CKPT_STEP_DIR}/actor")"
        if [ "${linked_target}" != "${SOURCE_CKPT_STEP_DIR}/actor" ]; then
            echo "Unexpected existing actor symlink: ${FORK_CKPT_STEP_DIR}/actor -> ${linked_target}" >&2
            exit 1
        fi
    elif [ -e "${FORK_CKPT_STEP_DIR}/actor" ]; then
        echo "Refusing to overwrite existing fork actor path: ${FORK_CKPT_STEP_DIR}/actor" >&2
        exit 1
    else
        ln -s "${SOURCE_CKPT_STEP_DIR}/actor" "${FORK_CKPT_STEP_DIR}/actor"
    fi

    echo "Prepared fork checkpoint: ${FORK_CKPT_STEP_DIR}"
    echo "Actor checkpoint source: ${SOURCE_CKPT_STEP_DIR}/actor"
}

prepare_fork_checkpoint

# Sampling weights are stored in:
# recipe/vsearch/config/insight_doc_rl_category_answerability_sampling_weights_prompt24k_simple_source_custom_unans014_mc_false_e05_plus_arxiv_struct1k_llm_20260722.yaml.
# The train parquet uses simplified data_source keys:
#   - original rows: {category}_{answerability}
#   - MC false-E calibration rows: {category}_answerable_mc_false_e
#   - LLM-rewritten structural rows:
#       arxiv_struct_{single,list}_answerable_{r025,r035,r05}
# This fork resumes actor/optimizer/lr-scheduler state from the source run's
# global_step_450 checkpoint without modifying original checkpoint files. It
# does not load the source run's data.pt, so the new dataset and refill sampler
# start from their own fresh state.
run_experiment \
    +custom_reward_function.reward_kwargs.insight_qwen_judge_mode=legacy \
    data.batch_sampler.enabled=True \
    data.batch_sampler.class_name=VSearchWeightedRandomRefillBatchSampler \
    data.batch_sampler.weights_file=/scratch/ywxzml3j/likaican/src/verl-qwen3-vl/recipe/vsearch/config/insight_doc_rl_category_answerability_sampling_weights_prompt24k_simple_source_custom_unans014_mc_false_e05_plus_arxiv_struct1k_llm_20260722.yaml \
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
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${FORK_CKPT_STEP_DIR}" \
    custom_reward_function.reward_kwargs.max_retries=15 \
    custom_reward_function.reward_kwargs.retry_interval=90 \
    +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${EXP_NAME}/${EVAL_NAME}" \
    "$@"
