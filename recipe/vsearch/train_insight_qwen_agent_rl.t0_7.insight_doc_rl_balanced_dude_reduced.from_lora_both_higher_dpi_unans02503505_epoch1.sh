#!/bin/bash

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

source ./recipe/vsearch/_base.sh

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:$PYTHONPATH"
export VERL_PROJ_DIR="$PWD"

# For LoRA continuation, keep MODEL_PATH as the base HF model and pass the
# PEFT adapter through actor_rollout_ref.model.lora_adapter_path below.
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
export LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-/scratch/ywxzml3j/likaican/temp/insight_qwen_agent_lora_unanswerable_medium_lane0_20260517/lora_both_higher_dpi_unanswerable02503505_len65536_sp2_bs32_rank32_alpha64_freeze_vt_medium_only_epoch1/sft_checkpoints/global_step_559/huggingface_base_lora_export/lora_adapter}"

# export WORK_DIR=/scratch/ywxzml3j/likaican/mms1_rl
export WORK_DIR="${WORK_DIR:-/home/ywxzml3j/ywxzml3juser40/mms1_rl}"
export PROJECT_NAME="${PROJECT_NAME:-insight_doc}"
export EXP_NAME="${EXP_NAME:-insight_doc_rl_balanced_dude_reduced_u25_qwen3_insight_qwen_agent_rl_t0_7_from_lora_both_higher_dpi_unans02503505_e1}"
export EVAL_NAME="${EVAL_NAME:-dude200_mmlongbench200_o3bench0502_insight_qwen_agent}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-nano}"
export OPENAI_CLIENT_TIMEOUT="${OPENAI_CLIENT_TIMEOUT:-60}"

export TRAIN_FILES="${TRAIN_FILES:-[/scratch/ywxzml3j/likaican/data/insight_doc/insight_doc_rl_balanced_dude_reduced_merged_u25-insight_qwen_agent.parquet]}"
export VAL_FILES="${VAL_FILES:-[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-insight_qwen_agent.test.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/dude200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/mmlongbench200-insight_qwen_agent.parquet,/scratch/ywxzml3j/likaican/temp/eval_data_0502_parquets/o3bench0502-insight_qwen_agent.parquet]}"
export NUM_VAL_TRIALS="${NUM_VAL_TRIALS:-1}"

USE_DYNAMIC_BSZ=True

MAX_IMG_TOKENS_TRAIN=4K
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
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=11 \
    actor_rollout_ref.model.custom_chat_template=null \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}" \
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
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    trainer.test_freq=50 \
    trainer.total_training_steps=1000 \
    custom_reward_function.reward_kwargs.max_retries=15 \
    custom_reward_function.reward_kwargs.retry_interval=90 \
    +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="/scratch/ywxzml3j/likaican/mms1_rl/exported_conversations/multi_agent_vsearch/${EXP_NAME}/${EVAL_NAME}" \
    "$@"
