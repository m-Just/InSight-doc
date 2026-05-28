#!/bin/bash

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

source ./recipe/vsearch/_base.sh

# export CUDA_LAUNCH_BLOCKING=1

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:$PYTHONPATH"
export VERL_PROJ_DIR="$PWD"
# export MODEL_PATH='Qwen/Qwen2.5-VL-7B-Instruct'
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
# export MODEL_PATH='m-Just/InSight-o3-vS'
export WORK_DIR=/scratch/ywxzml3j/likaican/mms1_rl
export PROJECT_NAME="multi_agent_vsearch"
export EXP_NAME="${EXP_NAME:-arxiv_0307_sample_qwen3_region_loc_bbox_issue_fixed}"
# export EVAL_NAME='my_eval'
# export EVAL_NAME='veqa_batch_0350_mveqa_batch_0352_sample_50_maxp40_simple_prompt'
# export EVAL_NAME='veqa_batch_0350_mveqa_batch_0352_sample_10_maxp40_simple_prompt_answer_tag_export.run2'
export EVAL_NAME="${EVAL_NAME:-veqa_batch_0350_mveqa_batch_0352_sample_102_maxp40_simple_prompt_answer_tag_export}"

# AGENT_LOOP_CONFIG_PATH="recipe/vsearch/config/agent_gpt-5-mini_vr2.yaml"
AGENT_LOOP_CONFIG_PATH="${AGENT_LOOP_CONFIG_PATH:-recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2.yaml}"
export JUDGE_MODEL=gpt-5-nano
# export OPENAI_BASE_URL='https://globalai.vip/v1'
# export OPENAI_BASE_URL='https://hk.globalai.vip/v1'
# export OPENAI_API_KEY='sk-FrezcU4JuYifou4Mw0QKc8RRtyztHHd1jv2w6bMw3Xr6ihar'
# export OPENAI_API_KEY='sk-kcDJ68GnTGnQneEy7BVT2a2X3J3DUIrI73OVe16CV9E5t02O'
# export OPENAI_API_KEY='sk-clSsrDVChQi76UbAJz5stAnGjEXvLjzjsU3gOqFGUG7r0xmO'
# export OPENAI_BASE_URL='https://az.gptplus5.com/v1'
# export OPENAI_API_KEY='sk-TF4MriCPDbUsIaUXRY21hfEDu2OlnjDBssqMnX3i8I3RvmIx'
# export OPENAI_CLIENT_TIMEOUT=60
export OPENAI_CLIENT_TIMEOUT=1800

export TRAIN_FILES='[/scratch/ywxzml3j/likaican/data/InSightDocRegionLocalization/all-vsearcher_qwen3vl.parquet]'
# export VAL_FILES='[/scratch/ywxzml3j/likaican/data/vstar_bench/full-deepeyes_prompt-vreasoner.parquet,/scratch/ywxzml3j/likaican/data/o3_bench/release_v1_3-deepeyes_prompt-vreasoner.parquet]'
# export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_filtered_cs_reduced_sample_50_max_pages_50-vreasoner_v2.sample_10.parquet]'
# export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_sample_50_maxp40-vreasoner_v2.parquet]'
# export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_sample_10_maxp40-vreasoner_v2.parquet]'
export VAL_FILES="${VAL_FILES:-[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_veqa_batch_0350_mveqa_batch_0352_maxp40-vreasoner_v2.test.parquet]}"
export NUM_VAL_TRIALS=1

USE_DYNAMIC_BSZ=True
# If use dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=4K
MAX_IMG_TOKENS_VAL=4K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=0         # dump a small number of validation samples each step

VAL_BEFORE_TRAIN=True
VAL_ONLY=True
if [ "$VAL_ONLY" == "True" ]; then
    CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME/$EVAL_NAME}"
else
    CONVERSATION_EXPORT_DIR="${CONVERSATION_EXPORT_DIR:-$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME}"
fi
export VSEARCH_PROFILE_DIR="${VSEARCH_PROFILE_DIR:-$CONVERSATION_EXPORT_DIR/_profile}"


run_experiment \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-8}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    actor_rollout_ref.rollout.load_format="${LOAD_FORMAT:-dummy}" \
    trainer.val_only_hf_model_rollout="${VAL_ONLY_HF_MODEL_ROLLOUT:-false}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    +data.batch_sampler.weights.insight_doc_region_localization=1.0 \
    data.max_response_length=49152 \
    data.max_prompt_length=49152 \
    actor_rollout_ref.rollout.max_model_len=$(( 64 * 1024 )) \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 64 * 1024 )) \
    +actor_rollout_ref.rollout.agent.vsearcher_loop_cls=VSearcherLoopQwen3VL \
    +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="$CONVERSATION_EXPORT_DIR" \
    +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_resume_mode=skip_completed \
    +actor_rollout_ref.rollout.agent.vreasoner_v2_profile_dir="$VSEARCH_PROFILE_DIR" \
    actor_rollout_ref.model.custom_chat_template=null \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
    custom_reward_function.reward_kwargs.reward_type=basic_weighted_addition \
    custom_reward_function.reward_kwargs.reward_weights.tool=0.0 \
    custom_reward_function.reward_kwargs.format_reward.simple=True \
    trainer.debug_skip_worker_init="${DEBUG_SKIP_WORKER_INIT:-False}" \
    data.val_batch_size=16 \
    data.validation_shuffle=False \
    custom_reward_function.reward_kwargs.max_retries=15 \
    custom_reward_function.reward_kwargs.retry_interval=90 \
    "$@"

echo "CONVERSATION_EXPORT_DIR: $CONVERSATION_EXPORT_DIR"
echo "VSEARCH_PROFILE_DIR: $VSEARCH_PROFILE_DIR"




# AGENT_LOOP_CONFIG_PATH="recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2.yaml"
# export EVAL_NAME="veqa_batch_0350_r2_mveqa_batch_0352_r2-dpi200_aug_noaug_maxp40-zoom_factor2-gpt"
# if [ "$VAL_ONLY" == "True" ]; then
#     CONVERSATION_EXPORT_DIR="$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME/$EVAL_NAME"
# else
#     CONVERSATION_EXPORT_DIR="$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME"
# fi

# if [ "$MODEL_PATH" == "Qwen/Qwen3-VL-8B-Instruct" ]; then
#     run_experiment \
#         +data.batch_sampler.weights.insight_doc_region_localization=1.0 \
#         data.max_response_length=32768 \
#         data.max_prompt_length=32768 \
#         data.validation_max_prompt_length=32768 \
#         actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH" \
#         +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 64 * 1024 )) \
#         +actor_rollout_ref.rollout.agent.vsearcher_loop_cls=VSearcherLoopQwen3VL \
#         +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="$CONVERSATION_EXPORT_DIR" \
#         actor_rollout_ref.model.custom_chat_template=null \
#         actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
#         actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
#         actor_rollout_ref.rollout.val_kwargs.top_k=20 \
#         actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
#         custom_reward_function.reward_kwargs.reward_type=basic_weighted_addition \
#         custom_reward_function.reward_kwargs.reward_weights.tool=0.0 \
#         custom_reward_function.reward_kwargs.format_reward.simple=True \
#         "$@"

# else
#     echo "Invalid model path: $MODEL_PATH"
#     exit 1
# fi


# AGENT_LOOP_CONFIG_PATH="recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor3.yaml"
# export EVAL_NAME="veqa_batch_0350_r2_mveqa_batch_0352_r2-dpi200_aug_noaug_maxp40-zoom_factor3-gpt"
# if [ "$VAL_ONLY" == "True" ]; then
#     CONVERSATION_EXPORT_DIR="$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME/$EVAL_NAME"
# else
#     CONVERSATION_EXPORT_DIR="$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME"
# fi

# if [ "$MODEL_PATH" == "Qwen/Qwen3-VL-8B-Instruct" ]; then
#     run_experiment \
#         +data.batch_sampler.weights.insight_doc_region_localization=1.0 \
#         data.max_response_length=32768 \
#         data.max_prompt_length=32768 \
#         data.validation_max_prompt_length=32768 \
#         actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH" \
#         +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 64 * 1024 )) \
#         +actor_rollout_ref.rollout.agent.vsearcher_loop_cls=VSearcherLoopQwen3VL \
#         +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="$CONVERSATION_EXPORT_DIR" \
#         actor_rollout_ref.model.custom_chat_template=null \
#         actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
#         actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
#         actor_rollout_ref.rollout.val_kwargs.top_k=20 \
#         actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
#         custom_reward_function.reward_kwargs.reward_type=basic_weighted_addition \
#         custom_reward_function.reward_kwargs.reward_weights.tool=0.0 \
#         custom_reward_function.reward_kwargs.format_reward.simple=True \
#         "$@"

# else
#     echo "Invalid model path: $MODEL_PATH"
#     exit 1
# fi


# AGENT_LOOP_CONFIG_PATH="recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_initial_0_5.yaml"
# export EVAL_NAME="veqa_batch_0350_r2_mveqa_batch_0352_r2-dpi200_aug_noaug_maxp40-zoom_factor2_initial_0_5-gpt"
# if [ "$VAL_ONLY" == "True" ]; then
#     CONVERSATION_EXPORT_DIR="$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME/$EVAL_NAME"
# else
#     CONVERSATION_EXPORT_DIR="$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME"
# fi

# if [ "$MODEL_PATH" == "Qwen/Qwen3-VL-8B-Instruct" ]; then
#     run_experiment \
#         +data.batch_sampler.weights.insight_doc_region_localization=1.0 \
#         data.max_response_length=32768 \
#         data.max_prompt_length=32768 \
#         data.validation_max_prompt_length=32768 \
#         actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH" \
#         +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 64 * 1024 )) \
#         +actor_rollout_ref.rollout.agent.vsearcher_loop_cls=VSearcherLoopQwen3VL \
#         +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="$CONVERSATION_EXPORT_DIR" \
#         actor_rollout_ref.model.custom_chat_template=null \
#         actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
#         actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
#         actor_rollout_ref.rollout.val_kwargs.top_k=20 \
#         actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
#         custom_reward_function.reward_kwargs.reward_type=basic_weighted_addition \
#         custom_reward_function.reward_kwargs.reward_weights.tool=0.0 \
#         custom_reward_function.reward_kwargs.format_reward.simple=True \
#         "$@"

# else
#     echo "Invalid model path: $MODEL_PATH"
#     exit 1
# fi


# AGENT_LOOP_CONFIG_PATH="recipe/vsearch/config/agent_gpt-5-mini_vr2_zoom_factor2_max_calls10.yaml"
# export EVAL_NAME="veqa_batch_0350_r2_mveqa_batch_0352_r2-dpi200_aug_noaug_maxp40-zoom_factor2_max_calls10-gpt"
# if [ "$VAL_ONLY" == "True" ]; then
#     CONVERSATION_EXPORT_DIR="$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME/$EVAL_NAME"
# else
#     CONVERSATION_EXPORT_DIR="$WORK_DIR/exported_conversations/$PROJECT_NAME/$EXP_NAME"
# fi

# if [ "$MODEL_PATH" == "Qwen/Qwen3-VL-8B-Instruct" ]; then
#     run_experiment \
#         +data.batch_sampler.weights.insight_doc_region_localization=1.0 \
#         data.max_response_length=32768 \
#         data.max_prompt_length=32768 \
#         data.validation_max_prompt_length=32768 \
#         actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH" \
#         +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 64 * 1024 )) \
#         +actor_rollout_ref.rollout.agent.vsearcher_loop_cls=VSearcherLoopQwen3VL \
#         +actor_rollout_ref.rollout.agent.vreasoner_v2_conversation_export_dir="$CONVERSATION_EXPORT_DIR" \
#         actor_rollout_ref.model.custom_chat_template=null \
#         actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
#         actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
#         actor_rollout_ref.rollout.val_kwargs.top_k=20 \
#         actor_rollout_ref.rollout.val_kwargs.presence_penalty=1.5 \
#         custom_reward_function.reward_kwargs.reward_type=basic_weighted_addition \
#         custom_reward_function.reward_kwargs.reward_weights.tool=0.0 \
#         custom_reward_function.reward_kwargs.format_reward.simple=True \
#         "$@"

# else
#     echo "Invalid model path: $MODEL_PATH"
#     exit 1
# fi
