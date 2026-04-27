#!/bin/bash

cd /scratch/ywxzml3j/likaican/src/verl-qwen3-vl

source ./recipe/vsearch/_base.sh

# export CUDA_LAUNCH_BLOCKING=1

export PYTHONPATH="/scratch/ywxzml3j/likaican/src/InSight-o3:$PYTHONPATH"
export VERL_PROJ_DIR="$PWD"
# export MODEL_PATH='Qwen/Qwen2.5-VL-7B-Instruct'
# export MODEL_PATH='Qwen/Qwen3-VL-8B-Instruct'
export MODEL_PATH='m-Just/InSight-o3-vS'
export WORK_DIR=/scratch/ywxzml3j/likaican/mms1_rl
export PROJECT_NAME="multi_agent_vsearch"
# export EXP_NAME="arxiv_0307_sample_filtered_cs_reduced_sample_50_max_pages_50-vreasoner_v2-sample_10-gpt-5-mini-qwen3-initial-rescale-0.25-prompt-refine10"
export EXP_NAME="arxiv_0307_sample_filtered_cs_reduced_sample_50_max_pages_50-vreasoner_v2-sample_10-insight-o3-vS-initial-rescale-0.25-prompt-refine10-correct_image_process"
export EVAL_NAME='my_eval'

# export API_MODEL_FOR_AGENT=gpt-5-mini
# export API_MODEL_FOR_AGENT=gemini-2.5-flash
export JUDGE_MODEL=gpt-5-nano
export OPENAI_BASE_URL='https://globalai.vip/v1'
# export OPENAI_BASE_URL='https://hk.globalai.vip/v1'
# export OPENAI_API_KEY='sk-FrezcU4JuYifou4Mw0QKc8RRtyztHHd1jv2w6bMw3Xr6ihar'
# export OPENAI_API_KEY='sk-kcDJ68GnTGnQneEy7BVT2a2X3J3DUIrI73OVe16CV9E5t02O'
export OPENAI_API_KEY='sk-clSsrDVChQi76UbAJz5stAnGjEXvLjzjsU3gOqFGUG7r0xmO'
# export OPENAI_BASE_URL='https://az.gptplus5.com/v1'
# export OPENAI_API_KEY='sk-TF4MriCPDbUsIaUXRY21hfEDu2OlnjDBssqMnX3i8I3RvmIx'
export OPENAI_CLIENT_TIMEOUT=60

export TRAIN_FILES='[]'
# export VAL_FILES='[/scratch/ywxzml3j/likaican/data/vstar_bench/full-deepeyes_prompt-vreasoner.parquet,/scratch/ywxzml3j/likaican/data/o3_bench/release_v1_3-deepeyes_prompt-vreasoner.parquet]'
export VAL_FILES='[/scratch/ywxzml3j/likaican/temp/arxiv_0307_sample_filtered_cs_reduced_sample_50_max_pages_50-vreasoner_v2.sample_10.parquet]'
export NUM_VAL_TRIALS=1

USE_DYNAMIC_BSZ=True
# If use dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=4K
MAX_IMG_TOKENS_VAL=4K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=10         # dump a small number of validation samples each step

VAL_BEFORE_TRAIN=True
VAL_ONLY=True


if [ "$MODEL_PATH" == "Qwen/Qwen3-VL-8B-Instruct" ]; then
        # actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_gemini-3-flash-preview_vr2.yaml" \
    run_experiment \
        actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_gpt-5-mini_vr2.yaml" \
        +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$(( 32 * 1024 )) \
        +actor_rollout_ref.rollout.agent.vsearcher_loop_cls=VSearcherLoopQwen3VL \
        actor_rollout_ref.model.custom_chat_template=null \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        custom_reward_function.reward_kwargs.format_reward.simple=True \
        "$@"

else

    # run_experiment \
    #     actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_gpt-5-high_vr2.yaml" \
    #     "$@"

    # run_experiment \
    #     actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_gemini-3-flash-preview_vr2.yaml" \
    #     "$@"

    run_experiment \
        actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_gpt-5-mini_vr2.yaml" \
        "$@"

fi