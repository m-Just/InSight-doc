#!/bin/bash

set -o errexit    # exit the script if any command fails
set -o pipefail   # exit the script if any command in a pipeline fails
set -o noclobber  # prevent overwriting files with redirection
set -o nounset    # treat unset variables as an error when substituting

if [ -z "${SLURM_JOB_ID:-}" ]; then
    NUM_CPUS=null
else
    NUM_CPUS=$(nproc)
fi

generate_wandb_run_id() {
    # Generate 8 character random string using alphanumeric characters
    (set +o pipefail; cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
}

run_experiment () {
    echo "Running experiment: $EXP_NAME"
    export EXP_NAME
    if [ "$#" -gt 0 ]; then
        echo "Additional arguments:"
        for arg in "$@"; do echo "  $arg"; done
    fi

    # Set up experiment directory and run mode
    local exp_path=$WORK_DIR/ckpts/$PROJECT_NAME/$EXP_NAME
    local train_dump_dir=$WORK_DIR/train_results/$PROJECT_NAME/$EXP_NAME
    local val_dump_dir=$WORK_DIR/val_results/$PROJECT_NAME/$EXP_NAME
    local val_only=${VAL_ONLY:-False}

    if [ "$val_only" == "True" ]; then
        echo "Run mode: val"
        if [ -n "${EVAL_NAME:-}" ]; then
            echo "Eval name: $EVAL_NAME"
            val_dump_dir=$val_dump_dir/$EVAL_NAME
            local log_file=$exp_path/val_${EVAL_NAME}.log
        else
            local log_file=$exp_path/val.log
        fi
        local logger="${LOGGER:-['console']}"
    else
        echo "Run mode: train"
        local log_file=$exp_path/train.log
        local logger="${LOGGER:-['console','wandb']}"
        mkdir -p "$train_dump_dir"
    fi

    mkdir -p "$exp_path"
    mkdir -p "$val_dump_dir"
    echo "Experiment dir: $exp_path"

    # Set up resume from step
    local resume_from_step=${RESUME_FROM_STEP:-null}
    if [ "$resume_from_step" != "null" ]; then
        echo "Resume from step: $resume_from_step"
    fi

    # Set up wandb run id (if wandb is enabled)
    local wandb_run_id=null
    if [[ "$logger" == *wandb* ]]; then
        # Automatically resume wandb run if it exists
        if [ -s "$exp_path/wandb_run_id" ]; then
            wandb_run_id=$(cat "$exp_path/wandb_run_id")
            if [ ! -z "${WANDB_RUN_ID:-}" ] && [ "$WANDB_RUN_ID" != "$wandb_run_id" ]; then
                echo "Error: wandb run id exists: $wandb_run_id" >&2
                exit 1
            fi
            echo "Resume wandb run: $wandb_run_id"
        else
            if [ -z "${WANDB_RUN_ID:-}" ]; then
                wandb_run_id="$(generate_wandb_run_id)"
                echo "New wandb run: $wandb_run_id"
            else
                wandb_run_id="$WANDB_RUN_ID"
                echo "Use wandb run: $wandb_run_id"
            fi
            echo "$wandb_run_id" > "$exp_path/wandb_run_id"
        fi
    fi

    # Record slurm job id (if slurm is used)
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "$SLURM_JOB_ID" >> "$exp_path/slurm_job_ids"
    fi

    # Compute max pixels, prompt length, and response length
    K_to_1024 () {
        if [[ "$1" =~ ^([0-9]+)[K]$ ]]; then
            echo $(( BASH_REMATCH[1] * 1024 ))
        else
            echo "$1"
        fi
    }

    MAX_MODEL_LEN=$(( 32 * 1024 ))
    MAX_IMG_TOKENS_TRAIN="${MAX_IMG_TOKENS_TRAIN:-4K}"
    MAX_IMG_TOKENS_VAL="${MAX_IMG_TOKENS_VAL:-16K}"

    _max_img_tokens_train=$(K_to_1024 "$MAX_IMG_TOKENS_TRAIN")
    MAX_PIXELS=$(( _max_img_tokens_train * 28 * 28 ))
    MAX_PROMPT_LENGTH=$(( _max_img_tokens_train + 1024 ))
    MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$(( 1024 * 9 ))}

    _max_img_tokens_val=$(K_to_1024 "$MAX_IMG_TOKENS_VAL")
    MAX_PIXELS_VAL=$(( _max_img_tokens_val * 28 * 28 ))
    MAX_PROMPT_LENGTH_VAL=$(( _max_img_tokens_val + 1024 ))
    _max_img_tokens=$(( _max_img_tokens_train > _max_img_tokens_val ? _max_img_tokens_train : _max_img_tokens_val ))

    (
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
        export WANDB_RESUME=allow
        export WANDB_RUN_ID="$wandb_run_id"
        # export HF_HUB_OFFLINE=1
        export VLLM_USE_V1=1
        export HYDRA_FULL_ERROR=1

        tmp_avail_gb=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
        if [ "$tmp_avail_gb" -lt 256 ]; then
            export TMPDIR="$WORK_DIR/tmp"
            export RAY_TMPDIR="$WORK_DIR/tmp"
            echo "WARNING: /tmp low on space (${tmp_avail_gb}GB); using TMPDIR and RAY_TMPDIR as $WORK_DIR/tmp"
            mkdir -p "$TMPDIR"
        fi
        
        python3 -m verl.trainer.main_ppo \
            --config-path "${VERL_CONFIG_DIR:-$VERL_PROJ_DIR/recipe/vsearch/config}" \
            --config-name "qwen_2_5_vl_7b_async.yaml" \
            ray_kwargs.ray_init.num_cpus="$NUM_CPUS" \
            data.train_files="$TRAIN_FILES" \
            data.train_batch_size=24 \
            data.val_files="$VAL_FILES"  \
            data.val_batch_size="${VAL_BATCH_SIZE:-128}" \
            actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
            data.max_pixels="$MAX_PIXELS" \
            data.max_prompt_length="$MAX_PROMPT_LENGTH" \
            data.max_response_length="$MAX_RESPONSE_LENGTH" \
            data.validation_max_pixels="$MAX_PIXELS_VAL" \
            data.validation_max_prompt_length="$MAX_PROMPT_LENGTH_VAL" \
            actor_rollout_ref.nccl_timeout="$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" \
            actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}" \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( 1024 * 14 )) \
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( 1024 * 28 )) \
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(( 1024 * 28 )) \
            actor_rollout_ref.rollout.temperature=1 \
            actor_rollout_ref.model.path="$MODEL_PATH" \
            actor_rollout_ref.actor.ppo_mini_batch_size=24 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.n=8 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.rollout.agent.agent_loop_config_path="recipe/vsearch/config/agent_${API_MODEL_FOR_AGENT:-gpt-5-mini}.yaml" \
            custom_reward_function.reward_kwargs.judge_model="${JUDGE_MODEL:-gpt-5-nano}" \
            custom_reward_function.reward_kwargs.fallback_judge_model="${FALLBACK_JUDGE_MODEL:-}" \
            trainer.logger="$logger" \
            trainer.project_name="$PROJECT_NAME" \
            trainer.experiment_name="$EXP_NAME" \
            trainer.default_local_dir="$exp_path" \
            trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
            trainer.val_before_train_n="${NUM_VAL_TRIALS:-1}" \
            trainer.val_only="$val_only" \
            trainer.save_freq=50 \
            trainer.test_freq=50 \
            trainer.refresh_freq=5 \
            trainer.total_training_steps=150 \
            trainer.resume_from_step="$resume_from_step" \
            trainer.train_dump_dir="$train_dump_dir" \
            trainer.val_dump_dir="$val_dump_dir" \
            trainer.max_val_sample_dump_per_data_source="${MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE:-5}" \
            "$@" 2>&1 | tee -a "$log_file"
    )
}
