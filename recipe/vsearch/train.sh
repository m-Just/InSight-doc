#!/bin/bash

source ./recipe/vsearch/_base.sh

USE_DYNAMIC_BSZ=True
# If use dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=4K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=5         # dump a small number of validation samples each step
MIN_SUCCESS_RATE=0.99                         # ensure that most samples are judged successfully

VAL_ONLY=True
VAL_BEFORE_TRAIN_N="${NUM_VAL_TRIALS:-1}"

run_experiment \
    +data.batch_sampler.weights.info_vqa_region_localization=0.5 \
    +data.batch_sampler.weights.merged_compound=0.5 \
    trainer.max_val_sample_dump_per_data_source="$MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE" \
    custom_reward_function.reward_kwargs.min_success_rate="$MIN_SUCCESS_RATE"