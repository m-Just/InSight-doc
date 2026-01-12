#!/bin/bash

source ./recipe/vsearch/_base.sh

USE_DYNAMIC_BSZ=True
# If use dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=4K
MAX_IMG_TOKENS_VAL=16K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=5         # dump a small number of validation samples each step

VAL_ONLY=False

run_experiment \
    +data.batch_sampler.weights.info_vqa_region_localization=0.5 \
    +data.batch_sampler.weights.merged_compound=0.5