#!/bin/bash

source ./recipe/vsearch/_base.sh

USE_DYNAMIC_BSZ=True
# If using dynamic batch size, tune `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu`
# Otherwise, tune `ppo_micro_batch_size_per_gpu` and `log_prob_micro_batch_size_per_gpu`

MAX_IMG_TOKENS_TRAIN=4K
MAX_IMG_TOKENS_VAL=16K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=5  # dump a small number of validation samples each step

VAL_ONLY=False

# set a lower timeout for API requests to prevent laggy API requests from slowing down training
# increase `OPENAI_CLIENT_TIMEOUT` if you see many API timeouts
export OPENAI_CLIENT_TIMEOUT=60

run_experiment \
    +data.batch_sampler.weights.info_vqa_region_localization=0.5 \
    +data.batch_sampler.weights.viscot_vstar_collage=0.5