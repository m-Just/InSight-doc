#!/bin/bash

source ./recipe/vsearch/_base.sh

TRAIN_FILES=null

MAX_IMG_TOKENS_VAL=16K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=10000000  # dump all validation samples (this can take a lot of disk space)
MIN_SUCCESS_RATE=1.0                          # ensure that all samples are judged successfully

VAL_ONLY=True
VAL_BEFORE_TRAIN_N="${NUM_VAL_TRIALS:-1}"

run_experiment \
    custom_reward_function.reward_kwargs.judge_model="${JUDGE_MODEL:-gpt-5-nano}" \
    custom_reward_function.reward_kwargs.min_success_rate="$MIN_SUCCESS_RATE" \
    trainer.max_val_sample_dump_per_data_source="$MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE"