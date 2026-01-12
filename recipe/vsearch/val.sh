#!/bin/bash

source ./recipe/vsearch/_base.sh

TRAIN_FILES='[]'

MAX_IMG_TOKENS_VAL=16K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=10000000  # dump all validation samples (this may take some disk space)

VAL_ONLY=True

run_experiment \
    custom_reward_function.reward_kwargs.judge_model="${JUDGE_MODEL:-gpt-5-nano}" \
    trainer.max_val_sample_dump_per_data_source="$MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE"