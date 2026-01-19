#!/bin/bash

source ./recipe/vsearch/_base.sh

TRAIN_FILES='[]'

MAX_IMG_TOKENS_VAL=16K

MAX_VAL_SAMPLE_DUMP_PER_DATA_SOURCE=10000000  # dump all validation samples (this may take some disk space)

VAL_ONLY=True
VAL_BEFORE_TRAIN=True

EVAL_NAME=my_eval
run_experiment