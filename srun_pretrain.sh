#!/bin/bash
GPUS=8
GPUS_PER_NODE=8
PARTITION=MIA_LLM
JOB_NAME=MOTORpretrain

export TOKENIZERS_PARALLELISM=0

srun -n${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python Pretrain.py  --output_dir ./output/Pretrain
