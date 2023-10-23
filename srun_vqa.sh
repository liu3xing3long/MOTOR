#!/bin/bash
GPUS=1
GPUS_PER_NODE=1
PARTITION=MIA_LLM
JOB_NAME=MOTORgenerate

export TOKENIZERS_PARALLELISM=0
pretrained_ckpt=./output/Pretrain/checkpoint_29.pth

#srun -n${GPUS} \
#    -p ${PARTITION} \
#    --job-name=${JOB_NAME} \
#    --gres=gpu:${GPUS_PER_NODE} \
#    --ntasks-per-node=${GPUS_PER_NODE} \
#    --kill-on-bad-exit=1 \
#    python VQA.py --test_C --add_typeatt2 --pretrained $pretrained_ckpt \
#           --setting VQA-RAD --output_dir results/VQA/

