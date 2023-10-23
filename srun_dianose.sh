#!/bin/bash
GPUS=1
GPUS_PER_NODE=1
PARTITION=MIA_LLM
JOB_NAME=MOTORdianose

export TOKENIZERS_PARALLELISM=0
pretrained_ckpt=./output/Pretrain/checkpoint_29.pth

srun -n${GPUS} \
    -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    python Diagnose_BLIP.py --output_dir ./output/Diagnose --dataset_name mimic_cxr --pretrained $pretrained_ckpt \
           --distributed False --save_dir results/mimic/Diagnose



#GPUS_PER_NODE=1
#NTASK=1
#PARTITION=MIA_LLM
#JOB_NAME=MedBLIP
#srun -n${GPUS} \
#    -p ${PARTITION} \
#    --job-name=${JOB_NAME} \
#    --gres=gpu:${GPUS_PER_NODE} \
#    --ntasks-per-node=${GPUS_PER_NODE} \
#    --kill-on-bad-exit=1 \
#    python Diagnose_BLIP.py --output_dir output/Diagnose --dataset_name chexpert --pretrained $pretrained_ckpt \
#           --distributed False --save_dir results/chexpert/Diagnose
