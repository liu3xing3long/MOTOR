#!/bin/bash

python -m torch.distributed.run --nproc_per_node=1 Pretrain.py  --output_dir ./output/Pretrain2
