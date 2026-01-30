#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29508 ./test_shadow.py --win_size 8 --train_ps 256 --save_images