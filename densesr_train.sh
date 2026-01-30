#!/bin/bash

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node 1 --master_port 29500 densesr_train_DDP.py --win_size 8 --train_ps 256 --dino_version dinov3 --dino_model vitl16