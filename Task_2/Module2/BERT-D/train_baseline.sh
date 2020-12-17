#!/bin/bash

# This script demonstrates how to train BERT-D

# set path to dataset here
version="baseline"
dataroot="data"
num_gpus=1


python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_addr 127.0.0.2 --master_port 29517 baseline.py \
    --negative_sample_method "all" \
    --params_file baseline/configs/selection/params.json \
    --dataroot data \
    --exp_name ks-all-${version}-dstc9all-256dh-128kn-4cand-10epoch


