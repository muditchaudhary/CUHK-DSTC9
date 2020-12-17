#!/bin/bash

# This script demonstrates how to train BERT-DE model 

# set path to dataset here
version="baseline"
dataroot="data"
num_gpus=1

# use --negative_sample_method to modify the setting in params.json for this training run
# the updated parameters will be saved to {checkpoint}/params.json
# note that the default negative_sample_method for testing is "oracle"
# which filters the candidates based on the ground truth entity, so the number of candidates
# is way less than the total number of snippets
python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_addr 127.0.0.2 --master_port 29511 baseline.py \
    --negative_sample_method "all" \
    --params_file baseline/configs/selection/params.json \
    --dataroot data \
    --exp_name ks-all-${version}-dstc9all-256dh-128kn-5cand-10epoch
