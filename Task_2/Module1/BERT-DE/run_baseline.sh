#!/bin/bash

version="baseline"

# This is a demonstration on how to generate results with BERT-DE
# The input files are knowledge.json and logs.json. labels.json is not required

# Prepare directories for intermediate results of each subtask
mkdir -p pred/test-data/dstc9all-256dh-5cand


# Next we do knowledge selection based on the predictions generated previously
# Use --labels_file to take the results from the previous task
# Use --output_file to generate labels.json with predictions
python3 baseline.py --eval_only --checkpoint runs/ks-all-${version}-dstc9all-256dh-128kn-5cand-10epoch/checkpoint-13660 \
   --eval_all_snippets \
   --dataroot data_eval \
   --eval_dataset test \
   --labels_file data_eval/test/pseudo_input.ktd.json \
   --output_file pred/test-data/dstc9all-256dh-5cand/team2_13660.ks.json
