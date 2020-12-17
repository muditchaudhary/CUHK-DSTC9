#!/bin/bash

version="baseline"

# Prepare directories for intermediate results of each subtask
mkdir -p pred/val

python3 baseline.py --eval_only --checkpoint runs/ks-all-${version}-combined8domainsall-256dh-128kn-4cand-10epoch/checkpoint-50450 \
   --eval_all_snippets \
   --dataroot data_eval/ \
   --eval_dataset test \
   --labels_file data_eval/test/team1-3step-Sunday.json \
   --output_file pred/test-data/combined8domainsall-256dh-128kn-4cand-10epoch/team1-3step-team2-bert-Sunday.json


