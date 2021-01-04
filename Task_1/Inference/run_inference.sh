#!/bin/bash

python official_eval.py --params="../Domain_cls/params/params.json" --eval_only --checkpoint="./models/checkpoint-2400_new" --dataroot="./dum_data" --outfile="preds_test.json"