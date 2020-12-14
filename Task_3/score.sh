#!/bin/bash
version="2snippets_f"
python scripts/scores.py --dataset val --dataroot data/ --outfile results/multi_${version}_val.json --scorefile results/multi_${version}_val_score.json
