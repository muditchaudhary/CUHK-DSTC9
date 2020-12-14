 #!/bin/bash

version="4snippets_xl"
labels=put your label file (task 2 output) here

python3 multiknow.py --generate runs/multiknow-${version} \
        --generation_params_file configs/generation/inference_params.json \
        --eval_dataset val --dataroot data \
        --labels_file ${labels} \
        --output_file results/multi_${version}_val.json
