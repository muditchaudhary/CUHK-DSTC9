
version="4snippets_f"
dataroot="data"
num_gpus=2

#
python3 -m torch.distributed.launch --nproc_per_node ${num_gpus} --master_addr 127.0.0.2 --master_port 28079 multiknow.py \
    --params_file configs/generation/gpt2-large-multiknow-params_1.json\
    --dataroot data \
    --exp_name multiknow-large-${version}
