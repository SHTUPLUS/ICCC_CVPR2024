python -m torch.distributed.run --master_port 23159  --nproc_per_node=$2 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2.yaml --job-name $1