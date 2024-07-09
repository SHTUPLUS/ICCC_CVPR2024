python -m torch.distributed.run --master_port 13122 --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/okvqa_zeroshot_opt2.7b_eval.yaml \
