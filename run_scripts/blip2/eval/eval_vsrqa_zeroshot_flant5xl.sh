python -m torch.distributed.run --nproc_per_node=4 \
evaluate.py --cfg-path lavis/projects/blip2/eval/vsr_zeroshot_flant5xl_eval.yaml