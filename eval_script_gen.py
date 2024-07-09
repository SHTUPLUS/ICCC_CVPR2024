import os
import sys
import random

path = sys.argv[1]
# model_type = sys.argv[2]
# path = "lavis/output/BLIP2-instrut/Pretrain_stage2/2023092112454-instruct_vicuna_new_aug_pretrain-fromv1.1_prompt_for_qformer-train"

all_pth = []
for each in  os.listdir(path):
    if '.pth' in each:
        all_pth.append(each)



ib_v7b_task = dict(gqa = "lavis/projects/blip2/eval/gqa_zeroshot_instruct_blip_vicuna7b_eval.yaml",
              vqav2 = "lavis/projects/blip2/eval/vqav2_zeroshot_instruct_blip_vicuna7b_eval.yaml",
              okvqa = "lavis/projects/blip2/eval/okvqa_zeroshot_instruct_blip_vicuna7b_eval.yaml",
              cococap ="lavis/projects/blip2/eval/caption_coco_instructblip_vicuna7b_eval.yaml",
              nocap='lavis/projects/blip2/eval/caption_nocap_instructblip_vicuna7b_eval.yaml',
              vsr='lavis/projects/blip2/eval/vsr_zeroshot_instructblip_vicuna7b_eval.yaml')


b2_67b_task = dict(gqa = "lavis/projects/blip2/eval/gqa_zeroshot_opt6.7b_eval.yaml",
                    coco_cap="lavis/projects/blip2/eval/caption_coco_opt6.7b_eval.yaml",
                    okvqa = "lavis/projects/blip2/eval/okvqa_zeroshot_opt6.7b_eval.yaml",
                    nocap='lavis/projects/blip2/eval/caption_nocap_opt6.7b_eval.yaml',
                    vqav2 = "lavis/projects/blip2/eval/vqav2_zeroshot_opt6.7b_eval.yaml",)

b2_27b_task = dict(gqa = "lavis/projects/blip2/eval/gqa_zeroshot_opt2.7b_eval.yaml",
                    okvqa = "lavis/projects/blip2/eval/okvqa_zeroshot_opt2.7b_eval.yaml",
                    coco_cap="lavis/projects/blip2/eval/caption_coco_opt2.7b_eval.yaml",
                    nocap='lavis/projects/blip2/eval/caption_nocap_opt2.7b_eval.yaml',
                    vqav2 = "lavis/projects/blip2/eval/vqav2_zeroshot_opt2.7b_eval.yaml",)

if 'opt2.7b' in path:
    d_task = b2_27b_task
elif 'opt6.7b' in path:
    d_task = b2_67b_task
elif 'vicuna7b' in path:
    d_task = ib_v7b_task

eval_script = """
python -m torch.distributed.run --master_port 23699 --nproc_per_node=4 evaluate.py --cfg-path "{cfg_script}" --job-name "{j_name}" --ckpt-path "{ckpt_path}"
"""

run_script = []
for task_name in d_task.keys():
    all_script = []
    idx = 1
    while True:
        if idx > len(all_pth):
            break
        
        each_ckpt_pth = all_pth[idx - 1]
        # idx = int(idx * 2)
        idx = int(idx + 1) 
        # if random.random() > 0.7:
        #     idx += 1

        each_ckpt_pth = os.path.join(path, each_ckpt_pth)
        job_name = "-".join(each_ckpt_pth.split('/')[-3:])
        job_name = job_name.split('.pth')[0]
        script = eval_script.format(**{
            'cfg_script': d_task[task_name],
            'j_name':job_name,
            'ckpt_path': each_ckpt_pth
        })
        all_script.append(script)

    exp_name = path.split('/')[-1]
    scr_dir = f'run_scripts/batch_eval/{exp_name}-{task_name}.sh'
    with open(scr_dir, 'w') as f:
        f.write('\n\n'.join(all_script))
    
    run_script.append(f"bash {scr_dir}")

run_script = ' ;'.join(run_script)
print(run_script)