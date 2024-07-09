# Official Implementation for [Learning by Correction: Efficient Tuning Task for Zero-Shot Generative Vision-Language Reasoning (CVPR 2024)](https://arxiv.org/pdf/2404.00909.pdf).

## Table of Contents
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Model Zoo](#modelzoo)
  - [Training](#train)
  - [Evaluation](#eval)
  - [Reference](#ref)


## Abstract
Generative vision-language models (VLMs) have shown like image captioning and visual question answering. However, improving their zero-shot reasoning typically requires second-stage instruction tuning, which relies heavily on human-labeled or large language model-generated challenge, we introduce Image-Conditioned Caption Correction (ICCC), a novel pre-training task designed to enhance VLMs’ zero-shot performance without the need for labeled task-aware data. 
The ICCC task compels VLMs to rectify mismatches between visual and language concepts, thereby enhancing instruction following and text generation conditioned on visual inputs. Leveraging language structure and a lightweight dependency parser, we construct data labeling and computation costs. Experimental results on BLIP-2 and InstructBLIP demonstrate significant improvements in zero-shot image-text generation-based VL tasks through ICCC instruction tuning.

## Installation

1. Creating conda environment and install pytorch

```bash
conda create -n lavis python=3.8
conda activate lavis


# CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

2. Install other dependencies:
```bash
pip install -r requirements_iccc.txt
# the hugging face version: v4.29.2
```
Our work is built upon LAVIS, sharing the majority of its requirements.


3. Build from source

```bash
pip install -e .
```

4. Download pre-trained parameters of LLMs and VLMs:

We employ the same parameters as those used in the Large Language Models (LLMs) and Vision-Language Models (VLMs) implemented in  [LAVIS](https://github.com/salesforce/LAVIS/blob/main/lavis/configs/models/blip2). However, we enhance loading efficiency by manually downloading and replacing the model parameters' URL with a local directory.

For LLM Vicuna, please start by preparing the Vicuna 7B v1.1 weights available [here](https://huggingface.co/lmsys/vicuna-7b-v1.1). Then, modify the ``llm_model`` in the [Model Config](lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml) to point to the folder containing the Vicuna weights.


## Datasets

We leverage the Visual Genome and COCO image-text datasets, consistent with BLIP in LAVIS. To prepare the data, please refer to the [instructions provided by LAVIS](https://opensource.salesforce.com/LAVIS/latest/benchmark#blip).

#### Dependency Parsing for ICCC Task
Prior to ICCC training, data generation involves parsing image-text data using a dependency parser to extract dependency structures and concept bases for training data construction.
```bash
python iccc_data_generation.py path/to/data/dir data_filename_without_suffix
# Example for the COCO dataset:
# python iccc_data_generation.py data/coco/annotations coco_karpathy_val
```
After parsing, ensure the data file path is updated with ICCC in configuration (`lavis/configs/datasets/coco/defaults_cap_iccc.yaml` for COCO and `lavis/configs/datasets/vg/defaults_cap_iccc.yaml` for Visual Genome).

## Model Zoo
###  BLIP-2 
|                 |      |        |       |      |        |       |        |            |
|-----------------|------|--------|-------|------|---------------|-------|--------|------------|
|                 |      |        |       |      |   NoCaps|       |        |            |
|         | GQA  | OK-VQA | VQAv2 | VSR  | BLUE@4 | SPICE | CIDERr | Checkpoint |
| OPT2.7B         | 33.5 | 26.6   | 51.9  | 48.3 | 43.6          | 13.8  | 105.7  |            |
| OPT2.7B w/ ICCC | 38.2 | 29.5   | 54.3  | 47.6 | 46.0          | 14.3  | 111.9  |            |
| OPT6.7B         | 35.5 | 30.7   | 52.6  | 48.5 | 41.5          | 13.0  | 101.4  |            |
| OPT6.7B w/ ICCC | 38.3 | 31.7   | 58.8  | 51.5 | 44.1          | 13.5  | 106.9  |            |

## Training and Evaluation

### ICCC Fine-tuning

#### BLIP2 OPT2.7
```bash
python -m torch.distributed.run --master_port 23619  --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_opt2.7_iccc_iter.yaml --job-name blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt --swap-ratio 0.15 --aug-ratio 0.30
```

#### BLIP2 OPT6.7
```bash
python -m torch.distributed.run --master_port 23619  --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_opt6.7_iccc_iter.yaml --job-name blip2-new_aug_opt6.7-0.3-aug-0.15_swap --swap-ratio 0.15 --aug-ratio 0.30
```

### Evaluation

#### Evaluate All Downstream Tasks Simultaneously
Use `eval_script_gen.py` to generate evaluation scripts for all downstream tasks.
This Python script generates the command for starting evaluation in the `batch_eval` directory.
```bash
python eval_script_gen.py /path/to/experiment/dir 
```


Execute the generated output bash commands will evaluate all intermediate checkpoints of fine-tuning for each downstream task:
```bash
bash run_scripts/batch_eval/2024041915535-blip2-new_aug_opt2.7-0.3-aug-0.15_swap_lr5e6-train-gqa.sh ;
bash run_scripts/batch_eval/2024041915535-blip2-new_aug_opt2.7-0.3-aug-0.15_swap_lr5e6-train-okvqa.sh ;
bash run_scripts/batch_eval/2024041915535-blip2-new_aug_opt2.7-0.3-aug-0.15_swap_lr5e6-train-coco_cap.sh ;
bash run_scripts/batch_eval/2024041915535-blip2-new_aug_opt2.7-0.3-aug-0.15_swap_lr5e6-train-nocap.sh ;
bash run_scripts/batch_eval/2024041915535-blip2-new_aug_opt2.7-0.3-aug-0.15_swap_lr5e6-train-vqav2.sh;
```

#### Perform Evaluation on Each Downstream Task Individually
You can also use the default `evaluate.py` from the LAVIS evaluation toolkit. Provide `--cfg-path` and `--ckpt-path` to specify the target downstream tasks and model parameters for training.
Below are examples illustrating the usage:

##### BLIP2 OPT2.7
```bash
# GQA
python -m torch.distributed.run --master_port 23199 --nproc_per_node=4 evaluate.py --cfg-path "lavis/projects/blip2/eval/gqa_zeroshot_opt2.7b_eval.yaml" --job-name "Pretrain_stage2-2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train-checkpoint_7500" --ckpt-path "lavis/output/BLIP2/Pretrain_stage2/2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train/checkpoint_7500.pth"

# VQAv2
python -m torch.distributed.run --master_port 23199 --nproc_per_node=4 evaluate.py --cfg-path "lavis/projects/blip2/eval/vqav2_zeroshot_opt2.7b_eval.yaml" --job-name "Pretrain_stage2-2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train-checkpoint_7500" --ckpt-path "lavis/output/BLIP2/Pretrain_stage2/2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train/checkpoint_7500.pth"

# OKVQA
python -m torch.distributed.run --master_port 23199 --nproc_per_node=4 evaluate.py --cfg-path "lavis/projects/blip2/eval/okvqa_zeroshot_opt2.7b_eval.yaml" --job-name "Pretrain_stage2-2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train-checkpoint_7500" --ckpt-path "lavis/output/BLIP2/Pretrain_stage2/2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train/checkpoint_7500.pth"

# NoCAP
python -m torch.distributed.run --master_port 23199 --nproc_per_node=4 evaluate.py --cfg-path "lavis/projects/blip2/eval/caption_nocap_opt2.7b_eval.yaml" --job-name "Pretrain_stage2-2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train-checkpoint_7500" --ckpt-path "lavis/output/BLIP2/Pretrain_stage2/2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train/checkpoint_7500.pth"

# COCOcap
python -m torch.distributed.run --master_port 23199 --nproc_per_node=4 evaluate.py --cfg-path "lavis/projects/blip2/eval/caption_coco_opt2.7b_eval.yaml" --job-name "Pretrain_stage2-2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train-checkpoint_7500" --ckpt-path "lavis/output/BLIP2/Pretrain_stage2/2023103113105-blip2-new_aug_opt2.7-0.3-aug-0.15_swap-simple_prompt-role-all-train/checkpoint_7500.pth"

```

## Paper and Citing 
If you find this project helps your research, please kindly consider citing our papers in your publications. 

```bibtex
@misc{li2024learning,
      title={Learning by Correction: Efficient Tuning Task for Zero-Shot Generative Vision-Language Reasoning}, 
      author={Rongjie Li and Yu Wu and Xuming He},
      year={2024},
      eprint={2404.00909},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledge

This repository is built on [LAVIS](https://github.com/salesforce/LAVIS).

## License
[BSD 3-Clause License](LICENSE.txt)