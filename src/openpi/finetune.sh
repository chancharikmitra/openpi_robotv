#!/bin/bash
#SBATCH --job-name=Pi0FAST-FT       # Job name
#SBATCH --output=/scr2/yusenluo/openpi_robotv/src/openpi/slurm_output/finetune/head_lora_tune_KNN_cosine_K=40_without_img.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=gpu:2               # Number of GPUs                
#SBATCH --cpus-per-task=24               # Number of CPU cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=256G 

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh

conda activate openpi

#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/compute_norm_stats.py --config-name=pi0_fast_droid_h5_head_lora_tune_on_robot

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_droid_h5_finetune --exp-name=all_heads_lora_tune_without_img_on_robot --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_droid_h5_full_finetune --exp-name=full_finetune --overwrite


#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_droid_h5_head_lora_tune_debug --exp-name=head_lora_tune_SAV_best_pick_20_margin_without_img --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_droid_h5_head_lora_tune_on_robot --exp-name=head_lora_tune_SAV_worst_pick_20_margin_without_img --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_droid_h5_head_lora_tune_on_robot --exp-name=head_lora_tune_KNN_cosine_K=40_without_img --overwrite