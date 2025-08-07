#!/bin/bash
#SBATCH --job-name=SAV       # Job name
#SBATCH --output=/scr2/yusenluo/openpi_robotv/src/openpi/slurm_output/SAV_train_full_finetune.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=gpu:4                 # Number of GPUs                
#SBATCH --cpus-per-task=32               # Number of CPU cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=256G 

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh

conda activate openpi

#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/compute_norm_stats.py --config-name=pi0_fast_droid_h5_head_tune_debug

#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python /scr2/yusenluo/openpi_robotv/scripts/train.py pi0_fast_droid_h5_finetune --exp-name=20_pick_lora_finetune --overwrite
#XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_droid_h5_full_finetune --exp-name=full_finetune --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=false python scripts/train.py pi0_fast_droid_h5_head_tune_debug --exp-name=head_tune_debug_mask --overwrite


#python SAV_Infer.py

#python SAV.py
#python attention_visualize.py

#python rename_h5_datasets.py

#python SAV_PI.py