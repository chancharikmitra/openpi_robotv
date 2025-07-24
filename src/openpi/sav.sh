#!/bin/bash
#SBATCH --job-name=SAV       # Job name
#SBATCH --output=slurm_output/SAV_EVAL_keyframe_Wipe.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=shard:16                  # Number of GPUs                
#SBATCH --cpus-per-task=2               # Number of CPU cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=128G 

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh

conda activate openpi

# python SAV_Infer.py

python SAV.py
#python attention_visualize.py

#python rename_h5_datasets.py

#python SAV_PI.py