#!/bin/bash
#SBATCH --job-name=SAV       # Job name
#SBATCH --output=/scr2/yusenluo/openpi_robotv/src/openpi/slurm_output/SAV_train_pick1_keyframe.txt   # Output file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --gres=gpu:2              # Number of GPUs                
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --time=24:00:00
#SBATCH --mem=256G 

source /scr/yusenluo/anaconda3/etc/profile.d/conda.sh

conda activate openpi


python /scr2/yusenluo/openpi_robotv/src/openpi/generate_activation_dataset_on_robot.py

#python SAV.py
#python attention_visualize.py

#python rename_h5_datasets.py

#python SAV_PI.py