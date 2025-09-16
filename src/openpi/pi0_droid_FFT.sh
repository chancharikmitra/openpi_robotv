

In openpi_robotv/src/openpi/training/config.py , replace the line 877 with your own path to the h5 file. You can get the h5 dataset from /home/yusenluo/openpi_robotv/pick_green_cube_20.h5, it is on BAIR cluster and should be accessible for you.


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/compute_norm_stats.py --config-name=pi0_fast_droid_h5_finetune_sanity_check_pick_green_cube_FFT

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_droid_h5_finetune_sanity_check_pick_green_cube_FFT --exp-name=FFT_all_heads_sanity_check_pick_green_cube_droid_joint_velocity_20 --overwrite