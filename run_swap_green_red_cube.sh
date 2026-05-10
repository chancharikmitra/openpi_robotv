#!/bin/bash
# How to fine-tune pi05 on the swap_green_red_cube_20 LeRobot dataset.
#
# Prereqs (one-time):
#   1. Install openpi env (conda env named "openpi" with JAX/Flax + lerobot).
#   2. Log in to HuggingFace so LeRobotDataset can pull yusenluo9z/swap_green_red_cube_20:
#        huggingface-cli login
#      (or set HF_TOKEN in your environment).
#   3. Clone this repo and cd into its root.
#
# Pick a free GPU and run:
#   bash run_swap_green_red_cube.sh
# Or override the GPU index:
#   GPU=0 bash run_swap_green_red_cube.sh
#
# Two variants are trained sequentially:
#   - pi05_KNN_..._swap_green_red_cube           (KNN top-20 attention heads, LoRA r=32)
#   - pi05_All_heads_LoRA_swap_green_red_cube    (all heads, LoRA r=32)
#
# Each variant: norm_stats first, then 3000 train steps (batch=32, cosine LR).

set -uo pipefail

# ---- Conda activation ----
# Adjust the path below if your conda lives elsewhere.
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$CONDA_SH" ]]; then
    # Try the lab path used on this machine.
    CONDA_SH="/scr/yusenluo/anaconda3/etc/profile.d/conda.sh"
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate openpi

# ---- Env ----
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PWD/.cache/openpi}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE=0

CONFIGS=(
    "pi05_KNN_heads_robo_steering_freeze_KV_SIGLIP_ActionExpert_MLP_swap_green_red_cube"
    "pi05_All_heads_LoRA_swap_green_red_cube"
)

# Continue to the next config even if one fails. Final exit is non-zero
# if any config failed.
overall_rc=0
for cfg in "${CONFIGS[@]}"; do
    echo "===== [$(date)] norm_stats: $cfg ====="
    python scripts/compute_norm_stats.py --config-name "$cfg"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "!!!!! norm_stats failed for $cfg (exit=$rc), skipping train and moving on"
        overall_rc=1
        continue
    fi

    echo "===== [$(date)] train: $cfg ====="
    python scripts/train.py "$cfg" --exp-name="${cfg}" --overwrite
    rc=$?
    echo "===== [$(date)] $cfg train exit=$rc ====="
    if [[ $rc -ne 0 ]]; then
        overall_rc=1
    fi
done

echo "===== [$(date)] all done. overall_rc=$overall_rc ====="
exit $overall_rc
