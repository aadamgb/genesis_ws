#!/bin/bash
#SBATCH --job-name=genesis_train
#SBATCH --partition=amdgpufast        # 4h limit, starts fastest; use amdgpu (1 day) for longer runs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Usage (from /mnt/personal/$USER/genesis_ws):
#   sbatch cluster/train.sbatch                          # default config
#   sbatch cluster/train.sbatch task=goto e=run1 seed=2  # any hydra overrides

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /mnt/personal/$USER/envs/rl

cd /mnt/personal/$USER/genesis_ws
nvidia-smi
python train.py v=false "$@"
