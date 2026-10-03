#!/bin/bash
#SBATCH --partition=gpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/%x_%j.out

# One train.py run on the RCI cluster; the job name is the experiment name e, the arguments are hydra overrides.
# NOTES (a markdown file) is copied into the run's log dir as README.md once training has started.
# Submit from the repo root:
#   NOTES=runs/notes/E1.md sbatch --job-name=E1_enc_base runs/rci_train.sh task=adapt_goto_encoder cm=SRT B=16384

source /etc/profile
PY=/mnt/personal/elghaada/envs/rl/bin/python

nvidia-smi
$PY train.py e="$SLURM_JOB_NAME" "$@" &
pid=$!
if [ -n "$NOTES" ]; then
  sleep 120  # train.py recreates the log dir at start
  log_dir=$(ls -dt logs/*/*_"$SLURM_JOB_NAME" | head -1)
  cp "$NOTES" "$log_dir/README.md" && echo "notes -> $log_dir/README.md"
fi
wait $pid
