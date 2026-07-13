#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/mzl7/rolling_island_catch_dataset_2026-07-06_logs/%x_%A.out

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
DATASET_ROOT="${DATASET_ROOT:-/gpfs/radev/scratch/sous/mzl7/rolling_island_catch_dataset_2026-07-06}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"

export PYTHONPATH="/home/mzl7/scratch:${PYTHONPATH:-}"

/home/mzl7/.conda/envs/env_isaaclab/bin/python -m ball_roll_interception_scripts.dataset_generation finalize \
  --family "${FAMILY}" \
  --dataset-root "${DATASET_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}"
