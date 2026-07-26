#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/mzl7/rolling_island_catch_dataset_2026-07-06_logs/%x_%A_%a.out

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
COUNT="${COUNT:-1500}"
DATASET_ROOT="${DATASET_ROOT:-/gpfs/radev/scratch/sous/mzl7/rolling_island_catch_dataset_2026-07-06}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
FPS="${FPS:-30}"
DURATION="${DURATION:-2.5}"

export MUJOCO_GL=egl
export PYTHONPATH="/home/mzl7/scratch:${PYTHONPATH:-}"

/home/mzl7/.conda/envs/env_isaaclab/bin/python -m ball_roll_interception_scripts.dataset_generation episode \
  --family "${FAMILY}" \
  --episode-index "${SLURM_ARRAY_TASK_ID}" \
  --episode-count "${COUNT}" \
  --dataset-root "${DATASET_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --duration "${DURATION}"
