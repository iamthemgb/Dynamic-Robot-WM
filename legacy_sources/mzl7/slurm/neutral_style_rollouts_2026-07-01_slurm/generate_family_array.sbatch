#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/mzl7/neutral_style_rollouts_2026-07-01_logs/%x_%A_%a.out

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
COUNT="${COUNT:-750}"
DATASET_ROOT="${DATASET_ROOT:-/home/mzl7/scratch/neutral_style_rollouts_2026-07-01}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
DURATION="${DURATION:-2.5}"

export MUJOCO_GL=egl
export PYTHONPATH="/home/mzl7/scratch:${PYTHONPATH:-}"

/home/mzl7/.conda/envs/env_isaaclab/bin/python -m scripts_mujoco_projectile_catch_robocasa_train2500_yaml_scenes.dataset_generation episode \
  --family "${FAMILY}" \
  --episode-index "${SLURM_ARRAY_TASK_ID}" \
  --episode-count "${COUNT}" \
  --dataset-root "${DATASET_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --duration "${DURATION}"
