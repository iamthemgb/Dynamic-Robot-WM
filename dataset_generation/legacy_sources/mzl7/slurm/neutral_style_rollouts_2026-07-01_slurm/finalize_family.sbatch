#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/mzl7/neutral_style_rollouts_2026-07-01_logs/%x_%A.out

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
DATASET_ROOT="${DATASET_ROOT:-/home/mzl7/scratch/neutral_style_rollouts_2026-07-01}"

export PYTHONPATH="/home/mzl7/scratch:${PYTHONPATH:-}"

/home/mzl7/.conda/envs/env_isaaclab/bin/python -m scripts_mujoco_projectile_catch_robocasa_train2500_yaml_scenes.dataset_generation finalize \
  --family "${FAMILY}" \
  --dataset-root "${DATASET_ROOT}"
