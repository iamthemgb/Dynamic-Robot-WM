#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A.out

# Finalize + validate one family of the grouped rolling dataset.
# Submit FROM old_dataset_gen/:
#   FAMILY=rolling_layout38_style42 DATASET_ROOT=... sbatch slurm/finalize_rolling_groups.sbatch

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
OLD_ROOT="${OLD_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${OLD_ROOT}/projectile_ball_catch_export}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"

export PYTHONPATH="${OLD_ROOT}:${BUNDLE_ROOT}/robosuite:${BUNDLE_ROOT}/robocasa_full:${PYTHONPATH:-}"
export ROBOCASA_ASSETS_ROOT="${BUNDLE_ROOT}/robocasa_full/robocasa/models/assets"

"${PYTHON_BIN}" -m ball_rolling_dynamics_scripts.group_dataset_generation finalize \
  --family "${FAMILY}" \
  --dataset-root "${DATASET_ROOT}"

"${PYTHON_BIN}" -m ball_rolling_dynamics_scripts.validate_group_dataset \
  --dataset-root "${DATASET_ROOT}" \
  --family "${FAMILY}"
