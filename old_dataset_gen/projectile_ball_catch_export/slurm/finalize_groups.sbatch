#!/bin/bash
#SBATCH --partition=rtx-devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A.out

# Run after the group-generation array finishes (per family):
#   cd <bundle root>
#   FAMILY=style020_seed0 DATASET_ROOT=... sbatch slurm/finalize_groups.sbatch
# Writes <family>/metadata.parquet, <family>/dataset_info.json, and refreshes
# the dataset-level splits.json. (No GPU needed.)

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${SLURM_SUBMIT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"

export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m projectile_ball_catch_robocasa_kitchen_scenes_scripts.group_dataset_generation finalize \
  --family "${FAMILY}" \
  --dataset-root "${DATASET_ROOT}"
