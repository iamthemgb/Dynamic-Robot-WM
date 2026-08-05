#!/bin/bash
#SBATCH --partition=rtx-devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A.out

# Run after the generate array finishes:
#   cd <bundle root>
#   FAMILY=style020_seed0 sbatch slurm/finalize_family.sbatch
# (No GPU needed; runs fine on a CPU partition too.)

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${SLURM_SUBMIT_DIR}}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"

export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m projectile_ball_catch_robocasa_kitchen_scenes_scripts.dataset_generation finalize \
  --family "${FAMILY}" \
  --dataset-root "${DATASET_ROOT}"
