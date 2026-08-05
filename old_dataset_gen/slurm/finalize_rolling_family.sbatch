#!/bin/bash
#SBATCH --partition=rtx-devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A.out

# Run after the generate array finishes:
#   cd <old_dataset_gen>
#   FAMILY=rolling_layout38_style42 DATASET_ROOT=... sbatch slurm/finalize_rolling_family.sbatch
# (No GPU needed; runs fine on a CPU partition too.)

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
OLD_ROOT="${OLD_ROOT:-${SLURM_SUBMIT_DIR}}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"

export PYTHONPATH="${OLD_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m ball_rolling_dynamics_scripts.dataset_generation finalize \
  --family "${FAMILY}" \
  --dataset-root "${DATASET_ROOT}"
