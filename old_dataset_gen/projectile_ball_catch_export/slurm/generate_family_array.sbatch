#!/bin/bash
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out

# Submit FROM the bundle root so the relative logs/ path and default
# BUNDLE_ROOT resolve correctly:
#   cd <bundle root>
#   FAMILY=style020_seed0 sbatch --array=0-749 slurm/generate_family_array.sbatch

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
COUNT="${COUNT:-750}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${SLURM_SUBMIT_DIR}}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
DURATION="${DURATION:-2.5}"

export MUJOCO_GL=egl
export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m projectile_ball_catch_robocasa_kitchen_scenes_scripts.dataset_generation episode \
  --family "${FAMILY}" \
  --episode-index "${SLURM_ARRAY_TASK_ID}" \
  --episode-count "${COUNT}" \
  --dataset-root "${DATASET_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --duration "${DURATION}"
