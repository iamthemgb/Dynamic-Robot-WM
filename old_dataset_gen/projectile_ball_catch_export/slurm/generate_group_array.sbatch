#!/bin/bash
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out

# v3 state-conditional grouped dataset: one GROUP per array task (all siblings
# generated sequentially; requeued tasks skip already-finished siblings).
# Submit FROM the bundle root so the relative logs/ path and default
# BUNDLE_ROOT resolve correctly:
#   cd <bundle root>
#   FAMILY=style020_seed0 DATASET_ROOT=... PHASE_CONFIG=configs/p1_full.json \
#     sbatch --array=0-499 slurm/generate_group_array.sbatch

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
PHASE_CONFIG="${PHASE_CONFIG:?PHASE_CONFIG must be exported (e.g. configs/p0_pilot.json)}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${SLURM_SUBMIT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
DURATION="${DURATION:-2.5}"

export MUJOCO_GL=egl
export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m projectile_ball_catch_robocasa_kitchen_scenes_scripts.group_dataset_generation group \
  --family "${FAMILY}" \
  --group-index "${SLURM_ARRAY_TASK_ID}" \
  --dataset-root "${DATASET_ROOT}" \
  --phase-config "${PHASE_CONFIG}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --duration "${DURATION}"
