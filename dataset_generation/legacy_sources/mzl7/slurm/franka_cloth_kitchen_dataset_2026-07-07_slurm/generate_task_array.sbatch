#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/radev/scratch/sous/mzl7/franka_cloth_kitchen_dataset_2026-07-07_logs/%x_%A_%a.out

set -euo pipefail

TASK="${TASK:?TASK must be exported}"
DATASET_ROOT="${DATASET_ROOT:-/gpfs/radev/scratch/sous/mzl7/franka_cloth_kitchen_dataset_2026-07-07}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-24}"
DURATION="${DURATION:-5.0}"

export MUJOCO_GL=egl
export PYTHONPATH="/home/mzl7/scratch:${PYTHONPATH:-}"

/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper/.venv/bin/python \
  -m mujoco_franka_cloth_kitchen_scene_previews.generate_previews episode \
  --task "${TASK}" \
  --episode-index "${SLURM_ARRAY_TASK_ID}" \
  --dataset-root "${DATASET_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --duration "${DURATION}"
