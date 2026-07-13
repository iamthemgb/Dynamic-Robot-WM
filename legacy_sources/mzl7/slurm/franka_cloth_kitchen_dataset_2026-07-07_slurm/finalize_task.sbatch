#!/bin/bash
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/scratch/sous/mzl7/franka_cloth_kitchen_dataset_2026-07-07_logs/%x_%A.out

set -euo pipefail

TASK="${TASK:?TASK must be exported}"
DATASET_ROOT="${DATASET_ROOT:-/gpfs/radev/scratch/sous/mzl7/franka_cloth_kitchen_dataset_2026-07-07}"
COUNT="${COUNT:-1500}"

export PYTHONPATH="/home/mzl7/scratch:${PYTHONPATH:-}"

/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper/.venv/bin/python \
  -m mujoco_franka_cloth_kitchen_scene_previews.generate_previews finalize \
  --task "${TASK}" \
  --dataset-root "${DATASET_ROOT}" \
  --expected-count "${COUNT}"
