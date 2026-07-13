#!/usr/bin/env bash
#SBATCH --job-name=dynamic-native-rigid-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/native_smoke/slurm-%j.out

set -euo pipefail

REPO=/gpfs/radev/project/sous/zl664/dataset_generation
OUTPUT=${DYNAMIC_ROBOT_NATIVE_SMOKE_OUTPUT:?Set DYNAMIC_ROBOT_NATIVE_SMOKE_OUTPUT to a new dataset directory}

cd "$REPO"
export MUJOCO_GL=egl

uv sync --frozen --extra mujoco --extra test
uv run dynamic-robot-dataset generate \
  --backend native_mujoco \
  --family falling_catch \
  --subfamily centered_vertical_drop \
  --branches success,no_op \
  --num-bundles 1 \
  --output "$OUTPUT"
uv run dynamic-robot-dataset finalize --dataset "$OUTPUT" --name native_rigid_smoke
uv run dynamic-robot-dataset validate --dataset "$OUTPUT"
