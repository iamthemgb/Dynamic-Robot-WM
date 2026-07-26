#!/usr/bin/env bash
#SBATCH --job-name=dynamic-robot-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/smoke_tests/slurm-%j.out

set -euo pipefail

REPO=/gpfs/radev/project/sous/zl664/dataset_generation
cd "$REPO"
export MUJOCO_GL=egl

uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
uv run python tools/run_smoke_suite.py \
  --config configs/families/smoke_120.yaml \
  --output outputs/smoke_tests/canonical_smoke_120
