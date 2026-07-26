#!/usr/bin/env bash
# Execute the shards of one planned scale block on A40 GPUs.
#
# Usage (one array task per shard; throttle with %N):
#   sbatch --array=0-15%8 slurm/scale_generate_a40.sbatch <DATASET_ROOT>
# Example:
#   sbatch --array=0-15%8 slurm/scale_generate_a40.sbatch \
#     /gpfs/radev/scratch/sous/mzl7/dynamic_rollouts/F1a/block-0000
#
# Interruptions are safe: per-episode attempt receipts make every shard
# resumable by resubmitting the identical array task.
#
#SBATCH --job-name=dynamic-robot-scale-generate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/scale/slurm-generate-%A_%a.out

set -euo pipefail

DATASET_ROOT="${1:?usage: scale_generate_a40.sbatch DATASET_ROOT}"
REPO="${SCALE_REPO:-/gpfs/radev/scratch/sous/mzl7/dataset_generation_fork}"
cd "$REPO"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load uv/0.9.17
export MUJOCO_GL=egl

uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
uv run --frozen python tools/scale_generation_driver.py run-shards \
  --dataset "$DATASET_ROOT" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --max-episodes-per-process 1
