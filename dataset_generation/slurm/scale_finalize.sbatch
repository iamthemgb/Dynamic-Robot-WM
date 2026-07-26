#!/usr/bin/env bash
# Seal and strict-QC one completed scale block (CPU only).
#
# Usage:
#   sbatch slurm/scale_finalize.sbatch <DATASET_ROOT>
# Set SCALE_DEEP_VIDEO=1 to decode every MP4 during QC (slow).
#
#SBATCH --job-name=dynamic-robot-scale-finalize
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/scale/slurm-finalize-%j.out

set -euo pipefail

DATASET_ROOT="${1:?usage: scale_finalize.sbatch DATASET_ROOT}"
REPO="${SCALE_REPO:-/gpfs/radev/scratch/sous/mzl7/dataset_generation_fork}"
DEEP_FLAG=""
if [ "${SCALE_DEEP_VIDEO:-0}" = "1" ]; then
  DEEP_FLAG="--deep-video-checks"
fi
cd "$REPO"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load uv/0.9.17

uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
uv run --frozen python tools/scale_generation_driver.py finalize \
  --dataset "$DATASET_ROOT" $DEEP_FLAG
