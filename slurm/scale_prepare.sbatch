#!/usr/bin/env bash
# Mint and immutably plan one scale block (CPU only; no rendering).
#
# Usage:
#   sbatch --array=0-<blocks-1> slurm/scale_prepare.sbatch <LEAF> <OUTPUT_ROOT>
# Example:
#   sbatch --array=0-27 slurm/scale_prepare.sbatch F1a \
#     /gpfs/radev/scratch/sous/mzl7/dynamic_rollouts
#
#SBATCH --job-name=dynamic-robot-scale-prepare
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/scale/slurm-prepare-%A_%a.out

set -euo pipefail

LEAF="${1:?usage: scale_prepare.sbatch LEAF OUTPUT_ROOT}"
OUTPUT_ROOT="${2:?usage: scale_prepare.sbatch LEAF OUTPUT_ROOT}"
REPO="${SCALE_REPO:-/gpfs/radev/scratch/sous/mzl7/dataset_generation_fork}"
cd "$REPO"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load uv/0.9.17

uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
uv run --frozen python tools/scale_generation_driver.py mint-plan \
  --leaf "$LEAF" \
  --block "${SLURM_ARRAY_TASK_ID}" \
  --output-root "$OUTPUT_ROOT" \
  --prepare-workers "${SLURM_CPUS_PER_TASK:-8}"
