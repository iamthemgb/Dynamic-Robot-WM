#!/usr/bin/env bash
# End-to-end calibration block for one leaf on one A40: mint + plan +
# generate + finalize ~100 sampled episodes and print timing/size metrics.
#
# Usage:
#   sbatch slurm/scale_calibrate_a40.sbatch <LEAF> <OUTPUT_ROOT> [EPISODES]
# Example:
#   sbatch slurm/scale_calibrate_a40.sbatch F1a \
#     /gpfs/radev/scratch/sous/mzl7/dynamic_rollouts 100
#
#SBATCH --job-name=dynamic-robot-scale-calibrate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/scale/slurm-calibrate-%j.out

set -euo pipefail

LEAF="${1:?usage: scale_calibrate_a40.sbatch LEAF OUTPUT_ROOT [EPISODES] [BLOCK]}"
OUTPUT_ROOT="${2:?usage: scale_calibrate_a40.sbatch LEAF OUTPUT_ROOT [EPISODES] [BLOCK]}"
EPISODES="${3:-100}"
BLOCK="${4:-9000}"
REPO="${SCALE_REPO:-/gpfs/radev/scratch/sous/mzl7/dataset_generation_fork}"
cd "$REPO"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load uv/0.9.17
export MUJOCO_GL=egl

uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
uv run --frozen python tools/scale_generation_driver.py calibrate \
  --leaf "$LEAF" \
  --episodes "$EPISODES" \
  --block "$BLOCK" \
  --episode-start "$((9000000 + BLOCK * 1000))" \
  --output-root "$OUTPUT_ROOT" \
  --shards 4 \
  --prepare-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --deep-video-checks
