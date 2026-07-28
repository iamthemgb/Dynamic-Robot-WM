#!/bin/bash
# Idempotent cache refill (latents shards 0-3, t5 on task 4). The full cache
# already exists from the cloth-plan work; this only fills gaps.
#SBATCH --job-name=proj_smoke_precompute
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --constraint="a100|h100|l40s"
#SBATCH --array=0-4
#SBATCH --output=/gpfs/radev/scratch/sous/mzl7/wan_projectile_smoke_cache/logs/precompute_%A_%a.log

set -euo pipefail
# Submit from the repo root (the dir containing projectile_smoke/).
cd "${SLURM_SUBMIT_DIR}"
export WAN21_ROOT=${WAN21_ROOT:-/gpfs/radev/scratch/sous/mzl7/Wan2.1}
PY=~/.venvs/wan21/bin/python

if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
  $PY projectile_smoke/data/precompute_latents.py \
      --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
else
  $PY projectile_smoke/data/precompute_t5.py
fi
