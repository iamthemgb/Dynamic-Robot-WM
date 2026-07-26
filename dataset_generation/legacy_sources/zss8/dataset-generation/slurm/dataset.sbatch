#!/bin/bash
#SBATCH --job-name=fc_f1a
#SBATCH --partition=gpu_devel
#SBATCH --array=0-3%2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/radev/project/sous/zss8/dataset-generation/logs/f1a_%A_%a.out
#SBATCH --error=/gpfs/radev/project/sous/zss8/dataset-generation/logs/f1a_%A_%a.out

# Full F1_A centered-vertical-drop dataset (1500 episodes, 2 views each).
# Sharded across the array; each shard writes <OUT>/shardNNN. Merge afterwards
# with:  python -m franka_catch.merge --root <OUT> --shards N
#
# Overrides:  sbatch --export=ALL,EPISODES=1500,OUT=/path,NSHARDS=4 slurm/dataset.sbatch
source /gpfs/radev/project/sous/zss8/dataset-generation/slurm/env.sh

NSHARDS=${NSHARDS:-4}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
EPISODES=${EPISODES:-1500}
OUT=${OUT:-/gpfs/radev/scratch/sous/zss8/franka_catch/f1a_centered_v1}
SEED=${SEED:-1000}

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) shard=${SHARD}/${NSHARDS} episodes=${EPISODES} out=${OUT}"
cd "${PROJECT_DIR}"

"${PYTHON}" -m franka_catch.generate \
    --out "${OUT}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --fps 30 --control-hz 120 --duration 4.0 \
    --width 512 --height 512 \
    --rich-json \
    --num-shards "${NSHARDS}" --shard "${SHARD}"
