#!/bin/bash
#SBATCH --job-name=rope_ds
#SBATCH --partition=gpu_devel
#SBATCH --array=0-26%6
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/radev/project/sous/zss8/dataset-generation/franka_rope/logs/rope_ds_%A_%a.out
#SBATCH --error=/gpfs/radev/project/sous/zss8/dataset-generation/franka_rope/logs/rope_ds_%A_%a.out

# Franka-rope deformable dataset with broad background variety (plain studio,
# kitchen, warehouse, workshop, lab, office, patio, garage, ...) across 9 tasks.
# The (task x seed) grid is round-robin sharded over the array; each shard
# writes its own episodes<NNN>.jsonl / .parquet into ${OUT}. Only pixels vary
# across backgrounds -- rope/table/arm physics are identical.
#
# Defaults: 9 tasks x 72 seeds x 3 material variants = 1944 episodes over 27
# shards (72 eps/shard, ~4.4 h/shard at ~218 s/episode full-res; gpu_devel QOS
# caps wall-time at 6 h). Override with e.g.
#   sbatch --export=SEEDS_PER_TASK=100,VARIANTS=4 ...
# (keep NSHARDS dividing 9*SEEDS_PER_TASK, and eps/shard*220s < 6 h).

source /gpfs/radev/project/sous/zss8/dataset-generation/franka_rope/slurm/rope_env.sh
umask 0027

NSHARDS=${NSHARDS:-27}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
SEEDS_PER_TASK=${SEEDS_PER_TASK:-72}
VARIANTS=${VARIANTS:-3}
OUT=${OUT:-/gpfs/radev/scratch/sous/zss8/franka_rope/rope_deformable_bgvariety_perview_v1}
DURATION=${DURATION:-5.0}
VIDEO_MODE=${VIDEO_MODE:-split}

mkdir -p "${OUT}" "${PROJECT_DIR}/franka_rope/logs"
chgrp sous "${OUT}" || true
chmod g+rx,g+s "${OUT}" || true

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) shard=${SHARD}/${NSHARDS} seeds=${SEEDS_PER_TASK} variants=${VARIANTS} out=${OUT}"
cd "${PROJECT_DIR}"

set +e
"${PYTHON}" -m franka_rope.generate \
    --out "${OUT}" \
    --seeds-per-task "${SEEDS_PER_TASK}" --variants "${VARIANTS}" \
    --width 640 --height 480 --fps 24 --duration "${DURATION}" \
    --video-mode "${VIDEO_MODE}" \
    --num-shards "${NSHARDS}" --shard-index "${SHARD}"
STATUS=$?
set -e

if [ -d "${OUT}" ]; then
    chgrp -R sous "${OUT}" || true
    find "${OUT}" -type d -exec chmod g+rx,g+s {} + || true
    find "${OUT}" -type f -exec chmod g+r {} + || true
fi

exit "${STATUS}"
