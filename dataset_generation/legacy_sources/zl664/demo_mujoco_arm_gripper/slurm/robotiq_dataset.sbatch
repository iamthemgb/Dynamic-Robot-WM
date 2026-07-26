#!/bin/bash
#SBATCH --job-name=rq_f1ab
#SBATCH --partition=gpu_devel
#SBATCH --array=0-7%2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/radev/home/zl664/project/demo_mujoco_arm_gripper/logs/robotiq_f1ab_%A_%a.out
#SBATCH --error=/gpfs/radev/home/zl664/project/demo_mujoco_arm_gripper/logs/robotiq_f1ab_%A_%a.out

# Full Robotiq thick-pad F1_A/F1_B dataset.
# Defaults write 3000 total episodes as 8 shards under:
#   /gpfs/radev/scratch/sous/zl664/robotiq_catch/f1_ab_robotiq_thick_pad_v1/shardNNN
#
# Merge after all shards finish:
#   PYTHONPATH=/gpfs/radev/home/zl664/project/demo_mujoco_arm_gripper:/gpfs/radev/project/sous/mzl7/robocasa \
#   /gpfs/radev/project/sous/zss8/dataset-generation/.venv/bin/python -m robotiq_catch.merge \
#     --root /gpfs/radev/scratch/sous/zl664/robotiq_catch/f1_ab_robotiq_thick_pad_v1 --shards 8

source /gpfs/radev/home/zl664/project/demo_mujoco_arm_gripper/slurm/robotiq_env.sh
umask 0027

NSHARDS=${NSHARDS:-8}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
EPISODES_PER_SUBFAMILY=${EPISODES_PER_SUBFAMILY:-1500}
OUT=${OUT:-/gpfs/radev/scratch/sous/zl664/robotiq_catch/f1_ab_robotiq_thick_pad_v1}
SEED=${SEED:-5100}
RETRY_LIMIT=${RETRY_LIMIT:-12}
SHARD_ROOT="${OUT}/shard$(printf "%03d" "${SHARD}")"

mkdir -p "${OUT}"
chgrp sous "${OUT}" || true
chmod g+rx,g+s "${OUT}" || true

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) shard=${SHARD}/${NSHARDS} eps_per_subfamily=${EPISODES_PER_SUBFAMILY} retry_limit=${RETRY_LIMIT} out=${OUT}"
cd "${PROJECT_DIR}"

set +e
"${PYTHON}" -m robotiq_catch.generate \
    --out "${OUT}" \
    --episodes-per-subfamily "${EPISODES_PER_SUBFAMILY}" \
    --seed "${SEED}" \
    --fps 30 --control-hz 120 --duration 2.5 \
    --width 832 --height 480 \
    --rich-json \
    --retry-limit "${RETRY_LIMIT}" \
    --num-shards "${NSHARDS}" --shard "${SHARD}" \
    --overwrite
STATUS=$?
set -e

if [ -d "${SHARD_ROOT}" ]; then
    chgrp -R sous "${SHARD_ROOT}" || true
    find "${SHARD_ROOT}" -type d -exec chmod g+rx,g+s {} +
    find "${SHARD_ROOT}" -type f -exec chmod g+r {} +
fi

exit "${STATUS}"
