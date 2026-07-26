#!/bin/bash
#SBATCH --job-name=rq_merge
#SBATCH --partition=gpu_devel
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/radev/project/sous/zss8/dataset-generation/robotiq_arm_gripper/logs/robotiq_merge_%j.out
#SBATCH --error=/gpfs/radev/project/sous/zss8/dataset-generation/robotiq_arm_gripper/logs/robotiq_merge_%j.out

# Merge the 8 shards into one LeRobot dataset. Submit with a dependency on the
# full array so it runs automatically once every shard succeeds:
#   sbatch --dependency=afterok:<full_array_jobid> slurm/robotiq_merge.sbatch
source /gpfs/radev/project/sous/zss8/dataset-generation/robotiq_arm_gripper/slurm/robotiq_env.sh
umask 0027

NSHARDS=${NSHARDS:-8}
OUT=${OUT:-/gpfs/radev/scratch/sous/zss8/robotiq_catch/f1_ab_robotiq_thick_pad_v2_closer_bg}
cd "${PROJECT_DIR}"
"${PYTHON}" -m robotiq_catch.merge --root "${OUT}" --shards "${NSHARDS}"
STATUS=$?
if [ -d "${OUT}" ]; then
    chgrp -R sous "${OUT}" || true
    find "${OUT}" -type d -exec chmod g+rx,g+s {} +
    find "${OUT}" -type f -exec chmod g+r {} +
fi
exit "${STATUS}"
