#!/bin/bash
# Submit the Franka-cloth (bland-table) dataset: one SLURM array per task family
# (episode index i -> seed i//3, variant i%3) plus a dependent finalize job per
# family. Each episode randomizes cloth color/material, robot base + approach
# pose, table color, background/skybox color, floor color and light intensity;
# the scripted outcome_branch (success/near_miss/execution_failure/wrong_action,
# 50/20/20/10) is keyed on the seed.
#
#   COUNT=12 CONCURRENCY=4 ./submit_all.sh     # pilot (4 seeds/task, 48 videos)
#   ./submit_all.sh                            # full 1500/task (6000 videos)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/gpfs/radev/scratch/sous/mzl7/franka_cloth_previews_dataset_2026-07-08}"
COUNT="${COUNT:-1500}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-24}"
DURATION="${DURATION:-5.0}"
CONCURRENCY="${CONCURRENCY:-8}"

tasks=(
  "poke_cloth"
  "lift_corner_release"
  "fold_edge_fixed_line"
  "dual_franka_tshirt_fold_box"
)

mkdir -p "${DATASET_ROOT}"

for task in "${tasks[@]}"; do
  array_job_id="$(sbatch --parsable \
    --job-name="cloth_${task}" \
    --array="0-$((COUNT - 1))%${CONCURRENCY}" \
    --export="ALL,TASK=${task},DATASET_ROOT=${DATASET_ROOT},WIDTH=${WIDTH},HEIGHT=${HEIGHT},FPS=${FPS},DURATION=${DURATION}" \
    "${SCRIPT_DIR}/generate_task_array.sbatch")"
  finalize_job_id="$(sbatch --parsable \
    --job-name="finalize_${task}" \
    --dependency="afterany:${array_job_id}" \
    --export="ALL,TASK=${task},DATASET_ROOT=${DATASET_ROOT},COUNT=${COUNT}" \
    "${SCRIPT_DIR}/finalize_task.sbatch")"
  printf '%s array=%s finalize=%s\n' "${task}" "${array_job_id}" "${finalize_job_id}"
done
