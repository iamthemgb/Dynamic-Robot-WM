#!/bin/bash
# Submit the Franka-cloth kitchen dataset: one SLURM array per task family
# (episode index i -> seed i//3, variant i%3) plus a finalize job per family.
#
#   COUNT=24 CONCURRENCY=8 ./submit_all.sh     # pilot (8 seeds/task)
#   ./submit_all.sh                            # full 1500/task

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/gpfs/radev/scratch/sous/mzl7/franka_cloth_kitchen_dataset_2026-07-07}"
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
