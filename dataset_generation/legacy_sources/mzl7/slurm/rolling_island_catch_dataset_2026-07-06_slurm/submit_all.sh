#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/gpfs/radev/scratch/sous/mzl7/rolling_island_catch_dataset_2026-07-06}"
COUNT="${COUNT:-1500}"
WIDTH="${WIDTH:-960}"
HEIGHT="${HEIGHT:-540}"
FPS="${FPS:-30}"
DURATION="${DURATION:-2.5}"
CONCURRENCY="${CONCURRENCY:-8}"

families=(
  "rolling_layout38_style42"
  "rolling_layout48_style41"
  "rolling_layout51_style34"
)

for family in "${families[@]}"; do
  array_job_id="$(sbatch --parsable \
    --job-name="catch_${family}" \
    --array="0-$((COUNT - 1))%${CONCURRENCY}" \
    --export="ALL,FAMILY=${family},COUNT=${COUNT},DATASET_ROOT=${DATASET_ROOT},WIDTH=${WIDTH},HEIGHT=${HEIGHT},FPS=${FPS},DURATION=${DURATION}" \
    "${SCRIPT_DIR}/generate_family_array.sbatch")"
  finalize_job_id="$(sbatch --parsable \
    --job-name="finalize_${family}" \
    --dependency="afterok:${array_job_id}" \
    --export="ALL,FAMILY=${family},DATASET_ROOT=${DATASET_ROOT},WIDTH=${WIDTH},HEIGHT=${HEIGHT}" \
    "${SCRIPT_DIR}/finalize_family.sbatch")"
  printf '%s %s %s\n' "${family}" "${array_job_id}" "${finalize_job_id}"
done
