#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/home/mzl7/scratch/neutral_style_rollouts_2026-07-01}"
COUNT="${COUNT:-750}"

families=(
  "style020_seed0"
  "style020_seed0_opposite_camera"
  "style055_seed1000"
  "style055_seed1000_opposite_camera"
)

for family in "${families[@]}"; do
  array_job_id="$(sbatch --parsable \
    --job-name="catch_${family}" \
    --array="0-$((COUNT - 1))" \
    --export="ALL,FAMILY=${family},COUNT=${COUNT},DATASET_ROOT=${DATASET_ROOT}" \
    "${SCRIPT_DIR}/generate_family_array.sbatch")"
  finalize_job_id="$(sbatch --parsable \
    --job-name="finalize_${family}" \
    --dependency="afterok:${array_job_id}" \
    --export="ALL,FAMILY=${family},DATASET_ROOT=${DATASET_ROOT}" \
    "${SCRIPT_DIR}/finalize_family.sbatch")"
  printf '%s %s %s\n' "${family}" "${array_job_id}" "${finalize_job_id}"
done
