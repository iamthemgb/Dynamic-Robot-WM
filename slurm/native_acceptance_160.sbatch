#!/usr/bin/env bash
#SBATCH --job-name=dynamic-native-acceptance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/native_acceptance/slurm-%j.out

set -euo pipefail

REPO=/gpfs/radev/project/sous/zl664/dataset_generation
OUTPUT=${DYNAMIC_ROBOT_ACCEPTANCE_OUTPUT:?Set DYNAMIC_ROBOT_ACCEPTANCE_OUTPUT to a new dataset directory}

cd "$REPO"
export MUJOCO_GL=egl

uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
set +e
uv run dynamic-robot-dataset generate-suite \
  --config configs/families/native_acceptance_160.yaml \
  --output "$OUTPUT"
SUITE_STATUS=$?

uv run dynamic-robot-dataset finalize \
  --dataset "$OUTPUT" \
  --name native_acceptance_160 \
  --description "Immutable 160-branch native/contact-tiering acceptance suite"
FINALIZE_STATUS=$?

uv run dynamic-robot-dataset build-splits --dataset "$OUTPUT" --seed 0
SPLIT_STATUS=$?

uv run dynamic-robot-dataset qc --dataset "$OUTPUT"
QC_STATUS=$?

uv run dynamic-robot-dataset stats --dataset "$OUTPUT" --output "$OUTPUT/qc/statistics.json"
STATS_STATUS=$?
set -e

# Acceptance mismatches are deliberately not retried or relabeled.  Always run
# finalization and reporting so a failed suite leaves inspectable evidence, then
# propagate the first non-zero status to Slurm.
for STATUS in "$SUITE_STATUS" "$FINALIZE_STATUS" "$SPLIT_STATUS" "$QC_STATUS" "$STATS_STATUS"; do
  if [[ "$STATUS" -ne 0 ]]; then
    exit "$STATUS"
  fi
done
