#!/bin/bash
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out

# Submit FROM old_dataset_gen/ so the relative logs/ path and default
# OLD_ROOT resolve correctly:
#   cd <old_dataset_gen>
#   FAMILY=rolling_layout38_style42 DATASET_ROOT=... sbatch --array=0-749 slurm/generate_rolling_family_array.sbatch

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
COUNT="${COUNT:-750}"
OLD_ROOT="${OLD_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${OLD_ROOT}/projectile_ball_catch_export}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
DURATION="${DURATION:-2.5}"

export MUJOCO_GL=egl
export PYTHONPATH="${OLD_ROOT}:${BUNDLE_ROOT}/robosuite:${BUNDLE_ROOT}/robocasa_full:${PYTHONPATH:-}"
export ROBOCASA_ASSETS_ROOT="${BUNDLE_ROOT}/robocasa_full/robocasa/models/assets"

# Fail fast before rendering if the scene cannot be built natively
# (yaml_fallback kitchens are not acceptable output for these families).
"${PYTHON_BIN}" - <<PYEOF
from ball_rolling_dynamics_scripts.dataset_generation import FAMILY_SPECS
from ball_rolling_dynamics_scripts.yaml_scene import resolve_scene
spec = FAMILY_SPECS["${FAMILY}"]
scene = resolve_scene(spec["layout_id"], spec["style_id"])
if scene.backend != "robocasa_native":
    raise SystemExit(
        f"Refusing to generate: backend={scene.backend} native_error={scene.native_error}"
    )
PYEOF

"${PYTHON_BIN}" -m ball_rolling_dynamics_scripts.dataset_generation episode \
  --family "${FAMILY}" \
  --episode-index "${SLURM_ARRAY_TASK_ID}" \
  --episode-count "${COUNT}" \
  --dataset-root "${DATASET_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --duration "${DURATION}"

# The episode is committed as a canonical-layout marker + sharded artifacts
# (data/videos/... under ${DATASET_ROOT}/${FAMILY}); read the recorded backend
# back through the marker rather than a per-episode metadata.json.
BACKEND="$("${PYTHON_BIN}" -m ball_rolling_dynamics_scripts.dataset_generation episode-backend \
  --family "${FAMILY}" \
  --episode-index "${SLURM_ARRAY_TASK_ID}" \
  --dataset-root "${DATASET_ROOT}")"
if [ "${BACKEND}" != "robocasa_native" ]; then
  echo "ERROR: episode rendered with backend=${BACKEND}, expected robocasa_native" >&2
  exit 1
fi
