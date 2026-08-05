#!/bin/bash
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out

# One array task = one counterfactual group (all siblings, sequential;
# requeue-safe: already-committed siblings are skipped via their markers).
# Submit FROM old_dataset_gen/ so the relative logs/ path and default
# OLD_ROOT resolve correctly:
#   cd <old_dataset_gen>
#   FAMILY=rolling_layout38_style42 DATASET_ROOT=... PHASE_CONFIG=configs/rolling_groups_p1.json \
#     sbatch --array=0-332 slurm/generate_rolling_group_array.sbatch

set -euo pipefail

FAMILY="${FAMILY:?FAMILY must be exported}"
PHASE_CONFIG="${PHASE_CONFIG:?PHASE_CONFIG must be exported}"
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

"${PYTHON_BIN}" -m ball_rolling_dynamics_scripts.group_dataset_generation group \
  --family "${FAMILY}" \
  --group-index "${SLURM_ARRAY_TASK_ID}" \
  --phase-config "${PHASE_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --duration "${DURATION}"

# Post-generation guard: every sibling of this group must have rendered with
# the native backend (read back through the committed markers).
"${PYTHON_BIN}" - <<PYEOF
import json
from pathlib import Path
from ball_rolling_dynamics_scripts.group_dataset_generation import group_id_for

group_id = group_id_for("${FAMILY}", int("${SLURM_ARRAY_TASK_ID}"))
group_json = Path("${DATASET_ROOT}") / "${FAMILY}" / "groups" / group_id / "group.json"
group = json.loads(group_json.read_text(encoding="utf-8"))
bad = [
    sibling["episode_id"]
    for sibling in group["siblings"]
    if sibling.get("scene_resolution_backend") != "robocasa_native"
]
if bad:
    raise SystemExit(f"ERROR: non-native backend in siblings: {bad}")
print(f"{group_id}: {len(group['siblings'])} siblings, backend robocasa_native")
PYEOF
