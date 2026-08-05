#!/bin/bash
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:15:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%A_%a.out

# Persistent generation worker: holds ONE GPU for the whole walltime and
# pulls groups off a shared work list until it is empty or the time window
# closes — the long-allocation alternative to one-task-per-group arrays.
# Launch N of them (e.g. --array=0-4 for a 5-GPU pool) via
# submit_rolling_workers.sh, which also builds the work list.
#
#   cd <old_dataset_gen>
#   bash slurm/submit_rolling_workers.sh <DATASET_ROOT> <PHASE_CONFIG> [N=5] [HOURS=6]
#
# Work list: one "<family> <group_index>" per line. Claim protocol: mkdir
# under DATASET_ROOT/.claims/ — atomic on NFSv3 (flock is not). Claims older
# than STALE_MIN whose group.json never appeared are reclaimed, and episode
# generation itself is marker-skippable, so a crashed worker costs nothing
# and duplicated claims at worst redo one group.

set -euo pipefail

WORKLIST="${WORKLIST:?WORKLIST must be exported (see submit_rolling_workers.sh)}"
DATASET_ROOT="${DATASET_ROOT:?DATASET_ROOT must be exported}"
PHASE_CONFIG="${PHASE_CONFIG:?PHASE_CONFIG must be exported}"
OLD_ROOT="${OLD_ROOT:-${SLURM_SUBMIT_DIR}}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${OLD_ROOT}/projectile_ball_catch_export}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"
WORK_SECONDS="${WORK_SECONDS:-21600}"
STALE_MIN="${STALE_MIN:-90}"
DRAIN_MARGIN_S="${DRAIN_MARGIN_S:-900}"

export MUJOCO_GL=egl
export PYTHONPATH="${OLD_ROOT}:${BUNDLE_ROOT}/robosuite:${BUNDLE_ROOT}/robocasa_full:${PYTHONPATH:-}"
export ROBOCASA_ASSETS_ROOT="${BUNDLE_ROOT}/robocasa_full/robocasa/models/assets"

CLAIMS="${DATASET_ROOT}/.claims"
mkdir -p "${CLAIMS}"
DEADLINE=$(( $(date +%s) + WORK_SECONDS - DRAIN_MARGIN_S ))
WORKER="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
DONE_COUNT=0

group_complete() {
  # complete <=> group.json exists (written only after every sibling commits)
  [ -f "${DATASET_ROOT}/$1/groups/$2/group.json" ]
}

claim() {
  local dir="${CLAIMS}/$1"
  if mkdir "${dir}" 2>/dev/null; then
    echo "${WORKER} $(date +%s)" > "${dir}/owner"
    return 0
  fi
  # reclaim: stale claim with no finished group behind it
  local ts now
  ts=$(stat -c %Y "${dir}" 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ $(( now - ts )) -gt $(( STALE_MIN * 60 )) ]; then
    rm -rf "${dir}" 2>/dev/null || true
    if mkdir "${dir}" 2>/dev/null; then
      echo "${WORKER} $(date +%s) reclaimed" > "${dir}/owner"
      return 0
    fi
  fi
  return 1
}

echo "worker ${WORKER}: window $(( WORK_SECONDS / 60 )) min, list ${WORKLIST}"
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
  PICKED=""
  while read -r fam gidx; do
    [ -z "${fam}" ] && continue
    gid="${fam}_g$(printf '%06d' "${gidx}")"
    group_complete "${fam}" "${gid}" && continue
    claim "${gid}" || continue
    if group_complete "${fam}" "${gid}"; then
      continue                      # finished between the check and the claim
    fi
    PICKED="${fam} ${gidx} ${gid}"
    break
  done < "${WORKLIST}"
  if [ -z "${PICKED}" ]; then
    echo "worker ${WORKER}: work list drained after ${DONE_COUNT} group(s)"
    exit 0
  fi
  set -- ${PICKED}
  fam="$1"; gidx="$2"; gid="$3"
  echo "worker ${WORKER}: generating ${gid}"
  "${PYTHON_BIN}" -m ball_rolling_dynamics_scripts.group_dataset_generation group \
    --family "${fam}" --group-index "${gidx}" \
    --phase-config "${PHASE_CONFIG}" --dataset-root "${DATASET_ROOT}"
  "${PYTHON_BIN}" - "${DATASET_ROOT}/${fam}/groups/${gid}/group.json" <<'PYEOF'
import json, sys
group = json.loads(open(sys.argv[1], encoding="utf-8").read())
bad = [s["episode_id"] for s in group["siblings"]
       if s.get("scene_resolution_backend") != "robocasa_native"]
if bad:
    raise SystemExit(f"non-native backend in {bad}")
PYEOF
  DONE_COUNT=$(( DONE_COUNT + 1 ))
done
echo "worker ${WORKER}: time window closing, drained cleanly after ${DONE_COUNT} group(s)"
