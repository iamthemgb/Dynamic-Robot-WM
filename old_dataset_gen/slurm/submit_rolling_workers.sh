#!/bin/bash
# Launch a persistent worker pool for grouped rolling generation: N jobs of
# one GPU each, every job holding its GPU for HOURS and pulling groups from a
# shared work list (see rolling_gen_worker.sbatch). Builds the work list from
# the phase config minus the groups already complete on disk. Run FROM
# old_dataset_gen/:
#
#   bash slurm/submit_rolling_workers.sh <DATASET_ROOT> <PHASE_CONFIG> [N=5] [HOURS=6]
#
# DRY_RUN=1 prints instead of submitting.

set -euo pipefail
shopt -s inherit_errexit

DATASET_ROOT="${1:?usage: submit_rolling_workers.sh DATASET_ROOT PHASE_CONFIG [N] [HOURS]}"
PHASE_CONFIG="${2:?usage: submit_rolling_workers.sh DATASET_ROOT PHASE_CONFIG [N] [HOURS]}"
N_WORKERS="${3:-5}"
HOURS="${4:-6}"
OLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

cd "${OLD_ROOT}"
mkdir -p logs "${DATASET_ROOT}"

WORKLIST="${DATASET_ROOT}/.worklist"
"${PYTHON_BIN}" - "${PHASE_CONFIG}" "${DATASET_ROOT}" "${WORKLIST}" <<'PYEOF'
import json
import sys
from pathlib import Path

config_path, dataset_root, worklist = sys.argv[1:4]
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
families = ("rolling_layout38_style42", "rolling_layout48_style41",
            "rolling_layout51_style34")
groups = int(config["groups_per_family"])
lines, done = [], 0
for family in families:
    for index in range(groups):
        gid = f"{family}_g{index:06d}"
        if (Path(dataset_root) / family / "groups" / gid / "group.json").exists():
            done += 1
            continue
        lines.append(f"{family} {index}")
Path(worklist).write_text("\n".join(lines) + ("\n" if lines else ""),
                          encoding="utf-8")
print(f"work list: {len(lines)} group(s) to generate, {done} already complete")
PYEOF

if [ ! -s "${WORKLIST}" ]; then
  echo "nothing to do — every group is already complete"
  exit 0
fi

WORK_SECONDS=$(( HOURS * 3600 ))
TIME_LIMIT=$(printf '%02d:%02d:00' $(( HOURS )) 15)   # walltime = window + drain margin

CMD=(sbatch --job-name=roll-gen-worker
     --array="0-$(( N_WORKERS - 1 ))"
     --time="${TIME_LIMIT}"
     --export=ALL,WORKLIST="${WORKLIST}",DATASET_ROOT="${DATASET_ROOT}",PHASE_CONFIG="${PHASE_CONFIG}",WORK_SECONDS="${WORK_SECONDS}"
     slurm/rolling_gen_worker.sbatch)

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY ${CMD[*]}"
else
  "${CMD[@]}"
fi
echo "${N_WORKERS} worker(s) x 1 GPU x ${HOURS}h window over $(wc -l < "${WORKLIST}") group(s)"
