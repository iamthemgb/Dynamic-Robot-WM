#!/bin/bash
# Smoke test for the v3 state-conditional grouped dataset code.
# Uses ONLY the `plan` dry-run subcommand: no simulation, no rendering, and no
# writes outside a temp dir — safe on a login node, no GPU needed.
#   cd <bundle root> && bash smoke_test_group_dataset.sh

set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"
MODULE=projectile_ball_catch_robocasa_kitchen_scenes_scripts.group_dataset_generation

export PYTHONPATH="${BUNDLE_ROOT}:${PYTHONPATH:-}"
cd "${BUNDLE_ROOT}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

plan_json() {
  # robosuite prints a mimicgen WARNING to stdout before the JSON; keep from the first '{'.
  "${PYTHON_BIN}" -m "${MODULE}" plan --family "$1" --group-index "$2" \
    --phase-config configs/p0_pilot.json 2>/dev/null | sed -n '/^{/,$p' > "$3"
}

echo "[1/4] dry-run group construction, both families"
plan_json style020_seed0 0 "${WORKDIR}/f1.json"
plan_json style055_seed1000 9 "${WORKDIR}/f2.json"

echo "[2/4] determinism across reruns"
plan_json style020_seed0 0 "${WORKDIR}/f1_again.json"
diff -q "${WORKDIR}/f1.json" "${WORKDIR}/f1_again.json" > /dev/null

echo "[3/4] structure and distribution checks"
"${PYTHON_BIN}" - "${WORKDIR}/f1.json" "${WORKDIR}/f2.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    group = json.load(open(path))
    siblings = group["siblings"]
    assert len(siblings) >= 2, f"{group['group_id']}: need >=2 siblings"
    states = [
        tuple(s["state_context"]["ball_initial_position_m"])
        + tuple(s["state_context"]["ball_initial_velocity_mps"])
        + (s["state_context"]["release_time_s"],)
        for s in siblings
    ]
    assert len(set(states)) == len(states), f"{group['group_id']}: duplicate sibling ICs"
    assert group["split"] in {"train", "val", "test"}
    assert group["physics_regime"] == "nominal_constant_v3"
    assert group["shared_context"]["ball_color"], "shared appearance missing"
    print(f"    {group['group_id']}: {len(siblings)} siblings, split={group['split']}, "
          f"branches={[s['branch'] for s in siblings]}")
PY
"${PYTHON_BIN}" - <<'PY'
from collections import Counter
from pathlib import Path

from projectile_ball_catch_robocasa_kitchen_scenes_scripts.group_dataset_generation import (
    _branch_schedule, _load_phase_config, group_id_for, split_for_group)

config = _load_phase_config(Path("configs/p1_full.json"))
splits = Counter(
    split_for_group(group_id_for(family, g), config["split_percent"])
    for family in ("style020_seed0", "style055_seed1000")
    for g in range(config["groups_per_family"])
)
assert 850 <= splits["train"] <= 950, f"train split off target: {dict(splits)}"
schedule = _branch_schedule("style020_seed0", config)
counts = Counter(schedule)
assert counts["success"] == 1000 and counts["wrong_action"] == 200, dict(counts)
assert schedule == _branch_schedule("style020_seed0", config), "branch schedule not deterministic"
print(f"    p1 splits {dict(splits)}; branch counts {dict(counts)}; schedule deterministic")
PY

echo "[4/4] validator CLI loads"
"${PYTHON_BIN}" -m projectile_ball_catch_robocasa_kitchen_scenes_scripts.validate_group_dataset --help > /dev/null 2>&1

echo "SMOKE TEST PASSED"
