#!/bin/bash
# Dry-run smoke test for the grouped rolling dataset mechanism (no GPU, no
# rendering): plan determinism, sibling speed stratification, split hashing.
# Run from old_dataset_gen/:  bash smoke_test_rolling_groups.sh

set -euo pipefail

OLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-${OLD_ROOT}/projectile_ball_catch_export}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python}"
export PYTHONPATH="${OLD_ROOT}:${BUNDLE_ROOT}/robosuite:${BUNDLE_ROOT}/robocasa_full:${PYTHONPATH:-}"
export ROBOCASA_ASSETS_ROOT="${BUNDLE_ROOT}/robocasa_full/robocasa/models/assets"

FAMILY="${FAMILY:-rolling_layout38_style42}"
CONFIG="${CONFIG:-${OLD_ROOT}/configs/rolling_groups_p0.json}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

plan() {
  # robosuite prints a mimicgen WARNING to stdout before the JSON; keep from '{'
  "${PYTHON_BIN}" -m ball_rolling_dynamics_scripts.group_dataset_generation plan \
    --family "${FAMILY}" --group-index "$1" --phase-config "${CONFIG}" \
    | sed -n '/^{/,$p'
}

echo "[1/4] plan determinism (group 0 twice)"
plan 0 > "${WORK}/g0a.json"
plan 0 > "${WORK}/g0b.json"
diff "${WORK}/g0a.json" "${WORK}/g0b.json"
echo "  identical"

echo "[2/4] group distinctness (group 1)"
plan 1 > "${WORK}/g1.json"

echo "[3/4] sibling separation, strata, caps"
"${PYTHON_BIN}" - "$WORK/g0a.json" "$WORK/g1.json" <<'PYEOF'
import itertools
import json
import sys

groups = [json.loads(open(path, encoding="utf-8").read()) for path in sys.argv[1:]]
sep = None
for group in groups:
    plan = group["speed_plan"]
    assert plan["stratified"], "expected stratified speeds"
    sep = plan["min_speed_separation_mps"]
    speeds = [s["state_context"]["speed_mps"] for s in group["siblings"]]
    strata = sorted(s["speed_stratum"] for s in group["siblings"])
    assert strata == list(range(len(speeds))), f"bad strata permutation {strata}"
    for a, b in itertools.combinations(speeds, 2):
        assert abs(a - b) >= sep - 1e-9, f"{group['group_id']}: speeds {a:.3f},{b:.3f} too close"
    for sibling in group["siblings"]:
        speed = sibling["state_context"]["speed_mps"]
        assert speed <= sibling["speed_cap_mps"] + 1e-9, "speed above per-heading cap"
        assert speed <= plan["group_speed_cap_mps"] + 1e-9, "speed above group cap"
v0 = {tuple(s["state_context"]["ball_initial_velocity_mps"]) for s in groups[0]["siblings"]}
v1 = {tuple(s["state_context"]["ball_initial_velocity_mps"]) for s in groups[1]["siblings"]}
assert not (v0 & v1), "groups 0 and 1 share a sibling velocity"
print(f"  {len(groups)} groups, sep >= {sep} m/s, strata valid, caps respected")
PYEOF

echo "[4/4] split hashing (group-atomic, ~90/5/5 over 999 ids)"
"${PYTHON_BIN}" - <<'PYEOF'
from collections import Counter
from ball_rolling_dynamics_scripts.group_dataset_generation import group_id_for, split_for_group

percent = {"train": 90, "val": 5, "test": 5}
counts = Counter(
    split_for_group(group_id_for(family, index), percent)
    for family in ("rolling_layout38_style42", "rolling_layout48_style41", "rolling_layout51_style34")
    for index in range(333)
)
total = sum(counts.values())
assert total == 999
assert 0.85 <= counts["train"] / total <= 0.95, counts
assert counts["val"] and counts["test"], counts
print(f"  splits over {total} group ids: {dict(counts)}")
PYEOF

echo "SMOKE OK"
