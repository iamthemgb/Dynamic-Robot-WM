# Reproduction commands

Commands in this file have three different evidence levels:

1. the preserved 120-branch diagnostic artifact was actually accepted at the
   commit recorded below;
2. production-revision unit/dry-run commands validate code and configuration but
   do not create release data; and
3. calibration and stage-gate commands evaluate new artifacts and must not be
   reported as passed until their reports exist.

No command here submits a Slurm job, starts Wan training, or launches the
10-hour/100-hour pilots. The pilot YAML files themselves set `submit: false`.
The retired custom-tool SLURM wrappers were removed.

The accepted diagnostic smoke artifacts were generated from commit
`ef1b3c927dce4297441a268ba2cd7e7a97dfb4dc`. Metadata timestamps and the random
dataset UUID naturally change on a rerun; task plans, trajectories, labels,
media content, splits, and schema structure are deterministic for the recorded
seed and environment.

## Environment and tests

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
PYTHONDONTWRITEBYTECODE=1 uv run --frozen pytest -q -p no:cacheprovider
```

The accepted run reported 34 passing tests. The environment digest stored in
`meta/provenance.parquet` is computed from installed distribution metadata and
`uv.lock`; it does not require the optional `pip` module.

## Production-revision contract checks

Run the current tests from the revision being evaluated. Do not substitute the
historical “34 passing” count above for the current test result:

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
PYTHONDONTWRITEBYTECODE=1 uv run --frozen pytest -q -p no:cacheprovider
```

Verify access to the real Panda-hand and Panda+Robotiq generators without
importing, executing, or writing to them:

```bash
uv run --frozen dynamic-robot-dataset inspect-embodiments \
  --require-rigid-sources
```

Validate that the pilot plans are non-submitting and their unique-hour
allocations sum exactly to 10 and 100. This loads configuration only:

```bash
uv run --frozen python - <<'PY'
from dynamic_robot_dataset.common.suites import load_pilot_plan

for path, expected in (
    ("configs/pilots/pilot_10h.yaml", 10.0),
    ("configs/pilots/pilot_100h.yaml", 100.0),
):
    plan = load_pilot_plan(path)
    assert plan["submit"] is False
    assert sum(float(row["unique_hours"]) for row in plan["allocations"]) == expected
    print(path, expected, "unique hours; submit=false")
PY
```

Verify that the retired suite is fail-closed. This command must exit non-zero
and create no output:

```bash
uv run --frozen dynamic-robot-dataset generate-suite \
  --config configs/families/native_acceptance_160.yaml \
  --dry-run
```

## Corrected real-gripper generation

The real source roots may be overridden as follows:

```bash
export FRANKA_HAND_SOURCE_ROOT=/gpfs/radev/project/sous/zss8/dataset-generation/demo_mujoco_arm_gripper
export ROBOTIQ_SOURCE_ROOT=/gpfs/radev/home/zl664/project/demo_mujoco_arm_gripper
export FRANKA_DEFORMABLE_SOURCE_ROOT=/gpfs/radev/home/zl664/project/demo_mujoco_deformable
uv run --frozen dynamic-robot-dataset inspect-embodiments --require-rigid-sources
```

`source_mujoco` execution is deliberately blocked until the LeRobot-to-v2
normalizer and independent objective replay pass. Do not run the original
source generators into the canonical output tree and call that normalized data.

## Physics calibration

Running calibration without observations is an intentional blocked check and
returns a non-zero status. Its report must list free fall, bounce, sliding
deceleration, rolling slip, and contact stability as missing:

```bash
uv run --frozen dynamic-robot-dataset calibrate-physics \
  --config configs/physics/rigid_ranges_v1.yaml \
  --output /tmp/rigid_ranges_blocked.json
```

After a native calibration suite has produced a content-addressed observation
artifact, evaluate it explicitly:

```bash
uv run --frozen dynamic-robot-dataset calibrate-physics \
  --config configs/physics/rigid_ranges_v1.yaml \
  --observations /path/to/native_calibration_observations.json \
  --output /path/to/native_calibration_report.json
```

The observation JSON must use
`dynamic-robot-native-calibration-observations/v1`, bind the resolved catalog
hash, include hash-verified source artifacts with the distinct roles `episodes`
and `qc_report`, provide raw trials for all five required oracles, and declare
the proposed admitted support. Its `approval_artifact` entry must point to a
hash-verified `dynamic-robot-calibration-approval/v1` JSON that names a reviewer
and binds the same catalog and observation payload. The command recomputes the
oracles; an input `passed: true` value is ignored. Candidate values remain
development-only unless the output report says both `passed: true` and
`release_eligible: true`. MuJoCo solver parameters are not a replacement for
measured effective friction or restitution.

## Real-gripper acceptance and release gates

There is intentionally no acceptance generation command yet. The retired
`native_acceptance_160` definition is blocked and its outputs were removed. The
corrected suite must be named `real_gripper_acceptance_160_v2`, contain only
`franka_hand` and `robotiq_2f85_thick_pad`, and pass canonical-v2 conversion,
objective replay, physics, visual, balance, counterfactual, and split checks.

The 10-hour readiness evaluator expects an already-generated canonical dataset,
its canonical `qc/dataset_report.json`, and a separate passed full-native
acceptance artifact. The QC report must use `dynamic-robot-qc-report/v2` and
bind the exact dataset root, complete finalized metadata manifest, complete
episode membership, and every episode media/table hash. Split diagnostics must
bind the current episode and split tables. `--acceptance-report` must name the
canonical
`.suite_execution.json` inside the acceptance dataset declared by that report;
its sibling `.suite_plan.json` must also exist. The evaluator does not generate
or submit a pilot:

```bash
uv run --frozen dynamic-robot-dataset evaluate-readiness \
  --dataset-root /path/to/canonical_10h_candidate \
  --gate configs/release_gates/10h.yaml \
  --acceptance-report /path/to/real_gripper_acceptance_160_v2/.suite_execution.json \
  --wan-root /path/to/optional_wan_export \
  --output /path/to/canonical_10h_candidate/qc/readiness/native_10h_v1.json
```

The 100-hour gate additionally requires an external model-evaluation JSON
artifact. The evaluator checks the configured trajectory/contact improvements,
action ranking, held-out physics result, and visual-quality regression flag.
The artifact provenance must bind the exact candidate episode-table and QC
report hashes. It must give existing `model_artifact_path` and
`evaluation_manifest_path` files whose bytes match their declared lowercase
SHA-256 values:

```bash
uv run --frozen dynamic-robot-dataset evaluate-readiness \
  --dataset-root /path/to/canonical_100h_candidate \
  --gate configs/release_gates/100h.yaml \
  --acceptance-report /path/to/real_gripper_acceptance_160_v2/.suite_execution.json \
  --prerequisite-report /path/to/canonical_10h_candidate/qc/readiness/native_10h_v1.json \
  --wan-root /path/to/optional_wan_export \
  --model-evaluation /path/to/external_model_evaluation.json \
  --output /path/to/canonical_100h_candidate/qc/readiness/native_100h_v1.json
```

Only a report at `DATASET/qc/readiness/GATE_ID.json` can serve as a later
prerequisite. Omit `--output` for an exploratory stdout-only evaluation.

The retired custom-tool report cannot be substituted for the future real-
gripper acceptance artifact in either command. Both readiness evaluations are
therefore expected to remain blocked in this revision. Neither gate
reallocates a missing family quota. The 300-hour and 1,000-hour stages have no
launcher in this revision.

## Free-contact real-gripper development examples

This command renders six development examples using actuator-only robot motion
and native free-contact object dynamics after initialization. It does not
enable `source_mujoco` release generation:

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
MUJOCO_GL=egl .venv/bin/python tools/generate_free_contact_examples.py \
  --config configs/examples/free_contact_real_grippers_v1.yaml \
  --output outputs/examples/free_contact_real_grippers_v1 \
  --overwrite
```

Review `contact_sheet.png`, each episode's `main.mp4` and `secondary.mp4`, and
the measured outcome/evidence in `metadata.json`. The command aborts if a
controller mutates robot or object qpos/qvel after initialization or if any
hashed source file changes during generation.

## Additional rigid-scenario development examples

This command renders ten Panda-hand examples covering catch/retain,
rolling/sliding, rebound, finite-surface transitions, and hand deflection. It
uses world fixtures where the scenario requires a surface, but never mounts a
tray, bin, or paddle on the robot:

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
MUJOCO_GL=egl .venv/bin/python tools/generate_dynamic_scenario_examples.py \
  --config configs/examples/dynamic_scenarios_real_grippers_v1.yaml \
  --output outputs/examples/dynamic_scenarios_real_grippers_v1 \
  --overwrite
```

The suite is development-only and deliberately retains measured failure cases.
Inspect `suite_report.json` and per-episode `metadata.json` before treating an
example as physically valid; `release_eligible` is false for every episode.

## Canonical 120-branch smoke run

This shell invocation corresponds to the script argv stored in provenance
(provenance records `tools/run_smoke_suite.py ...`, without the `uv run python`
launcher prefix):

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv run python tools/run_smoke_suite.py \
  --config configs/families/smoke_120.yaml \
  --output outputs/smoke_tests/canonical_smoke_120
```

The command refuses to overwrite an existing run. To reproduce without moving
the accepted artifacts, choose a new output basename; its Wan export is written
next to it automatically:

```bash
uv run python tools/run_smoke_suite.py \
  --config configs/families/smoke_120.yaml \
  --output outputs/smoke_tests/reproduced_smoke_120
```

## Standalone verification

```bash
uv run dynamic-robot-dataset validate \
  --dataset-root outputs/smoke_tests/canonical_smoke_120

uv run python tools/validate_physics.py \
  --dataset-root outputs/smoke_tests/canonical_smoke_120
```

The smoke runner already performs finalization, deterministic group-aware
split construction, deep video/metadata/label/duplicate/leakage QC, the Wan
round trip, and contact-sheet creation. The standalone commands rerun the two
main validators read-only.

## Key accepted artifacts

```text
outputs/smoke_tests/canonical_smoke_120/.generation.json
outputs/smoke_tests/canonical_smoke_120/meta/.complete.json
outputs/smoke_tests/canonical_smoke_120/qc/smoke_summary.json
outputs/smoke_tests/canonical_smoke_120/qc/dataset_report.json
outputs/smoke_tests/canonical_smoke_120/qc/contact_sheets/samples.json
outputs/smoke_tests/canonical_smoke_120_wan/manifest.jsonl
outputs/smoke_tests/canonical_smoke_120_wan/checksums.json
outputs/smoke_tests/canonical_smoke_120_wan/source_root.json
```
