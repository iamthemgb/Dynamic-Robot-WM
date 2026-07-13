# Reproduction commands

Commands in this file have three different evidence levels:

1. the preserved 120-branch diagnostic artifact was actually accepted at the
   commit recorded below;
2. production-revision unit/dry-run commands validate code and configuration but
   do not create release data; and
3. native calibration, acceptance, and stage-gate commands create or evaluate
   new artifacts and must not be reported as passed until their reports exist.

No command here submits a Slurm job, starts Wan training, or launches the
10-hour/100-hour pilots. The pilot YAML files themselves set `submit: false`.
The native Slurm wrappers are provided but were not submitted; using `sbatch`
is a separate, explicit operator action.

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

Expand the native acceptance definition without simulating or writing a
dataset. The JSON result must report exactly 160 cases:

```bash
uv run --frozen dynamic-robot-dataset generate-suite \
  --config configs/families/native_acceptance_160.yaml \
  --dry-run \
  > /tmp/native_acceptance_160.plan.json

uv run --frozen python - <<'PY'
import json
from pathlib import Path

plan = json.loads(Path("/tmp/native_acceptance_160.plan.json").read_text())
assert plan["case_count"] == 160
assert plan["execution_backend_counts"] == {
    "native_mujoco": 128,
    "diagnostic_quarantine": 32,
}
assert plan["full_native"] is False
print(plan["suite"], plan["case_count"], plan["execution_backend_counts"])
PY
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

Execute the 128 rigid acceptance branches without rendering. This is a
development physics/outcome audit only; the JSON explicitly records that it is
not a rendered, calibrated, finalized, or fully native 160-case artifact:

```bash
uv run --frozen python tools/audit_native_rigid_suite.py \
  --config configs/families/native_acceptance_160.yaml
```

## Native rigid smoke generation

Set the read-only MuJoCo Menagerie root (or `FRANKA_MJCF_PATH`) and generate a
small dedicated output. `native_mujoco` owns both simulation and rendering; do
not pass the legacy `--renderer` plug-in for this path:

```bash
export MUJOCO_MENAGERIE_ROOT=/path/to/mujoco_menagerie
export MUJOCO_GL=egl

uv run --frozen dynamic-robot-dataset generate \
  --backend native_mujoco \
  --family falling_catch \
  --subfamily catch_retain \
  --num-bundles 1 \
  --branches success_seeking,near_miss,contact_failure,no_op \
  --views main,secondary \
  --seed 20260713 \
  --scene-style clean_franka_lab \
  --output outputs/native_smoke/catch_retain_v2

uv run --frozen dynamic-robot-dataset finalize \
  --dataset-root outputs/native_smoke/catch_retain_v2
uv run --frozen dynamic-robot-dataset build-splits \
  --dataset-root outputs/native_smoke/catch_retain_v2 \
  --config configs/splits/default.yaml
uv run --frozen dynamic-robot-dataset qc \
  --dataset-root outputs/native_smoke/catch_retain_v2
uv run --frozen dynamic-robot-dataset stats \
  --dataset-root outputs/native_smoke/catch_retain_v2 \
  --output outputs/native_smoke/catch_retain_v2/qc/statistics.json
```

This smoke output is not release data while the referenced parameter-range
profile is provisional. Inspect the QC report, decoded videos, contacts, state
tables, objective recomputation, and backend provenance rather than treating a
zero process exit as physical validation.

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

## Native acceptance and release gates

The following is the full 160-case output command. Its executor routes 128 rigid
cases through `native_mujoco` and 32 cloth/rope/negative controls through
`diagnostic_quarantine`. It is provided for exact reproduction, but it was not
run in this revision and must not be submitted as a production-scale job. Its
execution report records `full_native: false`, and stage readiness must remain
failed until native deformable backends exist:

```bash
uv run --frozen dynamic-robot-dataset generate-suite \
  --config configs/families/native_acceptance_160.yaml \
  --output outputs/acceptance/native_acceptance_160
```

Before simulation, the command writes the immutable expected-membership ledger
`.suite_plan.json`. It records every case UUID, backend route, counterfactual
sibling, intervention field, and fixed-field hash. Every case then receives one
immutable committed/failed attempt record; outcome mismatch is never retried.
The aggregate `.suite_execution.json` records committed and failed counts,
measured outcomes, backend counts, and `full_native: false`. A failed case can
only be replaced by a new generator/suite version, not silently regenerated in
the same immutable suite.

Once generation is complete, run the same finalization/QC sequence and record
duration denominators independently:

```bash
uv run --frozen dynamic-robot-dataset finalize \
  --dataset-root outputs/acceptance/native_acceptance_160
uv run --frozen dynamic-robot-dataset build-splits \
  --dataset-root outputs/acceptance/native_acceptance_160 \
  --config configs/splits/default.yaml
uv run --frozen dynamic-robot-dataset qc \
  --dataset-root outputs/acceptance/native_acceptance_160
uv run --frozen dynamic-robot-dataset stats \
  --dataset-root outputs/acceptance/native_acceptance_160 \
  --output outputs/acceptance/native_acceptance_160/qc/statistics.json
```

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
  --acceptance-report /path/to/passed_full_native_acceptance/.suite_execution.json \
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
  --acceptance-report /path/to/passed_full_native_acceptance/.suite_execution.json \
  --prerequisite-report /path/to/canonical_10h_candidate/qc/readiness/native_10h_v1.json \
  --wan-root /path/to/optional_wan_export \
  --model-evaluation /path/to/external_model_evaluation.json \
  --output /path/to/canonical_100h_candidate/qc/readiness/native_100h_v1.json
```

Only a report at `DATASET/qc/readiness/GATE_ID.json` can serve as a later
prerequisite. Omit `--output` for an exploratory stdout-only evaluation.

The currently defined 128-native/32-quarantine acceptance run always records
`full_native: false`, so its report cannot be substituted for the future
full-native acceptance artifact in either command. Both readiness evaluations
are therefore expected to remain blocked in this revision. Neither gate
reallocates a missing family quota. The 300-hour and 1,000-hour stages have no
launcher in this revision.

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
