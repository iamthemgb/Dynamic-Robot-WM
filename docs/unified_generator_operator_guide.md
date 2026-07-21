# Unified dynamic-manipulation generator: operator guide

This is the canonical operating procedure for generation in this repository.
It separates three questions that must never be conflated:

1. Is a scenario declared by the corpus taxonomy?
2. Can a backend execute that exact leaf/embodiment/task tuple?
3. Has that leaf passed enough evidence gates to receive review, pilot, or
   production quota?

A declared capability is not a release claim. At this checkpoint 15 rigid
leaf modules have physical fixed-review implementations. Thirteen are
review-executable: P0a-d, F1a-d, F2a, F2c, F2e, F2f, and F3b. F2b and F2d
remain execution-blocked by their Robotiq nominal catches; the remaining five
modules are fail-closed contracts. Every leaf remains release-blocked until
its own evidence gate passes. Large-scale generation is not yet allowed.

## Sources of truth

The runtime taxonomy is
[`configs/corpus/dynamic_manipulation_v2.yaml`](../configs/corpus/dynamic_manipulation_v2.yaml).
It contains exactly 20 leaves and defines each leaf's backend, embodiments,
task variants, rates, evaluator, required metadata, release state, blockers,
and canonical Python module. Do not add a scenario by bypassing this registry
in a script.

All leaf modules are collected under
[`src/dynamic_robot_dataset/scenarios/`](../src/dynamic_robot_dataset/scenarios/).
Their filenames include both the corpus ID and subfamily, for example
`f2b_ramp_launch.py`. Inspect the complete inventory or one contract with:

```bash
python -m dynamic_robot_dataset.scenarios list
python -m dynamic_robot_dataset.scenarios show F2b
```

The modules declare fixture and controller policy and own recipe dispatch;
the backend retains only shared IK, actuator control, rendering, mutation
auditing, and persistence. The older `families/*/adapter.py` files are
analytical diagnostics, not canonical rendered scenario implementations.

Backend support and hash pins are declared in
[`configs/backends/capabilities_v1.yaml`](../configs/backends/capabilities_v1.yaml).
The resolver fails closed for an unknown leaf, backend, embodiment, or task
variant. The supported ownership boundary is:

| Backend | Leaves | Current boundary |
|---|---|---|
| `source_mujoco` | P0a-d, F1a-d, F2a-f, F3a, F3b, F3d | Rigid implementation and review work; no leaf is released by implementation alone |
| `source_genesis_fluid` | F3c | Blocked until the current-and-pickup scene and conservation evidence exist |
| `source_mujoco_deformable` | D1, D2 | Blocked until frictional grasps and deformable replay checks replace equality-connect proxies |
| `native_mujoco` | none | Permanently blocked; synthetic tray/bin/paddle regression code only |

All canonical episodes use `dynamic-robot-dataset/v2`, two synchronized views
(`main`, `secondary`), 832x480 H.264/yuv420p video at 30 Hz, and the same
writer/evaluator/QC interfaces across backends. “Unified” does not mean that
MuJoCo rigid contact, Genesis fluid physics, and MuJoCo deformables have the
same internal solver.

## Read-only dependencies

External generators, robot models, and RoboCasa are hash-pinned inputs. They
must never be used as an output root or edited during a run. The common path
guard rejects writes beneath the protected roots.

The canonical RoboCasa root is:

```text
/gpfs/radev/project/sous/mzl7/robocasa
```

RoboCasa assets are selected through
[`configs/assets/robocasa_catalog_v1.yaml`](../configs/assets/robocasa_catalog_v1.yaml),
not by selecting arbitrary files at runtime. The checked-in catalog has four
content-bound review candidates covering lab, kitchen, workbench, storage, and
tabletop. Runtime strips arbitrary `rc_*` imports, injects exactly the selected
catalog asset, verifies XML/mesh/texture/license hashes, keeps collision off,
and checks its transformed AABB against the task swept volume. Release remains
blocked until rendered occlusion evidence is recorded. A procedural or
missing-asset fallback is not an acceptable substitute.

Set environment variables only to select those read-only inputs:

```bash
export ROBOCASA_ROOT=/gpfs/radev/project/sous/mzl7/robocasa
export ROBOCASA_ASSETS_ROOT=/gpfs/radev/project/sous/mzl7/robocasa/robocasa/models/assets
export MUJOCO_GL=egl
```

Use an output beneath this repository's `outputs/` directory or another path
that is not a protected source tree.

## Install and preflight

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
uv run --frozen dynamic-robot-dataset --help
uv run --frozen dynamic-robot-dataset inspect-embodiments --require-rigid-sources
uv run --frozen pytest -q tests/unit/test_corpus_registry.py
```

Before any run, inspect both registries and the selected leaf's blockers. A
backend in `blocked` state may be described and tested, but it receives zero
execution quota. `review` permits only the fixed acceptance rollouts. `pilot`
permits only its bounded pilot. Only `released` may receive scale quota.

## Canonical run lifecycle

Every multi-worker run follows exactly three state-changing phases. Inputs to
the planning phase must already contain fully resolved, serializable
`SourceScenarioSpec` declarations with canonical lowercase UUIDs and contiguous
episode indices.

### 1. Plan once

```bash
uv run --frozen dynamic-robot-dataset plan-run \
  --config run_inputs/resolved_run.yaml \
  --episodes run_inputs/episodes.yaml \
  --output outputs/example_run \
  --shards 2 \
  --chunk-size 1000 \
  --width 832 --height 480 \
  --fps-num 30 --fps-den 1 \
  --codec libx264 --pixel-format yuv420p \
  --crf 18 --preset medium
```

`plan-run` writes an immutable content-addressed episode/shard ledger. Episode
`i` belongs to shard `i % shard_count`; worker count cannot change identity.
The resume hash includes resolved configuration, video encoding, chunking,
layout, and split settings. Use `--resume` only with byte-equivalent resolved
inputs and identical writer settings.

### 2. Execute deterministic shards

```bash
uv run --frozen dynamic-robot-dataset run-shard \
  --dataset outputs/example_run \
  --shard-id 0

uv run --frozen dynamic-robot-dataset run-shard \
  --dataset outputs/example_run \
  --shard-id 1
```

The command resolves the owned executor from the immutable backend declaration;
do not point it at a collaborator's mutable script. Arbitrary
`module:function` execution is diagnostic-only and requires both `--executor`
and `--unsafe-executor`. Each worker stages privately. An ordinary exception
represents an interruption and may resume the same plan. A measured runtime
physics failure is written as an immutable failed-attempt receipt and is not
retried to obtain a desired label. Only scene-construction failures may
resample, and each attempt seed must be persisted.

### 3. Finalize and seal exactly once

```bash
uv run --frozen dynamic-robot-dataset finalize-run \
  --dataset outputs/example_run \
  --info run_inputs/dataset_info.yaml \
  --tasks run_inputs/tasks.yaml \
  --cameras run_inputs/cameras.yaml \
  --provenance run_inputs/provenance.yaml \
  --counterfactual-families run_inputs/counterfactual_families.yaml
```

Finalization fails if any planned UUID is absent, any unplanned UUID exists, a
failure receipt occupies planned membership, or another finalizer owns the
seal. It hashes every finalized metadata artifact and rejects all later episode
commits. Never repair a sealed run in place; increment the scenario/controller
profile and plan a new run with the same acceptance seeds.

Run strict validation after sealing:

```bash
uv run --frozen dynamic-robot-dataset validate \
  --dataset outputs/example_run --strict-all
uv run --frozen dynamic-robot-dataset qc \
  --dataset outputs/example_run --strict-all
uv run --frozen dynamic-robot-dataset stats --dataset outputs/example_run
```

## Single-worker preview wrapper

`generate` is the bounded single-worker preview surface. It must resolve the
same corpus/capability tuple and materialize through the same scenario,
writer, and evaluator path as `plan-run`/`run-shard`/`finalize-run`; it is not a
second generator implementation.

```bash
uv run --frozen dynamic-robot-dataset generate \
  --backend source_mujoco \
  --review-suite-root outputs/review/unified_acceptance_v1 \
  --review-case F1a-review-00 \
  --output outputs/previews/f1a_seed_7401
```

If the selected backend/leaf is still blocked, the command must stop before
creating episode data. Diagnostic rendering and the permanently blocked
`native_mujoco` implementation never establish review or release evidence.

## Fixed review suite

Create the immutable 20×6 acceptance plan and hash-bound artifact requests:

```bash
uv run --frozen dynamic-robot-dataset review-suite \
  --output outputs/review/unified_acceptance_v1

uv run --frozen dynamic-robot-dataset review-suite \
  --output outputs/review/unified_acceptance_v1 \
  --validate-only

uv run --frozen dynamic-robot-dataset review-suite \
  --output outputs/review/unified_acceptance_v1 \
  --validate-only --execute \
  --dataset-output outputs/review/f1a_fixed_six \
  --leaf-id F1a
```

This command plans 120 logical episodes, 240 canonical MP4s, and synchronized
event strips for both views. It does not make blocked cases executable. The
scene order for every leaf is fixed: clean R0, then RoboCasa lab, kitchen,
workbench, storage, and tabletop. Dual-compatible leaves use three Panda and
three Robotiq cases; P0 leaves use six passive variants; D1/D2 are Panda-only
for v1.

The generated request ledger requires scenario, source-manifest, QC, video,
and event-strip hashes. Review covers continuity, penetration, grasp support,
rebound/deflection plausibility, sticking, tunneling, clipping, fixture
support, occlusion, and RoboCasa intersections. Any automated or human failure
keeps that leaf blocked. Repair the versioned profile and rerun the same seed;
never substitute an easier seed.

Execution uses one immutable shard per selected case and launches the owned
worker in a fresh process, sequentially. This bounds native MuJoCo/renderer
memory without changing episode UUIDs, RNG streams, or plan membership. A
complete six-case leaf publishes a hash-bound pending human-review ledger; it
does not invent decisions or change release state.

Complete the review with an external decision document. The document uses
`dynamic-robot-human-review-decisions/v1`, has one reviewer identity and one
timezone-aware timestamp, and contains exactly one item for every pending
rollout. Every item supplies `leaf_id`, `rollout_index`, the artifact's
`binding_sha256`, all ten boolean checks, and a notes string. Then publish it:

```bash
uv run --frozen dynamic-robot-dataset review-finalize \
  --dataset /gpfs/radev/project/sous/zl664/dataset_generation_runs/review_F2c_fixed6_v18_6bd2d99 \
  --decisions review_inputs/f2c_decisions.json \
  --output-root /gpfs/radev/project/sous/zl664/dataset_reviews
```

`review-finalize` rehashes the seal, finalized metadata, strict QC report,
source scenarios, source manifests, both videos, both event strips, and the
media-pack bindings before accepting a decision. It publishes an immutable
ledger, dataset binding, and per-leaf activation report outside the sealed
dataset. A failed check is preserved as a failed review; it is never converted
to an approval. A second publication for the same seal is treated as a
conflict by the catalog.

For F2b/F2d/F2e/F2f repair work, reproduce the non-rendered fixed-seed verdict
before requesting any media generation:

```bash
MUJOCO_GL=egl uv run --frozen python tools/run_rigid_breadth_fixed_six.py \
  --leaves F2b F2d F2e F2f \
  --output /gpfs/radev/project/sous/zl664/dataset_generation_runs/diagnostics/rigid_breadth_fixed_six_next.json

MUJOCO_GL=egl uv run --frozen python tools/run_rigid_breadth_timestep_halving.py \
  --leaves F2b F2d F2e F2f \
  --output /gpfs/radev/project/sous/zl664/dataset_generation_runs/diagnostics/rigid_breadth_timestep_halving_next.json
```

The report hashes its source files, review plan, cases, measurements, mutation
audit, and independent evaluator replay. It is diagnostic evidence only and
never training data. A leaf that is below 6/6, lacks canonical replay, or has
pending 600/1200 Hz admission remains blocked; `review-suite` media and pilot
commands must not be used to bypass that result. In particular, F2f surface
admission is catalog- and seed-replayed and each admission boolean requires a
SHA-256 evidence artifact.

Regenerate and verify that F2f artifact whenever a bound source file changes:

```bash
MUJOCO_GL=egl uv run --frozen python tools/calibrate_f2f_surface_catalog.py \
  --jobs 4 \
  --candidate plane_mid_285 \
  --candidate plane_table_400 \
  --candidate barrier_yaw_neg_15 \
  --candidate barrier_yaw_neg_25 \
  --output configs/physics/f2f_surface_admission_v1.json

uv run --frozen python tools/calibrate_f2f_surface_catalog.py \
  --verify configs/physics/f2f_surface_admission_v1.json
```

The current honest fixed verdict is F2b 5/6, F2d 5/6, F2e 6/6, and F2f
6/6. F2b and F2d have no admitted rate: their Robotiq nominal catches cannot
simultaneously satisfy retention, visible penetration, and passive-finger
acceleration. F2e and F2f select 1200 Hz. F2f admits four sampled geometries,
but only the two plane candidates currently have genuine retained positive
catches for both embodiments. Its barrier candidates remain review-only until
the Robotiq positive catch is repaired.

## Activation and scale gates

Activation is independent per leaf. The required order is:

1. Six fixed review rollouts pass automated QC, saved-artifact evaluator
   replay, and hash-bound human review.
2. A 100-episode pilot for that leaf has zero hard physics, schema, or
   synchronization failures and passes a two-shard interruption/resume
   rehearsal.
3. Accepted leaves may enter the 10-unique-hour gate.
4. A passed, content-bound 10-hour report is prerequisite evidence for the
   100-unique-hour gate.
5. Hundreds-of-hours generation starts only after the 100-hour gate passes.

Camera streams count once. Endpoint padding and derived Wan clips do not add
hours. A blocked leaf receives zero quota and its quota is not reallocated.
Legacy `configs/release_gates/10h.yaml` and `100h.yaml` still encode the retired
160-case/native acceptance identity; they remain fail-closed historical gate
inputs until their readiness-artifact verifier is migrated to the new 120-case
review ledger. Do not use them to claim this unified suite passed.

## Rigid profile evidence

[`configs/physics/rigid_600_1200_calibration_v1.yaml`](../configs/physics/rigid_600_1200_calibration_v1.yaml)
records the current wall-rebound and Robotiq timestep-halving measurements. The
600 Hz candidate is used only for fixed review classes that satisfy strict QC;
F2e/F2f require the 1200 Hz reference; F2b and F2d have no admitted rate. The
file also binds the honest fixed-six/timestep reports, the 100-probe F2f
surface evidence, and rejected predictive-pad/base diagnostics so numerical
false positives cannot activate a leaf. Neither rate is production-admitted.
Admission still requires sealed rendered rollouts and completed hash-bound
human review.

## Conversion tool boundary

The collaborator tree
`/gpfs/radev/project/sous/zss8/dataset-generation/harmonized/` is a read-only
migration reference, not the runtime generator. `harmonize.py` provides
plan/scaffold/convert/generate/validate/status surfaces, `taxonomy.py` holds its
conversion taxonomy, and `converters/` maps existing datasets into its layout.
Only the v3-ish converter is currently wired, and conversion must run in that
framework's virtual environment. For existing data, start with
`convert --limit 5`, validate, and only then run the full conversion. For new
physics data, use the owned registry, backend, writer, evaluator, and review
commands in this repository; conversion output is not physics-acceptance
evidence.

## Quarantine boundary

Existing F1a state-rewritten/latch-assisted data, analytical renderers,
schematic smoke suites, and custom-attachment MuJoCo outputs belong only under
`legacy_assisted` or diagnostic namespaces. They never count toward review
membership, pilot success, training manifests, or generated hours.
