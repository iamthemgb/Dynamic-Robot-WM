# Unified Dynamic-Robotics Dataset Generator

This repository is the canonical orchestration, schema, writer, evaluator, and
QC layer for the 20-leaf dynamic-manipulation corpus. It uses the physics
backend appropriate to each leaf while emitting one
`dynamic-robot-dataset/v2` contract.

The repository is fail-closed. P0a-d and F1a-d currently have executable
review implementations; the other 12 leaves have zero execution quota. All 20
remain release-blocked pending their fixed acceptance evidence, so executable
review code is not a claim that large-scale generation is ready.

For the exact operating procedure, start with the
[unified generator operator guide](docs/unified_generator_operator_guide.md).

## Canonical boundary

Two checked-in registries are the runtime sources of truth:

- [`configs/corpus/dynamic_manipulation_v2.yaml`](configs/corpus/dynamic_manipulation_v2.yaml)
  defines exactly P0a-d, F1a-d, F2a-f, F3a-d, D1, and D2, including each
  leaf's backend, embodiments, variants, rates, evaluator, metadata, release
  state, and blockers.
- [`configs/backends/capabilities_v1.yaml`](configs/backends/capabilities_v1.yaml)
  declares supported leaf/embodiment tuples and content hashes for read-only
  source dependencies.

| Backend | Corpus leaves | Status |
|---|---|---|
| `source_mujoco` | P0a-d, F1a-d, F2a-f, F3a/F3b/F3d | Rigid review implementation; blocked until per-leaf evidence passes |
| `source_genesis_fluid` | F3c | Blocked; current-and-pickup fluid scene is unfinished |
| `source_mujoco_deformable` | D1/D2 | Blocked; frictional grasp and replay acceptance are unfinished |
| `native_mujoco` | none | Permanently blocked custom tray/bin/paddle regression backend |

“Unified” means one registry, scenario contract, writer, evaluator/QC
interface, and orchestration layer. It does not claim that rigid, fluid, and
deformable simulators use identical internal physics.

## Dataset contract

Canonical episodes require:

- exactly `round(duration × 30)` synchronized frames from one persisted
  rollout at timestamps `k/30`, with simulator timestamps and synchronization
  error recorded;
- `main` and `secondary` H.264/yuv420p views at 832x480 and 30 Hz;
- actual actuator commands as actions: seven arm controls plus Panda finger or
  Robotiq tendon control;
- saved state/contact/event evidence and independent objective replay;
- source, XML, mesh, texture, configuration, video, table, and finalized
  metadata hashes;
- strict hard-physics checks, including penetration, energy, restitution,
  tunneling, mutation-boundary, and task-specific contact evidence;
- outcome mismatches preserved as measured attempts, never retried to obtain an
  intended label.

Legacy state-rewritten, welded, latched, scripted, diagnostic, or custom-tool
episodes remain in `legacy_assisted`/diagnostic quarantine. They never count as
training data or generated hours.

## Install

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
uv run --frozen dynamic-robot-dataset --help
```

The canonical RoboCasa dependency is read-only:

```bash
export ROBOCASA_ROOT=/gpfs/radev/project/sous/mzl7/robocasa
export ROBOCASA_ASSETS_ROOT=/gpfs/radev/project/sous/mzl7/robocasa/robocasa/models/assets
export MUJOCO_GL=egl
```

External generator, RoboCasa, robot-model, scratch-data, and Wan-training trees
are protected inputs. Never choose one as a dataset output root.

The catalog contains four content-bound, visual-only candidates covering the
five R1 profiles. Runtime strips arbitrary external `rc_*` selections and
injects exactly the catalog candidate for the requested profile. Candidates
remain release-blocked until rendered occlusion review is recorded.

## Immutable orchestration

Large or multi-worker runs use one three-phase lifecycle:

```bash
uv run --frozen dynamic-robot-dataset plan-run \
  --config run_inputs/resolved_run.yaml \
  --episodes run_inputs/episodes.yaml \
  --output outputs/example_run \
  --shards 2

uv run --frozen dynamic-robot-dataset run-shard \
  --dataset outputs/example_run \
  --shard-id 0

uv run --frozen dynamic-robot-dataset run-shard \
  --dataset outputs/example_run \
  --shard-id 1

uv run --frozen dynamic-robot-dataset finalize-run \
  --dataset outputs/example_run \
  --info run_inputs/dataset_info.yaml \
  --tasks run_inputs/tasks.yaml \
  --cameras run_inputs/cameras.yaml \
  --provenance run_inputs/provenance.yaml \
  --counterfactual-families run_inputs/counterfactual_families.yaml
```

`plan-run` binds exact membership, deterministic shard assignment, encoding,
chunking, layout, and split settings. Workers stage privately. `finalize-run`
requires exact plan membership, acquires an exclusive seal, hashes every
metadata artifact, and rejects post-seal commits.

`run-shard` resolves the owned backend from the immutable plan. Supplying a
`module:function` executor is diagnostic-only and requires the explicit
`--unsafe-executor` opt-in.

`generate` is reserved for bounded single-worker previews over the same owned
backend path. It must fail before materialization when the selected capability
or leaf is blocked; diagnostic rendering is not production evidence.

## Fixed review and release sequence

Create or validate the immutable acceptance plan with:

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

The plan is exactly 20 leaves × 6 fixed seeds: 120 logical episodes, 240
canonical videos, and two-view event strips. The scene sequence is clean R0,
then real RoboCasa lab, kitchen, workbench, storage, and tabletop. Planning does
not give blocked leaves execution quota.

For a quick single-case preview, select the same immutable review case through
the bounded wrapper:

```bash
uv run --frozen dynamic-robot-dataset generate \
  --backend source_mujoco \
  --review-suite-root outputs/review/unified_acceptance_v1 \
  --review-case F1a-review-00 \
  --output outputs/previews/f1a_r0
```

### F3b RoboCasa kitchen previews

F3b borrows only the content-pinned scene, YAML, camera, and visual-asset
helpers from Michael's read-only `ball_roll_interception_scripts` tree. Its
`controller.py` is neither pinned nor imported; robot motion remains the owned
actuator-only, free-contact controller in this repository.

For R1, the independent asset seed selects one of the 50 RoboCasa training
styles. The selected layout stays fixed, while official countertop, cabinet,
floor, and wall textures vary. Only sized sink, dishwasher, and refrigerator
models outside the task island are admitted as collision-disabled background
meshes. Zero-size placeholders, island appliances, and the occluding stove
models fail closed.

Generate the fixed six-case F3b review with:

```bash
export MUJOCO_GL=egl
uv run --frozen dynamic-robot-dataset review-suite \
  --output outputs/review/unified_acceptance_v1 \
  --validate-only --execute \
  --dataset-output outputs/review/f3b_robocasa_fixed_six \
  --leaf-id F3b
```

The resulting videos are written to
`videos/observation.images.main/chunk-000/` and
`videos/observation.images.secondary/chunk-000/`. Episode 0 is the fixed clean
R0 reference; episodes 1--5 are the seeded RoboCasa kitchen appearances.

Complete six-case leaves produce a hash-bound
`reviews/human_review_ledger.pending.json`; the ledger contains no fabricated
human decisions and cannot activate a leaf until reviewers fill and validate
it.

Each leaf activates independently only after:

1. all six fixed rollouts pass automated QC, saved-artifact objective replay,
   and hash-bound human review;
2. its 100-episode pilot has zero hard failures and passes a two-shard
   interruption/resume rehearsal;
3. the existing 10-unique-hour gate passes; and
4. the subsequent 100-unique-hour gate passes before hundreds-of-hours scale.

Camera streams count once. Blocked leaves get zero quota, and quota is not
reallocated. The old 10h/100h YAMLs still name the retired 160-case/native
acceptance suite and therefore remain historical fail-closed inputs until the
readiness verifier is migrated to the new hash-bound 120-case ledger.

## Physics and randomization status

[`configs/randomization/default.yaml`](configs/randomization/default.yaml)
defines independent RNG streams. R0 fixes physics and appearance; R1 varies
admitted RoboCasa appearance; R2 adds validated object, camera, robot-start,
latency, and controller variation only after R1 passes. Counterfactual siblings
retain their fixed streams.

[`configs/physics/rigid_600_1200_calibration_v1.yaml`](configs/physics/rigid_600_1200_calibration_v1.yaml)
records exploratory wall-rebound and Robotiq 600/1200 Hz measurements. Both
profiles remain explicitly unadmitted until the full timestep-halving matrix,
hash-bound artifacts, fixed reviews, and human approval exist.

## Collaborator harmonizer boundary

`/gpfs/radev/project/sous/zss8/dataset-generation/harmonized/` is a read-only
migration/conversion reference. Its `harmonize.py`, `taxonomy.py`, and
`converters/` scaffold or convert existing data into a canonical layout; they
do not provide free-contact physics generation or acceptance evidence. Only
the v3-ish converter adapter is currently wired, and conversion must use that
framework's own virtual environment. Use a five-episode `convert --limit 5`
smoke test before a full existing-dataset conversion, but use this repository's
owned review/generation path for new simulated episodes.

## Repository map

- `src/dynamic_robot_dataset/common`: contracts, registries, randomization,
  writer, orchestration, review, physics QC, splits, and provenance.
- `src/dynamic_robot_dataset/backends`: owned backend controllers plus the
  permanently blocked regression backend.
- `configs`: corpus/backend registries, media, assets, physics, randomization,
  review, pilots, and release gates.
- `docs`: canonical operations plus historical design rationale.
- `migration` and `legacy_sources`: source inventory and immutable quarantine
  provenance; never a production execution path.
- `tests`: unit, failure-injection, concurrency, and rendered integration
  contracts.

The older [production revision](docs/production_revision.md) and migration
reports are retained for historical rationale. Where their retired
`native_mujoco`/160-case terminology conflicts with the registries and operator
guide above, the registries and operator guide are authoritative.
