# Unified Dynamic-Robotics Dataset Generator

This repository is the non-destructive, Franka-first generator for
dynamics-centered robotics data. It separates persistent physics, transient
state, and actions; derives outcomes from saved evidence instead of branch
intent; and keeps synchronized camera views as observations of one physical
rollout rather than counting them as separate experiences.

The production revision adds a typed native-MuJoCo lifecycle, the
`dynamic-robot-dataset/v2` contract, deterministic connected-group 80/10/10
splits, calibrated release gates, exact duration accounting, and staged 10-hour
and 100-hour pilot plans. It does **not** launch either pilot or declare the
160-case acceptance suite passed.

## Current boundary

| Component | Status |
|---|---|
| Native Franka rigid backend | Implemented for falling/catch/retention, rolling/sliding/transitions, and projectile/rebound/deflection; still subject to calibration and acceptance gates |
| v2 schema, outcomes, counterfactual declarations, QC, statistics, splits, and Wan export | Implemented with v1 read compatibility |
| Diagnostic 120-branch suite | Preserved as an accepted pipeline regression fixture; never training data |
| Native acceptance 160-branch suite | Immutable definition implemented; execution routes 128 rigid cases through `native_mujoco` and 32 deformable/negative controls through `diagnostic_quarantine`, so it cannot pass full native coverage yet |
| Cloth and rope | Objective/tiering contracts and rope target repair implemented; native deformable production validation remains open |
| Foam ball, beanbag, pouch | Gated behind rigid-contact acceptance; beanbag/pouch also need validated shell and self-contact models |
| Production-scale generation and Wan training | Out of scope; no jobs are submitted or modified |

The three scratch source trees, selected collaborator-owned project source
trees listed in `common/paths.py`, `legacy_sources/`, and the separate
`wan_scripts` workspace are protected by the common path guard. Legacy copying
is limited to the allowlisted small source/configuration snapshot already
recorded under `migration/`. Generated datasets, media, environments,
third-party repositories, fluids, checkpoints, and active jobs are not touched.

See [the production revision](docs/production_revision.md) for the exact schema,
scenario, suite, duration, and release contracts. Source-lineage evidence and
remaining migration uncertainty are in
[the implementation report](migration/implementation_report.md).

## Canonical contract

- Python 3.10 with dependencies locked by `uv.lock`.
- Atomic, resumable MP4 + Parquet episodes; resume requires the same resolved
  configuration hash and never overwrites an episode.
- Two synchronized calibrated H.264/`yuv420p` streams at 832x480 and 30 FPS.
- Event-adaptive duration with PTS-derived timestamps; physical time is never
  slowed to fill a model window.
- SI units, right-handed world/task frames, +Z up, WXYZ quaternions.
- Closed measured outcomes, concrete failure evidence, objective evaluator
  identity, and independent recomputation from persisted tables.
- Separate action-counterfactual bundles and physics-counterfactual families,
  joined by one leakage-prevention split group.
- Per-frame phase, motion mode, surface, contact role, transition events, and
  simulator-observed assistance intervals.
- Explicit `free_contact`, `assisted_contact`, `scripted_motion`, `unverified`,
  and quarantine publication boundaries.
- Default training export includes only objective-verified, calibrated,
  free-contact, physics-QC-passing, hard-QC-passing v2 episodes.

The Wan export remains derived and model-specific: 832x480, 24 FPS, 121 frames,
one manifest row per logical episode. `video` names the main view and the
secondary synchronized view remains a sidecar in the same row. Endpoint frames
may repeat when source context is shorter than the fixed window, and that
padding is reported separately.

## Install

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv sync --frozen --extra mujoco --extra deformable --extra video --extra test
uv run --frozen dynamic-robot-dataset --help
```

Set `MUJOCO_MENAGERIE_ROOT` (or `FRANKA_MJCF_PATH`) for the native Franka
backend. Optional RoboCasa/RobotWin roots are referenced through environment
variables and are never vendored. An asset does not become production-admitted
merely because it loads: scale, collision, rendering, content hash, and license
notice checks must pass.

## Safe workflow

Inventory and dry-run operations are read-only:

```bash
uv run --frozen dynamic-robot-dataset inventory \
  --output migration/source_inventory.refresh.json

uv run --frozen dynamic-robot-dataset generate \
  --family falling_catch \
  --subfamily centered_vertical_drop \
  --num-bundles 2 \
  --branches success_seeking,near_miss,contact_failure,no_op \
  --views main,secondary \
  --seed 0 \
  --randomization-level R1 \
  --scene-style clean_franka_lab \
  --dry-run

uv run --frozen dynamic-robot-dataset generate-suite \
  --config configs/families/native_acceptance_160.yaml \
  --dry-run
```

Native execution must be selected explicitly. It owns both simulation and
rendering; the legacy renderer hook is not a production-physics substitute:

```bash
export MUJOCO_GL=egl
uv run --frozen dynamic-robot-dataset generate \
  --backend native_mujoco \
  --family falling_catch \
  --subfamily catch_retain \
  --num-bundles 1 \
  --branches success_seeking,near_miss,contact_failure,no_op \
  --output outputs/native_catch_retain
```

Canonical datasets are then finalized, split, validated, measured, and
optionally exported:

```bash
uv run --frozen dynamic-robot-dataset finalize \
  --dataset-root outputs/native_catch_retain
uv run --frozen dynamic-robot-dataset build-splits \
  --dataset-root outputs/native_catch_retain \
  --config configs/splits/default.yaml
uv run --frozen dynamic-robot-dataset qc \
  --dataset-root outputs/native_catch_retain
uv run --frozen dynamic-robot-dataset stats \
  --dataset-root outputs/native_catch_retain
uv run --frozen dynamic-robot-dataset export-wan \
  --dataset-root outputs/native_catch_retain \
  --output outputs/native_catch_retain_wan
```

Exact verification, calibration, gate, and diagnostic commands are in
[`migration/reproduction_commands.md`](migration/reproduction_commands.md).

## Two suites, two meanings

`configs/families/smoke_120.yaml` is the preserved diagnostic suite: 24
falling/catch, 16 rolling, 30 projectile/rebound and sweeps, 16 cloth, 18 rope,
12 soft-body, and four quarantined proxy branches. Its state renderer is
schematic. All 120 logical episodes are explicitly excluded from training.

```bash
uv run --frozen python tools/run_smoke_suite.py \
  --config configs/families/smoke_120.yaml \
  --output outputs/smoke_tests/reproduced_smoke_120
```

`configs/families/native_acceptance_160.yaml` is separate: 40 falling/catch,
48 rolling/sliding, 40 projectile/rebound, 12 cloth tiering, 16 rope repair,
and four negative controls. It covers three styles and both cameras. Its full
executor sends the 128 rigid cases to `native_mujoco` and the remaining 32
cloth/rope/negative-control cases to an explicit `diagnostic_quarantine` path.
That preserves exact coverage without pretending deformable proxies are native,
and the suite report records `full_native: false`; stage readiness must remain
failed until real native deformable backends exist. A dry run proves only
deterministic suite expansion; no 160-case artifact was generated by this
revision. The execution gate requires every planned rigid outcome class—not
only a generic success/failure pair—to appear in independently measured labels.
The checked-out implementation also completed a local no-render audit of all
128 rigid branches. All required measured outcome classes were present, while
124 branches passed native physics QC and four were correctly quarantined for
exceeding the declared Franka joint-velocity and/or joint-acceleration gates
(acceptance cases 54, 58, 73, and 82). These measured failures are not waived
or relabeled. This does not substitute for the missing rendered/finalized
160-case artifact or native deformable coverage.

`slurm/native_rigid_smoke.sbatch` and
`slurm/native_acceptance_160.sbatch` are explicit wrappers only. Creating them
does not submit a job; each requires a caller-provided new output directory and
an explicit `sbatch` invocation.

## Calibration, gates, and hours

`configs/physics/rigid_ranges_v1.yaml` is provisional and release-ineligible.
`calibrate-physics` requires native free-fall, bounce, slide, roll/slip, and
contact-stability evidence before admitting support. MuJoCo solver values are
not labelled as measured physical restitution or friction. Calibration accepts
only the versioned observation schema with hash-verified episode/QC sources,
recomputable raw oracle trials, admitted support inside the candidate ranges,
and a separately hash-bound reviewer approval; an asserted `passed` field is
not evidence.

The 10-hour and 100-hour YAML files under `configs/pilots/` are non-submitting
plans (`submit: false`). Readiness is evaluated from QC-passed unique logical
episode duration. Camera streams, Wan clips, and endpoint padding have separate
denominators and never inflate a scaling target. The 10-hour quota is not
reallocated if no deformable family is ready; the 100-hour gate additionally
requires an external model-evaluation artifact. Both gates require a canonical
`dynamic-robot-qc-report/v2` bound to the exact episode table and an explicit
`--acceptance-report` for a passed, 160-branch, fully native suite. The current
128-native/32-quarantine definition deliberately fails that full-native gate;
it cannot unlock either pilot. QC/readiness also rehash finalized metadata and
episode media/tables, and prerequisite reports are accepted only at
`DATASET/qc/readiness/GATE_ID.json`. Model-evaluation claims require real,
hash-matching model and evaluation-manifest paths. No pilot has been launched.

## Repository map

- `src/dynamic_robot_dataset/backends`: typed lifecycle and integrated native
  MuJoCo backend.
- `src/dynamic_robot_dataset/common`: v2 schema, atomic writer, calibration,
  objectives, statistics, QC, split, provenance, and Wan export.
- `src/dynamic_robot_dataset/families`: planning adapters, diagnostic fixtures,
  and task-specific objective/tiering logic.
- `configs`: v2 schema, cameras, randomization, physics, suites, pilots, and
  release gates.
- `docs`: production architecture and release policy.
- `migration`: source inventory, allowlisted copy provenance, source hashes,
  known issues, implementation status, and exact commands.
- `legacy_sources`: immutable byte-identical allowlisted source snapshots.
- `tests`: schema, family, backend, writer, QC, split, and round-trip contracts.

## Important limitations

The missing historical projectile module
`scripts_mujoco_projectile_catch_robocasa_train2500_yaml_scenes` has not been
reconstructed speculatively. The available kitchen projectile package matches
the observed production interface and metadata, not proven historical bytes or
rollout behavior. Two mzl7 catch directories also remain inaccessible.

Native rigid code is not synonymous with calibrated data. Until calibration,
the native suite, and release gates pass, no output should be described as a
production Wan training corpus. Native cloth/rope/soft-object simulation remains
an explicit blocker. Fluids, complex knotting/bags, chaotic multi-object scenes,
uncontrolled multi-bounce motion, and production-scale jobs remain outside this
revision.
