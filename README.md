# Unified Dynamic-Robotics Dataset Generator

This repository is the non-destructive home for dynamic-object
robotics dataset generation. It keeps persistent physics, transient state, and
actions as separate named modalities; measures outcomes rather than copying
branch intent into labels; and records the provenance and assistance status of
every episode.

The three scratch source trees, the selected collaborator-owned project source
trees listed in `common/paths.py`, `legacy_sources/`, and the separate
`wan_scripts` workspace are write-protected by the common path guard.
Byte-for-byte legacy copies are limited to allowlisted source and small
configuration files in `legacy_sources/`; generated datasets, media, caches,
environments, third-party trees, and fluid code are excluded.

Generator provenance is a real Git commit only after this repository has its
first commit. Before that point it is recorded as `unknown`; no smoke output
from an uncommitted tree should be treated as reproducible or canonical.

## Canonical contract

- Python 3.10, dependencies locked by `uv.lock`.
- MP4 + Parquet layout modelled after LeRobotDataset v3.
- Two synchronized, calibrated camera streams at 832×480, 30 FPS.
- H.264 video with `yuv420p`, event-adaptive episode duration, and explicit
  PTS-derived timestamps.
- SI units, right-handed world/task frames, +Z up, WXYZ quaternions.
- Named state/action fields and explicit action mode.
- Separate action-counterfactual bundles, physics-counterfactual families, and
  leakage-prevention split groups.
- Per-frame assistance masks and explicit `free_contact`, `assisted_contact`,
  `scripted_motion`, or `unverified` publication tiers.
- Only objective-label, free-contact, physics-QC-passing, hard-QC-passing
  episodes enter the default training manifest.

The lightweight family adapters use fixed-duration deterministic trajectories
for tests and smoke fixtures. They are not native production simulators. A
production backend must choose the event-adaptive durations in `plan.md` and
retain pre-event context, the interaction, and terminal evidence without
slowing physical time.

The canonical source format is independent of any video model. The Wan export
is a derived 832×480, 24 FPS, 121-frame view that pads endpoints without
changing physical time.

## Install

```bash
cd /gpfs/radev/project/sous/zl664/dataset_generation
uv sync --extra mujoco --extra deformable --extra video --extra test
uv run dynamic-robot-dataset --help
```

Copy `configs/assets/asset_roots.example.yaml` to the ignored
`configs/assets/asset_roots.local.yaml`, or set `ROBOCASA_ROOT`,
`ROBOTWIN_ROOT`, `ROBOTWIN_2_ROOT`, and `MUJOCO_MENAGERIE_ROOT`. External asset
repositories are referenced, never vendored.

## Typical workflow

```bash
# Produce a raw read-only snapshot. This is not the curated migration decision
# ledger and can be a large traversal when run on an entire scratch root.
uv run dynamic-robot-dataset inventory \
  --output migration/source_inventory.refresh.json

# Preview deterministic plans only.
uv run dynamic-robot-dataset generate \
  --family falling_catch \
  --subfamily centered_vertical_drop \
  --num-bundles 2 \
  --branches success,near_miss,contact_failure,bad_action \
  --views main,secondary \
  --seed 0 \
  --randomization-level R1 \
  --scene-style clean_franka_lab \
  --output outputs/example \
  --dry-run

# Load the same plan from YAML (still read-only).
uv run dynamic-robot-dataset generate \
  --config examples/falling_catch.yaml \
  --dry-run

# A real non-dry run requires an explicitly selected native renderer backend.
# The callable must return videos, calibrated cameras, and provenance fields.
uv run dynamic-robot-dataset generate \
  --config examples/falling_catch.yaml \
  --renderer your_native_backend.module:render_episode

# Then finalize and validate the generated canonical dataset.
uv run dynamic-robot-dataset finalize --dataset-root outputs/example
uv run dynamic-robot-dataset build-splits --dataset-root outputs/example
uv run dynamic-robot-dataset qc --dataset-root outputs/example
uv run dynamic-robot-dataset validate --dataset-root outputs/example

# Produce the model-specific derived export.
uv run dynamic-robot-dataset export-wan \
  --dataset-root outputs/example \
  --output outputs/example_wan
```

Generation refuses to overwrite an episode. `--resume` is accepted only when
the recorded configuration hash matches exactly. Finalization writes staged
files with a marker-last transaction and stores only relative paths. This
repository deliberately does not pretend its analytic family adapters are a
native MuJoCo renderer: non-dry `generate` exits with an actionable error when
no renderer plugin is supplied.

## Exact smoke suite

`configs/families/smoke_120.yaml` defines a diagnostic suite of exactly 120 episode branches: 24
falling/catch, 16 rolling interception, 30 projectile/rebound (including the
three five-point physics sweeps), 16 cloth, 18 rope, 12 soft-body, and four
quarantined scripted-motion legacy proxies. It uses three background styles and
two synchronized views.

```bash
uv run python tools/run_smoke_suite.py \
  --config configs/families/smoke_120.yaml \
  --output outputs/smoke_tests/canonical_smoke_120
```

Its state renderer is intentionally labelled diagnostic and every episode is
excluded from the default training manifest. It tests formats, synchronization,
resume, QC, and export; it is not evidence of native simulator/render quality.
Run it only after the first repository commit so the command, commit, and
environment in its reproduction manifest are meaningful. A SLURM wrapper is
available under `slurm/` for an EGL-capable A40 node.

## Repository map

- `src/dynamic_robot_dataset/common`: schema, atomic writers, timing, cameras,
  QC, splits, provenance, and derived export.
- `src/dynamic_robot_dataset/families`: cleaned family adapters and objective
  evaluators. Lightweight proxy simulators are named honestly and are not
  substitutes for native production physics.
- `legacy_sources`: immutable allowlisted copies with SHA-256 provenance.
- `migration`: source inventory, canonical/excluded decisions, asset catalog,
  source hashes, blockers, and implementation report.
- `configs`: schema, camera, randomization, split, family, and sweep contracts.
- `tools`: stable entry points for inventory, validation, QC, physics checks,
  contact sheets, splits, and exports.
- `tests`: unit contracts plus a real tiny MP4/Parquet/split/QC/Wan integration
  test with an injected metadata-finalization crash and resume.

## Important limitations

The historical module named
`train2500_yaml_scenes` was not present in any accessible source root. The
available RoboCasa kitchen projectile package reproduces the observed
production family definitions, but byte identity with that missing historical
module is not claimed. Native deformable generators retain simulator- and
asset-specific limitations; tasks without robust objective evaluators are
labelled `unverified` and quarantined rather than guessed.

There is currently no bundled native production renderer. The diagnostic smoke
renderer is never a default fallback, and analytic adapter outputs carry
non-production quality flags even if an external renderer visualizes them.
`tools/export_cosmos.py` intentionally refuses to invent an export because the
plan defines no versioned Cosmos video or manifest contract; Wan is the only
model-specific export implemented here.

See `migration/implementation_report.md` for verified source lineage, copied
files, exclusions, smoke results, and remaining scale-up blockers. Exact shell
invocations and accepted artifact paths are in
`migration/reproduction_commands.md`.
