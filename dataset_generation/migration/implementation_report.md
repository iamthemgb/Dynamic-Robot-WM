# Migration implementation report

Date: 2026-07-13

## Result

The read-only source-copy phase is complete. It selected and copied the
accessible canonical non-fluid generators, retained meaningful version
differences, and recorded missing, duplicate, scripted, preview, and development
lineages without modifying any source tree. A subsequent production-oriented
revision added a native rigid test backend and stricter v2 release contracts.
The embodiment correction below supersedes that backend for production. No
real-gripper acceptance suite or production corpus has passed.

- **148** allowlisted files (**1,418,073 bytes**) were copied byte-for-byte:
  59 from mzl7, 29 from zl664, and 60 from zss8.
- All 148 source-after-copy, source-at-copy snapshot, and legacy-copy SHA-256
  values match. No source file changed during migration.
- `source_mapping.yaml` contains one record for every copied file: 123 name a
  maintained adapter/evaluator target, while 25 documentation and launcher
  records deliberately have `new_refactored_path: null` because they are
  provenance-only.
- `asset_catalog.parquet` contains 4,584 external descriptor records and a
  non-null SHA-256 for every record. No external asset repository was copied.
- Generated videos, trajectories, datasets, logs, caches, environments,
  backups, pilots, debug/smoke outputs, third-party trees, and fluids were not
  copied.

## Embodiment correction

The initial maintained `native_mujoco` rigid path was subsequently found to
mount a synthetic tray, bin, or paddle on the Franka flange. It did not use the
Panda hand or Robotiq clamp found in the selected source generators. Its local
acceptance/smoke outputs and logs were removed, its two SLURM launchers were
deleted, and public CLI execution now rejects that backend before writing any
output. The code remains only as a quarantined low-level regression fixture.

The corrected production allowlist is exactly `franka_hand` and
`robotiq_2f85_thick_pad`. Read-only source adapters now identify and hash:

- zss8 Panda-hand centered-drop catch;
- zss8 Panda-hand table-bounce catch;
- zl664 Panda+Robotiq centered-drop and direct-projectile catch; and
- the zl664 Franka cloth preview, which remains non-release because lift/fold
  use equality-connect proxy grasps.

All required source files for those four adapters are currently readable.
The catch sources still hold/rewrite ball state after capture, the Panda bounce
source also rewrites the bounce response, and cloth lift/fold use equality-
connect grasps. All are therefore marked non-release. `source_mujoco` generation remains
blocked until LeRobot-to-v2 normalization and independent objective replay are
implemented. Release gates now require the not-yet-produced
`real_gripper_acceptance_160_v2` suite and reject custom attachment names and
provenance. This section supersedes the historical custom-tool acceptance
claims retained below for auditability.

## Production generator revision

The maintained repository now implements the following code-level changes:

- a typed `ScenarioSpec -> EpisodePlan -> NativeMuJoCoBackend.run()` lifecycle
  for Franka rigid dynamics, with simulation and rendering owned by the same
  backend;
- actuator-driven Franka control, native MuJoCo object contacts, explicit
  initialization-only object state, synchronized main/secondary views, and
  frame/controller/event/object-state tables;
- `dynamic-robot-dataset/v2` writes and v1 read compatibility, including closed
  measured outcomes, versioned failure evidence, objective evaluator
  provenance, frame semantics, assistance mechanisms, camera mapping,
  controller/tool provenance, and calibrated-range provenance;
- symmetric action/physics counterfactual declarations with expected members,
  intervention fields, fixed-field hashes, and one connected leakage group;
- deterministic connected-group 80/10/10 stratification, exact logical/stream/
  derived/padding duration accounting, intended-versus-actual confusion
  reports, and visual/outcome association warnings;
- versioned physics calibration, 10-hour and 100-hour readiness gates, and
  non-submitting pilot plans; and
- objective/tiering helpers for retained cloth/rope tasks, including a guard
  that rejects the historical `RB_6` versus `RB_first` rope equality-target
  mismatch.

The rigid parameter catalog remains `provisional` and `release_eligible: false`.
Loading plausible candidate bounds is not calibration: free-fall, bounce,
sliding-deceleration, rolling-slip, and contact-stability observations plus
explicit release approval are required. The calibrator verifies the observation
schema and catalog hash, hashes the referenced episode and QC source artifacts,
recomputes every oracle from raw trials, checks admitted support, and verifies a
separate reviewer approval bound to that exact evidence. Self-reported pass
flags are not accepted. Solver settings remain separate from measured effective
friction and restitution.

The native five-point regression confirms that this gate is non-vacuous.
Gravity and sliding-friction responses are monotonic. Under the versioned
`native-restitution-solver-map/v1` controlled impact regime, requested values
0.05/0.25/0.50/0.75/0.95 measure approximately
0.103/0.252/0.501/0.753/0.955 from separated pre/post-contact samples and pass
the local target-error check. This is diagnostic solver-response evidence, not
range approval: promotion still requires hash-bound multi-condition trials,
source episode/QC evidence, admitted-support checks, and independent reviewer
approval.

Two immutable suites now have different meanings:

| Suite | Definition | Status and admissible claim |
|---|---|---|
| Diagnostic smoke | `configs/families/smoke_120.yaml`, exactly 120 logical branches | Existing accepted regression artifact; schema/video/QC/Wan pipeline evidence only, never training data |
| Retired custom-tool acceptance | `configs/families/native_acceptance_160.yaml`, exactly 160 branches | Retained as a regression/provenance definition with `execution_allowed: false`; never training or gate evidence |
| Corrected real-gripper acceptance | `real_gripper_acceptance_160_v2` | Required by release gates but intentionally undefined until the source adapters and v2 normalizer pass |

The retired acceptance definition contains 40 falling/catch/retention, 48
rolling/sliding/transition, 40 projectile/rebound/deflection, 12 cloth tiering,
16 rope repair/tiering, and four assisted/scripted negative controls. It uses
three visual styles and two synchronized cameras. It no longer executes through
the public CLI because the 128 rigid cases used the wrong custom attachment.
Its structure may inform the corrected suite, but neither names nor labels can
convert its outputs into real-gripper data.

Native cloth, rope, foam, beanbag, and pouch production models have not been
accepted. Cloth/rope objective code and the rope constraint repair do not prove
native deformable topology, self-contact, tunnelling, strain, settling, or
visibility quality. The retired 160-case definition cannot be described as a
passed suite.
The installed MuJoCo 3.10 build compiles the retained cable rope models but
lacks the `mujoco.elasticity.shell` plug-in required by the legacy shell-cloth
model. A downgraded edge-only grid is explicitly not accepted as equivalent
cloth physics.

The staged pilot configurations are plans, not launchers. Both set
`submit: false`; no 10-hour or 100-hour corpus was generated or submitted by
this revision. All targets mean QC-passed unique logical episode-hours. The
10-hour deformable quota is not reallocated if blocked, and the 100-hour gate
also requires an accepted 10-hour readiness report and an external
model-evaluation artifact. Both gates require an exact
`dynamic-robot-qc-report/v2` and the canonical report for a passed, fully native
160-branch real-gripper acceptance suite. The retired suite cannot satisfy that
condition. See
`docs/production_revision.md` for the complete release boundary.

### Code-level verification

The frozen current test command in `reproduction_commands.md` is the
authoritative verification command. The suite now includes the real-gripper
allowlist, read-only source hashing, public retirement guard, release-gate
end-effector enforcement, and the existing v1/v2 schema, counterfactual,
split, QC, MP4/Parquet, and Wan round-trip contracts.

Earlier no-render measurements over the 128 rigid cases characterize only the
retired custom attachment implementation. They must not be cited as Panda-hand
or Robotiq acceptance evidence, regardless of their physics or kinematic QC
results.

## Canonical decisions

| Unified category | Selected lineage | Status |
|---|---|---|
| Cloth | mzl7 neutral-tabletop and kitchen packages; continuous materials retained as a kitchen mode | Copied; labels and assisted grasps require canonical evaluation/flags |
| Rolling interception | mzl7 `ball_roll_interception_scripts` | Exact production package copied; success-conditioned retry must not survive refactoring |
| Projectile interception | mzl7 `projectile_ball_catch_robocasa_kitchen_scenes_scripts` | Available implementation matches observed production family IDs, defaults, and metadata schema; copied with historical-name and behavioral uncertainty |
| Robotiq catch v1 | zl664 current Robotiq package and MuJoCo support | Copied; `_gpu` output excluded as duplicate trajectory lineage |
| Franka catch v1/v2 | zss8 current Franka catch package | Copied; versions remain explicit because timing, images, trajectories, and outcomes differ |
| Scripted bounce | zss8 current bounce package | Copied only for reproducibility and permanently classified `scripted_motion` |
| Robotiq catch v2 | zss8 Robotiq copy | Copied as a meaningful camera/control/visual variant of v1 |
| Rope | zss8 per-view package | Copied; composite output excluded and equality/snags require explicit treatment |
| Soft body | zl664 preview package | Copied as current accessible source; no output becomes release-eligible without verified metrics |

The compared but unselected projectile forks are documented with component tree
hashes in `source_inventory.json`. Hashes and diffs establish that they are
related forks with divergent scene/variant code. Any predecessor ordering is an
inference from wrapper references, names, and feature accretion because no source
history is available.

## Remaining blockers and uncertainties

1. The historical projectile package named by production Slurm,
   `scripts_mujoco_projectile_catch_robocasa_train2500_yaml_scenes`, does not
   exist at the source root. The selected available package matches the observed
   production interface, defaults, and metadata schema. That similarity does not
   prove equivalent rollout behavior or byte identity to the missing package.
2. mzl7's `scripts_mujoco_nominal_catch` and
   `scripts_mujoco_lat_drift_catch` still return `EACCES`. Accessible zl664/zss8
   catch implementations let the selected migration scope proceed without those
   directories. Their contents were not inspected, and no source-equivalence or
   feature-parity claim is made.
3. The selected source trees are not Git worktrees. Provenance uses file and
   tree SHA-256 values instead of fabricated commits.
4. RobotWin assets are SAPIEN-oriented. They require explicit conversion,
   collision, articulation, and scale validation before MuJoCo use.
5. Physics/label defects listed in `known_issues.md` remain refactoring and QC
   gates; copying a legacy source does not endorse its labels or physics.

## Artifacts

- `source_inventory.json`: evidence-based selected/missing component inventory
  and grouped exclusion status. Its broad excluded-lineage record is not an
  exhaustive per-directory manifest; see `known_issues.md` for the granularity
  boundary.
- `source_mapping.yaml`: exact 148-file migration ledger.
- `source_before_after_hashes.json`: source immutability and copy-integrity proof.
- `known_issues.md`: canonical/excluded table and concrete label/physics issues.
- `migration_plan.md`: fixed lineage, schema, asset, and validation decisions.
- `asset_catalog.parquet`: external asset descriptor inventory.
- `legacy_sources/{mzl7,zl664,zss8}`: immutable byte-identical snapshots.

The maintained generator code must use the new common CLI and schema; legacy
Slurm files are provenance only because they contain stale absolute paths and,
in one case, the missing historical module name.

## Maintained implementation boundary

The family adapters still implement common planning and deterministic
diagnostic solvers. Their analytical rigid trajectories and scripted
cloth/rope/soft-body geometry remain non-production fixtures; rendering one of
those trajectories with an external plug-in does not make it native. All such
records carry exclusion flags and stay out of the default training manifest.

The old `native_mujoco` implementation is a quarantined custom-attachment
regression path. Public generation rejects it. The corrected `source_mujoco`
boundary targets the selected real Panda-hand and Panda+Robotiq source
generators, but canonical execution is blocked until the v2 normalizer and
objective replay exist. Object pose/velocity initialization-only rules,
calibrated parameter support, physics/visual QC, complete counterfactual
declarations, and hard dataset QC still apply after that integration.

Cloth and rope currently have maintained native-objective and tiering
contracts, not an accepted general free-contact deformable backend.
Soft-body generation remains gated, and dual-Franka box folding, shake-wave,
complex bags, knots, fluids, and chaotic scenes remain suspended or deferred.

In `source_mapping.yaml`, `new_refactored_path` names the maintained adapter or
evaluator target and `major_changes` records the required treatment. Neither
field claims that every legacy controller, camera, robot state, or task variant
has already reached native feature parity.

## Canonical smoke acceptance

The committed diagnostic pipeline was exercised end to end from generator
commit `ef1b3c927dce4297441a268ba2cd7e7a97dfb4dc`. Its environment fingerprint
is `d68ecfb3adbd54470f07f9c8d9aed10678cbd10ca5ec433c732ef45a8a0e159f` and
includes installed Python distributions plus the `uv.lock` digest. The
corresponding shell invocation is recorded in `reproduction_commands.md`; the
finalized provenance Parquet records its script argv and resolved identities.
The counts below describe that preserved historical diagnostic artifact. The
production revision changes the default splitter to connected-group 80/10/10;
it does not rewrite the artifact's existing 108/6/6 split table.

| Family | Branches | Frames/episode | Duration (s) | Objective outcomes | Physics-QC pass |
|---|---:|---:|---:|---|---:|
| Falling catch | 24 | 97 | 3.2333 | 6 success, 18 measured failures | 24 |
| Rolling interception | 16 | 136 | 4.5333 | 5 success, 11 measured failures | 16 |
| Projectile/rebound and sweeps | 30 | 115 | 3.8333 | 14 success, 16 measured failures | 30 |
| Cloth | 16 | 151 | 5.0333 | 4 success, 12 measured failures | 0 (proxy) |
| Rope | 18 | 151 | 5.0333 | 9 success, 9 measured failures | 0 (proxy) |
| Soft body | 12 | 136 | 4.5333 | 12 proxy outcomes, unverified for release | 0 (proxy) |
| Legacy proxy quarantine | 4 | 76 | 2.5333 | 4 quarantined | 0 (proxy) |

Acceptance facts:

- Exactly 120 unique episodes and 240 synchronized source videos were written.
  Every source video independently decoded as H.264/yuv420p, 832×480, 30 FPS,
  with uniform PTS and the declared 76–151 frame count.
- Source duration is 500.8 episode-seconds, or 1,001.6 seconds counting both
  views. The source tree occupies 14,035,437 bytes.
- The derived Wan export contains 120 manifest rows and 240 H.264/yuv420p
  videos at 832×480, 24 FPS, exactly 121 frames/5.0417 seconds each. That is
  605 episode-seconds or 1,210 view-seconds; the export occupies 5,928,633
  bytes. All 241 export checksums and referenced source hashes match.
- The three five-point gravity, friction, and effective-restitution sweeps are
  configurations and smoke episodes only. No production 10,000-video sweep was
  generated.
- Split counts are exactly 108 train, 6 validation, and 6 test. No action
  bundle, physics family, scene seed, initial-state identity, trajectory
  identity, parent lineage, or exact-media duplicate group crosses splits.
- Deep dataset QC and the independent physics validator report zero hard or
  global failures. The full frozen test suite reports 34 passing tests.
- Publication tiers contain 70 free-contact, 48 scripted-motion, and 2
  assisted-contact records. The overlapping unverified manifest contains 50.
  All 120 records carry `smoke_diagnostic_renderer`, all are quarantined, and
  `default_training.jsonl` is intentionally empty.

Duplicate validation found 18 exact nonrelease groups and 8,336 perceptual
similarity candidates. All exact groups are now confined to one split; exact
groups crossing splits are a hard validator failure. These duplicates and the
8,354 resulting global warnings are expected evidence of the schematic smoke
renderer, not production diversity. Per-episode QC also retains 650 warnings,
primarily because generic visual visibility/occlusion and independent label
agreement cannot be established from schematic frames.

## Visual review

All seven family contact sheets were inspected. The exact UUIDs, outcomes,
camera names, frame indices, and relative MP4 paths are retained in
`outputs/smoke_tests/canonical_smoke_120/qc/contact_sheets/samples.json`.

- Both cameras are synchronized, in frame, and complete from initial context
  through the terminal state. No sampled video is corrupt, truncated, frozen,
  or missing its second view.
- The rigid diagnostic trajectories visibly distinguish success, miss,
  near-miss, contact failure, and bad-action cases, and the rebound sweep shows
  changing motion. This is useful for checking labels and timing only.
- The renderer is deliberately schematic: it does not show a robot body,
  articulated gripper, realistic contact geometry, occlusion, materials, or
  penetration detail. The three “background styles” are palette/table changes,
  not production scene diversity.
- Cloth and rope proxies have low pixel footprint and repeated no-action
  trajectories. Soft-body deformation is represented by an exaggerated glyph
  radius. These are format/QC fixtures, not qualitative evidence for training.
- Style coverage is confounded by family: falling catch is clean-lab only and
  rolling is RoboCasa-style only; projectile outcomes are not style-balanced.
  Production jobs must balance style independently within family and outcome.

The reviewed contact sheets therefore support pipeline acceptance but not
native simulator, physics, rendering, or Wan-training acceptance.

## Immutability recheck

A post-commit read-only audit verified all 148 allowlisted original files and
all 148 `legacy_sources` copies against the ledgers, with zero content, size,
ownership, or Git differences. All 17 recorded source component trees matched,
covering 193 inventoried files. Ten literal mtime comparisons differed by one
microsecond because nanosecond timestamps were serialized through floating
point; their hashes and other stat evidence match.

The separate Wan workspace has no before-hash ledger and was changing during
this audit because pre-existing Slurm job `2108566` (`wan_setup`) was compiling
FlashAttention. No dataset-generation path appeared there, and runtime write
guards rejected all source, `legacy_sources`, and Wan-workspace probes. This
report therefore claims no writes by this implementation, not an unverifiable
whole-workspace before/after hash equality.

## Acceptance boundary

The migration ledgers and preserved diagnostic smoke/Wan round trip are
accepted as pipeline evidence. The v2 schema, atomic/resumable writer,
counterfactual declarations, connected splits, statistics, calibration and
readiness evaluators are implemented code. Real-gripper source inspection is
implemented; real-gripper canonical generation is not.
They have not converted the historical diagnostic artifact into training data,
and they do not establish production-corpus acceptance.

The repository must not be used to advertise a production Wan training corpus
until real-gripper v2 integration, native physics calibration, the corrected
160-case suite, objective and visual QC, counterfactual/leakage checks, and the
relevant stage gate all pass.
Native deformable/soft-object acceptance is still missing. The preserved
diagnostic default manifest correctly contains zero episodes, and no pilot or
training job was launched by this revision.
