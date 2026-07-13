# Migration implementation report

Date: 2026-07-13

## Result

The read-only source-copy phase is complete. It selected and copied the
accessible canonical non-fluid generators, retained meaningful version
differences, and recorded missing, duplicate, scripted, preview, and development
lineages without modifying any source tree. This is not a declaration that
native generator integration or end-to-end migration acceptance is complete.

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

The current refactored family layer implements the common planning, identity,
state/action/physics separation, objective evaluators, failure codes, physics
checks, and exact smoke interface. Its rigid families use named abstract
free-contact solvers (`native_mujoco=false`), while cloth, rope, soft-body, and
legacy cases use explicitly quarantined scripted geometry/response proxies.
These are useful for deterministic schema and pipeline validation, but they are
not native replacements for the copied Franka, Robotiq, MuJoCo-flex, or rope
generators. All diagnostic-renderer/proxy records carry exclusion flags and do
not enter the default training manifest. Native embodiment and renderer
integration remains a production scale-up blocker.

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

The schema, atomic/resumable writer, splits, validators, migration ledgers,
diagnostic smoke suite, and Wan-format round trip are implemented and accepted.
Native Franka/Robotiq MuJoCo rendering and native cloth/rope/soft-body backends
are not bundled. Consequently the repository must not be used to generate a
production Wan training manifest until those native integrations pass the same
gates; the present default manifest correctly contains zero episodes.
