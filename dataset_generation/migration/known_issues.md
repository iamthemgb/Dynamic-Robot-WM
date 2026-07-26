# Migration findings and known issues

Inventory date: 2026-07-13. All source inspection and copying was read-only. The
148 files under `legacy_sources/` are byte-identical snapshots; see
`source_before_after_hashes.json` for the per-file proof.

## Canonical and excluded lineages

| Owner | Source or output lineage | Decision | Evidence and caveat |
|---|---|---|---|
| mzl7 | `mujoco_franka_cloth_previews` | Keep: neutral-tabletop cloth source | Production wrapper names this module. The 6,000-episode output is physically and visually distinct from kitchen cloth despite `previews` in its name. Labels are unverified and grasping is assisted. |
| mzl7 | `mujoco_franka_cloth_kitchen_scene_previews` | Keep: kitchen preset and continuous cloth source | Both production wrappers invoke this module; continuous materials are a CLI mode, not a separate generator. |
| mzl7 | cloth pilot directories and 1920x480 mosaics | Exclude | Pilots duplicate canonical records. Mosaics combine three primary views and are derived media. |
| mzl7 | `ball_roll_interception_scripts` | Keep: canonical rolling-island source | Production Slurm invokes the package directly for the three 1,500-episode families. Its dataset entry point imports a demo-named module for calibration constants and a camera hook, so that file is an effective production dependency and was retained. |
| mzl7 | `projectile_ball_catch_robocasa_kitchen_scenes_scripts` | Keep: selected available projectile implementation | It defines four 750-episode family configurations, the branch schedule, YAML/native scene loader, 832x480/30 FPS/2.5 s defaults, two views, and metadata keys that match the observed production output. This supports interface/default/schema similarity, not equivalence to the missing historical implementation. |
| mzl7 | `scripts_mujoco_projectile_catch_robocasa_train2500_yaml_scenes` | Missing historical source | The production Slurm wrappers invoke this exact module name, but the directory is absent. The available implementation above is not claimed to be a byte-identical rename. |
| mzl7 | base/cannon/randomized/Robocasa/train2500 projectile clones | Exclude as canonical generators | These are related scene/variant experiments. Most lack `dataset_generation.py`; hashes show shared modules plus divergent scene/variant code. Their likely ordering is inferred rather than proven by repository history. The train2500 package enumerates scene IDs but is not the historical YAML production package. |
| mzl7 | `scripts_mujoco_nominal_catch`, `scripts_mujoco_lat_drift_catch` | Not copied; non-blocking for selected scope | Both still return `EACCES`. Independently inspected Franka and Robotiq implementations cover the selected catch scope, but the inaccessible contents are unknown and no equivalence claim is made. |
| zl664 | current `robotiq_catch` plus `scripts_mujoco` | Keep: canonical thick-pad v1 source | Its 3,000-episode output is the complete 832x480/2.5 s lineage. Source lives in an unversioned project tree, not the scratch output tree. |
| zl664 | `f1_ab_robotiq_thick_pad_v1_gpu` | Exclude duplicate output | All 3,000 Parquet trajectories are bit-identical to v1, root summaries match, and sampled videos are visual rerenders of the same episodes. |
| zl664 | failed shards, aesthetic examples, verification patch outputs | Exclude | Incomplete, superseded, or explicit validation data. |
| zl664 | `mujoco_franka_softbody_previews` | Keep source, mark unverified | This is the accessible current non-fluid flex source, but no canonical production corpus or calibrated objective evaluator was found. |
| zss8 | current `franka_catch` | Keep: Franka vertical-drop source | The v1/v2 outputs differ in resolution, duration, trajectories, and some outcomes; keep them as named versions, not duplicate hours. |
| zss8 | current `franka_bounce` | Keep only as quarantined legacy source | Source explicitly makes the visible surface non-colliding and changes ball pose/velocity analytically at impact. It is `scripted_motion`, never free-contact physics supervision. |
| zss8 | current Robotiq copy | Keep: meaningful v2 variant | Relative to zl664 v1 it changes defaults to 512x512/4 s, adds min-jerk reach/retrieve motion, closer elevated cameras, and broader visual jitter. It shares the task lineage but is not a byte duplicate. |
| zss8 | current `franka_rope` per-view generator | Keep | This generates the canonical front/top/wrist lineage. |
| zss8 | older composite rope output | Exclude duplicate representation | Episode IDs and sampled NPZ/XML hashes match per-view; only the encoded presentation differs. |
| zss8 | `.backups`, debug helpers, previews, smoke data, contact sheets | Exclude | Superseded source snapshots or derived/development outputs. |
| all | fluid generators and media | Exclude | Fluids are explicitly outside this repository's scope. |

### Exclusion inventory granularity

`source_inventory.json` records selected, missing, and inaccessible generator
components individually, but its final `excluded.generated_and_source_lineages`
record deliberately groups broad policy exclusions. It is therefore not an
exhaustive recursive manifest of every excluded directory or output:

| Grouped exclusion | Evidence boundary |
|---|---|
| zl664 `_gpu`, failed shards, and patch/aesthetic outputs | Dataset-level duplicate or supersession finding; no separate component record for every output directory |
| zss8 composite rope and development artifacts | Composite conclusion uses episode IDs and sampled NPZ/XML comparisons; it is not a byte-identity claim for every encoded file |
| mzl7 cloth pilots, mosaics, previews, and derived media | Grouped by representation/development policy rather than enumerated file by file |
| Fluids, caches, environments, third-party trees, backups, logs, and smoke/debug outputs | Scope or hygiene exclusions; neither copied nor exhaustively inventoried |

Exact per-file lineage is asserted only for the 148 copied files in
`source_mapping.yaml` and `source_before_after_hashes.json`.

### Projectile candidate comparison

The accessible projectile directories are related forks, not interchangeable
names for one package. Static file hashes and code diffs establish relatedness
and feature differences, but not chronology. The ordering below is an inference
from directory names, production-wrapper references, and apparent feature
accretion because the source trees have no Git history:

| Candidate | Distinguishing implementation | Production decision |
|---|---|---|
| `scripts_mujoco_projectile_catch` | Baseline six-scene projectile/catch helpers; no dataset-family entry point | Inferred early related fork |
| `..._cannon_clone` | Changes `scene_builder.py` and `variants.py` to add a cannon model | Visual experiment |
| `..._cannon_randomized_zoom_clone` | Adds sampled visual settings, camera/zoom variation and corresponding metadata | Visual experiment |
| `..._randomized_kitchen_clone` | Adds procedural kitchen visual randomization without the canonical YAML/native scene pipeline | Visual experiment |
| `..._robocasa_env_clone` | Randomly selects RoboCasa train layout/style IDs and adds fixture-aware scene assets | Inferred earlier environment experiment |
| `..._robocasa_train2500_envs` | Deterministically enumerates 50 train layouts x 50 train styles and resolves a limited fixture set; still lacks `dataset_generation.py` | Likely related predecessor; not the invoked historical package |
| `projectile_ball_catch_robocasa_kitchen_scenes_scripts` | Adds the complete YAML/native resolver, fixed four-family dataset entry point, branch schedule, two-view renderer, and production metadata schema | Selected complete available implementation; historical behavior equivalence unproven |

Shared modules such as `assets.py` and `glb_exporter.py` are byte-identical
across most forks, while scene, variant, metadata, and entry-point code diverges.
This is why only the complete available package matching the observed production
interface was copied rather than vendoring every clone.

## Blockers and uncertainty

- The rolling and accessible projectile sources are now readable; the original
  permission gate for those task families is cleared.
- The exact historical projectile module
  `scripts_mujoco_projectile_catch_robocasa_train2500_yaml_scenes` remains
  missing. The observed family/default/schema contract can be reconstructed from
  the available source and output metadata, but historical rollout behavior and
  byte provenance cannot be established.
- `scripts_mujoco_nominal_catch` and `scripts_mujoco_lat_drift_catch` remain
  permission denied. They are non-blocking for the selected scope because other
  inspected catch lineages are available, but no implementation-equivalence
  claim is made about them.
- None of the selected source directories is a Git working tree. Provenance is
  therefore content-addressed with per-file and component SHA-256 digests.
- Asset catalog entries reference external repositories. They do not assert
  that every model is licensed for redistribution, safe to load, correctly
  scaled, or directly compatible with MuJoCo. RobotWin descriptors are marked
  SAPIEN-oriented and require conversion plus scale/collision validation.

## Data and physics issues requiring correction

1. **Intended branch is not actual outcome.** The legacy rolling and projectile
   writers prefill `outcome` and `failure_mode` from the requested branch. The
   controller's measured `success` disagrees in 1,196/4,500 rolling and
   371/3,000 projectile records. Success-branch rolling generation also retries
   with changed random draws, biasing the distribution.
2. **Assisted capture must be explicit.** Cloth and rope use equality-connect
   proxy grasps. Ball controllers can pin a captured sphere to a gripper-relative
   offset. These lineages belong in `assisted_contact`, not `free_contact`.
3. **The bounce corpus is scripted.** `franka_bounce/controller.py` overwrites
   the ball at impact, and its surface has no collision. Bounce v2 includes 247
   intended-success/actual-failure records with `failure_mode=none`; measured
   pre-impact speeds are also inconsistent with the short drop.
4. **Cloth outcomes are not objective labels.** All 14,004 production cloth
   records declare a success branch, yet no task success metric is stored. At
   least one sampled box-fold clip visibly ends outside the box while labelled
   success.
5. **Rope success is qualified by snags.** `thread_through_ring` has 0/216
   successes and `shake_wave` 47/216. Of 1,380 labelled successes, 804 also set
   `snag_flag=true`; release success must require task geometry and no unintended
   snag.
6. **Schemas are incompatible.** Sources differ in directory layout, IDs,
   resolutions, frame rates, action dimensions and meanings, timestamp rates,
   quaternion documentation, camera calibration, label names, and splits.
7. **Paths are not portable.** Several legacy records and wrappers contain
   `/home/mzl7` or other absolute checkout paths. They are preserved only as
   provenance and must not be emitted by the maintained CLI.
8. **Camera/time defects exist.** Audited examples include cropped balls,
   black room voids, tight wrist views, overexposed rope, occluded task geometry,
   and long static terminal tails. Requested durations also differ from encoded
   inclusive-frame durations.

## Asset catalog limitations

`asset_catalog.parquet` contains 4,584 readable descriptor files:

| Source project | Rows | Configured root convention |
|---|---:|---|
| RoboCasa | 4,373 | `ROBOCASA_ROOT`, paths beginning `robocasa/models/assets/` |
| MuJoCo Menagerie | 13 | `MUJOCO_MENAGERIE_ROOT` |
| RoboTwin 1.0 | 2 | `ROBOTWIN_ROOT`, paths beginning `models/` |
| RoboTwin 2.0 | 196 | `ROBOTWIN_2_ROOT`, paths beginning `assets/objects/` |

Every row has a content SHA-256 and byte size. `scale` is deliberately null
because source descriptors do not provide one portable, validated scale value;
inventing it would be unsafe. License fields point to notices within each
configured repository and are references, not redistribution approval.
