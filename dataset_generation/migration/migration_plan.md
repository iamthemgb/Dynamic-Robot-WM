# Source migration plan

This document fixes the source-lineage decisions used by the unified generator.
It complements the machine-readable inventory and per-file mapping ledger; it
does not authorize changes to any source root.

## 1. Immutable-source procedure

1. Treat the three scratch roots and the collaborator project trees as
   read-only. Never generate beside, relabel, chmod, delete, or normalize them.
2. Select sources only after checking production wrappers, imports, output
   metadata, and representative records. A directory name alone is not evidence.
3. Copy only the 148 allowlisted source/documentation/wrapper files recorded in
   `source_mapping.yaml`. Do not copy media, output datasets, logs, caches,
   environments, backups, third-party repositories, pilots, or fluid code.
4. Preserve copied bytes. `source_before_after_hashes.json` must report both
   source reads and the legacy copy as identical before any refactoring begins.
5. Refactor only under `src/dynamic_robot_dataset`; never edit the snapshots in
   `legacy_sources`.

## 2. Family mapping

| Unified family | Selected source | Required treatment |
|---|---|---|
| `deformable/cloth` | Both mzl7 cloth packages | Share one adapter; retain kitchen/tabletop and preset/continuous as named variants. Mark equality grasp assisted, drop mosaics as primary views, and compute task-specific objective outcomes. |
| `rigid_dynamic/rolling_interception` | mzl7 `ball_roll_interception_scripts` | Preserve the three layouts. Remove success-conditioned retries, emit measured outcomes, record assistance, and add synchronized state/action/object/contact data. |
| `rigid_dynamic/projectile_rebound` | mzl7 available YAML-scene projectile package | Preserve the four production scene/camera variants as an assisted legacy mode. Implement free-contact rebound separately, without pose or velocity resets. |
| `rigid_dynamic/falling_catch` | zl664 Robotiq v1, zss8 Franka, zss8 Robotiq v2 | Use embodiment adapters with named states/actions. Keep v1/v2 timing, camera, and control differences explicit. Never merge their records or call them independent task families. |
| `rigid_dynamic/projectile_rebound:legacy_scripted_bounce` | zss8 `franka_bounce` | Preserve for reproducibility only. Force release tier `scripted_motion` and exclude it from default physics supervision. |
| `deformable/rope` | zss8 per-view rope | Keep three primary views, equality-assistance state, contacts and trajectory. Success requires task geometry and no unintended snag. |
| `deformable/soft_body` | zl664 soft-body preview package | Migrate the accessible source but keep outputs `unverified` until objective metrics and stability tests pass. |

No bag or dynamic-handoff generator is registered without an inspected source
and evaluator. Fluids remain excluded.

## 3. Canonicalization rules

- Convert output into the versioned MP4/Parquet contract with portable relative
  paths, explicit SI units, right-handed `+Z`, WXYZ quaternions, calibrated
  cameras, exact time bases, and separate controller-rate and video-rate tables.
- Store persistent physics separately from transient state and action. Unknown,
  unavailable, or simulator-nonidentifiable values are masked, never guessed.
- Keep `intended_branch`, objective `actual_outcome`, and non-null
  `failure_code` separate. Outcome mismatch is data, not a retry condition.
- Record assisted-grasp/latch/equality state at every frame and publish distinct
  `free_contact`, `assisted_contact`, `scripted_motion`, and `unverified` tiers.
- Use `counterfactual_bundle_id` for alternate action branches,
  `physics_counterfactual_family_id` for physics-only intervention, and a common
  `split_group_id` to prevent leakage across either relation.
- Hold camera, assets, colors, initial state, and open-loop action fixed inside a
  physics counterfactual family. Validate invariant hashes before release.
- Preserve source version names and lineage hashes. The missing historical
  projectile module stays explicitly missing; do not rewrite history by naming
  the available snapshot as a byte-identical replacement.

## 4. Asset policy

- Resolve external assets through `ROBOCASA_ROOT`, `ROBOTWIN_ROOT`,
  `ROBOTWIN_2_ROOT`, and `MUJOCO_MENAGERIE_ROOT`; do not vendor those trees.
- Use `asset_catalog.parquet` as the content-addressed descriptor inventory.
- Check that the referenced notice exists before an asset is eligible.
- Require explicit conversion and geometry/scale validation for SAPIEN-oriented
  RoboTwin assets before using them in MuJoCo.
- Treat null scale as unknown. Do not infer meters from mesh dimensions alone.

## 5. Verification and acceptance

Before a migrated family becomes release-eligible:

1. Verify its legacy copy hash and resolved configuration hash.
2. Run a deterministic one-episode replay and compare metadata/trajectory
   invariants; visual bytes need not match when encoding is intentionally
   standardized.
3. Exercise both intended success and failure branches without retrying for a
   desired outcome.
4. Validate media decode, synchronized time bases, finite states/actions,
   coordinate transforms, contact events, camera calibration, non-null failure
   codes, assistance flags, and counterfactual invariants.
5. Run family physics checks. Scripted bounce and unverified deformables cannot
   pass the free-contact gate by construction.
6. Confirm source files still match `source_before_after_hashes.json` and that no
   generated artifact resides below a source root.

The migration is complete only when the unified smoke set passes these gates or
the failing episodes are explicitly quarantined with evidence. Production-scale
generation is a separate operation.
