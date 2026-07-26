# Embodiment correction

Date: 2026-07-13

## Reason

The first maintained `native_mujoco` backend mounted `native_task_tool`
geometry on the Franka flange. Depending on the scenario it used
`shallow_tray`, `deep_tray`, `small_bin`, `flat_paddle`, or `angled_paddle`.
Those rollouts are not Panda-hand or Robotiq-clamp data and cannot be corrected
by relabeling metadata.

The production embodiment contract is now limited to:

- `franka_hand` (`franka_panda`); and
- `robotiq_2f85_thick_pad`
  (`franka_panda_nohand_plus_robotiq_2f85`).

## Removed generated artifacts

The following ignored outputs/logs were removed from the maintained workspace.
No source dataset, collaborator generator, Wan checkpoint, or running job was
modified.

```text
outputs/native_acceptance_160_20260713_a3b6eb88
outputs/native_acceptance_160_20260713_b0f35e20
outputs/native_acceptance_160_20260713_bb07ace
outputs/native_rigid_smoke_20260713_6370047
outputs/native_rigid_smoke_20260713_bb07ace
outputs/slurm-native-acceptance-2110527.out
outputs/slurm-native-acceptance-2110715.out
outputs/slurm-native-acceptance-2110749.out
outputs/slurm-native-smoke-2110519.out
outputs/slurm-native-smoke-2110520.out
outputs/slurm-native-smoke-2110526.out
/gpfs/radev/home/zl664/.codex/visualizations/2026/07/13/019f5b99-dc28-7661-88f1-6f0f74e60df9/acceptance_examples
```

The versioned SLURM launchers `slurm/native_rigid_smoke.sbatch` and
`slurm/native_acceptance_160.sbatch` and the direct no-render launcher
`tools/audit_native_rigid_suite.py` were also deleted. The original
120-episode diagnostic regression fixture remains; it is explicitly
non-training data and did not claim to use the corrected production embodiment.

## Source access verified

`dynamic-robot-dataset inspect-embodiments --require-rigid-sources` resolves
and SHA-256 hashes the required files for:

- zss8 `franka_catch.generate` (Panda hand, 18-state/8-action);
- zss8 `franka_bounce.generate` (Panda hand, 18-state/8-action);
- zl664 `robotiq_catch.generate` (Panda+Robotiq, 24-state/15-action); and
- zl664 `mujoco_franka_cloth_previews.generate_previews` (Panda hand,
  non-release preview/assisted tier).

All four are currently accessible. This inspection is read-only and records
`release_candidate: false` and `release_ready: false`. The catch sources use
post-capture ball-state holding, the Panda bounce source additionally rewrites
the bounce state, and cloth lift/fold use equality-connect grasps. They are
correct-embodiment references and refactoring inputs, not corrected training
episodes.

## Fail-closed behavior

- Public `generate --backend native_mujoco` is no longer accepted by the CLI.
- Config-loaded requests for that backend raise before planning or writing.
- `configs/families/native_acceptance_160.yaml` is marked
  `execution_allowed: false` and `retired_custom_attachment_definition`.
- Pilot configs target `source_mujoco` and retain `submit: false`.
- Release gates require `real_gripper_acceptance_160_v2`, both allowed end
  effectors, and no custom flange attachment.
- Readiness validates finalized `tool_type` values and scans backend provenance
  for the retired attachment identifiers.

`source_mujoco` execution is intentionally not enabled yet. The next required
implementation is an immutable adapter/normalizer from the existing LeRobot
outputs into `dynamic-robot-dataset/v2`, followed by independent objective
recomputation and a new real-gripper smoke suite.

## Corrected free-contact development examples

The small `free_contact_real_grippers_v1` development suite exercises the
corrected control boundary without enabling release generation. It uses only
the Panda hand and mounted Robotiq 2F-85, initializes robot/object state once,
then permits only MuJoCo actuator commands and native freejoint evolution. A
runtime guard compares robot and object qpos/qvel immediately before and after
every controller call and aborts on any post-initialization rewrite. No latch,
weld, equality capture, tray, bin, or paddle is added.

The 2026-07-13 example artifact contains six logical episodes and two views per
episode at 832x480, 30 FPS, 61 frames (2.0 seconds). The two Panda episodes
objectively retain the ball for at least 0.4 seconds with bilateral contact and
maximum measured finger penetration below 2 mm. The Panda offset branch is a
near miss. The three Robotiq examples are measured failures/no-op and remain
development-only. In particular, the stationary Robotiq no-op exposes about
13.5 mm maximum pad overlap, so it is evidence that Robotiq collision/contact
geometry still requires repair, not a release candidate.

```text
outputs/examples/free_contact_real_grippers_v1/contact_sheet.png
outputs/examples/free_contact_real_grippers_v1/manifest.jsonl
outputs/examples/free_contact_real_grippers_v1/suite_report.json
outputs/examples/free_contact_real_grippers_v1/episodes/*/{main,secondary}.mp4
outputs/examples/free_contact_real_grippers_v1/episodes/*/trajectory.parquet
```

The artifact reports zero robot/object state rewrites for every rollout and
verifies that all hashed source files are unchanged. It is deliberately marked
`release_eligible: false`: canonical-v2 conversion, independent objective
replay, calibration/admission approval, and the full acceptance suite are still
required.

## Additional rigid-scenario development examples

The `dynamic_scenarios_real_grippers_v1` suite adds two examples for each of
five rigid scenario categories: catch/retain, rolling/sliding,
projectile/rebound, mode transition, and robot-induced deflection. The Panda
hand is the only end effector in this suite. No tray, bin, paddle, latch, weld,
or equality capture is mounted to the robot. Narrow tracks, finite platforms,
ramps, and rebound walls are explicit world fixtures used to create the
corresponding contact modes; they are not flange tools.

The artifact contains ten logical episodes (25 seconds) and two synchronized
views per episode (50 encoded stream-seconds). Every MP4 is H.264/yuv420p at
832x480 and 30 FPS with 76 frames. The controller-rate trajectories contain
601 rows at 240 Hz, except the contact-sensitive table-rebound example, which
contains 2,501 rows at 1 kHz. Runtime guards report zero robot and object state
rewrites after initialization, and source hashes are unchanged.

Six examples objectively pass: rolling interception, sliding redirect, table
rebound, wall rebound, roll-off-edge, and ramp-to-flight. Both hand-deflection
branches make contact but fail the requested redirection outcome, so they are
preserved as measured contact failures. Both catch/retain branches fail physics
QC because the ball overlaps the Panda hand collision geometry by about 8.3 mm;
one is labeled partial success and the other contact failure. None of these
development examples is release-eligible.

```text
outputs/examples/dynamic_scenarios_real_grippers_v1/contact_sheet.png
outputs/examples/dynamic_scenarios_real_grippers_v1/manifest.jsonl
outputs/examples/dynamic_scenarios_real_grippers_v1/suite_report.json
outputs/examples/dynamic_scenarios_real_grippers_v1/episodes/*/{main,secondary}.mp4
outputs/examples/dynamic_scenarios_real_grippers_v1/episodes/*/trajectory.parquet
```
