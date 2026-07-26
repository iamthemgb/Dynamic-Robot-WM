# Production generator revision

This document describes the production-oriented revision of the maintained
generator. It is an implementation and release contract, not evidence that a
production corpus has already been generated. The immutable source-migration
decisions remain in
[`original_consolidation_plan.md`](archive/original_consolidation_plan.md) and
`migration/`; neither this
revision nor its commands write to a source dataset or to the Wan training
workspace.

Implementation effort is intentionally weighted about 70% toward deepening the
existing falling, rolling, projectile, and rebound core and 30% toward
retention, deflection, finite-surface transitions, and recovery. New unrelated
task families are not used as a substitute for label, physics, balance, and
variation repairs.

## What is production and what is diagnostic

The repository now distinguishes three paths:

| Path | Purpose | Physics/rendering claim | Training eligibility |
|---|---|---|---|
| `source_mujoco` | Planned canonical adapter over the existing real Panda-hand and Panda+Robotiq generators | Must preserve source MuJoCo states, actions, contacts, and synchronized rendering while normalizing to v2 | Fail-closed until the normalizer and objective replay are implemented and tested |
| retired `native_mujoco` | Low-level regression for the first custom attachment implementation | MuJoCo advances state, but the flange carries a synthetic tray/bin/paddle instead of the required grippers | Never training or acceptance data; public execution disabled |
| diagnostic family adapters and state renderer | Fast schema, writer, split, QC, and Wan-export regression | Deterministic analytical/scripted fixtures, not native production physics | Always excluded |

The old `--renderer module:function` hook is retained only for diagnostic and
backward-compatible workflows. A renderer plug-in cannot turn an analytical
adapter trajectory into native physics. Corrected production execution must follow:

```text
real Panda-hand or Panda+Robotiq source generator
  -> source episode plus synchronized LeRobot tables/videos
  -> source_mujoco canonical-v2 adapter
  -> synchronized states, actions, contacts, cameras, and frames
  -> atomic episode writer
  -> objective recomputation, QC, splits, and Wan export
```

The allowlist is exactly `franka_hand` and `robotiq_2f85_thick_pad`. The latter
is an explicit source variant of the real Robotiq 2F-85, not a generic tool
alias. Source access and hashes are checked by `inspect-embodiments`; access does
not establish release readiness. Current successful catch controllers hold or
rewrite ball state after capture, and the Panda bounce source also rewrites the
bounce response, so they are embodiment references rather than release
candidates. Object motion after initialization must remain native MuJoCo motion,
and canonical provenance must bind the exact source files, robot/gripper model,
action schema, and conversion version.

## Version-2 dataset contract

New outputs use `dynamic-robot-dataset/v2`. The reader accepts v1 records and
upgrades them in memory, but a legacy record does not silently become release
eligible. The canonical video/table contract remains two synchronized H.264
`yuv420p` views at 832x480 and 30 FPS, plus frame-, controller-, event-, and
object-state Parquet tables in SI units, a right-handed frame with +Z up, and
WXYZ quaternions.

The v2 episode contract adds:

- the closed measured-outcome classes `success`, `partial_success`,
  `near_miss`, `contact_failure`, `miss`, `no_op`, `wrong_action`, `invalid`,
  and `unverified`;
- a primary failure code, versioned failure tags, and the compatibility
  `failure_mode`, which must agree;
- an objective evaluator ID/version, threshold-set hash, persisted evidence,
  and key-event name/time;
- per-frame task phase, motion mode, active surface, contact role, and explicit
  transition events;
- controller, latency, robot-start, tool-calibration, camera-calibration, and
  physics-range provenance;
- zero or more simulator-observed assistance mechanisms, including target
  bodies/elements and activation intervals;
- a declared counterfactual-family table containing expected siblings,
  intervention fields, and hashes of every field that must stay fixed.

Intended branch and measured outcome are always separate. A mismatch is saved
as evidence; it is never relabelled or retried merely to satisfy the intended
class. `no_op` is a first-class measured and intended class.

Native rigid records use the built-in
`native_rigid_state_event/1.2.0` evaluator. The online label path and the QC
path invoke the same versioned evaluator over saved frame/event/transition evidence; QC
reconstructs the immutable scenario spec from episode provenance and never
reads branch intent to decide the outcome.

### Counterfactual symmetry

Action siblings keep physics, scene, initial state, appearance, cameras, and
assets fixed. Physics siblings replay the same timestamped action and keep the
scene, initial state, appearance, cameras, and assets fixed while changing only
the declared physics field. Both relations share the connected
`split_group_id`. A missing expected sibling, an undeclared intervention, or a
changed fixed-field hash excludes the family from release.

The default split is deterministic connected-group 80/10/10. It stratifies as
far as the connected groups permit over family, subfamily, measured outcome,
background style, asset, tool, and physics bin. Sparse strata are reported as
readiness failures; the splitter never breaks a counterfactual or duplicate
group to improve counts.

### Release eligibility

An episode can enter the default training manifest only when all of the
following are true:

1. its v2 label is produced and independently recomputed by the declared
   objective evaluator from saved artifacts;
2. it is `free_contact`, with simulator-derived assistance evidence showing no
   hidden latch or constraint;
3. its physics values come from a calibrated, content-addressed range profile;
4. native physics checks and hard dataset QC pass;
5. it has no disqualifying quality flag; and
6. its complete counterfactual and split groups pass leakage checks.

Assisted, scripted, proxy, unverified, and diagnostic records remain useful as
named tiers or negative controls, but not as default physics supervision.

## Scenario scope

The revision deliberately deepens a small dynamics-centered scenario library.

| Area | Maintained direction | Current release boundary |
|---|---|---|
| Falling/catch | centered Panda-hand drops, Panda-hand bounce-catch, and centered/lateral Panda+Robotiq catch are the currently verified source lineages | Expand only by actuating the real fingers/clamp; retention means a measured physical grasp, never a tray or hidden latch |
| Rolling/sliding | both directions, rolling/sliding transitions, slopes, finite surfaces, and direct interception remain the highest-priority design target | No corrected real-gripper production adapter exists yet; old paddle/bin cases are retired, not relabelled |
| Projectile/rebound | direct Robotiq interception and Panda bounce-catch are the current source lineages; wall/table/edge transitions remain desired | One or two well-defined contacts; no synthetic paddle attachment, and measured bounce response is never inferred from solver settings |
| Cloth | freeze volume; objectively evaluate poke; tier lift/release and fold-edge as assisted unless free contact is demonstrated | Dual-Franka box folding remains suspended |
| Rope | repair drag, tug, lift/drape, wrap, and a small ring-thread pilot; bind the grasp equality to the requested segment | Shake-wave is suspended; ring thread is not released until the corrected target and metric pass |
| Soft objects | interfaces and configs only; foam-ball hand/clamp catch is first | Blocked until rigid-contact gates pass; beanbag and pouch additionally require validated shell/self-contact models |
| Deferred | handoff beyond catch recovery, complex bags, knots, fluids, chaotic multi-object and uncontrolled multi-bounce tasks | No production generator in this revision |

The cloth and rope modules currently provide objective/tiering contracts and the
rope equality-target repair. The accessible Franka cloth package is explicitly a
preview: poke is contact-based, while lift/fold use equality-connect proxy
grasps. These are not, by themselves, completed free-contact deformable data.

`configs/families/soft_objects_gated.yaml` makes the soft-object boundary
machine-readable. Its production flag is false; a config flag cannot override
missing rigid, shell, or self-contact evidence, and diagnostic/scripted motion
cannot satisfy the gate.

Visual randomization is bundle-seeded and independent of intended or measured
outcome. The three named styles are clean Franka laboratory, RoboCasa kitchen
tabletop, and RobotWin cluttered tabletop. Loading an external asset is not
admission: scale, collision, rendering, content hash, and license-notice checks
are all required. Appearance, light, cameras, and object identity stay fixed
within counterfactual siblings.

## Diagnostic fixture and retired suite

`configs/families/smoke_120.yaml` remains the immutable fast diagnostic
regression fixture. It contains exactly 120 logical branches and two views per
branch. Its accepted historical artifact checks encoding, synchronization,
metadata, resume, splitting, QC, contact sheets, and Wan conversion. Every
episode is marked diagnostic and excluded from default training.

`configs/families/native_acceptance_160.yaml` preserves the previous 160-case
definition for provenance and low-level regression:

| Category | Branches |
|---|---:|
| Falling, catch, and retention | 40 |
| Rolling, sliding, and tabletop transitions | 48 |
| Projectile, rebound, and deflection | 40 |
| Cloth evaluator/tiering | 12 |
| Rope repair/tiering | 16 |
| Scripted/assisted negative controls | 4 |

It is now marked `retired_custom_attachment_definition` and public dry-run and
execution both fail before creating output. Its 128 rigid cases were generated
with synthetic tray/bin/paddle flange attachments and cannot be repaired by
renaming their `tool_type`. All artifacts from that path in this workspace and
both related SLURM launchers were removed.

The corrected acceptance identity is `real_gripper_acceptance_160_v2`. It is
intentionally not defined until source normalization and objective replay are
implemented. Every release gate requires that exact suite name, both
`franka_hand` and `robotiq_2f85_thick_pad`, and absence of custom flange
attachments. Readiness recomputes this from finalized records and backend
provenance rather than trusting a report flag.

## Physics calibration and staged gates

`configs/physics/rigid_ranges_v1.yaml` contains candidate bounds, not approved
production support. `calibrate-physics` refuses to promote them without native
evidence for free fall, bounce, sliding deceleration, rolling slip, and contact
stability. Solver parameters are stored separately from measured effective
friction and restitution. After measured stable support is admitted, its central
80% forms train-ID support and stable outer values form held-out OOD support.

Calibration evidence is not a self-asserted pass flag. The observations artifact
must use `dynamic-robot-native-calibration-observations/v1`, bind the resolved
range-catalog hash, reference hash-verified `episodes` and `qc_report` source
artifacts, and contain raw trials that the calibrator recomputes for every
required oracle. It must also reference a separate
`dynamic-robot-calibration-approval/v1` artifact whose reviewer, catalog hash,
and observation hash bind that exact evidence. A missing or mismatched source,
trial, support range, or approval keeps the catalog release-ineligible.

The local no-render sweep regression is deliberately diagnostic, not a range
approval. Gravity and sliding-friction responses are monotonic. The versioned
`native-restitution-solver-map/v1` profile measures approximately 0.103, 0.252,
0.501, 0.753, and 0.955 in its declared impact-speed regime for requested
values 0.05, 0.25, 0.50, 0.75, and 0.95. All five pass the regression tolerance,
but this only validates that controlled solver-to-response mapping. Promotion
still requires hash-bound observations across the candidate support, source
episode/QC evidence, and independent reviewer approval; solver values are not
reported as physical restitution.

The pilot files are plans only and set `submit: false`:

- `pilot_10h.yaml`: 2.5 unique hours falling/retention, 3.5 rolling/sliding,
  2.5 projectile/rebound, and 1.5 verified free-contact deformables;
- `pilot_100h.yaml`: 35 unique hours rolling/sliding, 25 falling/retention,
  25 projectile/rebound, 10 deformables, and 5 handoff/recovery.

No pilot is launched by loading or validating these files. The 10-hour gate
stays blocked if its deformable quota is not release-ready; the quota is not
silently moved. The 100-hour gate additionally requires the accepted 10-hour
readiness report and an external model-evaluation artifact showing improved
trajectory/contact prediction, action ranking, and held-out physics performance
without a material visual-quality regression. The 300- and 1,000-hour stages
remain policy-blocked until a positive scaling curve exists.

Both the 10-hour and 100-hour evaluators require `--acceptance-report` pointing
to the canonical `.suite_execution.json` inside the acceptance dataset named by
that report. The report and its immutable `.suite_plan.json` must exist, contain
all 160 committed branches, pass the suite gates, and say both
`passed_execution: true` and `full_native: true`. It must also be named
`real_gripper_acceptance_160_v2`, cover both allowed end effectors, and contain
no custom flange attachment. The retired suite cannot satisfy this condition.
Readiness also
requires the canonical `qc/dataset_report.json` to use
`dynamic-robot-qc-report/v2` and bind the exact dataset root,
the complete finalized metadata manifest, complete episode-UUID membership,
and every episode media/table byte hash. Split diagnostics separately bind the
current `episodes.parquet` and `splits.parquet`. A readiness report used by a
later stage must be stored at `DATASET/qc/readiness/GATE_ID.json`; its gate,
dataset, QC, acceptance, and dependency hashes are revalidated. For the
100-hour gate, the model-evaluation artifact must additionally bind those exact
episode and QC hashes and name existing model/evaluation-manifest files whose
bytes match the declared SHA-256 values.

## Duration accounting

Every report uses named denominators:

| Quantity | Definition |
|---|---|
| Unique source episode hours | Sum of logical rollout durations; synchronized cameras count once |
| Encoded source stream hours | Sum of every stored source video duration; two views normally count twice |
| Unique derived clip hours | Sum of one fixed-window duration per Wan manifest row |
| Encoded derived stream hours | Sum of all encoded Wan sidecar streams |
| Endpoint-padding hours | Repeated endpoint time summed across exported Wan streams because a fixed window exceeds source context |
| Release hours | QC-passed, release-eligible unique source episode hours, also broken down by family, tier, outcome, and split |

All 10/100/300/1,000-hour targets mean QC-passed **unique source episode
hours**. Camera views and padded Wan duration are never counted as additional
physical experience. The statistics report also emits intended-versus-actual
confusion counts and flags visual/outcome associations above Cramer's V 0.1 for
review.

## Known blockers and uncertainty

- The provisional rigid parameter ranges still need measured native calibration
  and explicit release approval.
- The complete 160-case native artifact has not yet been accepted; an exact
  suite expansion alone is insufficient.
- Native cloth, rope, foam, beanbag, and pouch model validation remains separate
  work. Deformable evaluator code does not prove topology, self-contact,
  tunneling, strain, or visibility quality.
- The installed MuJoCo 3.10 build supports the retained native cable models but
  does not expose the required `mujoco.elasticity.shell` plug-in. An edge-only
  flex grid is not being relabelled as a production shell-cloth model; native
  cloth release remains blocked until a validated model/plugin is available.
- The historical projectile module
  `scripts_mujoco_projectile_catch_robocasa_train2500_yaml_scenes` remains
  missing; the selected accessible implementation is behaviorally similar but
  not claimed byte-identical.
- Two mzl7 catch source directories remain inaccessible. Other lineages cover
  the selected scope, but feature equivalence is unknown.
- RobotWin assets are not automatically MuJoCo-ready or redistribution-cleared.

These blockers are intentional release stops, not reasons to guess labels,
physics semantics, or provenance.
