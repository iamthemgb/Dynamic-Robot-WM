You are consolidating and repairing several robotics dataset-generation codebases on a GPFS cluster.

## Objective

Inspect the existing generation scripts under:

* `/gpfs/radev/scratch/sous/mzl7`
* `/gpfs/radev/scratch/sous/zss8`
* `/gpfs/radev/scratch/sous/zl664`

Then copy and refactor the relevant generation code into:

```text
/gpfs/radev/project/sous/zl664/dataset_generation
```

The destination should become the single organized, reproducible codebase for future dynamic-object robotics dataset generation.

Do not blindly copy everything. First identify the canonical generator for each task family, known duplicates, failed runs, derived media, inconsistent schemas, and invalid label logic.

Do not move, delete, rename, or modify anything under the three source roots. Treat all source directories as read-only.

Do not copy generated videos, checkpoints, caches, temporary frames, or large output datasets into the new repository. Copy only scripts, configurations, small required assets, schema definitions, and documentation. For large shared assets, create configurable path references instead.

Do not generate the full production dataset in this task. Only produce small smoke-test datasets after the refactoring is complete.

Fluids are out of scope and must not be implemented or migrated.

---

# Known issues that must be addressed

The existing datasets are not mutually plug-compatible. Known problems include:

1. Different directory layouts, IDs, video resolutions, FPS, durations, camera names, trajectory rates, action dimensions, coordinate conventions, and labels.

2. Intended branch labels are sometimes incorrectly treated as actual outcomes.

3. Cloth datasets often claim success without an objective task metric.

4. Rolling and projectile datasets contain substantial disagreement between intended outcome and actual controller success.

5. Some bounce episodes are intended successes but actually fail while recording `failure_mode=none`.

6. Some bounce trajectories appear physically suspicious and need validation against gravity, impact velocity, and restitution.

7. Rope task labels are unreliable:

   * `thread_through_ring` has essentially no successful examples.
   * `shake_wave` has very low success.
   * many labelled successes also have `snag_flag=true`.

8. Existing physics may use equality constraints, latches, or assisted grasping. These episodes must explicitly record when assistance is active and must not be described as ordinary free-contact dynamics.

9. Some source trees contain duplicates:

   * a `_gpu` Robotiq run under `zl664` duplicates the corresponding canonical trajectory lineage;
   * the composite rope rendering duplicates the per-view rope dataset;
   * cloth mosaics are derived representations and should not be primary camera streams.

10. Existing paths sometimes contain stale absolute output paths or asset paths belonging to other users.

11. Existing train/validation/test splits are inconsistent or absent.

12. Camera calibration is generally missing, and camera naming is inconsistent.

13. Existing videos sometimes have:

* cropped or tiny balls;
* long static tails;
* black room voids;
* overexposed rope;
* occluded critical contacts;
* mosaics that reduce useful detail.

These issues must be fixed at the generator and schema level before large-scale generation.

---

# Phase 0: read-only inventory and migration plan

Before copying or modifying code, recursively inspect the three source roots.

Search for:

```text
*.py
*.sh
*.yaml
*.yml
*.json
*.toml
*.xml
*.mjcf
*.urdf
README*
requirements*
environment*
pyproject.toml
```

Search file contents for terms such as:

```text
generate
render
rollout
scenario
mujoco
mjx
genesis
robocasa
robotwin
franka
robotiq
cloth
rope
rolling
projectile
bounce
catch
success
failure_mode
outcome
```

Create:

```text
/gpfs/radev/project/sous/zl664/dataset_generation/migration/
  source_inventory.json
  source_mapping.yaml
  known_issues.md
  migration_plan.md
```

For every relevant source component, record:

```text
source_owner
absolute_source_path
task_family
subfamily
simulator
robot_model
renderer
current_output_format
known_generated_dataset
known_quality_issues
duplicate_or_canonical_status
proposed_destination
copy_status
access_status
```

If a directory is permission-denied, record the exact path and error. Do not infer or fabricate its contents.

Stop and report clearly if the missing directory contains a generator that is essential and no equivalent accessible implementation exists.

---

# Canonical source lineages

Use the audit findings as initial guidance, but verify all decisions from code and metadata.

Expected relevant sources include:

## `mzl7`

* kitchen and neutral-tabletop cloth generation;
* continuous-physics cloth generation;
* rolling-island interception;
* projectile-ball interception in RoboCasa-style scenes.

Do not copy pilot datasets, example renderers, temporary previews, Wan outputs, smoke tests, or generated media.

Treat cloth mosaics as derived visualizations, not primary observations.

## `zl664`

* canonical Robotiq thick-pad catch generator;
* any current MuJoCo/MJX dynamic-preview or realistic Franka generators that are genuinely reusable.

Do not choose the known `_gpu` repeated dataset lineage as a separate canonical generator if it is only a duplicate execution of the same trajectories.

## `zss8`

* Franka vertical-drop and bounce generators;
* newer Robotiq variants if they represent genuinely different timing/geometry;
* per-view rope generator.

Do not copy the older composite rope rerender as a separate dataset generator.

Keep distinct generator versions only when there is a meaningful physical, timing, robot, or rendering difference. Record such versions explicitly rather than silently merging them.

---

# Destination repository structure

Create this structure:

```text
dataset_generation/
  README.md
  pyproject.toml
  requirements/
    base.txt
    mujoco.txt
    deformable.txt
  configs/
    schema/
      dataset_schema_v1.yaml
      failure_codes_v1.yaml
    assets/
      asset_roots.example.yaml
    cameras/
    randomization/
    splits/
    families/
  src/
    dynamic_robot_dataset/
      __init__.py
      cli.py
      common/
        schema.py
        paths.py
        provenance.py
        episode_writer.py
        video_writer.py
        synchronization.py
        cameras.py
        randomization.py
        outcomes.py
        contacts.py
        qc.py
        splits.py
        hashing.py
      families/
        rigid_dynamic/
          falling_catch/
          rolling_interception/
          projectile_rebound/
          dynamic_handoff/
        deformable/
          cloth/
          rope/
          soft_body/
          bags/
  legacy_sources/
    mzl7/
    zl664/
    zss8/
  tools/
    inventory_sources.py
    validate_dataset.py
    validate_physics.py
    inspect_labels.py
    build_asset_catalog.py
    build_splits.py
    make_contact_sheets.py
    export_wan.py
    export_cosmos.py
  tests/
    unit/
    integration/
  migration/
  examples/
  outputs/
    smoke_tests/
```

Copy the original relevant source scripts into `legacy_sources/` without changing them. Preserve their relative paths where practical.

Implement cleaned generators under `src/dynamic_robot_dataset/families/`.

For every migrated file, maintain a migration ledger in:

```text
migration/source_mapping.yaml
```

Record:

```text
original_path
copied_legacy_path
new_refactored_path
major_changes
known_behavior_difference
```

---

# Common command-line interface

Implement a common CLI such as:

```bash
python -m dynamic_robot_dataset.cli generate \
  --family falling_catch \
  --subfamily centered_vertical_drop \
  --num-bundles 10 \
  --branches success,near_miss,contact_failure,bad_action \
  --views main,secondary \
  --seed 0 \
  --randomization-level R1 \
  --output /path/to/output
```

Support at least:

```text
--family
--subfamily
--variant
--robot-model
--tool-type
--num-bundles
--branches
--views
--seed
--randomization-level
--scene-style
--output
--resume
--dry-run
```

A bundle means one initial physical scene with multiple counterfactual action branches.

---

# Canonical dataset format

Use a LeRobotDataset-v3-style MP4 + Parquet layout.

The implementation should use portable relative paths and sharding. Temporary per-episode files may be used during generation, but the finalized dataset must be packable into the common sharded layout.

Use:

```text
dataset_root/
  meta/
    info.json
    episodes.parquet
    tasks.parquet
    cameras.parquet
    splits.parquet
    provenance.parquet
  data/
    chunk-000/
      file-000.parquet
  videos/
    observation.images.main/
      chunk-000/
        file-000.mp4
    observation.images.secondary/
      chunk-000/
        file-000.mp4
  high_rate/
    chunk-000/
      file-000.parquet
  events/
    chunk-000/
      file-000.parquet
  qc/
    episode_qc.parquet
    dataset_report.json
```

Do not use a stitched multiview mosaic as a primary observation. Mosaics may be generated only as derived QA artifacts.

---

# Canonical video and timing specification

Use a two-layer format.

## Canonical source episodes

```text
resolution: 832 x 480
source FPS: 30
camera views: two synchronized views by default
duration: event-adaptive
codec: H.264/yuv420p MP4
units: SI
```

Recommended event-adaptive durations:

```text
simple falling catch: 2.5–3.5 seconds
off-center/drifted falling catch: 3–4 seconds
rolling/sliding interception: 3.5–6 seconds
projectile/rebound: 3.5–6 seconds
catch-then-retain: 4–6 seconds
cloth/rope/deformable: 4–6 seconds
recovery or multi-stage task: up to 8 seconds
```

Do not stretch or slow physics to fit a fixed clip duration.

Each clip must contain:

```text
pre-event context
the dynamic event
post-contact outcome or settling evidence
```

Avoid long static tails.

## Model-specific derived export

Implement an exporter for the initial Wan experiment:

```text
832 x 480
24 FPS
121 frames
approximately 5.04 seconds
event-centered crop or padding
```

Padding may repeat initial/final frames only when required by the model. It must never change physical time or rescale motion speed.

Keep the canonical source independent of Wan/Cosmos assumptions.

---

# Camera standard

Require two synchronized cameras for rigid dynamic tasks:

```text
observation.images.main
observation.images.secondary
```

Suggested roles:

```text
main:
  close three-quarter external view

secondary:
  side view for falling/projectile/rebound
  top-oblique view for rolling/rope/cloth
  wrist/tool view only when useful and sufficiently contextualized
```

Optional additional streams:

```text
observation.images.top
observation.images.wrist
observation.depth.main
observation.segmentation.main
```

For every camera, save:

```text
camera_name
intrinsic matrix
distortion model/coefficients
world-to-camera transform
camera-to-world transform
resolution
FPS
near/far planes
renderer
```

Camera streams must be synchronized to the same episode clock and event timestamps.

Automatically reject or flag episodes where:

```text
the target is cropped during important frames
the target is too small
the critical contact is occluded in both views
the scene is severely overexposed
the render contains large unintended black voids
```

---

# Unified coordinate and action conventions

Declare the following in dataset-level metadata:

```text
units: SI
world frame: right-handed
up axis: +Z
quaternion order: WXYZ
pose convention: position followed by quaternion
velocity frame: world unless explicitly named otherwise
```

Never store an unnamed state or action vector without declaring its fields.

For every robot episode, save named features for:

```text
robot_model
base_pose_world

q[7]
dq[7]
ddq[7], if available

tau_cmd[7], if applicable
tau_measured[7], if available
tau_external[7], if available

q_target[7], if applicable
dq_target[7], if applicable

ee_pose_world
ee_twist_world
cartesian_target_pose
cartesian_target_twist

gripper_q
gripper_dq
gripper_width
gripper_width_target
gripper_open_close_command
gripper_force_command
gripper_force_measured, if available

controller_type
controller_gains
action_mode
```

Supported action modes should be explicitly named, for example:

```text
joint_position_absolute
joint_position_delta
joint_velocity
joint_torque
ee_pose_absolute
ee_pose_delta
ee_twist
```

Do not merge incompatible action semantics into the same unnamed tensor.

---

# Frame-aligned and high-rate data

The canonical frame Parquet must contain one synchronized row per rendered video timestamp.

Include:

```text
episode_index
frame_index
timestamp
video_frame_index
task_index
robot state
action
primary target-object state
contact flags
event flags
```

Also save a high-rate sidecar at the simulator/controller rate when available.

Metadata must declare:

```text
sim_hz
control_hz
video_hz
timestamp dtype
video time base
```

Use exact timestamps or PTS-based synchronization. Do not assume `frame_index / FPS` when encoded video timing differs.

For scenes with multiple objects, store:

```text
primary target state in the frame table
additional object states in a long-format object-state table keyed by object_id and timestamp
```

---

# Episode identity and provenance

Every episode branch must contain:

```text
episode_uuid
episode_index
counterfactual_bundle_id
scene_seed
branch_seed
family
subfamily
variant
robot_model
tool_type
intended_branch
actual_outcome
task_success
partial_success_score
failure_mode
split
parent_episode_uuid, if rerendered
source_generator
source_generator_version
generator_git_commit
config_hash
asset_ids
asset_hashes
simulator_name
simulator_version
renderer
creation_timestamp
```

Use globally unique IDs. Do not depend only on local integer episode numbers.

All output paths must be relative to the dataset root.

---

# Intended branch versus actual outcome

These must be separate fields.

```text
intended_branch:
  the behavior the generator attempted

actual_outcome:
  the result measured after simulation
```

Never assign `task_success` from the intended branch name.

Implement a task-specific objective evaluator for every included subfamily.

Every evaluator must return:

```text
task_success: bool
partial_success_score: float or null
failure_mode: required string
metrics: dictionary of objective quantities
label_confidence
label_status
```

Rules:

1. If `task_success=false`, `failure_mode` must not be `none`, null, or empty.

2. If no objective evaluator exists, set:

```text
label_status=unverified
```

and exclude the episode from release/training by default.

3. Assisted or proxy-physics episodes must include:

```text
assisted_grasp
assisted_retention
equality_constraint_active
latch_active
constraint_activation_time
constraint_deactivation_time
```

Prefer per-frame masks when assistance changes during the episode.

4. A contact is not automatically a success.

5. A small minimum distance is not automatically a success.

---

# Target branch distribution

Use the following simple intended branch mix per subfamily:

```text
40% success-seeking
30% near-miss
20% contact-failure
10% bad-action or no-op
```

Definitions:

```text
success-seeking:
  nominal action intended to complete the task

near-miss:
  small spatial or timing error

contact-failure:
  contact occurs, but the object bounces, slips, snags, escapes,
  redirects incorrectly, or is not retained

bad-action/no-op:
  no motion, wrong direction, unreachable target, or clearly poor control
```

Generate counterfactual branches from the same initial scene whenever possible:

```text
same scene
same background
same camera poses
same object parameters
same release state
same physics parameters
different robot action branch
```

After simulation, compute actual outcomes objectively.

The released dataset should target an actual distribution close to:

```text
40% success
30% near-miss
20% contact failure
10% bad/no-op
```

Report actual achieved distributions by family and subfamily. Do not relabel episodes merely to reach the target distribution.

---

# Physics metadata

Save all parameters required to reproduce each rollout.

For rigid objects:

```text
asset_id
shape
dimensions
mass
density
center of mass
inertia
friction coefficients
rolling friction, if modeled
torsional friction, if modeled
restitution
initial pose
initial linear velocity
initial angular velocity
release time
external impulses
gravity
solver settings
simulation timestep
substeps
```

For surfaces/tools:

```text
surface asset ID
surface normal
geometry
friction
restitution
tool dimensions
tool mass/inertia
tool mount transform
```

For cloth:

```text
stretch stiffness
bend stiffness
shear stiffness
density
thickness
cloth-table friction
cloth-tool/gripper friction
damping
mesh resolution
assisted grasp flags
```

For rope/cable:

```text
length
radius
density
stretch stiffness
bend stiffness
damping
rope-table friction
rope-tool/gripper friction
segment count
assisted grasp flags
```

For soft bodies:

```text
Young's modulus
Poisson ratio
density
damping
friction
restitution
plasticity
mesh resolution
```

Also save contact-event tables containing:

```text
timestamp
object_a
object_b
contact point
contact normal
penetration depth
normal force or impulse
relative velocity
```

---

# Physics validation

Implement automated checks.

## Falling and projectile checks

Verify:

```text
saved position finite differences agree with saved velocity
saved velocity finite differences approximately agree with acceleration
free-fall segments are consistent with configured gravity
no unexplained impulses occur before contact
```

## Bounce checks

Verify:

```text
pre-impact velocity is plausible for release height and initial velocity
impact frame matches surface crossing
post-impact normal velocity is consistent with configured restitution
tangential change is consistent with friction
no large unexplained energy gain
```

Episodes that violate tolerances must be marked:

```text
physics_qc_pass=false
```

They must not enter the release dataset by default.

## Contact checks

Detect:

```text
explosive penetration
persistent interpenetration
object teleportation
constraint instability
unrealistic velocity spikes
```

## Assisted dynamics

Separate:

```text
free_contact
assisted_contact
scripted_motion
```

in metadata. Do not mix them silently.

---

# Correct existing label logic

## Cloth

Do not accept universal success labels.

Implement or document objective metrics per task.

Examples:

```text
poke_cloth:
  intended contact region reached and sufficient cloth displacement

lift_corner_release:
  correct corner lifted above threshold and final placement satisfies target

fold_edge_fixed_line:
  required cloth region crosses fold line with sufficient overlap/alignment

dual_franka_tshirt_fold_box:
  sufficient fraction of cloth vertices/area lies inside target box,
  with bounded spill outside the box
```

If a robust metric cannot be implemented for a cloth task, label it unverified and exclude it from supervised outcome training.

## Rolling/projectile/catch

Separate:

```text
intended_branch
controller_reached_target
object_contacted_tool
object_entered_receptacle
object_retained_until_end
actual_success
failure_mode
```

## Bounce

Repair any path where actual failure produces `failure_mode=none`.

Validate suspicious impact speeds before using the generator for scale-up.

## Rope

Define task-specific metrics for:

```text
coil_on_table
drag_endpoint
lift_and_drape
shake_wave
sweep
thread_through_ring
tug
twirl_overhead
wrap_around_post
```

A task must not count as success merely because contact occurred.

For `thread_through_ring`, require the endpoint or specified rope section to cross the ring plane and remain on the target side.

For snag-sensitive tasks, decide explicitly whether snagging invalidates success. Save both:

```text
task_success
snag_flag
```

Do not allow contradictory labels without an explicit `partial_success` interpretation.

---

# Background and asset randomization

Build an asset catalog from accessible local RoboCasa, RoboTwin, MuJoCo Menagerie, and project assets.

Do not copy entire external asset repositories into this project. Store configurable roots in:

```text
configs/assets/asset_roots.example.yaml
```

Support environment variables such as:

```text
ROBOCASA_ROOT
ROBOTWIN_ROOT
MUJOCO_MENAGERIE_ROOT
```

Create:

```text
migration/asset_catalog.parquet
```

with:

```text
asset_id
source_project
relative_path
asset_type
category
license_or_notice_path
compatible_simulator
scale
known_issues
```

Randomize independently of outcome:

```text
room/background scene
table/workbench
floor
wall
lighting intensity and position
camera exposure
camera pose jitter
object color
object texture
object geometry
tool material
tool geometry
distractor objects
lab clutter
```

Initial background-style mix:

```text
60% clean Franka lab
25% RoboCasa-style kitchen/tabletop
15% RoboTwin-style cluttered tabletop
```

Do not allow background, ball color, camera, or asset family to correlate with intended or actual outcome.

For each counterfactual bundle, keep visual randomization fixed across branches unless the experiment explicitly studies visual intervention.

Save:

```text
scene_asset_id
background_style
lighting_id
object_asset_id
object_color_id
tool_asset_id
camera_preset_id
```

---

# Split design

Create dataset-local splits.

Use bundle-aware, scene-aware grouping:

```text
train: 90%
validation: 5%
test: 5%
```

All branches from the same counterfactual bundle must remain in the same split.

Avoid leakage across:

```text
scene seed
initial state
rerender lineage
background scene
near-duplicate trajectories
```

Also generate optional OOD evaluation manifests for:

```text
unseen object assets
unseen backgrounds
unseen physics ranges
unseen camera poses
unseen tool geometries
```

---

# Automated quality control

Implement `tools/validate_dataset.py`.

Check every episode for:

```text
MP4 decodes fully
expected codec/pixel format
correct resolution
correct FPS/time base
camera streams synchronized
frame count consistent with metadata
Parquet timestamps monotonic
required metadata present
relative paths valid
content hashes valid
no frozen video
limited static tail
object visible during key event
object not critically cropped
contact event within clip
post-event evidence present
objective label agrees with metrics
failure code present for failures
physics QC passed
no split leakage
no exact duplicate
```

Add approximate/perceptual duplicate detection for rendered videos.

Create:

```text
qc/dataset_report.json
qc/episode_qc.parquet
qc/failure_summary.csv
qc/outcome_distribution.csv
qc/physics_validation.csv
qc/contact_sheets/
```

Reject or quarantine episodes failing hard checks.

---

# Smoke-test generation only

After migration and refactoring, generate a small smoke-test set.

Do not generate more than approximately 200 episode branches total.

Include at least:

```text
falling catch:
  success
  near miss
  contact failure
  no-op

rolling interception:
  success
  near miss
  contact failure
  no-op

projectile/rebound:
  success
  wrong rebound prediction
  contact failure
  no-op

cloth:
  at least one task with objective evaluator

rope:
  at least one successful and one failed task with objective evaluator
```

Render two synchronized camera views.

Use at least three background styles where assets are available.

Run all dataset, physics, label, video, and split validators.

Create contact sheets for manual inspection.

---

# Tests

Add unit tests for:

```text
schema validation
failure-code enforcement
branch-versus-outcome separation
timestamp synchronization
camera calibration serialization
coordinate/quaternion conversion
split grouping
counterfactual bundle grouping
physics finite-difference checks
bounce restitution checks
constraint/assistance flags
portable relative paths
```

Add integration tests that generate a tiny dataset and run the full validator.

---

# Required deliverables

At completion, provide:

1. A concise summary of what was inspected.

2. A table of canonical versus excluded source generators.

3. Exact source-to-destination mappings.

4. A list of permission or missing-asset blockers.

5. The organized destination repository.

6. The common schema and CLI.

7. Refactored generators for all accessible existing non-fluid families.

8. Objective outcome evaluators or explicit unverified labels.

9. Physics validation tools.

10. Background/asset randomization infrastructure.

11. Bundle-aware split generation.

12. A small smoke-test dataset with two views.

13. Contact sheets and QC reports.

14. Exact commands required to reproduce the smoke test.

15. A list of remaining issues that must be fixed before generating 10, 100, or 1,000 hours.

---

# Hard constraints

* Never modify the source roots.
* Never delete or move source files.
* Never silently overwrite existing destination files.
* Never copy large generated datasets into the code repository.
* Never invent contents for permission-denied directories.
* Never call an intended-success branch an actual success without objective validation.
* Never leave `failure_mode` empty for an actual failure.
* Never mix assisted and free-contact physics without explicit metadata.
* Never use fluid simulation.
* Never begin full-scale generation during this task.
* Prefer a correct, testable migration over supporting every legacy option.
* Preserve provenance for every copied/refactored generator.

Begin with the read-only inventory and migration report. Then centralize and refactor the accessible canonical generators, run the smoke tests, and finish with a concise implementation report.
