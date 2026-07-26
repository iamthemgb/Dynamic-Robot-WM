# mujoco_franka_cloth_previews

Preview pipeline (NOT the final dataset generator) for the first
**Franka-cloth deformable** dataset family. Prioritizes runnable examples,
MP4 previews, and clean metadata over physical realism.

## Task subfamilies

| task_id | embodiment | description |
|---|---|---|
| `poke_cloth` | single_franka | closed-gripper fingertip poke into cloth, then retract |
| `lift_corner_release` | single_franka | grasp one cloth corner, lift, release mid-air |
| `fold_edge_fixed_line` | single_franka | fold one cloth edge over a fixed line (red marker) |
| `dual_franka_tshirt_fold_box` | **dual_franka (full, no proxy)** | two namespaced Pandas fold a T-shirt-proxy cloth into a shallow box |

Dual Franka is a *full* dual-arm MJCF via `<attach ... prefix="L_/R_">`
(automatic namespacing) — `dual_franka_proxy=false` everywhere.

## Requirements

- MuJoCo Python >= 3.3 (flexcomp + `mujoco.elasticity.shell` plugin + MjSpec)
- `numpy`, `imageio`, `imageio-ffmpeg`, `Pillow`
- Panda model from a local MuJoCo Menagerie checkout (searched paths are in
  `scene_builder._MENAGERIE_CANDIDATES`; currently resolves to
  `demo_mujoco_arm_gripper/third_party/mujoco_menagerie`)
- Offscreen GL: `MUJOCO_GL=egl` (set automatically if unset)

A known-good interpreter on this cluster:
`/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper/.venv/bin/python`

## Usage

### All-in-one mode (small runs)

From the directory *containing* this package:

```bash
MUJOCO_GL=egl \
/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper/.venv/bin/python \
  -m mujoco_franka_cloth_previews.generate_previews \
  --out mujoco_franka_cloth_previews/outputs/preview_001 \
  --num-seeds 2 --physics-variants 3 \
  --width 640 --height 480 --fps 24 --duration 5.0
```

Optional: `--tasks poke_cloth lift_corner_release` to run a subset.

### Dataset mode (SLURM arrays, 1500 videos/task)

One episode per array element; episode index `i` maps to `seed = i // 3`,
`variant = i % 3` (idempotent per episode id):

```bash
MUJOCO_GL=egl PYTHONPATH=/home/mzl7/scratch \
  python -m mujoco_franka_cloth_previews.generate_previews episode \
    --task poke_cloth --episode-index 0 --dataset-root /path/to/dataset
# after the array finishes, aggregate one task's metadata:
  python -m mujoco_franka_cloth_previews.generate_previews finalize \
    --task poke_cloth --dataset-root /path/to/dataset --expected-count 1500
```

The SLURM harness in `../franka_cloth_previews_dataset_2026-07-08_slurm/`
submits one array + dependent finalize per task:

```bash
COUNT=12 CONCURRENCY=4 ./submit_all.sh   # pilot (48 videos)
./submit_all.sh                          # full 1500/task (6000 videos)
```

## Domain randomization & outcome split

Per episode the pipeline randomizes:

- **cloth color/texture** (`tasks._random_cloth_appearance`) — random HSV rgba,
  40% flat / 35% checker / 25% gradient;
- **cloth material** (`materials.PRESETS[variant]`) — 5 physics presets;
- **arm/hand pose** (`tasks._jitter_base`, `_jitter_endpoints`) — base xy/yaw
  jitter + approach/hover-height jitter;
- **table color, background/skybox color, floor color, light intensity**
  (`scene_builder.random_scene_appearance`), keyed on `(task, seed, variant)`.

Each episode also carries a scripted **`outcome_branch`** (in metadata) that is a
deterministic function of the **seed** — a 4-category success/fail split:

| branch | share | meaning |
|---|---|---|
| `success` | 50% | task executes correctly |
| `near_miss` | 20% | grasp/poke closes just off the cloth (no effect) |
| `execution_failure` | 20% | grasp slips / poke stops short mid-motion |
| `wrong_action` | 10% | correct grasp, wrong target/direction |

Branch is keyed on seed (not variant) so all physics variants of one seed share
the scripted outcome (counterfactual bundle). `finalize` prints the realized
per-task branch histogram.

## Video format

Matches the existing preview convention: **480p per camera panel, 24 FPS,
121 frames (fps x duration + 1, inclusive of t=0 and t=duration), ~5 s.**
Each MP4 is a horizontal composite of `front | top | wrist` panels
(640x480 each).

## Outputs (`--out` root)

```
videos/    {episode_id}.mp4           front|top|wrist composite
videos/    {episode_id}__{view}.mp4   per-view captioned clips (main/top/wrist)
metadata/  {episode_id}.json          full per-episode metadata (incl. outcome_branch, scene)
states/    {episode_id}.npz           frame-aligned robot/cloth/contact state
scenes/    {episode_id}.xml           compiled scene MJCF
episodes.jsonl / episodes_{task}.jsonl  consolidated metadata (all-in-one / dataset finalize)
contact_sheet.png / contact_sheet_{task}.png  sampled labeled thumbnails
```

In dataset mode all artifacts (including `scenes/`) are written under
`--dataset-root`; in all-in-one mode they go under `--out`.

`episode_id = {task_id}__seed{SSS}__var{VV}`;
`counterfactual_bundle_id = {task_id}__seed{SSS}` — the same seed shares
identical scripted actions across physics variants, which differ only in
material preset (see `materials.PRESETS`).

### npz keys

- `time (F,)`, `cloth_keypoints (F,9,3)`, `keypoint_names (9,)`,
  `cloth_vertices (F,V,3)` (full flex vertex positions, float32)
- per arm (`arm0` or `left`/`right`):
  `{arm}_q, {arm}_dq, {arm}_q_target, {arm}_tau_cmd (F,7)`,
  `{arm}_ee_pos (F,3)`, `{arm}_ee_quat (F,4)`, `{arm}_ee_vel (F,6)`,
  `{arm}_gripper_width, {arm}_gripper_cmd, {arm}_gripper_force_cmd (F,)`
- contacts involving the cloth, sampled at every video frame (cap 32/frame):
  `contact_time, contact_frame, contact_pos, contact_force (world), contact_other`

## Physics / implementation notes

- Cloth: `flexcomp type="grid"` (dim=2 shell), stretch via edge equality,
  bending via the `mujoco.elasticity.shell` plugin. `stretch/bend/shear`
  stiffness metadata are thin-shell continuum values derived from
  (young, poisson, thickness) — nominal, not fitted.
- Robot: Menagerie Panda position-servo actuators; Cartesian scripted
  waypoints tracked with damped-least-squares differential IK
  (`control_mode=joint_position_servo`, `action_mode=cartesian_ee_waypoints_diff_ik`).
  Full 7-dof `q/dq/q_target/tau_cmd` recorded — no faked robot state.
- **Proxy grasp**: cloth vertex welded to the hand body by a `connect`
  equality toggled at scripted times, synchronized with gripper commands
  (`grasp_mechanism=equality_connect_proxy` in metadata).
- T-shirt is a rectangular cloth with collar/sleeve visual markers
  (`tshirt_proxy=true`).

## TODOs (preview -> dataset)

- Replace equality-connect proxy grasp with real friction-based contact
  grasping (close fingers on cloth, no weld).
- True T-shaped flex mesh for the T-shirt task.
- Cloth self-collision tuning + higher-resolution grids.
- Per-episode domain randomization of lighting/textures — **done** (cloth
  color/texture, table/background/floor color, light intensity).
- Measured (physics-based) success detector to cross-check the scripted
  `outcome_branch` labels.
- zarr option for vertex trajectories if episodes get long.
