# franka_rope — Franka rope/cable/string deformable dataset (zss8)

Scaled, background-varied successor to zl664's `mujoco_franka_rope_previews`
preview pipeline. **All code and data live under `zss8` only**; the Franka
Panda MJCF is referenced read-only from zl664's MuJoCo Menagerie checkout
(never copied or modified).

- **Code**: `/gpfs/radev/project/sous/zss8/dataset-generation/franka_rope/`
- **Data**: `/gpfs/radev/scratch/sous/zss8/franka_rope/`
- **venv**: `/gpfs/radev/project/sous/zss8/dataset-generation/.venv` (mujoco 3.10)

## What's new vs the preview pipeline

1. **Background domain randomization** (`backgrounds.py`) — the single
   hard-coded studio look is replaced by 10 sampled scene themes, each with
   per-episode colour/texture/light jitter. Requested "plain" and "kitchen"
   are both included:

   `plain_studio`, `white_cyclorama`, `kitchen`, `warehouse`, `wood_workshop`,
   `lab`, `office_desk`, `outdoor_patio`, `garage`, `bright_tabletop`.

   A theme controls only **cosmetics**: skybox gradient, floor texture/repeat/
   reflectance, table *appearance* (colour + optional marble/plank/stone
   checker), headlight, light rig, and optional **visual-only** backdrop walls
   and props (`contype=0 conaffinity=0` — they can never collide). The table
   **geometry and friction, the rope, the arm, the cameras and gravity are
   identical across every theme**, so a counterfactual bundle rendered under
   "kitchen" vs "plain" has bit-identical dynamics — the model learns rope
   physics, not a backdrop.

2. **Five extra dynamics-rich tasks** on top of the four spec tasks.

3. **More rope materials** (`materials.py`, 8 presets) and **shard/seed/variant
   CLI** (`generate.py`) for SLURM fan-out.

## Tasks

| task_id | description | task metadata |
|---|---|---|
| `tug_endpoint` | grasp one endpoint, tug up ~0.3 m, hold, release | `release_time` |
| `drag_endpoint` | grasp one endpoint, drag along a curved low path | `release_time` |
| `wrap_around_post` | orbit grasped endpoint ~210° around a post | `post_pose_world`, `winding_number_around_post` |
| `thread_through_ring` | lower a dangling tail through a fixed horizontal ring | `ring_pose_world`, `ring_radius`, `ring_passage_success` |
| `shake_wave` | lift endpoint, oscillate laterally → traveling waves | `wave_tip_amplitude` |
| `lift_and_drape` | carry endpoint up and over a bar so the rope drapes | `bar_pose_world`, `drape_success` |
| `coil_on_table` | spiral the endpoint inward to coil the rope | `coil_center_world`, `coil_turns` |
| `twirl_overhead` | lift the endpoint and move it in a circle so the rope trails | `twirl_center_world`, `twirl_tip_radius` |
| `sweep_aside` | non-prehensile: push the rope sideways, gripper closed | `sweep_midpoint_displacement` |

All tasks use a full Franka (`embodiment=single_franka`), position-servo
actuators, damped-least-squares differential IK tracking scripted Cartesian
waypoints. Endpoint "grasp" is an equality-connect weld (not a friction pinch);
`sweep_aside` is genuinely non-prehensile (no weld ever activates).

## Rope model

`rope_model_type = mujoco_composite_cable`: MuJoCo composite `type="cable"`
(rigid capsule links + ball joints) with the `mujoco.elasticity.cable` plugin
for bend/twist. 8 presets in `materials.PRESETS` (medium_rope, light_string,
stiff_cable, thick_hemp, nylon_cord, red_paracord, thin_wire,
green_garden_hose) vary length/radius/density/stiffness/friction/colour.
Honesty notes (`stretch_stiffness=null` — inextensible by construction; bend/
twist are derived EI/GJ; grasp is a weld) are emitted per episode under
`parameter_implementation_notes`, unchanged from the preview pipeline.

## Usage

Local smoke (from `dataset-generation/`):

```bash
MUJOCO_GL=egl .venv/bin/python -m franka_rope.generate \
    --out /gpfs/radev/scratch/sous/zss8/franka_rope/smoke \
    --seeds-per-task 1 --variants 2 \
    --width 640 --height 480 --fps 24 --duration 5.0
```

Force one theme (e.g. to render a held-out eval set): `--theme kitchen`.
Subset of tasks: `--tasks shake_wave lift_and_drape`.

Scale-up on SLURM:

```bash
cd franka_rope
sbatch slurm/rope_smoke.sbatch                       # verify all 9 tasks
sbatch slurm/rope_dataset.sbatch                     # 9x60x3 = 1620 eps, 12 shards
# bigger: sbatch --export=SEEDS_PER_TASK=100,VARIANTS=4 slurm/rope_dataset.sbatch
```

Backgrounds are chosen per `(task, seed)` bundle, so a bundle's material
variants share a theme (clean counterfactuals) while seeds/tasks span all
themes. Each shard writes `episodes<NNN>.jsonl` / `.parquet`; concatenate the
per-shard jsonl for the full manifest.

## Outputs (`--out` root)

```
videos/<camera>/ {episode_id}.mp4     one mp4 PER VIEW (split mode, default)
metadata/        {episode_id}.json    full per-episode record (incl. scene block)
states/          {episode_id}.npz     trajectories (see below)
scenes/          {episode_id}.xml     compiled MJCF actually simulated
contact_sheets/  contact_sheet*.png   snapshot grid (theme + material + label)
episodes*.jsonl / episodes*.parquet   consolidated metadata (all views)
episodes*__<camera>.jsonl             per-view manifest (single-view dataset)
```

### Video mode: split (default) vs composite

`--video-mode split` (default) writes **one mp4 per camera view** into a
per-camera subdirectory — `videos/front/`, `videos/top/`, `videos/arm0wrist/` —
so each view is its own standalone dataset of videos, and per-view manifests
`episodes*__<camera>.jsonl` list just that view's clips. Physics, `states/`,
`metadata/` and `scenes/` are shared across views (only the pixels differ).
Per-episode JSON records `video_mode`, `videos_by_camera` (camera → mp4 path)
and a `video_paths` list of every view.

`--video-mode composite` restores the legacy single `videos/{episode_id}.mp4`
with `front|top|wrist` hstacked side by side.

`episode_id = {task_id}__seed{SSS}__var{VV}`;
`counterfactual_bundle_id = {task_id}__seed{SSS}`.

Per-episode JSON adds a `scene` block: `background_theme` + a
`background_summary` (sampled skybox/floor/table colours, light count, whether
a backdrop was drawn) + a note that the background is cosmetic-only.

### npz keys

`time (F,)`, `rope_centerline_points_world (F,N,3)`, `rope_keypoints (F,5,3)`
(endpoint_0, quarter_1, midpoint, quarter_3, endpoint_1),
`endpoint_positions_world (F,2,3)`; `arm0_q/dq/q_target/tau_cmd (F,7)`,
`arm0_ee_pos/quat/vel`, `arm0_gripper_width/cmd/force_cmd`; rope contacts
sampled per frame (cap 32): `contact_time/frame/pos/force/other`.

## TODOs (inherited)

- Real friction pinch grasp of the rope (replace the endpoint weld).
- Tunable stretch (segmented capsule chain + slide-joint springs).
- Rope-rope knotting; multi-post figure-8 wrapping.
