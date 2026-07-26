# MuJoCo Arm Gripper Ball Preview

This folder is a copied, gripper-only slice of
`/scratch/zl664_yale/hs2272/simple_phys_data/3d/demo_mujoco`.

It generates UR5e + Robotiq 2F-85 gripper previews for an incoming ball. It does
not pack Lance/HDF5 datasets. The only generated outputs are temporary scene
JSON, temporary render shards, preview frames, contact sheets, and MP4 videos.

## Run

```bash
cd /scratch/zl664_yale/world_model_robotics
sbatch --export=ALL,MODE=videos demo_mujoco_arm_gripper/job_arm_gripper_ball_check.sh
```

## Local Example

From this directory, generate one isolated `arm_gripper_ball` example with:

```bash
bash examples/run_arm_gripper_ball_example.sh
```

This writes a scene JSON, render shard, preview frames, and contact sheet under:

```text
examples/example_arm_gripper_ball_run/
```

To also export MP4 previews, run:

```bash
MODE=videos bash examples/run_arm_gripper_ball_example.sh
```

Outputs are written under:

```text
demo_mujoco_arm_gripper/outputs/arm_gripper_ball/
demo_mujoco_arm_gripper/_staging/shards/arm_gripper_ball/v0/
demo_mujoco_arm_gripper/preview/arm_gripper_ball/
```

The MP4 files are:

```text
demo_mujoco_arm_gripper/preview/arm_gripper_ball/videos/episode_000.mp4
demo_mujoco_arm_gripper/preview/arm_gripper_ball/summary.mp4
```

Use `MODE=frames` to write only PNG preview frames/contact sheets, or
`MODE=mount` to render Robotiq mount debug markers.

## Franka End-Effector Showcase

To generate short comparison videos of objects/tools mounted to the Franka
wrist, run:

```bash
bash examples/franka_end_effector_examples/run.sh
```

This writes one MP4 per attachment under:

```text
examples/franka_end_effector_examples/output/
```

The available attachments include the native Panda hand, Robotiq 2F-85,
suction cup, tray/scoop, paddle/pusher, hook/probe, multi-finger hand sketch,
and sensor/tool head.

## Main Files

- `generate.py`: samples gripper-ball episodes and writes `outputs/.../scene.json`.
- `sim_support.py`: builds the UR5e + Robotiq MuJoCo model and contains grasp/latch logic.
- `render.py`: renders scene JSON into temporary HDF5 shards.
- `export_preview_frames.py`: writes PNG frames and `contact_sheet.png`.
- `export_preview_videos.py`: writes MP4 videos from temporary render shards.
- `third_party/mujoco_menagerie/universal_robots_ur5e`: UR5e assets.
- `third_party/mujoco_menagerie/robotiq_2f85`: Robotiq gripper assets.

## Interception Subfamily Pipeline

The arm-gripper pipeline now includes the direct projectile task plus five
additional Earth-gravity interception subfamilies:

```text
arm_gripper_ball
arm_gripper_ramp_projectile
arm_gripper_floor_bounce
arm_gripper_wall_rebound
arm_gripper_multi_rebound
arm_gripper_arbitrary_bounce
```

Each generated episode uses gravity `[0, 0, -9.81]`, stores the subfamily label,
success/failure plan, ball/robot/tool initial state, surface geometry,
contact/material parameters, key event frames/timestamps, per-frame ball state,
per-frame end-effector state, and validation metadata. Success examples may be
retained gripper catches (`outcome: "capture"`) or physically meaningful
post-surface tool interceptions (`outcome: "block"`). Failure examples are
sampled with larger timing/spatial offsets and must miss after the required
surface contacts.

For every arm-gripper interception episode, `render.py` keeps the legacy
single-view `pixels` dataset and also writes:

```text
pixels_views[frame, view, height, width, rgb]
camera_view_names = ["side_view_a", "side_view_b", "on_device"]
```

Run a small two-outcome example set for all subfamilies with:

```bash
bash examples/run_interception_subfamilies_example.sh
```

This writes scene JSON under `examples/interception_subfamilies_run/outputs`,
HDF5 render shards under `_staging/shards`, and per-view MP4 previews such as
`episode_000_side_view_a.mp4`, `episode_000_side_view_b.mp4`, and
`episode_000_on_device.mp4` under each task's `preview/.../videos` directory.

The `metadata.validation` block checks Earth gravity, projectile-motion
segments, expected surface contact order, penetration over the critical event
window, label/outcome agreement, gripper catch or block consistency, three-view
camera presence, and ball visibility/trackability.

## RoboCasa Robotiq Visibility Checks

`examples/generate_robocasa_robotiq_thick_pad_interception_example.py` renders
three synchronized views: two oblique side views (`main_camera` and
`closeup_camera`) plus the on-device `wrist_rgb` camera. Before rendering, the
script runs a cheap pre-simulation of the selected scenario and places the two
side cameras from the resulting ball trajectory envelope, so launch, contact,
interception, rebound/slip, and final outcome remain in frame.

Each generated `metadata.json` contains `camera_framing` and
`visibility_validation`. The validation projects the ball into every camera,
checks visible and trackable frame ratios, verifies key event frames, and rejects
the rollout for resampling if framing fails.
