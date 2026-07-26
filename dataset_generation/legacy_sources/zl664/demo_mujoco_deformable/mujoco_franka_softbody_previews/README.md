# mujoco_franka_softbody_previews

Preview-generation pipeline for the third Franka-deformable dataset family:
**plush / sponge / soft-body packing** with a Franka Panda.

This is a *preview* pipeline (runnable MP4s, stable scripted interactions,
honest metadata), not the final dataset generator.

## Tasks

| task_id | behavior | contact realism |
|---|---|---|
| `press_release` | fingertips press the sponge, hold, retract; recovery measured | real contact |
| `squeeze_with_gripper` | fingers flank the sponge and close to a width target | real finger-pad contact (`squeeze_proxy=false`) |
| `push_toward_box_opening` | closed gripper pushes the sponge toward an open-front box | real contact |
| `compress_and_close_lid` | compress sponge below a box rim, retract, hinged lid closes | real dynamic hinged lid, force-bounded actuator (`lid_proxy=false`) |

## Soft body

`softbody_model_type = "mujoco_flexcomp_grid"`: MuJoCo volumetric flex
(`flexcomp type="grid" dim="3"`), tetrahedral FEM with built-in continuum
elasticity. Default object 9x8x6 vertices (432 nodes, 1680 tets), outer size
~0.098 x 0.080 x 0.061 m; the squeeze task uses a pinch-scale 5x4x4 object
(~0.048 x 0.039 x 0.041 m). `plush_proxy = true` (rectangular sponge proxy,
no plush mesh). Vertex spacing is kept below the fingertip pad width (17 mm)
and the object the gripper squeezes is sized to the pad face — otherwise the
pads wedge between vertex rows / knife into a large sponge and the hand
appears to pass through a rigid object.

Material presets (counterfactual physics variants): `soft_plush`
(E=2e4 Pa, nu=0.30), `medium_sponge` (E=8e4, nu=0.35), `stiff_foam`
(E=3e5, nu=0.40).

### MuJoCo 3.3.1 stability caveats (empirical, important)

- Built-in flex `<elasticity damping>` > 0 is **unconditionally unstable**
  (BADQACC + silent auto-reset loops, even in free fall). Elasticity damping
  is set to 0; dissipation is implemented as flex `<edge damping>` and the
  low/medium/high damping axis maps to edge damping values.
- Explicit elastic forces bound the stable timestep, so `dt_sim` varies per
  preset: 2.5e-4 / 2e-4 / 1.25e-4 s. Bundles share actions but not dt.
- MuJoCo contact stiffness is inertia-scaled; ~0.1 g flex vertices make the
  default solref (0.01 s) so soft that rigid bodies tunnel through the flex.
  The flex contact solref timeconst is therefore set to 2*dt_sim (stability
  floor) — recorded as `contact_solref` and in the implementation notes.
- `poisson = 0.45` at high Young's modulus diverges at every dt tried;
  presets cap poisson at 0.40.
- The runner detects MuJoCo's silent auto-reset (monotonic `data.time`
  check) and fails the episode loudly instead of producing a corrupt video.

Restitution and plasticity are **not implemented** (see
`parameter_implementation_notes` in every episode JSON).

## Robot

Full Menagerie Panda (`embodiment = "single_franka"`), joint position-servo
actuators driven by damped-least-squares differential IK over scripted
Cartesian waypoints. One documented modification: the gripper servo
(actuator8) gains are scaled 5x via MjSpec — the stock kp=100 tendon servo
saturates at ~2.4 N pinch, too weak to visibly squeeze foam
(`gripper_servo_modification` in robot metadata). Wrist yaw is held at the home yaw (fingers travel along
world x; the squeeze object is laid out to match). Full q/dq/tau/EE state is
recorded; `estimated_ee_wrench` is a least-squares solve from
`qfrc_constraint` (approximate).

## Run

```bash
cd /gpfs/radev/project/sous/zl664/demo_mujoco_deformable
MUJOCO_GL=egl /gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper/.venv/bin/python \
  -m mujoco_franka_softbody_previews.generate_previews \
  --out mujoco_franka_softbody_previews/outputs/preview_001 \
  --num-seeds 2 --physics-variants 3 \
  --width 640 --height 480 --fps 24 --duration 5.0
```

Videos are 480p per camera panel (front | top | wrist), 24 FPS, 121 frames
(~5 s). Episode naming: `{task}__seed{SSS}__var{VV}`;
`counterfactual_bundle_id = {task}__seed{SSS}` (seed fixes layout + scripted
action; variants change only material physics).

## Outputs

- `videos/*.mp4` — composite front|top|wrist previews
- `metadata/*.json` + `episodes.jsonl` (authoritative; parquet skipped when
  pandas is unavailable)
- `states/*.npz` — frame-aligned robot state, flex vertex positions, tet
  indices, keypoints, COM, bbox, tet-mesh volume, compression ratio,
  contacts, box occupancy / lid state where applicable
- `contact_sheets/contact_sheet.png` — labeled thumbnail per episode
- `scenes/*.xml` — compiled per-episode MJCF (in the package dir)
