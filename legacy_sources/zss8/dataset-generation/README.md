# Franka Reactive Ball-Catch World-Model Dataset (MuJoCo)

Large-scale simulation data for a **physics-aware robotic world model**: a Franka
Emika Panda **reactively intercepts a falling ball**. The ball is released at t=0
from a high drop point and is already falling; the arm starts at home and reaches
the **predicted ballistic intercept** (arriving just before the ball). This is the
corrected behavior — the ball drops first, *then* the arm reaches to catch — not
"arm pre-positions, then ball drops".

Built in MuJoCo (Earth gravity g=9.81), in the PI's module style. Episodes span
the success/failure taxonomy the downstream video/world-model post-training + RL
needs (action-conditioned causality requires failures).

Family **F1**, subfamily **F1_A_centered_vertical_drop** (centered, near-vertical
drop, current focus).

## Taxonomy (per episode)
| branch | share | behavior | outcome |
|--------|-------|----------|---------|
| `success`           | 50% | reach intercept, pinch, hold to end | caught (~92% reliable) |
| `spatial_near_miss` | 20% | intercept target offset laterally | ball falls beside the hand |
| `contact_failure`   | 20% | reach path but pads too wide / late / shallow | touch then slip |
| `wrong_action`      | 10% | idle / random joints / wrong-way reach | never secures |

Measured split ≈ 46% success / 54% failure. `branch` (intended) and `outcome`
(measured) are both recorded.

## Environments (balanced 1/3 each)
`clean_lab`, `office` (clean lab/office backgrounds, tabletop-mounted robot), and
`robocasa_kitchen` (real RoboCasa fixtures/objects imported read-only from
`/gpfs/radev/project/sous/mzl7/robocasa/...`, discovered at runtime so kitchens
vary clip to clip).

## Output — LeRobotDataset v3-style (in scratch, 10 TB)
Everything is written under **`/gpfs/radev/scratch/sous/zss8/franka_catch/`**
(project space is only 1 TB; scratch is 10 TB).

```
<root>/
  meta/info.json            feature schema, fps, control_hz, counts
  meta/episodes.jsonl       one rich record per episode (schema below)
  meta/tasks.jsonl
  meta/dataset_summary.json split / outcome / variant counts
  meta/rich/episode_*.json  per-episode rich metadata sidecars
  data/chunk-000/episode_XXXXXX.parquet   per-frame state/action @ 120 Hz
  videos/observation.images.main/chunk-000/episode_XXXXXX.mp4   832x480 H.264 30 fps
  videos/observation.images.side/chunk-000/episode_XXXXXX.mp4
```

- **2 synchronized views** (`main_camera`, `side_camera`) at ~45–60° left/right,
  832×480, H.264 MP4, 30 fps.
- **Proprio parquet** at 120 Hz: `observation.state` (18: arm q7, fingers 2, ee 3,
  ball pos 3, ball vel 3), `action` (8: arm q cmd 7 + gripper cmd), `timestamp`,
  `video_frame_index` (maps control rows → video frames), `next.done`.
- Rich per-episode metadata (family/subfamily/branch/outcome/failure_mode/views/
  object_physics/randomization/events/metrics/...).

## Run
```bash
cd /gpfs/radev/project/sous/zss8/dataset-generation

# 10-sample smoke test (GPU) for qualitative inspection
sbatch slurm/smoke.sbatch                          # -> scratch/.../smoke10

# Full 1500-episode dataset (array of 4 shards, 2 concurrent on gpu_devel)
sbatch slurm/dataset.sbatch                         # -> scratch/.../f1a_centered_v1/shardNNN
# then merge the shards into one dataset:
PYTHONPATH=demo_mujoco_arm_gripper .venv/bin/python -m franka_catch.merge \
    --root /gpfs/radev/scratch/sous/zss8/franka_catch/f1a_centered_v1 --shards 4
```

Rendering is EGL on a GPU node (`MUJOCO_GL=egl`; set in `slurm/env.sh`). OSMesa
CPU rendering is unavailable on this cluster; physics-only sim runs on CPU.

See `MEMORY.md` for the full build log, design decisions, and cluster notes.
