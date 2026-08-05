# Projectile Ball Catch — self-contained export bundle

Everything needed to regenerate the `projectile_ball_catch_robocasa_kitchen_scenes_rollouts`
dataset on another cluster. The folder layout is load-bearing: the scene code resolves its
dependencies as **siblings of the scripts package** (`repo_root()` = the parent of the package
directory, i.e. this bundle root). Keep the top-level structure exactly as is:

```
projectile_ball_catch_export/            <- bundle root; put this dir on PYTHONPATH
├── projectile_ball_catch_robocasa_kitchen_scenes_scripts/   <- the generator package
├── robocasa_full/                       <- RoboCasa code + models/assets (fixtures, textures, scene YAMLs)
├── robosuite/                           <- imported by RoboCasa (added to sys.path by the scripts)
├── demo_mujoco_arm_gripper/
│   └── third_party/mujoco_menagerie/franka_emika_panda/     <- Franka Panda MJCF + meshes
├── slurm/                               <- portable sbatch templates
└── logs/                                <- sbatch output lands here (relative to submit dir)
```

Do NOT pip-install robosuite/robocasa in the target env — the scripts prepend the vendored
copies to `sys.path` (`yaml_scene._prepare_native_import_paths`) and the vendored assets must
match the vendored code.

## Environment

Reference env (what generated the original dataset): Python 3.11.15 with
mujoco 3.3.1, numpy 2.2.5, scipy 1.15.3, pandas 3.0.3, pyarrow 24.0.0,
Pillow 11.3.0, PyYAML 6.0.3, imageio 2.37.3, imageio-ffmpeg 0.6.0.
(`trimesh` + `pygltflib` are only needed for the optional GLB exporter.)

Required environment variables:

```bash
export PYTHONPATH=/path/to/projectile_ball_catch_export
export MUJOCO_GL=egl        # headless GPU rendering; needs an allocated GPU + EGL libs
```

## Smoke test (run this first on the new cluster)

```bash
cd /path/to/projectile_ball_catch_export
PYTHONPATH=$PWD MUJOCO_GL=egl python -m projectile_ball_catch_robocasa_kitchen_scenes_scripts.run_rollout \
  --seed 0 --layout-id 11 --style-id 20 --smoke --output-dir /tmp/pbc_smoke
```

The third-from-last printed line (also `scene_resolution_backend` in
`/tmp/pbc_smoke/metadata.json`) must be `robocasa_native`. If it says `yaml_fallback`,
the kitchen rendered as untextured boxes; the exact import/asset error is in the
`scene_resolution_warning` field of `metadata.json` — fix that and re-run.

## Mass generation (SLURM)

One array task per episode; 4 families × 750 episodes. Adjust `#SBATCH --partition`
to your cluster. Submit from the bundle root:

```bash
cd /path/to/projectile_ball_catch_export
export DATASET_ROOT=/path/to/output_rollouts
export PYTHON_BIN=/path/to/env/bin/python     # optional; defaults to `python` on PATH

for FAMILY in style020_seed0 style020_seed0_opposite_camera style055_seed1000 style055_seed1000_opposite_camera; do
  FAMILY=$FAMILY sbatch --array=0-749 --job-name=gen_$FAMILY slurm/generate_family_array.sbatch
done

# after all arrays finish, build each family's metadata.parquet index:
for FAMILY in style020_seed0 style020_seed0_opposite_camera style055_seed1000 style055_seed1000_opposite_camera; do
  FAMILY=$FAMILY sbatch --job-name=fin_$FAMILY slurm/finalize_family.sbatch
done
```

Episodes are deterministic in `(family, episode_index)`, so failed array tasks can be
resubmitted individually (`--array=17,203`) without affecting the rest.

## What was trimmed from the vendored repos

To keep the bundle small (~3.7 GB instead of ~8 GB), these were excluded — none are used
by this pipeline:

- `robocasa_full`: `.git`, `docs`, `tests`, `robocasa/models/assets/groot_dataset_assets/`,
  and most of `robocasa/models/assets/objects/lightwheel/` (~1.4 GB of manipulable kitchen
  objects). Kept from `objects/`: `lightwheel/stool/` and `lightwheel/paper_towel_holder/` —
  the only categories the native KitchenArena places for layouts 11/31 + styles 20/55
  (verified empirically; anything missing surfaces as a `FileNotFoundError` in
  `scene_resolution_warning`).
- `robosuite`: `.git`, `docs`, `tests`
- `mujoco_menagerie`: everything except `franka_emika_panda/`

If you reuse this bundle for OTHER layout/style IDs and get `yaml_fallback` with a
`FileNotFoundError` under `objects/lightwheel/<category>/`, copy that category from the
original repo into the same relative path.

Known quirk: `yaml_scene.robocasa_assets_root()` has a hardcoded fallback path
`/home/mzl7/scratch/robocasa_full/...` from the original cluster. It is harmless here —
the primary lookup (`<bundle root>/robocasa_full/...`) always wins as long as the layout
above is preserved.
