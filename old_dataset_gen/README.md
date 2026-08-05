# old_dataset_gen — legacy dataset-generation scripts

Script-only copy of the legacy MuJoCo dataset-generation pipelines. The vendored
third-party trees, downloaded assets, generated datasets, and logs are **not in
git** — they are reproducible from the pins below, and the full runnable bundle
lives on the cluster at
`/scratch/zl664_yale/world_model_robotics/Dynamic-Robot-WM-main/old_dataset_gen/`.

## Contents

| Path | What it is |
|---|---|
| `projectile_ball_catch_export/` | Self-contained export bundle for the projectile-ball-catch dataset (see its `README.md` for env setup, smoke test, and SLURM mass generation) |
| `ball_rolling_dynamics_scripts/` | Passive ball-rolling corpus generator (velocity-only variation, v2 format, counterfactual grouping) |
| `ball_rolling_dynamics_example/` | One example rolling episode (LeRobot-style shards) |
| `configs/`, `slurm/`, `smoke_test_rolling_groups.sh` | Rolling-pipeline configs, sbatch templates, smoke test |
| `COUNTERFACTUAL_PROJECTILE_CATCH_PLAN.md` | Design doc for the counterfactual projectile-catch dataset |

## Vendored dependencies (not committed)

`projectile_ball_catch_export`'s layout is load-bearing: the scene code resolves
its dependencies as siblings of the scripts package. Three third-party trees sit
there in the runnable bundle and are pinned in
`projectile_ball_catch_export/VENDOR_PINS.json`. All three were diffed
byte-for-byte against their pinned upstream commits (2026-08-05): **zero local
modifications** — the pins fully reproduce them.

To materialize a runnable bundle from this git copy, inside
`projectile_ball_catch_export/`:

```bash
# robosuite (docs/ and tests/ can be pruned, they are unused)
git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
git -C robosuite checkout 85abee228d1c43ab1939bce33028099945d453b4

# robocasa code...
git clone https://github.com/robocasa/robocasa.git robocasa_full
git -C robocasa_full checkout be22d659b02db8f6d7f3a3c3edc742934fdcbaae
# ...plus its downloaded kitchen assets (~3.1G)
python robocasa_full/robocasa/scripts/download_kitchen_assets.py

# Franka Panda MJCF + meshes
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git /tmp/menagerie
git -C /tmp/menagerie sparse-checkout set franka_emika_panda
git -C /tmp/menagerie checkout accb6df40a9a1d1e49eff88157f6818b63a49335
mkdir -p demo_mujoco_arm_gripper/third_party/mujoco_menagerie
cp -r /tmp/menagerie/franka_emika_panda demo_mujoco_arm_gripper/third_party/mujoco_menagerie/
```

Do **not** pip-install robosuite/robocasa — the scripts prepend the vendored
copies to `sys.path`, and the vendored assets must match the vendored code
(see the bundle README).

Then follow `projectile_ball_catch_export/README.md`: the smoke test must report
`scene_resolution_backend: robocasa_native` (`yaml_fallback` means the kitchen
assets failed to load).
