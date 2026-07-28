# projectile_smoke — loss-curve validation of the projectile physics embedding

Implements `~/scratch/wan_projectile_smoke_train_plan.txt` (2026-07-15).
Data: ONLY `projectile_ball_catch_robocasa_kitchen_scenes_rollouts` (4×750
episodes; style058 preview skipped). Deliverable: one 2k-step GPU run whose
curves show the physics embedding trains (recon aux down, EMA train L_fm
down) and, via the paired shuffle-gap eval, whether it is *consumed*.

This copy is **self-contained**: the cloth-plan (`physics_finetune`) modules it
depends on are vendored in rather than imported from a sibling checkout —
`models/wan_wrapper.py`, `models/lora.py`, `training/flow_match.py`,
`tests/test_flow_convention.py`, plus `mujoco_bridge.py` (frame resampling)
from the Wan2.1 repo root. All were copied verbatim on 2026-07-28 from the
versions that produced the recorded runs; only their `sys.path`/import lines
were rewired. It also reuses the existing latent/T5 caches (all 3,000
projectile episodes were already encoded during the cloth work; captions
verified byte-identical). Never edits `wan/`.

## External prerequisites (not vendored)

- **The upstream Wan2.1 repo** (`wan/` package: DiT, VAE, T5, schedulers).
  Scripts locate it via the `WAN21_ROOT` env var, defaulting to
  `/gpfs/radev/scratch/sous/mzl7/Wan2.1`.
- **Wan2.1-T2V-1.3B weights** — path set by `model_dir` in the configs
  (default `/gpfs/radev/scratch/sous/mzl7/wan_models/Wan2.1-T2V-1.3B`).
- **Python env** with torch/imageio/opencv/yaml/matplotlib
  (`~/.venvs/wan21` on the cluster).
- **Caches + datasets** at the absolute paths below (cluster-specific).

SLURM scripts must be submitted **from this repo's root** (they `cd
"${SLURM_SUBMIT_DIR}"`), e.g. `sbatch projectile_smoke/slurm/tests.sbatch`.

## Pipeline

```
P0 (CPU, done inline)
  data/captions.py           captions_blind.jsonl (+ verifies vs cached T5)
  data/splits.py             2400/300, stratified family x failure_mode,
                             physics-identity twin assertion (no twins exist
                             in this dataset - verified)
  data/physics_vec.py        14-d camera-frame vector + degenerate-dim guard
                             (nothing dropped; ball_radius std/scale ~1e-2)
  data/check_reprojection.py BLOCKING go/no-go, writes PASS marker
P1 (GPU)
  sbatch projectile_smoke/slurm/tests.sbatch
    flow sign test / encoder+LoRA test / VAE round-trip (PASS marker) /
    overfit + save-resume test
P2 (GPU)
  sbatch projectile_smoke/slurm/smoke.sbatch     # configs/smoke_lora.yaml
  sbatch projectile_smoke/slurm/smoke.sbatch projectile_smoke/configs/smoke_frozen.yaml
  eval/plot_curves.py --run-dir <out_dir>        # -> curves.png deliverable
```

Cache: `/gpfs/radev/scratch/sous/mzl7/wan_projectile_smoke_cache/`
(latents/, t5_blind/ symlink into wan_physics_cache). Runs:
`/gpfs/radev/scratch/sous/mzl7/wan_projectile_smoke_runs/`.

## Dataset facts that differ from the plan's assumptions (verified 2026-07-15)

- **No opposite-camera twins**: all 3,000 episodes have unique physics, so
  the 300 val episodes are 300 independent clusters (better statistical
  power than the plan's ~150 estimate). The twin-leakage assertion in
  splits.py and the cluster machinery still exist and would engage on a
  twinned regeneration.
- **ball_radius survives the degenerate-dim guard** (std/scale 1.1e-2 >
  1e-3): it is genuinely sampled, not constant. Final vector stays 14-d.
- `events.release_frame` is `round(release_time_s*fps)` of a continuous
  release (U(0.18,0.34)s); `first_contact_frame==0` means launcher-rest
  contact. `ballistic_intercept_time_s` is episode-referenced (verified,
  median residual 17 ms). The reprojection check fits a sub-frame release
  offset and passes on all 4 families (median 0.9-6.6 px, close-time 3D
  residual 6-9 mm).
- LoRA r=8 on cross-attn q,k,v,o of all 30 blocks is **2.95M** params (the
  plan's "~6M" double-counted).
