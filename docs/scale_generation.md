# Diagnostic scale generation (fork)

This fork adds a sampled-scale generation pathway for the 11 review-executable
rigid leaves (P0a–d, F1a–d, F2a, F2c, F3b).  It reuses the canonical
plan-run → run-shard → finalize-run pipeline, writer, QC, and evaluators
unchanged.  Every episode is diagnostic: `training_eligible=false`,
`release_eligible=false`, `release_state=blocked`.  Release-hour accounting is
structurally zero; consume the data by filtering per-episode `physics_qc_pass`.

## Campaign targets

`configs/scale/hours_v1.yaml` encodes the target unique video hours per leaf
(6.25 h P0a–d, 31.25 h F1a–d, 35 h F2a/F2c, 50 h F3b — 270 h ≈ 508,808
episodes in 232 blocks).  Inspect with:

```bash
uv run --frozen python tools/scale_generation_driver.py plan-hours
```

## What is new

- `sampled_scale` initial-state mode: versioned bounded samplers per leaf
  (`scenarios/_rigid_shared.py`), compiled at the 1200 Hz reference rate,
  each episode carrying a hash-bound sampling contract.
- `common/scale_suite.py`: deterministic `ScaleSuiteCase` minting under an
  independent master seed/UUID namespace (`SCALE_MASTER_SEED = 20260721`).
  The fixed 120-case review plan hash is unchanged (test-pinned).
- `common/scale_execution.py` + `common/scale_prepare_worker.py`: a scale
  bridge parallel to the fixed-review bridge; run-shard dispatch selects the
  executor by declaration schema, and each bridge rejects the other's
  declarations.
- `common/scale_generation.py` + `tools/scale_generation_driver.py`: block
  planning (2,048/4,096 episodes per immutable dataset), isolated per-episode
  worker execution, sealing, strict QC.
- Path guard: exactly one writable subtree,
  `/gpfs/radev/scratch/sous/mzl7/dynamic_rollouts`; the rest of the scratch
  root stays write-protected.
- F2f note: its surface-admission evidence is content-bound to upstream
  generator bytes and is honestly revoked in this fork (tests skip); F2f is
  not part of scale generation.

## Operating procedure

Stage 0 — calibration (GPU node, required before any fan-out):

```bash
export MUJOCO_GL=egl
uv run --frozen python tools/scale_generation_driver.py calibrate \
  --leaf F1a --episodes 100 --shards 4 \
  --output-root /gpfs/radev/scratch/sous/mzl7/dynamic_rollouts
```

Repeat per leaf.  Gates before mass launch: >=95% physics-QC pass and >=85%
intended-vs-actual agreement on nominal branches; use the reported
`run_seconds_per_episode` and `bytes_per_episode` to size arrays and storage.

Stage 1 — plan blocks (CPU):

```bash
sbatch --array=0-27 slurm/scale_prepare.sbatch F1a \
  /gpfs/radev/scratch/sous/mzl7/dynamic_rollouts
```

Stage 2 — generate (A40 GPUs, one array task per shard, resumable):

```bash
sbatch --array=0-15%8 slurm/scale_generate_a40.sbatch \
  /gpfs/radev/scratch/sous/mzl7/dynamic_rollouts/F1a/block-0000
```

Stage 3 — finalize + strict QC (CPU):

```bash
sbatch slurm/scale_finalize.sbatch \
  /gpfs/radev/scratch/sous/mzl7/dynamic_rollouts/F1a/block-0000
```

Aggregate: `tools/scale_generation_driver.py report --output-root ...`.

## Testing

`tests/unit/test_scale_samplers.py`, `test_scale_suite.py`, and
`test_scale_execution_bridge.py` cover sampler bounds/determinism, minting,
gating, dispatch separation, and pin the upstream review-plan hash
(`48ee988f…dddc99b`).  Render-dependent test modules require a GPU/EGL node
(they abort on login nodes); run them inside a GPU job.
