# generalized_physics.real

Real-corpus campaigns against the actual Wan 2.1 backbone. One shared
training core, three corpus families layered on top, distinguished by
filename prefix:

| prefix | corpus | reader / records | cache builder | campaign driver |
|---|---|---|---|---|
| *(none)* | `f1_10h` manipulation (first campaign) | `f1_episode`, `f1_groups`, `f1_records` | `encode_cache` | `run_campaign` (phases 0–3) |
| `pbc_` | projectile-ball-catch (`pbc_state_groups`, state-conditional counterfactual groups) | `pbc_episode`, `pbc_records` | `pbc_encode_cache` | `pbc_run_campaign` |
| `roll_` | passive ball-rolling (`v1_velocity_groups`, velocity-only variation) | `roll_episode`, `roll_records` | `roll_encode_cache` | `roll_run_campaign` (phases 1–3) + `roll_oracle_ceiling` (pre-flight ceiling gate) |

Each campaign reuses its predecessor instead of forking it, so the layering is

    roll_*  →  pbc_*  →  f1 / un-prefixed  →  shared core

`pbc_encode_cache` imports `f1_records.fit_normalizer` and `prompts.bucket_key`;
`roll_episode` / `roll_records` re-export `pbc_episode.read_frames` and
`pbc_records.camera_matrix`; `roll_run_campaign` drives the same
`phase{1,2,3}_real` chain under a `roll_`-prefixed cache/run namespace, with
the muffling-diagnosis fixes (ROI-weighted fm loss, high-σ oversampling,
held-out paired σ-grid eval). Un-prefixed modules are therefore either
**f1-specific** — the first campaign predates the prefix convention
(`encode_cache`, `run_campaign`, `f1_*`, `prompts`) — or **corpus-agnostic
core**:

| module | role |
|---|---|
| `paths` | cluster-local paths + per-arm checkpoint specs; everything path-shaped lives here |
| `wan_loader` | real Wan DiT / VAE / umT5 loading; enforces one vendored `wan` tree per process |
| `backend` | injects the real DiT into the stock `training/` phase modules |
| `flow_match` | rectified-flow utilities in Wan's convention |
| `phase1_real`, `phase2_real`, `phase3_real` | oracle conditioning / student distillation / substitution, resume-capable |
| `cache_io` | on-disk latent cache → the dict contract of `data/counterfactual_dataset.build_cache` |
| `instrument` | `PhaseRecorder` patch-while-active logging for the DiT-free phases |
| `t5_cache` | precomputed umT5 embeddings for the synthetic prompt table |
| `plots` | campaign figures/tables across arms |

## Entry points

All run as `python -m generalized_physics.real.<module>`; the slurm drivers
live in `mzl7/physics_wan/slurm/`:

| module | slurm |
|---|---|
| `encode_cache` | `10_encode.sbatch` |
| `t5_cache` | `15_t5_cache.sbatch`, `pbc_15_t5.sbatch`, `roll_15_t5.sbatch` |
| `run_campaign` | `20_train.sbatch` |
| `pbc_encode_cache` | `pbc_10_index.sbatch`, `pbc_11_encode.sbatch` |
| `pbc_run_campaign` | `pbc_20_train.sbatch` (+ `pbc_99_smoke.sbatch`) |
| `roll_encode_cache` | `roll_10_index.sbatch`, `roll_11_encode.sbatch` |
| `roll_oracle_ceiling` | `roll_16_ceiling.sbatch` |
| `roll_run_campaign` | `roll_20_train.sbatch`, `roll_25_phase2.sbatch`, `roll_30_phase3.sbatch` |
| `plots` | manual / `30_submit_campaign.sh` |

## Why flat, not subpackages

The corpus split above could be `f1/`, `pbc/`, `roll/` subpackages, but the
flat module paths are load-bearing: 16 slurm scripts plus `physics_wan`'s
`probes/` and `samples/` reference `generalized_physics.real.<module>` by
name, and the completed campaigns' run/cache directories were produced under
them. Grouping is by prefix; this file is the map.
