# Projectile-Catch State-Conditional Dataset — Generation Plan

Status: v3 — CODE IMPLEMENTED 2026-07-29 (generation not yet run). New files: `group_dataset_generation.py` (group/plan/finalize CLI), `validate_group_dataset.py`, `slurm/generate_group_array.sbatch`, `slurm/finalize_groups.sbatch`, `configs/p0_pilot.json`, `configs/p1_full.json`; one-line `group` block added to `metadata.py`. Smoke test: `bash smoke_test_group_dataset.sh` from the bundle root (dry-run only — no simulation/rendering, login-node safe); verifies deterministic group construction, shared context per group, distinct ICs per sibling, 90/5/5 splits, exact branch mix, validator CLI. Passing as of 2026-07-29. Target pipeline: `projectile_ball_catch_export/projectile_ball_catch_robocasa_kitchen_scenes_scripts` (the self-contained export bundle, `pbc-native` env, `robocasa_native` backend).

v3 change (supersedes v2): **no physical property varies — physics is fixed at nominal everywhere** (gravity −9.81, no drag, stock friction/restitution). The conditioning signal the model must learn is the **state context (initial conditions)**. Groups remain: siblings share scene/appearance/camera and differ in their initial conditions. The physics-variation grid from v2 is deferred, not deleted — see Appendix A.

## 1. Goal and rationale

Build a projectile-catch dataset organized into **groups of state siblings**. A group fixes the scene — camera, appearance, robot base/home state, ball radius/mass/color — and contains M siblings, each a rollout from its **own initial conditions** (launch pose/velocity, release time, catch target, branch, and the catch/gripper commands planned for that IC) under **identical, nominal physics**.

Why this design, given the f1_10h null result (conditioning gates closed; tokens suppressed as useless):

- **Test the conditioning pathway on the easiest signal first.** Initial state is maximally and visibly causal: the ball's arc is a deterministic function of its initial pose/velocity under fixed physics. If the model can't learn to use *state* tokens, no physics-token experiment can succeed — and if it can, we have a validated pathway before layering in physics variation (Appendix A).
- **Appearance-matched negatives.** Within a group, siblings look identical until the ball moves. A "wrong state" condition drawn from a same-group sibling is contextually plausible — same kitchen, same camera, same ball — and can only be rejected by actually relating the state tokens to the observed dynamics, not by style cues.
- **No physics confounds by construction.** Physics tokens are dataset-wide constants, recorded in metadata for schema continuity; they carry zero information and are *expected* to be ignored. The state tokens are the experiment.

## 2. Definitions

- **Group** `g`: one sampled scene context (family, appearance, camera, robot base, ball properties) + M state siblings.
- **Sibling** `i`: one rollout of IC `i` under nominal physics. Episode ID: `{family}_g{group:06d}_s{i:02d}`.
- **View**: `main_camera` or `side_camera`. Every sibling is rendered from both, stored as separate files (already the pipeline default: `main.mp4`, `side.mp4`).

Wrong-state negatives for training are drawn **only within a group**: for sibling i, take the conditioning bundle of another sibling j — same scene, different launch/timing — so the mismatch is detectable only through dynamics (§11 on why the *full* bundle is swapped).

### One group, pictured

```
GROUP style020_seed0_g000123
shared by every cell: scene, appearance, camera pose, robot base/home,
                      ball radius/mass/color, physics (nominal, constant
                      dataset-wide), split assignment

              s00 (IC₀)        s01 (IC₁)        s02 (IC₂)        s03 (IC₃)
              cmds(IC₀)        cmds(IC₁)        cmds(IC₂)        cmds(IC₃)
            ┌────────────────┬────────────────┬────────────────┬────────────────┐
  nominal   │    g_s00       │    g_s01       │    g_s02       │    g_s03       │
  physics   │  ▶main ▶side   │  ▶main ▶side   │  ▶main ▶side   │  ▶main ▶side   │
            └────────────────┴────────────────┴────────────────┴────────────────┘
                       → siblings differ ONLY in state context:
              launch pose/velocity, release time, catch target, branch,
              and the open-loop commands planned from that IC

     wrong-state negative for s00 = the conditioning bundle (state + action
     tokens) of s01/s02/s03 — appearance-matched, rejectable only via dynamics
```

One cell = one rollout = `sYY/{main.mp4, side.mp4, metadata.json}` (both views rendered, stored separately). The whole group is one atomic unit for train/val/test splitting.

## 3. What is shared vs. varied

**Shared across the whole group** (drawn once from the group RNG, frozen):

| Shared item | Where it lives today |
|---|---|
| Layout/style/camera variant | `FAMILY_SPECS`, `visual_settings` |
| Camera pose incl. jitter | `sample.camera_jitter` → MJCF cameras (static) |
| Lighting, floor jitter, texture/background/color variants | `lighting_intensity`, `floor_material_jitter`, `visual_settings["dataset_*_variant"]` |
| Ball radius, mass, color | `ball_radius`, `ball_mass`, `ball_color` |
| Robot base pose + home config | `robot_base_*`, `home_q` |
| Physics (nominal, same in every group) | `<option gravity>`, `BALL_FRICTION`, `BALL_SOLREF/SOLIMP` |

**Per sibling** (drawn from the sibling's IC RNG):

| Varied item | Where it lives today |
|---|---|
| Launch position jitter, catch target jitter | `_build_episode_sample` jitters |
| Launch angle (58–70°), speed scale (0.98–1.02) → initial velocity | `_projectile_launch_velocity` |
| Release time (0.18–0.34 s) | `release_time_s` |
| Branch (success / near-miss / contact-failure / wrong-action) + controller offsets/lead times | branch tags, `controller_target_offset`, `*_lead_time_offset_s` |
| Command schedule (planned from this IC; physics is the true nominal physics, so planner and simulator agree) | controller plan |

RNG discipline: group RNG ← `(family, group_index)`; sibling IC RNG ← `(family, group_index, "ic", i)`. Appearance and state are independent streams, so no appearance↔state correlation.

To widen state diversity beyond today's jitters (the point of the dataset), the IC sampler broadens the existing ranges moderately: launch XY jitter ±0.16/±0.10 m (2× today), launch angle 55–72°, release time 0.15–0.40 s — subject to the P0 in-frame and catch-reachability checks, then frozen.

## 4. Physics: constant, recorded

Every episode uses today's constants — gravity `(0,0,-9.81)`, no fluid forces, `BALL_FRICTION=(5.0, 0.01, 0.001)`, `BALL_SOLREF=(0.004, 1.0)`, `BALL_SOLIMP=(0.95, 0.99, 0.001)`. `physics_tokens` stay in the metadata exactly as the schema defines them (constants are still tokens — the loader interface doesn't change), which lets physics variation be layered in later (Appendix A) without any schema migration. A hard validation gate asserts physics tokens are identical across the entire dataset.

## 5. Controller and commands

**No controller surgery is needed in v3.** Each sibling's commands are planned from its own IC by the existing `MujocoInterceptionController` under the true (nominal) physics — planner and simulator agree, so the legacy closed-loop details are harmless:

- The reactive close trigger and the analytic `predicted_close_time_s` nearly coincide under nominal physics; there is no counterfactual physics for the reactive path to leak.
- The grasp-capture weld stays at its legacy behavior (on for success branch, off for the failure branches, as `_build_episode_sample` already sets) — nothing we vary is masked by it.
- Trajectory recording stays as-is; there is no cross-sibling command-identity requirement (commands *should* differ per sibling — they encode the IC).

One consistency note worth recording in metadata: commands are a deterministic function of the state context, so `action_context` and `state_context` are mutually redundant on success-branch episodes. This is why wrong-condition negatives swap the full bundle (§11).

## 6. Dataset layout and metadata

```
DATASET_ROOT/
  <family>/                          # style020_seed0, style055_seed1000
    groups/
      <family>_g000123/
        group.json                   # shared context + per-sibling state/command summary + split
        s00/ {main.mp4, side.mp4, metadata.json}
        s01/ ...
        s02/ ...
        s03/ ...
    metadata.parquet                 # one row per sibling (finalize step)
    dataset_info.json
  splits.json                        # group_id -> train/val/test
  generation_config.json             # IC ranges, M, physics constants, code version
```

`metadata.json` keeps the existing five-section schema (`physics_tokens` / `state_context` / `geometry_context` / `action_context` / `outcomes`) unchanged, plus one new top-level block:

```json
"group": {
  "group_id": "style020_seed0_g000123",
  "sibling_index": 2,
  "sibling_ids": ["..._s00", "..._s01", "..._s02", "..._s03"],
  "split": "train",
  "physics_regime": "nominal_constant_v3"
}
```

`group.json` duplicates the shared context and tabulates every sibling's `state_context` (+ branch, close time, intercept), so a training loader can build wrong-state negatives without opening sibling episode JSONs.

## 7. Splits — group-atomic

Split is a function of `group_id` only: stable hash (blake2s of `group_id`) mod 100 → `<90` train, `<95` val, else test. Written once to `splits.json` at finalize, stamped into every sibling's metadata and the parquet. Entire groups stay in one split; both views of a sibling live in one episode dir, so views can never straddle splits. Validation asserts no group appears in two splits.

## 8. Phasing and scale

Two scene families (`style020_seed0`, `style055_seed1000`); `*_opposite_camera` variants optional later. Siblings per group: **M = 4** (config knob).

- **P0 — pilot (blocking, <1 GPU-day):** 2 families × 10 groups × 4 siblings = 80 rollouts. Deliverables: (a) shared-context identity within groups; (b) IC diversity and trajectory divergence between siblings (§10); (c) widened IC ranges keep the ball ≥80% in-frame post-release in both views and keep the success branch succeeding ≥70% (else tighten ranges once, then freeze); (d) eyeball MP4 contact sheets.
- **P1 — full generation:** 2 × 500 groups × 4 = **4,000 rollouts** (matches the N=4000 stratified-cache scale used in the Wan campaign). Optionally stretch to 750 groups/family (6,000) if early training looks data-hungry.

At ~2 min/rollout (832×480@30, 2.5 s, two views, rtx-batch): P1 ≈ 135 GPU-h, one **group per array task** (4 × 2 min ≈ 8 min/task). Storage ≈ 12–20 GB.

Branch mix: existing schedule (50% success / 20% near-miss / 20% contact-failure / 10% wrong-action) assigned per sibling from its IC RNG, balanced within each family.

## 9. Code changes (file by file, in the export bundle)

v3 needs **no changes** to `scene_builder.py`, `controller.py`, or physics handling — the physics path is untouched. `dataset_generation.py` and legacy behavior stay as-is.

1. **New** `group_dataset_generation.py`
   - `build_group(family, group_index, config)`: group RNG → shared context (appearance, camera jitter, ball props — the parts of `sample_episode` + `_build_episode_sample` that v3 freezes per group); sibling IC RNG → per-sibling launch/catch/release/branch draws with the widened ranges; reuses the existing branch → controller-offset logic.
   - CLI: `group` (generate all M siblings for one group sequentially; write `group.json`) and `finalize` (parquet + `splits.json` + `dataset_info.json` + split stamping).
2. `metadata.py` — add the small `group` block (group id, sibling index/ids, split, physics regime). Nothing else changes; `physics_tokens` from module constants is *correct* in v3 since physics is constant.
3. **New** `validate_group_dataset.py` — the gates in §10, runnable on a partial dataset.
4. `slurm/generate_group_array.sbatch` (array = group index; `FAMILY`, `DATASET_ROOT`, `PHASE_CONFIG` env vars; rtx-batch) and `slurm/finalize_groups.sbatch` (rtx-devel/CPU).

## 10. Validation gates (run after P0 and after P1)

1. **Shared-context identity (hard):** camera metadata, appearance fields, ball radius/mass/color identical across all siblings of a group.
2. **IC diversity (hard):** within a group, state contexts genuinely differ — min pairwise distance over (launch position, launch velocity, release time) above a floor; no duplicate ICs.
3. **Physics constancy (hard):** `physics_tokens` bitwise identical across the entire dataset.
4. **State→trajectory divergence (soft, thresholded):** median pairwise world-space ball-position divergence between same-group siblings at `release + 0.4 s` > 15 cm, so wrong-state negatives are unambiguous rather than near-duplicates.
5. **State–command consistency:** each sibling's recorded intercept position matches the ballistic prediction from its own `state_context` (catches planner/sampler drift).
6. **In-frame check:** ball reprojection visible ≥ 80% of post-release frames in both views (cameras' `pos/xyaxes/fovy` are in metadata).
7. **Split integrity:** groups atomic across splits; per-split branch distributions roughly uniform.
8. **Confound scan:** no correlation between state draws and appearance variables (independent RNG streams by construction — verify anyway; the f1_10h postmortem earns the paranoia).

## 11. Training integration (Wan / generalized_physics)

- **Conditioning target = `state_context`** (initial ball pose/velocity, release time; optionally catch target), tokenized the same way physics tokens were. Physics tokens may be included as constants for interface continuity; expect their gates to stay closed — that is not a failure in v3.
- **View sampling:** cache-build indexes episodes at sibling granularity with both view paths; the loader samples ONE view per (episode, epoch) draw — view is augmentation, not identity. Optional phase-2 student: fuse both views (separate files, identical timing).
- **Wrong-state negatives — swap the full bundle:** commands are a deterministic function of the IC, so if only `state_context` were swapped, the model could detect the mismatch by cross-checking state tokens against action tokens without watching the video. Negatives must swap state *and* action context together from the same donor sibling, so the wrong bundle is internally consistent and only inconsistent with the *video*. (`group.json` holds everything needed.)
- **Group-aware batching:** GroupBatcher already wants dense group ids; `group_id` maps directly (relabel after subsetting, per the cache_io convention).
- **Sanity probe before any big run:** linear-probe whether the frozen backbone can read the initial state from early frames; then verify the state-conditioning gates open on a short 1.3B run before committing bigger arms.

## 12. Open decisions

- Final widened IC ranges (one adjustment allowed at P0, then frozen).
- M = 4 vs 6 siblings/group; group count 500 vs 750 per family.
- Whether to include the catch target and branch tags in the conditioning bundle or keep conditioning to ball state + release time only.
- Opposite-camera families as additional groups — post-P1 call.

## 13. Runbook (once implemented)

```bash
cd /scratch/zl664_yale/world_model_robotics/Dynamic-Robot-WM-main/old_dataset_gen/projectile_ball_catch_export

# Pilot
FAMILY=style020_seed0 DATASET_ROOT=/scratch/zl664_yale/world_model_robotics/datasets/pbc_state_groups_p0 \
  PHASE_CONFIG=configs/p0_pilot.json sbatch --array=0-9 slurm/generate_group_array.sbatch

# Validate
python -m projectile_ball_catch_robocasa_kitchen_scenes_scripts.validate_group_dataset \
  --dataset-root .../pbc_state_groups_p0

# Full generation, then finalize
FAMILY=style020_seed0 DATASET_ROOT=.../pbc_state_groups_v1 PHASE_CONFIG=configs/p1_full.json \
  sbatch --array=0-499 slurm/generate_group_array.sbatch
FAMILY=style020_seed0 DATASET_ROOT=.../pbc_state_groups_v1 sbatch slurm/finalize_groups.sbatch
```

Env: `pbc-native` python (`/scratch/zl664_yale/world_model_robotics/envs/pbc-native/bin/python`), `MUJOCO_GL=egl`, submit from the bundle root (relative `logs/`), partitions rtx-batch / rtx-devel, `--requeue --open-mode=append`.

---

## Appendix A — Deferred: physics-variation grid (v2 design, for later phases)

Once state-token conditioning is demonstrated (gates open, wrong-state gap positive), physics variation can be layered onto the same group structure with no schema migration: each group's sibling set becomes a **grid** of physics blocks × the shared IC draws. Summary of the v2 design (full details in git history of this file):

- **Blocks** = fixed dataset-wide physics constants: gravity ×0.60 / ×1.40 (`<option gravity>`, planner keeps nominal), drag (`<option density>`=5.0 + `fluidshape="ellipsoid"` on the ball), ball–gripper friction 0.10 via explicit `<contact><pair>` entries (per-geom values would be masked by MuJoCo's max-combination rule), bouncy contact `solref=(0.010, 0.25)` with `priority="1"` on the ball geom in all blocks.
- The same ICs are reused across blocks, so IC-matched cross-block pairs are exact counterfactuals and wrong-*physics* negatives stay within-group and state/action-matched.
- Requires the v2 controller changes: `open_loop_commands` flag (kill the reactive close trigger; plan under nominal `sample.gravity` with sim physics in new `sim_gravity`/... fields), commanded-`q_cmd` trajectory recording (post-step qpos picks up physics-dependent sub-timestep sag), and a grasp-capture decision for the contact axes (the weld overrides exactly the physics those blocks vary).
- Hard gates: cross-block bitwise command identity per IC; single-axis purity per block; divergence thresholds (>8 cm at release+0.4 s for gravity/drag; ≥40% outcome-flip for gravity blocks).
