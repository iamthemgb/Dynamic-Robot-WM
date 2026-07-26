# generalized_physics

Implementation of `~/scratch/Generalized_Adaptive_Physics_Embedding_Plan.tex`:
variable-schema simulation supervision, causal video inference, and compact
Wan conditioning. Successor to `adaptive_physics/` (fixed 14-D parameter
vector) — the Wan interface is kept, both physics sources are generalized.

## Six core modules

| module | file | contract |
|---|---|---|
| frozen Wan VAE + latent cache | `data/cache_wan_latents.py` | provenance hash; causal chunking; delta boundary mask |
| metadata teacher | `models/metadata_teacher.py` | variable typed records `(key, scope, unit, value)` → DeepSets → belief **b ∈ R⁶⁴**; permutation/padding invariant |
| causal student | `models/causal_student.py` | ordered VAE latents + per-bin control GRU → 2-layer causal GRU → same 64-D belief, per-bin outputs |
| shared projector | `models/physics_projector.py` | 64 → **8 × 256** tokens (fixed, Wan-width-independent); learned null code; ordinary small init |
| physics adapters | `wan/physics_adapter.py` | 4 zero-gated cross-attn adapters at quarter depth; 256-wide bottleneck; width read from `wan_config()` |
| DiT LoRA | `wan/dit_lora.py` | rank 16, α 16, dropout 0.05 on `.q`/`.v`; **B zero-init**; single on/off switch for adapter-only attribution |

Supporting pieces: `models/metadata_records.py` (registry vocabularies, scope
sanity — raw `body_17`-style IDs are rejected, batching masks),
`data/metadata_registry.yaml`, `data/normalize_metadata.py` (per-key
identity/log/logit transforms + train-set stats manifest),
`data/synthetic_env.py` (two families with DIFFERENT schemas: 4-record
projectile_impact, 5-record push_slide), `data/counterfactual_dataset.py`
(same-group wrong codes), `data/change_point_dataset.py`,
`wan/mock_wan.py` (CPU mock with Wan-style named `.q/.k/.v/.o` projections so
LoRA targeting is identical on mock and real), and
`wan/adaptive_wan_integration.py` (real Wan2.1 wrapped-block insertion,
following `projectile_adaptive_smoke/models/adaptive_wan.py`).

## Training phases

```
training/phase0_representation_probe.py    # ceiling + causal controls gate
training/phase1_oracle_wan.py              # teacher + adapters + LoRA, shared-noise ranking
training/phase2_student_distillation.py    # student ← frozen teacher (belief MSE + query loss)
training/phase3_student_substitution.py    # 40/40/10/10 mixture, student FM at one endpoint
training/phase4_sliding_adaptation.py      # change points, sliding-window recomputation
```

## Run

```bash
cd /gpfs/radev/home/mzl7/scratch/wan_scripts_new
PY=~/.venvs/wan21/bin/python
$PY -m generalized_physics.tests.run_all       # 27 unit tests, CPU, ~30 s
$PY -m generalized_physics.run_smoke_e2e       # tiny 5-phase chain, ~10 s
$PY -m generalized_physics.run_smoke_e2e smoke # full smoke config
```

Unit tests cover the plan's checklist: record permutation/padding invariance,
scope sanity, temporal-order/causal masking, action-swap + one-bin impulse
shift, per-bin impulse timing, projector contract (teacher/student symmetry,
fixed tokens under variable schemas), zero-gate equivalence WITH LoRA
installed, LoRA zero-init identity + off-switch restoration + parameter
grouping, shared-noise ranking, and gradient coverage.

## Notes

* Everything trains against the CPU mock; swap `wan/mock_wan.py` for
  `wan/adaptive_wan_integration.py` + real cached latents for the 1.3B run.
* The mock DiT's output head is small-random (a pretrained stand-in): with a
  zero head the frozen backbone would output exactly zero and no gradient
  would ever reach the adapters or LoRA.
* Tiny-scale smoke validates integration only; gates (phase-0 controls,
  correct/wrong/null separation, gap closure) need the full smoke budget.
