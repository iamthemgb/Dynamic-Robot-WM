"""Two-stage rbi campaign (velocity conditioning -> action tokens).

    python -m generalized_physics.real.rbi_run_campaign --arm wan21_t2v_1p3b
    python -m generalized_physics.real.rbi_run_campaign --arm wan21_t2v_1p3b \
        --tiny --limit 16                     # single-GPU smoke
    python -m generalized_physics.real.rbi_run_campaign --arm wan21_t2v_1p3b \
        --stage action [--p1-ckpt ...]        # stage 2 on the stage-1 ckpt

The rolling-ball interception corpus failed three zl664/physics_training
runs (rbi_phase1_problems_and_fixes.tex); this campaign reruns it under the
passing generalized-physics recipe with the fix package:

  * stage ``physics`` — the roll phase-1 recipe verbatim against the
    main-view rbi cache: velocity-only conditioning with NO action pathway
    anywhere in the model (tex B1 fix 1: within every velocity pair the
    action plan is identical, so action tokens are pure competition);
    zero-init gates in a 10x-LR group (B2); rank gap monitor-only (C1);
    teacher trained jointly at w_meta=0.2 (the passing recipe); single-term
    ROI-weighted flow loss with roi_lambda=32 — the weighted-mask form of
    backend.py, calibrated so ROI cells keep the ~33:1 per-cell emphasis
    the failed additive fm+4*roi form had at this corpus's ~12.6% ROI
    fraction (NOT the old two-term objective); high-sigma oversampling;
    5-GPU DDP; held-out bootstrap-CI gates (C3).
  * stage ``action`` — loads the stage-1 checkpoint FROZEN (teacher,
    decoder, projector, physics adapters, LoRA), installs the action
    pathway (rbi_action) and trains only it, paired against the
    opposite-ACTION sibling (action_group_id), plain unweighted fm loss.

Known caveat, accepted upstream: the corpus's velocity ROI oracle ceiling
is ~22x below the passive corpus's (0.041 vs 0.892); gates may fail for
corpus reasons — rbi_oracle_ceiling's report is folded into the summary so
that verdict is attributable.

Gate semantics: stage physics PASSes on val gap_null > 0 AND a bootstrap
95% CI for val gap_wrong strictly above 0 at some grid sigma (full-frame or
ROI); stage action on the analogous wrong-ACTION CI. Stage action refuses
to start unless stage physics passed (``--force`` overrides).
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..training import common
from . import backend, cache_io, phase1_real, phase2_real, rbi_action, \
    rbi_phase_action
from .paths import ARMS
from .roll_run_campaign import _gap_wrong_cis, _strip_per_episode, \
    bootstrap_ci, make_sigma_sampler
from ..config import smoke_config

RBI_ARMS = ("wan21_t2v_1p3b",)
VAL_SIGMA_GRID = (0.9, 0.99)
DATASET = "rolling_ball_velocity_4k_v1"


def rbi_arm(name, cache_name=None):
    base = ARMS[name]
    return replace(base, name=f"rbi_{base.name}",
                   cache_name=cache_name or "rbi_wan21_vae")


def _compat_cfg(arm):
    """The phase-1 cfg recipe (copied from roll_run_campaign — the stage-1
    state dicts will not load under anything else)."""
    cfg = smoke_config()
    cfg.vae.latent_channels = arm.latent_channels
    cfg.dit.prefix_bins = 2
    return cfg


def _init_distributed():
    """(rank, world, device); copied from roll_run_campaign, including the
    2h NCCL timeout that outlives rank 0's solo eval (R10)."""
    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, "cuda"
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist.get_rank(), world, f"cuda:{local}"


def _write_manifest(run_dir, key, gates, wall_s):
    manifest_path = run_dir / "run_manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})
    manifest[key] = {"done": True, "gates": gates,
                     "wall_s": round(wall_s, 1),
                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _oracle_ceiling(arm):
    p = Path(arm.cache_dir) / "oracle_ceiling.json"
    return json.loads(p.read_text()) if p.exists() else None


def _run_dir(arm, tiny):
    run_dir = arm.run_dir
    if tiny:
        run_dir = run_dir.parent.parent / "runs_tiny" / arm.name
    return run_dir


def _load_val(arm, args, device, registry):
    """Rank-0 held-out split; --tiny smokes on pilot caches whose val split
    is empty fall back to a small train slice (multiple of 4 so the
    action-pair grouping of stage 2 stays whole)."""
    import pandas as pd

    val_split, val_limit = "val", None
    if args.tiny:
        idx = pd.read_parquet(Path(arm.cache_dir) / "index.parquet")
        if int((idx["split"] == "val").sum()) < 2:
            val_split, val_limit = "train", 8
    val_cache, _, _ = cache_io.load_cache(
        arm.cache_dir, device=device, split=val_split, limit=val_limit,
        registry=registry)
    return val_cache


def _swap_to_action_groups(cache):
    """Shallow-copy the cache with group_id = dense action-pair ids, read
    from the already-filtered cache["index"] (never re-read from disk — the
    row selection must match). Stage 2's ActionPairBatcher then pairs
    opposite-action siblings."""
    idx = cache["index"]
    ag = idx["action_group_id"].to_numpy()
    _, counts = np.unique(ag, return_counts=True)
    if not (counts == 2).all():
        raise SystemExit(
            "action-pair grouping broken on this subset (members per pair: "
            f"{sorted(set(counts.tolist()))}) — --limit must be a multiple "
            "of 4 for --stage action")
    relabel = {g: i for i, g in enumerate(sorted(set(ag.tolist())))}
    out = dict(cache)
    out["group_id"] = torch.tensor([relabel[g] for g in ag],
                                   dtype=torch.long)
    return out


def _phase1_ckpt(run_dir, args):
    ck = (Path(args.p1_ckpt) if args.p1_ckpt
          else phase2_real.latest_checkpoint(run_dir / "phase1"))
    if ck is None or not Path(ck).exists():
        raise SystemExit(f"no stage-1 checkpoint under {run_dir / 'phase1'} "
                         "— run --stage physics first or pass --p1-ckpt")
    return Path(ck)


def _require_physics_pass(run_dir, args):
    if args.force or args.p1_ckpt:
        return
    sp = run_dir / "phase1" / "summary.json"
    if not sp.exists():
        raise SystemExit(f"--stage action blocked: {sp} missing "
                         "(run --stage physics first, or --force)")
    s = json.loads(sp.read_text())
    if not all(s["gates"].values()):
        raise SystemExit(f"--stage action blocked: stage physics gates "
                         f"{json.dumps(s['gates'])} (--force to override)")


# ------------------------------------------------------------ physics ----

def _run_physics(args):
    arm = rbi_arm(args.arm, cache_name=args.cache_name)
    rank, world, device = _init_distributed()
    is_main = rank == 0
    run_dir = _run_dir(arm, args.tiny)
    steps = 5 if args.tiny else args.p1_steps
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = _compat_cfg(arm)
    cfg.phase1_steps = steps
    cfg.train.batch_size = args.batch_size
    cfg.train.roi_lambda = args.roi_lambda
    cfg.train.sigma_high_frac = args.sigma_high_frac

    if is_main:
        print(f"=== {arm.name} physics: cache {arm.cache_dir} "
              f"(world {world}, roi_lambda {cfg.train.roi_lambda}) ===",
              flush=True)
    cache, registry, normalizer = cache_io.load_cache(
        arm.cache_dir, device=device, split="train", limit=args.limit)
    text_ctx = cache_io.load_text_ctx(arm.cache_dir, cache["prompt_id"],
                                      device=device)
    val_cache = val_text_ctx = None
    if is_main:
        val_cache = _load_val(arm, args, device, registry)
        val_text_ctx = cache_io.load_text_ctx(arm.cache_dir,
                                              val_cache["prompt_id"],
                                              device=device)
        print(f"train episodes: {len(cache['z'])}  "
              f"val: {len(val_cache['z'])}  Tz={cache['z'].shape[2]}  "
              f"velocity pairs={int(cache['group_id'].max()) + 1}  "
              f"roi={'roi' in cache}", flush=True)

    sampler = None
    if cfg.train.sigma_high_frac > 0.0:
        lo, hi = cfg.train.sigma_high_range
        sampler = make_sigma_sampler(arm.shift, cfg.train.sigma_high_frac,
                                     lo, hi, device)
    backend.install_real_backend(arm, cache, text_ctx, cfg, device=device,
                                 action_dim=1, sigma_sampler=sampler,
                                 roi_lambda=cfg.train.roi_lambda)

    t0 = time.time()
    p1 = phase1_real.run(cfg, cache, registry, normalizer,
                         run_dir / "phase1", steps=steps,
                         grad_accum=args.grad_accum,
                         eval_interval=args.eval_interval,
                         val_cache=val_cache, val_text_ctx=val_text_ctx,
                         val_sigma_grid=VAL_SIGMA_GRID, val_n=args.val_n)
    if not is_main:
        import torch.distributed as dist

        dist.barrier()
        return
    m, mv = p1["metrics"], p1["val_metrics"]
    cis = _gap_wrong_cis(mv)
    gates = {
        "gap_null_positive_val": mv["gap_null"] > 0,
        "gap_wrong_ci_positive": any(lo > 0 for lo, _ in cis.values()),
    }
    summary = {
        "phase": "physics", "arm": arm.name, "dataset": DATASET,
        "view": "main", "steps": steps,
        "roi_lambda": cfg.train.roi_lambda,
        "roi_loss_form": "single weighted mask (1 + lambda*roi, "
                         "mean-normalized) — never fm + lambda*roi_bonus",
        "sigma_high_frac": cfg.train.sigma_high_frac,
        "action_pathway": "absent by design in this stage (tex B1 fix 1)",
        "metrics_train": m,
        "metrics_val": {k: v for k, v in mv.items() if k != "sigma_grid"},
        "val_sigma_grid": {
            sig: _strip_per_episode(entry)
            for sig, entry in (mv.get("sigma_grid") or {}).items()},
        "gap_wrong_ci95": cis,
        "oracle_ceiling": _oracle_ceiling(arm),
        "gates": gates,
        "wall_s": round(time.time() - t0, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    d = run_dir / "phase1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_manifest(run_dir, "phase1", gates, time.time() - t0)
    print(f"[physics {arm.name}] val gap_wrong {mv['gap_wrong']:.6f} "
          f"gap_null {mv['gap_null']:.6f}", flush=True)
    print(f"gap_wrong CI95: {json.dumps(cis)}", flush=True)
    print(f"gates: {json.dumps(gates)}", flush=True)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()


# ------------------------------------------------------------- action ----

def _run_action(args):
    if args.limit is not None and args.limit % 4:
        raise SystemExit("--stage action needs --limit % 4 == 0 (rows are "
                         "grouped in per-rbi-group quads)")
    arm = rbi_arm(args.arm, cache_name=args.cache_name)
    rank, world, device = _init_distributed()
    is_main = rank == 0
    run_dir = _run_dir(arm, args.tiny)
    steps = 5 if args.tiny else args.action_steps
    _require_physics_pass(run_dir, args)
    p1_ckpt = _phase1_ckpt(run_dir, args)

    cfg = _compat_cfg(arm)
    cfg.train.batch_size = args.batch_size
    cfg.train.roi_lambda = 0.0        # the arm IS the action signal
    cfg.train.sigma_high_frac = args.sigma_high_frac

    if is_main:
        print(f"=== {arm.name} action: cache {arm.cache_dir} "
              f"(world {world}) p1 ckpt {p1_ckpt} ===", flush=True)
    cache, registry, normalizer = cache_io.load_cache(
        arm.cache_dir, device=device, split="train", limit=args.limit)
    text_ctx = cache_io.load_text_ctx(arm.cache_dir, cache["prompt_id"],
                                      device=device)
    cache_a = _swap_to_action_groups(cache)
    val_cache_a = val_text_ctx = None
    if is_main:
        val_cache = _load_val(arm, args, device, registry)
        val_text_ctx = cache_io.load_text_ctx(arm.cache_dir,
                                              val_cache["prompt_id"],
                                              device=device)
        val_cache_a = _swap_to_action_groups(val_cache)
        print(f"train episodes: {len(cache['z'])}  "
              f"val: {len(val_cache['z'])}  action pairs="
              f"{int(cache_a['group_id'].max()) + 1}", flush=True)

    # identical init on all ranks; per-rank noise diverges inside run()
    common.set_seed(cfg.train.seed)
    heads = phase2_real.load_phase1_heads(cfg, registry, p1_ckpt, device,
                                          with_projector=True)
    dit = backend.RealWanDiT(cfg, arm, device=device)
    dit.model.load_adaptive_state_dict(heads["adaptive"])
    # freeze BEFORE installing the shims: both walks below address blocks by
    # their pre-shim parameter names
    dit.model.physics.requires_grad_(False)
    for p in dit.model.lora_parameters():
        p.requires_grad_(False)
    bank, ref = rbi_action.install_action_bank(dit)
    encoder = rbi_action.ActionTokenEncoder().float().to(device)

    sampler = None
    if cfg.train.sigma_high_frac > 0.0:
        lo, hi = cfg.train.sigma_high_range
        sampler = make_sigma_sampler(arm.shift, cfg.train.sigma_high_frac,
                                     lo, hi, device)
    fm = rbi_action.make_action_fm_loss(dit, text_ctx, encoder, ref,
                                        sigma_sampler=sampler)
    val_fm = (rbi_action.make_action_fm_loss(dit, val_text_ctx, encoder, ref)
              if val_cache_a is not None else None)

    t0 = time.time()
    p2 = rbi_phase_action.run(
        cfg, cache_a, heads["teacher"], heads["projector"], dit, fm,
        encoder, bank, p1_ckpt, heads["step"],
        run_dir / "phase_action", steps=steps, grad_accum=args.grad_accum,
        eval_interval=args.eval_interval, p_null_action=args.p_null_action,
        val_cache=val_cache_a, val_fm=val_fm,
        val_sigma_grid=VAL_SIGMA_GRID, val_n=args.val_n)
    if not is_main:
        import torch.distributed as dist

        dist.barrier()
        return
    m, mv = p2["metrics"], p2["val_metrics"]
    cis = _gap_wrong_cis(mv)
    gates = {
        "gap_wrong_action_ci_positive": any(lo > 0
                                            for lo, _ in cis.values()),
    }
    summary = {
        "phase": "action", "arm": arm.name, "dataset": DATASET,
        "view": "main", "steps": steps,
        "p1_ckpt": str(p1_ckpt), "p1_step": heads["step"],
        "p_null_action": args.p_null_action,
        "sigma_high_frac": cfg.train.sigma_high_frac,
        "frozen": "teacher, decoder, projector, physics adapters+gates, "
                  "LoRA — fingerprint-checked every eval",
        "frozen_fingerprint": p2["frozen_fingerprint"],
        "n_trainable": p2["n_trainable"],
        "loss": "plain unweighted fm (no ROI weighting: the arm is the "
                "action signal)",
        "metrics_train": m,
        "metrics_val": {k: v for k, v in mv.items() if k != "sigma_grid"},
        "val_sigma_grid": {
            sig: _strip_per_episode(entry)
            for sig, entry in (mv.get("sigma_grid") or {}).items()},
        "gap_wrong_action_ci95": cis,
        "gap_null_action_positive_val": mv["gap_null"] > 0,   # informational
        "oracle_ceiling": _oracle_ceiling(arm),
        "gates": gates,
        "wall_s": round(time.time() - t0, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    d = run_dir / "phase_action"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_manifest(run_dir, "phase_action", gates, time.time() - t0)
    print(f"[action {arm.name}] val gap_wrong_action {mv['gap_wrong']:.6f} "
          f"gap_null_action {mv['gap_null']:.6f}", flush=True)
    print(f"gap_wrong_action CI95: {json.dumps(cis)}", flush=True)
    print(f"gates: {json.dumps(gates)}", flush=True)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(RBI_ARMS))
    ap.add_argument("--stage", choices=("physics", "action"),
                    default="physics")
    ap.add_argument("--p1-steps", type=int, default=4000)
    ap.add_argument("--action-steps", type=int, default=2000)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--roi-lambda", type=float, default=32.0,
                    help="stage physics only; single weighted-mask term "
                         "(33:1 per-cell parity with the failed additive "
                         "fm+4*roi form at ~12.6%% ROI fraction)")
    ap.add_argument("--sigma-high-frac", type=float, default=0.3)
    ap.add_argument("--p-null-action", type=float, default=0.0,
                    help="stage action: prob of a no-action-context TRAINING "
                         "draw. Default 0: the bypass null is exactly the "
                         "frozen stage-1 model (nothing to train, zero "
                         "gradient) — the eval triple still measures it")
    ap.add_argument("--eval-interval", type=int, default=1000,
                    help="both stages (final-step eval always runs)")
    ap.add_argument("--val-n", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap train episodes (smoke); action stage needs "
                         "a multiple of 4")
    ap.add_argument("--tiny", action="store_true",
                    help="5 steps; smoke only, separate run dir")
    ap.add_argument("--cache-name", default=None)
    ap.add_argument("--p1-ckpt", default=None,
                    help="stage action: stage-1 trainer.pt (default: "
                         "latest under the run dir's phase1)")
    ap.add_argument("--force", action="store_true",
                    help="stage action: proceed even if stage physics "
                         "gates failed")
    args = ap.parse_args()

    if args.stage == "action":
        return _run_action(args)
    return _run_physics(args)


if __name__ == "__main__":
    main()
