"""Phases 1-3 (velocity conditioning -> student) on the ball-rolling corpus.

    python -m generalized_physics.real.roll_run_campaign --arm wan21_t2v_1p3b
    python -m generalized_physics.real.roll_run_campaign --arm wan21_t2v_1p3b \
        --tiny --limit 16            # adapter smoke on a pilot cache
    python -m generalized_physics.real.roll_run_campaign --arm wan21_t2v_1p3b \
        --phase 2 [--delta-z off]    # student distillation from the p1 ckpt
    python -m generalized_physics.real.roll_run_campaign --arm wan21_t2v_1p3b \
        --phase 3                    # substitution; picks the p2 gate winner

Phase 2 (``phase2_real``) is DiT-free and single-GPU: it reloads the frozen
phase-1 heads from the phase-1 checkpoint and trains only the causal
student, one run dir per input variant (``phase2`` = [Z; dZ] via
``use_delta_z``, ``phase2_nodz`` = stock z-only). Phase 3 refuses to start
unless at least one phase-2 variant passed its gates, then substitutes that
student's codes into the conditioning mixture (``phase3_real``) with the
same ROI-weighted, sigma-oversampled fm loss as phase 1, and reports a
held-out sigma-grid gap/closure eval.

Reuses the pbc campaign machinery wholesale against a ``roll_``-prefixed
cache/run namespace, with the muffling-diagnosis fixes wired in:

  * ROI-weighted flow loss (``cfg.train.roi_lambda``, default 4.0 here) —
    the ball tube is ~1% of latent cells, so the unweighted mean starves the
    conditioning gradient (problem 1);
  * high-sigma oversampling (``sigma_high_frac`` of training draws uniform
    on ``sigma_high_range``) — the conditioning signal lives at sigma -> 1
    (problem 2);
  * held-out VAL paired eval with a sigma grid, ROI-restricted variants and
    per-episode bootstrap CIs (problems 2/5); the pbc runs had no held-out
    measurement at all;
  * defaults sized for ~4-5 epochs: batch 2 x grad-accum 2, 4000 steps
    (problem 5).

Gate semantics: PASS requires val gap_null > 0 AND a bootstrap 95% CI for
val gap_wrong strictly above 0 at some grid sigma (full-frame or ROI).
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..config import smoke_config
from ..training import common
from . import backend, cache_io, phase1_real, phase2_real, phase3_real
from .flow_match import shift_sigma
from .paths import ARMS

ROLL_ARMS = ("wan21_t2v_1p3b", "wan22_ti2v_5b")
VAL_SIGMA_GRID = (0.9, 0.99)


def roll_arm(name, cache_name=None):
    base = ARMS[name]
    vae = "wan21" if base.vae_kind == "wan21" else "wan22"
    return replace(base, name=f"roll_{base.name}",
                   cache_name=cache_name or f"roll_{vae}_vae")


def make_sigma_sampler(shift, frac, lo, hi, device):
    """Mixture: with prob ``frac`` draw sigma ~ U(lo, hi), else the standard
    shifted-uniform schedule. Training-only (backend skips it for evals)."""

    def sampler(b):
        pick = torch.rand(b, device=device) < frac
        base = shift_sigma(torch.rand(b, device=device), shift)
        high = lo + (hi - lo) * torch.rand(b, device=device)
        return torch.where(pick, high, base)

    return sampler


def bootstrap_ci(values, n_boot=10000, seed=0):
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(1)
    return (float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def _strip_per_episode(entry):
    out = {k: v for k, v in entry.items()
           if k not in ("per_episode", "roi")}
    if "roi" in entry:
        out["roi"] = {k: v for k, v in entry["roi"].items()
                      if k != "per_episode"}
    return out


def _gap_wrong_cis(val_metrics):
    """{sigma or sigma+'_roi': [ci_lo, ci_hi]} from per-episode gap lists."""
    out = {}
    for sig, entry in (val_metrics.get("sigma_grid") or {}).items():
        if "per_episode" in entry:
            out[sig] = list(bootstrap_ci(entry["per_episode"]["gap_wrong"]))
        roi = entry.get("roi") or {}
        if "per_episode" in roi:
            out[f"{sig}_roi"] = list(
                bootstrap_ci(roi["per_episode"]["gap_wrong"]))
    return out


def _init_distributed():
    """(rank, world, device). Under torchrun each rank owns one GPU and
    phase1_real averages gradients across ranks; single-process runs are
    untouched."""
    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1, "cuda"
    from datetime import timedelta

    import torch.distributed as dist

    # rank 0 runs the full eval suite (train + val sigma grid, ~15 min)
    # alone while the other ranks wait in the next step's all-reduce; the
    # default 10-min NCCL watchdog kills that wait (job 246368).
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    return dist.get_rank(), world, f"cuda:{local}"


def _assert_single_process(phase):
    import os

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise SystemExit(f"phase {phase} is single-process; "
                         "launch without torchrun")


def _compat_cfg(arm):
    """The phase-1 cfg recipe. Head dims come from PipelineConfig defaults
    and latent geometry from the arm; the phase-1 state dicts will not load
    under anything else."""
    cfg = smoke_config()
    cfg.vae.latent_channels = arm.latent_channels
    cfg.dit.prefix_bins = 2
    return cfg


def _load_val(arm, args, device, registry):
    """Held-out split for the single-process phases. --tiny smokes may run
    on pilot caches whose val split is empty; a small train slice keeps the
    val code path exercised regardless."""
    val_split, val_limit = "val", None
    if args.tiny:
        import pandas as pd

        idx = pd.read_parquet(Path(arm.cache_dir) / "index.parquet")
        if int((idx["split"] == "val").sum()) < 2:
            val_split, val_limit = "train", 8
    val_cache, _, _ = cache_io.load_cache(
        arm.cache_dir, device=device, split=val_split, limit=val_limit,
        registry=registry)
    return val_cache


def _phase1_ckpt(arm, args):
    ck = (Path(args.p1_ckpt) if args.p1_ckpt
          else phase2_real.latest_checkpoint(arm.run_dir / "phase1"))
    if ck is None or not Path(ck).exists():
        raise SystemExit(
            f"no phase-1 checkpoint under {arm.run_dir / 'phase1'} — "
            "run phase 1 first or pass --p1-ckpt")
    return Path(ck)


def _write_manifest(run_dir, key, gates, wall_s):
    manifest_path = run_dir / "run_manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})
    manifest[key] = {"done": True, "gates": gates,
                     "wall_s": round(wall_s, 1),
                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _run_phase2(args):
    _assert_single_process(2)
    arm = roll_arm(args.arm, cache_name=args.cache_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = arm.run_dir
    steps = args.p2_steps
    if args.tiny:
        run_dir = run_dir.parent.parent / "runs_tiny" / arm.name
        steps = 5
    delta_on = args.delta_z == "on"
    variant = "phase2" if delta_on else "phase2_nodz"
    out_dir = run_dir / variant

    cfg = _compat_cfg(arm)
    cfg.phase2_steps = steps
    cfg.train.batch_size = args.p2_batch
    cfg.student.use_delta_z = delta_on

    ckpt = _phase1_ckpt(arm, args)
    print(f"=== {arm.name} {variant} "
          f"({'z+dZ' if delta_on else 'z-only'}): cache {arm.cache_dir} "
          f"p1 ckpt {ckpt} device {device} ===", flush=True)
    cache, registry, normalizer = cache_io.load_cache(
        arm.cache_dir, device=device, split="train", limit=args.limit)
    val_cache = _load_val(arm, args, device, registry)
    print(f"train episodes: {len(cache['z'])}  val: {len(val_cache['z'])}  "
          f"Tz={cache['z'].shape[2]}  roi={'roi' in cache}", flush=True)
    heads = phase2_real.load_phase1_heads(cfg, registry, ckpt, device)

    t0 = time.time()
    p2 = phase2_real.run(cfg, cache, heads["teacher"], heads["decoder"],
                         out_dir, steps=steps, val_cache=val_cache)
    mv = p2["val_metrics"]
    ci_group = bootstrap_ci(mv["per_group_acc"])
    ci_episode = bootstrap_ci(mv["per_episode_acc"])
    gates = {
        # primary: siblings share everything but velocity, so >0.5 means
        # the belief carries velocity content; per-GROUP resampling (n=37)
        # because siblings are not independent draws
        "paired_acc_group_ci_above_half": ci_group[0] > 0.5,
        "time_shuffle_degrades":
            mv["query_time_shuffled"] > mv["query_loss"],
    }
    summary = {
        "phase": 2, "arm": arm.name, "variant": variant,
        "dataset": "roll_groups_v1", "steps": steps,
        "batch_size": cfg.train.batch_size,
        "use_delta_z": delta_on,
        "p1_ckpt": str(ckpt), "p1_step": heads["step"],
        "metrics_train": p2["metrics"],
        "metrics_val": {k: v for k, v in mv.items()
                        if k not in ("per_episode_acc", "prefix_curve")},
        "prefix_curve": mv["prefix_curve"],
        "paired_acc_ci95": {"group": list(ci_group),
                            "episode": list(ci_episode)},
        "gates": gates,
        "wall_s": round(time.time() - t0, 1),
        "peak_mem_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 2)
                        if torch.cuda.is_available() else None),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_manifest(run_dir, variant, gates, time.time() - t0)
    print(f"[{variant} {arm.name}] paired_acc {mv['paired_acc']:.4f} "
          f"group CI95 [{ci_group[0]:.4f}, {ci_group[1]:.4f}] "
          f"best_end {json.dumps(mv['best_end'])}", flush=True)
    print(f"gates: {json.dumps(gates)}", flush=True)


def _pick_phase2_winner(run_dir):
    """The chain's hard gate: phase 3 refuses to run unless a phase-2
    variant passed, and takes the better group-CI lower bound."""
    cands = []
    for variant in ("phase2", "phase2_nodz"):
        sp = run_dir / variant / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        if all(s["gates"].values()):
            cands.append((s["paired_acc_ci95"]["group"][0], variant, s))
    if not cands:
        raise SystemExit(
            f"phase 3 blocked: no phase-2 variant under {run_dir} passed "
            "its gates (phase2/summary.json, phase2_nodz/summary.json)")
    return max(cands, key=lambda c: c[0])


def _run_phase3(args):
    _assert_single_process(3)
    arm = roll_arm(args.arm, cache_name=args.cache_name)
    device = "cuda"
    run_dir = arm.run_dir
    steps = args.p3_steps
    if args.tiny:
        run_dir = run_dir.parent.parent / "runs_tiny" / arm.name
        steps = 5
    ci_lo, p2_variant, s2 = _pick_phase2_winner(run_dir)
    p2_ckpt = phase2_real.latest_checkpoint(run_dir / p2_variant)
    if p2_ckpt is None:
        raise SystemExit(f"phase 3: {run_dir / p2_variant} has a summary "
                         "but no checkpoint")

    cfg = _compat_cfg(arm)
    cfg.phase3_steps = steps
    cfg.train.batch_size = args.batch_size
    cfg.train.roi_lambda = args.roi_lambda
    cfg.train.sigma_high_frac = args.sigma_high_frac
    cfg.student.use_delta_z = bool(s2["use_delta_z"])
    p1_ckpt = _phase1_ckpt(arm, args)
    print(f"=== {arm.name} phase3: student {p2_variant} "
          f"(group CI lo {ci_lo:.4f}) p1 {p1_ckpt} p2 {p2_ckpt} ===",
          flush=True)

    cache, registry, normalizer = cache_io.load_cache(
        arm.cache_dir, device=device, split="train", limit=args.limit)
    text_ctx = cache_io.load_text_ctx(arm.cache_dir, cache["prompt_id"],
                                      device=device)
    val_cache = _load_val(arm, args, device, registry)
    val_text_ctx = cache_io.load_text_ctx(arm.cache_dir,
                                          val_cache["prompt_id"],
                                          device=device)
    print(f"train episodes: {len(cache['z'])}  val: {len(val_cache['z'])}  "
          f"roi={'roi' in cache}", flush=True)

    sampler = None
    if cfg.train.sigma_high_frac > 0.0:
        lo, hi = cfg.train.sigma_high_range
        sampler = make_sigma_sampler(arm.shift, cfg.train.sigma_high_frac,
                                     lo, hi, device)
    backend.install_real_backend(arm, cache, text_ctx, cfg, device=device,
                                 action_dim=1, sigma_sampler=sampler,
                                 roi_lambda=cfg.train.roi_lambda)
    dit = common.build_dit(cfg)
    heads = phase2_real.load_phase1_heads(cfg, registry, p1_ckpt, device,
                                          with_projector=True)
    dit.model.load_adaptive_state_dict(heads["adaptive"])
    st2 = torch.load(p2_ckpt, map_location="cpu", weights_only=False)
    if st2["use_delta_z"] != cfg.student.use_delta_z:
        raise SystemExit(f"{p2_ckpt} use_delta_z={st2['use_delta_z']} "
                         f"but winner summary says {cfg.student.use_delta_z}")
    student = common.build_student(cfg, action_dim=1)
    student.load_state_dict(st2["student"])

    phase1 = {"cache": cache, "teacher": heads["teacher"],
              "decoder": heads["decoder"], "projector": heads["projector"],
              "dit": dit, "registry": registry, "normalizer": normalizer}
    t0 = time.time()
    p3 = phase3_real.run(cfg, phase1, {"student": student},
                         run_dir / "phase3", steps=steps,
                         grad_accum=args.grad_accum)

    # held-out sigma-grid eval; fm_loss must serve VAL rows from the val
    # text table (row-index trap, same as phase 1's _val_eval)
    tr = cfg.train
    with_roi = "roi" in val_cache
    grid = {}
    orig_fm = common.fm_loss
    common.fm_loss = backend.make_real_fm_loss(dit, val_text_ctx,
                                               roi_lambda=tr.roi_lambda)
    try:
        mixed = phase3_real._eval_micro(
            cfg, val_cache, heads["teacher"], heads["projector"], dit,
            student, seed=tr.seed + 6, n=args.val_n)
        for sig in VAL_SIGMA_GRID:
            entry = phase3_real._eval_micro(
                cfg, val_cache, heads["teacher"], heads["projector"], dit,
                student, seed=tr.seed + 6, n=args.val_n, sigma=sig,
                per_episode=True)
            if with_roi:
                entry["roi"] = phase3_real._eval_micro(
                    cfg, val_cache, heads["teacher"], heads["projector"],
                    dit, student, seed=tr.seed + 6, n=args.val_n, sigma=sig,
                    roi_only=True, per_episode=True)
            grid[str(sig)] = entry
    finally:
        common.fm_loss = orig_fm

    cis, cis_wrong, oracle = {}, {}, {}
    for sig, entry in grid.items():
        variants = [(sig, entry)]
        roi_e = entry.get("roi")
        if roi_e is not None and "per_episode" in roi_e:
            variants.append((f"{sig}_roi", roi_e))
        for key, e in variants:
            cis[key] = list(bootstrap_ci(e["per_episode"]["gap_student"]))
            cis_wrong[key] = list(
                bootstrap_ci(e["per_episode"]["gap_student_wrong"]))
            oracle[key] = e["oracle_gap"]
    gates = {
        # primary (content): the student's own-video code must beat its
        # sibling-video code — student-vs-null passes for a RANDOM student
        # via the presence bias (verified in the tiny smoke), exactly the
        # muffling failure mode
        "gap_student_wrong_ci_positive": any(
            lo > 0 and oracle[k] > 1e-4
            for k, (lo, _hi) in cis_wrong.items()),
        # secondary: student codes beat null where the teacher gap itself
        # is meaningful (else closure is noise-over-noise, the f1 mode)
        "gap_student_ci_positive": any(
            lo > 0 and oracle[k] > 1e-4 for k, (lo, _hi) in cis.items()),
    }
    summary = {
        "phase": 3, "arm": arm.name, "dataset": "roll_groups_v1",
        "steps": steps, "batch_size": tr.batch_size,
        "grad_accum": args.grad_accum, "roi_lambda": tr.roi_lambda,
        "sigma_high_frac": tr.sigma_high_frac,
        "student_variant": p2_variant,
        "use_delta_z": cfg.student.use_delta_z,
        "p1_ckpt": str(p1_ckpt), "p2_ckpt": str(p2_ckpt),
        "metrics_train": p3["metrics"],
        "metrics_val_mixed": mixed,
        "val_sigma_grid": {sig: _strip_per_episode(entry)
                           for sig, entry in grid.items()},
        "gap_student_ci95": cis, "gap_student_wrong_ci95": cis_wrong,
        "oracle_gap": oracle,
        "gates": gates,
        "wall_s": round(time.time() - t0, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    d = run_dir / "phase3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_manifest(run_dir, "phase3", gates, time.time() - t0)
    closures = {sig: entry.get("gap_closure")
                for sig, entry in grid.items()}
    print(f"[phase3 {arm.name}] mixed closure "
          f"{mixed.get('gap_closure')} grid closures {json.dumps(closures)}",
          flush=True)
    print(f"gap_student CI95: {json.dumps(cis)}", flush=True)
    print(f"gap_student_wrong CI95: {json.dumps(cis_wrong)}", flush=True)
    print(f"gates: {json.dumps(gates)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ROLL_ARMS))
    ap.add_argument("--phase", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--p1-steps", type=int, default=4000)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--roi-lambda", type=float, default=4.0)
    ap.add_argument("--sigma-high-frac", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap train episodes (smoke)")
    ap.add_argument("--tiny", action="store_true",
                    help="5 steps; smoke only, separate run dir")
    ap.add_argument("--cache-name", default=None)
    # phase 2/3 (single-process) options
    ap.add_argument("--p2-steps", type=int, default=4000)
    ap.add_argument("--p2-batch", type=int, default=16)
    ap.add_argument("--delta-z", choices=("on", "off"), default="on",
                    help="student input variant: on = [Z; dZ] (phase2 dir), "
                         "off = stock z-only (phase2_nodz dir)")
    ap.add_argument("--p1-ckpt", default=None,
                    help="phase-1 trainer.pt (default: latest under the "
                         "arm's real phase1 run dir)")
    ap.add_argument("--p3-steps", type=int, default=1000)
    ap.add_argument("--val-n", type=int, default=48,
                    help="phase-3 held-out eval episodes per grid point")
    args = ap.parse_args()

    if args.phase == 2:
        return _run_phase2(args)
    if args.phase == 3:
        return _run_phase3(args)

    arm = roll_arm(args.arm, cache_name=args.cache_name)
    rank, world, device = _init_distributed()
    is_main = rank == 0
    run_dir = arm.run_dir
    steps = args.p1_steps
    if args.tiny:
        run_dir = run_dir.parent.parent / "runs_tiny" / arm.name
        steps = 5
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = smoke_config()
    cfg.vae.latent_channels = arm.latent_channels
    cfg.dit.prefix_bins = 2
    cfg.phase1_steps = steps
    cfg.train.batch_size = args.batch_size
    cfg.train.roi_lambda = args.roi_lambda
    cfg.train.sigma_high_frac = args.sigma_high_frac

    if is_main:
        print(f"=== {arm.name}: cache {arm.cache_dir} (world {world}) ===",
              flush=True)
    cache, registry, normalizer = cache_io.load_cache(
        arm.cache_dir, device=device, split="train", limit=args.limit)
    text_ctx = cache_io.load_text_ctx(arm.cache_dir, cache["prompt_id"],
                                      device=device)
    # rank 0 alone evaluates, so only it needs the val split in memory.
    # --tiny smokes may run on pilot caches whose val split is empty; a
    # small train slice keeps the val code path exercised regardless.
    val_cache = val_text_ctx = None
    if is_main:
        val_split = "val"
        val_limit = None
        if args.tiny:
            import pandas as pd

            idx = pd.read_parquet(Path(arm.cache_dir) / "index.parquet")
            if int((idx["split"] == "val").sum()) < 2:
                val_split, val_limit = "train", 8
        val_cache, _, _ = cache_io.load_cache(
            arm.cache_dir, device=device, split=val_split, limit=val_limit,
            registry=registry)
        val_text_ctx = cache_io.load_text_ctx(arm.cache_dir,
                                              val_cache["prompt_id"],
                                              device=device)
        print(f"train episodes: {len(cache['z'])}  "
              f"val: {len(val_cache['z'])}  Tz={cache['z'].shape[2]}  "
              f"groups={int(cache['group_id'].max()) + 1}  "
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
                         val_cache=val_cache, val_text_ctx=val_text_ctx,
                         val_sigma_grid=VAL_SIGMA_GRID)
    if not is_main:
        import torch.distributed as dist

        dist.barrier()        # wait out rank 0's final eval, then exit clean
        return
    m, mv = p1["metrics"], p1["val_metrics"]
    cis = _gap_wrong_cis(mv)
    gates = {
        "gap_null_positive_val": mv["gap_null"] > 0,
        "gap_wrong_ci_positive": any(lo > 0 for lo, _ in cis.values()),
    }
    summary = {
        "phase": 1, "arm": arm.name, "dataset": "roll_groups_v1",
        "steps": steps, "roi_lambda": cfg.train.roi_lambda,
        "sigma_high_frac": cfg.train.sigma_high_frac,
        "metrics_train": m,
        "metrics_val": {k: v for k, v in mv.items() if k != "sigma_grid"},
        "val_sigma_grid": {
            sig: _strip_per_episode(entry)
            for sig, entry in (mv.get("sigma_grid") or {}).items()},
        "gap_wrong_ci95": cis,
        "gates": gates,
        "wall_s": round(time.time() - t0, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    d = run_dir / "phase1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    manifest_path = run_dir / "run_manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})
    manifest["phase1"] = {"done": True, "gates": gates,
                          "wall_s": summary["wall_s"],
                          "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[phase1 {arm.name}] val gap_wrong {mv['gap_wrong']:.6f} "
          f"gap_null {mv['gap_null']:.6f}", flush=True)
    print(f"gap_wrong CI95: {json.dumps(cis)}", flush=True)
    print(f"gates: {json.dumps(gates)}", flush=True)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()        # release the non-main ranks parked in main()


if __name__ == "__main__":
    main()
