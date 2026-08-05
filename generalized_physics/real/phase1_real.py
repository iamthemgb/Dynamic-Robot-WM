"""Phase 1 (oracle Wan conditioning) against the real DiT, with resume.

Loss math is identical to ``training/phase1_oracle_wan.run``::

    u < p_null          -> l_fm(null tokens)
    u < p_null+p_noise  -> l_fm(projector(b_t + noise))
    else                -> l_fm(projector(b_t))
    loss = w_fm*l_fm + w_meta*l_meta

The plan's shuffled-code rank hinge (eqs. 16-19) is deliberately NOT
optimized: training the correct-below-wrong margin is circular with the
correct/wrong gap used as the phase gate.  Instead, every
``rank_monitor_interval`` steps the wrong-code loss is recomputed under
``torch.no_grad`` with the SAME (tau, eps) and the signed diagnostic
``rank_gap = l_w - l_fm`` is logged (positive = the physics code is
informative); the held-out ``gap_wrong``/``gap_null`` from ``paired_eval``
remain the checking points.  ``cfg.train.w_rank`` is ignored.

Reimplemented rather than instrumented because phase 1 on the 14B is a
multi-hour job on ``PreemptMode=REQUEUE`` partitions, and the stock loop has
no resume point. One deliberate departure, from
``projectile/train_real.py`` (written for exactly this hardware regime):

  * a third optimizer group runs the zero-init adapter gates at 10x LR with
    no weight decay, otherwise they never leave zero.

Returns the same dict keys as the stock ``run()`` so phases 2/3/4 chain
unchanged.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..data.counterfactual_dataset import GroupBatcher
from ..training import common
from .instrument import append_row, open_train_log

EMA_SPAN = 100


def _dist_info():
    """(rank, world_size); (0, 1) when torch.distributed is not initialized."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _sync_grads(params, world_size):
    """Average gradients across ranks (poor-man's DDP).

    The stock DDP wrapper is a bad fit here: the loss reaches the DiT through
    the monkey-patched ``common.fm_loss`` closure rather than a module
    forward, and which parameters receive grads varies per step (the learned
    null code only trains on null-cond draws). Averaging explicit grads after
    backward — zero-filling the absent ones so every rank reduces the same
    tensor list — gives identical updates on every rank with none of DDP's
    graph assumptions.
    """
    import torch.distributed as dist

    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad)
        p.grad /= world_size


def build_all(cfg, registry):
    teacher = common.build_teacher(cfg, registry)
    decoder = common.build_decoder(cfg, teacher)
    projector = common.build_projector(cfg)
    dit = common.build_dit(cfg)
    return teacher, decoder, projector, dit


def build_optimizer(cfg, teacher, decoder, projector, dit,
                    gate_lr_mult=10.0):
    tr = cfg.train
    gates = dit.physics.gate_params()
    non_gate = (list(projector.parameters()) + dit.physics.non_gate_params()
                + dit.model.lora_parameters())
    groups = [
        {"params": common.dedupe_params(list(teacher.parameters())
                                        + list(decoder.parameters())),
         "lr": tr.lr_teacher_student, "weight_decay": tr.weight_decay},
        {"params": common.dedupe_params(non_gate),
         "lr": tr.lr_projector_adapters, "weight_decay": tr.weight_decay},
        {"params": gates, "lr": tr.lr_projector_adapters * gate_lr_mult,
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    warmup = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 1.0))
    return opt, sched


def save_checkpoint(out_dir, step, ema_fm, dit, teacher, decoder, projector,
                    opt, sched):
    d = Path(out_dir) / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "ema_fm": ema_fm,
        "adaptive": dit.model.adaptive_state_dict(),   # never the frozen base
        "teacher": teacher.state_dict(),
        "decoder": decoder.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "python": random.getstate(),
                "numpy": np.random.get_state()},
    }, d / "trainer.pt")
    # keep the last two checkpoints only; the frozen base is re-loadable
    ckpts = sorted(Path(out_dir).glob("ckpt_*/trainer.pt"))
    for old in ckpts[:-2]:
        old.unlink()
        old.parent.rmdir()
    print(f"saved {d}", flush=True)


def latest_checkpoint(out_dir):
    cks = sorted(Path(out_dir).glob("ckpt_*/trainer.pt"))
    return cks[-1] if cks else None


def _fm_supports_roi():
    import inspect

    try:
        return "roi_only" in inspect.signature(common.fm_loss).parameters
    except (TypeError, ValueError):
        return False


@torch.no_grad()
def paired_eval(cfg, cache, teacher, projector, dit, seed=0, n=64,
                micro=8, sigma_grid=(), per_episode=False,
                roi_variants=False):
    """Correct / wrong / null flow losses under shared (tau, eps).

    Micro-batched: a batch-64 forward through the 14B at seq_len 23,400 is a
    memory spike for no benefit, so the paired triple runs in chunks.

    Defaults reproduce the historical metric dict exactly. Extensions
    (muffling diagnosis, problems 1/2/5):

    * ``sigma_grid``: additionally evaluate the triple at each FIXED sigma
      (the conditioning signal lives at sigma -> 1; the mixed-sigma average
      buries it). Results land in ``metrics["sigma_grid"][str(sigma)]``.
    * ``per_episode``: sigma-grid entries run micro=1 and return per-episode
      gap lists so the caller can bootstrap a CI for gap_wrong.
    * ``roi_variants``: where the bound ``common.fm_loss`` supports the
      ``roi_only`` reduction and the cache carries roi masks, sigma-grid
      entries also report gaps measured over the ball tube alone (``roi``
      subdict) — the only region where sibling codes can differ at all.
    """
    was_training = dit.model.training
    dit.model.eval()          # LoRA dropout off -- it would add variance
    teacher.eval()            # exactly where correct/wrong are compared
    projector.eval()

    batcher = GroupBatcher(cache, seed=seed)
    batch = batcher.paired_batch(min(n, len(cache["z"])))
    b_t = teacher.forward_batch(batch["rec"])
    b_w = teacher.forward_batch(batch["rec_wrong"])
    g = torch.Generator().manual_seed(seed)
    B = len(batch["idx"])
    with_roi = roi_variants and "roi" in batch and _fm_supports_roi()

    def triple(sl, sigma=None, roi_only=False):
        """(l_c, l_w, l_n) for one chunk under shared (tau, eps)."""
        sub = {"z": batch["z"][sl], "idx": batch["idx"][sl]}
        if "roi" in batch:
            sub["roi"] = batch["roi"][sl]
        kw = {"roi_only": True} if roi_only else {}
        tau = (None if sigma is None
               else torch.full((sl.stop - sl.start,), float(sigma)))
        l_c, tau, eps = common.fm_loss(dit, sub, projector(b_t[sl]), 0,
                                       tau=tau, generator=g, **kw)
        l_w, _, _ = common.fm_loss(dit, sub, projector(b_w[sl]), 0,
                                   tau=tau, eps=eps, **kw)
        l_n, _, _ = common.fm_loss(dit, sub,
                                   projector.null_tokens(sl.stop - sl.start),
                                   0, tau=tau, eps=eps, **kw)
        return float(l_c), float(l_w), float(l_n)

    tot = {"correct": 0.0, "wrong": 0.0, "null": 0.0}
    for s in range(0, B, micro):
        sl = slice(s, min(s + micro, B))
        l_c, l_w, l_n = triple(sl)
        w = (sl.stop - sl.start) / B
        tot["correct"] += l_c * w
        tot["wrong"] += l_w * w
        tot["null"] += l_n * w

    metrics = {"loss_correct": tot["correct"], "loss_wrong": tot["wrong"],
               "loss_null": tot["null"],
               "gap_wrong": tot["wrong"] - tot["correct"],
               "gap_null": tot["null"] - tot["correct"]}

    if sigma_grid:
        grid_micro = 1 if per_episode else micro
        metrics["sigma_grid"] = {}
        for sigma in sigma_grid:
            entry = {}
            for roi_only in ([False, True] if with_roi else [False]):
                cs, ws, ns = [], [], []
                for s in range(0, B, grid_micro):
                    sl = slice(s, min(s + grid_micro, B))
                    c_, w_, n_ = triple(sl, sigma=sigma, roi_only=roi_only)
                    cs.append(c_)
                    ws.append(w_)
                    ns.append(n_)
                agg = {
                    "loss_correct": sum(cs) / len(cs),
                    "gap_wrong": sum(ws) / len(ws) - sum(cs) / len(cs),
                    "gap_null": sum(ns) / len(ns) - sum(cs) / len(cs),
                }
                if per_episode:
                    agg["per_episode"] = {
                        "gap_wrong": [round(b - a, 6)
                                      for a, b in zip(cs, ws)],
                        "gap_null": [round(b - a, 6)
                                     for a, b in zip(cs, ns)],
                    }
                if roi_only:
                    entry["roi"] = agg
                else:
                    entry.update(agg)
            metrics["sigma_grid"][str(sigma)] = entry

    if was_training:
        dit.model.train()
        teacher.train()
        projector.train()
    return metrics


def run(cfg, cache, registry, normalizer, out_dir, steps=None, grad_accum=2,
        eval_interval=200, ckpt_interval=100, log_interval=10,
        rank_monitor_interval=10, val_cache=None, val_text_ctx=None,
        val_sigma_grid=(0.9, 0.99), val_n=48):
    """Phase-1 training loop with checkpoint/resume.

    ``val_cache``/``val_text_ctx`` (both required together) add a held-out
    paired eval each ``eval_interval``, written to ``eval_val_log.jsonl``
    (separate file: plots.py consumes eval_log.jsonl and must keep seeing
    train-split-only rows). The val eval rebinds ``common.fm_loss`` to a
    closure over the VAL cache's text context for its duration —
    ``load_text_ctx`` maps by row index into its own cache's prompt table,
    so serving val rows through the train closure would silently fetch
    wrong embeddings.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = open_train_log(out_dir)
    steps = steps or cfg.phase1_steps
    tr = cfg.train
    if (val_cache is None) != (val_text_ctx is None):
        raise ValueError("val_cache and val_text_ctx must be given together")

    # Data-parallel mode (torchrun): every rank owns one GPU and draws its
    # OWN batches; gradients are averaged in _sync_grads so parameter
    # trajectories stay bit-identical across ranks (same seed at build time
    # -> same init; same synced grads + deterministic AdamW -> same updates).
    # Rank 0 alone logs, evaluates, and checkpoints.
    rank, world = _dist_info()
    is_main = rank == 0

    common.set_seed(tr.seed)                    # identical init on all ranks
    batcher = GroupBatcher(cache, seed=tr.seed + 10007 * rank)
    teacher, decoder, projector, dit = build_all(cfg, registry)
    opt, sched = build_optimizer(cfg, teacher, decoder, projector, dit)
    rng = np.random.default_rng(tr.seed + 10007 * rank)
    trainable = [p for gp in opt.param_groups for p in gp["params"]]
    if world > 1:
        # diverge the per-rank noise streams (dropout, tau, eps) AFTER the
        # shared-init build; weight identity is maintained by grad averaging
        torch.manual_seed(tr.seed + 31 * rank + 1)
        torch.cuda.manual_seed_all(tr.seed + 31 * rank + 1)

    step, ema_fm = 0, None
    ck = latest_checkpoint(out_dir)
    if ck:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        dit.model.load_adaptive_state_dict(st["adaptive"])
        teacher.load_state_dict(st["teacher"])
        decoder.load_state_dict(st["decoder"])
        projector.load_state_dict(st["projector"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, ema_fm = st["step"], st["ema_fm"]
        if world == 1:
            torch.set_rng_state(st["rng"]["torch"])
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
            random.setstate(st["rng"]["python"])
            np.random.set_state(st["rng"]["numpy"])
        else:
            # the checkpoint holds rank 0's states; other ranks (and rank 0,
            # for cross-rank symmetry) reseed deterministically past the
            # consumed steps instead
            torch.manual_seed(tr.seed + 31 * rank + 1 + 977 * step)
            torch.cuda.manual_seed_all(tr.seed + 31 * rank + 1 + 977 * step)
        # advance the batch rng deterministically past consumed draws
        rng = np.random.default_rng(tr.seed + 10007 * rank + 977 * step)
        if is_main:
            print(f"resumed {ck} at step {step}", flush=True)

    dit.model.train()
    for m in (teacher, decoder, projector):
        m.train()

    while step < steps:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        acc = {"fm": 0.0, "meta": 0.0}
        conds, gap_vals = [], []
        for micro in range(grad_accum):
            batch = batcher.paired_batch(tr.batch_size)
            b_t = teacher.forward_batch(batch["rec"])
            l_meta = decoder.loss(b_t, batch["rec"])
            u = rng.random()
            if u < tr.p_null:
                cond = "null"
                tokens = projector.null_tokens(tr.batch_size)
                l_fm, tau, eps = common.fm_loss(dit, batch, tokens, 0)
            else:
                cond = "noisy" if u < tr.p_null + tr.p_noise else "clean"
                b_in = b_t
                if cond == "noisy":
                    b_in = b_t + tr.noise_scale * torch.randn_like(b_t)
                l_fm, tau, eps = common.fm_loss(dit, batch,
                                                projector(b_in), 0)
                # Diagnostic only, never optimized: wrong-code loss under
                # the SAME (tau, eps).  A positive gap means the physics
                # code is informative to the frozen-noise prediction.
                if (rank_monitor_interval and micro == 0 and is_main
                        and (step + 1) % rank_monitor_interval == 0):
                    with torch.no_grad():
                        b_w = teacher.forward_batch(batch["rec_wrong"])
                        l_w, _, _ = common.fm_loss(
                            dit, batch, projector(b_w), 0, tau=tau, eps=eps)
                    gap_vals.append(float(l_w) - float(l_fm))
            loss = tr.w_fm * l_fm + tr.w_meta * l_meta
            (loss / grad_accum).backward()
            conds.append(cond)
            acc["fm"] += float(l_fm) / grad_accum
            acc["meta"] += float(l_meta) / grad_accum
        if world > 1:
            _sync_grads(trainable, world)
        torch.nn.utils.clip_grad_norm_(trainable, tr.grad_clip)
        opt.step()
        sched.step()
        step += 1
        ema_fm = (acc["fm"] if ema_fm is None
                  else ema_fm + (acc["fm"] - ema_fm) * 2 / (EMA_SPAN + 1))

        if not is_main:
            # non-main ranks only train; skip logging, eval and checkpoints
            continue
        g_mean, g_max = dit.physics.gate_stats()
        b_norm = float(torch.stack(
            [p.detach().norm()
             for p in dit.model.lora_parameters()[1::2]]).mean())
        rank_gap = (round(sum(gap_vals) / len(gap_vals), 6)
                    if gap_vals else "")
        append_row(csv_path, phase="phase1", step=step,
                   wall_s=round(time.time() - t0, 2),
                   loss=round(acc["fm"] + acc["meta"], 6),
                   fm=round(acc["fm"], 6), ema_fm=round(ema_fm, 6),
                   rank_gap=rank_gap,
                   meta=round(acc["meta"], 6), cond="|".join(conds),
                   gate_mean=f"{g_mean:.6f}", gate_max=f"{g_max:.6f}",
                   lora_b_norm=f"{b_norm:.6f}",
                   lr_head=sched.get_last_lr()[0],
                   lr_wan=sched.get_last_lr()[1],
                   mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        if step % log_interval == 0:
            gap_text = f"{rank_gap}" if gap_vals else "-"
            print(f"p1 step {step}/{steps} fm {acc['fm']:.4f} "
                  f"ema {ema_fm:.4f} gap {gap_text} "
                  f"meta {acc['meta']:.4f} gates {g_mean:.4f}/{g_max:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

        if step % eval_interval == 0 or step == steps:
            m = paired_eval(cfg, cache, teacher, projector, dit,
                            seed=tr.seed + 1)
            with open(out_dir / "eval_log.jsonl", "a") as f:
                f.write(json.dumps({"phase": 1, "step": step,
                                    "metrics": m}) + "\n")
            print(f"[eval p1] {json.dumps(m)}", flush=True)
            if val_cache is not None:
                mv = _val_eval(cfg, val_cache, val_text_ctx, teacher,
                               projector, dit, tr, val_sigma_grid, val_n)
                with open(out_dir / "eval_val_log.jsonl", "a") as f:
                    f.write(json.dumps({"phase": 1, "step": step,
                                        "split": "val",
                                        "metrics": mv}) + "\n")
                print(f"[eval p1 val] gap_wrong {mv['gap_wrong']:.6f} "
                      f"gap_null {mv['gap_null']:.6f}", flush=True)
            save_checkpoint(out_dir, step, ema_fm, dit, teacher, decoder,
                            projector, opt, sched)
        elif step % ckpt_interval == 0:
            save_checkpoint(out_dir, step, ema_fm, dit, teacher, decoder,
                            projector, opt, sched)

    metrics, val_metrics = None, None
    if is_main:
        metrics = paired_eval(cfg, cache, teacher, projector, dit,
                              seed=tr.seed + 1)
        if val_cache is not None:
            val_metrics = _val_eval(cfg, val_cache, val_text_ctx, teacher,
                                    projector, dit, tr, val_sigma_grid,
                                    val_n)
    teacher.requires_grad_(False)
    decoder.requires_grad_(False)
    return {"teacher": teacher, "decoder": decoder, "projector": projector,
            "dit": dit, "metrics": metrics, "val_metrics": val_metrics,
            "cache": cache, "registry": registry, "normalizer": normalizer}


def _val_eval(cfg, val_cache, val_text_ctx, teacher, projector, dit, tr,
              sigma_grid, n):
    """Held-out paired eval under a val-cache fm_loss binding (see run())."""
    from . import backend as _backend

    orig = common.fm_loss
    common.fm_loss = _backend.make_real_fm_loss(
        dit, val_text_ctx, roi_lambda=getattr(tr, "roi_lambda", 0.0))
    try:
        return paired_eval(cfg, val_cache, teacher, projector, dit,
                           seed=tr.seed + 2, n=n, sigma_grid=sigma_grid,
                           per_episode=True, roi_variants=True)
    finally:
        common.fm_loss = orig
