"""Phase 2 (student distillation) against a cached corpus, with resume.

Loss math is identical to ``training/phase2_student_distillation.run``::

    end  ~ U(2, Tz+1)                       # random causal prefix
    L_S  = ||b_S - sg(b_T)||^2 + w_query_s * L_query(b_S)

Reimplemented rather than instrumented for the same reason as
``phase1_real.py``: the stock loop has no checkpoint, no logging, no
held-out measurement, and its causal controls are wrong for this corpus:

  * ``query_action_swapped`` rolls the action tensor, an exact no-op on
    all-zero action caches (the rolling corpus has no actions);
  * the time shuffle permutes ``z`` but leaves ``delta`` in file order, so
    a ``use_delta_z`` student would keep reading correct motion from the
    dZ channels — here the control recomputes delta from the shuffled z.

Additions over stock: DiT-free reload of the frozen phase-1 heads from a
phase-1 checkpoint (``load_phase1_heads``), per-sample ROI-visibility loss
masking (windows whose bins never contain the tracked object carry no
signal; on the rolling cache this is measured to never trigger — kept as a
cheap generalization guard), deterministic full-val-split evaluation with
sibling paired accuracy + per-end prefix curves, and requeue-safe
checkpoint/resume. Appended eval jsonl files may repeat steps after a
requeue; consumers dedupe by step, keep-last.

The CI arithmetic on the returned per-group/per-episode lists lives in the
campaign runner (``roll_run_campaign.bootstrap_ci``) — importing it here
would be circular.
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


def load_phase1_heads(cfg, registry, ckpt_path, device, with_projector=False):
    """Rebuild the frozen phase-1 heads from a phase-1 ``trainer.pt``.

    DiT-free: builds teacher/decoder(/projector) from cfg + registry and
    loads their state dicts, so phase 2 never touches Wan, the text cache,
    or ``install_real_backend`` (whose rebinding would poison
    ``common.fm_loss`` for DiT-free phases). ``cfg`` must reproduce the
    phase-1 recipe or the state dicts will not load. The decoder shares the
    teacher's RecordEmbedder; loading teacher first, decoder second rewrites
    the shared ``embed.*`` tensors with identical values.

    Returns ``{"teacher", "decoder", ("projector",) "adaptive", "step",
    "path"}`` — heads on ``device``, ``.eval()``, gradients off;
    ``adaptive`` is the raw adapter+LoRA state dict for a phase-3 caller
    (kept here so the 127 MB checkpoint is read once).
    """
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    teacher = common.build_teacher(cfg, registry)
    teacher.load_state_dict(st["teacher"])
    decoder = common.build_decoder(cfg, teacher)
    decoder.load_state_dict(st["decoder"])
    out = {"teacher": teacher, "decoder": decoder, "step": st["step"],
           "adaptive": st["adaptive"], "path": str(ckpt_path)}
    if with_projector:
        projector = common.build_projector(cfg)
        projector.load_state_dict(st["projector"])
        out["projector"] = projector
    for k in ("teacher", "decoder", "projector"):
        if k in out:
            out[k] = out[k].to(device)
            out[k].eval()
            out[k].requires_grad_(False)
    return out


def build_optimizer(cfg, student):
    tr = cfg.train
    opt = torch.optim.AdamW(student.parameters(), lr=tr.lr_teacher_student,
                            weight_decay=tr.weight_decay, betas=(0.9, 0.95))
    warmup = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 1.0))
    return opt, sched


def save_checkpoint(out_dir, step, ema, student, opt, sched, use_delta_z):
    d = Path(out_dir) / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "ema": ema, "use_delta_z": bool(use_delta_z),
        "student": student.state_dict(),
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "python": random.getstate(),
                "numpy": np.random.get_state()},
    }, d / "trainer.pt")
    for old in sorted(Path(out_dir).glob("ckpt_*/trainer.pt"))[:-2]:
        old.unlink()
        old.parent.rmdir()
    print(f"saved {d}", flush=True)


def latest_checkpoint(out_dir):
    cks = sorted(Path(out_dir).glob("ckpt_*/trainer.pt"))
    return cks[-1] if cks else None


def visibility_from_roi(cache):
    """bool [N, Tz] — True where the latent bin contains any ROI cell.
    None for caches without roi masks (masking then disabled)."""
    if "roi" not in cache:
        return None
    return (cache["roi"] != 0).flatten(2).any(-1)


def window_visible(vis, idx, start, end):
    """bool [B]: does episode idx's window [start, end) contain a visible
    bin?"""
    return vis[idx.to(vis.device), start:end].any(1)


def _recompute_delta(z):
    d = torch.zeros_like(z)
    d[:, :, 1:] = z[:, :, 1:] - z[:, :, :-1]
    return d


def shuffled_window(win, perm):
    """Time-shuffle control that is honest for delta-fed students: permute
    the latent bins AND recompute delta from the shuffled sequence (the
    stock control leaves ``delta`` in file order). ``delta_valid`` marks
    only the first shuffled bin invalid — every other consecutive pair is a
    real (if scrambled) difference."""
    perm = perm.to(win["z"].device)
    z = win["z"][:, :, perm]
    dv = torch.ones_like(win["delta_valid"])
    dv[:, 0] = 0.0
    return dict(win, z=z, delta=_recompute_delta(z), delta_valid=dv)


def _shuffle_perm(n, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    if n > 1 and bool((perm == torch.arange(n)).all()):
        perm = torch.randperm(n, generator=g)
    return perm


def _beliefs(cfg, cache, student, idx, end, chunk=64, transform=None):
    """belief_final for episodes ``idx`` at prefix endpoint ``end``,
    chunked. Uses GroupBatcher only as a deterministic gather (its rng is
    never consulted by ``_gather``/``window``)."""
    batcher = GroupBatcher(cache, seed=0)
    outs = []
    for s in range(0, len(idx), chunk):
        b = batcher._gather(np.asarray(idx[s:s + chunk]))
        win = batcher.window(b, W=cfg.window, end=end)
        if transform is not None:
            win = transform(win)
        outs.append(common.student_window_forward(student, win)
                    ["belief_final"])
    return torch.cat(outs)


def _paired_acc(b_s, b_t, gid):
    """Sibling paired accuracy. For episode i with same-group siblings j:
    acc_i = mean_j[ ||b_S_i - b_T_i||^2 < ||b_S_i - b_T_j||^2 ].
    Returns (mean, per_episode list, per_group list). Groups always have
    >=2 members (load_cache drops singletons)."""
    d2 = torch.cdist(b_s, b_t).pow(2).cpu().numpy()
    own = np.diag(d2)
    n = len(gid)
    per_ep = np.zeros(n)
    for i in range(n):
        sib = np.nonzero((gid == gid[i]) & (np.arange(n) != i))[0]
        per_ep[i] = float((own[i] < d2[i, sib]).mean())
    uniq = sorted(set(gid.tolist()))
    per_group = [float(per_ep[gid == g].mean()) for g in uniq]
    return float(per_ep.mean()), [round(float(a), 4) for a in per_ep], \
        [round(a, 4) for a in per_group]


@torch.no_grad()
def probe_eval(cfg, cache, teacher, decoder, student, n=256, chunk=64):
    """Deterministic train-split probe (first n episodes, full trailing
    window) — the cheap between-eval trend line for eval_log.jsonl."""
    was = student.training
    student.eval()
    N = min(n, len(cache["z"]))
    idx = np.arange(N)
    Tz = cache["z"].shape[2]
    rec = {k: v[:N] for k, v in cache["rec_batch"].items()}
    b_t = teacher.forward_batch(rec)
    b_s = _beliefs(cfg, cache, student, idx, Tz, chunk)
    out = {"n": N,
           "distill_mse": float(((b_s - b_t) ** 2).mean()),
           "query_loss": float(decoder.loss(b_s, rec))}
    if was:
        student.train()
    return out


@torch.no_grad()
def val_eval(cfg, val_cache, teacher, decoder, student, chunk=64,
             shuffle_seed=0):
    """Deterministic full-split evaluation (every episode exactly once; the
    only randomness is the fixed-seed shuffle permutation).

    Returns per-episode/per-group sibling-accuracy lists for the caller to
    bootstrap, the fixed-perm time-shuffle control (delta recomputed), a
    per-end prefix curve, and — because the primary gate anchors at end=Tz
    where the ball is slowest and sometimes gone — the best-end accuracy.
    ``query_action_swapped`` is computed only when the action stream is
    non-degenerate; on the rolling corpus it is identically zero and the
    control is meaningless (reported as None).
    """
    was = student.training
    student.eval()
    N = len(val_cache["z"])
    Tz = val_cache["z"].shape[2]
    idx = np.arange(N)
    gid = val_cache["group_id"].cpu().numpy()
    rec = val_cache["rec_batch"]

    b_t = teacher.forward_batch(rec)
    var_bt = float(b_t.var(0, unbiased=False).mean().clamp_min(1e-8))
    b_s = _beliefs(cfg, val_cache, student, idx, Tz, chunk)
    distill = float(((b_s - b_t) ** 2).mean())
    query = float(decoder.loss(b_s, rec))
    acc, per_ep, per_group = _paired_acc(b_s, b_t, gid)

    W = min(cfg.window, Tz)
    perm = _shuffle_perm(W, shuffle_seed)
    b_shuf = _beliefs(cfg, val_cache, student, idx, Tz, chunk,
                      transform=lambda w: shuffled_window(w, perm))
    query_shuffled = float(decoder.loss(b_shuf, rec))

    query_swapped = None
    if float(val_cache["actions"].abs().max()) > 0:
        b_swap = _beliefs(
            cfg, val_cache, student, idx, Tz, chunk,
            transform=lambda w: dict(w, actions=w["actions"].roll(1, dims=0)))
        query_swapped = float(decoder.loss(b_swap, rec))

    curve = []
    for end in range(2, Tz + 1):
        b_e = _beliefs(cfg, val_cache, student, idx, end, chunk)
        acc_e, _, _ = _paired_acc(b_e, b_t, gid)
        curve.append({"end": end,
                      "distill_mse": round(float(((b_e - b_t) ** 2).mean()),
                                           6),
                      "paired_acc": round(acc_e, 4)})
    best = max(curve, key=lambda c: c["paired_acc"])

    vis = visibility_from_roi(val_cache)
    acc_visible, n_visible = None, None
    if vis is not None:
        keep = window_visible(vis, torch.as_tensor(idx), max(0, Tz - W), Tz)
        keep_np = keep.cpu().numpy()
        n_visible = int(keep_np.sum())
        if 0 < n_visible:
            acc_visible = float(np.asarray(per_ep)[keep_np].mean())

    if was:
        student.train()
    return {
        "n": N, "distill_mse": distill,
        "distill_mse_norm": distill / var_bt, "var_bt": var_bt,
        "query_loss": query, "query_time_shuffled": query_shuffled,
        "query_action_swapped": query_swapped,
        "paired_acc": acc, "per_episode_acc": per_ep,
        "per_group_acc": per_group,
        "paired_acc_visible": acc_visible, "n_visible": n_visible,
        "prefix_curve": curve,
        "best_end": {"end": best["end"], "paired_acc": best["paired_acc"]},
    }


def run(cfg, cache, teacher, decoder, out_dir, steps=None, eval_interval=200,
        ckpt_interval=500, log_interval=10, val_cache=None, vis_mask=True,
        student=None):
    """Phase-2 training loop with checkpoint/resume (single process).

    Returns a superset of the stock dict — ``{"student", "metrics",
    "val_metrics"}`` — so phase 3 chains via ``["student"]`` unchanged.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = open_train_log(out_dir)
    steps = steps or cfg.phase2_steps
    tr = cfg.train
    device = cache["delta_valid"].device

    common.set_seed(tr.seed + 2)
    batcher = GroupBatcher(cache, seed=tr.seed + 2)
    rng = np.random.default_rng(tr.seed + 2)
    if student is None:
        student = common.build_student(
            cfg, action_dim=int(cache["actions"].shape[-1]))
    student = student.to(device)
    opt, sched = build_optimizer(cfg, student)
    vis = visibility_from_roi(cache) if vis_mask else None
    Tz = cache["z"].shape[2]
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[p2] student {n_params / 1e6:.2f}M params "
          f"use_delta_z={cfg.student.use_delta_z} device={device} "
          f"train_eps={len(cache['z'])} Tz={Tz} "
          f"vis_mask={'on' if vis is not None else 'off'}", flush=True)

    step, ema = 0, None
    ck = latest_checkpoint(out_dir)
    if ck:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        if st["use_delta_z"] != cfg.student.use_delta_z:
            raise RuntimeError(
                f"checkpoint {ck} has use_delta_z={st['use_delta_z']} but "
                f"cfg.student.use_delta_z={cfg.student.use_delta_z} — "
                "resuming the wrong variant into this run dir")
        student.load_state_dict(st["student"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, ema = st["step"], st["ema"]
        torch.set_rng_state(st["rng"]["torch"])
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        random.setstate(st["rng"]["python"])
        np.random.set_state(st["rng"]["numpy"])
        # advance the batch/end rng deterministically past consumed draws
        rng = np.random.default_rng(tr.seed + 2 + 977 * step)
        batcher = GroupBatcher(cache, seed=tr.seed + 2 + 977 * step)
        print(f"resumed {ck} at step {step}", flush=True)

    teacher.eval()
    decoder.eval()
    student.train()

    while step < steps:
        t0 = time.time()
        for _attempt in range(20):
            batch = batcher.episode_batch(tr.batch_size)
            end = int(rng.integers(2, Tz + 1))
            keep = None
            if vis is not None:
                keep = window_visible(vis, batch["idx"],
                                      max(0, end - cfg.window), end)
                if not bool(keep.any()):
                    continue
            break
        else:
            raise RuntimeError("20 consecutive fully-invisible batches — "
                               "roi masks and window geometry disagree")
        win = batcher.window(batch, W=cfg.window, end=end)
        b_s = common.student_window_forward(student, win)["belief_final"]
        with torch.no_grad():
            b_t = teacher.forward_batch(batch["rec"])
        rec = batch["rec"]
        n_kept = len(batch["idx"])
        if keep is not None and not bool(keep.all()):
            b_s, b_t = b_s[keep], b_t[keep]
            rec = {k: v[keep] for k, v in rec.items()}
            n_kept = int(keep.sum())
        l_distill = ((b_s - b_t) ** 2).mean()
        l_query = decoder.loss(b_s, rec)
        loss = l_distill + tr.w_query_s * l_query
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), tr.grad_clip)
        opt.step()
        sched.step()
        step += 1
        loss_f = float(loss.detach())
        ema = (loss_f if ema is None
               else ema + (loss_f - ema) * 2 / (EMA_SPAN + 1))

        append_row(csv_path, phase="phase2", step=step,
                   wall_s=round(time.time() - t0, 2),
                   loss=round(loss_f, 6),
                   distill=round(float(l_distill.detach()), 6),
                   query=round(float(l_query.detach()), 6),
                   ema_fm=round(ema, 6),
                   cond=f"vis={n_kept}/{len(batch['idx'])}|end={end}",
                   lr_head=sched.get_last_lr()[0],
                   mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)
                   if torch.cuda.is_available() else "")
        if step % log_interval == 0:
            print(f"p2 step {step}/{steps} loss {loss_f:.4f} "
                  f"ema {ema:.4f} distill {float(l_distill.detach()):.4f} "
                  f"query {float(l_query.detach()):.4f} "
                  f"({time.time() - t0:.2f}s)", flush=True)

        if step % eval_interval == 0 or step == steps:
            m = probe_eval(cfg, cache, teacher, decoder, student)
            with open(out_dir / "eval_log.jsonl", "a") as f:
                f.write(json.dumps({"phase": 2, "step": step,
                                    "metrics": m}) + "\n")
            print(f"[eval p2] {json.dumps(m)}", flush=True)
            if val_cache is not None:
                mv = val_eval(cfg, val_cache, teacher, decoder, student)
                with open(out_dir / "eval_val_log.jsonl", "a") as f:
                    f.write(json.dumps({"phase": 2, "step": step,
                                        "split": "val",
                                        "metrics": mv}) + "\n")
                print(f"[eval p2 val] paired_acc {mv['paired_acc']:.4f} "
                      f"(best end {mv['best_end']['end']} "
                      f"{mv['best_end']['paired_acc']:.4f}) "
                      f"distill_norm {mv['distill_mse_norm']:.4f} "
                      f"shuffle {mv['query_time_shuffled']:.4f} "
                      f"vs {mv['query_loss']:.4f}", flush=True)
            save_checkpoint(out_dir, step, ema, student, opt, sched,
                            cfg.student.use_delta_z)
        elif step % ckpt_interval == 0:
            save_checkpoint(out_dir, step, ema, student, opt, sched,
                            cfg.student.use_delta_z)

    metrics = probe_eval(cfg, cache, teacher, decoder, student)
    val_metrics = (val_eval(cfg, val_cache, teacher, decoder, student)
                   if val_cache is not None else None)
    return {"student": student, "metrics": metrics,
            "val_metrics": val_metrics}
