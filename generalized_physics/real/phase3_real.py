"""Phase 3 (student substitution, 40/40/10/10 mixture) with resume.

Loss math identical to ``training/phase3_student_substitution.run``::

    l_distill = ||b_S - sg(b_T)||^2
    l_query   = decoder.loss(b_S, rec)
    tokens    ~ 40% teacher | 40% student | 10% corrupted teacher | 10% null
    loss      = l_distill + w_query_s * l_query + w_fm_s * l_fm(tokens)

Trainable: student (base LR) + adapters + LoRA (low LR, gates at 10x).
Teacher, decoder, projector frozen. The student-window endpoint is sampled per
step exactly as in the stock loop.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..data.counterfactual_dataset import GroupBatcher
from ..training import common
from ..training.phase3_student_substitution import evaluate as stock_evaluate
from .instrument import append_row, open_train_log


def save_checkpoint(out_dir, step, dit, student, opt):
    d = Path(out_dir) / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "adaptive": dit.model.adaptive_state_dict(),
        "student": student.state_dict(),
        "optimizer": opt.state_dict(),
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


def run(cfg, phase1, phase2, out_dir, steps=None, grad_accum=2,
        ckpt_interval=100, log_interval=10, gate_lr_mult=10.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = open_train_log(out_dir)
    steps = steps or cfg.phase3_steps
    tr = cfg.train

    common.set_seed(tr.seed + 4)
    cache = phase1["cache"]
    teacher, decoder = phase1["teacher"], phase1["decoder"]
    projector, dit = phase1["projector"], phase1["dit"]
    student = phase2["student"]
    projector.requires_grad_(False)
    batcher = GroupBatcher(cache, seed=tr.seed + 4)

    gates = dit.physics.gate_params()
    wan_side = [p for p in common.dedupe_params(
        dit.physics.non_gate_params() + dit.model.lora_parameters())
        if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": student.parameters(), "lr": tr.lr_teacher_student,
         "weight_decay": tr.weight_decay},
        {"params": wan_side, "lr": tr.lr_projector_adapters,
         "weight_decay": tr.weight_decay},
        {"params": gates, "lr": tr.lr_projector_adapters * gate_lr_mult,
         "weight_decay": 0.0},
    ], betas=(0.9, 0.95))
    rng = np.random.default_rng(tr.seed + 4)
    Tz = cache["z"].shape[2]
    prefix = cfg.dit.prefix_bins
    mix = np.cumsum([tr.mix_teacher, tr.mix_student, tr.mix_corrupt,
                     tr.mix_null])
    trainable = [p for gp in opt.param_groups for p in gp["params"]]

    step = 0
    ck = latest_checkpoint(out_dir)
    if ck:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        dit.model.load_adaptive_state_dict(st["adaptive"])
        student.load_state_dict(st["student"])
        opt.load_state_dict(st["optimizer"])
        step = st["step"]
        torch.set_rng_state(st["rng"]["torch"])
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        random.setstate(st["rng"]["python"])
        np.random.set_state(st["rng"]["numpy"])
        rng = np.random.default_rng(tr.seed + 4 + 977 * step)
        print(f"resumed {ck} at step {step}", flush=True)

    dit.model.train()
    student.train()
    while step < steps:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        acc = {"distill": 0.0, "query": 0.0, "fm": 0.0}
        conds = []
        for _ in range(grad_accum):
            batch = batcher.episode_batch(tr.batch_size)
            end = int(rng.integers(prefix, Tz + 1))
            win = batcher.window(batch, W=cfg.window, end=end)
            b_s = common.student_window_forward(student, win)["belief_final"]
            with torch.no_grad():
                b_t = teacher.forward_batch(batch["rec"])
            l_distill = ((b_s - b_t) ** 2).mean()
            l_query = decoder.loss(b_s, batch["rec"])
            u = rng.random()
            if u < mix[0]:
                cond, tokens = "teacher", projector(b_t)
            elif u < mix[1]:
                cond, tokens = "student", projector(b_s)
            elif u < mix[2]:
                cond = "corrupt"
                tokens = projector(b_t + tr.noise_scale
                                   * torch.randn_like(b_t))
            else:
                cond, tokens = "null", projector.null_tokens(tr.batch_size)
            l_fm, _, _ = common.fm_loss(dit, batch, tokens, prefix)
            loss = l_distill + tr.w_query_s * l_query + tr.w_fm_s * l_fm
            (loss / grad_accum).backward()
            conds.append(cond)
            acc["distill"] += float(l_distill) / grad_accum
            acc["query"] += float(l_query) / grad_accum
            acc["fm"] += float(l_fm) / grad_accum
        torch.nn.utils.clip_grad_norm_(trainable, tr.grad_clip)
        opt.step()
        step += 1

        g_mean, g_max = dit.physics.gate_stats()
        append_row(csv_path, phase="phase3", step=step,
                   wall_s=round(time.time() - t0, 2),
                   loss=round(acc["distill"] + acc["query"] + acc["fm"], 6),
                   fm=round(acc["fm"], 6), distill=round(acc["distill"], 6),
                   query=round(acc["query"], 6), cond="|".join(conds),
                   gate_mean=f"{g_mean:.6f}", gate_max=f"{g_max:.6f}",
                   lr_head=opt.param_groups[0]["lr"],
                   lr_wan=opt.param_groups[1]["lr"],
                   mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        if step % log_interval == 0:
            print(f"p3 step {step}/{steps} distill {acc['distill']:.4f} "
                  f"query {acc['query']:.4f} fm {acc['fm']:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if step % ckpt_interval == 0 or step == steps:
            save_checkpoint(out_dir, step, dit, student, opt)

    metrics = _eval_micro(cfg, cache, teacher, projector, dit, student,
                          seed=tr.seed + 5)
    with open(out_dir / "eval_log.jsonl", "a") as f:
        f.write(json.dumps({"phase": 3, "step": step,
                            "metrics": metrics}) + "\n")
    return {"student": student, "dit": dit, "metrics": metrics}


@torch.no_grad()
def _eval_micro(cfg, cache, teacher, projector, dit, student, seed=0, n=64,
                micro=8, sigma=None, roi_only=False, per_episode=False):
    """Stock oracle-gap-closure metric, micro-batched for the 14B.

    Extensions (defaults reproduce the historical dict, except that
    ``gap_closure`` is now None — undefined — when ``oracle_gap`` is below
    1e-4: dividing by a noise-floor denominator produced garbage ratios on
    the f1 arms and must never read as a result again):

    * ``sigma``: evaluate the triple at one FIXED sigma (the conditioning
      signal lives at sigma -> 1; the mixed average buries it);
    * ``roi_only``: ball-tube-restricted reduction, where the bound
      ``common.fm_loss`` supports it and the cache carries roi;
    * ``per_episode``: forces micro=1 and returns per-episode
      ``gap_teacher``/``gap_student`` (null minus conditioned) and
      ``gap_student_wrong`` lists so the caller can bootstrap CIs.

    ``roi`` is threaded into every chunk (the real fm_loss indexes
    ``batch["roi"]`` whenever it was built with ``roi_lambda > 0``; leaving
    it out crashed any ROI-weighted eval).

    A fourth condition guards against the presence-vs-null bias (muffling
    diagnosis: ANY non-null code beats the null code long before content
    matters, so student-vs-null passes for a random student): the student
    is also run on a SIBLING episode's window and its code substituted
    under the same (tau, eps). ``gap_student_wrong`` = wrong-video minus
    own-video student loss is positive only when the student's code
    carries episode content the DiT uses — the student analog of phase 1's
    ``gap_wrong``.
    """
    dit.model.eval()               # LoRA dropout off for paired comparison
    student.eval()
    batcher = GroupBatcher(cache, seed=seed)
    batch = batcher.episode_batch(min(n, len(cache["z"])))
    win = batcher.window(batch, W=cfg.window)
    b_s = common.student_window_forward(student, win)["belief_final"]
    b_t = teacher.forward_batch(batch["rec"])
    gid = cache["group_id"].numpy()
    rng_w = np.random.default_rng(seed + 1)
    wrong = []
    for k in batch["idx"].tolist():
        members = batcher.groups[gid[k]]
        wrong.append(int(rng_w.choice(members[members != k])))
    win_w = batcher.window(batcher._gather(np.asarray(wrong)), W=cfg.window)
    b_sw = common.student_window_forward(student, win_w)["belief_final"]
    g = torch.Generator().manual_seed(seed)
    B = len(batch["idx"])
    if per_episode:
        micro = 1
    kw = {"roi_only": True} if roi_only else {}
    tot = {"t": 0.0, "s": 0.0, "sw": 0.0, "n": 0.0}
    per = {"t": [], "s": [], "sw": [], "n": []}
    for s0 in range(0, B, micro):
        sl = slice(s0, min(s0 + micro, B))
        sub = {"z": batch["z"][sl], "idx": batch["idx"][sl]}
        if "roi" in batch:
            sub["roi"] = batch["roi"][sl]
        tau = (None if sigma is None
               else torch.full((sl.stop - sl.start,), float(sigma)))
        l_t, tau, eps = common.fm_loss(dit, sub, projector(b_t[sl]), 0,
                                       tau=tau, generator=g, **kw)
        l_s, _, _ = common.fm_loss(dit, sub, projector(b_s[sl]), 0,
                                   tau=tau, eps=eps, **kw)
        l_sw, _, _ = common.fm_loss(dit, sub, projector(b_sw[sl]), 0,
                                    tau=tau, eps=eps, **kw)
        l_n, _, _ = common.fm_loss(dit, sub,
                                   projector.null_tokens(sl.stop - sl.start),
                                   0, tau=tau, eps=eps, **kw)
        w = (sl.stop - sl.start) / B
        tot["t"] += float(l_t) * w
        tot["s"] += float(l_s) * w
        tot["sw"] += float(l_sw) * w
        tot["n"] += float(l_n) * w
        if per_episode:
            per["t"].append(float(l_t))
            per["s"].append(float(l_s))
            per["sw"].append(float(l_sw))
            per["n"].append(float(l_n))
    gap = tot["n"] - tot["t"]
    closure = (tot["n"] - tot["s"]) / gap if gap > 1e-4 else None
    out = {"loss_teacher": tot["t"], "loss_student": tot["s"],
           "loss_student_wrong": tot["sw"], "loss_null": tot["n"],
           "oracle_gap": gap, "gap_closure": closure,
           "gap_student_wrong": tot["sw"] - tot["s"]}
    if per_episode:
        out["per_episode"] = {
            "gap_teacher": [round(b - a, 6)
                            for a, b in zip(per["t"], per["n"])],
            "gap_student": [round(b - a, 6)
                            for a, b in zip(per["s"], per["n"])],
            "gap_student_wrong": [round(b - a, 6)
                                  for a, b in zip(per["s"], per["sw"])],
        }
    return out
