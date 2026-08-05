"""Stage 2 (action-token training) on the rbi cache, with resume.

Runs AFTER the stage-1 (physics) gate: everything stage 1 trained — teacher,
decoder, projector, physics adapters + gates, LoRA — is loaded frozen, and
only the new ``ActionTokenEncoder`` + action adapter bank train (zero-init
gates in a dedicated 10x-LR group, the passing recipe's cure for the
gate-LR race, tex B2). The pairing INVERTS stage 1: the wrong donor shares
the velocity state and swaps the action plan (``action_group_id``), so the
paired gap isolates the action stream exactly as stage 1 isolated velocity.

Loss is plain unweighted flow matching: the arm is the action signal and it
dominates the frame, so no ROI weighting applies (the ball ROI is the wrong
footprint). No metadata loss (the teacher is frozen), no rank hinge (the
wrong-action gap is a monitor-only diagnostic here, as in phase 1).

Unlike phase 1, there are no null-conditioning TRAINING draws by default
(``p_null_action=0``): phase 1's null branch trains a learned null code,
but stage 2's null is the action pathway bypassed — exactly the frozen
stage-1 model, which nothing here can improve. Such a draw touches no
trainable parameter (its loss has no grad_fn — backward is guarded), so it
is pure wasted compute; the eval triple still measures the bypass null for
``gap_null_action``.

A new module rather than a parametrization of ``phase1_real.run``: stage 2
inverts the optimizer groups, drops the meta loss, changes pairing and
checkpoint keys — threading that through the shared trainer would touch a
file the passing campaigns depend on. The proven skeleton (DDP grad
averaging, EMA, CSV/JSONL logging, keep-last-2 checkpoint/resume) is
copied; ``_dist_info``/``_sync_grads`` are imported from it.

Frozen-stage integrity: a fingerprint (per-module sum of |params|) of the
frozen stack is taken at start and re-checked at every eval — a wiring
mistake must never silently train physics.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..data.counterfactual_dataset import GroupBatcher
from ..wan.dit_lora import lora_modules
from .instrument import append_row, open_train_log
from .phase1_real import _dist_info, _sync_grads

EMA_SPAN = 100


class ActionPairBatcher(GroupBatcher):
    """Wrong donor = the opposite-action sibling; also gathers its actions.

    The stock ``paired_batch`` keeps only the donor's records — useless for
    an action swap, where the records (velocity) are identical by
    construction and the ACTIONS are what differs.
    """

    def paired_batch(self, batch_size):
        batch = self.episode_batch(batch_size)
        gid = self.c["group_id"].numpy()
        wrong = []
        for k in batch["idx"].tolist():
            members = self.groups[gid[k]]
            wrong.append(int(self.rng.choice(members[members != k])))
        wrong = torch.as_tensor(wrong, dtype=torch.long)
        batch["wrong_idx"] = wrong
        batch["actions_wrong"] = self.c["actions"][wrong]
        return batch


def build_optimizer(cfg, action_encoder, bank, gate_lr_mult=10.0):
    tr = cfg.train
    groups = [
        {"params": list(action_encoder.parameters())
                   + bank.non_gate_params(),
         "lr": tr.lr_projector_adapters, "weight_decay": tr.weight_decay},
        {"params": bank.gate_params(),
         "lr": tr.lr_projector_adapters * gate_lr_mult, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    warmup = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 1.0))
    return opt, sched


def save_checkpoint(out_dir, step, ema_fm, action_encoder, bank, opt, sched,
                    p1_ckpt, p1_step):
    d = Path(out_dir) / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "ema_fm": ema_fm,
        "action_encoder": action_encoder.state_dict(),
        "action_bank": bank.state_dict(),
        "block_idx": list(bank.block_idx),
        "p1_ckpt": str(p1_ckpt), "p1_step": p1_step,
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "python": random.getstate(),
                "numpy": np.random.get_state()},
    }, d / "trainer.pt")
    # frozen physics/LoRA are never re-saved: re-loadable from p1_ckpt
    for old in sorted(Path(out_dir).glob("ckpt_*/trainer.pt"))[:-2]:
        old.unlink()
        old.parent.rmdir()
    print(f"saved {d}", flush=True)


def latest_checkpoint(out_dir):
    cks = sorted(Path(out_dir).glob("ckpt_*/trainer.pt"))
    return cks[-1] if cks else None


def frozen_fingerprint(teacher, projector, dit):
    """Cheap exact-drift detector over the frozen stack. Values are pure
    functions of the (frozen) tensors, so any change is a training leak."""
    def s(params):
        return float(sum(p.detach().float().abs().sum().item()
                         for p in params))
    return {
        "teacher": s(teacher.parameters()),
        "projector": s(projector.parameters()),
        "physics_bank": s(dit.model.physics.parameters()),
        "lora": s(dit.model.lora_parameters()),
    }


def _relora_eval(dit):
    """R5: the model runs train() for gradient checkpointing, but the frozen
    LoRA's dropout (p=0.05) must stay off — re-assert after any mode flip."""
    for m in lora_modules(dit.model.dit.blocks):
        m.eval()


@torch.no_grad()
def paired_action_eval(cache, teacher, projector, dit, fm, seed=0, n=64,
                       micro=8, sigma_grid=(), per_episode=False):
    """Correct-action / wrong-action / no-action losses, shared (tau, eps).

    Same metric-dict shape as ``phase1_real.paired_eval`` (gap_wrong,
    gap_null, sigma_grid with per-episode lists) so the campaign's CI
    helpers apply unchanged — but full-frame only: the ball ROI has no
    meaning for an arm swap.
    """
    was_training = dit.model.training
    dit.model.eval()

    batcher = ActionPairBatcher(cache, seed=seed)
    batch = batcher.paired_batch(min(n, len(cache["z"])))
    b_t = teacher.forward_batch(batch["rec"])
    tokens = projector(b_t)
    g = torch.Generator().manual_seed(seed)
    B = len(batch["idx"])

    def triple(sl, sigma=None):
        sub = {"z": batch["z"][sl], "idx": batch["idx"][sl]}
        tau = (None if sigma is None
               else torch.full((sl.stop - sl.start,), float(sigma)))
        l_c, tau, eps = fm(dit, sub, tokens[sl], 0, tau=tau, generator=g,
                           action_seq=batch["actions"][sl])
        l_w, _, _ = fm(dit, sub, tokens[sl], 0, tau=tau, eps=eps,
                       action_seq=batch["actions_wrong"][sl])
        l_n, _, _ = fm(dit, sub, tokens[sl], 0, tau=tau, eps=eps,
                       action_seq=None)
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
            cs, ws, ns = [], [], []
            for s in range(0, B, grid_micro):
                sl = slice(s, min(s + grid_micro, B))
                c_, w_, n_ = triple(sl, sigma=sigma)
                cs.append(c_)
                ws.append(w_)
                ns.append(n_)
            entry = {
                "loss_correct": sum(cs) / len(cs),
                "gap_wrong": sum(ws) / len(ws) - sum(cs) / len(cs),
                "gap_null": sum(ns) / len(ns) - sum(cs) / len(cs),
            }
            if per_episode:
                entry["per_episode"] = {
                    "gap_wrong": [round(b - a, 6) for a, b in zip(cs, ws)],
                    "gap_null": [round(b - a, 6) for a, b in zip(cs, ns)],
                }
            metrics["sigma_grid"][str(sigma)] = entry

    if was_training:
        dit.model.train()
    _relora_eval(dit)
    return metrics


def run(cfg, cache, teacher, projector, dit, fm, action_encoder, bank,
        p1_ckpt, p1_step, out_dir, steps=None, grad_accum=2,
        eval_interval=1000, ckpt_interval=100, log_interval=10,
        action_gap_interval=10, p_null_action=0.0, val_cache=None,
        val_fm=None, val_sigma_grid=(0.9, 0.99), val_n=48):
    """Stage-2 training loop with checkpoint/resume.

    ``fm``/``val_fm`` are ``rbi_action.make_action_fm_loss`` closures bound
    to the TRAIN and VAL text contexts respectively (the row-index trap of
    ``load_text_ctx`` applies here exactly as in phase 1 — never serve val
    rows through the train closure). ``teacher``/``projector``/``dit``
    arrive frozen; only ``action_encoder`` + ``bank`` own the optimizer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = open_train_log(out_dir)
    tr = cfg.train
    if (val_cache is None) != (val_fm is None):
        raise ValueError("val_cache and val_fm must be given together")

    rank, world = _dist_info()
    is_main = rank == 0

    fingerprint = frozen_fingerprint(teacher, projector, dit)
    n_trainable = sum(p.numel() for p in action_encoder.parameters()) \
        + sum(p.numel() for p in bank.parameters())
    if is_main:
        print(f"stage-2 trainables: encoder+bank {n_trainable/1e6:.2f}M "
              f"params; frozen fingerprint {json.dumps(fingerprint)}",
              flush=True)

    batcher = ActionPairBatcher(cache, seed=tr.seed + 10007 * rank)
    opt, sched = build_optimizer(cfg, action_encoder, bank)
    rng = np.random.default_rng(tr.seed + 10007 * rank)
    trainable = [p for gp in opt.param_groups for p in gp["params"]]
    if world > 1:
        # diverge per-rank noise AFTER the shared-init build (campaign seeds
        # torch before constructing encoder/bank); weight identity across
        # ranks is maintained by grad averaging
        torch.manual_seed(tr.seed + 31 * rank + 1)
        torch.cuda.manual_seed_all(tr.seed + 31 * rank + 1)

    step, ema_fm = 0, None
    ck = latest_checkpoint(out_dir)
    if ck:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        if st["p1_ckpt"] != str(p1_ckpt):
            raise RuntimeError(f"resume mismatch: checkpoint was trained on "
                               f"{st['p1_ckpt']}, this run loads {p1_ckpt}")
        action_encoder.load_state_dict(st["action_encoder"])
        bank.load_state_dict(st["action_bank"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, ema_fm = st["step"], st["ema_fm"]
        if world == 1:
            torch.set_rng_state(st["rng"]["torch"])
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
            random.setstate(st["rng"]["python"])
            np.random.set_state(st["rng"]["numpy"])
        else:
            torch.manual_seed(tr.seed + 31 * rank + 1 + 977 * step)
            torch.cuda.manual_seed_all(tr.seed + 31 * rank + 1 + 977 * step)
        rng = np.random.default_rng(tr.seed + 10007 * rank + 977 * step)
        if is_main:
            print(f"resumed {ck} at step {step}", flush=True)

    dit.model.train()             # gradient checkpointing needs train()
    _relora_eval(dit)             # ...but frozen-LoRA dropout stays off (R5)
    action_encoder.train()
    bank.train()

    steps = steps or 2000
    while step < steps:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        acc_fm = 0.0
        conds, gap_vals = [], []
        for micro in range(grad_accum):
            batch = batcher.paired_batch(tr.batch_size)
            with torch.no_grad():
                tokens = projector(teacher.forward_batch(batch["rec"]))
            u = rng.random()
            if u < p_null_action:
                cond = "null"
                l_fm, tau, eps = fm(dit, batch, tokens, 0, action_seq=None)
            else:
                cond = "act"
                l_fm, tau, eps = fm(dit, batch, tokens, 0,
                                    action_seq=batch["actions"])
                # Diagnostic only, never optimized: wrong-ACTION loss under
                # the SAME (tau, eps); positive gap = the action tokens are
                # informative to the frozen-noise prediction.
                if (action_gap_interval and micro == 0 and is_main
                        and (step + 1) % action_gap_interval == 0):
                    with torch.no_grad():
                        l_w, _, _ = fm(dit, batch, tokens, 0, tau=tau,
                                       eps=eps,
                                       action_seq=batch["actions_wrong"])
                    gap_vals.append(float(l_w) - float(l_fm.detach()))
            if l_fm.requires_grad:
                # a null-action draw runs the fully frozen stage-1 model:
                # its loss has no graph and its gradient is exactly zero
                (l_fm / grad_accum).backward()
            conds.append(cond)
            acc_fm += float(l_fm.detach()) / grad_accum
        if world > 1:
            _sync_grads(trainable, world)
        torch.nn.utils.clip_grad_norm_(trainable, tr.grad_clip)
        opt.step()
        sched.step()
        step += 1
        ema_fm = (acc_fm if ema_fm is None
                  else ema_fm + (acc_fm - ema_fm) * 2 / (EMA_SPAN + 1))

        if not is_main:
            continue
        g_mean, g_max = bank.gate_stats()
        rank_gap = (round(sum(gap_vals) / len(gap_vals), 6)
                    if gap_vals else "")
        append_row(csv_path, phase="action", step=step,
                   wall_s=round(time.time() - t0, 2),
                   loss=round(acc_fm, 6), fm=round(acc_fm, 6),
                   ema_fm=round(ema_fm, 6), rank_gap=rank_gap,
                   cond="|".join(conds),
                   gate_mean=f"{g_mean:.6f}", gate_max=f"{g_max:.6f}",
                   lr_head=sched.get_last_lr()[0],
                   lr_wan=sched.get_last_lr()[1],
                   mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        if step % log_interval == 0:
            gap_text = f"{rank_gap}" if gap_vals else "-"
            print(f"act step {step}/{steps} fm {acc_fm:.4f} "
                  f"ema {ema_fm:.4f} gap {gap_text} "
                  f"gates {g_mean:.4f}/{g_max:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

        if step % eval_interval == 0 or step == steps:
            now = frozen_fingerprint(teacher, projector, dit)
            if now != fingerprint:
                raise RuntimeError(f"frozen stack drifted: {now} != "
                                   f"{fingerprint} — a stage-1 module is "
                                   "receiving gradient")
            m = paired_action_eval(cache, teacher, projector, dit, fm,
                                   seed=tr.seed + 1)
            with open(out_dir / "eval_log.jsonl", "a") as f:
                f.write(json.dumps({"phase": "action", "step": step,
                                    "metrics": m}) + "\n")
            print(f"[eval act] {json.dumps(m)}", flush=True)
            if val_cache is not None:
                mv = paired_action_eval(val_cache, teacher, projector, dit,
                                        val_fm, seed=tr.seed + 2, n=val_n,
                                        sigma_grid=val_sigma_grid,
                                        per_episode=True)
                with open(out_dir / "eval_val_log.jsonl", "a") as f:
                    f.write(json.dumps({"phase": "action", "step": step,
                                        "split": "val",
                                        "metrics": mv}) + "\n")
                print(f"[eval act val] gap_wrong {mv['gap_wrong']:.6f} "
                      f"gap_null {mv['gap_null']:.6f}", flush=True)
            save_checkpoint(out_dir, step, ema_fm, action_encoder, bank,
                            opt, sched, p1_ckpt, p1_step)
        elif step % ckpt_interval == 0:
            save_checkpoint(out_dir, step, ema_fm, action_encoder, bank,
                            opt, sched, p1_ckpt, p1_step)

    metrics = val_metrics = None
    if is_main:
        metrics = paired_action_eval(cache, teacher, projector, dit, fm,
                                     seed=tr.seed + 1)
        if val_cache is not None:
            val_metrics = paired_action_eval(val_cache, teacher, projector,
                                             dit, val_fm, seed=tr.seed + 2,
                                             n=val_n,
                                             sigma_grid=val_sigma_grid,
                                             per_episode=True)
        now = frozen_fingerprint(teacher, projector, dit)
        if now != fingerprint:
            raise RuntimeError("frozen stack drifted by end of training")
    return {"metrics": metrics, "val_metrics": val_metrics,
            "frozen_fingerprint": fingerprint,
            "n_trainable": n_trainable}
