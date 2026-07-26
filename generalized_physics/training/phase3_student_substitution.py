"""Phase 3: student substitution and mixed fine-tuning (plan eqs. 23-25).

Wan is conditioned with the mixture 40% teacher / 40% student / 10%
corrupted teacher / 10% null code. The student-conditioned flow loss is
applied at ONE sampled prefix endpoint per batch (never every endpoint).
Trainable: student (base LR) + physics adapters + DiT LoRA (low LR), with
teacher-conditioned batches retained as the anchor. Teacher, decoder, and
projector stay frozen.

L_sub = L_distill + 0.5 * L_query^S + 0.25 * L_FM^S
"""

import numpy as np
import torch

from ..data.counterfactual_dataset import GroupBatcher
from . import common


def run(cfg, phase1, phase2):
    common.set_seed(cfg.train.seed + 4)
    cache = phase1["cache"]
    teacher, decoder = phase1["teacher"], phase1["decoder"]
    projector, dit = phase1["projector"], phase1["dit"]
    student = phase2["student"]
    projector.requires_grad_(False)
    batcher = GroupBatcher(cache, seed=cfg.train.seed + 4)
    tr = cfg.train
    opt = torch.optim.AdamW([
        {"params": student.parameters(), "lr": tr.lr_teacher_student},
        {"params": [p for p in common.wan_side_params(cfg, dit, projector)
                    if p.requires_grad],     # adapters + LoRA; proj frozen
         "lr": tr.lr_projector_adapters},
    ], weight_decay=tr.weight_decay)
    rng = np.random.default_rng(tr.seed + 4)
    Tz = cache["z"].shape[2]
    prefix = cfg.dit.prefix_bins
    mix = np.cumsum([tr.mix_teacher, tr.mix_student, tr.mix_corrupt,
                     tr.mix_null])

    for step in range(cfg.phase3_steps):
        batch = batcher.episode_batch(tr.batch_size)
        end = int(rng.integers(prefix, Tz + 1))
        win = batcher.window(batch, W=cfg.window, end=end)
        out = common.student_window_forward(student, win)
        b_s = out["belief_final"]
        with torch.no_grad():
            b_t = teacher.forward_batch(batch["rec"])

        l_distill = ((b_s - b_t) ** 2).mean()
        l_query = decoder.loss(b_s, batch["rec"])
        u = rng.random()
        if u < mix[0]:
            tokens = projector(b_t)
        elif u < mix[1]:
            tokens = projector(b_s)             # gradients reach the student
        elif u < mix[2]:
            tokens = projector(b_t + tr.noise_scale * torch.randn_like(b_t))
        else:
            tokens = projector.null_tokens(tr.batch_size)
        l_fm, _, _ = common.fm_loss(dit, batch, tokens, prefix)
        loss = l_distill + tr.w_query_s * l_query + tr.w_fm_s * l_fm
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], tr.grad_clip)
        opt.step()

    metrics = evaluate(cfg, cache, teacher, projector, dit, student,
                       seed=tr.seed + 5)
    return {"student": student, "dit": dit, "metrics": metrics}


@torch.no_grad()
def evaluate(cfg, cache, teacher, projector, dit, student, seed=0, n=32):
    """Oracle-gap closure: no-physics -> student -> teacher, shared noise."""
    batcher = GroupBatcher(cache, seed=seed)
    batch = batcher.episode_batch(min(n, len(cache["z"])))
    win = batcher.window(batch, W=cfg.window)
    b_s = common.student_window_forward(student, win)["belief_final"]
    b_t = teacher.forward_batch(batch["rec"])
    g = torch.Generator().manual_seed(seed)
    prefix = cfg.dit.prefix_bins
    l_t, tau, eps = common.fm_loss(dit, batch, projector(b_t), prefix,
                                   generator=g)
    l_s, _, _ = common.fm_loss(dit, batch, projector(b_s), prefix,
                               tau=tau, eps=eps)
    l_n, _, _ = common.fm_loss(dit, batch,
                               projector.null_tokens(len(batch["idx"])),
                               prefix, tau=tau, eps=eps)
    gap = float(l_n - l_t)
    closure = float((l_n - l_s) / gap) if abs(gap) > 1e-9 else 0.0
    return {"loss_teacher": float(l_t), "loss_student": float(l_s),
            "loss_null": float(l_n), "oracle_gap": gap,
            "gap_closure": closure}


if __name__ == "__main__":
    from ..config import tiny_config
    from . import phase1_oracle_wan, phase2_student_distillation
    cfg = tiny_config()
    p1 = phase1_oracle_wan.run(cfg)
    p2 = phase2_student_distillation.run(cfg, p1)
    print(run(cfg, p1, p2)["metrics"])
