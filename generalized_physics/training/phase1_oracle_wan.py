"""Phase 1: metadata teacher + oracle Wan conditioning.

Trains the teacher, query decoder, shared projector, four physics adapters,
and the DiT LoRA matrices on paired counterfactual groups. The base DiT and
VAE stay frozen (LoRA is the only trainable Wan-side capacity besides the
adapters).

L_T = w_fm * L_FM(conditioned) + w_meta * L_meta.

The plan's shuffled-code rank hinge (eqs. 16-19) is deliberately NOT part
of the optimized objective any more: pushing L_FM(correct) below
L_FM(wrong) during training is circular with the correct/wrong gap used to
certify that the physics code has been learned. The gap is now measurement
only — ``evaluate`` computes the held-out correct/wrong/null separation
under shared (tau, eps) and the phase gate reads ``gap_wrong``/``gap_null``
from there. ``cfg.train.w_rank``/``rank_margin`` are ignored.
Corruption: p_null batches use the learned null code, p_noise batches add
small Gaussian noise to the teacher belief — preparing the adapters for
imperfect student codes.
"""

import numpy as np
import torch

from ..data.counterfactual_dataset import GroupBatcher, build_cache
from . import common


def paired_flow_losses(dit, projector, batch, b_correct, b_wrong,
                       prefix_bins, generator=None):
    """Correct/wrong/null FM losses under one shared (tau, eps)."""
    l_c, tau, eps = common.fm_loss(dit, batch, projector(b_correct),
                                   prefix_bins, generator=generator)
    l_w, _, _ = common.fm_loss(dit, batch, projector(b_wrong), prefix_bins,
                               tau=tau, eps=eps)
    with torch.no_grad():
        l_n, _, _ = common.fm_loss(
            dit, batch, projector.null_tokens(len(tau)), prefix_bins,
            tau=tau, eps=eps)
    return l_c, l_w, l_n


def run(cfg, cache=None, registry=None, normalizer=None):
    common.set_seed(cfg.train.seed)
    vae = common.build_vae(cfg)
    if cache is None:
        cache, registry, normalizer = build_cache(cfg, vae,
                                                  seed=cfg.train.seed)
    batcher = GroupBatcher(cache, seed=cfg.train.seed)
    teacher = common.build_teacher(cfg, registry)
    decoder = common.build_decoder(cfg, teacher)
    projector = common.build_projector(cfg)
    dit = common.build_dit(cfg)

    tr = cfg.train
    opt = torch.optim.AdamW([
        {"params": common.dedupe_params(
            list(teacher.parameters()) + list(decoder.parameters())),
         "lr": tr.lr_teacher_student},
        {"params": common.wan_side_params(cfg, dit, projector),
         "lr": tr.lr_projector_adapters},
    ], weight_decay=tr.weight_decay)
    rng = np.random.default_rng(tr.seed)
    prefix = cfg.dit.prefix_bins

    for step in range(cfg.phase1_steps):
        batch = batcher.paired_batch(tr.batch_size)
        b_t = teacher.forward_batch(batch["rec"])
        u = rng.random()
        if u < tr.p_null:
            tokens = projector.null_tokens(tr.batch_size)
        else:
            b_in = b_t
            if u < tr.p_null + tr.p_noise:
                b_in = b_t + tr.noise_scale * torch.randn_like(b_t)
            tokens = projector(b_in)
        l_fm, tau, eps = common.fm_loss(dit, batch, tokens, prefix)
        l_meta = decoder.loss(b_t, batch["rec"])
        loss = tr.w_fm * l_fm + tr.w_meta * l_meta
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], tr.grad_clip)
        opt.step()

    metrics = evaluate(cfg, cache, teacher, projector, dit,
                       seed=tr.seed + 1)
    # gate passed -> the teacher is frozen for every later phase
    teacher.requires_grad_(False)
    decoder.requires_grad_(False)
    return {"teacher": teacher, "decoder": decoder, "projector": projector,
            "dit": dit, "metrics": metrics, "cache": cache,
            "registry": registry, "normalizer": normalizer}


@torch.no_grad()
def evaluate(cfg, cache, teacher, projector, dit, seed=0, n=32):
    """Same-group correct/wrong/null separation under shared noise."""
    batcher = GroupBatcher(cache, seed=seed)
    batch = batcher.paired_batch(min(n, len(cache["z"])))
    b_t = teacher.forward_batch(batch["rec"])
    b_w = teacher.forward_batch(batch["rec_wrong"])
    g = torch.Generator().manual_seed(seed)
    l_c, l_w, l_n = paired_flow_losses(dit, projector, batch, b_t, b_w,
                                       cfg.dit.prefix_bins, generator=g)
    return {"loss_correct": float(l_c), "loss_wrong": float(l_w),
            "loss_null": float(l_n),
            "gap_wrong": float(l_w - l_c), "gap_null": float(l_n - l_c)}


if __name__ == "__main__":
    from ..config import tiny_config
    print(run(tiny_config())["metrics"])
