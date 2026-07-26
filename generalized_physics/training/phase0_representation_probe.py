"""Phase 0: representation ceiling and data validation (plan gate).

Trains the causal student plus a temporary metadata query probe on the
student belief, then answers, BEFORE any Wan training:
  * do post-event predictions beat a constant baseline (R^2 > 0)?
  * do the causal controls (time shuffle, action swap, one-bin action
    shift) materially degrade probe accuracy?

Proceed to Phase 1 only when both hold — otherwise the student is not
identifying physics from intervention-response history.
"""

import torch

from ..data.counterfactual_dataset import GroupBatcher, build_cache
from ..models.metadata_query_decoder import MetadataQueryDecoder
from ..models.metadata_records import RecordEmbedder
from . import common


def probe_loss(decoder, belief, rec):
    return decoder.loss(belief, rec)


def _control_variants(win, rng):
    """time-shuffled / action-swapped / action-shifted copies of a window."""
    W = win["z"].shape[2]
    perm = torch.randperm(W, generator=rng)
    shuffled = dict(win, z=win["z"][:, :, perm])
    swapped = dict(win, actions=win["actions"].roll(1, dims=0))
    shifted = dict(win, bin_index=(win["bin_index"] + 1).clamp(max=W - 1))
    return {"time_shuffle": shuffled, "action_swap": swapped,
            "action_shift": shifted}


def run(cfg, cache=None, registry=None, normalizer=None):
    common.set_seed(cfg.train.seed)
    vae = common.build_vae(cfg)
    if cache is None:
        cache, registry, normalizer = build_cache(cfg, vae,
                                                  seed=cfg.train.seed)
    batcher = GroupBatcher(cache, seed=cfg.train.seed)
    student = common.build_student(cfg)
    embedder = RecordEmbedder(registry, cfg.teacher.key_embed,
                              cfg.teacher.scope_embed, cfg.teacher.unit_embed)
    probe = MetadataQueryDecoder(embedder, cfg.teacher.belief_dim,
                                 hidden=cfg.teacher.record_width)
    opt = torch.optim.AdamW(
        list(student.parameters()) + list(probe.parameters()),
        lr=cfg.train.lr_teacher_student, weight_decay=cfg.train.weight_decay)

    for step in range(cfg.phase0_steps):
        batch = batcher.episode_batch(cfg.train.batch_size)
        win = batcher.window(batch, W=batch["z"].shape[2])   # full history
        out = common.student_window_forward(student, win)
        loss = probe_loss(probe, out["belief_final"], batch["rec"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(student.parameters()) + list(probe.parameters()),
            cfg.train.grad_clip)
        opt.step()

    # -- evaluation: post-event gain and causal controls -------------------
    student.eval()
    probe.eval()
    rng = torch.Generator().manual_seed(cfg.train.seed + 1)
    with torch.no_grad():
        batch = batcher.episode_batch(min(64, len(cache["z"])))
        win = batcher.window(batch, W=batch["z"].shape[2])
        out = common.student_window_forward(student, win)
        Tz = batch["z"].shape[2]
        per_bin = {}
        for t in (max(1, Tz // 4), Tz - 1):    # early vs late evidence
            per_bin[t] = float(probe_loss(probe, out["belief"][:, t],
                                          batch["rec"]))
        base = float(probe_loss(probe, out["belief_final"], batch["rec"]))
        controls = {}
        for name, w in _control_variants(win, rng).items():
            o = common.student_window_forward(student, w)
            controls[name] = float(probe_loss(probe, o["belief_final"],
                                              batch["rec"]))
    early, late = per_bin[max(1, Tz // 4)], per_bin[Tz - 1]
    metrics = {
        "probe_loss": base,
        "probe_early": early,
        "probe_late": late,
        "post_event_gain": early - late,
        "controls": controls,
        "controls_degrade": all(v > base for v in controls.values()),
    }
    return {"student": student, "probe": probe, "metrics": metrics,
            "cache": cache, "registry": registry, "normalizer": normalizer}


if __name__ == "__main__":
    from ..config import tiny_config
    res = run(tiny_config())
    print(res["metrics"])
