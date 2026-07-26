"""Phase 2: distill the causal student from the frozen teacher (plan eqs.
20-22).

For each episode and a random causal prefix endpoint:
    L_S = ||b_S - sg(b_T)||^2 + 0.5 * L_query(b_S)
The teacher, query decoder, projector, adapters, LoRA, and Wan base all
stay frozen; only the student trains. Validation runs the causal controls
(time shuffle / action swap) every cycle.
"""

import numpy as np
import torch

from ..data.counterfactual_dataset import GroupBatcher
from . import common


def run(cfg, phase1, student=None):
    common.set_seed(cfg.train.seed + 2)
    cache = phase1["cache"]
    teacher, decoder = phase1["teacher"], phase1["decoder"]
    batcher = GroupBatcher(cache, seed=cfg.train.seed + 2)
    student = student or common.build_student(cfg)
    opt = torch.optim.AdamW(student.parameters(),
                            lr=cfg.train.lr_teacher_student,
                            weight_decay=cfg.train.weight_decay)
    rng = np.random.default_rng(cfg.train.seed + 2)
    Tz = cache["z"].shape[2]

    for step in range(cfg.phase2_steps):
        batch = batcher.episode_batch(cfg.train.batch_size)
        end = int(rng.integers(2, Tz + 1))       # random causal prefix
        win = batcher.window(batch, W=cfg.window, end=end)
        out = common.student_window_forward(student, win)
        b_s = out["belief_final"]
        with torch.no_grad():
            b_t = teacher.forward_batch(batch["rec"])
        l_distill = ((b_s - b_t) ** 2).mean()
        l_query = decoder.loss(b_s, batch["rec"])
        loss = l_distill + cfg.train.w_query_s * l_query
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),
                                       cfg.train.grad_clip)
        opt.step()

    metrics = evaluate(cfg, cache, teacher, decoder, student,
                       seed=cfg.train.seed + 3)
    return {"student": student, "metrics": metrics}


@torch.no_grad()
def evaluate(cfg, cache, teacher, decoder, student, seed=0, n=32):
    batcher = GroupBatcher(cache, seed=seed)
    batch = batcher.episode_batch(min(n, len(cache["z"])))
    win = batcher.window(batch, W=cfg.window)
    out = common.student_window_forward(student, win)
    b_t = teacher.forward_batch(batch["rec"])
    distill = float(((out["belief_final"] - b_t) ** 2).mean())
    query = float(decoder.loss(out["belief_final"], batch["rec"]))

    W = win["z"].shape[2]
    perm = torch.randperm(W, generator=torch.Generator().manual_seed(seed))
    shuffled = common.student_window_forward(student, dict(win,
                                             z=win["z"][:, :, perm]))
    swapped = common.student_window_forward(student, dict(win,
                                            actions=win["actions"].roll(
                                                1, dims=0)))
    return {
        "distill_mse": distill, "query_loss": query,
        "query_time_shuffled": float(decoder.loss(
            shuffled["belief_final"], batch["rec"])),
        "query_action_swapped": float(decoder.loss(
            swapped["belief_final"], batch["rec"])),
    }


if __name__ == "__main__":
    from ..config import tiny_config
    from . import phase1_oracle_wan
    cfg = tiny_config()
    print(run(cfg, phase1_oracle_wan.run(cfg))["metrics"])
