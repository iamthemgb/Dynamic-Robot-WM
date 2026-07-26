"""Phase 4: adaptive inference by sliding-window recomputation (plan).

No separate updater: at each deployment step the newest VAE chunk and
executed control are appended and the student is simply re-run on the
latest W latent bins. This script fine-tunes the student on change-point
windows (current-regime teacher targets for windows before, straddling,
and after the switch) and then evaluates the recovery profile: belief
distance to the current-regime teacher belief as a function of the window
endpoint relative to the change bin.
"""

import numpy as np
import torch

from ..data.change_point_dataset import ChangePointWindows
from ..data.counterfactual_dataset import build_cache
from . import common


def run(cfg, phase1, student, change_frame=None):
    common.set_seed(cfg.train.seed + 6)
    teacher = phase1["teacher"]
    registry, normalizer = phase1["registry"], phase1["normalizer"]
    vae = common.build_vae(cfg)
    change_frame = change_frame or cfg.env.n_frames // 2
    cache, _, _ = build_cache(cfg, vae, seed=cfg.train.seed + 6,
                              n_groups=cfg.n_eval_groups,
                              change_frame=change_frame,
                              normalizer=normalizer, registry=registry)
    windows = ChangePointWindows(cache, cfg.window,
                                 seed=cfg.train.seed + 6)
    opt = torch.optim.AdamW(student.parameters(),
                            lr=cfg.train.lr_teacher_student,
                            weight_decay=cfg.train.weight_decay)
    rng = np.random.default_rng(cfg.train.seed + 6)
    ends = windows.sweep_ends()

    for step in range(cfg.phase4_windows):
        end = int(rng.choice(ends))              # before/straddle/after mix
        win = windows.batch(cfg.train.batch_size, end)
        out = common.student_window_forward(student, win)
        with torch.no_grad():
            b_t = teacher.forward_batch(win["rec"])
        loss = ((out["belief_final"] - b_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(),
                                       cfg.train.grad_clip)
        opt.step()

    # compare window sizes as inference settings (plan: W in {4, 8, 12});
    # with W ~ episode length no window lies fully after the change, so the
    # shorter window supplies the clean "after" recovery number
    metrics = {}
    for w in dict.fromkeys((cfg.window, 4)):
        wins = (windows if w == cfg.window else
                ChangePointWindows(cache, w, seed=cfg.train.seed + 6))
        metrics[f"W{w}"] = evaluate(cfg, wins, teacher, student)
    return {"student": student, "metrics": metrics, "change_cache": cache}


@torch.no_grad()
def evaluate(cfg, windows, teacher, student, n=32):
    """Deployment loop: recompute the belief at every window endpoint and
    track distance to the current-regime teacher belief."""
    profile = {}
    for end in windows.sweep_ends():
        win = windows.batch(n, end)
        b_s = common.student_window_forward(student, win)["belief_final"]
        b_t = teacher.forward_batch(win["rec"])
        profile[end] = {"phase": win["phase"],
                        "belief_mse": float(((b_s - b_t) ** 2).mean())}
    def _mean(phase):
        vals = [v["belief_mse"] for v in profile.values()
                if v["phase"] == phase]
        return float(np.mean(vals)) if vals else None

    # with W comparable to the episode length no window lies fully after
    # the change; the straddle profile is then the recovery signal
    return {"profile": profile,
            "mse_before": _mean("before"),
            "mse_straddle": _mean("straddle"),
            "mse_after": _mean("after"),
            "change_bin": windows.change_bin}


if __name__ == "__main__":
    from ..config import tiny_config
    from . import phase1_oracle_wan, phase2_student_distillation
    cfg = tiny_config()
    p1 = phase1_oracle_wan.run(cfg)
    p2 = phase2_student_distillation.run(cfg, p1)
    m = run(cfg, p1, p2["student"])["metrics"]
    print({w: {k: v for k, v in mw.items() if k != "profile"}
           for w, mw in m.items()})
