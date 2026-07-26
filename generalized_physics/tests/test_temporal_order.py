"""Temporal-order sensitivity and causal masking (plan unit tests 4, 7)."""

import torch

from ..training import common
from .util import run_tests, small_world


def _setup():
    cfg, cache, _, _, batcher = small_world()
    student = common.build_student(cfg).eval()
    batch = batcher.episode_batch(4)
    win = batcher.window(batch, W=batch["z"].shape[2])
    return student, win


def test_time_shuffle_changes_belief():
    student, win = _setup()
    W = win["z"].shape[2]
    perm = torch.randperm(W, generator=torch.Generator().manual_seed(3))
    while torch.equal(perm, torch.arange(W)):
        perm = torch.randperm(W)
    with torch.no_grad():
        b = common.student_window_forward(student, win)["belief_final"]
        b_s = common.student_window_forward(
            student, dict(win, z=win["z"][:, :, perm]))["belief_final"]
    assert (b - b_s).abs().max() > 1e-6, "student ignores temporal order"


def test_causal_masking():
    """Per-bin belief at bin t must not change when later bins change."""
    student, win = _setup()
    W = win["z"].shape[2]
    t = W // 2
    corrupted = dict(win, z=win["z"].clone())
    corrupted["z"][:, :, t + 1:] += 10.0
    keep = win["bin_index"] <= t          # controls after bin t may differ too
    corrupted["actions"] = win["actions"].clone()
    corrupted["actions"][:, ~keep] += 5.0
    with torch.no_grad():
        b = common.student_window_forward(student, win)["belief"]
        b_c = common.student_window_forward(student, corrupted)["belief"]
    torch.testing.assert_close(b[:, :t + 1], b_c[:, :t + 1],
                               atol=1e-5, rtol=1e-5)
    assert (b[:, -1] - b_c[:, -1]).abs().max() > 1e-6, \
        "future corruption never reached later beliefs (dead inputs?)"


if __name__ == "__main__":
    run_tests([test_time_shuffle_changes_belief, test_causal_masking],
              __file__)
