"""Action-swap and impulse-time sensitivity (plan unit tests 5, 6)."""

import torch

from ..models.causal_student import PerBinControlEncoder
from ..training import common
from .util import run_tests, small_world


def test_action_swap_changes_belief():
    cfg, cache, _, _, batcher = small_world()
    student = common.build_student(cfg).eval()
    batch = batcher.episode_batch(4)
    win = batcher.window(batch, W=batch["z"].shape[2])
    with torch.no_grad():
        b = common.student_window_forward(student, win)["belief_final"]
        b_swap = common.student_window_forward(
            student, dict(win, actions=win["actions"].roll(1, dims=0)))[
            "belief_final"]
    assert (b - b_swap).abs().max() > 1e-6, "student ignores actions"


def test_impulse_shift_one_bin_changes_belief():
    cfg, cache, _, _, batcher = small_world()
    student = common.build_student(cfg).eval()
    batch = batcher.episode_batch(4)
    win = batcher.window(batch, W=batch["z"].shape[2])
    stride = cache["temporal_stride"]
    shifted = dict(win, actions=win["actions"].roll(stride, dims=1))
    with torch.no_grad():
        b = common.student_window_forward(student, win)["belief_final"]
        b_shift = common.student_window_forward(student, shifted)[
            "belief_final"]
    assert (b - b_shift).abs().max() > 1e-6, \
        "one-latent-bin impulse shift is invisible to the student"


def test_per_bin_encoder_keeps_impulse_timing():
    """Mean pooling would erase WITHIN-bin timing; the GRU must not."""
    torch.manual_seed(0)
    enc = PerBinControlEncoder(1, width=16).eval()
    B, S = 2, 4
    bin_index = torch.tensor([0, 0, 0, 0])
    early = torch.zeros(B, S, 1)
    early[:, 0] = 1.0
    late = torch.zeros(B, S, 1)
    late[:, -1] = 1.0
    with torch.no_grad():
        e_early = enc(early, bin_index, 1)
        e_late = enc(late, bin_index, 1)
    assert (e_early - e_late).abs().max() > 1e-6, \
        "control encoder destroyed within-bin impulse timing"


def test_empty_bin_yields_zeros():
    torch.manual_seed(0)
    enc = PerBinControlEncoder(1, width=8).eval()
    controls = torch.randn(2, 3, 1)
    bin_index = torch.tensor([1, 1, 2])          # bin 0 has no samples
    with torch.no_grad():
        e = enc(controls, bin_index, 3)
    assert e[:, 0].abs().max() == 0.0, "empty bin should encode to zeros"
    assert e[:, 1].abs().max() > 0


if __name__ == "__main__":
    run_tests([test_action_swap_changes_belief,
               test_impulse_shift_one_bin_changes_belief,
               test_per_bin_encoder_keeps_impulse_timing,
               test_empty_bin_yields_zeros], __file__)
