"""Verify the training-time flow-matching convention against Wan's inference
scheduler: feeding the exact velocity (noise - x0) at each step must recover x0.

Run: ~/.venvs/wan21/bin/python projectile_smoke/tests/test_flow_convention.py
"""

import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("WAN21_ROOT", "/gpfs/radev/scratch/sous/mzl7/Wan2.1"))
sys.path.insert(0, str(REPO))

from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from projectile_smoke.training.flow_match import make_targets, sample_sigmas  # noqa: E402


def test_scheduler_recovers_x0():
    torch.manual_seed(0)
    x0 = torch.randn(1, 4, 8, 8)

    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sched.set_timesteps(50, device="cpu", shift=5.0)

    sample = torch.randn(1, 4, 8, 8)  # pure noise at sigma_max
    for t in sched.timesteps:
        sigma = sched.sigmas[sched.step_index if sched.step_index is not None
                             else 0]
        # exact velocity under our convention: v = (x_t - x0) / sigma
        v = (sample - x0) / max(sigma.item(), 1e-8)
        sample = sched.step(v, t, sample, return_dict=False)[0]

    err = (sample - x0).abs().max().item()
    assert err < 1e-3, f"scheduler did not recover x0 (max err {err})"
    print(f"OK: scheduler recovers x0, max err {err:.2e}")


def test_make_targets_consistency():
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, 3, 8, 8)
    sigmas = sample_sigmas(2, shift=5.0, device="cpu")
    x_t, v, t = make_targets(x0, sigmas)
    s = sigmas.view(-1, 1, 1, 1, 1)
    # x0 must be recoverable via the scheduler's inversion: x0 = x_t - sigma*v
    err = (x_t - s * v - x0).abs().max().item()
    assert err < 1e-5, err
    assert torch.all((t >= 0) & (t <= 1000))
    print(f"OK: make_targets consistent, max err {err:.2e}")


if __name__ == "__main__":
    test_make_targets_consistency()
    test_scheduler_recovers_x0()
