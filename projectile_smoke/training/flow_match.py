"""Rectified-flow training objective matching Wan's inference conventions.

Wan samples with FlowUniPCMultistepScheduler(shift applied via set_timesteps):
sigma(u) = shift*u / (1 + (shift-1)*u), t = sigma * num_train_timesteps,
x_t = (1-sigma)*x0 + sigma*noise, model predicts velocity v = noise - x0
(integrating dx/dsigma = v from sigma=1 to 0 recovers x0). The sign
convention is unit-tested against the scheduler in tests/test_flow_convention.py.
"""

import torch


def shift_sigma(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)


def sample_sigmas(batch_size, shift, device, generator=None):
    u = torch.rand(batch_size, device=device, generator=generator)
    return shift_sigma(u, shift)


def make_targets(x0, sigmas, noise=None, generator=None):
    """x0: [B,C,F,H,W] clean latents. Returns (x_t, v_target, t)."""
    if noise is None:
        noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype,
                            generator=generator)
    s = sigmas.view(-1, *([1] * (x0.dim() - 1))).to(x0.dtype)
    x_t = (1.0 - s) * x0 + s * noise
    v_target = noise - x0
    t = sigmas * 1000.0
    return x_t, v_target, t


def flow_loss(pred_list, v_target):
    """pred_list: list of [C,F,H,W] from WanModel; v_target: [B,C,F,H,W]."""
    pred = torch.stack(pred_list)
    return torch.nn.functional.mse_loss(pred.float(), v_target.float())
