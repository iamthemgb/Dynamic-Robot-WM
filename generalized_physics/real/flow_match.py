"""Rectified-flow utilities in Wan's convention.

``physics_finetune.training.flow_match`` (imported by projectile/train_real.py)
does not exist on this filesystem. These four functions reconstruct it from the
convention that ``train_real.paired_eval`` -- which *is* present -- uses
verbatim:

    x_t      = (1 - sigma) * x0 + sigma * eps
    v_target = eps - x0
    t        = sigma * 1000

  !! SIGN TRAP !!
``wan/mock_wan.py::flow_sample`` uses the OPPOSITE convention:
``x_tau = (1-tau)*eps + tau*z1`` with ``v* = z1 - eps``, i.e. tau=1 means DATA.
Wan uses sigma=1 means NOISE. ``training.common.fm_loss`` must therefore be
*replaced* wholesale, never adapted. ``tests/test_flow_convention.py`` gates
this: with the frozen base model, predicting ``eps - x0`` must score far better
than ``x0 - eps``.
"""

import torch


def shift_sigma(u, shift: float):
    """Map uniform u in (0,1) to a shifted sigma. shift=1 is the identity."""
    if not torch.is_tensor(u):
        u = torch.tensor(float(u))
    return shift * u / (1.0 + (shift - 1.0) * u)


def _randn(shape, ref, generator):
    """randn matching ``ref``'s device/dtype, tolerating a CPU generator.

    ``phase1_oracle_wan.evaluate`` seeds a plain ``torch.Generator()`` (CPU)
    and threads it through for reproducible paired evaluation. torch refuses a
    CPU generator on a CUDA allocation, so draw on the generator's own device
    and move. Sampling on CPU also keeps the eval reproducible across GPUs.
    """
    if generator is not None and generator.device.type != ref.device.type:
        out = torch.randn(shape, generator=generator, dtype=torch.float32,
                          device=generator.device)
        return out.to(device=ref.device, dtype=ref.dtype)
    return torch.randn(shape, generator=generator, device=ref.device,
                       dtype=ref.dtype)


def sample_sigmas(batch: int, shift: float, device=None, generator=None):
    if generator is not None and generator.device.type != torch.device(
            device or "cpu").type:
        u = torch.rand(batch, generator=generator, device=generator.device)
        u = u.to(device)
    else:
        u = torch.rand(batch, device=device, generator=generator)
    return shift_sigma(u, shift)


def make_targets(x0, sigmas, eps=None, generator=None):
    """-> (x_t, v_target, t, eps).

    Passing an explicit ``eps`` (and reusing ``sigmas``) is what makes the
    paired correct/wrong ranking comparison valid: both conditions must see
    identical noise, otherwise the loss difference is dominated by sampling
    variance rather than by the conditioning.
    """
    if eps is None:
        eps = _randn(x0.shape, x0, generator)
    eps = eps.to(device=x0.device, dtype=x0.dtype)
    view = (-1,) + (1,) * (x0.dim() - 1)
    s = sigmas.to(device=x0.device, dtype=x0.dtype).view(view)
    x_t = (1.0 - s) * x0 + s * eps
    v_target = eps - x0
    t = sigmas.to(x0.device).float() * 1000.0
    return x_t, v_target, t, eps


def flow_loss(pred, target, weight=None):
    """MSE against the flow target. Accepts WanModel's list-of-tensor output.

    ``weight`` (optional) is a per-cell weight broadcastable against the
    squared error, e.g. ``[B, 1, Tz, Hz, Wz]`` for a spatial ROI weighting.
    Callers are expected to normalize it (mean 1) so the loss scale is
    unchanged; ``None`` is the exact unweighted path.
    """
    if isinstance(pred, (list, tuple)):
        pred = torch.stack([p.float() for p in pred])
    err = (pred.float() - target.float()).pow(2)
    if weight is None:
        return err.mean()
    return (err * weight.float()).mean()
