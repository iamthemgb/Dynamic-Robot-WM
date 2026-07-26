"""LoRA zero-init and switch (plan unit test 10).

With B = 0 the LoRA-equipped DiT matches the base DiT exactly; after
training moves B, the off-switch restores base behavior exactly; the LoRA
parameter group contains only A/B and the wrapped base Linears stay
frozen."""

import copy

import torch

from ..training import common
from ..wan.dit_lora import (LoRALinear, apply_lora, lora_modules,
                            lora_params, set_lora_enabled)
from .util import randomize_out, run_tests, small_world


def _fixed_inputs(cfg, batcher):
    """One frozen input tuple, so every model sees identical data."""
    batch = batcher.episode_batch(2)
    prefix, future = common.split_prefix_future(batch["z"],
                                                cfg.dit.prefix_bins)
    g = torch.Generator().manual_seed(1)
    x_tau = torch.randn(future.shape, generator=g)
    tau = torch.rand(future.shape[0], generator=g)
    return (x_tau, tau, prefix, batch["actions"], batch["bin_index"], None)


def _outputs(dit, inputs):
    with torch.no_grad():
        return dit(*inputs)


def test_zero_init_is_exact_identity():
    cfg, cache, _, _, batcher = small_world()
    inputs = _fixed_inputs(cfg, batcher)
    base = randomize_out(common.build_dit(cfg, with_lora=False)).eval()
    lora = copy.deepcopy(base)
    wrapped = apply_lora(lora.blocks, targets=cfg.lora.targets,
                         rank=cfg.lora.rank, alpha=cfg.lora.alpha,
                         dropout=0.0)
    assert wrapped, "no projections wrapped"
    assert all(n.endswith((".q", ".v")) for n in wrapped)
    torch.testing.assert_close(_outputs(base, inputs),
                               _outputs(lora, inputs),
                               atol=0.0, rtol=0.0)


def test_switch_restores_base_after_training():
    cfg, cache, _, _, batcher = small_world()
    inputs = _fixed_inputs(cfg, batcher)
    base = randomize_out(common.build_dit(cfg, with_lora=False)).eval()
    lora = copy.deepcopy(base)
    apply_lora(lora.blocks, targets=cfg.lora.targets, rank=cfg.lora.rank,
               alpha=cfg.lora.alpha, dropout=0.0)
    with torch.no_grad():                       # simulate a trained state
        for m in lora_modules(lora.blocks):
            m.lora_b.normal_(std=0.05)
    v_base = _outputs(base, inputs)
    assert (v_base - _outputs(lora, inputs)).abs().max() > 1e-7, \
        "perturbed LoRA changed nothing (dead path)"
    set_lora_enabled(lora.blocks, False)        # adapter-only mode
    torch.testing.assert_close(v_base, _outputs(lora, inputs),
                               atol=0.0, rtol=0.0)
    set_lora_enabled(lora.blocks, True)
    assert (v_base - _outputs(lora, inputs)).abs().max() > 1e-7


def test_param_group_and_freezing():
    cfg, cache, _, _, batcher = small_world()
    dit = common.build_dit(cfg, with_lora=True)
    params = lora_params(dit)
    mods = lora_modules(dit)
    assert len(params) == 2 * len(mods)
    for m in mods:
        assert not m.base.weight.requires_grad, "base weight not frozen"
        assert m.lora_a.requires_grad and m.lora_b.requires_grad
        assert m.lora_b.abs().max() == 0.0, "B must start at zero"


def test_gradient_reaches_lora_b():
    """At zero init dL/dA = 0 (B gates it), but dL/dB must be nonzero."""
    cfg, cache, _, _, batcher = small_world()
    dit = randomize_out(common.build_dit(cfg, with_lora=True)).train()
    proj = common.build_projector(cfg)
    batch = batcher.episode_batch(2)
    loss, _, _ = common.fm_loss(dit, batch, proj.null_tokens(2),
                                cfg.dit.prefix_bins,
                                generator=torch.Generator().manual_seed(0))
    loss.backward()
    got = any(m.lora_b.grad is not None and m.lora_b.grad.abs().max() > 0
              for m in lora_modules(dit))
    assert got, "no gradient reached any LoRA B matrix"


def test_wraps_only_linear():
    try:
        LoRALinear(torch.nn.Conv1d(3, 3, 1))
    except TypeError:
        return
    raise AssertionError("LoRALinear accepted a non-Linear module")


if __name__ == "__main__":
    run_tests([test_zero_init_is_exact_identity,
               test_switch_restores_base_after_training,
               test_param_group_and_freezing,
               test_gradient_reaches_lora_b,
               test_wraps_only_linear], __file__)
