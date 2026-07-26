"""Zero-gate equivalence (plan unit test 9): with all adapter gates at
their exact-zero init AND zero-initialized LoRA installed, conditioning on
physics tokens reproduces the frozen base output bit-for-bit; physics_ctx=
None bypasses the adapters entirely."""

import torch

from ..training import common
from .util import randomize_out, run_tests, small_world


def _fm_inputs(cfg, batcher):
    batch = batcher.episode_batch(2)
    prefix, future = common.split_prefix_future(batch["z"],
                                                cfg.dit.prefix_bins)
    g = torch.Generator().manual_seed(0)
    x_tau = torch.randn(future.shape, generator=g)
    tau = torch.rand(future.shape[0], generator=g)
    return batch, prefix, x_tau, tau


def test_zero_gate_with_lora_matches_base():
    cfg, cache, _, _, batcher = small_world()
    dit = randomize_out(common.build_dit(cfg, with_lora=True)).eval()
    proj = common.build_projector(cfg).eval()
    batch, prefix, x_tau, tau = _fm_inputs(cfg, batcher)
    tokens = proj(torch.randn(2, cfg.teacher.belief_dim))
    with torch.no_grad():
        v_base = dit(x_tau, tau, prefix, batch["actions"],
                     batch["bin_index"], None)
        v_cond = dit(x_tau, tau, prefix, batch["actions"],
                     batch["bin_index"], tokens)
    torch.testing.assert_close(v_base, v_cond, atol=0.0, rtol=0.0)


def test_opened_gate_breaks_equivalence():
    """The equivalence must be BECAUSE of the gates: opening one gate with
    nonzero tokens must change the output (no dead physics path)."""
    cfg, cache, _, _, batcher = small_world()
    dit = randomize_out(common.build_dit(cfg, with_lora=True)).eval()
    proj = common.build_projector(cfg).eval()
    batch, prefix, x_tau, tau = _fm_inputs(cfg, batcher)
    tokens = proj(torch.randn(2, cfg.teacher.belief_dim))
    first = next(iter(dit.physics.adapters.values()))
    with torch.no_grad():
        first.gate.fill_(0.5)
        v_base = dit(x_tau, tau, prefix, batch["actions"],
                     batch["bin_index"], None)
        v_cond = dit(x_tau, tau, prefix, batch["actions"],
                     batch["bin_index"], tokens)
    assert (v_base - v_cond).abs().max() > 1e-7, "physics path is dead"


def test_quarter_depth_placement():
    from ..wan.physics_adapter import quarter_depth_indices
    assert quarter_depth_indices(30, 4) == [3, 11, 18, 26]
    assert quarter_depth_indices(4, 4) == [0, 1, 2, 3]
    idx = quarter_depth_indices(40, 8)
    assert len(set(idx)) == 8 and idx == sorted(idx)


def test_gradient_coverage():
    """Plan unit test 12: every intended trainable parameter gets a finite
    gradient — adapters (incl. gates), LoRA A/B, projector."""
    cfg, cache, _, _, batcher = small_world()
    dit = randomize_out(common.build_dit(cfg, with_lora=True)).train()
    proj = common.build_projector(cfg).train()
    # open the gates slightly so gradients reach past them
    for a in dit.physics.adapters.values():
        torch.nn.init.constant_(a.gate, 0.1)
    batch = batcher.episode_batch(2)
    tokens = proj(torch.randn(2, cfg.teacher.belief_dim))
    loss, tau, eps = common.fm_loss(dit, batch, tokens, cfg.dit.prefix_bins,
                                    generator=torch.Generator()
                                    .manual_seed(0))
    # a null-code term so the learned null belief participates too
    l_null, _, _ = common.fm_loss(dit, batch, proj.null_tokens(2),
                                  cfg.dit.prefix_bins, tau=tau, eps=eps)
    (loss + l_null).backward()
    for name, p in list(dit.named_parameters()) + \
            list(proj.named_parameters()):
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    frozen = [n for n, p in dit.named_parameters()
              if not p.requires_grad and p.grad is not None
              and p.grad.abs().max() > 0]
    assert not frozen, f"frozen params received gradients: {frozen[:3]}"


if __name__ == "__main__":
    run_tests([test_zero_gate_with_lora_matches_base,
               test_opened_gate_breaks_equivalence,
               test_quarter_depth_placement,
               test_gradient_coverage], __file__)
