"""Shared-noise ranking (plan unit test 11): correct and wrong codes reuse
identical (tau, eps), so the paired loss difference isolates the code."""

import torch

from ..training import common
from ..training.phase1_oracle_wan import paired_flow_losses
from ..wan.mock_wan import flow_sample
from .util import run_tests, small_world


def test_flow_sample_reuses_noise():
    z = torch.randn(3, 4, 2, 2, 2)
    x1, tau, eps, v1 = flow_sample(z, generator=torch.Generator()
                                   .manual_seed(0))
    x2, tau2, eps2, v2 = flow_sample(z, tau=tau, eps=eps)
    torch.testing.assert_close(x1, x2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(tau, tau2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(v1, v2, atol=0.0, rtol=0.0)


def test_identical_codes_give_identical_losses():
    """Under shared (tau, eps), the same belief must give the same loss —
    if not, the pairing leaks noise into the comparison."""
    cfg, cache, registry, _, batcher = small_world()
    teacher = common.build_teacher(cfg, registry).eval()
    proj = common.build_projector(cfg).eval()
    dit = common.build_dit(cfg).eval()
    batch = batcher.paired_batch(2)
    b = teacher.forward_batch(batch["rec"])
    with torch.no_grad():
        l_c, l_w, _ = paired_flow_losses(dit, proj, batch, b, b,
                                         cfg.dit.prefix_bins,
                                         generator=torch.Generator()
                                         .manual_seed(0))
    torch.testing.assert_close(l_c, l_w, atol=0.0, rtol=0.0)


def test_wrong_code_is_same_group():
    cfg, cache, _, _, batcher = small_world()
    batch = batcher.paired_batch(8)
    gid = cache["group_id"]
    for row, k in enumerate(batch["idx"].tolist()):
        same_group = [i for i in range(len(gid))
                      if gid[i] == gid[k] and i != k]
        candidates = [cache["rec_batch"]["values"][i] for i in same_group]
        match = any(torch.equal(batch["rec_wrong"]["values"][row], c)
                    for c in candidates)
        assert match, "wrong code did not come from the same group"


if __name__ == "__main__":
    run_tests([test_flow_sample_reuses_noise,
               test_identical_codes_give_identical_losses,
               test_wrong_code_is_same_group], __file__)
