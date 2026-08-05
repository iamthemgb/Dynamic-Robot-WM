"""V6 -- one real fwd+bwd per arm through GeneralizedAdaptiveWan.

Checks, per arm:
  * the checkpoint loads and in_dim matches the cache's latent channels
  * adapters land at the expected quarter-depth block indices
  * the zero-init contract holds on the REAL model: with physics_ctx=None and
    LoRA disabled the wrapped model reproduces the frozen base bit-exactly
    (the mock-level version of this is tests/test_zero_gate_equivalence.py)
  * a full forward+backward completes and gradient reaches both the adapters
    and the LoRA B matrices
  * reports peak VRAM and step latency, which is what the sbatch sizing rests on

    python -m generalized_physics.real.tests.test_real_wan_step wan21_t2v_1p3b
"""

import sys
import time

import torch

from ...config import smoke_config
from ...wan.physics_adapter import quarter_depth_indices
from .. import wan_loader as W
from ..backend import RealWanDiT, make_real_fm_loss
from ..paths import ARMS

LATENT_HW = {"wan21": (15, 60, 104), "wan22": (15, 30, 52)}


def main(arm_name="wan21_t2v_1p3b"):
    arm = ARMS[arm_name]
    dev = "cuda"
    torch.manual_seed(0)

    cfg = smoke_config()
    cfg.vae.latent_channels = arm.latent_channels
    cfg.dit.prefix_bins = 2

    t0 = time.time()
    dit = RealWanDiT(cfg, arm, device=dev)
    load_s = time.time() - t0
    wc = dit.wan_config()
    expect = quarter_depth_indices(wc["n_blocks"], cfg.adapters.n_adapters)
    placed = list(dit.physics.block_idx)
    n_base = sum(p.numel() for p in dit.model.dit.parameters())
    n_train = sum(p.numel() for p in dit.model.trainable_params())
    print(f"  loaded {load_s:5.1f}s | d_model {wc['d_model']} blocks "
          f"{wc['n_blocks']} | base {n_base/1e9:.2f}B frozen, "
          f"{n_train/1e6:.1f}M trainable")
    print(f"  adapters at {placed}  expected {expect}  "
          f"{'OK' if placed == expect else 'MISMATCH'}")

    Tz, Hz, Wz = LATENT_HW[arm.vae_kind]
    x0 = torch.randn(1, arm.latent_channels, Tz, Hz, Wz, device=dev)
    seq_len = W.seq_len_of(x0.shape)
    ctx = [torch.zeros(1, 4096, device=dev, dtype=torch.bfloat16)]
    t = torch.full((1,), 500.0, device=dev)

    # --- zero-init contract on the real model -----------------------------
    dit.model.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        dit.set_lora_enabled(False)
        a = dit.model(x=list(x0), t=t, context=ctx, seq_len=seq_len,
                      physics_ctx=None)[0].float()
        dit.set_lora_enabled(True)
        b = dit.model(x=list(x0), t=t, context=ctx, seq_len=seq_len,
                      physics_ctx=None)[0].float()
    same = torch.equal(a, b)
    print(f"  zero-init identity (LoRA off vs on, no physics): "
          f"max|d|={float((a-b).abs().max()):.3e}  {'OK' if same else 'BROKEN'}")

    # --- real training step ------------------------------------------------
    dit.model.train()
    fm = make_real_fm_loss(dit, lambda idx: ctx)
    tokens = torch.randn(1, cfg.projector.k_tokens, cfg.projector.d_phys,
                         device=dev, requires_grad=True)
    batch = {"z": x0, "idx": torch.zeros(1, dtype=torch.long)}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss, tau, eps = fm(dit, batch, tokens, cfg.dit.prefix_bins)
    loss.backward()
    torch.cuda.synchronize()
    step_s = time.time() - t0

    g_ad = sum(1 for p in dit.physics.parameters()
               if p.grad is not None and p.grad.abs().sum() > 0)
    lora_b = dit.model.lora_parameters()[1::2]
    g_lo = sum(1 for p in lora_b if p.grad is not None and p.grad.abs().sum() > 0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  loss {float(loss):.4f} | seq_len {seq_len} | fwd+bwd {step_s:.2f}s"
          f" | peak {peak:.1f} GB")
    print(f"  grad reaches adapters {g_ad}/{len(list(dit.physics.parameters()))}"
          f", LoRA-B {g_lo}/{len(lora_b)}, tokens "
          f"{'yes' if tokens.grad is not None else 'NO'}")

    ok = (placed == expect and same and g_ad > 0 and g_lo > 0
          and torch.isfinite(loss))
    print(f"  VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
