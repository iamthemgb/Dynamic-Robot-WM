"""P1 pre-flight: encoder shapes + gradient flow through the frozen DiT's
text_embedding into encoder params + LoRA wiring. CPU-safe (loads the real
WanModel in bf16, ~3 GB RAM; forward uses a tiny fake latent).

Run: ~/.venvs/wan21/bin/python projectile_smoke/tests/test_encoder.py
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from projectile_smoke.models.lora import (CROSS_ATTN_TARGETS, LoRALinear,  # noqa: E402
                                          apply_lora)
from projectile_smoke.models.wan_wrapper import PhysicsWan  # noqa: E402

from projectile_smoke.models.physics_encoder import ProjectilePhysicsEncoder  # noqa: E402

PHYS_DIM = 14
MODEL_DIR = "/gpfs/radev/scratch/sous/mzl7/wan_models/Wan2.1-T2V-1.3B"


def test_encoder_standalone():
    torch.manual_seed(0)
    enc = ProjectilePhysicsEncoder(phys_dim=PHYS_DIM, num_tokens=8, n_freqs=8)
    enc.init_base_from_t5([torch.randn(30, 4096)])
    x = torch.randn(3, PHYS_DIM)

    tokens = enc(x)
    assert tokens.shape == (3, 8, 4096), tokens.shape
    # zero-init delta + gate: tokens identical to base and physics-independent
    assert torch.equal(tokens[0], tokens[1])
    assert torch.allclose(tokens[0], enc.base_tokens)
    assert abs(float(torch.tanh(enc.gate)) - 0.1) < 1e-6

    # fourier features deterministic + right size: D*(2F+1)
    ff = enc.fourier(x)
    assert ff.shape == (3, PHYS_DIM * 17)
    assert torch.equal(ff, enc.fourier(x))

    aux = enc.aux_regress(torch.randn(3, 8, 1536))
    assert aux.shape == (3, PHYS_DIM)
    print("OK: encoder shapes, zero-init gating, fourier determinism")


def test_infonce():
    torch.manual_seed(0)
    enc = ProjectilePhysicsEncoder(phys_dim=PHYS_DIM, with_contrastive=True)
    enc.init_base_from_t5([torch.randn(30, 4096)])
    loss = enc.infonce_loss(torch.randn(16, PHYS_DIM))
    assert loss.isfinite()
    print(f"OK: infonce loss computes ({loss.item():.3f})")


def test_grad_through_frozen_text_embedding_and_lora():
    torch.manual_seed(0)
    on_gpu = torch.cuda.is_available()
    enc = ProjectilePhysicsEncoder(phys_dim=PHYS_DIM, num_tokens=8)
    enc.init_base_from_t5([torch.randn(30, 4096)])
    # wan's attention fallback hard-casts to bf16 internally; only cuda
    # autocast bridges that, so the full forward runs fp32-safe paths on CPU
    # and the complete DiT forward+backward on GPU (slurm/tests.sbatch).
    dtype = torch.bfloat16 if on_gpu else torch.float32
    model = PhysicsWan(enc, model_dir=MODEL_DIR, torch_dtype=dtype)
    model.freeze_dit()

    lora_params = apply_lora(model.dit, targets=CROSS_ATTN_TARGETS,
                             rank=8, alpha=8)
    n_lora = sum(p.numel() for p in lora_params)
    n_layers = sum(isinstance(m, LoRALinear) for m in model.dit.modules())
    assert n_layers == 4 * len(model.dit.blocks), n_layers
    # 30 blocks x 4 linears x (8x1536 + 1536x8) = 2.95M. (The plan's "~6M"
    # estimate double-counted; the architecture matches the plan exactly.)
    assert n_lora == len(model.dit.blocks) * 4 * 2 * 8 * model.dit.dim, n_lora

    device = torch.device("cuda" if on_gpu else "cpu")
    model.to(device)

    # encoder -> inject -> frozen text_embedding -> aux head, with backward:
    # THE anti-collapse path; must push grads into the encoder on any device
    context = [torch.randn(20, 4096, device=device)]
    phys = torch.randn(1, PHYS_DIM, device=device)
    ctx_out, tokens = model.inject(context, phys)
    assert ctx_out[0].shape == (28, 4096) and tokens.shape == (1, 8, 4096)
    projected = model.dit.text_embedding(tokens.to(model.base_dtype))
    assert projected.shape == (1, 8, 1536), projected.shape
    aux = enc.aux_regress(projected.float())
    aux.pow(2).mean().backward()
    frozen = list(model.dit.text_embedding.parameters())
    assert all(not p.requires_grad and p.grad is None for p in frozen)
    grads = {n: p.grad for n, p in enc.named_parameters()}
    for name in ["base_tokens", "delta.4.weight", "regressor.2.weight"]:
        g = grads[name]
        assert g is not None and g.abs().sum() > 0, f"no grad into {name}"
    # at init delta outputs exactly zero (zero-init final linear), so the
    # gate's grad is exactly zero - it only starts moving once delta does
    assert grads["gate"] is not None and grads["gate"].abs().sum() == 0
    with torch.no_grad():  # nudge delta off zero: gate must then get grad
        enc.delta[-1].weight.normal_(0, 0.01)
    enc.zero_grad()
    aux2 = enc.aux_regress(model.dit.text_embedding(
        model.inject(context, phys)[1].to(model.base_dtype)).float())
    aux2.pow(2).mean().backward()
    assert enc.gate.grad is not None and enc.gate.grad.abs().sum() > 0
    with torch.no_grad():
        enc.delta[-1].weight.zero_()
    print(f"OK: grads flow through frozen text_embedding into encoder; "
          f"LoRA {n_lora / 1e6:.2f}M params on {n_layers} cross-attn linears")

    if not on_gpu:
        print("SKIPPED (no GPU): full DiT forward+backward incl. LoRA grads "
              "- covered by slurm/tests.sbatch")
        return

    enc.zero_grad()
    # tiny fake latent: [16, 1, 8, 8] -> 16 tokens
    x = [torch.randn(16, 1, 8, 8, device=device)]
    t = torch.tensor([500.0], device=device)
    with torch.autocast("cuda", torch.bfloat16):
        pred, aux = model(x=x, t=t, context=context, seq_len=16, phys_vec=phys)
    assert pred[0].shape == (16, 1, 8, 8), pred[0].shape
    assert aux.shape == (1, PHYS_DIM)
    loss = pred[0].float().pow(2).mean() + aux.float().pow(2).mean()
    loss.backward()
    # zero-init lora_b: on the FIRST backward lora_b gets nonzero grad while
    # lora_a's grad (flowing through the zero lora_b) is exactly zero
    bs = [p for p in lora_params if p.shape[0] == model.dit.dim]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bs), \
        "no grad into any lora_b - LoRA not in the graph"
    for name in ["base_tokens", "delta.4.weight"]:
        g = dict(enc.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, f"no grad into {name}"
    print("OK: full DiT forward+backward - LoRA and encoder both receive grads")


if __name__ == "__main__":
    test_encoder_standalone()
    test_infonce()
    test_grad_through_frozen_text_embedding_and_lora()
    print("ALL ENCODER TESTS PASSED")
