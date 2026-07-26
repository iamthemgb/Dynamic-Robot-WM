"""Rank-16 LoRA on the frozen DiT attention projections (plan eq. 14).

W x -> W x + (alpha/r) * B A x,  A [r, d_in] small random, B [d_out, r] ZERO.

Zero-init B means the LoRA path contributes exactly nothing at the start of
training, so zero-gate equivalence holds with LoRA installed. LoRA is
backbone capacity, NOT a physics input — the belief reaches Wan only through
the physics tokens and adapters.

Attribution contract: all LoRA matrices live in one parameter group behind a
single on/off switch (set_lora_enabled), so every experiment can be re-run
in adapter-only mode. The base Linear stays frozen and bit-identical, and
disabling the switch restores base behavior exactly even after training.

Targeting: attribute-name suffixes of attention projections ("q", "v" by
default). Both the real Wan DiT (WanSelfAttention/WanCrossAttention with
.q/.k/.v/.o Linears) and mock_wan.NamedAttention use this naming.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=16, alpha=16.0, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base)}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.enabled = True
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            delta = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
            out = out + self.scale * delta
        return out


def apply_lora(module, targets=("q", "v"), rank=16, alpha=16.0, dropout=0.05):
    """Wrap every nn.Linear whose attribute name is in `targets` (anywhere
    under `module`) with LoRALinear. Returns the qualified names wrapped."""
    wrapped = []
    for name, child in list(module.named_modules()):
        for attr, sub in list(child.named_children()):
            if attr in targets and isinstance(sub, nn.Linear):
                setattr(child, attr, LoRALinear(sub, rank, alpha, dropout))
                wrapped.append(f"{name}.{attr}" if name else attr)
    if not wrapped:
        raise ValueError(f"no nn.Linear children named {targets} found")
    return wrapped


def lora_modules(module):
    return [m for m in module.modules() if isinstance(m, LoRALinear)]


def lora_params(module):
    """The single LoRA parameter group (A and B matrices only)."""
    return [p for m in lora_modules(module)
            for p in (m.lora_a, m.lora_b)]


def set_lora_enabled(module, enabled: bool):
    """The attribution switch: adapter-only mode when False."""
    for m in lora_modules(module):
        m.enabled = enabled


def lora_state_dict(module):
    return {n: p.detach().clone()
            for n, p in module.named_parameters()
            if "lora_a" in n or "lora_b" in n}
