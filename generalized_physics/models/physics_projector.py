"""Shared belief -> physics-token projector (plan: Module C, eq. 12).

C = reshape(W2 GELU(W1 b)) + E_slot   in   R^{K x d_phys},  K=8, d_phys=256.

The token interface is FIXED and Wan-independent: d_phys never tracks the
DiT width (the adapters map 256 -> d_model). One projector serves teacher
and student beliefs — that is the interface-drift guard — and it consumes
the deterministic belief only.

Initialization: ordinary small weights. The exact-zero initialization lives
on the ADAPTER residual gates, not here (zeroing both would starve the
projector of gradient signal once the gates open).

The learned null code lives here so "no physics" is a trained condition.
"""

import torch
import torch.nn as nn


class PhysicsProjector(nn.Module):
    def __init__(self, belief_dim=64, k_tokens=8, d_phys=256, hidden=256):
        super().__init__()
        self.belief_dim = belief_dim
        self.k_tokens = k_tokens
        self.d_phys = d_phys
        self.w1 = nn.Linear(belief_dim, hidden)
        self.w2 = nn.Linear(hidden, k_tokens * d_phys)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.zeros_(self.w2.bias)
        self.slot = nn.Parameter(torch.randn(k_tokens, d_phys) * 0.02)
        self.null_code = nn.Parameter(torch.zeros(belief_dim))

    def forward(self, belief):
        """belief [B, belief_dim] -> [B, K, d_phys]."""
        if belief.dim() != 2 or belief.shape[1] != self.belief_dim:
            raise ValueError(
                f"projector expects the deterministic belief [B, "
                f"{self.belief_dim}]; got {tuple(belief.shape)}. Uncertainty "
                "must not be passed as a side channel.")
        h = torch.nn.functional.gelu(self.w1(belief))
        out = self.w2(h).reshape(-1, self.k_tokens, self.d_phys)
        return out + self.slot[None]

    def null_tokens(self, batch_size):
        """Learned null conditioning through the same pathway."""
        return self.forward(self.null_code[None].expand(batch_size, -1))
