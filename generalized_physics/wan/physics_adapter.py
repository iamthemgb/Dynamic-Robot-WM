"""Zero-gated physics cross-attention adapters (plan eq. 13).

H' = H + alpha * W_o Attn(W_q LN(H), W_k C_phys, W_v C_phys)

The attention runs in the FIXED d_phys=256 bottleneck: W_q maps the block
width (read from the loaded model, never hard-coded) down to d_phys, W_o
maps back up. alpha is a learned scalar initialized to exactly zero, so a
freshly inserted adapter is a no-op and the frozen Wan path starts
unperturbed — this is the only exact-zero init on the physics path (the
projector uses ordinary small init).

Placement: four adapters at approximately quarter-depth block indices
(quarter_depth_indices).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quarter_depth_indices(n_blocks, n_adapters=4):
    """Evenly spaced block indices, e.g. 30 blocks -> [3, 11, 18, 26]."""
    if n_adapters > n_blocks:
        raise ValueError(f"{n_adapters} adapters > {n_blocks} blocks")
    idx = [round((i + 0.5) * n_blocks / n_adapters - 0.5)
           for i in range(n_adapters)]
    if len(set(idx)) != n_adapters:
        idx = list(range(n_adapters))
    return idx


class PhysicsAdapter(nn.Module):
    def __init__(self, d_model, d_phys=256, heads=4, dropout=0.0):
        super().__init__()
        if d_phys % heads:
            raise ValueError(f"d_phys {d_phys} not divisible by heads {heads}")
        self.heads = heads
        self.d_phys = d_phys
        self.norm = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_phys)
        self.w_k = nn.Linear(d_phys, d_phys)
        self.w_v = nn.Linear(d_phys, d_phys)
        self.w_o = nn.Linear(d_phys, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))    # alpha, exact zero init

    def _split(self, x):
        B, N, D = x.shape
        return x.reshape(B, N, self.heads, D // self.heads).transpose(1, 2)

    def forward(self, h, physics_ctx):
        """h [B, N, d_model]; physics_ctx [B, K, d_phys]."""
        q = self._split(self.w_q(self.norm(h)))
        k = self._split(self.w_k(physics_ctx))
        v = self._split(self.w_v(physics_ctx))
        a = F.scaled_dot_product_attention(q, k, v)
        B, _, N, _ = a.shape
        a = a.transpose(1, 2).reshape(B, N, self.d_phys)
        return h + self.gate * self.dropout(self.w_o(a))


class PhysicsAdapterBank(nn.Module):
    """The four adapters plus their placement, as one module.

    Built against a wan_config() dict so the block width is always read from
    the loaded model. adapter_for(i) returns the adapter for block i or None;
    the host DiT (mock or wrapped real Wan) calls it after each block.
    """

    def __init__(self, wan_config, d_phys=256, n_adapters=4, heads=4,
                 dropout=0.0):
        super().__init__()
        if "d_model" not in wan_config or "n_blocks" not in wan_config:
            raise KeyError("wan_config must provide d_model and n_blocks — "
                           "read them from the loaded model")
        self.block_idx = quarter_depth_indices(int(wan_config["n_blocks"]),
                                               n_adapters)
        self.adapters = nn.ModuleDict({
            str(i): PhysicsAdapter(int(wan_config["d_model"]), d_phys,
                                   heads, dropout)
            for i in self.block_idx})

    def adapter_for(self, block_index):
        key = str(block_index)
        return self.adapters[key] if key in self.adapters else None

    def gate_params(self):
        return [a.gate for a in self.adapters.values()]

    def non_gate_params(self):
        return [p for a in self.adapters.values()
                for n, p in a.named_parameters() if n != "gate"]

    def gate_stats(self):
        g = torch.stack([a.gate.detach().abs().squeeze(0)
                         for a in self.adapters.values()])
        return float(g.mean()), float(g.max())
