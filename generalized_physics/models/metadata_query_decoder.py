"""Metadata query decoder (plan eq. 7-8).

v_hat_j = D_omega(b, E_k(k_j), E_s(s_j), E_u(u_j))

The variable-schema replacement for fixed-vector reconstruction: it is
evaluated only on records that exist in the episode (valid_mask), so missing
parameters need no padding in the semantic target. The same decoder serves
the teacher loss (Phase 1), the student query loss (Phases 2-3), and — as a
separately constructed probe — the Phase 0 representation-ceiling test.
"""

import torch
import torch.nn as nn


class MetadataQueryDecoder(nn.Module):
    def __init__(self, embedder, belief_dim=64, hidden=256):
        """embedder: the shared RecordEmbedder (same instance as the
        teacher's, so queries live in the record embedding space)."""
        super().__init__()
        self.embed = embedder
        self.net = nn.Sequential(
            nn.Linear(belief_dim + embedder.out_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, belief, key_ids, scope_ids, unit_ids):
        """belief [B, d_b]; id tensors [B, J] -> v_hat [B, J, 1]."""
        emb = self.embed(key_ids, scope_ids, unit_ids)           # [B, J, e]
        J = emb.shape[1]
        b = belief[:, None].expand(-1, J, -1)
        return self.net(torch.cat([b, emb], dim=-1))

    def loss(self, belief, batch):
        """Masked MSE over the records present in the episode (plan eq. 8)."""
        pred = self.forward(belief, batch["key_ids"], batch["scope_ids"],
                            batch["unit_ids"])
        m = batch["valid_mask"][..., None]
        se = (pred - batch["values"]) ** 2 * m
        return se.sum() / m.sum().clamp(min=1.0)
