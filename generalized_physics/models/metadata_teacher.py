"""Variable-length metadata teacher (plan: Module A).

r_j = f_rec([E_k(k_j); E_s(s_j); E_u(u_j); v_tilde_j])          (record MLP)
q   = [sum_j phi(r_j) / sqrt(n); log(1 + n)]                    (DeepSets)
b^T = rho(q)  in  R^64

Permutation invariant and linear in record count; masked slots contribute
exactly zero to the sum and to n. No per-sample LayerNorm on the belief
(it couples code dimensions); a small output scale and optional tanh bound
it instead.
"""

import torch
import torch.nn as nn

from .metadata_records import RecordEmbedder


def _mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Linear(d_hidden, d_out))


class MetadataTeacher(nn.Module):
    def __init__(self, registry, record_width=256, key_embed=64,
                 scope_embed=32, unit_embed=16, belief_dim=64,
                 output_scale=1.0, tanh_output=False, embedder=None):
        super().__init__()
        self.belief_dim = belief_dim
        self.output_scale = output_scale
        self.tanh_output = tanh_output
        self.embed = embedder or RecordEmbedder(
            registry, key_embed, scope_embed, unit_embed)
        d_r = record_width
        self.f_rec = _mlp(self.embed.out_dim + 1, d_r, d_r)
        self.phi = _mlp(d_r, d_r, d_r)
        self.rho = _mlp(d_r + 1, d_r, belief_dim)
        # keep the initial belief small; the projector/adapters see values
        # near zero at the start of training regardless of schema
        last = self.rho[-1]
        nn.init.normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)

    def forward(self, key_ids, scope_ids, unit_ids, values, valid_mask):
        """key/scope/unit_ids [B, J] long; values [B, J, 1]; valid_mask [B, J]
        -> belief [B, belief_dim]."""
        emb = self.embed(key_ids, scope_ids, unit_ids)          # [B, J, e]
        r = self.f_rec(torch.cat([emb, values], dim=-1))        # [B, J, d_r]
        m = valid_mask[..., None]
        pooled = (self.phi(r) * m).sum(dim=1)                   # [B, d_r]
        n = valid_mask.sum(dim=1, keepdim=True)                 # [B, 1]
        pooled = pooled / n.clamp(min=1.0).sqrt()
        q = torch.cat([pooled, torch.log1p(n)], dim=-1)
        b = self.rho(q) * self.output_scale
        return torch.tanh(b) if self.tanh_output else b

    def forward_batch(self, batch):
        """Convenience for collate_records() dicts."""
        return self.forward(batch["key_ids"], batch["scope_ids"],
                            batch["unit_ids"], batch["values"],
                            batch["valid_mask"])
