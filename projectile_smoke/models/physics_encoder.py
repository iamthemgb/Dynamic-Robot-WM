"""Fourier-feature physics encoder for the projectile smoke run.

Same architecture family as the cloth encoder (base tokens init to real T5
stats + tanh-gated zero-init MLP delta + aux recon head), with one addition:
the z-scored input vector goes through Fourier features (8 frequencies per
dim, 2^0..2^7) before the MLP - projectile geometry needs cm-level resolution
(highest frequency ~ 0.05 std units ~ 1.5 cm in position dims) that raw
scalars don't provide.

Interface-compatible with physics_finetune/models/wan_wrapper.PhysicsWan:
  forward(phys [B,D])          -> tokens [B,K,4096]
  aux_regress(proj [B,K,1536]) -> [B,D]   (recon decoder, anti-collapse aux)
  init_base_from_t5(embeddings)

Optional InfoNCE head (plan: include in config, OFF by default): pooled
token embedding -> projection -> normalized; two small-noise augmentations of
the same vector are positives, other episodes in the contrastive batch are
negatives. Safe here because physics is continuously sampled (no
preset-collapse risk), but recon aux suffices for the smoke goal.
"""

import math

import torch
import torch.nn as nn

T5_DIM = 4096
WAN_DIM = 1536


class FourierFeatures(nn.Module):

    def __init__(self, dim, n_freqs=8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freqs).float())
        self.out_dim = dim * (2 * n_freqs + 1)

    def forward(self, x):
        ang = x.unsqueeze(-1) * self.freqs          # [B, D, F]
        feats = torch.cat([x.unsqueeze(-1), ang.sin(), ang.cos()], dim=-1)
        return feats.flatten(1)                      # [B, D*(2F+1)]


class ProjectilePhysicsEncoder(nn.Module):

    def __init__(self, phys_dim, num_tokens=8, n_freqs=8, hidden=(256, 512),
                 gate_init=0.1, with_contrastive=False):
        super().__init__()
        self.phys_dim = phys_dim
        self.num_tokens = num_tokens
        self.fourier = FourierFeatures(phys_dim, n_freqs)

        self.base_tokens = nn.Parameter(torch.zeros(num_tokens, T5_DIM))
        self._base_initialized = False

        h1, h2 = hidden
        self.delta = nn.Sequential(
            nn.Linear(self.fourier.out_dim, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, num_tokens * T5_DIM),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.gate = nn.Parameter(torch.tensor(math.atanh(gate_init)))

        # recon decoder: mean-pooled post-text_embedding tokens -> phys vector
        self.regressor = nn.Sequential(
            nn.Linear(WAN_DIM, 256), nn.SiLU(), nn.Linear(256, phys_dim))

        self.contrastive_head = (
            nn.Sequential(nn.Linear(T5_DIM, 512), nn.SiLU(), nn.Linear(512, 128))
            if with_contrastive else None)

    @torch.no_grad()
    def init_base_from_t5(self, t5_embeddings):
        """Match base-token statistics to real caption T5 embeddings (the
        frozen model treats zero-pad slots as semantically ignorable)."""
        flat = torch.cat([e.reshape(-1).float() for e in t5_embeddings])
        self.base_tokens.normal_(flat.mean().item(), flat.std().item())
        self._base_initialized = True

    def token_delta(self, phys_vec):
        b = phys_vec.shape[0]
        return self.delta(self.fourier(phys_vec)).view(b, self.num_tokens, T5_DIM)

    def forward(self, phys_vec):
        """phys_vec: [B, phys_dim] z-scored -> tokens [B, K, 4096]."""
        assert self._base_initialized, "call init_base_from_t5 first"
        return (self.base_tokens.unsqueeze(0)
                + torch.tanh(self.gate) * self.token_delta(phys_vec))

    def aux_regress(self, projected_tokens):
        """projected_tokens: [B, K, 1536] (post text_embedding) -> [B, phys_dim]."""
        return self.regressor(projected_tokens.mean(dim=1))

    def infonce_loss(self, phys_batch, noise_std=0.02, temperature=0.1):
        """SimCLR-style: two noise-augmented views of each vector; other
        episodes in the batch are negatives. phys_batch: [N, phys_dim]."""
        assert self.contrastive_head is not None, "built with with_contrastive=False"
        z = []
        for _ in range(2):
            aug = phys_batch + noise_std * torch.randn_like(phys_batch)
            pooled = self.token_delta(aug).mean(dim=1)          # [N, 4096]
            z.append(nn.functional.normalize(self.contrastive_head(pooled), dim=-1))
        logits = z[0] @ z[1].T / temperature                     # [N, N]
        labels = torch.arange(len(phys_batch), device=phys_batch.device)
        return 0.5 * (nn.functional.cross_entropy(logits, labels)
                      + nn.functional.cross_entropy(logits.T, labels))
