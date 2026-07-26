"""Mock Wan VAE and DiT with the tensor contracts of the real integration.

Adapted from adaptive_physics/wan/mock_wan.py so every phase script and unit
test runs on CPU without checkpoints; the real integration swaps these
classes and reads all widths from wan_config() at runtime.

Differences from the adaptive_physics mock, per the generalized plan:
  * attention uses NamedAttention with separate .q/.k/.v/.o nn.Linear
    children — the same attribute naming as the real Wan2.1 blocks — so
    dit_lora.apply_lora targets mock and real DiT identically;
  * physics conditioning enters through a PhysicsAdapterBank at quarter-
    depth blocks, consuming FIXED [B, K, d_phys=256] tokens (the adapter
    maps 256 -> d_model internally; the token width never tracks the model);
  * physics_ctx=None bypasses the adapters entirely (exact base model).

MockWanVAE: frozen random causal encoder — first latent from frame 0 alone,
then one latent per temporal_stride frames (Wan2.1's causal chunking);
exposes checkpoint_hash for cache provenance.

MockWanDiT: rectified-flow velocity model over future latent tokens with
prefix cross-attention, zero-gated action cross-attention, and the physics
adapters. x_tau = (1 - tau) * eps + tau * z1, target v* = z1 - eps.
"""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics_adapter import PhysicsAdapterBank


def sinusoidal(n, d, device=None):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(d // 2, device=device, dtype=torch.float32)[None]
    ang = pos / torch.pow(10000.0, 2 * i / d)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class MockWanVAE(nn.Module):
    """Frozen deterministic stand-in for the Wan video VAE encoder."""

    def __init__(self, latent_channels=8, temporal_stride=4, spatial_stride=8,
                 in_channels=1, seed=1234):
        super().__init__()
        self.latent_channels = latent_channels
        self.temporal_stride = temporal_stride
        self.spatial_stride = spatial_stride
        g = torch.Generator().manual_seed(seed)
        d_first = in_channels * spatial_stride ** 2
        d_block = in_channels * temporal_stride * spatial_stride ** 2
        w1 = torch.randn(d_first, latent_channels, generator=g)
        w2 = torch.randn(d_block, latent_channels, generator=g)
        self.register_buffer("w_first", w1 / w1.shape[0] ** 0.5)
        self.register_buffer("w_block", w2 / w2.shape[0] ** 0.5)
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def checkpoint_hash(self) -> str:
        h = hashlib.sha256()
        for b in (self.w_first, self.w_block):
            h.update(b.numpy().tobytes())
        return h.hexdigest()[:16]

    def _s2d(self, x):
        s = self.spatial_stride
        *lead, C, H, W = x.shape
        x = x.reshape(*lead, C, H // s, s, W // s, s)
        x = x.movedim(-3, -4).movedim(-1, -3)
        return x.reshape(*lead, C * s * s, H // s, W // s)

    @torch.no_grad()
    def encode(self, frames):
        """frames [B, T, C, H, W] -> latents [B, Cz, Tz, Hz, Wz];
        (T - 1) % temporal_stride == 0 (causal chunking)."""
        B, T, C, H, W = frames.shape
        st = self.temporal_stride
        assert (T - 1) % st == 0, f"frame count {T} incompatible with {st}"
        first = self._s2d(frames[:, 0])
        z0 = torch.einsum("bchw,cd->bdhw", first, self.w_first)
        blocks = frames[:, 1:].reshape(B, (T - 1) // st, st * C, H, W)
        zb = torch.einsum("btchw,cd->btdhw", self._s2d(blocks), self.w_block)
        z = torch.cat([z0[:, None], zb], dim=1)
        return z.movedim(1, 2).contiguous()


class NamedAttention(nn.Module):
    """Multi-head attention with separate q/k/v/o Linears (Wan-style naming,
    so LoRA can target .q and .v by attribute name)."""

    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        h = self.heads

        def split(t):
            return t.reshape(B, -1, h, D // h).transpose(1, 2)

        a = F.scaled_dot_product_attention(split(self.q(x)),
                                           split(self.k(ctx)),
                                           split(self.v(ctx)))
        return self.o(a.transpose(1, 2).reshape(B, N, D))


class BinAlignedActionEncoder(nn.Module):
    """Causal GRU over control-rate actions gathered at the last control
    step of each latent bin -> one token per bin with a temporal position."""

    def __init__(self, action_dim, d_model):
        super().__init__()
        self.gru = nn.GRU(action_dim, d_model, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, action_state, bin_index, n_bins):
        B = action_state.shape[0]
        h, _ = self.gru(action_state)
        out = h.new_zeros(B, n_bins, h.shape[-1])
        for b in range(n_bins):
            mask = bin_index <= b
            if mask.any():
                out[:, b] = h[:, int(mask.nonzero()[-1])]
        tokens = self.proj(out)
        return tokens + sinusoidal(n_bins, tokens.shape[-1],
                                   action_state.device)[None]


class _Block(nn.Module):
    """DiT block: self-attn + prefix cross-attn + gated action cross-attn
    + FFN. Physics adapters are applied OUTSIDE, by the DiT block loop."""

    def __init__(self, d, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = NamedAttention(d, heads)
        self.norm2 = nn.LayerNorm(d)
        self.prefix_attn = NamedAttention(d, heads)
        self.norm_a = nn.LayerNorm(d)
        self.action_attn = NamedAttention(d, heads)
        self.action_gate = nn.Parameter(torch.zeros(1))
        self.norm3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(),
                                 nn.Linear(4 * d, d))

    def forward(self, h, prefix_ctx, action_ctx):
        h = h + self.self_attn(self.norm1(h))
        h = h + self.prefix_attn(self.norm2(h), prefix_ctx)
        h = h + torch.tanh(self.action_gate) * \
            self.action_attn(self.norm_a(h), action_ctx)
        h = h + self.mlp(self.norm3(h))
        return h


class MockWanDiT(nn.Module):
    """Rectified-flow velocity model with quarter-depth physics adapters."""

    def __init__(self, latent_channels, d_model=96, heads=8, blocks=4,
                 action_dim=1, d_phys=256, k_tokens=8, n_adapters=4,
                 adapter_heads=4):
        super().__init__()
        self.d_model = d_model
        self.k_tokens = k_tokens
        self.d_phys = d_phys
        self.in_proj = nn.Linear(latent_channels, d_model)
        self.prefix_proj = nn.Linear(latent_channels, d_model)
        self.action_enc = BinAlignedActionEncoder(action_dim, d_model)
        self.tau_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(),
                                     nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList(_Block(d_model, heads)
                                    for _ in range(blocks))
        self.physics = PhysicsAdapterBank(
            {"d_model": d_model, "n_blocks": blocks},
            d_phys=d_phys, n_adapters=min(n_adapters, blocks),
            heads=adapter_heads)
        # small random head, emulating a PRETRAINED frozen base model: a
        # zero head would keep the frozen backbone's output (and therefore
        # every gradient into the adapters/LoRA) at exactly zero forever
        self.out = nn.Linear(d_model, latent_channels)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def wan_config(self) -> dict:
        """Runtime source of truth for widths — never hard-code these."""
        return {"d_model": self.d_model, "n_blocks": len(self.blocks),
                "physics_tokens": self.k_tokens, "d_phys": self.d_phys}

    def _tokens(self, z, proj, t_offset=0):
        B, C, T, H, W = z.shape
        x = proj(z.movedim(1, -1).reshape(B, T * H * W, C))
        d = self.d_model
        pt = sinusoidal(T + t_offset, d, z.device)[t_offset:]
        ph = sinusoidal(H, d, z.device)
        pw = sinusoidal(W, d, z.device)
        pos = pt[:, None, None] + ph[None, :, None] + pw[None, None, :]
        return x + pos.reshape(1, T * H * W, d)

    def forward(self, x_tau, tau, prefix_z, action_state, bin_index,
                physics_ctx):
        """
        x_tau        [B, Cz, Tf, Hz, Wz]  noised future latents
        tau          [B]                  flow time in [0, 1]
        prefix_z     [B, Cz, Tp, Hz, Wz]  clean prefix latents
        action_state [B, Tc, da]          control-rate script
        bin_index    [Tc]                 control step -> latent bin
        physics_ctx  [B, K, d_phys] or None (None = exact base model)
        """
        B, C, Tf, H, W = x_tau.shape
        Tp = prefix_z.shape[2]
        h = self._tokens(x_tau, self.in_proj, t_offset=Tp)
        h = h + self.tau_mlp(sinusoidal(1024, self.d_model, x_tau.device)[
            (tau * 1023).long()])[:, None]
        prefix_ctx = self._tokens(prefix_z, self.prefix_proj)
        action_ctx = self.action_enc(action_state, bin_index, Tp + Tf)
        for i, blk in enumerate(self.blocks):
            h = blk(h, prefix_ctx, action_ctx)
            if physics_ctx is not None:
                adapter = self.physics.adapter_for(i)
                if adapter is not None:
                    h = adapter(h, physics_ctx)
        v = self.out(h)
        return v.reshape(B, Tf, H, W, C).movedim(-1, 1)


def flow_sample(z_future, tau=None, eps=None, generator=None):
    """Shared-noise rectified flow sample -> (x_tau, tau, eps, v_star).

    Paired correct/wrong comparisons MUST reuse the returned (tau, eps) —
    see training/phase1_oracle_wan.paired_flow_losses.
    """
    B = z_future.shape[0]
    if eps is None:
        eps = torch.randn(z_future.shape, generator=generator,
                          device=z_future.device, dtype=z_future.dtype)
    if tau is None:
        tau = torch.rand(B, generator=generator, device=z_future.device)
    t = tau.view(B, *([1] * (z_future.dim() - 1)))
    x_tau = (1 - t) * eps + t * z_future
    v_star = z_future - eps
    return x_tau, tau, eps, v_star
