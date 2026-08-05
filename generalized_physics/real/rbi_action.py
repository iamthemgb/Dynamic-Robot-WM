"""Stage-2 action pathway: token encoder, block shims, action-aware fm loss.

Everything rbi-local — ``backend.py`` / ``adaptive_wan_integration.py`` /
``physics_adapter.py`` are NOT edited, so the stage-1 path (and the roll /
pbc campaigns) stay bit-identical. The action bank exists only inside the
stage-2 process:

  * ``ActionTokenEncoder`` — verbatim port of the zl664 encoder
    (``phase1_state/action_encoder.py``), frame-aligned [B, 73, 8] controls
    -> [B, 8, 256] tokens. The full open-loop plan is known before rollout,
    so it need not be causal.
  * ``install_action_bank`` — post-hoc re-wraps the four quarter-depth
    ``_AdapterBlock``s of an already-built ``RealWanDiT`` with
    ``_ActionShim``: frozen block + frozen physics adapter run unchanged
    (still gradient-checkpointed inside the inner block), then the new
    zero-gated action adapter is applied — the same physics-then-action
    ordering and placement as the zl664 two-bank model
    (``phase1_state/wan/adaptive.py``). Reusing ``PhysicsAdapterBank`` for
    the action bank gives zero-init gates and the
    gate_params/non_gate_params/gate_stats surface for free.
  * ``make_action_fm_loss`` — ``backend.make_real_fm_loss`` minus the ROI
    weighting (the arm IS the action signal; the ball ROI is the wrong
    footprint), plus an ``action_seq`` kwarg. ``action_seq=None`` leaves the
    shim a no-op — that IS the null-action condition (pathway bypassed),
    used both for the p_null_action training draws and the eval null.

Install order matters: call ``load_adaptive_state_dict`` (and freeze LoRA)
BEFORE ``install_action_bank`` — both walk ``dit.model.dit.blocks`` by
parameter name, and the shim nests the original block under ``.inner``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..wan.adaptive_wan_integration import _AdapterBlock
from ..wan.physics_adapter import PhysicsAdapterBank
from . import wan_loader as W
from .flow_match import flow_loss, make_targets, sample_sigmas

ACTION_DIM = 8
ACTION_TOKENS = 8
ACTION_WIDTH = 256                # == d_phys: the adapters share the class
ACTION_MAX_FRAMES = 128


class ActionTokenEncoder(nn.Module):
    """Encode frame-aligned open-loop controls into fixed conditioning tokens.

    Port of zl664 ``phase1_state/action_encoder.py:ActionTokenEncoder`` with
    the rbi geometry as defaults. No access to outcomes or measured motion —
    input is the planned control echo, [B, T, action_dim].
    """

    def __init__(self, action_dim=ACTION_DIM, width=ACTION_WIDTH,
                 n_tokens=ACTION_TOKENS, max_frames=ACTION_MAX_FRAMES):
        super().__init__()
        self.width = width
        self.n_tokens = n_tokens
        self.input = nn.Sequential(
            nn.Linear(action_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.position = nn.Parameter(torch.randn(max_frames, width) * 0.02)
        self.temporal = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.slot = nn.Parameter(torch.randn(n_tokens, width) * 0.02)
        self.norm = nn.LayerNorm(width)

    def forward(self, actions):
        if actions.ndim != 3 or actions.shape[-1] != self.input[0].in_features:
            raise ValueError(f"expected [B,T,{self.input[0].in_features}] "
                             f"actions, got {tuple(actions.shape)}")
        if actions.shape[1] > self.position.shape[0]:
            raise ValueError("action sequence exceeds positional table")
        x = self.input(actions)
        x = x + self.position[: x.shape[1]][None]
        x = self.temporal(x.transpose(1, 2))
        x = F.adaptive_avg_pool1d(x, self.n_tokens).transpose(1, 2)
        return self.norm(x + self.slot[None])


class _ActionShim(nn.Module):
    """Frozen ``_AdapterBlock`` + the new zero-gated action adapter.

    Not checkpointed itself (no nested checkpointing; the inner block still
    checkpoints block+physics-adapter when training). Zero-init gate makes a
    fresh shim an exact no-op, as does ``ref["ctx"] is None``.
    """

    def __init__(self, inner, adapter, ref):
        super().__init__()
        self.inner = inner
        self.adapter = adapter
        self._ref = ref               # plain dict, not a submodule

    def forward(self, x, **kwargs):
        x = self.inner(x, **kwargs)
        ctx = self._ref["ctx"]
        if ctx is not None:
            x = self.adapter(x, ctx.to(x.dtype)).to(x.dtype)
        return x


def install_action_bank(dit, heads=4, dropout=0.0):
    """-> (bank, ref). Re-wraps the quarter-depth blocks of a built
    ``RealWanDiT`` in place. Call AFTER ``load_adaptive_state_dict`` and
    after freezing LoRA (both address blocks by their pre-shim names)."""
    bank = PhysicsAdapterBank(dit.model.wan_config(), d_phys=ACTION_WIDTH,
                              n_adapters=4, heads=heads,
                              dropout=dropout).float().to(dit.device)
    ref = {"ctx": None}
    blocks = dit.model.dit.blocks
    for i in bank.block_idx:
        if not isinstance(blocks[i], _AdapterBlock):
            raise RuntimeError(
                f"block {i} is {type(blocks[i]).__name__}, not _AdapterBlock "
                "— backend internals moved, or the bank was installed twice")
        blocks[i] = _ActionShim(blocks[i], bank.adapter_for(i), ref)
    return bank, ref


def make_action_fm_loss(dit, text_ctx, action_encoder, ref,
                        sigma_sampler=None):
    """``backend.make_real_fm_loss`` with an action pathway, no ROI weight.

    Same ``(loss, tau, eps)`` contract, so shared-noise paired triples keep
    working: the caller passes the same ``(tau, eps)`` back in for the
    wrong-action and null-action conditions. ``tokens`` stays the frozen
    physics-token argument; ``action_seq`` ([B, 73, 8] float or None) drives
    the shims via ``ref``.
    """
    device = dit.device
    shift = dit.arm.shift
    per_token_t = dit.arm.repo == "wan22"

    def action_fm_loss(_dit, batch, tokens, prefix_bins, tau=None, eps=None,
                       generator=None, action_seq=None):
        x0 = batch["z"].to(device=device, dtype=torch.float32)
        b = x0.shape[0]
        if tau is None:
            if sigma_sampler is not None and generator is None:
                tau = sigma_sampler(b)
            else:
                tau = sample_sigmas(b, shift, device=device,
                                    generator=generator)
        tau = tau.to(device)
        x_t, v_target, t, eps = make_targets(x0, tau, eps=eps,
                                             generator=generator)
        seq_len = W.seq_len_of(x0.shape)
        if per_token_t:
            t = t[:, None].expand(-1, seq_len)
        ctx = text_ctx(batch["idx"])
        a_tokens = None
        if action_seq is not None:
            a_tokens = action_encoder(
                action_seq.to(device=device, dtype=torch.float32))
        ref["ctx"] = a_tokens
        try:
            with torch.autocast("cuda", torch.bfloat16):
                pred = dit.model(x=list(x_t), t=t, context=ctx,
                                 seq_len=seq_len,
                                 physics_ctx=tokens.to(device))
        finally:
            ref["ctx"] = None
        return flow_loss(pred, v_target), tau, eps

    return action_fm_loss
