"""Real Wan2.1 insertion: frozen DiT + quarter-depth adapters + LoRA.

Mirrors the proven wrapped-block pattern of
Wan2.1/projectile_adaptive_smoke/models/adaptive_wan.py (wan/ is never
edited; blocks are wrapped, not modified), with the generalized plan's
Wan-side capacity:

  * PhysicsAdapterBank at quarter-depth blocks, consuming the FIXED
    [B, K=8, d_phys=256] tokens from the shared projector (the adapter maps
    256 -> the DiT width read from the loaded model);
  * rank-16 zero-init LoRA on the .q/.v attention projections of every
    block (real WanSelfAttention/WanT2VCrossAttention use .q/.k/.v/.o
    Linears, so dit_lora.apply_lora targets them by attribute name exactly
    as it does on the mock);
  * a single LoRA on/off switch for adapter-only attribution runs.

Requires the Wan2.1 repo on sys.path and a downloaded checkpoint; nothing
here is imported by the CPU tests or phase scripts.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .dit_lora import apply_lora, lora_params, set_lora_enabled
from .physics_adapter import PhysicsAdapterBank

WAN_REPO = Path("/gpfs/radev/home/mzl7/scratch/Wan2.1")
MODEL_DIR = "/gpfs/radev/scratch/sous/mzl7/wan_models/Wan2.1-T2V-1.3B"


class _AdapterBlock(nn.Module):
    """Frozen (LoRA-augmented) WanAttentionBlock + optional physics adapter.

    Per-call physics context travels through a shared mutable holder set by
    GeneralizedAdaptiveWan.forward, because the stock block loop passes a
    fixed kwargs dict. Gradient checkpointing covers block+adapter together
    (ctx is a positional tensor input so its grads survive the recompute).
    """

    def __init__(self, block, adapter, ctx_ref):
        super().__init__()
        self.block = block
        self.adapter = adapter
        self._ctx_ref = ctx_ref          # plain dict, not a submodule

    def _run(self, x, ctx, kwargs):
        x = self.block(x, **kwargs)
        if ctx is not None and self.adapter is not None:
            x = self.adapter(x, ctx.to(x.dtype)).to(x.dtype)
        return x

    def forward(self, x, **kwargs):
        ctx = self._ctx_ref["ctx"]
        if self.training and torch.is_grad_enabled():
            if ctx is None or self.adapter is None:
                return checkpoint(lambda x_: self.block(x_, **kwargs), x,
                                  use_reentrant=False)
            return checkpoint(lambda x_, c_: self._run(x_, c_, kwargs), x,
                              ctx, use_reentrant=False)
        return self._run(x, ctx, kwargs)


class GeneralizedAdaptiveWan(nn.Module):
    def __init__(self, cfg, model_dir=MODEL_DIR, torch_dtype=torch.bfloat16):
        """cfg: config.PipelineConfig (projector/adapters/lora sections)."""
        super().__init__()
        if str(WAN_REPO) not in sys.path:
            sys.path.insert(0, str(WAN_REPO))
        from wan.modules.model import WanModel

        self.dit = WanModel.from_pretrained(model_dir,
                                            torch_dtype=torch_dtype)
        self.dit.requires_grad_(False)

        wan_cfg = self.wan_config()
        self.physics = PhysicsAdapterBank(
            wan_cfg, d_phys=cfg.projector.d_phys,
            n_adapters=cfg.adapters.n_adapters,
            heads=cfg.adapters.heads, dropout=cfg.adapters.dropout).float()

        self._ctx_ref = {"ctx": None}
        wrapped = []
        for i, blk in enumerate(self.dit.blocks):
            wrapped.append(_AdapterBlock(blk, self.physics.adapter_for(i),
                                         self._ctx_ref))
        self.dit.blocks = nn.ModuleList(wrapped)

        # LoRA on every block's q/v projections (self- and cross-attention);
        # zero-init B keeps the wrapped model bit-exact the frozen base
        self.lora_names = apply_lora(
            self.dit.blocks, targets=cfg.lora.targets, rank=cfg.lora.rank,
            alpha=cfg.lora.alpha, dropout=cfg.lora.dropout)

    # -- contract ----------------------------------------------------------
    def wan_config(self):
        """Widths read from the loaded model, never hard-coded."""
        return {"d_model": self.dit.dim, "n_blocks": len(self.dit.blocks)}

    # -- parameter groups / switches --------------------------------------
    def adapter_params(self):
        return list(self.physics.parameters())

    def lora_parameters(self):
        return lora_params(self.dit.blocks)

    def trainable_params(self):
        return self.adapter_params() + self.lora_parameters()

    def set_lora_enabled(self, enabled: bool):
        """Adapter-only attribution mode when False."""
        set_lora_enabled(self.dit.blocks, enabled)

    # -- state (never save the frozen base DiT) ---------------------------
    def adaptive_state_dict(self):
        return {"physics": self.physics.state_dict(),
                "lora": {n: p.detach().clone()
                         for n, p in self.dit.blocks.named_parameters()
                         if "lora_" in n},
                "lora_names": self.lora_names}

    def load_adaptive_state_dict(self, state):
        assert state["lora_names"] == self.lora_names, \
            "checkpoint LoRA placement differs from config"
        self.physics.load_state_dict(state["physics"])
        own = dict(self.dit.blocks.named_parameters())
        with torch.no_grad():
            for n, v in state["lora"].items():
                own[n].copy_(v)

    # -- forward -----------------------------------------------------------
    def forward(self, x, t, context, seq_len, physics_ctx=None):
        """WanModel.forward contract plus physics tokens.

        physics_ctx: [B, K, d_phys] from the shared projector (teacher or
        student belief), or None to bypass the adapters (with LoRA disabled
        as well, that is exactly the frozen base model).
        """
        self._ctx_ref["ctx"] = physics_ctx
        try:
            return self.dit(x, t=t, context=context, seq_len=seq_len)
        finally:
            self._ctx_ref["ctx"] = None
