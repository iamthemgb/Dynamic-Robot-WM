"""Wrapper around the stock WanModel: physics-token injection + training utils.

`wan/` is never modified — physics tokens are concatenated onto each caption's
T5 embedding before WanModel.forward pads the context list to 512 slots
(`context_lens=None`, so all slots are attended in every block's cross-attn).
"""

import os
import sys

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.environ.get("WAN21_ROOT", "/gpfs/radev/scratch/sous/mzl7/Wan2.1"))  # noqa: E402

from wan.modules.model import WanModel  # noqa: E402

MODEL_DIR = "/gpfs/radev/scratch/sous/mzl7/wan_models/Wan2.1-T2V-1.3B"


class _CkptBlock(nn.Module):
    """Gradient-checkpoint shim with the same call signature as the block."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(
                lambda x_, **kw: self.block(x_, **kw), x,
                use_reentrant=False, **kwargs)
        return self.block(x, **kwargs)


class PhysicsWan(nn.Module):

    def __init__(self, physics_encoder, model_dir=MODEL_DIR, torch_dtype=torch.bfloat16):
        super().__init__()
        self.dit = WanModel.from_pretrained(model_dir, torch_dtype=torch_dtype)
        self.encoder = physics_encoder
        self.text_len = self.dit.text_len

    def enable_gradient_checkpointing(self):
        if not isinstance(self.dit.blocks[0], _CkptBlock):
            self.dit.blocks = nn.ModuleList(_CkptBlock(b) for b in self.dit.blocks)

    def freeze_dit(self):
        self.dit.requires_grad_(False)

    def inject(self, context, phys_vec):
        """context: list of [L,4096]; returns (list of [L+K,4096], tokens [B,K,4096])."""
        tokens = self.encoder(phys_vec)
        out = []
        for i, u in enumerate(context):
            assert u.size(0) + tokens.size(1) <= self.text_len, \
                f"caption too long ({u.size(0)}) for {tokens.size(1)} physics tokens"
            out.append(torch.cat([u, tokens[i].to(u.dtype)], dim=0))
        return out, tokens

    def forward(self, x, t, context, seq_len, phys_vec=None):
        """Same as WanModel.forward, plus optional physics conditioning.

        Returns (denoised list, aux_pred [B,10] or None).
        """
        aux_pred = None
        if phys_vec is not None:
            context, tokens = self.inject(context, phys_vec)
            projected = self.dit.text_embedding(tokens.to(self.base_dtype))
            aux_pred = self.encoder.aux_regress(projected.float())
        out = self.dit(x, t=t, context=context, seq_len=seq_len)
        return out, aux_pred

    @property
    def base_dtype(self):
        return self.dit.text_embedding[0].weight.dtype
