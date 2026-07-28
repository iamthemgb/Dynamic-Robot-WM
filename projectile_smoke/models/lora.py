"""Minimal hand-rolled LoRA for WanModel (peft is unnecessary here).

apply_lora() swaps target nn.Linear layers for LoRALinear in place; the base
weight stays frozen. merge_lora() folds A@B into the base weight for
deployment / DPO reference models.
"""

import torch
import torch.nn as nn

# module paths inside each WanAttentionBlock
DEFAULT_TARGETS = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
)
CROSS_ATTN_TARGETS = ("cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o")


class LoRALinear(nn.Module):

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        # fp32 params for optimizer stability; autocast downcasts in matmul
        self.lora_a = nn.Parameter(
            torch.randn(rank, base.in_features, dtype=torch.float32) * (1.0 / rank))
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=torch.float32))

    def forward(self, x):
        a = self.lora_a.to(x.dtype)
        b = self.lora_b.to(x.dtype)
        return self.base(x) + (x @ a.T @ b.T) * self.scale

    @torch.no_grad()
    def merge(self):
        delta = (self.lora_b @ self.lora_a) * self.scale
        self.base.weight += delta.to(self.base.weight.dtype)


def _resolve(block, path):
    obj = block
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _parent_and_leaf(block, path):
    parts = path.split(".")
    parent = block
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def apply_lora(wan_model, targets=DEFAULT_TARGETS, rank=32, alpha=32):
    """Wrap target linears in every block. Returns list of LoRA parameters."""
    params = []
    for block in wan_model.blocks:
        inner = block.block if hasattr(block, "block") else block  # ckpt shim
        for path in targets:
            parent, leaf = _parent_and_leaf(inner, path)
            base = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
            assert isinstance(base, nn.Linear), f"{path} is {type(base)}"
            wrapped = LoRALinear(base, rank, alpha)
            if leaf.isdigit():
                parent[int(leaf)] = wrapped
            else:
                setattr(parent, leaf, wrapped)
            params += [wrapped.lora_a, wrapped.lora_b]
    return params


def lora_state_dict(wan_model):
    return {k: v for k, v in wan_model.state_dict().items()
            if "lora_a" in k or "lora_b" in k}


def load_lora_state_dict(wan_model, state):
    missing, unexpected = wan_model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected


def merge_lora(wan_model):
    for m in wan_model.modules():
        if isinstance(m, LoRALinear):
            m.merge()
