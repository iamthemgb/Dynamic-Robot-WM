"""Shared utilities for the phase scripts."""

import numpy as np
import torch

from ..models.causal_student import CausalStudent
from ..models.metadata_query_decoder import MetadataQueryDecoder
from ..models.metadata_teacher import MetadataTeacher
from ..models.physics_projector import PhysicsProjector
from ..wan.dit_lora import apply_lora, lora_params
from ..wan.mock_wan import MockWanDiT, MockWanVAE, flow_sample


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def r2_score(pred, target):
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0) + 1e-12
    return 1.0 - ss_res / ss_tot


# -- builders (all widths flow from cfg or wan_config, never literals) -----

def build_vae(cfg):
    v = cfg.vae
    return MockWanVAE(v.latent_channels, v.temporal_stride, v.spatial_stride,
                      seed=v.seed)


def build_teacher(cfg, registry):
    t = cfg.teacher
    return MetadataTeacher(
        registry, record_width=t.record_width, key_embed=t.key_embed,
        scope_embed=t.scope_embed, unit_embed=t.unit_embed,
        belief_dim=t.belief_dim, output_scale=t.output_scale,
        tanh_output=t.tanh_output)


def build_decoder(cfg, teacher):
    return MetadataQueryDecoder(teacher.embed, cfg.teacher.belief_dim,
                                hidden=cfg.teacher.record_width)


def build_student(cfg, action_dim=1, state_dim=0):
    s = cfg.student
    return CausalStudent(
        latent_channels=cfg.vae.latent_channels, action_dim=action_dim,
        state_dim=state_dim, belief_dim=cfg.teacher.belief_dim,
        width=s.width, gru_layers=s.gru_layers, gru_hidden=s.gru_hidden,
        ctrl_width=s.ctrl_width, use_delta_z=s.use_delta_z,
        dropout=s.dropout)


def build_projector(cfg):
    p = cfg.projector
    return PhysicsProjector(cfg.teacher.belief_dim, p.k_tokens, p.d_phys,
                            p.hidden)


def build_dit(cfg, action_dim=1, with_lora=True):
    d = cfg.dit
    dit = MockWanDiT(cfg.vae.latent_channels, d_model=d.d_model,
                     heads=d.heads, blocks=d.blocks, action_dim=action_dim,
                     d_phys=cfg.projector.d_phys,
                     k_tokens=cfg.projector.k_tokens,
                     n_adapters=cfg.adapters.n_adapters,
                     adapter_heads=cfg.adapters.heads)
    # freeze the backbone, then add trainable capacity: adapters + LoRA
    dit.requires_grad_(False)
    dit.physics.requires_grad_(True)
    if with_lora:
        apply_lora(dit.blocks, targets=cfg.lora.targets, rank=cfg.lora.rank,
                   alpha=cfg.lora.alpha, dropout=cfg.lora.dropout)
    return dit


def dedupe_params(params):
    """Drop duplicate Parameter objects (shared submodules such as the
    teacher/decoder embedder) while preserving order."""
    return list(dict.fromkeys(params))


def wan_side_params(cfg, dit, projector):
    """The projector/adapter/LoRA parameter group (single low LR)."""
    return dedupe_params(list(projector.parameters())
                         + list(dit.physics.parameters())
                         + lora_params(dit))


# -- forward glue ----------------------------------------------------------

def student_window_forward(student, win):
    """Run the student on a GroupBatcher.window() dict."""
    return student(win["z"], win["actions"], win["bin_index"],
                   delta_z=win.get("delta") if student.use_delta_z else None,
                   delta_valid=win.get("delta_valid")
                   if student.use_delta_z else None)


def split_prefix_future(z, prefix_bins):
    return z[:, :, :prefix_bins], z[:, :, prefix_bins:]


def fm_loss(dit, batch, tokens, prefix_bins, tau=None, eps=None,
            generator=None):
    """Flow-matching loss on the future bins; returns (loss, tau, eps).

    Paired conditions MUST pass the same (tau, eps) back in — the
    shared-noise ranking contract.
    """
    prefix_z, future_z = split_prefix_future(batch["z"], prefix_bins)
    x_tau, tau, eps, v_star = flow_sample(future_z, tau=tau, eps=eps,
                                          generator=generator)
    v = dit(x_tau, tau, prefix_z, batch["actions"], batch["bin_index"],
            tokens)
    return ((v - v_star) ** 2).mean(), tau, eps
