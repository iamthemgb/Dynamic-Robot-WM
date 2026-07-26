"""Projector contract (plan unit test 8): teacher and student beliefs map
to the SAME fixed token interface, independent of task schema and Wan
width; the belief mean is the only accepted input."""

import torch

from ..config import tiny_config
from ..models.physics_projector import PhysicsProjector
from ..training import common
from .util import (collate_one, example_records, fitted_normalizer,
                   run_tests, small_world)


def test_shared_shape_for_teacher_and_student():
    cfg, cache, registry, norm, batcher = small_world()
    teacher = common.build_teacher(cfg, registry).eval()
    student = common.build_student(cfg).eval()
    proj = common.build_projector(cfg).eval()
    batch = batcher.episode_batch(3)
    win = batcher.window(batch, W=batch["z"].shape[2])
    with torch.no_grad():
        c_t = proj(teacher.forward_batch(batch["rec"]))
        c_s = proj(common.student_window_forward(student, win)[
            "belief_final"])
    K, d = cfg.projector.k_tokens, cfg.projector.d_phys
    assert c_t.shape == c_s.shape == (3, K, d)


def test_token_width_independent_of_wan_width():
    cfg = tiny_config()
    proj = PhysicsProjector(cfg.teacher.belief_dim, cfg.projector.k_tokens,
                            cfg.projector.d_phys, cfg.projector.hidden)
    for d_model in (32, 64, 128):     # the DiT width may change...
        dit = common.build_dit(cfg, with_lora=False)
        assert proj.d_phys == cfg.projector.d_phys  # ...the tokens never do
    tokens = proj(torch.zeros(2, cfg.teacher.belief_dim))
    assert tokens.shape[-1] == cfg.projector.d_phys


def test_variable_schema_same_tokens():
    """4-record and 2-record episodes produce identically shaped tokens."""
    cfg, _, registry, norm, _ = small_world()
    teacher = common.build_teacher(cfg, registry).eval()
    proj = common.build_projector(cfg).eval()
    with torch.no_grad():
        c4 = proj(teacher.forward_batch(
            collate_one(example_records(), registry, norm)))
        c2 = proj(teacher.forward_batch(
            collate_one(example_records()[:2], registry, norm)))
    assert c4.shape == c2.shape


def test_rejects_non_belief_input():
    cfg = tiny_config()
    proj = PhysicsProjector(cfg.teacher.belief_dim)
    for bad in (torch.zeros(2, cfg.teacher.belief_dim + 1),
                torch.zeros(2, cfg.teacher.belief_dim, 2)):
        try:
            proj(bad)
        except ValueError:
            continue
        raise AssertionError(f"projector accepted shape {tuple(bad.shape)}")


def test_null_tokens_same_interface():
    cfg = tiny_config()
    proj = PhysicsProjector(cfg.teacher.belief_dim, cfg.projector.k_tokens,
                            cfg.projector.d_phys)
    assert proj.null_tokens(5).shape == (5, cfg.projector.k_tokens,
                                         cfg.projector.d_phys)


if __name__ == "__main__":
    run_tests([test_shared_shape_for_teacher_and_student,
               test_token_width_independent_of_wan_width,
               test_variable_schema_same_tokens,
               test_rejects_non_belief_input,
               test_null_tokens_same_interface], __file__)
