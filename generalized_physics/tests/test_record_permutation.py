"""Record-order invariance + scope sanity (plan unit tests 1 and 3)."""

import torch

from ..config import tiny_config
from ..models.metadata_records import MetadataRegistry, TypedRecord
from ..models.metadata_teacher import MetadataTeacher
from .util import collate_one, example_records, fitted_normalizer, run_tests


def _teacher(registry):
    cfg = tiny_config().teacher
    torch.manual_seed(0)
    return MetadataTeacher(registry, record_width=cfg.record_width,
                           key_embed=cfg.key_embed,
                           scope_embed=cfg.scope_embed,
                           unit_embed=cfg.unit_embed,
                           belief_dim=cfg.belief_dim).eval()


def test_permutation_invariance():
    registry = MetadataRegistry()
    norm = fitted_normalizer(registry)
    teacher = _teacher(registry)
    recs = example_records()
    with torch.no_grad():
        b0 = teacher.forward_batch(collate_one(recs, registry, norm))
        b1 = teacher.forward_batch(collate_one(recs[::-1], registry, norm))
        b2 = teacher.forward_batch(
            collate_one([recs[2], recs[0], recs[3], recs[1]], registry,
                        norm))
    torch.testing.assert_close(b0, b1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b0, b2, atol=1e-5, rtol=1e-5)


def test_variable_record_count_same_shape():
    registry = MetadataRegistry()
    norm = fitted_normalizer(registry)
    teacher = _teacher(registry)
    with torch.no_grad():
        b4 = teacher.forward_batch(
            collate_one(example_records(), registry, norm))
        b2 = teacher.forward_batch(
            collate_one(example_records()[:2], registry, norm))
    assert b4.shape == b2.shape == (1, teacher.belief_dim)


def test_scope_sanity_raw_ids_rejected():
    registry = MetadataRegistry()
    bad = [TypedRecord("mass", "body_17", "kg", 1.0)]
    try:
        registry.validate(bad)
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("raw simulator entity ID accepted as scope")
    pair_bad = [TypedRecord("restitution", "body_1--support_surface",
                            "dimensionless", 0.5)]
    try:
        registry.validate(pair_bad)
    except (ValueError, KeyError):
        pass
    else:
        raise AssertionError("raw entity ID inside pair scope accepted")


def test_duplicate_key_scope_rejected():
    registry = MetadataRegistry()
    dup = [TypedRecord("mass", "primary_object", "kg", 1.0),
           TypedRecord("mass", "primary_object", "kg", 2.0)]
    try:
        registry.validate(dup)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate (key, scope) accepted")


if __name__ == "__main__":
    run_tests([test_permutation_invariance,
               test_variable_record_count_same_shape,
               test_scope_sanity_raw_ids_rejected,
               test_duplicate_key_scope_rejected], __file__)
