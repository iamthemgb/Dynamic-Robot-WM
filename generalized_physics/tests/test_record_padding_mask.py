"""Padding invariance: masked slots change nothing (plan unit test 2)."""

import torch

from ..config import tiny_config
from ..models.metadata_records import MetadataRegistry
from ..models.metadata_teacher import MetadataTeacher
from .util import collate_one, example_records, fitted_normalizer, run_tests


def test_padding_invariance():
    registry = MetadataRegistry()
    norm = fitted_normalizer(registry)
    t = tiny_config().teacher
    torch.manual_seed(0)
    teacher = MetadataTeacher(registry, record_width=t.record_width,
                              key_embed=t.key_embed,
                              scope_embed=t.scope_embed,
                              unit_embed=t.unit_embed,
                              belief_dim=t.belief_dim).eval()
    recs = example_records()
    with torch.no_grad():
        b = teacher.forward_batch(collate_one(recs, registry, norm))
        b_pad = teacher.forward_batch(
            collate_one(recs, registry, norm, pad_to=16))
    torch.testing.assert_close(b, b_pad, atol=1e-6, rtol=1e-6)


def test_padded_values_cannot_leak():
    """Garbage in masked value slots must not reach the belief."""
    registry = MetadataRegistry()
    norm = fitted_normalizer(registry)
    t = tiny_config().teacher
    torch.manual_seed(0)
    teacher = MetadataTeacher(registry, record_width=t.record_width,
                              key_embed=t.key_embed,
                              scope_embed=t.scope_embed,
                              unit_embed=t.unit_embed,
                              belief_dim=t.belief_dim).eval()
    batch = collate_one(example_records(), registry, norm, pad_to=12)
    poisoned = dict(batch)
    poisoned["values"] = batch["values"].clone()
    poisoned["values"][:, 4:] = 1e6
    with torch.no_grad():
        b = teacher.forward_batch(batch)
        b_p = teacher.forward_batch(poisoned)
    torch.testing.assert_close(b, b_p, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    run_tests([test_padding_invariance, test_padded_values_cannot_leak],
              __file__)
