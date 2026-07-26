"""Shared fixtures for the unit tests (CPU, tiny config)."""

import torch

from ..config import tiny_config
from ..data.counterfactual_dataset import GroupBatcher, build_cache
from ..models.metadata_records import (MetadataRegistry, TypedRecord,
                                       collate_records)
from ..data.normalize_metadata import MetadataNormalizer
from ..training import common

PAIR = "primary_object--support_surface"


def example_records():
    return [
        TypedRecord("gravity_z", "global", "m_per_s2", -9.81),
        TypedRecord("mass", "primary_object", "kg", 0.20),
        TypedRecord("dynamic_friction", PAIR, "dimensionless", 0.35),
        TypedRecord("restitution", PAIR, "dimensionless", 0.72),
    ]


def fitted_normalizer(registry):
    sets = [example_records(),
            [TypedRecord("mass", "primary_object", "kg", 1.5),
             TypedRecord("dynamic_friction", PAIR, "dimensionless", 0.1),
             TypedRecord("gravity_z", "global", "m_per_s2", -9.81),
             TypedRecord("restitution", PAIR, "dimensionless", 0.4)]]
    return MetadataNormalizer(registry).fit(sets)


def small_world(seed=0):
    """cfg, cache, registry, normalizer, batcher — one tiny dataset."""
    cfg = tiny_config()
    common.set_seed(seed)
    vae = common.build_vae(cfg)
    cache, registry, normalizer = build_cache(cfg, vae, seed=seed)
    return cfg, cache, registry, normalizer, GroupBatcher(cache, seed=seed)


def collate_one(records, registry, normalizer, pad_to=None):
    return collate_records([records], registry, normalizer, pad_to=pad_to)


def randomize_out(dit, seed=7):
    """The mock DiT's output head is zero-initialized (the flow-matching
    training convention), which makes every output identically zero at init
    and output-comparison tests vacuous. Tests that compare outputs
    randomize it first."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        dit.out.weight.copy_(torch.randn(dit.out.weight.shape,
                                         generator=g) * 0.05)
        dit.out.bias.copy_(torch.randn(dit.out.bias.shape,
                                       generator=g) * 0.05)
    return dit


def run_tests(fns, name):
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    if failed:
        raise SystemExit(f"{name}: {failed} failed")
    print(f"{name}: all passed")
