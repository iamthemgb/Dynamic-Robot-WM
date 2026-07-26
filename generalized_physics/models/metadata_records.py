"""Typed metadata records: schema, registry vocabularies, batching masks.

A record is m = (key, scope, unit, value) — one scalar per record (vector
quantities are pre-split into gravity_x/gravity_y/gravity_z). Within one
episode the (key, scope) pair must be unique. Scopes are canonical semantic
roles from the registry; raw simulator entity IDs are rejected here so they
can never enter the teacher (plan: scope sanity).

Index 0 of every vocabulary is PAD; padded slots are excluded by valid_mask
and the teacher's masked aggregation, so padding is purely a batching detail.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

_RAW_ID = re.compile(r"^(body|geom|joint|actuator|site)_\d+$")


@dataclass(frozen=True)
class TypedRecord:
    key: str
    scope: str
    unit: str
    value: float


class MetadataRegistry:
    """Closed vocabularies for keys, scopes, and units + per-key transforms."""

    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).parents[1] / "data" / \
            "metadata_registry.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)
        self.path = path
        self.scopes = list(raw["scopes"])
        self.units = list(raw["units"])
        self.properties = dict(raw["properties"])
        self.keys = list(self.properties)
        # index 0 = PAD in every vocabulary
        self.key_to_id = {k: i + 1 for i, k in enumerate(self.keys)}
        self.scope_to_id = {s: i + 1 for i, s in enumerate(self.scopes)}
        self.unit_to_id = {u: i + 1 for i, u in enumerate(self.units)}

    @property
    def n_keys(self):
        return len(self.keys) + 1

    @property
    def n_scopes(self):
        return len(self.scopes) + 1

    @property
    def n_units(self):
        return len(self.units) + 1

    def transform_of(self, key):
        return self.properties[key]["transform"]

    def validate(self, records):
        """Raise on unknown keys/scopes/units, raw entity IDs, or duplicate
        (key, scope) pairs. Returns the records unchanged."""
        seen = set()
        for r in records:
            for part in r.scope.split("--"):
                if _RAW_ID.match(part):
                    raise ValueError(
                        f"raw simulator entity ID {part!r} in scope "
                        f"{r.scope!r}; canonicalize to a semantic role first")
            if r.key not in self.key_to_id:
                raise KeyError(f"unregistered property key {r.key!r}")
            if r.scope not in self.scope_to_id:
                raise KeyError(f"unregistered scope {r.scope!r}")
            if r.unit not in self.unit_to_id:
                raise KeyError(f"unregistered unit {r.unit!r}")
            allowed = self.properties[r.key]["allowed_scopes"]
            if r.scope not in allowed:
                raise ValueError(
                    f"scope {r.scope!r} not allowed for key {r.key!r} "
                    f"(allowed: {allowed})")
            ks = (r.key, r.scope)
            if ks in seen:
                raise ValueError(f"duplicate (key, scope) pair {ks}")
            seen.add(ks)
        return records


def collate_records(record_sets, registry, normalizer=None, pad_to=None):
    """Pad a list of variable-length record sets into one masked batch.

    record_sets : list of B lists of TypedRecord (each validated)
    normalizer  : data.normalize_metadata.MetadataNormalizer or None (raw
                  values pass through; training must always use a normalizer)
    pad_to      : optional minimum J (e.g. to test padding invariance)

    Returns dict of tensors: key_ids/scope_ids/unit_ids [B, J] long,
    values [B, J, 1] float (normalized), valid_mask [B, J] float.
    """
    B = len(record_sets)
    J = max((len(rs) for rs in record_sets), default=1)
    J = max(J, pad_to or 1)
    key_ids = torch.zeros(B, J, dtype=torch.long)
    scope_ids = torch.zeros(B, J, dtype=torch.long)
    unit_ids = torch.zeros(B, J, dtype=torch.long)
    values = torch.zeros(B, J, 1)
    mask = torch.zeros(B, J)
    for b, rs in enumerate(record_sets):
        registry.validate(rs)
        for j, r in enumerate(rs):
            key_ids[b, j] = registry.key_to_id[r.key]
            scope_ids[b, j] = registry.scope_to_id[r.scope]
            unit_ids[b, j] = registry.unit_to_id[r.unit]
            v = normalizer.normalize(r.key, r.value) if normalizer else r.value
            values[b, j, 0] = float(v)
            mask[b, j] = 1.0
    return {"key_ids": key_ids, "scope_ids": scope_ids, "unit_ids": unit_ids,
            "values": values, "valid_mask": mask}


class RecordEmbedder(torch.nn.Module):
    """Shared key/scope/unit embeddings (index 0 = PAD) used by the teacher,
    the query decoder, and the phase-0 probe, so queries and records live in
    one embedding space."""

    def __init__(self, registry, key_dim=64, scope_dim=32, unit_dim=16):
        super().__init__()
        self.key = torch.nn.Embedding(registry.n_keys, key_dim, padding_idx=0)
        self.scope = torch.nn.Embedding(registry.n_scopes, scope_dim,
                                        padding_idx=0)
        self.unit = torch.nn.Embedding(registry.n_units, unit_dim,
                                       padding_idx=0)
        self.out_dim = key_dim + scope_dim + unit_dim

    def forward(self, key_ids, scope_ids, unit_ids):
        """[..., J] id tensors -> [..., J, out_dim]."""
        return torch.cat([self.key(key_ids), self.scope(scope_ids),
                          self.unit(unit_ids)], dim=-1)
