"""Per-key value normalization with training-set statistics (plan eq. 6).

v_tilde = (g_k(v) - mu_k) / (sigma_k + eps), where g_k is the registry
transform: identity for bounded values, log for positive spans, clipped logit
for (0, 1) quantities. mu_k / sigma_k / transform are stored in a JSON
manifest so cached datasets carry their normalization provenance.
"""

import json
import math
from pathlib import Path

_EPS = 1e-6
_LOGIT_CLIP = 1e-4


def _fwd(transform, v):
    if transform == "identity":
        return float(v)
    if transform == "log":
        if v <= 0:
            raise ValueError(f"log transform requires positive value, got {v}")
        return math.log(v)
    if transform == "logit_clipped":
        p = min(max(float(v), _LOGIT_CLIP), 1.0 - _LOGIT_CLIP)
        return math.log(p / (1.0 - p))
    raise KeyError(f"unknown transform {transform!r}")


def _inv(transform, t):
    if transform == "identity":
        return float(t)
    if transform == "log":
        return math.exp(t)
    if transform == "logit_clipped":
        return 1.0 / (1.0 + math.exp(-t))
    raise KeyError(f"unknown transform {transform!r}")


class MetadataNormalizer:
    def __init__(self, registry, stats=None):
        self.registry = registry
        self.stats = stats or {}   # key -> {"mu": float, "sigma": float}

    def fit(self, record_sets):
        """record_sets: iterable of lists of TypedRecord (training split
        only). Computes per-key mu/sigma of transformed values."""
        acc = {}
        for rs in record_sets:
            for r in rs:
                acc.setdefault(r.key, []).append(
                    _fwd(self.registry.transform_of(r.key), r.value))
        self.stats = {}
        for key, vals in acc.items():
            n = len(vals)
            mu = sum(vals) / n
            var = sum((v - mu) ** 2 for v in vals) / max(n - 1, 1)
            self.stats[key] = {"mu": mu, "sigma": math.sqrt(var)}
        return self

    def normalize(self, key, value):
        s = self.stats[key]
        t = _fwd(self.registry.transform_of(key), value)
        return (t - s["mu"]) / (s["sigma"] + _EPS)

    def denormalize(self, key, v_tilde):
        s = self.stats[key]
        t = v_tilde * (s["sigma"] + _EPS) + s["mu"]
        return _inv(self.registry.transform_of(key), t)

    def save(self, path):
        manifest = {"registry": str(self.registry.path),
                    "transforms": {k: self.registry.transform_of(k)
                                   for k in self.stats},
                    "stats": self.stats}
        Path(path).write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, registry, path):
        manifest = json.loads(Path(path).read_text())
        for k, tr in manifest["transforms"].items():
            if registry.transform_of(k) != tr:
                raise ValueError(
                    f"manifest transform {tr!r} for key {k!r} does not match "
                    f"the registry ({registry.transform_of(k)!r})")
        return cls(registry, manifest["stats"])
