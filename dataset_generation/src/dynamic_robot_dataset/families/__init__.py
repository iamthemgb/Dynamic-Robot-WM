"""Registry for deterministic family planning and smoke simulation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from .base import EpisodePlan, FamilyAdapter, GenerationRequest, SimulationResult


_CANONICAL = (
    "falling_catch",
    "rolling_interception",
    "projectile_rebound",
    "cloth",
    "rope",
    "soft_body",
    "legacy_proxy_quarantine",
)

_ALIASES = {
    "falling": "falling_catch",
    "catch": "falling_catch",
    "rolling": "rolling_interception",
    "projectile": "projectile_rebound",
    "rebound": "projectile_rebound",
    "softbody": "soft_body",
    "legacy_proxy": "legacy_proxy_quarantine",
}


def list_families(*, include_quarantine: bool = True) -> tuple[str, ...]:
    return _CANONICAL if include_quarantine else _CANONICAL[:-1]


@lru_cache(maxsize=None)
def get_family(name: str) -> FamilyAdapter:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized == "falling_catch":
        from .rigid_dynamic.falling_catch import FallingCatchAdapter

        return FallingCatchAdapter()
    if normalized == "rolling_interception":
        from .rigid_dynamic.rolling_interception import RollingInterceptionAdapter

        return RollingInterceptionAdapter()
    if normalized == "projectile_rebound":
        from .rigid_dynamic.projectile_rebound import ProjectileReboundAdapter

        return ProjectileReboundAdapter()
    if normalized == "cloth":
        from .deformable.cloth import ClothAdapter

        return ClothAdapter()
    if normalized == "rope":
        from .deformable.rope import RopeAdapter

        return RopeAdapter()
    if normalized == "soft_body":
        from .deformable.soft_body import SoftBodyAdapter

        return SoftBodyAdapter()
    if normalized == "legacy_proxy_quarantine":
        from .legacy_proxy_quarantine import LegacyProxyQuarantineAdapter

        return LegacyProxyQuarantineAdapter()
    raise KeyError(f"unknown family {name!r}; available: {', '.join(_CANONICAL)}")


def plan(config: Mapping[str, Any] | GenerationRequest) -> list[EpisodePlan]:
    family = config.family if isinstance(config, GenerationRequest) else str(config["family"])
    return get_family(family).plan(config)


def generate(config: Mapping[str, Any] | GenerationRequest) -> list[SimulationResult]:
    family = config.family if isinstance(config, GenerationRequest) else str(config["family"])
    return get_family(family).generate(config)


__all__ = [
    "EpisodePlan",
    "FamilyAdapter",
    "GenerationRequest",
    "SimulationResult",
    "generate",
    "get_family",
    "list_families",
    "plan",
]

