"""F2e — ordered physical multi-surface rebound and retained catch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from ._rigid_shared import rigid_module


SCENARIO = rigid_module("F2e", "projectile_rebound", "multi_surface_rebound", fixture_policy="ordered grounded floor/table and wall fixtures", controller_kind="catch_after_ordered_rebounds", trajectory="jerk_limited_predictive_reach", retention_required=True, hand_orientation="pick_down", compact_pickup_ready=True, robotiq_tendon_profile="pickup", interior_joint_margin_rad=0.01)


def _finite_vector(value: tuple[float, ...], size: int, label: str) -> None:
    if len(value) != size or any(not math.isfinite(float(item)) for item in value):
        raise ValueError(f"{label} must contain {size} finite values")


@dataclass(frozen=True, slots=True)
class OrderedContactContract:
    ordered_surface_ids: tuple[str, ...]
    ordered_surface_normals_world_xyz: tuple[tuple[float, float, float], ...]
    minimum_separated_pre_post_samples: int = 2
    minimum_inter_contact_free_flight_s: float = 0.04
    reject_contact_chatter: bool = True
    schema_version: str = "ordered-surface-contacts/v1"

    def validate(self) -> None:
        if self.schema_version != "ordered-surface-contacts/v1":
            raise ValueError("unsupported ordered-contact contract")
        if len(self.ordered_surface_ids) < 2 or len(set(self.ordered_surface_ids)) != len(
            self.ordered_surface_ids
        ):
            raise ValueError("ordered contacts require at least two unique stable IDs")
        if len(self.ordered_surface_normals_world_xyz) != len(self.ordered_surface_ids):
            raise ValueError("ordered surface normals do not match the ID sequence")
        for normal in self.ordered_surface_normals_world_xyz:
            _finite_vector(normal, 3, "ordered surface normal")
            if abs(np.linalg.norm(normal) - 1.0) > 1e-6:
                raise ValueError("ordered surface normals must be normalized")
        if self.minimum_separated_pre_post_samples < 2:
            raise ValueError("ordered contacts require separated pre/post samples")
        if self.minimum_inter_contact_free_flight_s <= 0:
            raise ValueError("ordered contacts require a positive free-flight gap")
        if not self.reject_contact_chatter:
            raise ValueError("contact chatter rejection cannot be disabled")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


__all__ = ["SCENARIO", "OrderedContactContract"]
