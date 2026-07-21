"""F2b — physically supported ramp launch followed by interception."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from ._rigid_shared import rigid_module


SCENARIO = rigid_module("F2b", "projectile_rebound", "ramp_launch", fixture_policy="owned grounded ramp with four supports", controller_kind="catch_or_deflect", trajectory="jerk_limited_predictive_reach", retention_required=False, hand_orientation="pick_down", compact_pickup_ready=True, robotiq_reach_arrival_lead_s=0.13, robotiq_tendon_profile="f2c", interior_joint_margin_rad=0.01)


def _finite_vector(value: tuple[float, ...], size: int, label: str) -> None:
    if len(value) != size or any(not math.isfinite(float(item)) for item in value):
        raise ValueError(f"{label} must contain {size} finite values")


@dataclass(frozen=True, slots=True)
class SurfaceTransitionContract:
    support_surface_id: str
    support_normal_world_xyz: tuple[float, float, float]
    minimum_support_contact_s: float = 0.08
    minimum_free_flight_s: float = 0.08
    require_ordered_termination: bool = True
    schema_version: str = "surface-to-free-flight/v1"

    def validate(self) -> None:
        if self.schema_version != "surface-to-free-flight/v1":
            raise ValueError("unsupported surface-transition contract")
        if not self.support_surface_id:
            raise ValueError("surface-transition contract lacks a stable surface ID")
        _finite_vector(self.support_normal_world_xyz, 3, "support normal")
        if abs(np.linalg.norm(self.support_normal_world_xyz) - 1.0) > 1e-6:
            raise ValueError("support normal must be normalized")
        if self.minimum_support_contact_s <= 0 or self.minimum_free_flight_s <= 0:
            raise ValueError("surface-transition durations must be positive")
        if not self.require_ordered_termination:
            raise ValueError("support termination must remain ordered")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


__all__ = ["SCENARIO", "SurfaceTransitionContract"]
