"""F2e — ordered physical multi-surface rebound and retained catch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from ._rigid_shared import rigid_module


def _finite_vector(value: tuple[float, ...], size: int, label: str) -> None:
    if len(value) != size or any(not math.isfinite(float(item)) for item in value):
        raise ValueError(f"{label} must contain {size} finite values")


F2E_FLOOR_TO_WALL_REPAIR_SCHEMA = "dynamic-robot-f2e-floor-to-wall-repair/v1"


@dataclass(frozen=True, slots=True)
class FloorToWallRepairProfile:
    """Measured 1200 Hz repair for only the F2e floor-to-wall variant.

    The physical floor and wall construction remains derived from the original
    fixed seed.  These constants remove unused wall extent from the robot
    workspace and bind the controller to the measured post-wall free-flight
    trajectory.  They are not applied to ``flight_to_table_bounce``.
    """

    wall_tangent_half_extent_m: float = 0.10
    catch_event_time_s: float = 0.590
    measured_target_bias_world_xyz_m: tuple[float, float, float] = (
        0.07385,
        -0.036865,
        0.09596,
    )
    robotiq_aim_bias_world_xyz_m: tuple[float, float, float] = (
        -0.040,
        0.0,
        0.022,
    )
    negative_controller_aim_delta_world_xyz_m: tuple[float, float, float] = (
        0.0,
        -0.10,
        0.0,
    )
    required_simulation_hz: int = 1200
    rejected_comparison_simulation_hz: int = 600
    schema_version: str = F2E_FLOOR_TO_WALL_REPAIR_SCHEMA

    def validate(self) -> None:
        if self.schema_version != F2E_FLOOR_TO_WALL_REPAIR_SCHEMA:
            raise ValueError("unsupported F2e floor-to-wall repair profile")
        if not 0.05 <= self.wall_tangent_half_extent_m <= 0.20:
            raise ValueError("F2e repaired wall extent is outside its measured range")
        if not 0.5 < self.catch_event_time_s < 0.7:
            raise ValueError("F2e repaired catch event lies outside the rebound window")
        for label, value in (
            ("measured target bias", self.measured_target_bias_world_xyz_m),
            ("Robotiq aim bias", self.robotiq_aim_bias_world_xyz_m),
            (
                "negative controller aim delta",
                self.negative_controller_aim_delta_world_xyz_m,
            ),
        ):
            _finite_vector(value, 3, label)
        if self.robotiq_aim_bias_world_xyz_m[2] <= 0.0:
            raise ValueError("F2e Robotiq aim must retain positive jaw clearance")
        if self.negative_controller_aim_delta_world_xyz_m != (0.0, -0.10, 0.0):
            raise ValueError("F2e negative controller branch lost its safe fixed miss")
        if (
            self.rejected_comparison_simulation_hz,
            self.required_simulation_hz,
        ) != (600, 1200):
            raise ValueError("F2e floor-to-wall timestep decision is not calibrated")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


FLOOR_TO_WALL_REPAIR = FloorToWallRepairProfile()
FLOOR_TO_WALL_REPAIR.validate()


SCENARIO = rigid_module(
    "F2e",
    "projectile_rebound",
    "multi_surface_rebound",
    fixture_policy="ordered grounded floor/table and wall fixtures",
    controller_kind="catch_after_ordered_rebounds",
    trajectory="jerk_limited_predictive_reach",
    retention_required=True,
    hand_orientation="pick_down",
    compact_pickup_ready=True,
    robotiq_tendon_profile="pickup",
    interior_joint_margin_rad=0.01,
)


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


__all__ = [
    "F2E_FLOOR_TO_WALL_REPAIR_SCHEMA",
    "FLOOR_TO_WALL_REPAIR",
    "FloorToWallRepairProfile",
    "OrderedContactContract",
    "SCENARIO",
]
