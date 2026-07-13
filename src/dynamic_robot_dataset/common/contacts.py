"""Contact-event and assistance metadata types."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .schema import DynamicsMode, SchemaValidationError


def _finite_vector(value: tuple[float, ...], length: int, name: str) -> None:
    if len(value) != length or not all(math.isfinite(component) for component in value):
        raise SchemaValidationError(f"{name} must contain {length} finite values")


@dataclass(slots=True, frozen=True)
class ContactEvent:
    """One contact observation in the canonical world frame."""

    timestamp: float
    object_a: str
    object_b: str
    point_world_m: tuple[float, float, float]
    normal_world: tuple[float, float, float]
    penetration_depth_m: float
    normal_force_n: float | None = None
    normal_impulse_n_s: float | None = None
    relative_velocity_world_m_s: tuple[float, float, float] | None = None
    expected_fixture_contact: bool = False
    snag: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.timestamp) or self.timestamp < 0:
            raise SchemaValidationError("Contact timestamp must be finite and non-negative")
        if not self.object_a or not self.object_b or self.object_a == self.object_b:
            raise SchemaValidationError("Contact objects must be distinct and named")
        _finite_vector(self.point_world_m, 3, "point_world_m")
        _finite_vector(self.normal_world, 3, "normal_world")
        normal_length = math.sqrt(sum(value * value for value in self.normal_world))
        if not 0.95 <= normal_length <= 1.05:
            raise SchemaValidationError("Contact normal must be normalized")
        if not math.isfinite(self.penetration_depth_m) or self.penetration_depth_m < 0:
            raise SchemaValidationError("penetration_depth_m must be finite and non-negative")
        for name, value in (
            ("normal_force_n", self.normal_force_n),
            ("normal_impulse_n_s", self.normal_impulse_n_s),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise SchemaValidationError(f"{name} must be null or finite and non-negative")
        if self.relative_velocity_world_m_s is not None:
            _finite_vector(self.relative_velocity_world_m_s, 3, "relative_velocity_world_m_s")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(slots=True, frozen=True)
class AssistanceSample:
    """Per-frame mask for non-free-contact mechanisms."""

    timestamp: float
    assisted_grasp: bool = False
    assisted_retention: bool = False
    equality_constraint_active: bool = False
    latch_active: bool = False

    @property
    def active(self) -> bool:
        return any(
            (
                self.assisted_grasp,
                self.assisted_retention,
                self.equality_constraint_active,
                self.latch_active,
            )
        )


@dataclass(slots=True)
class AssistanceSummary:
    """Episode-level summary derived from frame-level assistance masks."""

    dynamics_mode: DynamicsMode
    assisted_grasp: bool
    assisted_retention: bool
    equality_constraint_active: bool
    latch_active: bool
    constraint_activation_time: float | None
    constraint_deactivation_time: float | None

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[AssistanceSample],
        *,
        scripted_motion: bool = False,
    ) -> "AssistanceSummary":
        ordered = sorted(samples, key=lambda sample: sample.timestamp)
        active = [sample for sample in ordered if sample.active]
        mode = (
            DynamicsMode.SCRIPTED_MOTION
            if scripted_motion
            else DynamicsMode.ASSISTED_CONTACT
            if active
            else DynamicsMode.FREE_CONTACT
        )
        return cls(
            dynamics_mode=mode,
            assisted_grasp=any(sample.assisted_grasp for sample in ordered),
            assisted_retention=any(sample.assisted_retention for sample in ordered),
            equality_constraint_active=any(sample.equality_constraint_active for sample in ordered),
            latch_active=any(sample.latch_active for sample in ordered),
            constraint_activation_time=active[0].timestamp if active else None,
            constraint_deactivation_time=active[-1].timestamp if active else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "dynamics_mode": self.dynamics_mode.value}


def normalize_assistance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy ``*_time_s`` keys to the canonical SI field names."""

    result = dict(value)
    for canonical, legacy in (
        ("constraint_activation_time", "constraint_activation_time_s"),
        ("constraint_deactivation_time", "constraint_deactivation_time_s"),
    ):
        if canonical not in result and legacy in result:
            result[canonical] = result.pop(legacy)
    for field_name in (
        "assisted_grasp",
        "assisted_retention",
        "equality_constraint_active",
        "latch_active",
    ):
        result.setdefault(field_name, False)
    result.setdefault("constraint_activation_time", None)
    result.setdefault("constraint_deactivation_time", None)
    return result


def normalize_contact_row(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy adapter contact keys to the canonical event schema."""

    result = dict(value)
    aliases = {
        "contact_point_m": "point_world_m",
        "contact_normal": "normal_world",
        "normal_impulse_Ns": "normal_impulse_n_s",
        "normal_force_N": "normal_force_n",
        "relative_velocity_pre_mps": "relative_velocity_world_m_s",
        "relative_velocity_post_mps": "relative_velocity_post_world_m_s",
        "impulse_Ns": "impulse_world_n_s",
    }
    for legacy, canonical in aliases.items():
        if legacy in result and canonical not in result:
            result[canonical] = result.pop(legacy)
    return result


def select_task_event_time(events: Iterable[Mapping[str, Any]]) -> float | None:
    """Select the first task interaction, excluding passive support impacts.

    A floor bounce is physics context for projectile tasks, not the rebound
    intervention around which a model clip should be centered.
    """

    passive_tokens = ("floor", "table", "ground", "wall", "static_surface")
    candidates: list[float] = []
    for event in events:
        object_b = str(event.get("object_b", "")).lower()
        if object_b and not any(token in object_b for token in passive_tokens):
            candidates.append(float(event["timestamp"]))
    return min(candidates) if candidates else None
