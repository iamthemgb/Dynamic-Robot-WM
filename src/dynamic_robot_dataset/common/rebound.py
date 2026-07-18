"""Shared, persisted-artifact rebound acceptance measurements.

The rebound leaf is semantic: merely touching a surface is not a rebound.
This module keeps the online source backend and the independent persisted-row
evaluator on one fixed threshold/evidence contract without importing simulator
code into the common evaluator layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


REBOUND_ACCEPTANCE_SCHEMA = "dynamic-robot-rebound-acceptance/v1"


@dataclass(frozen=True, slots=True)
class ReboundAcceptanceThresholds:
    """Minimum visible separation plus the existing no-energy-gain ceiling."""

    minimum_effective_restitution: float = 0.15
    maximum_effective_restitution: float = 1.05
    minimum_outgoing_normal_speed_m_s: float = 0.15
    minimum_normal_separation_m: float = 0.01
    minimum_normal_separation_radius_fraction: float = 0.5
    minimum_separation_duration_s: float = 1.0 / 30.0
    schema_version: str = REBOUND_ACCEPTANCE_SCHEMA

    def validate(self) -> None:
        values = (
            self.minimum_effective_restitution,
            self.maximum_effective_restitution,
            self.minimum_outgoing_normal_speed_m_s,
            self.minimum_normal_separation_m,
            self.minimum_normal_separation_radius_fraction,
            self.minimum_separation_duration_s,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("rebound acceptance thresholds must be finite")
        if self.schema_version != REBOUND_ACCEPTANCE_SCHEMA:
            raise ValueError("unsupported rebound acceptance schema")
        if not (
            0.0
            < self.minimum_effective_restitution
            <= self.maximum_effective_restitution
            <= 1.05
        ):
            raise ValueError("rebound restitution bounds are invalid")
        if any(value <= 0.0 for value in values[2:]):
            raise ValueError("rebound separation thresholds must be positive")

    def minimum_required_separation_m(self, object_radius_m: float) -> float:
        radius = float(object_radius_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("rebound object radius must be positive and finite")
        return max(
            self.minimum_normal_separation_m,
            self.minimum_normal_separation_radius_fraction * radius,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReboundAcceptanceThresholds":
        try:
            result = cls(
                minimum_effective_restitution=float(
                    value["minimum_effective_restitution"]
                ),
                maximum_effective_restitution=float(
                    value["maximum_effective_restitution"]
                ),
                minimum_outgoing_normal_speed_m_s=float(
                    value["minimum_outgoing_normal_speed_m_s"]
                ),
                minimum_normal_separation_m=float(
                    value["minimum_normal_separation_m"]
                ),
                minimum_normal_separation_radius_fraction=float(
                    value["minimum_normal_separation_radius_fraction"]
                ),
                minimum_separation_duration_s=float(
                    value["minimum_separation_duration_s"]
                ),
                schema_version=str(value["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("rebound acceptance mapping is incomplete") from error
        result.validate()
        return result


DEFAULT_REBOUND_ACCEPTANCE = ReboundAcceptanceThresholds()
DEFAULT_REBOUND_ACCEPTANCE.validate()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector(value: Any, size: int) -> tuple[float, ...] | None:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != size or any(not math.isfinite(item) for item in result):
        return None
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _state_contact_count(row: Mapping[str, Any]) -> int | None:
    raw = row.get("contact.count", row.get("contact_count"))
    if isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def measure_rebound_kinematics(
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    object_radius_m: float,
    thresholds: ReboundAcceptanceThresholds = DEFAULT_REBOUND_ACCEPTANCE,
) -> dict[str, Any]:
    """Measure the first task-surface rebound from unlabelled persisted rows."""

    thresholds.validate()
    required_separation = thresholds.minimum_required_separation_m(object_radius_m)
    base: dict[str, Any] = {
        "schema_version": REBOUND_ACCEPTANCE_SCHEMA,
        "thresholds": thresholds.to_dict(),
        "required_minimum_normal_separation_m": required_separation,
        "applicable": False,
        "separated_pre_post_contact_samples": False,
        "measured_contact_normal": False,
        "effective_restitution": None,
        "incoming_normal_velocity_m_s": None,
        "outgoing_normal_velocity_m_s": None,
        "maximum_normal_separation_m": 0.0,
        "separation_duration_s": 0.0,
        "effective_restitution_within_limits": False,
        "outgoing_normal_speed_sufficient": False,
        "normal_separation_sufficient": False,
        "separation_duration_sufficient": False,
        "no_unexplained_contact_energy_gain": False,
        "rebound_acceptance_pass": False,
    }
    surface = [
        row for row in event_rows if row.get("contact_category") == "task_surface"
    ]
    event_times = [_number(row.get("timestamp")) for row in surface]
    finite_event_times = [value for value in event_times if value is not None]
    if not finite_event_times:
        return base
    event_time = min(finite_event_times)
    base["applicable"] = True
    event_normals = [
        normal
        for row in surface
        if (_number(row.get("timestamp")) is not None)
        and abs(float(row["timestamp"]) - event_time) <= 1e-9
        and (normal := _vector(row.get("normal_world"), 3)) is not None
    ]
    if not event_normals:
        return base
    mean_normal = tuple(
        sum(row[axis] for row in event_normals) / len(event_normals)
        for axis in range(3)
    )
    normal_norm = math.sqrt(_dot(mean_normal, mean_normal))
    if normal_norm <= 1e-12:
        return base
    normal = tuple(value / normal_norm for value in mean_normal)
    base["measured_contact_normal"] = True
    base["event_time_s"] = event_time
    base["normal_world"] = list(normal)

    states = [
        (timestamp, row)
        for row in state_rows
        if (timestamp := _number(row.get("timestamp"))) is not None
        and _vector(row.get("object.position"), 3) is not None
        and _vector(row.get("object.linear_velocity"), 3) is not None
    ]
    states.sort(key=lambda item: item[0])
    incoming_candidates = [
        item
        for item in states
        if item[0] < event_time and _state_contact_count(item[1]) == 0
    ]
    outgoing_candidates = [
        item
        for item in states
        if item[0] > event_time and _state_contact_count(item[1]) == 0
    ]
    if not incoming_candidates or not outgoing_candidates:
        final_velocity = (
            _vector(states[-1][1].get("object.linear_velocity"), 3)
            if states
            else None
        )
        initial_supported_contact = bool(
            states
            and abs(states[0][0] - event_time) <= 1e-9
            and _state_contact_count(states[0][1]) not in {None, 0}
        )
        settled = bool(
            (incoming_candidates or initial_supported_contact)
            and states
            and _state_contact_count(states[-1][1]) not in {None, 0}
            and final_velocity is not None
            and abs(_dot(final_velocity, normal)) <= 0.1
        )
        if settled:
            base["effective_restitution"] = 0.0
            base["no_unexplained_contact_energy_gain"] = True
        return base

    incoming_time, incoming = incoming_candidates[-1]
    outgoing_time, outgoing = outgoing_candidates[0]
    incoming_velocity = _vector(incoming.get("object.linear_velocity"), 3)
    outgoing_velocity = _vector(outgoing.get("object.linear_velocity"), 3)
    assert incoming_velocity is not None and outgoing_velocity is not None
    vn_in = _dot(incoming_velocity, normal)
    vn_out = _dot(outgoing_velocity, normal)
    effective = vn_out / -vn_in if vn_in < -1e-4 and vn_out > 0.0 else math.inf

    outgoing_index = next(
        index for index, item in enumerate(states) if item[1] is outgoing
    )
    free_interval: list[tuple[float, Mapping[str, Any]]] = []
    for timestamp, row in states[outgoing_index:]:
        if _state_contact_count(row) != 0:
            break
        free_interval.append((timestamp, row))
    event_state = min(states, key=lambda item: abs(item[0] - event_time))[1]
    event_position = _vector(event_state.get("object.position"), 3)
    assert event_position is not None
    separations = []
    for _, row in free_interval:
        position = _vector(row.get("object.position"), 3)
        assert position is not None
        separations.append(
            _dot(tuple(value - origin for value, origin in zip(position, event_position)), normal)
        )
    maximum_separation = max([0.0, *separations])
    separation_duration = (
        max(0.0, free_interval[-1][0] - event_time) if free_interval else 0.0
    )
    restitution_within_limits = bool(
        math.isfinite(effective)
        and thresholds.minimum_effective_restitution
        <= effective
        <= thresholds.maximum_effective_restitution
    )
    speed_sufficient = bool(
        math.isfinite(vn_out)
        and vn_out >= thresholds.minimum_outgoing_normal_speed_m_s
    )
    separation_sufficient = maximum_separation >= required_separation
    duration_sufficient = (
        separation_duration + 1e-12 >= thresholds.minimum_separation_duration_s
    )
    no_energy_gain = bool(
        math.isfinite(effective)
        and effective <= thresholds.maximum_effective_restitution
    )
    accepted = bool(
        restitution_within_limits
        and speed_sufficient
        and separation_sufficient
        and duration_sufficient
        and no_energy_gain
    )
    base.update(
        {
            "separated_pre_post_contact_samples": True,
            "event_position_m": list(event_position),
            "incoming_sample_time_s": incoming_time,
            "outgoing_sample_time_s": outgoing_time,
            "incoming_normal_velocity_m_s": vn_in,
            "outgoing_normal_velocity_m_s": vn_out,
            "effective_restitution": effective,
            "maximum_normal_separation_m": maximum_separation,
            "separation_duration_s": separation_duration,
            "effective_restitution_within_limits": restitution_within_limits,
            "outgoing_normal_speed_sufficient": speed_sufficient,
            "normal_separation_sufficient": separation_sufficient,
            "separation_duration_sufficient": duration_sufficient,
            "no_unexplained_contact_energy_gain": no_energy_gain,
            "rebound_acceptance_pass": accepted,
        }
    )
    return base


__all__ = [
    "DEFAULT_REBOUND_ACCEPTANCE",
    "REBOUND_ACCEPTANCE_SCHEMA",
    "ReboundAcceptanceThresholds",
    "measure_rebound_kinematics",
]
