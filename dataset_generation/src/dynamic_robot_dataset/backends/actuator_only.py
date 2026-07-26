"""Fail-closed actuator-only controller boundary.

Production controllers receive an immutable observation and return the eight
commands that were actually applied to the simulator.  They never receive a
mutable simulator data object.  The mutation snapshot is intentionally kept at
the callback boundary as defence in depth: it also catches callbacks which
capture ``model`` or ``data`` through a closure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..common.embodiments import FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD


ACTION_SEMANTICS = "actual_actuator_command/v1"
ACTION_FIELD = "action.actuator_command"


class ActuatorOnlyViolation(RuntimeError):
    """A controller crossed the actuator-only mutation boundary."""


@dataclass(frozen=True, slots=True)
class ActuatorActionSpec:
    """Canonical applied-action layout for one real gripper embodiment."""

    embodiment: str
    actuator_names: tuple[str, ...]
    semantics: str = ACTION_SEMANTICS
    field_name: str = ACTION_FIELD

    @property
    def width(self) -> int:
        return len(self.actuator_names)

    def validate(self) -> None:
        if self.embodiment not in {FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD}:
            raise ValueError(f"unsupported actuator action embodiment {self.embodiment!r}")
        if self.semantics != ACTION_SEMANTICS or self.field_name != ACTION_FIELD:
            raise ValueError("canonical actions must use the applied-actuator v1 contract")
        if len(self.actuator_names) != 8 or len(set(self.actuator_names)) != 8:
            raise ValueError("real-gripper action layouts require eight unique actuators")


ACTION_SPECS: Mapping[str, ActuatorActionSpec] = {
    FRANKA_HAND: ActuatorActionSpec(
        embodiment=FRANKA_HAND,
        actuator_names=(
            "panda_joint_1_ctrl",
            "panda_joint_2_ctrl",
            "panda_joint_3_ctrl",
            "panda_joint_4_ctrl",
            "panda_joint_5_ctrl",
            "panda_joint_6_ctrl",
            "panda_joint_7_ctrl",
            "panda_finger_ctrl",
        ),
    ),
    ROBOTIQ_2F85_THICK_PAD: ActuatorActionSpec(
        embodiment=ROBOTIQ_2F85_THICK_PAD,
        actuator_names=(
            "panda_joint_1_ctrl",
            "panda_joint_2_ctrl",
            "panda_joint_3_ctrl",
            "panda_joint_4_ctrl",
            "panda_joint_5_ctrl",
            "panda_joint_6_ctrl",
            "panda_joint_7_ctrl",
            "robotiq_tendon_ctrl",
        ),
    ),
}


def action_spec(embodiment: str) -> ActuatorActionSpec:
    """Return the canonical action layout or fail closed."""

    try:
        value = ACTION_SPECS[str(embodiment)]
    except KeyError as error:
        raise ValueError(f"no canonical actuator layout for {embodiment!r}") from error
    value.validate()
    return value


def validate_actuator_command(
    command: Sequence[float],
    *,
    embodiment: str,
) -> np.ndarray:
    """Return a finite, copied eight-actuator command."""

    spec = action_spec(embodiment)
    values = np.asarray(command, dtype=np.float64)
    if values.shape != (spec.width,):
        raise ValueError(
            f"{embodiment} action must have shape ({spec.width},), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("actuator commands must be finite")
    return values.copy()


def validate_action_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    embodiment: str,
    action_semantics: str,
) -> list[str]:
    """Validate that persisted actions are applied controls, not desired poses."""

    problems: list[str] = []
    if action_semantics != ACTION_SEMANTICS:
        problems.append(
            f"action semantics must be {ACTION_SEMANTICS}, got {action_semantics!r}"
        )
    try:
        spec = action_spec(embodiment)
    except ValueError as error:
        return [str(error)]
    if not rows:
        return [*problems, "no persisted action rows"]
    forbidden_action_fields = sorted(
        {
            name
            for row in rows
            for name in row
            if name.startswith("action.")
            and name != spec.field_name
            and any(token in name.lower() for token in ("target", "desired", "joint_position"))
        }
    )
    if forbidden_action_fields:
        problems.append(
            "desired kinematic targets cannot be canonical actions: "
            + ", ".join(forbidden_action_fields)
        )
    for index, row in enumerate(rows):
        if spec.field_name not in row:
            problems.append(f"row {index} lacks {spec.field_name}")
            continue
        try:
            validate_actuator_command(row[spec.field_name], embodiment=embodiment)
        except (TypeError, ValueError) as error:
            problems.append(f"row {index} has invalid applied action: {error}")
    return problems


@dataclass(frozen=True, slots=True)
class ControlObservation:
    """Immutable, JSON-serialisable input exposed to a controller."""

    timestamp_s: float
    robot_joint_position: tuple[float, ...]
    robot_joint_velocity: tuple[float, ...]
    object_position_m: tuple[float, float, float]
    object_linear_velocity_m_s: tuple[float, float, float]
    extras: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        numeric = (
            self.timestamp_s,
            *self.robot_joint_position,
            *self.robot_joint_velocity,
            *self.object_position_m,
            *self.object_linear_velocity_m_s,
        )
        if self.timestamp_s < 0 or any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("control observations must be finite and have non-negative time")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActuatorTrajectoryLimits:
    """Model/calibration-derived limits used without silent command clipping."""

    maximum_velocity_per_s: tuple[float, ...]
    maximum_acceleration_per_s2: tuple[float, ...]
    maximum_jerk_per_s3: tuple[float, ...]

    def validate(self, *, width: int = 8) -> None:
        for label, values in (
            ("velocity", self.maximum_velocity_per_s),
            ("acceleration", self.maximum_acceleration_per_s2),
            ("jerk", self.maximum_jerk_per_s3),
        ):
            if len(values) != width or any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in values
            ):
                raise ValueError(
                    f"actuator {label} limits must contain {width} finite positive values"
                )


@dataclass(slots=True)
class ActuatorTrajectoryGuard:
    """Reject instantaneous closure and non-jerk-limited command sequences."""

    embodiment: str
    limits: ActuatorTrajectoryLimits
    _previous_time_s: float | None = None
    _previous_command: np.ndarray | None = None
    _previous_velocity: np.ndarray | None = None
    _previous_acceleration: np.ndarray | None = None

    def __post_init__(self) -> None:
        spec = action_spec(self.embodiment)
        self.limits.validate(width=spec.width)

    def validate_next(self, timestamp_s: float, command: Sequence[float]) -> np.ndarray:
        values = validate_actuator_command(command, embodiment=self.embodiment)
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("actuator trajectory timestamps must be finite and non-negative")
        if self._previous_time_s is None:
            self._previous_time_s = float(timestamp_s)
            self._previous_command = values.copy()
            return values
        dt = float(timestamp_s) - self._previous_time_s
        if dt <= 0:
            raise ActuatorOnlyViolation(
                "actuator trajectory timestamps must be strictly increasing"
            )
        assert self._previous_command is not None
        velocity = (values - self._previous_command) / dt
        maximum_velocity = np.asarray(self.limits.maximum_velocity_per_s)
        if np.any(np.abs(velocity) > maximum_velocity + 1e-12):
            raise ActuatorOnlyViolation(
                "actuator trajectory exceeds a calibrated velocity/closure limit"
            )
        acceleration: np.ndarray | None = None
        if self._previous_velocity is not None:
            acceleration = (velocity - self._previous_velocity) / dt
            maximum_acceleration = np.asarray(
                self.limits.maximum_acceleration_per_s2
            )
            if np.any(np.abs(acceleration) > maximum_acceleration + 1e-12):
                raise ActuatorOnlyViolation(
                    "actuator trajectory exceeds a calibrated acceleration limit"
                )
        if acceleration is not None and self._previous_acceleration is not None:
            jerk = (acceleration - self._previous_acceleration) / dt
            maximum_jerk = np.asarray(self.limits.maximum_jerk_per_s3)
            if np.any(np.abs(jerk) > maximum_jerk + 1e-12):
                raise ActuatorOnlyViolation(
                    "actuator trajectory is not jerk-limited"
                )
        self._previous_time_s = float(timestamp_s)
        self._previous_command = values.copy()
        self._previous_velocity = velocity
        if acceleration is not None:
            self._previous_acceleration = acceleration
        return values


_DATA_MUTATION_FIELDS = (
    # The callback is not allowed to pre-write even an actuator.  The owned
    # wrapper writes the validated command only after this snapshot comparison.
    "ctrl",
    "qpos",
    "qvel",
    "mocap_pos",
    "mocap_quat",
    "xfrc_applied",
    "qfrc_applied",
    "eq_active",
)

_MODEL_MUTATION_FIELDS = (
    "body_mass",
    "body_inertia",
    "dof_damping",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_contype",
    "geom_conaffinity",
    "actuator_ctrllimited",
    "actuator_ctrlrange",
    "actuator_forcelimited",
    "actuator_forcerange",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_dynprm",
    "eq_type",
    "eq_obj1id",
    "eq_obj2id",
    "eq_data",
)


def _copied_array(owner: Any, name: str) -> np.ndarray | None:
    value = getattr(owner, name, None)
    if value is None:
        return None
    return np.asarray(value).copy()


@dataclass(frozen=True, slots=True)
class MutationSnapshot:
    """State which a controller callback is forbidden to change."""

    data_fields: Mapping[str, np.ndarray]
    model_fields: Mapping[str, np.ndarray]
    option_fields: Mapping[str, np.ndarray]

    @classmethod
    def capture(cls, model: Any, data: Any) -> "MutationSnapshot":
        data_fields = {
            name: value
            for name in _DATA_MUTATION_FIELDS
            if (value := _copied_array(data, name)) is not None
        }
        model_fields = {
            name: value
            for name in _MODEL_MUTATION_FIELDS
            if (value := _copied_array(model, name)) is not None
        }
        option = getattr(model, "opt", None)
        option_fields: dict[str, np.ndarray] = {}
        if option is not None:
            for name in ("gravity", "timestep", "integrator", "solver", "iterations"):
                value = getattr(option, name, None)
                if value is not None:
                    option_fields[name] = np.asarray(value).copy()
        return cls(data_fields, model_fields, option_fields)

    def changed_fields(self, model: Any, data: Any) -> tuple[str, ...]:
        changed: list[str] = []
        for name, before in self.data_fields.items():
            after = np.asarray(getattr(data, name))
            if after.shape != before.shape or not np.array_equal(after, before, equal_nan=True):
                changed.append(f"data.{name}")
        for name, before in self.model_fields.items():
            after = np.asarray(getattr(model, name))
            if after.shape != before.shape or not np.array_equal(after, before, equal_nan=True):
                changed.append(f"model.{name}")
        option = getattr(model, "opt", None)
        for name, before in self.option_fields.items():
            after = np.asarray(getattr(option, name))
            if after.shape != before.shape or not np.array_equal(after, before, equal_nan=True):
                changed.append(f"model.opt.{name}")
        return tuple(changed)


def _assert_zero_applied_forces(data: Any) -> None:
    active: list[str] = []
    for name in ("xfrc_applied", "qfrc_applied"):
        value = getattr(data, name, None)
        if value is not None and np.any(np.asarray(value) != 0):
            active.append(f"data.{name}")
    if active:
        raise ActuatorOnlyViolation(
            "applied forces are forbidden in production rollouts: " + ", ".join(active)
        )


def _assert_forbidden_equalities_inactive(
    data: Any,
    forbidden_equality_ids: Sequence[int],
) -> None:
    if not forbidden_equality_ids:
        return
    active = getattr(data, "eq_active", None)
    if active is None:
        raise ActuatorOnlyViolation(
            "forbidden object-linked equalities were declared but eq_active is unavailable"
        )
    values = np.asarray(active)
    bad = [index for index in forbidden_equality_ids if index < 0 or index >= len(values) or bool(values[index])]
    if bad:
        raise ActuatorOnlyViolation(
            f"object-linked equality/weld/latch constraints are active or invalid: {bad}"
        )


def _assert_command_within_model_ranges(
    model: Any,
    actuator_ids: Sequence[int],
    command: np.ndarray,
) -> None:
    limited = getattr(model, "actuator_ctrllimited", None)
    ranges = getattr(model, "actuator_ctrlrange", None)
    if limited is None or ranges is None:
        raise ActuatorOnlyViolation(
            "compiled model does not expose actuator control-limit evidence"
        )
    limited_values = np.asarray(limited)
    range_values = np.asarray(ranges)
    violations: list[str] = []
    for action_index, (actuator_id, value) in enumerate(zip(actuator_ids, command)):
        index = int(actuator_id)
        if index >= len(limited_values) or index >= len(range_values):
            violations.append(f"action[{action_index}] has no model range")
            continue
        if not bool(limited_values[index]):
            violations.append(f"action[{action_index}] actuator is not control-limited")
            continue
        low, high = (float(item) for item in range_values[index])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            violations.append(f"action[{action_index}] has an invalid model range")
        elif not low <= float(value) <= high:
            violations.append(
                f"action[{action_index}]={float(value):g} outside [{low:g}, {high:g}]"
            )
    if violations:
        raise ActuatorOnlyViolation(
            "actuator command violates compiled model limits: " + "; ".join(violations)
        )


ControllerCallback = Callable[[ControlObservation], Sequence[float]]


def apply_actuator_only_callback(
    callback: ControllerCallback,
    observation: ControlObservation,
    *,
    embodiment: str,
    model: Any,
    data: Any,
    actuator_ids: Sequence[int],
    trajectory_guard: ActuatorTrajectoryGuard,
    forbidden_equality_ids: Sequence[int] = (),
) -> np.ndarray:
    """Run a controller and apply its command while enforcing the mutation wall.

    The returned vector is the exact clipped-free command written to ``ctrl``;
    callers persist this vector as ``action.actuator_command``.  Hardware/model
    range handling belongs in the controller profile and must reject rather
    than silently clip so saved actions remain identical to applied actions.
    """

    observation.validate()
    spec = action_spec(embodiment)
    if trajectory_guard.embodiment != embodiment:
        raise ValueError("trajectory guard embodiment differs from the controller embodiment")
    if len(actuator_ids) != spec.width or len(set(int(value) for value in actuator_ids)) != spec.width:
        raise ValueError("actuator_ids must map the eight canonical actions one-to-one")
    ctrl = np.asarray(getattr(data, "ctrl"))
    if any(int(index) < 0 or int(index) >= len(ctrl) for index in actuator_ids):
        raise ValueError("actuator_ids contain an out-of-range simulator actuator")
    _assert_zero_applied_forces(data)
    _assert_forbidden_equalities_inactive(data, forbidden_equality_ids)
    snapshot = MutationSnapshot.capture(model, data)
    command = validate_actuator_command(callback(observation), embodiment=embodiment)
    changed = snapshot.changed_fields(model, data)
    if changed:
        raise ActuatorOnlyViolation(
            "controller changed forbidden simulator state: " + ", ".join(changed)
        )
    _assert_zero_applied_forces(data)
    _assert_forbidden_equalities_inactive(data, forbidden_equality_ids)
    _assert_command_within_model_ranges(model, actuator_ids, command)
    trajectory_guard.validate_next(observation.timestamp_s, command)
    for actuator_id, value in zip(actuator_ids, command):
        ctrl[int(actuator_id)] = float(value)
    return command


__all__ = [
    "ACTION_FIELD",
    "ACTION_SEMANTICS",
    "ACTION_SPECS",
    "ActuatorActionSpec",
    "ActuatorOnlyViolation",
    "ActuatorTrajectoryGuard",
    "ActuatorTrajectoryLimits",
    "ControlObservation",
    "MutationSnapshot",
    "action_spec",
    "apply_actuator_only_callback",
    "validate_action_rows",
    "validate_actuator_command",
]
