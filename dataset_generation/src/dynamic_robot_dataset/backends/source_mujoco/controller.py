"""Owned actuator-only controllers for real Panda/Robotiq review rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from ..actuator_only import (
    ActuatorTrajectoryGuard,
    ActuatorTrajectoryLimits,
    ControlObservation,
    apply_actuator_only_callback,
)
from ...common.embodiments import FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD
from .profiles import RIGID_REVIEW_PROFILE, minimum_jerk_fraction


def feasible_reach_duration_s(joint_delta: Sequence[float]) -> float:
    """Shortest replayable minimum-jerk duration for one arm joint move.

    Inverts the quintic peak forms (velocity ``1.875*dq/T``, acceleration
    ``5.7735*dq/T**2``, jerk ``60*dq/T**3``) against the published Franka
    per-joint limits and the calibrated trajectory-guard jerk bound, each at
    the profile safety fraction.  The binding duration is the maximum over
    joints, floored at the profile minimum so the reach always reads as a
    deliberate motion rather than a snap.
    """

    delta = np.abs(np.asarray(joint_delta, dtype=np.float64))
    if delta.shape != (7,) or not np.isfinite(delta).all():
        raise ValueError("reach feasibility requires seven finite joint deltas")
    safety = RIGID_REVIEW_PROFILE.reach_limit_safety_fraction
    velocity_limit = safety * np.asarray(
        RIGID_REVIEW_PROFILE.franka_joint_velocity_limit_rad_s, dtype=np.float64
    )
    acceleration_limit = safety * np.asarray(
        RIGID_REVIEW_PROFILE.franka_joint_acceleration_limit_rad_s2,
        dtype=np.float64,
    )
    guard_jerk_limit = safety * np.asarray(
        calibrated_trajectory_limits(FRANKA_HAND).maximum_jerk_per_s3[:7],
        dtype=np.float64,
    )
    t_velocity = 1.875 * delta / velocity_limit
    t_acceleration = np.sqrt(5.7735 * delta / acceleration_limit)
    t_jerk = np.cbrt(60.0 * delta / guard_jerk_limit)
    binding = float(np.max(np.maximum(np.maximum(t_velocity, t_acceleration), t_jerk)))
    return max(RIGID_REVIEW_PROFILE.minimum_reach_duration_s, binding)


@dataclass(frozen=True, slots=True)
class OwnedControllerPlan:
    """Precomputed controls; no mutable simulator handle can enter a callback."""

    embodiment: str
    initial_arm_command: tuple[float, ...]
    intercept_arm_command: tuple[float, ...]
    open_gripper_command: float
    closed_gripper_command: float
    arm_motion_start_s: float
    arm_motion_end_s: float
    closure_start_s: float
    closure_end_s: float
    transport_arm_command: tuple[float, ...] | None = None
    transport_start_s: float = 1.0
    transport_end_s: float = 1.6
    capture_arm_command: tuple[float, ...] | None = None
    capture_start_s: float = 1.0
    capture_end_s: float = 1.0

    def validate(self) -> None:
        if self.embodiment not in {FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD}:
            raise ValueError(f"unsupported owned controller embodiment {self.embodiment!r}")
        for name, values in (
            ("initial arm", self.initial_arm_command),
            ("intercept arm", self.intercept_arm_command),
        ):
            if len(values) != 7 or any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} command must contain seven finite values")
        if self.transport_arm_command is not None and (
            len(self.transport_arm_command) != 7
            or any(not math.isfinite(value) for value in self.transport_arm_command)
        ):
            raise ValueError("transport arm command must contain seven finite values")
        if self.capture_arm_command is not None and (
            len(self.capture_arm_command) != 7
            or any(not math.isfinite(value) for value in self.capture_arm_command)
        ):
            raise ValueError("capture arm command must contain seven finite values")
        if any(
            not math.isfinite(value)
            for value in (self.open_gripper_command, self.closed_gripper_command)
        ):
            raise ValueError("gripper commands must be finite")
        if not 0 <= self.arm_motion_start_s < self.arm_motion_end_s:
            raise ValueError("arm motion interval is invalid")
        if not 0 <= self.closure_start_s < self.closure_end_s:
            raise ValueError("closure interval is invalid")
        closure_duration = self.closure_end_s - self.closure_start_s
        allowed_closure_durations = {RIGID_REVIEW_PROFILE.closure_duration_s}
        if self.embodiment == ROBOTIQ_2F85_THICK_PAD:
            allowed_closure_durations.add(
                RIGID_REVIEW_PROFILE.robotiq_pickup_closure_duration_s
            )
        if not any(
            abs(closure_duration - allowed) <= 1e-9
            for allowed in allowed_closure_durations
        ):
            raise ValueError("closure duration differs from the calibrated bounded profile")
        travel = max(
            abs(left - right)
            for left, right in zip(self.initial_arm_command, self.intercept_arm_command)
        )
        if travel < 1e-6:
            raise ValueError(
                "owned reaching plan commands no arm travel; a stationary "
                "pre-positioned interception is not an acceptable catch"
            )
        if not self.arm_motion_start_s < self.closure_start_s:
            raise ValueError("the reach must begin before the bounded closure")
        if self.transport_arm_command is not None:
            if not self.transport_start_s < self.transport_end_s:
                raise ValueError("transport interval is invalid")
            if self.transport_start_s < self.arm_motion_end_s:
                raise ValueError("transport cannot begin before reach arrival")
        if self.capture_arm_command is not None:
            if not self.arm_motion_end_s <= self.capture_start_s < self.capture_end_s:
                raise ValueError("capture phase must follow reach arrival")
            if self.transport_arm_command is None:
                raise ValueError("capture phase requires a final transport command")
            if self.transport_start_s < self.capture_end_s:
                raise ValueError("final transport cannot begin before capture completes")

    def command_at(self, timestamp_s: float) -> np.ndarray:
        """Return the exact eight planned commands at one timestamp."""

        timestamp = float(timestamp_s)
        initial = np.asarray(self.initial_arm_command, dtype=np.float64)
        intercept = np.asarray(self.intercept_arm_command, dtype=np.float64)
        arm = _interpolate(
            initial,
            intercept,
            timestamp,
            self.arm_motion_start_s,
            self.arm_motion_end_s,
        )
        transport_origin = intercept
        if self.capture_arm_command is not None and timestamp >= self.capture_start_s:
            capture = np.asarray(self.capture_arm_command, dtype=np.float64)
            arm = _interpolate(
                intercept,
                capture,
                timestamp,
                self.capture_start_s,
                self.capture_end_s,
            )
            transport_origin = capture
        if (
            self.transport_arm_command is not None
            and timestamp >= self.transport_start_s
        ):
            arm = _interpolate(
                transport_origin,
                np.asarray(self.transport_arm_command, dtype=np.float64),
                timestamp,
                self.transport_start_s,
                self.transport_end_s,
            )
        open_command = np.asarray((self.open_gripper_command,), dtype=np.float64)
        closed_command = np.asarray((self.closed_gripper_command,), dtype=np.float64)
        gripper = _interpolate(
            open_command,
            closed_command,
            timestamp,
            self.closure_start_s,
            self.closure_end_s,
        )
        return np.r_[arm, gripper]


def _interpolate(
    left: np.ndarray,
    right: np.ndarray,
    timestamp_s: float,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    if timestamp_s <= start_s:
        return left.copy()
    if timestamp_s >= end_s:
        return right.copy()
    alpha = minimum_jerk_fraction((timestamp_s - start_s) / (end_s - start_s))
    return left + alpha * (right - left)


def calibrated_trajectory_limits(embodiment: str) -> ActuatorTrajectoryLimits:
    """Limits encompass the calibrated quintic path without permitting a step."""

    if embodiment == FRANKA_HAND:
        # The guard estimates derivatives from 60 Hz zero-order-hold command
        # samples.  These bounds cover the finite-difference extrema of the
        # calibrated 100 ms quintic closure (including its boundary sample),
        # while the velocity/acceleration bounds still reject a step closure.
        gripper = (4000.0, 140000.0, 5_000_000.0)
    elif embodiment == ROBOTIQ_2F85_THICK_PAD:
        gripper = (2500.0, 90000.0, 4_000_000.0)
    else:
        raise ValueError(f"unsupported trajectory-limit embodiment {embodiment!r}")
    return ActuatorTrajectoryLimits(
        maximum_velocity_per_s=(3.5,) * 7 + (gripper[0],),
        maximum_acceleration_per_s2=(80.0,) * 7 + (gripper[1],),
        maximum_jerk_per_s3=(2400.0,) * 7 + (gripper[2],),
    )


class OwnedActuatorController:
    """Pure observation-to-command callback plus enforced application wall."""

    def __init__(self, plan: OwnedControllerPlan) -> None:
        plan.validate()
        self.plan = plan
        self.trajectory_guard = ActuatorTrajectoryGuard(
            plan.embodiment,
            calibrated_trajectory_limits(plan.embodiment),
        )

    def command(self, observation: ControlObservation) -> np.ndarray:
        """Return the eight commands; this method has no simulator reference."""

        observation.validate()
        return self.plan.command_at(float(observation.timestamp_s))

    def apply(
        self,
        observation: ControlObservation,
        *,
        model: Any,
        data: Any,
        actuator_ids: Sequence[int],
        forbidden_equality_ids: Sequence[int] = (),
    ) -> np.ndarray:
        """Apply through the shared state/model mutation boundary."""

        return apply_actuator_only_callback(
            self.command,
            observation,
            embodiment=self.plan.embodiment,
            model=model,
            data=data,
            actuator_ids=actuator_ids,
            trajectory_guard=self.trajectory_guard,
            forbidden_equality_ids=forbidden_equality_ids,
        )


__all__ = [
    "OwnedActuatorController",
    "OwnedControllerPlan",
    "calibrated_trajectory_limits",
    "feasible_reach_duration_s",
]
