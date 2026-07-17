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
        if any(
            not math.isfinite(value)
            for value in (self.open_gripper_command, self.closed_gripper_command)
        ):
            raise ValueError("gripper commands must be finite")
        if not 0 <= self.arm_motion_start_s < self.arm_motion_end_s:
            raise ValueError("arm motion interval is invalid")
        if not 0 <= self.closure_start_s < self.closure_end_s:
            raise ValueError("closure interval is invalid")
        if abs(
            (self.closure_end_s - self.closure_start_s)
            - RIGID_REVIEW_PROFILE.closure_duration_s
        ) > 1e-9:
            raise ValueError("closure duration differs from the calibrated bounded profile")
        if self.transport_arm_command is not None and not (
            self.transport_start_s < self.transport_end_s
        ):
            raise ValueError("transport interval is invalid")


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
        timestamp = float(observation.timestamp_s)
        initial = np.asarray(self.plan.initial_arm_command, dtype=np.float64)
        intercept = np.asarray(self.plan.intercept_arm_command, dtype=np.float64)
        arm = _interpolate(
            initial,
            intercept,
            timestamp,
            self.plan.arm_motion_start_s,
            self.plan.arm_motion_end_s,
        )
        if self.plan.transport_arm_command is not None:
            arm = _interpolate(
                intercept,
                np.asarray(self.plan.transport_arm_command, dtype=np.float64),
                timestamp,
                self.plan.transport_start_s,
                self.plan.transport_end_s,
            )
        open_command = np.asarray((self.plan.open_gripper_command,), dtype=np.float64)
        closed_command = np.asarray((self.plan.closed_gripper_command,), dtype=np.float64)
        gripper = _interpolate(
            open_command,
            closed_command,
            timestamp,
            self.plan.closure_start_s,
            self.plan.closure_end_s,
        )
        return np.r_[arm, gripper]

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
]
