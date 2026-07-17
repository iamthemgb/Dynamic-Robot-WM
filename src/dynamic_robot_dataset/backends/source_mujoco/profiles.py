"""Calibrated, review-only profiles for the owned ``source_mujoco`` backend.

The constants in this module are evidence, not release claims.  They preserve
the July 2026 timestep/contact experiments which motivated this backend while
the fixed six-case review suite is still outstanding.  Runtime code validates
the constants before compiling a model and persists the complete profile in
every result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


SOURCE_MUJOCO_PROFILE_SCHEMA = "dynamic-robot-source-mujoco-profile/v1"
SOURCE_MUJOCO_PROFILE_VERSION = "source-mujoco-rigid-review-2026-07-v4"


@dataclass(frozen=True, slots=True)
class RigidReviewProfile:
    """One fail-closed rigid solver/controller calibration profile."""

    profile_id: str = SOURCE_MUJOCO_PROFILE_VERSION
    simulation_hz: int = 600
    comparison_simulation_hz: int = 1200
    control_hz: int = 60
    video_hz: int = 30
    width: int = 832
    height: int = 480
    wall_solref: tuple[float, float] = (0.012, 0.7)
    wall_effective_restitution: float = 0.20
    robotiq_pad_condim: int = 3
    robotiq_pad_friction: tuple[float, float, float] = (0.9, 0.005, 0.0001)
    robotiq_pad_solref: tuple[float, float] = (0.012, 0.7)
    # v2: the physical thick-pad midpoint replaces the historical base site.
    # 115 is the smallest tested bounded command whose fixed-review contact
    # event agrees at 600 and 1200 Hz within the 1 cm / one-frame gate.
    robotiq_tendon_target: float = 115.0
    robotiq_tendon_force_range_n: tuple[float, float] = (-0.16, 0.16)
    closure_start_before_ballistic_s: float = 0.16
    closure_duration_s: float = 0.10
    maximum_gripper_penetration_m: float = 0.002
    maximum_task_surface_penetration_m: float = 0.003
    maximum_effective_restitution: float = 1.05
    maximum_free_flight_energy_drift_fraction: float = 0.05
    panda_ball_radius_range_m: tuple[float, float] = (0.024, 0.0245)
    robotiq_calibration_seed: int = 7401
    robotiq_calibration_penetration_m: float = 0.001547687056695994
    robotiq_first_bilateral_contact_600_s: float = 0.43666666666753823
    robotiq_first_bilateral_contact_1200_s: float = 0.4366666666664955
    robotiq_final_position_delta_m: float = 0.004186892649805686
    wall_600_1200_outcome_match: bool = True
    # Fixed review classes for which the 600 Hz candidate failed strict QC.
    # They execute at the calibrated 1200 Hz reference rather than receiving
    # a relaxed penetration or event-evidence threshold.
    reference_rate_required_case_classes: tuple[str, ...] = (
        "P0c/lower_initial_speed",
        "P0c/higher_initial_speed",
        "F1a/franka_hand/deterministic_negative_controller_timing",
        "F1b/franka_hand/deterministic_negative_controller_timing",
    )
    release_state: str = "blocked"
    schema_version: str = SOURCE_MUJOCO_PROFILE_SCHEMA

    def validate(self) -> None:
        if self.schema_version != SOURCE_MUJOCO_PROFILE_SCHEMA:
            raise ValueError("unsupported source_mujoco profile schema")
        if self.release_state != "blocked":
            raise ValueError("the review profile must not activate a release state")
        if (self.simulation_hz, self.comparison_simulation_hz) != (600, 1200):
            raise ValueError("the calibrated rigid profile is fixed at 600/1200 Hz")
        if self.simulation_hz % self.control_hz or self.simulation_hz % self.video_hz:
            raise ValueError("simulation rate must divide control and video schedules exactly")
        if (self.width, self.height, self.video_hz) != (832, 480, 30):
            raise ValueError("canonical source media must remain 832x480 at 30 Hz")
        if self.wall_solref != (0.012, 0.7):
            raise ValueError("wall restitution calibration changed without a profile version")
        if self.robotiq_pad_condim != 3:
            raise ValueError("Robotiq thick pads must use condim=3")
        if self.robotiq_pad_friction != (0.9, 0.005, 0.0001):
            raise ValueError("Robotiq thick-pad friction changed without a profile version")
        if self.robotiq_tendon_force_range_n != (-0.16, 0.16):
            raise ValueError("Robotiq calibrated tendon force range must remain +/-0.16 N")
        if not 0.0 < self.closure_duration_s <= self.closure_start_before_ballistic_s:
            raise ValueError("closure timing is not a bounded pre-intercept trajectory")
        if self.robotiq_calibration_penetration_m > self.maximum_gripper_penetration_m:
            raise ValueError("persisted Robotiq calibration exceeds the 2 mm admission limit")
        if not self.wall_600_1200_outcome_match:
            raise ValueError("600 Hz cannot be admitted when the 1200 Hz outcome differs")
        if self.reference_rate_required_case_classes != (
            "P0c/lower_initial_speed",
            "P0c/higher_initial_speed",
            "F1a/franka_hand/deterministic_negative_controller_timing",
            "F1b/franka_hand/deterministic_negative_controller_timing",
        ):
            raise ValueError("reference-rate exception classes changed without calibration")
        if self.wall_effective_restitution > self.maximum_effective_restitution:
            raise ValueError("wall calibration injects contact energy")
        numeric = (
            *self.wall_solref,
            self.wall_effective_restitution,
            *self.robotiq_pad_friction,
            *self.robotiq_pad_solref,
            self.robotiq_tendon_target,
            *self.robotiq_tendon_force_range_n,
            self.closure_start_before_ballistic_s,
            self.closure_duration_s,
            self.maximum_gripper_penetration_m,
            self.maximum_task_surface_penetration_m,
            self.maximum_effective_restitution,
            self.maximum_free_flight_energy_drift_fraction,
            *self.panda_ball_radius_range_m,
            self.robotiq_calibration_penetration_m,
            self.robotiq_first_bilateral_contact_600_s,
            self.robotiq_first_bilateral_contact_1200_s,
            self.robotiq_final_position_delta_m,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("calibration profile contains a non-finite value")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


RIGID_REVIEW_PROFILE = RigidReviewProfile()
RIGID_REVIEW_PROFILE.validate()


def minimum_jerk_fraction(value: float) -> float:
    """Quintic minimum-jerk interpolation in ``[0, 1]``."""

    if not math.isfinite(value):
        raise ValueError("minimum-jerk phase must be finite")
    phase = min(1.0, max(0.0, float(value)))
    return phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)


def exact_frame_schedule(
    duration_s: float,
    *,
    simulation_hz: int = 600,
    video_hz: int = 30,
) -> tuple[tuple[int, float], ...]:
    """Return exact simulation-step/frame timestamps without an endpoint frame.

    Canonical duration is represented by ``round(duration * fps)`` frames at
    timestamps ``k / fps``.  This deliberately avoids the historical extra
    frame at ``duration``.
    """

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if simulation_hz <= 0 or video_hz <= 0 or simulation_hz % video_hz:
        raise ValueError("simulation_hz must be a positive multiple of video_hz")
    frame_count = int(round(float(duration_s) * video_hz))
    if frame_count <= 0:
        raise ValueError("duration produces no canonical video frames")
    stride = simulation_hz // video_hz
    return tuple((index * stride, index / video_hz) for index in range(frame_count))


def timestep_comparison_failures(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
) -> tuple[str, ...]:
    """Apply the calibrated 600-vs-1200 admission thresholds."""

    failures: list[str] = []
    if coarse.get("outcome") != fine.get("outcome"):
        failures.append("outcome_changed_at_1200_hz")
    try:
        event_delta = abs(
            float(coarse["key_event_time_s"]) - float(fine["key_event_time_s"])
        )
    except (KeyError, TypeError, ValueError):
        failures.append("key_event_time_missing")
    else:
        if not math.isfinite(event_delta) or event_delta > 1.0 / 30.0:
            failures.append("key_event_shift_exceeds_one_frame")
    try:
        left = tuple(float(value) for value in coarse["key_event_position_m"])
        right = tuple(float(value) for value in fine["key_event_position_m"])
        position_delta = math.dist(left, right)
    except (KeyError, TypeError, ValueError):
        failures.append("key_event_position_missing")
    else:
        if len(left) != 3 or len(right) != 3 or position_delta > 0.01:
            failures.append("key_event_position_shift_exceeds_1cm")
    return tuple(failures)


__all__ = [
    "RIGID_REVIEW_PROFILE",
    "RigidReviewProfile",
    "SOURCE_MUJOCO_PROFILE_SCHEMA",
    "SOURCE_MUJOCO_PROFILE_VERSION",
    "exact_frame_schedule",
    "minimum_jerk_fraction",
    "timestep_comparison_failures",
]
