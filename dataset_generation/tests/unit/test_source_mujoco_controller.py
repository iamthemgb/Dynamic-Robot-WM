"""Unit contracts for the v9 ctrl-only reaching-catch controller plan."""

from __future__ import annotations

import numpy as np
import pytest

from dynamic_robot_dataset.backends.source_mujoco.controller import (
    OwnedControllerPlan,
    feasible_reach_duration_s,
)
from dynamic_robot_dataset.backends.source_mujoco.profiles import (
    RIGID_REVIEW_PROFILE,
)


READY = (0.0, -0.7, 0.1, -2.2, 0.05, 1.65, 0.78)
INTERCEPT = (0.02, -0.73, 0.0, -2.18, -0.04, 1.6, 0.79)


def _plan(**overrides) -> OwnedControllerPlan:
    values = dict(
        embodiment="franka_hand",
        initial_arm_command=READY,
        intercept_arm_command=INTERCEPT,
        open_gripper_command=255.0,
        closed_gripper_command=131.0,
        arm_motion_start_s=0.0,
        arm_motion_end_s=0.385,
        closure_start_s=0.28,
        closure_end_s=0.28 + RIGID_REVIEW_PROFILE.closure_duration_s,
        transport_arm_command=None,
    )
    values.update(overrides)
    plan = OwnedControllerPlan(**values)
    plan.validate()
    return plan


def test_feasible_reach_duration_floors_at_the_profile_minimum() -> None:
    tiny = feasible_reach_duration_s((1e-5,) * 7)
    assert tiny == RIGID_REVIEW_PROFILE.minimum_reach_duration_s


def test_feasible_reach_duration_grows_with_joint_travel() -> None:
    small = feasible_reach_duration_s((0.1,) * 7)
    large = feasible_reach_duration_s((0.8,) * 7)
    assert large > small >= RIGID_REVIEW_PROFILE.minimum_reach_duration_s


def test_feasible_reach_duration_rejects_malformed_deltas() -> None:
    with pytest.raises(ValueError, match="seven finite"):
        feasible_reach_duration_s((0.1,) * 6)
    with pytest.raises(ValueError, match="seven finite"):
        feasible_reach_duration_s((float("nan"),) * 7)


def test_zero_arm_travel_plan_is_rejected_as_pre_positioned() -> None:
    with pytest.raises(ValueError, match="stationary"):
        _plan(initial_arm_command=INTERCEPT)


def test_reach_must_begin_before_the_bounded_closure() -> None:
    with pytest.raises(ValueError, match="reach must begin"):
        _plan(arm_motion_start_s=0.30, arm_motion_end_s=0.60)


def test_command_at_is_minimum_jerk_between_ready_and_intercept() -> None:
    plan = _plan()
    start = plan.command_at(0.0)
    assert start[:7] == pytest.approx(np.asarray(READY))
    assert start[7] == 255.0
    arrived = plan.command_at(plan.arm_motion_end_s)
    assert arrived[:7] == pytest.approx(np.asarray(INTERCEPT))
    assert arrived[7] == 131.0
    midpoint = plan.command_at(0.5 * plan.arm_motion_end_s)
    expected_alpha = 0.5
    assert midpoint[:7] == pytest.approx(
        np.asarray(READY)
        + expected_alpha * (np.asarray(INTERCEPT) - np.asarray(READY))
    )
    quarter = plan.command_at(0.25 * plan.arm_motion_end_s)
    quarter_alpha = 0.25**3 * (10.0 - 15.0 * 0.25 + 6.0 * 0.25**2)
    assert quarter[:7] == pytest.approx(
        np.asarray(READY)
        + quarter_alpha * (np.asarray(INTERCEPT) - np.asarray(READY))
    )


def test_transport_override_does_not_erase_the_reach_before_it_starts() -> None:
    transport = tuple(value + 0.2 for value in INTERCEPT)
    plan = _plan(
        transport_arm_command=transport,
        transport_start_s=1.0,
        transport_end_s=1.6,
    )
    during_reach = plan.command_at(0.2)
    without_transport = _plan().command_at(0.2)
    assert during_reach[:7] == pytest.approx(without_transport[:7])
    assert plan.command_at(0.999)[:7] == pytest.approx(np.asarray(INTERCEPT))
    assert plan.command_at(1.6)[:7] == pytest.approx(np.asarray(transport))


def test_transport_cannot_begin_before_reach_arrival() -> None:
    with pytest.raises(ValueError, match="transport cannot begin"):
        _plan(
            transport_arm_command=tuple(value + 0.2 for value in INTERCEPT),
            transport_start_s=0.2,
            transport_end_s=0.8,
        )


def test_closure_may_complete_after_arrival_for_early_arrivals() -> None:
    plan = _plan(arm_motion_end_s=0.30)
    assert plan.closure_end_s > plan.arm_motion_end_s
    settled = plan.command_at(0.35)
    assert settled[:7] == pytest.approx(np.asarray(INTERCEPT))
    assert 131.0 < settled[7] < 255.0
