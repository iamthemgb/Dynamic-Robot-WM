"""Rebound measurement must not demand evidence past the persisted horizon."""

from dynamic_robot_dataset.common.rebound import (
    DEFAULT_REBOUND_ACCEPTANCE,
    measure_rebound_kinematics,
)


def _state(t: float, z: float, vz: float, contacts: int) -> dict:
    return {
        "timestamp": t,
        "object.position": (0.0, 0.0, z),
        "object.linear_velocity": (0.0, 0.0, vz),
        "contact.count": contacts,
    }


def _impact_event(t: float) -> dict:
    return {
        "contact_category": "task_surface",
        "timestamp": t,
        "normal_world": (0.0, 0.0, 1.0),
    }


def test_late_impact_is_structurally_unmeasurable_not_invalid() -> None:
    # First pad impact lands inside the final separation window (the P0b
    # calibration geometry: impact at 0.79-0.80 s of a 0.8 s episode).
    states = [
        _state(0.0, 0.60, -1.0, 0),
        _state(0.78, 0.03, -3.9, 0),
        _state(0.795, 0.024, -0.5, 1),
        _state(0.80, 0.024, -0.2, 1),
    ]
    result = measure_rebound_kinematics(
        states,
        [_impact_event(0.79)],
        object_radius_m=0.024,
    )
    assert result["rebound_window_truncated"] is True
    assert result["applicable"] is False
    assert result["effective_restitution"] is None


def test_early_impact_keeps_the_full_measurement() -> None:
    duration = DEFAULT_REBOUND_ACCEPTANCE.minimum_separation_duration_s
    states = [
        _state(0.0, 0.60, -1.0, 0),
        _state(0.19, 0.03, -3.9, 0),
        _state(0.20, 0.024, 0.0, 1),
        _state(0.21, 0.03, 1.9, 0),
        _state(0.21 + duration, 0.06, 1.6, 0),
        _state(0.30, 0.10, 0.9, 0),
    ]
    result = measure_rebound_kinematics(
        states,
        [_impact_event(0.20)],
        object_radius_m=0.024,
    )
    assert result["rebound_window_truncated"] is False
    assert result["applicable"] is True
    assert result["effective_restitution"] is not None
