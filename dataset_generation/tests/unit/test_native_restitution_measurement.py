from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import dynamic_robot_dataset.backends.mujoco_native.backend as native_backend_module
from dynamic_robot_dataset.backends.mujoco_native import (
    NativeMuJoCoBackend,
    make_scenario_spec,
    scenario_to_episode_plan,
)
from dynamic_robot_dataset.backends.mujoco_native.backend import (
    _RuntimeAudit,
    _Sample,
    _measure_static_fixture_restitution,
)


def _sample(
    timestamp: float,
    normal_velocity: float,
    *,
    separated: bool,
) -> _Sample:
    zeros3 = np.zeros(3, dtype=np.float64)
    zeros7 = np.zeros(7, dtype=np.float64)
    return _Sample(
        timestamp=timestamp,
        object_position=zeros3.copy(),
        object_quaternion=np.array((1.0, 0.0, 0.0, 0.0)),
        object_linear_velocity=np.array((0.0, 0.0, normal_velocity)),
        object_angular_velocity=zeros3.copy(),
        tool_position=zeros3.copy(),
        tool_quaternion=np.array((1.0, 0.0, 0.0, 0.0)),
        tool_rotation=np.eye(3, dtype=np.float64),
        joint_position=zeros7.copy(),
        joint_velocity=zeros7.copy(),
        joint_command=zeros7.copy(),
        tool_target_position=zeros3.copy(),
        tool_target_rpy=zeros3.copy(),
        phase="test",
        motion_mode="free_flight" if separated else "impact",
        active_surface="none" if separated else "table_surface",
        contact_role="none" if separated else "support",
        controller_enabled=False,
    )


def _event() -> dict[str, object]:
    return {
        "timestamp": 0.2,
        "object_b": "table_surface",
        "normal_world": [0.0, 0.0, 1.0],
        "expected_fixture_contact": True,
    }


def test_restitution_uses_last_incoming_and_first_outgoing_separated_samples() -> None:
    event = _event()
    samples = [
        _sample(0.0, -1.0, separated=True),
        _sample(0.1, -2.0, separated=True),
        # Solver-contact velocities must not be used as post-impact evidence.
        _sample(0.2, 1.8, separated=False),
        _sample(0.3, 1.5, separated=False),
        _sample(0.4, 1.0, separated=True),
        _sample(0.5, 0.8, separated=True),
    ]

    measurements, diagnostics = _measure_static_fixture_restitution(
        samples, [event]
    )

    assert diagnostics["eligible_fixture_contact_count"] == 1
    assert diagnostics["missing_outgoing_sample_count"] == 0
    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement["incoming_sample_time_s"] == pytest.approx(0.1)
    assert measurement["outgoing_sample_time_s"] == pytest.approx(0.4)
    assert measurement["measured_effective_restitution"] == pytest.approx(0.5)
    assert measurement["sample_count"] == 2
    assert "separated" in str(measurement["source"])
    assert event["restitution_measurement_status"] == "measured"
    assert event["restitution_outgoing_sample_time_s"] == pytest.approx(0.4)


def test_restitution_has_no_measurement_without_post_contact_separation() -> None:
    event = _event()
    samples = [
        _sample(0.1, -2.0, separated=True),
        _sample(0.2, 0.8, separated=False),
        _sample(0.3, 0.3, separated=False),
    ]

    measurements, diagnostics = _measure_static_fixture_restitution(
        samples, [event]
    )

    assert measurements == []
    assert diagnostics["missing_outgoing_sample_count"] == 1
    assert event["restitution_measurement_status"] == "no_separated_outgoing_sample"
    assert "restitution_outgoing_sample_time_s" not in event


def test_restitution_does_not_pair_samples_across_two_contacts() -> None:
    first = _event()
    second = {**_event(), "timestamp": 0.4, "object_b": "wall_fixture"}
    samples = [
        _sample(0.1, -2.0, separated=True),
        _sample(0.2, 0.7, separated=False),
        _sample(0.3, -0.8, separated=False),
        _sample(0.4, 0.5, separated=False),
        _sample(0.5, 0.4, separated=True),
    ]

    measurements, diagnostics = _measure_static_fixture_restitution(
        samples, [first, second]
    )

    assert measurements == []
    assert diagnostics["missing_outgoing_sample_count"] == 1
    assert diagnostics["missing_incoming_sample_count"] == 1
    assert first["restitution_measurement_status"] == "no_separated_outgoing_sample"
    assert second["restitution_measurement_status"] == "no_separated_incoming_sample"


def test_physics_qc_rejects_energy_gain_in_any_later_rebound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A safe first rebound must not hide a later energy-creating rebound."""

    pytest.importorskip("mujoco")
    spec = make_scenario_spec("table_bounce", seed=41)
    backend = NativeMuJoCoBackend(width=32, height=32)
    native_plan = backend.compile(scenario_to_episode_plan(spec))
    model, _data, _description = backend._compile_model(native_plan)

    measurements = [
        {
            "fixture": "table_surface",
            "contact_event_time_s": 0.2,
            "measured_effective_restitution": 0.50,
            "sample_count": 2,
        },
        {
            "fixture": "table_surface",
            "contact_event_time_s": 0.5,
            "measured_effective_restitution": 1.051,
            "sample_count": 2,
        },
    ]
    diagnostics = {
        "eligible_fixture_contact_count": 2,
        "missing_incoming_sample_count": 0,
        "missing_outgoing_sample_count": 0,
        "invalid_normal_count": 0,
        "invalid_direction_count": 0,
    }

    def fake_measurements(*_args: object, **_kwargs: object):
        return measurements, diagnostics

    monkeypatch.setattr(
        native_backend_module,
        "_measure_static_fixture_restitution",
        fake_measurements,
    )
    samples = [
        _sample(0.0, 0.0, separated=True),
        _sample(0.1, 0.0, separated=True),
    ]

    result = backend._physics_qc(
        model,
        spec,
        samples,
        [],
        _RuntimeAudit(),
    )

    checks = result["checks"]
    assert checks["measured_effective_restitution"] == pytest.approx(0.50)
    assert checks["rebound_measurement_count"] == 2
    assert checks["no_measured_contact_energy_gain"] is False
    assert result["physics_qc_pass"] is False
