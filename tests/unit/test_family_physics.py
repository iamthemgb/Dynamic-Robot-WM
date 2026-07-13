from __future__ import annotations

import math

import pytest

from dynamic_robot_dataset.families.base import GenerationRequest
from dynamic_robot_dataset.families.rigid_dynamic.projectile_rebound.adapter import (
    ProjectileReboundAdapter,
    named_physics_sweep,
)
from dynamic_robot_dataset.families.rigid_dynamic.solver import (
    analytic_first_impact,
    finite_difference_qc,
    simulate_ballistic_sphere,
    simulate_ballistic_sphere_with_paddle,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gravity", (4.905, 7.3575, 9.81, 12.2625, 14.715)),
        ("friction", (0.05, 0.15, 0.30, 0.60, 1.00)),
        ("restitution", (0.05, 0.25, 0.50, 0.75, 0.95)),
    ],
)
def test_five_point_sweeps_match_contract(name: str, expected: tuple[float, ...]) -> None:
    assert tuple(value for _, value in named_physics_sweep(name)) == expected


def test_physics_counterfactuals_reuse_scene_action_and_camera() -> None:
    adapter = ProjectileReboundAdapter()
    plans = adapter.plan(
        GenerationRequest(
            family="projectile_rebound",
            subfamily="bounce_sweep",
            physics_sweep="gravity",
            num_bundles=1,
            branches=("success_seeking",),
            views=("main", "secondary"),
            scene_style="clean_franka_lab",
        )
    )
    assert len(plans) == 5
    assert len({plan.physics_counterfactual_family_id for plan in plans}) == 1
    assert len({plan.action_hash for plan in plans}) == 1
    assert len({plan.invariant_hash for plan in plans}) == 1
    assert len({plan.scene_seed for plan in plans}) == 1
    assert len({plan.views for plan in plans}) == 1
    assert len({plan.physics_hash for plan in plans}) == 5


def test_rebound_oracle_uses_gravity_and_no_scripted_keyframes() -> None:
    impact = analytic_first_impact(
        initial_height_m=1.0,
        initial_vertical_velocity_mps=0.0,
        floor_center_height_m=0.04,
        gravity_z_mps2=-9.81,
    )
    assert impact is not None
    time_s, pre_velocity = impact
    assert time_s == pytest.approx(math.sqrt(2 * 0.96 / 9.81), rel=1e-8)
    assert pre_velocity == pytest.approx(-9.81 * time_s, rel=1e-8)

    trace = simulate_ballistic_sphere(
        initial_position_m=(0.0, 0.0, 1.0),
        initial_velocity_mps=(0.1, 0.0, 0.0),
        radius_m=0.04,
        mass_kg=0.07,
        gravity_mps2=(0.0, 0.0, -9.81),
        restitution=0.5,
        friction=0.2,
        duration_s=1.2,
        sim_hz=240,
        floor_z_m=0.0,
        object_id="ball",
        surface_id="floor",
    )
    assert trace.contacts
    event = trace.contacts[0]
    pre = event["relative_velocity_pre_mps"][2]
    post = event["relative_velocity_post_mps"][2]
    assert post / -pre == pytest.approx(0.5, abs=1e-12)
    assert max(abs(position[2] - 0.04) for position in trace.positions_m) > 0.1


def test_finite_difference_qc_accepts_clean_free_fall() -> None:
    trace = simulate_ballistic_sphere(
        initial_position_m=(0.0, 0.0, 5.0),
        initial_velocity_mps=(0.0, 0.0, 0.0),
        radius_m=0.04,
        mass_kg=0.07,
        gravity_mps2=(0.0, 0.0, -9.81),
        restitution=0.5,
        friction=0.2,
        duration_s=0.5,
        sim_hz=240,
        floor_z_m=0.0,
        object_id="ball",
        surface_id="floor",
    )
    qc = finite_difference_qc(
        trace,
        gravity_mps2=(0.0, 0.0, -9.81),
        contact_guard_s=1.0 / 120.0,
    )
    assert qc["finite_state"]
    assert qc["position_velocity_consistent"]
    assert qc["free_flight_acceleration_consistent"]


def test_oriented_paddle_uses_same_plane_and_impulse_normal() -> None:
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    normal = (-inverse_sqrt_two, inverse_sqrt_two, 0.0)
    initial = tuple(0.60 * value for value in normal)
    incoming = tuple(-value for value in normal)
    trace = simulate_ballistic_sphere_with_paddle(
        initial_position_m=initial,
        initial_velocity_mps=incoming,
        radius_m=0.05,
        mass_kg=1.0,
        gravity_mps2=(0.0, 0.0, 0.0),
        floor_restitution=0.0,
        floor_friction=0.0,
        paddle_restitution=0.5,
        paddle_friction=0.0,
        paddle_center=lambda _time: (0.0, 0.0, 0.0),
        paddle_normal=normal,
        paddle_radius_m=1.0,
        duration_s=1.0,
        sim_hz=240,
        floor_z_m=-10.0,
    )
    contacts = [event for event in trace.contacts if event["object_b"] == "robot_paddle"]
    assert len(contacts) == 1
    event = contacts[0]
    assert event["timestamp"] == pytest.approx(0.55, abs=1e-10)
    assert tuple(event["contact_normal"]) == pytest.approx(normal, abs=1e-12)
    assert sum(a * b for a, b in zip(event["contact_point_m"], normal)) == pytest.approx(
        0.0, abs=1e-10
    )
    pre_normal = sum(a * b for a, b in zip(event["relative_velocity_pre_mps"], normal))
    post_normal = sum(a * b for a, b in zip(event["relative_velocity_post_mps"], normal))
    assert pre_normal == pytest.approx(-1.0, abs=1e-10)
    assert post_normal == pytest.approx(0.5, abs=1e-10)
