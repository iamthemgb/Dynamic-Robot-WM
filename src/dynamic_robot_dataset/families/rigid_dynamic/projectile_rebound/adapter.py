"""Free-contact projectile/rebound family with parameter counterfactuals."""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping

from ...base import (
    EpisodePlan,
    FamilyAdapter,
    GenerationRequest,
    OutcomeResult,
    SimulationResult,
    assistance_record,
    classify_actual_outcome,
    default_physics_qc,
    physics_field,
    physics_value,
)
from ..solver import (
    analytic_first_impact,
    finite_difference_qc,
    moving_target_position,
    sample_trace,
    simulate_ballistic_sphere_with_paddle,
)


_SWEEPS: dict[str, tuple[float, ...]] = {
    "gravity": (4.905, 7.3575, 9.81, 12.2625, 14.715),
    "friction": (0.05, 0.15, 0.30, 0.60, 1.00),
    "restitution": (0.05, 0.25, 0.50, 0.75, 0.95),
}


def named_physics_sweep(name: str) -> tuple[tuple[str, float], ...]:
    """Return a stable five-point counterfactual sweep."""

    normalized = name.strip().lower()
    if normalized not in _SWEEPS:
        raise ValueError(f"unknown physics sweep {name!r}; choose from {tuple(_SWEEPS)}")
    return tuple((f"{normalized}_{index}", value) for index, value in enumerate(_SWEEPS[normalized]))


class ProjectileReboundAdapter(FamilyAdapter):
    family = "projectile_rebound"
    supported_subfamilies = (
        "default",
        "free_contact_rebound",
        "projectile_interception",
        "bounce_sweep",
        "direct_interception",
        "table_bounce",
        "wall_rebound",
        "angled_barrier_rebound",
        "paddle_deflection",
        "gravity_sweep",
        "restitution_sweep",
        "ramp_launch",
        "roll_off_edge",
        "floor_to_wall",
        "flight_to_table_bounce",
        "bounce_to_robot_interception",
    )
    default_duration_s = 3.8

    def scene_parameters(
        self, request: GenerationRequest, scene_seed: int, scene_index: int
    ) -> Mapping[str, Any]:
        rng = random.Random(scene_seed)
        return {
            "object_id": "projectile_ball",
            "initial_position_m": [0.0, rng.uniform(-0.025, 0.025), rng.uniform(1.08, 1.16)],
            "initial_velocity_mps": [rng.uniform(2.45, 2.60), rng.uniform(-0.025, 0.025), rng.uniform(-0.15, 0.05)],
            "floor_z_m": 0.0,
            "paddle_start_m": [1.35, -0.42, 0.29],
            "paddle_x_m": 1.35,
            "paddle_radius_m": 0.18,
            "goal_direction": [-1.0, 0.0, 0.0],
            "visual_seed": scene_seed,
            "background_style": request.scene_style,
            "native_scenario_profile": request.subfamily,
        }

    @staticmethod
    def _nominal_crossing(scene: Mapping[str, Any]) -> tuple[float, float]:
        position = scene["initial_position_m"]
        velocity = scene["initial_velocity_mps"]
        time_s = (float(scene["paddle_x_m"]) - float(position[0])) / float(velocity[0])
        # Piecewise analytic nominal ground bounce, used only to plan an action.
        impact = analytic_first_impact(
            initial_height_m=float(position[2]),
            initial_vertical_velocity_mps=float(velocity[2]),
            floor_center_height_m=0.04,
            gravity_z_mps2=-9.81,
        )
        if impact is None or time_s <= impact[0]:
            z = position[2] + velocity[2] * time_s - 0.5 * 9.81 * time_s * time_s
        else:
            elapsed = time_s - impact[0]
            rebound_vz = -0.62 * impact[1]
            z = 0.04 + rebound_vz * elapsed - 0.5 * 9.81 * elapsed * elapsed
        y = position[1] + velocity[1] * time_s
        return y, 0.29

    def branch_parameters(
        self,
        request: GenerationRequest,
        branch: str,
        branch_seed: int,
        scene: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        rng = random.Random(branch_seed)
        y, z = self._nominal_crossing(scene)
        if branch == "success_seeking":
            offset = (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01))
            normal = [-1.0, 0.0, 0.0]
            enabled = True
        elif branch in {"wrong_rebound_prediction", "contact_failure"}:
            offset = (rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015))
            normal = [-0.64, rng.choice((-1.0, 1.0)) * 0.768, 0.0]
            enabled = True
        elif branch == "near_miss":
            offset = (rng.choice((-1.0, 1.0)) * rng.uniform(0.26, 0.30), 0.0)
            normal = [-1.0, 0.0, 0.0]
            enabled = True
        else:
            offset = (rng.choice((-1.0, 1.0)) * rng.uniform(0.35, 0.48), rng.uniform(0.12, 0.2))
            normal = [-1.0, 0.0, 0.0]
            enabled = False
        return {
            "paddle_target_m": [float(scene["paddle_x_m"]), y + offset[0], z + offset[1]],
            "paddle_normal": normal,
            "motion_start_s": 0.05 if enabled else 2.0,
            "travel_time_s": 0.55,
            "controller_enabled": enabled,
            "desired_rebound_direction": list(scene["goal_direction"]),
        }

    @staticmethod
    def _physics(request: GenerationRequest, *, gravity: float = 9.81, friction: float = 0.15, restitution: float = 0.62) -> Mapping[str, Any]:
        return {
            "gravity": physics_field([0.0, 0.0, -gravity], "m/s^2"),
            "mass": physics_field(0.070, "kg"),
            "radius": physics_field(0.040, "m"),
            "floor_dynamic_friction": physics_field(friction, "1", interpretation="calibrated_effective"),
            "floor_static_friction": physics_field(None, "1", implemented=False, interpretation="unknown"),
            "floor_restitution": physics_field(restitution, "1", interpretation="calibrated_effective"),
            "paddle_dynamic_friction": physics_field(0.18, "1", interpretation="calibrated_effective"),
            "paddle_restitution": physics_field(0.35, "1", interpretation="calibrated_effective"),
            "simulation_timestep": physics_field(1.0 / request.sim_hz, "s"),
            "substeps": physics_field(1.0, "count"),
        }

    def physics_variants(
        self, request: GenerationRequest
    ) -> Iterable[tuple[str, Mapping[str, Any]]]:
        if not request.physics_sweep:
            yield "nominal", self._physics(request)
            return
        for variant, value in named_physics_sweep(request.physics_sweep):
            kwargs: dict[str, float] = {}
            kwargs[request.physics_sweep] = value
            yield variant, self._physics(request, **kwargs)

    def simulate(self, plan: EpisodePlan) -> SimulationResult:
        scene = plan.scene_parameters
        branch = plan.branch_parameters
        gravity = tuple(physics_value(plan.physics, "gravity"))
        initial_position = tuple(scene["initial_position_m"])
        initial_velocity = tuple(scene["initial_velocity_mps"])
        radius = float(physics_value(plan.physics, "radius"))
        mass = float(physics_value(plan.physics, "mass"))

        def paddle_center(time_s: float) -> tuple[float, float, float]:
            if not branch["controller_enabled"]:
                return tuple(scene["paddle_start_m"])
            return moving_target_position(
                time_s,
                start=tuple(scene["paddle_start_m"]),
                target=tuple(branch["paddle_target_m"]),
                start_time_s=float(branch["motion_start_s"]),
                travel_time_s=float(branch["travel_time_s"]),
            )

        trace = simulate_ballistic_sphere_with_paddle(
            initial_position_m=initial_position,
            initial_velocity_mps=initial_velocity,
            radius_m=radius,
            mass_kg=mass,
            gravity_mps2=gravity,
            floor_restitution=float(physics_value(plan.physics, "floor_restitution")),
            floor_friction=float(physics_value(plan.physics, "floor_dynamic_friction")),
            paddle_restitution=float(physics_value(plan.physics, "paddle_restitution")),
            paddle_friction=float(physics_value(plan.physics, "paddle_dynamic_friction")),
            paddle_center=paddle_center,
            paddle_normal=tuple(branch["paddle_normal"]),
            paddle_radius_m=float(scene["paddle_radius_m"]),
            duration_s=plan.duration_s,
            sim_hz=plan.rates_hz["simulation"],
            floor_z_m=float(scene["floor_z_m"]),
        )
        paddle_contacts = [event for event in trace.contacts if event["object_b"] == "robot_paddle"]
        ground_contacts = [event for event in trace.contacts if event["object_b"] == "floor"]
        contacted = bool(paddle_contacts)
        if contacted:
            outgoing = tuple(
                paddle_contacts[0].get(
                    "object_velocity_post_mps",
                    paddle_contacts[0]["relative_velocity_post_mps"],
                )
            )
            horizontal_speed = math.hypot(outgoing[0], outgoing[1])
            alignment = -outgoing[0] / max(horizontal_speed, 1e-9)
            lateral_ratio = abs(outgoing[1]) / max(horizontal_speed, 1e-9)
            redirected = outgoing[0] < -0.15 and alignment >= 0.75 and lateral_ratio <= 0.45
        else:
            outgoing = trace.velocities_mps[-1]
            alignment = -1.0
            lateral_ratio = 1.0
            redirected = False
        success = contacted and redirected
        paddle_distances = [
            math.sqrt(
                (position[1] - paddle_center(time_s)[1]) ** 2
                + (position[2] - paddle_center(time_s)[2]) ** 2
            )
            for time_s, position in zip(trace.timestamps_s, trace.positions_m)
            if abs(position[0] - paddle_center(time_s)[0]) <= 0.08
        ]
        min_distance = min(paddle_distances, default=10.0)
        if success:
            failure = "none"
        elif contacted:
            failure = "incorrect_rebound_direction"
        elif not branch["controller_enabled"]:
            failure = "controller_no_op"
        elif min_distance <= float(scene["paddle_radius_m"]) + 2.0 * radius:
            failure = "paddle_near_miss"
        else:
            failure = "paddle_missed_projectile"
        score = max(0.0, min(1.0, (alignment + 1.0) * 0.5)) if contacted else max(
            0.0, 1.0 - min_distance / 0.3
        )
        first_impact_oracle = analytic_first_impact(
            initial_height_m=initial_position[2],
            initial_vertical_velocity_mps=initial_velocity[2],
            floor_center_height_m=float(scene["floor_z_m"]) + radius,
            gravity_z_mps2=gravity[2],
        )
        impact_time_error = None
        restitution_error = None
        if ground_contacts and first_impact_oracle:
            impact_time_error = abs(float(ground_contacts[0]["timestamp"]) - first_impact_oracle[0])
            pre = float(ground_contacts[0]["relative_velocity_pre_mps"][2])
            post = float(ground_contacts[0]["relative_velocity_post_mps"][2])
            measured = post / max(-pre, 1e-9)
            restitution_error = abs(measured - float(physics_value(plan.physics, "floor_restitution")))
        first_paddle_time = min((float(event["timestamp"]) for event in paddle_contacts), default=math.inf)
        first_ground_time = min((float(event["timestamp"]) for event in ground_contacts), default=math.inf)
        oracle_applicable = first_ground_time < first_paddle_time
        outcome = OutcomeResult(
            success,
            1.0 if success else score,
            failure,
            {
                "controller_reached_target": branch["controller_enabled"],
                "object_contacted_tool": contacted,
                "correct_rebound_direction": redirected,
                "rebound_alignment": alignment,
                "rebound_lateral_speed_ratio": lateral_ratio,
                "minimum_projectile_paddle_radial_distance_m": min_distance,
                "ground_impact_count": len(ground_contacts),
            },
            0.98,
        )
        frame_indices = sample_trace(trace, plan.rates_hz["simulation"], plan.rates_hz["video"])
        states = [
            {
                "timestamp": trace.timestamps_s[index],
                "object.position": list(trace.positions_m[index]),
                "object.linear_velocity": list(trace.velocities_mps[index]),
                "object.angular_velocity": list(trace.angular_velocities_radps[index]),
                "robot.paddle_position": list(paddle_center(trace.timestamps_s[index])),
                "robot.paddle_normal": list(branch["paddle_normal"]),
            }
            for index in frame_indices
        ]
        actions = [
            {
                "timestamp": trace.timestamps_s[index],
                "command.paddle_target_position": list(branch["paddle_target_m"]),
                "command.paddle_normal": list(branch["paddle_normal"]),
                "command.enabled": bool(branch["controller_enabled"]),
            }
            for index in frame_indices
        ]
        high_rate = [
            {
                "timestamp": time_s,
                "object.position": list(position),
                "object.linear_velocity": list(velocity),
                "robot.paddle_position": list(paddle_center(time_s)),
            }
            for time_s, position, velocity in zip(
                trace.timestamps_s, trace.positions_m, trace.velocities_mps
            )
        ]
        qc = finite_difference_qc(
            trace,
            gravity_mps2=gravity,
            contact_guard_s=2.0 / plan.rates_hz["simulation"],
        )
        qc.update(
            {
                "analytic_impact_time_consistent": (not oracle_applicable) or (
                    impact_time_error is not None
                    and impact_time_error <= 2.0 / plan.rates_hz["simulation"]
                ),
                "analytic_oracle_applicability": "pre_tool_free_flight" if oracle_applicable else "not_applicable_after_prior_tool_contact",
                "restitution_consistent": restitution_error is not None and restitution_error <= 1e-6,
                "impact_time_error_s": impact_time_error if impact_time_error is not None else -1.0,
                "restitution_absolute_error": restitution_error if restitution_error is not None else -1.0,
                "no_post_impact_state_reset": True,
            }
        )
        return SimulationResult(
            plan=plan,
            frame_times_s=[trace.timestamps_s[index] for index in frame_indices],
            states=states,
            actions=actions,
            high_rate_states=high_rate,
            contacts=trace.contacts,
            outcome=outcome,
            actual_outcome=classify_actual_outcome(
                success=success,
                contacted=contacted,
                near_distance_m=min_distance,
                near_threshold_m=float(scene["paddle_radius_m"]) + 2.0 * radius,
                bad_action=not branch["controller_enabled"] and plan.intended_branch != "no_op",
                no_op=plan.intended_branch == "no_op",
            ),
            dynamics_mode="free_contact",
            release_tier="free_contact",
            assistance=assistance_record(),
            physics_qc=default_physics_qc(**qc),
            simulator={
                "name": "dynamic_robot_dataset.continuous_collision_rigid_solver",
                "version": "1",
                "native_mujoco": False,
                "contact_resolution": "event_time_impulse",
                "post_impact_keyframes": False,
                "coordinate_frame": "task_local_right_handed_z_up",
                "quaternion_order": "wxyz",
            },
            notes=("Analytic impact equations are validation oracles only; saved states come from the contact solver.",),
        )
