"""Deterministic falling-object catch adapter.

The smoke simulator models a freely falling sphere and a kinematically commanded
receptacle.  It is deliberately smaller than the migrated MuJoCo implementation;
metadata names this implementation so results cannot be mistaken for native
MuJoCo rollouts.
"""

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
    distance_xy,
    finite_difference_qc,
    moving_target_position,
    sample_trace,
    simulate_ballistic_sphere,
)


class FallingCatchAdapter(FamilyAdapter):
    family = "falling_catch"
    supported_subfamilies = (
        "default",
        "centered_vertical_drop",
        "off_center_drop",
        "drifted_drop",
        "direct_projectile",
        "catch_retain",
        "catch_transport",
        "catch_brake",
        "catch_tilt",
        "catch_recovery",
        "shallow_tray",
        "deep_tray",
    )
    default_duration_s = 3.2

    def scene_parameters(
        self, request: GenerationRequest, scene_seed: int, scene_index: int
    ) -> Mapping[str, Any]:
        rng = random.Random(scene_seed)
        drift = 0.0 if request.subfamily == "centered_vertical_drop" else rng.uniform(-0.08, 0.08)
        return {
            "object_id": "target_ball",
            "object_shape": "sphere",
            "initial_position_m": [rng.uniform(-0.025, 0.025), rng.uniform(-0.02, 0.02), rng.uniform(1.05, 1.2)],
            "initial_velocity_mps": [drift, rng.uniform(-0.025, 0.025), 0.0],
            "receptacle_start_m": [-0.35, 0.0, 0.42],
            "receptacle_radius_m": 0.105,
            "receptacle_surface_z_m": 0.40,
            "background_style": request.scene_style,
            "visual_seed": scene_seed,
            "native_scenario_profile": request.subfamily,
            "tool_geometry": (
                "deep_tray" if request.subfamily == "deep_tray" else
                "shallow_tray" if request.subfamily == "shallow_tray" else
                request.tool_type
            ),
        }

    def branch_parameters(
        self,
        request: GenerationRequest,
        branch: str,
        branch_seed: int,
        scene: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        rng = random.Random(branch_seed)
        ball_xy = scene["initial_position_m"][:2]
        offsets = {
            "success_seeking": (rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015)),
            "near_miss": (rng.choice((-1.0, 1.0)) * rng.uniform(0.115, 0.14), rng.uniform(-0.02, 0.02)),
            "contact_failure": (rng.choice((-1.0, 1.0)) * rng.uniform(0.07, 0.095), rng.uniform(-0.01, 0.01)),
            "bad_action": (rng.choice((-1.0, 1.0)) * rng.uniform(0.28, 0.4), rng.uniform(-0.1, 0.1)),
        }
        offset = offsets.get(branch, offsets["bad_action"])
        return {
            "target_position_m": [ball_xy[0] + offset[0], ball_xy[1] + offset[1], scene["receptacle_start_m"][2]],
            "motion_start_s": 0.0 if branch != "bad_action" else 1.8,
            "travel_time_s": 0.20 if branch != "bad_action" else 0.8,
            "controller_enabled": branch != "bad_action",
            "retention_hold_s": 0.45,
        }

    def physics_variants(
        self, request: GenerationRequest
    ) -> Iterable[tuple[str, Mapping[str, Any]]]:
        yield "nominal", {
            "gravity": physics_field([0.0, 0.0, -9.81], "m/s^2"),
            "mass": physics_field(0.065, "kg"),
            "radius": physics_field(0.035, "m"),
            "surface_dynamic_friction": physics_field(0.35, "1", interpretation="calibrated_effective"),
            "surface_static_friction": physics_field(None, "1", implemented=False, interpretation="unknown"),
            "restitution": physics_field(0.08, "1", interpretation="calibrated_effective"),
            "simulation_timestep": physics_field(1.0 / request.sim_hz, "s"),
        }

    def simulate(self, plan: EpisodePlan) -> SimulationResult:
        scene = plan.scene_parameters
        branch = plan.branch_parameters
        gravity = tuple(physics_value(plan.physics, "gravity"))
        radius = float(physics_value(plan.physics, "radius"))
        mass = float(physics_value(plan.physics, "mass"))
        friction = float(physics_value(plan.physics, "surface_dynamic_friction"))
        restitution = float(physics_value(plan.physics, "restitution"))
        initial_position = tuple(scene["initial_position_m"])
        initial_velocity = tuple(scene["initial_velocity_mps"])
        intercept_oracle = analytic_first_impact(
            initial_height_m=initial_position[2],
            initial_vertical_velocity_mps=initial_velocity[2],
            floor_center_height_m=float(scene["receptacle_surface_z_m"]) + radius,
            gravity_z_mps2=gravity[2],
        )
        intercepts_disk = False
        if intercept_oracle is not None:
            impact_time = intercept_oracle[0]
            ball_at_impact = (
                initial_position[0] + initial_velocity[0] * impact_time,
                initial_position[1] + initial_velocity[1] * impact_time,
                float(scene["receptacle_surface_z_m"]) + radius,
            )
            tool_at_impact = (
                moving_target_position(
                    impact_time,
                    start=tuple(scene["receptacle_start_m"]),
                    target=tuple(branch["target_position_m"]),
                    start_time_s=float(branch["motion_start_s"]),
                    travel_time_s=float(branch["travel_time_s"]),
                )
                if branch["controller_enabled"]
                else tuple(scene["receptacle_start_m"])
            )
            intercepts_disk = distance_xy(ball_at_impact, tool_at_impact) <= float(
                scene["receptacle_radius_m"]
            )
        trace = simulate_ballistic_sphere(
            initial_position_m=initial_position,
            initial_velocity_mps=initial_velocity,
            radius_m=radius,
            mass_kg=mass,
            gravity_mps2=gravity,
            restitution=restitution,
            friction=friction,
            duration_s=plan.duration_s,
            sim_hz=plan.rates_hz["simulation"],
            floor_z_m=(
                float(scene["receptacle_surface_z_m"]) if intercepts_disk else 0.0
            ),
            object_id=str(scene["object_id"]),
            surface_id="receptacle_support_disk" if intercepts_disk else "table_surface",
        )
        frame_indices = sample_trace(
            trace, plan.rates_hz["simulation"], plan.rates_hz["video"]
        )
        target_positions = [
            (
                moving_target_position(
                    time_s,
                    start=tuple(scene["receptacle_start_m"]),
                    target=tuple(branch["target_position_m"]),
                    start_time_s=float(branch["motion_start_s"]),
                    travel_time_s=float(branch["travel_time_s"]),
                )
                if branch["controller_enabled"]
                else tuple(scene["receptacle_start_m"])
            )
            for time_s in trace.timestamps_s
        ]
        distances = [distance_xy(ball, target) for ball, target in zip(trace.positions_m, target_positions)]
        contacted = any(event["object_b"] == "receptacle_support_disk" for event in trace.contacts)
        final_distance = distances[-1]
        final_speed = math.sqrt(sum(value * value for value in trace.velocities_mps[-1]))
        retained = contacted and final_distance <= 0.055 and final_speed <= 0.12
        success = bool(retained)
        min_distance = min(distances)
        if success:
            failure = "none"
        elif contacted:
            failure = "object_not_retained"
        elif min_distance <= 1.5 * float(scene["receptacle_radius_m"]):
            failure = "receptacle_near_miss"
        elif not branch["controller_enabled"]:
            failure = "controller_no_op"
        else:
            failure = "receptacle_missed_object"
        score = max(0.0, min(1.0, 1.0 - min_distance / (2.0 * float(scene["receptacle_radius_m"]))))
        outcome = OutcomeResult(
            success,
            1.0 if success else score,
            failure,
            {
                "controller_reached_target": bool(branch["controller_enabled"])
                and distance_xy(target_positions[-1], tuple(branch["target_position_m"])) < 1e-6,
                "object_contacted_tool": contacted,
                "object_entered_receptacle": contacted,
                "object_retained_until_end": retained,
                "minimum_object_tool_distance_m": min_distance,
                "final_object_tool_distance_m": final_distance,
                "final_object_speed_mps": final_speed,
            },
            0.9,
        )
        states = []
        actions = []
        for index in frame_indices:
            states.append(
                {
                    "timestamp": trace.timestamps_s[index],
                    "object.position": list(trace.positions_m[index]),
                    "object.linear_velocity": list(trace.velocities_mps[index]),
                    "object.angular_velocity": list(trace.angular_velocities_radps[index]),
                    "robot.receptacle_position": list(target_positions[index]),
                    "assistance.active": False,
                }
            )
            actions.append(
                {
                    "timestamp": trace.timestamps_s[index],
                    "command.receptacle_target_position": list(branch["target_position_m"]),
                    "command.enabled": bool(branch["controller_enabled"]),
                }
            )
        high_rate = [
            {
                "timestamp": time_s,
                "object.position": list(position),
                "object.linear_velocity": list(velocity),
                "robot.receptacle_position": list(target),
            }
            for time_s, position, velocity, target in zip(
                trace.timestamps_s, trace.positions_m, trace.velocities_mps, target_positions
            )
        ]
        qc = finite_difference_qc(
            trace,
            gravity_mps2=gravity,
            contact_guard_s=2.0 / plan.rates_hz["simulation"],
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
                near_threshold_m=1.5 * float(scene["receptacle_radius_m"]),
                bad_action=not branch["controller_enabled"] and plan.intended_branch != "no_op",
                no_op=plan.intended_branch == "no_op",
            ),
            dynamics_mode="free_contact",
            release_tier="free_contact",
            assistance=assistance_record(),
            physics_qc=default_physics_qc(**qc),
            simulator={
                "name": "dynamic_robot_dataset.rigid_event_solver",
                "version": "1",
                "native_mujoco": False,
                "coordinate_frame": "task_local_right_handed_z_up",
                "quaternion_order": "wxyz",
            },
            notes=(
                "Smoke solver uses a planar receptacle support; native MuJoCo adapters must replace it for production rendering.",
            ),
        )
