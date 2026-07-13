"""Planar free-contact rolling interception smoke model."""

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
from ..solver import moving_target_position


class RollingInterceptionAdapter(FamilyAdapter):
    family = "rolling_interception"
    supported_subfamilies = (
        "default",
        "rolling_island",
        "rolling_ball_interception",
        "straight_ball_left_to_right",
        "straight_ball_right_to_left",
        "rolling_sliding_transition",
        "sliding_puck",
        "sliding_cube",
        "small_slope",
        "ramp_to_table",
        "table_edge_fall",
        "friction_sweep",
        "temporary_occlusion",
        "container_receive",
        "paddle_stop",
        "paddle_redirect",
        "redirect_to_target",
    )
    default_duration_s = 4.5

    def scene_parameters(
        self, request: GenerationRequest, scene_seed: int, scene_index: int
    ) -> Mapping[str, Any]:
        rng = random.Random(scene_seed)
        reverse = request.subfamily == "straight_ball_right_to_left"
        direction = -1.0 if reverse else 1.0
        object_shape = (
            "cylinder" if request.subfamily == "sliding_puck" else
            "box" if request.subfamily == "sliding_cube" else
            "sphere"
        )
        return {
            "initial_position_m": [direction * -0.62, rng.uniform(-0.02, 0.02), 0.04],
            "initial_velocity_mps": [direction * rng.uniform(0.70, 0.82), rng.uniform(-0.015, 0.015), 0.0],
            "tool_start_m": [0.08, -0.36, 0.04],
            "tool_radius_m": 0.075,
            "goal_center_m": [-0.02, 0.0, 0.04],
            "goal_radius_m": 0.19,
            "table_bounds_m": [-0.8, 0.8, -0.55, 0.55],
            "visual_seed": scene_seed,
            "background_style": request.scene_style,
            "native_scenario_profile": request.subfamily,
            "object_shape": object_shape,
            "table_slope_deg": 3.0 if request.subfamily == "small_slope" else 0.0,
        }

    def branch_parameters(
        self,
        request: GenerationRequest,
        branch: str,
        branch_seed: int,
        scene: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        rng = random.Random(branch_seed)
        if branch == "success_seeking":
            y = rng.uniform(-0.012, 0.012)
            start_s, duration_s, enabled = 0.15, 0.55, True
        elif branch == "near_miss":
            y = rng.choice((-1.0, 1.0)) * rng.uniform(0.14, 0.17)
            start_s, duration_s, enabled = 0.15, 0.55, True
        elif branch == "contact_failure":
            y = rng.choice((-1.0, 1.0)) * rng.uniform(0.075, 0.095)
            start_s, duration_s, enabled = 0.15, 0.55, True
        else:
            y = rng.choice((-1.0, 1.0)) * rng.uniform(0.3, 0.4)
            start_s, duration_s, enabled = 2.2, 0.8, False
        return {
            "tool_target_m": [0.06, y, 0.04],
            "motion_start_s": start_s,
            "travel_time_s": duration_s,
            "controller_enabled": enabled,
        }

    def physics_variants(
        self, request: GenerationRequest
    ) -> Iterable[tuple[str, Mapping[str, Any]]]:
        yield "nominal", {
            "gravity": physics_field([0.0, 0.0, -9.81], "m/s^2"),
            "mass": physics_field(0.075, "kg"),
            "radius": physics_field(0.040, "m"),
            "rolling_resistance_acceleration": physics_field(0.085, "m/s^2", interpretation="calibrated_effective"),
            "tool_dynamic_friction": physics_field(0.30, "1", interpretation="calibrated_effective"),
            "tool_static_friction": physics_field(None, "1", implemented=False, interpretation="unknown"),
            "tool_restitution": physics_field(0.20, "1", interpretation="calibrated_effective"),
            "simulation_timestep": physics_field(1.0 / request.sim_hz, "s"),
        }

    def simulate(self, plan: EpisodePlan) -> SimulationResult:
        scene, branch = plan.scene_parameters, plan.branch_parameters
        dt = 1.0 / plan.rates_hz["simulation"]
        steps = int(round(plan.duration_s / dt))
        radius = float(physics_value(plan.physics, "radius"))
        tool_radius = float(scene["tool_radius_m"])
        mass = float(physics_value(plan.physics, "mass"))
        resistance = float(physics_value(plan.physics, "rolling_resistance_acceleration"))
        restitution = float(physics_value(plan.physics, "tool_restitution"))
        friction = float(physics_value(plan.physics, "tool_dynamic_friction"))
        p = list(scene["initial_position_m"])
        v = list(scene["initial_velocity_mps"])
        times = [0.0]
        positions = [tuple(p)]
        velocities = [tuple(v)]
        tool_positions = [tuple(scene["tool_start_m"])]
        contacts: list[Mapping[str, Any]] = []
        contacted = False
        for step in range(steps):
            time_s = step * dt
            tool = (
                moving_target_position(
                    time_s + dt,
                    start=tuple(scene["tool_start_m"]),
                    target=tuple(branch["tool_target_m"]),
                    start_time_s=float(branch["motion_start_s"]),
                    travel_time_s=float(branch["travel_time_s"]),
                )
                if branch["controller_enabled"]
                else tuple(scene["tool_start_m"])
            )
            speed = math.hypot(v[0], v[1])
            if speed > 0:
                new_speed = max(0.0, speed - resistance * dt)
                v[0] *= new_speed / speed
                v[1] *= new_speed / speed
            p[0] += v[0] * dt
            p[1] += v[1] * dt
            dx, dy = p[0] - tool[0], p[1] - tool[1]
            separation = math.hypot(dx, dy)
            contact_distance = radius + tool_radius
            if separation < contact_distance and separation > 1e-12:
                nx, ny = dx / separation, dy / separation
                vn = v[0] * nx + v[1] * ny
                if vn < 0:
                    pre = tuple(v)
                    tangent_x, tangent_y = v[0] - vn * nx, v[1] - vn * ny
                    tangent_scale = max(0.0, 1.0 - friction * (1.0 + restitution) * (-vn) / max(math.hypot(tangent_x, tangent_y), 1e-12))
                    v[0] = tangent_x * tangent_scale - restitution * vn * nx
                    v[1] = tangent_y * tangent_scale - restitution * vn * ny
                    # Minimal geometric correction resolves numerical overlap; it
                    # does not impose a post-impact trajectory or desired outcome.
                    p[0] = tool[0] + nx * contact_distance
                    p[1] = tool[1] + ny * contact_distance
                    impulse = [mass * (v[0] - pre[0]), mass * (v[1] - pre[1]), 0.0]
                    contacts.append(
                        {
                            "timestamp": time_s + dt,
                            "object_a": "rolling_ball",
                            "object_b": "robot_tool",
                            "contact_point_m": [tool[0] + nx * tool_radius, tool[1] + ny * tool_radius, radius],
                            "contact_normal": [nx, ny, 0.0],
                            "penetration_depth_m": contact_distance - separation,
                            "normal_impulse_Ns": abs(impulse[0] * nx + impulse[1] * ny),
                            "impulse_Ns": impulse,
                            "relative_velocity_pre_mps": list(pre),
                            "relative_velocity_post_mps": list(v),
                        }
                    )
                    contacted = True
            times.append(time_s + dt)
            positions.append(tuple(p))
            velocities.append(tuple(v))
            tool_positions.append(tool)
        goal = scene["goal_center_m"]
        distances_goal = [math.hypot(position[0] - goal[0], position[1] - goal[1]) for position in positions]
        tool_distances = [math.hypot(position[0] - tool[0], position[1] - tool[1]) for position, tool in zip(positions, tool_positions)]
        final_speed = math.hypot(velocities[-1][0], velocities[-1][1])
        retained = distances_goal[-1] <= float(scene["goal_radius_m"]) and final_speed <= 0.08
        success = contacted and retained
        min_tool_distance = min(tool_distances)
        if success:
            failure = "none"
        elif contacted:
            failure = "object_not_retained_in_goal"
        elif not branch["controller_enabled"]:
            failure = "controller_no_op"
        elif min_tool_distance <= radius + tool_radius + 0.05:
            failure = "tool_near_miss"
        else:
            failure = "tool_missed_rolling_object"
        score = max(0.0, min(1.0, 1.0 - distances_goal[-1] / (2.0 * float(scene["goal_radius_m"]))))
        outcome = OutcomeResult(
            success,
            1.0 if success else score,
            failure,
            {
                "controller_reached_target": bool(branch["controller_enabled"])
                and math.dist(tool_positions[-1], tuple(branch["tool_target_m"])) < 1e-6,
                "object_contacted_tool": contacted,
                "object_entered_receptacle": min(distances_goal) <= float(scene["goal_radius_m"]),
                "object_retained_until_end": retained,
                "final_goal_distance_m": distances_goal[-1],
                "final_object_speed_mps": final_speed,
                "minimum_object_tool_distance_m": min_tool_distance,
            },
            0.95,
        )
        frame_stride = plan.rates_hz["simulation"] // plan.rates_hz["video"]
        frame_indices = range(0, len(times), frame_stride)
        states = [
            {
                "timestamp": times[index],
                "object.position": list(positions[index]),
                "object.linear_velocity": list(velocities[index]),
                "object.angular_velocity": [-velocities[index][1] / radius, velocities[index][0] / radius, 0.0],
                "robot.tool_position": list(tool_positions[index]),
            }
            for index in frame_indices
        ]
        actions = [
            {
                "timestamp": times[index],
                "command.tool_target_position": list(branch["tool_target_m"]),
                "command.enabled": bool(branch["controller_enabled"]),
            }
            for index in frame_indices
        ]
        bounds = scene["table_bounds_m"]
        within_bounds = all(bounds[0] <= p0[0] <= bounds[1] and bounds[2] <= p0[1] <= bounds[3] for p0 in positions)
        finite = all(math.isfinite(value) for row in (*positions, *velocities) for value in row)
        penetration_ok = max(
            (float(event["penetration_depth_m"]) for event in contacts), default=0.0
        ) <= 0.02
        return SimulationResult(
            plan=plan,
            frame_times_s=[times[index] for index in frame_indices],
            states=states,
            actions=actions,
            high_rate_states=[
                {
                    "timestamp": time_s,
                    "object.position": list(position),
                    "object.linear_velocity": list(velocity),
                    "robot.tool_position": list(tool),
                }
                for time_s, position, velocity, tool in zip(times, positions, velocities, tool_positions)
            ],
            contacts=contacts,
            outcome=outcome,
            actual_outcome=classify_actual_outcome(
                success=success,
                contacted=contacted,
                near_distance_m=min_tool_distance,
                near_threshold_m=radius + tool_radius + 0.05,
                bad_action=not branch["controller_enabled"] and plan.intended_branch != "no_op",
                no_op=plan.intended_branch == "no_op",
            ),
            dynamics_mode="free_contact",
            release_tier="free_contact",
            assistance=assistance_record(),
            physics_qc=default_physics_qc(
                finite_state=finite,
                no_post_contact_scripted_reset=True,
                contact_penetration_bounded=penetration_ok,
                table_boundary_status="inside" if within_bounds else "exited_after_task_failure",
                max_contact_penetration_m=max((float(event["penetration_depth_m"]) for event in contacts), default=0.0),
            ),
            simulator={
                "name": "dynamic_robot_dataset.planar_rigid_contact_solver",
                "version": "1",
                "native_mujoco": False,
                "coordinate_frame": "task_local_right_handed_z_up",
                "quaternion_order": "wxyz",
            },
        )
