"""Rope metrics and quarantined geometry smoke adapter."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

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
)
from ..proxy import Vec3, centroid, interpolate_points, polyline_length


ROPE_TASKS = (
    "coil_on_table",
    "drag_endpoint",
    "lift_and_drape",
    "shake_wave",
    "sweep",
    "thread_through_ring",
    "tug",
    "twirl_overhead",
    "wrap_around_post",
)


def evaluate_rope(
    task: str,
    initial: Sequence[Vec3],
    final: Sequence[Vec3],
    *,
    snag_flag: bool,
    ring_plane_x_m: float = 0.55,
    ring_center_yz_m: Sequence[float] = (0.0, 0.20),
    ring_radius_m: float = 0.10,
) -> OutcomeResult:
    if len(initial) != len(final) or len(final) < 3:
        raise ValueError("rope evaluator requires equal point arrays")
    start_centroid, final_centroid = centroid(initial), centroid(final)
    endpoint_displacement = math.dist(initial[-1], final[-1])
    metrics: dict[str, Any]
    if task == "coil_on_table":
        center = centroid(final)
        radii = [math.hypot(point[0] - center[0], point[1] - center[1]) for point in final]
        mean_radius = sum(radii) / len(radii)
        radial_std = math.sqrt(sum((value - mean_radius) ** 2 for value in radii) / len(radii))
        table_fraction = sum(point[2] <= 0.04 for point in final) / len(final)
        base_success = 0.07 <= mean_radius <= 0.18 and radial_std <= 0.045 and table_fraction >= 0.9
        score = table_fraction * max(0.0, 1.0 - radial_std / 0.08)
        failure = "coil_geometry_not_formed"
        metrics = {"mean_coil_radius_m": mean_radius, "coil_radial_std_m": radial_std, "on_table_fraction": table_fraction}
    elif task == "drag_endpoint":
        displacement_xy = math.hypot(final[-1][0] - initial[-1][0], final[-1][1] - initial[-1][1])
        base_success = displacement_xy >= 0.22 and final[-1][2] <= 0.08
        score = min(1.0, displacement_xy / 0.22)
        failure = "endpoint_not_dragged_to_target"
        metrics = {"endpoint_xy_displacement_m": displacement_xy, "final_endpoint_height_m": final[-1][2]}
    elif task == "lift_and_drape":
        high_fraction = sum(point[2] >= 0.18 for point in final) / len(final)
        sides = any(point[0] < -0.08 for point in final) and any(point[0] > 0.08 for point in final)
        base_success = max(point[2] for point in final) >= 0.30 and high_fraction >= 0.2 and sides
        score = min(1.0, max(point[2] for point in final) / 0.3) * min(1.0, high_fraction / 0.2)
        failure = "rope_not_draped_over_support"
        metrics = {"maximum_height_m": max(point[2] for point in final), "high_segment_fraction": high_fraction, "segments_on_both_sides": sides}
    elif task == "shake_wave":
        y_values = [point[1] for point in final]
        amplitude = (max(y_values) - min(y_values)) / 2
        sign_changes = sum((a * b) < 0 for a, b in zip(y_values, y_values[1:]))
        base_success = amplitude >= 0.08 and sign_changes >= 2
        score = min(1.0, amplitude / 0.08) * min(1.0, sign_changes / 2)
        failure = "wave_amplitude_or_frequency_too_low"
        metrics = {"lateral_wave_amplitude_m": amplitude, "lateral_sign_changes": sign_changes}
    elif task == "sweep":
        displacement_y = final_centroid[1] - start_centroid[1]
        base_success = abs(displacement_y) >= 0.18
        score = min(1.0, abs(displacement_y) / 0.18)
        failure = "insufficient_rope_sweep"
        metrics = {"centroid_y_displacement_m": displacement_y}
    elif task == "thread_through_ring":
        endpoint = final[-1]
        crossed = endpoint[0] >= ring_plane_x_m + 0.04
        radial = math.hypot(endpoint[1] - ring_center_yz_m[0], endpoint[2] - ring_center_yz_m[1])
        # At least one segment must straddle the plane, proving passage rather
        # than an endpoint that began on the target side.
        straddled = any((a[0] - ring_plane_x_m) * (b[0] - ring_plane_x_m) <= 0 for a, b in zip(final, final[1:]))
        base_success = crossed and straddled and radial <= ring_radius_m
        score = float(crossed) * max(0.0, 1.0 - radial / (2.0 * ring_radius_m))
        failure = "rope_did_not_cross_and_remain_beyond_ring"
        metrics = {"endpoint_crossed_ring_plane": crossed, "rope_straddles_ring_plane": straddled, "endpoint_ring_radial_error_m": radial}
    elif task == "tug":
        endpoint_separation = math.dist(final[0], final[-1])
        stretch_ratio = endpoint_separation / max(polyline_length(initial), 1e-9)
        base_success = endpoint_displacement >= 0.18 and stretch_ratio >= 0.85
        score = min(1.0, endpoint_displacement / 0.18) * min(1.0, stretch_ratio / 0.85)
        failure = "insufficient_tug_displacement"
        metrics = {"endpoint_displacement_m": endpoint_displacement, "endpoint_separation_ratio": stretch_ratio}
    elif task == "twirl_overhead":
        elevated_fraction = sum(point[2] >= 0.30 for point in final) / len(final)
        xy_span = max(math.hypot(point[0] - final_centroid[0], point[1] - final_centroid[1]) for point in final)
        base_success = elevated_fraction >= 0.8 and xy_span >= 0.16
        score = elevated_fraction * min(1.0, xy_span / 0.16)
        failure = "rope_not_twirl_elevated"
        metrics = {"elevated_fraction": elevated_fraction, "horizontal_radial_span_m": xy_span}
    elif task == "wrap_around_post":
        angles = [math.atan2(point[1], point[0]) for point in final]
        unwrapped = [angles[0]]
        for angle in angles[1:]:
            delta = (angle - unwrapped[-1] + math.pi) % (2 * math.pi) - math.pi
            unwrapped.append(unwrapped[-1] + delta)
        turns = abs(unwrapped[-1] - unwrapped[0]) / (2 * math.pi)
        radial = sum(math.hypot(point[0], point[1]) for point in final) / len(final)
        base_success = turns >= 0.9 and 0.07 <= radial <= 0.16
        score = min(1.0, turns / 0.9) * max(0.0, 1.0 - abs(radial - 0.11) / 0.11)
        failure = "insufficient_wrap_around_post"
        metrics = {"wrap_turns": turns, "mean_post_radial_distance_m": radial}
    else:
        return OutcomeResult.unverified({"task": task}, "rope_metric_not_implemented")
    success = base_success and not snag_flag
    if success:
        failure = "none"
    elif snag_flag:
        failure = "unintended_snag"
    metrics.update({"snag_flag": snag_flag, "base_task_metric_pass": base_success})
    return OutcomeResult(success, 1.0 if success else score, failure, metrics, 0.55, "unverified")


class RopeAdapter(FamilyAdapter):
    family = "rope"
    supported_subfamilies = ("default",) + ROPE_TASKS
    default_duration_s = 5.0

    def scene_parameters(self, request: GenerationRequest, scene_seed: int, scene_index: int) -> Mapping[str, Any]:
        task = request.subfamily if request.subfamily != "default" else "drag_endpoint"
        return {
            "task": task,
            "length_m": 0.80,
            "segment_count": 24,
            "initial_z_m": 0.018,
            "ring_plane_x_m": 0.55,
            "ring_center_yz_m": [0.0, 0.20],
            "ring_radius_m": 0.10,
            "visual_seed": scene_seed,
            "background_style": request.scene_style,
        }

    def branch_parameters(self, request: GenerationRequest, branch: str, branch_seed: int, scene: Mapping[str, Any]) -> Mapping[str, Any]:
        magnitude = {"success_seeking": 1.0, "near_miss": 0.58, "contact_failure": 0.75, "bad_action": 0.0}.get(branch, 0.0)
        return {
            "control_magnitude": magnitude,
            "endpoint_attachment": branch != "bad_action",
            "snag_injected_for_qc": branch == "contact_failure",
        }

    def physics_variants(self, request: GenerationRequest) -> Iterable[tuple[str, Mapping[str, Any]]]:
        yield "proxy_nominal", {
            "length": physics_field(0.80, "m", interpretation="proxy"),
            "radius": physics_field(0.008, "m", interpretation="proxy"),
            "density": physics_field(0.025, "kg/m", interpretation="proxy"),
            "stretch_stiffness": physics_field(400.0, "N/m", interpretation="proxy"),
            "bend_stiffness": physics_field(0.015, "N*m", interpretation="proxy"),
            "damping": physics_field(0.10, "1", interpretation="proxy"),
            "rope_table_friction": physics_field(0.40, "1", interpretation="proxy"),
            "rope_tool_friction": physics_field(0.55, "1", interpretation="proxy"),
            "segment_count": physics_field(24.0, "count", interpretation="proxy"),
        }

    @staticmethod
    def _initial(scene: Mapping[str, Any]) -> list[Vec3]:
        count = int(scene["segment_count"]) + 1
        return [(-scene["length_m"] / 2 + scene["length_m"] * index / (count - 1), 0.0, scene["initial_z_m"]) for index in range(count)]

    @staticmethod
    def _final(task: str, initial: Sequence[Vec3], magnitude: float) -> list[Vec3]:
        count = len(initial)
        if magnitude == 0:
            return list(initial)
        if task == "coil_on_table":
            return [(0.12 * math.cos(3.0 * math.pi * i / (count - 1)) * magnitude + x * (1 - magnitude), 0.12 * math.sin(3.0 * math.pi * i / (count - 1)) * magnitude, z) for i, (x, _, z) in enumerate(initial)]
        if task == "drag_endpoint":
            return [(x + 0.30 * magnitude * (i / (count - 1)) ** 2, y + 0.03 * magnitude * i / (count - 1), z) for i, (x, y, z) in enumerate(initial)]
        if task == "lift_and_drape":
            return [(x * 0.55, y, z + 0.36 * magnitude * math.sin(math.pi * i / (count - 1))) for i, (x, y, z) in enumerate(initial)]
        if task == "shake_wave":
            return [(x, 0.12 * magnitude * math.sin(4 * math.pi * i / (count - 1)), z) for i, (x, _, z) in enumerate(initial)]
        if task == "sweep":
            return [(x, y + 0.24 * magnitude, z) for x, y, z in initial]
        if task == "thread_through_ring":
            return [(x + 0.25 * magnitude * (i / (count - 1)) ** 3, y, z + 0.182 * magnitude * (i / (count - 1)) ** 3) for i, (x, y, z) in enumerate(initial)]
        if task == "tug":
            return [(x + 0.22 * magnitude * (2 * i / (count - 1) - 1), y, z) for i, (x, y, z) in enumerate(initial)]
        if task == "twirl_overhead":
            return [(0.20 * magnitude * math.cos(2 * math.pi * i / (count - 1)) + x * (1 - magnitude), 0.20 * magnitude * math.sin(2 * math.pi * i / (count - 1)), z + 0.38 * magnitude) for i, (x, _, z) in enumerate(initial)]
        if task == "wrap_around_post":
            return [(0.11 * math.cos(2.4 * math.pi * magnitude * i / (count - 1)), 0.11 * math.sin(2.4 * math.pi * magnitude * i / (count - 1)), z + 0.01 * i / (count - 1)) for i, (_, _, z) in enumerate(initial)]
        return list(initial)

    def simulate(self, plan: EpisodePlan) -> SimulationResult:
        scene, branch = plan.scene_parameters, plan.branch_parameters
        initial = self._initial(scene)
        final = self._final(scene["task"], initial, float(branch["control_magnitude"]))
        snag = bool(branch["snag_injected_for_qc"])
        outcome = evaluate_rope(
            scene["task"], initial, final, snag_flag=snag,
            ring_plane_x_m=float(scene["ring_plane_x_m"]),
            ring_center_yz_m=scene["ring_center_yz_m"], ring_radius_m=float(scene["ring_radius_m"]),
        )
        count = int(round(plan.duration_s * plan.rates_hz["video"])) + 1
        times = [index / plan.rates_hz["video"] for index in range(count)]
        geometries = [interpolate_points(initial, final, t / plan.duration_s) for t in times]
        states = [
            {
                "timestamp": time_s,
                "rope.points": [list(point) for point in points],
                "rope.centroid": list(centroid(points)),
                "rope.endpoint": list(points[-1]),
                "rope.snag_flag": snag,
                "assistance.active": bool(branch["endpoint_attachment"]),
            }
            for time_s, points in zip(times, geometries)
        ]
        actions = [
            {
                "timestamp": time_s,
                "command.task": scene["task"],
                "command.magnitude": branch["control_magnitude"],
                "command.endpoint_attachment": branch["endpoint_attachment"],
            }
            for time_s in times
        ]
        contacted = bool(branch["endpoint_attachment"])
        return SimulationResult(
            plan=plan, frame_times_s=times, states=states, actions=actions,
            high_rate_states=[{"timestamp": row["timestamp"], "rope.endpoint": row["rope.endpoint"]} for row in states],
            contacts=(), outcome=outcome,
            actual_outcome=classify_actual_outcome(
                success=outcome.task_success, contacted=contacted,
                near_distance_m=0.0 if contacted else 1.0, near_threshold_m=0.1,
                bad_action=float(branch["control_magnitude"]) == 0.0,
            ),
            dynamics_mode="scripted_motion", release_tier="scripted_motion",
            assistance=assistance_record(
                assisted_grasp=bool(branch["endpoint_attachment"]),
                equality_constraint_active=bool(branch["endpoint_attachment"]),
                activation_time_s=0.0 if branch["endpoint_attachment"] else None,
                deactivation_time_s=plan.duration_s if branch["endpoint_attachment"] else None,
            ),
            physics_qc=default_physics_qc(
                finite_state=all(math.isfinite(value) for geometry in geometries for point in geometry for value in point),
                native_deformable_physics=False,
                no_unintended_snag=not snag,
            ),
            simulator={"name": "dynamic_robot_dataset.quarantined_rope_geometry_proxy", "version": "1", "native_mujoco_or_flex": False, "production_eligible": False},
            notes=("Synthetic rope geometry tests schemas and objective metrics only; it is not rope training data.",),
        )
