"""Cloth task metrics plus a quarantined deterministic geometry smoke adapter."""

from __future__ import annotations

import math
import random
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
from ..proxy import Vec3, centroid, grid_points, interpolate_points


CLOTH_TASKS = (
    "poke_cloth",
    "lift_corner_release",
    "fold_edge_fixed_line",
    "dual_franka_tshirt_fold_box",
)


def evaluate_cloth(
    task: str,
    initial: Sequence[Vec3],
    final: Sequence[Vec3],
    *,
    contacted_target_region: bool,
    max_corner_height_m: float = 0.0,
    box_bounds_m: Sequence[float] = (-0.18, 0.18, -0.16, 0.16),
) -> OutcomeResult:
    if len(initial) != len(final) or not initial:
        raise ValueError("cloth evaluator requires equal non-empty vertex arrays")
    metrics: dict[str, Any]
    if task == "poke_cloth":
        target_indices = [index for index, point in enumerate(initial) if math.hypot(point[0], point[1]) <= 0.11]
        displacement = sum(math.dist(initial[index], final[index]) for index in target_indices) / max(len(target_indices), 1)
        success = contacted_target_region and displacement >= 0.04
        score = min(1.0, displacement / 0.04) * (1.0 if contacted_target_region else 0.25)
        failure = "none" if success else ("target_region_not_contacted" if not contacted_target_region else "insufficient_cloth_displacement")
        metrics = {
            "target_region_contacted": contacted_target_region,
            "target_region_mean_displacement_m": displacement,
            "displacement_threshold_m": 0.04,
        }
    elif task == "lift_corner_release":
        corner = max(range(len(initial)), key=lambda index: initial[index][0] + initial[index][1])
        target_xy = (-0.16, -0.12)
        placement_error = math.hypot(final[corner][0] - target_xy[0], final[corner][1] - target_xy[1])
        lifted = max_corner_height_m >= 0.16
        released = final[corner][2] <= 0.06
        success = contacted_target_region and lifted and released and placement_error <= 0.07
        score = min(1.0, max_corner_height_m / 0.16) * max(0.0, 1.0 - placement_error / 0.2)
        failure = "none" if success else (
            "corner_not_grasped" if not contacted_target_region else
            "insufficient_corner_lift" if not lifted else
            "corner_not_released" if not released else
            "corner_placement_error"
        )
        metrics = {
            "correct_corner_contacted": contacted_target_region,
            "max_corner_height_m": max_corner_height_m,
            "corner_released": released,
            "final_corner_placement_error_m": placement_error,
        }
    elif task == "fold_edge_fixed_line":
        moving = [index for index, point in enumerate(initial) if point[0] > 0.02]
        crossed = sum(final[index][0] < -0.005 for index in moving) / max(len(moving), 1)
        alignment = sum(abs(final[index][0] + initial[index][0]) <= 0.045 for index in moving) / max(len(moving), 1)
        success = contacted_target_region and crossed >= 0.80 and alignment >= 0.70
        score = min(crossed, alignment)
        failure = "none" if success else ("fold_edge_not_grasped" if not contacted_target_region else "insufficient_fold_overlap")
        metrics = {
            "moving_region_crossed_fraction": crossed,
            "mirror_alignment_fraction": alignment,
            "fold_line_x_m": 0.0,
        }
    elif task == "dual_franka_tshirt_fold_box":
        x0, x1, y0, y1 = box_bounds_m
        inside = sum(x0 <= point[0] <= x1 and y0 <= point[1] <= y1 for point in final) / len(final)
        outside_distance = sum(
            max(x0 - point[0], 0.0, point[0] - x1) + max(y0 - point[1], 0.0, point[1] - y1)
            for point in final
        ) / len(final)
        success = contacted_target_region and inside >= 0.85 and outside_distance <= 0.015
        score = inside * max(0.0, 1.0 - outside_distance / 0.1)
        failure = "none" if success else ("cloth_not_grasped" if not contacted_target_region else "cloth_spill_outside_box")
        metrics = {
            "cloth_vertex_fraction_inside_box": inside,
            "mean_spill_distance_m": outside_distance,
            "box_bounds_m": list(box_bounds_m),
        }
    else:
        return OutcomeResult.unverified({"task": task}, "cloth_metric_not_implemented")
    # Metrics are objective, but this adapter's synthetic motion is quarantined;
    # native simulation adapters may upgrade label_status after validation.
    return OutcomeResult(success, 1.0 if success else score, failure, metrics, 0.55, "unverified")


class ClothAdapter(FamilyAdapter):
    family = "cloth"
    supported_subfamilies = ("default",) + CLOTH_TASKS
    default_duration_s = 5.0

    def scene_parameters(self, request: GenerationRequest, scene_seed: int, scene_index: int) -> Mapping[str, Any]:
        rng = random.Random(scene_seed)
        task = request.subfamily if request.subfamily != "default" else "poke_cloth"
        return {
            "task": task,
            "mesh_shape": [7, 7],
            "width_m": 0.52,
            "height_m": 0.42,
            "initial_z_m": 0.025,
            "yaw_rad": rng.uniform(-0.08, 0.08),
            "box_bounds_m": [-0.18, 0.18, -0.16, 0.16],
            "visual_seed": scene_seed,
            "background_style": request.scene_style,
        }

    def branch_parameters(self, request: GenerationRequest, branch: str, branch_seed: int, scene: Mapping[str, Any]) -> Mapping[str, Any]:
        magnitude = {
            "success_seeking": 1.0,
            "near_miss": 0.65,
            "contact_failure": 0.25,
            "bad_action": 0.0,
        }.get(branch, 0.0)
        return {
            "control_magnitude": magnitude,
            "contacted_target_region": branch in {"success_seeking", "contact_failure"},
            "attachment_enabled": scene["task"] != "poke_cloth" and branch != "bad_action",
            "release_fraction": 0.72,
        }

    def physics_variants(self, request: GenerationRequest) -> Iterable[tuple[str, Mapping[str, Any]]]:
        yield "proxy_nominal", {
            "stretch_stiffness": physics_field(650.0, "N/m", interpretation="proxy"),
            "bend_stiffness": physics_field(0.018, "N*m", interpretation="proxy"),
            "shear_stiffness": physics_field(220.0, "N/m", interpretation="proxy"),
            "density": physics_field(0.22, "kg/m^2", interpretation="proxy"),
            "thickness": physics_field(0.0012, "m", interpretation="proxy"),
            "cloth_table_friction": physics_field(0.45, "1", interpretation="proxy"),
            "cloth_tool_friction": physics_field(0.60, "1", interpretation="proxy"),
            "damping": physics_field(0.08, "1", interpretation="proxy"),
            "mesh_resolution": physics_field([7, 7], "vertices", interpretation="proxy"),
        }

    @staticmethod
    def _final_geometry(task: str, initial: Sequence[Vec3], magnitude: float, target_contact: bool) -> tuple[list[Vec3], float]:
        final = list(initial)
        max_corner_height = 0.0
        if task == "poke_cloth":
            center = (0.0, 0.0) if target_contact else (0.22, 0.16)
            final = [
                (x, y, z + 0.07 * magnitude * math.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / 0.012))
                for x, y, z in initial
            ]
        elif task == "lift_corner_release":
            corner = max(range(len(initial)), key=lambda index: initial[index][0] + initial[index][1])
            destination = (-0.16, -0.12, 0.025)
            if target_contact:
                final[corner] = tuple(initial[corner][axis] + magnitude * (destination[axis] - initial[corner][axis]) for axis in range(3))
                max_corner_height = 0.20 * magnitude
        elif task == "fold_edge_fixed_line":
            if target_contact:
                final = [((-x * magnitude + x * (1.0 - magnitude)) if x > 0 else x, y, z + (0.006 if x > 0 else 0.0)) for x, y, z in initial]
        elif task == "dual_franka_tshirt_fold_box":
            if target_contact:
                final = [(x * (1.0 - 0.48 * magnitude), y * (1.0 - 0.38 * magnitude), z + 0.008 * magnitude) for x, y, z in initial]
        return final, max_corner_height

    def simulate(self, plan: EpisodePlan) -> SimulationResult:
        scene, branch = plan.scene_parameters, plan.branch_parameters
        initial = grid_points(*scene["mesh_shape"], scene["width_m"], scene["height_m"], scene["initial_z_m"])
        final, max_corner_height = self._final_geometry(scene["task"], initial, float(branch["control_magnitude"]), bool(branch["contacted_target_region"]))
        outcome = evaluate_cloth(
            scene["task"],
            initial,
            final,
            contacted_target_region=bool(branch["contacted_target_region"]),
            max_corner_height_m=max_corner_height,
            box_bounds_m=scene["box_bounds_m"],
        )
        frames = int(round(plan.duration_s * plan.rates_hz["video"])) + 1
        frame_times = [index / plan.rates_hz["video"] for index in range(frames)]
        meshes = [interpolate_points(initial, final, time_s / plan.duration_s) for time_s in frame_times]
        states = [
            {
                "timestamp": time_s,
                "cloth.vertices": [list(point) for point in mesh],
                "cloth.centroid": list(centroid(mesh)),
                "assistance.active": bool(branch["attachment_enabled"] and time_s / plan.duration_s <= branch["release_fraction"]),
            }
            for time_s, mesh in zip(frame_times, meshes)
        ]
        actions = [
            {
                "timestamp": time_s,
                "command.task": scene["task"],
                "command.magnitude": branch["control_magnitude"],
                "command.attachment": bool(branch["attachment_enabled"] and time_s / plan.duration_s <= branch["release_fraction"]),
            }
            for time_s in frame_times
        ]
        contacted = bool(branch["contacted_target_region"])
        return SimulationResult(
            plan=plan,
            frame_times_s=frame_times,
            states=states,
            actions=actions,
            high_rate_states=[{"timestamp": row["timestamp"], "cloth.centroid": row["cloth.centroid"]} for row in states],
            contacts=(),
            outcome=outcome,
            actual_outcome=classify_actual_outcome(
                success=outcome.task_success,
                contacted=contacted,
                near_distance_m=0.0 if contacted else 1.0,
                near_threshold_m=0.1,
                bad_action=float(branch["control_magnitude"]) == 0.0 and plan.intended_branch != "no_op",
                no_op=plan.intended_branch == "no_op",
            ),
            dynamics_mode="scripted_motion",
            release_tier="scripted_motion",
            assistance=assistance_record(
                assisted_grasp=bool(branch["attachment_enabled"]),
                equality_constraint_active=bool(branch["attachment_enabled"]),
                activation_time_s=0.0 if branch["attachment_enabled"] else None,
                deactivation_time_s=plan.duration_s * float(branch["release_fraction"]) if branch["attachment_enabled"] else None,
                mechanism_id=("diagnostic-cloth-attachment" if branch["attachment_enabled"] else None),
                mechanism_type=("equality_constraint" if branch["attachment_enabled"] else None),
                constraint_ids=("diagnostic_cloth_attachment",) if branch["attachment_enabled"] else (),
                target_element_ids=("cloth_attachment_vertex",) if branch["attachment_enabled"] else (),
            ),
            physics_qc=default_physics_qc(
                finite_state=all(math.isfinite(value) for mesh in meshes for point in mesh for value in point),
                native_deformable_physics=False,
            ),
            simulator={
                "name": "dynamic_robot_dataset.quarantined_cloth_geometry_proxy",
                "version": "1",
                "native_mujoco_or_flex": False,
                "production_eligible": False,
            },
            notes=("Synthetic mesh motion tests schemas and objective metrics only; it is not cloth training data.",),
        )
