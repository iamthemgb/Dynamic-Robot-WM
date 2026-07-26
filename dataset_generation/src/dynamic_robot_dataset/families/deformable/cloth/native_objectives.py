"""Trajectory-based objectives for retained native cloth tasks."""

from __future__ import annotations

import math
from typing import Any, Sequence

from ...base import OutcomeResult


Vec3 = Sequence[float]
Mesh = Sequence[Vec3]


def evaluate_native_cloth(
    task: str,
    meshes: Sequence[Mesh],
    *,
    timestamps_s: Sequence[float],
    target_vertex_indices: Sequence[int],
    contacted_target_region: bool,
    equality_released: bool,
    fold_line_x_m: float = 0.0,
) -> OutcomeResult:
    """Recompute cloth success from saved mesh trajectories and task geometry."""

    if len(meshes) < 2 or len(meshes) != len(timestamps_s) or not meshes[0]:
        raise ValueError("cloth objective requires aligned non-empty mesh trajectories")
    count = len(meshes[0])
    if any(len(mesh) != count for mesh in meshes):
        raise ValueError("cloth topology changed during the episode")
    target = tuple(int(index) for index in target_vertex_indices)
    if not target or any(index < 0 or index >= count for index in target):
        raise ValueError("cloth target vertices are missing or invalid")
    initial, final = meshes[0], meshes[-1]
    metrics: dict[str, Any] = {"objective_success": False, "equality_released": equality_released}
    success = False
    score = 0.0
    failure = "cloth_metric_failed"

    if task == "poke_cloth":
        peak = max(
            sum(math.dist(initial[index], mesh[index]) for index in target) / len(target)
            for mesh in meshes
        )
        final_displacement = sum(math.dist(initial[index], final[index]) for index in target) / len(target)
        recovered = final_displacement <= max(0.025, 0.65 * peak)
        success = contacted_target_region and peak >= 0.04 and recovered
        failure = "target_region_not_contacted" if not contacted_target_region else "insufficient_cloth_displacement"
        score = min(1.0, peak / 0.04) * float(recovered)
        metrics.update(
            target_region_contacted=contacted_target_region,
            target_peak_mean_displacement_m=peak,
            target_final_mean_displacement_m=final_displacement,
            recovered_after_retraction=recovered,
        )
    elif task == "lift_corner_release":
        selected = target[0]
        heights = [float(mesh[selected][2]) for mesh in meshes]
        peak = max(heights)
        dwell = sum(
            right - left
            for left, right, height in zip(timestamps_s, timestamps_s[1:], heights[:-1])
            if height >= 0.16
        )
        settled = float(final[selected][2]) <= 0.07
        success = contacted_target_region and peak >= 0.16 and dwell >= 0.15 and equality_released and settled
        failure = (
            "corner_not_grasped" if not contacted_target_region else
            "insufficient_corner_lift" if peak < 0.16 or dwell < 0.15 else
            "corner_not_released"
        )
        score = min(1.0, peak / 0.16) * min(1.0, dwell / 0.15) * float(settled)
        metrics.update(selected_vertex_index=selected, peak_height_m=peak, lifted_dwell_s=dwell, post_release_settled=settled)
    elif task == "fold_edge_fixed_line":
        moving = target
        crossed = sum(float(final[index][0]) < fold_line_x_m - 0.005 for index in moving) / len(moving)
        overlap = sum(
            abs((float(final[index][0]) - fold_line_x_m) + (float(initial[index][0]) - fold_line_x_m)) <= 0.045
            for index in moving
        ) / len(moving)
        persistence_mesh = meshes[max(0, len(meshes) - max(2, len(meshes) // 10))]
        persistent = sum(float(persistence_mesh[index][0]) < fold_line_x_m for index in moving) / len(moving)
        success = contacted_target_region and crossed >= 0.80 and overlap >= 0.70 and persistent >= 0.75 and equality_released
        failure = "fold_edge_not_grasped" if not contacted_target_region else "insufficient_fold_overlap"
        score = min(crossed, overlap, persistent)
        metrics.update(
            fold_line_x_m=fold_line_x_m,
            moving_region_crossed_fraction=crossed,
            mirror_overlap_fraction=overlap,
            persistent_fold_fraction=persistent,
        )
    elif task == "dual_franka_tshirt_fold_box":
        return OutcomeResult.unverified(
            {"task": task, "suspension_reason": "native model and evaluator not validated"},
            "cloth_box_task_suspended",
        )
    else:
        return OutcomeResult.unverified({"task": task}, "native_cloth_evaluator_not_implemented")

    if success:
        failure = "none"
    metrics["objective_success"] = success
    return OutcomeResult(success, 1.0 if success else score, failure, metrics, 1.0, "verified_objective")
