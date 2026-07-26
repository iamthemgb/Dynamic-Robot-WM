"""Production rope constraint binding and trajectory-based objectives.

This module intentionally does not import the immutable legacy generator.  It
provides the corrected contract used by a cleaned native implementation: the
equality body is the task-requested rope segment, not a hard-coded endpoint.
"""

from __future__ import annotations

import html
import math
import re
from typing import Any, Mapping, Sequence

from ...base import OutcomeResult


Vec3 = Sequence[float]
Centerline = Sequence[Vec3]
Trajectory = Sequence[Centerline]
_ROPE_BODY = re.compile(r"^RB_(?:first|last|[1-9][0-9]*)$")


def validate_rope_grasp_target(requested_body: str, compiled_body: str) -> None:
    """Reject a task/scene equality mismatch such as RB_6 versus RB_first."""

    if not _ROPE_BODY.fullmatch(requested_body):
        raise ValueError(f"invalid rope grasp body: {requested_body!r}")
    if requested_body != compiled_body:
        raise ValueError(
            f"rope equality target mismatch: task requests {requested_body}, "
            f"compiled constraint targets {compiled_body}"
        )


def grasp_equality_xml(requested_body: str, *, hand_body: str = "A_hand") -> str:
    """Build the inactive equality using the exact task-selected segment."""

    if not _ROPE_BODY.fullmatch(requested_body):
        raise ValueError(f"invalid rope grasp body: {requested_body!r}")
    return (
        '<connect name="grasp_endpoint" '
        f'body1="{html.escape(requested_body, quote=True)}" '
        f'body2="{html.escape(hand_body, quote=True)}" anchor="0 0 0" '
        'active="false" solref="0.01 1"/>'
    )


def _distance(left: Vec3, right: Vec3) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _validate_trajectory(trajectory: Trajectory) -> None:
    if len(trajectory) < 2 or not trajectory[0]:
        raise ValueError("rope objective requires at least two non-empty centerlines")
    count = len(trajectory[0])
    if any(len(frame) != count for frame in trajectory):
        raise ValueError("rope topology changed during the episode")
    if any(
        len(point) != 3 or any(not math.isfinite(float(value)) for value in point)
        for frame in trajectory
        for point in frame
    ):
        raise ValueError("rope trajectory contains invalid points")


def evaluate_native_rope(
    task: str,
    trajectory: Trajectory,
    *,
    timestamps_s: Sequence[float],
    released: bool,
    snag_reason: str | None,
    target_world_m: Vec3 | None = None,
    ring_center_world_m: Vec3 | None = None,
    ring_inner_radius_m: float | None = None,
    free_tail_index: int = 0,
    post_center_world_m: Vec3 | None = None,
    bar_axis_x_m: float | None = None,
    bar_height_m: float | None = None,
) -> OutcomeResult:
    """Evaluate retained rope tasks from full native centerline evidence."""

    _validate_trajectory(trajectory)
    if len(timestamps_s) != len(trajectory) or any(
        right <= left for left, right in zip(timestamps_s, timestamps_s[1:])
    ):
        raise ValueError("rope timestamps must align and increase")
    initial, final = trajectory[0], trajectory[-1]
    endpoint_path = sum(_distance(left[-1], right[-1]) for left, right in zip(trajectory, trajectory[1:]))
    metrics: dict[str, Any] = {
        "objective_success": False,
        "released": released,
        "snag_reason": snag_reason,
        "endpoint_path_length_m": endpoint_path,
        "topology_point_count": len(initial),
    }
    base_success = False
    failure = "rope_metric_failed"
    score = 0.0

    if task == "drag_endpoint":
        if target_world_m is None:
            raise ValueError("drag_endpoint requires target_world_m")
        final_error = _distance(final[-1], target_world_m)
        max_height = max(float(frame[-1][2]) for frame in trajectory)
        displacement = _distance(initial[-1], final[-1])
        base_success = displacement >= 0.15 and final_error <= 0.06 and max_height <= 0.10 and released
        failure = "endpoint_not_dragged_to_target"
        score = min(1.0, displacement / 0.15) * max(0.0, 1.0 - final_error / 0.15)
        metrics.update(
            endpoint_displacement_m=displacement,
            final_target_error_m=final_error,
            maximum_endpoint_height_m=max_height,
        )
    elif task in {"tug", "tug_endpoint"}:
        heights = [float(frame[-1][2]) for frame in trajectory]
        peak = max(heights)
        dwell = sum(
            right - left
            for left, right, height in zip(timestamps_s, timestamps_s[1:], heights[:-1])
            if height >= 0.15
        )
        initial_length = sum(_distance(a, b) for a, b in zip(initial, initial[1:]))
        maximum_length = max(
            sum(_distance(a, b) for a, b in zip(frame, frame[1:])) for frame in trajectory
        )
        stretch_ratio = maximum_length / max(initial_length, 1e-9)
        base_success = peak >= 0.15 and dwell >= 0.20 and stretch_ratio <= 1.15 and released
        failure = "insufficient_tug_displacement"
        score = min(1.0, peak / 0.15) * min(1.0, dwell / 0.20)
        metrics.update(
            peak_endpoint_height_m=peak,
            lifted_dwell_s=dwell,
            maximum_centerline_stretch_ratio=stretch_ratio,
        )
    elif task == "thread_through_ring":
        if ring_center_world_m is None or ring_inner_radius_m is None:
            raise ValueError("thread_through_ring requires ring geometry")
        index = free_tail_index if free_tail_index >= 0 else len(initial) + free_tail_index
        if not 0 <= index < len(initial):
            raise ValueError("free_tail_index lies outside centerline")
        center = tuple(float(value) for value in ring_center_world_m)
        crossings = 0
        for left, right in zip(trajectory, trajectory[1:]):
            z0 = float(left[index][2]) - center[2]
            z1 = float(right[index][2]) - center[2]
            if z0 > 0 >= z1:
                radial = math.hypot(float(right[index][0]) - center[0], float(right[index][1]) - center[1])
                crossings += int(radial <= ring_inner_radius_m)
        final_below = float(final[index][2]) < center[2]
        final_straddle = min(float(point[2]) for point in final) < center[2] < max(float(point[2]) for point in final)
        radial_final = math.hypot(float(final[index][0]) - center[0], float(final[index][1]) - center[1])
        base_success = crossings > 0 and final_below and final_straddle and radial_final <= ring_inner_radius_m and released
        failure = "rope_did_not_cross_and_remain_beyond_ring"
        score = float(crossings > 0) * max(0.0, 1.0 - radial_final / (2.0 * ring_inner_radius_m))
        metrics.update(
            aperture_crossing_count=crossings,
            free_tail_final_below_ring=final_below,
            full_rope_straddles_ring_plane=final_straddle,
            free_tail_final_radial_error_m=radial_final,
        )
    elif task == "wrap_around_post":
        if post_center_world_m is None:
            raise ValueError("wrap_around_post requires post center")
        angles = [
            math.atan2(float(point[1]) - post_center_world_m[1], float(point[0]) - post_center_world_m[0])
            for point in final
        ]
        unwrapped = [angles[0]]
        for angle in angles[1:]:
            delta = (angle - unwrapped[-1] + math.pi) % (2.0 * math.pi) - math.pi
            unwrapped.append(unwrapped[-1] + delta)
        turns = abs(unwrapped[-1] - unwrapped[0]) / (2.0 * math.pi)
        base_success = turns >= 0.45 and released
        failure = "insufficient_wrap_around_post"
        score = min(1.0, turns / 0.45)
        metrics["centerline_winding_turns"] = turns
    elif task == "lift_and_drape":
        if bar_axis_x_m is None or bar_height_m is None:
            raise ValueError("lift_and_drape requires bar geometry")
        above = max(float(point[2]) for frame in trajectory for point in frame) > bar_height_m
        sides = any(float(point[1]) < -0.03 for point in final) and any(float(point[1]) > 0.03 for point in final)
        supported = min(_distance(point, (bar_axis_x_m, float(point[1]), bar_height_m)) for point in final) <= 0.04
        base_success = above and sides and supported and released
        failure = "rope_not_draped_over_support"
        score = (float(above) + float(sides) + float(supported) + float(released)) / 4.0
        metrics.update(went_above_bar=above, endpoints_on_opposite_sides=sides, bar_support_present=supported)
    else:
        return OutcomeResult.unverified({"task": task}, "native_rope_evaluator_not_implemented")

    success = base_success and snag_reason is None
    if success:
        failure = "none"
    elif snag_reason is not None:
        failure = "unintended_snag"
    metrics["objective_success"] = success
    return OutcomeResult(success, 1.0 if success else score, failure, metrics, 1.0, "verified_objective")
