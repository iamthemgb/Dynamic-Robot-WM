"""Versioned objective recomputation from persisted native state/event rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..base import NativeFamily, RigidScenario, ScenarioSpec, ToolKind
from ...families.base import OutcomeResult, stable_hash


EVALUATOR_ID = "native_rigid_state_event"
EVALUATOR_VERSION = "1.2.0"
OBJECTIVE_THRESHOLDS = {
    "retention_dwell_s": 0.35,
    "transport_terminal_error_m": 0.08,
    "brake_retention_fraction": 0.95,
    "tilt_minimum_rad": 0.12,
    "edge_excursion_tool_fraction": 0.45,
    "edge_recovery_tool_fraction": 0.60,
    "rolling_block_speed_m_s": 0.28,
    "goal_radius_m": 0.24,
    "projectile_reverse_speed_m_s": 0.10,
    "near_tool_distance_m": 0.28,
    # A wrong action is measured against an immutable, branch-independent
    # task target.  It is not inferred from the requested branch label.  Keep
    # this comfortably above the 0.22 m near-miss perturbation used by the
    # native planners, and require substantial measured robot motion so a
    # held/no-op controller cannot be misclassified as a wrong action.
    "wrong_action_target_error_m": 0.30,
    "wrong_action_minimum_tool_displacement_m": 0.08,
}
OBJECTIVE_THRESHOLD_SET_HASH = stable_hash(OBJECTIVE_THRESHOLDS)


@dataclass(frozen=True)
class ObjectiveEvaluation:
    outcome: OutcomeResult
    actual_outcome_class: str
    evaluator_id: str
    evaluator_version: str
    threshold_set_hash: str
    evidence: Mapping[str, Any]


def _vector(row: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float64)


def _rotation_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("saved tool quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _inside_tool(spec: ScenarioSpec, row: Mapping[str, Any]) -> bool:
    if spec.tool.kind in {ToolKind.FLAT_PADDLE, ToolKind.ANGLED_PADDLE}:
        return False
    relative = _vector(row, "object.position") - _vector(row, "robot.tool_position")
    local = _rotation_wxyz(row["robot.tool_quaternion_wxyz"]).T @ relative
    hx, hy, _hz = spec.tool.half_extents_m
    return bool(
        abs(float(local[0])) <= hy
        and abs(float(local[1])) <= hx
        and -(spec.tool.wall_height_m + 2.0 * spec.object.radius_m)
        <= float(local[2])
        <= spec.object.radius_m
    )


def _tool_relative(row: Mapping[str, Any]) -> np.ndarray:
    relative = _vector(row, "object.position") - _vector(
        row, "robot.tool_position"
    )
    return _rotation_wxyz(row["robot.tool_quaternion_wxyz"]).T @ relative


def _ordered(expected: Sequence[str], observed: Sequence[str]) -> bool:
    iterator = iter(observed)
    return all(any(value == target for value in iterator) for target in expected)


def _transition_surface_release_evidence(
    spec: ScenarioSpec,
    events: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    *,
    surface: str,
    terminal_time_s: float,
) -> dict[str, Any]:
    """Score one release from solver-rate, content-validated transitions."""

    surface_times = [
        float(event["timestamp"])
        for event in events
        if event.get("object_b") == surface
    ]
    release_x_raw = spec.extras.get("transition_release_x_m")
    direction = int(spec.extras.get("transition_direction", 1))
    minimum_dwell = float(
        spec.extras.get("transition_minimum_free_flight_dwell_s", 0.06)
    )
    first_surface = min(surface_times, default=None)
    base = {
        "passed": False,
        "surface_contact_time_s": first_surface,
        "free_flight_start_time_s": None,
        "free_flight_dwell_s": 0.0,
        "release_x_m": release_x_raw,
        "direction": direction,
        "minimum_dwell_s": minimum_dwell,
        "evidence_source": "persisted_transition_rows",
    }
    if first_surface is None or release_x_raw is None or direction not in {-1, 1}:
        return base
    release_x = float(release_x_raw)
    ordered = sorted(
        (dict(row) for row in transitions),
        key=lambda row: float(row.get("timestamp", 0.0)),
    )

    def valid_boundary(row: Mapping[str, Any]) -> bool:
        try:
            position = row["object_position_m"]
            return bool(
                row.get("event_type") == "surface_release_boundary_crossing"
                and row.get("surface") == surface
                and row.get("from") == "pre_release_region"
                and row.get("to") == "post_release_region"
                and abs(float(row["release_x_m"]) - release_x) <= 1e-9
                and int(row["direction"]) == direction
                and isinstance(position, Sequence)
                and len(position) == 3
                and all(math.isfinite(float(value)) for value in position)
                and direction * (float(position[0]) - release_x) >= -1e-6
                and float(row["timestamp"]) > first_surface + 1e-9
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return False

    boundary_rows = [
        row
        for row in ordered
        if row.get("event_type") == "surface_release_boundary_crossing"
        and row.get("surface") == surface
    ]
    valid_boundaries = [row for row in boundary_rows if valid_boundary(row)]
    if not valid_boundaries:
        return {
            **base,
            "invalid_boundary_row_count": len(boundary_rows),
        }
    first_boundary = valid_boundaries[0]
    boundary_time = float(first_boundary["timestamp"])
    preceding_motion = [
        row
        for row in ordered
        if row.get("event_type") == "motion_mode_transition"
        and first_surface + 1e-9
        < float(row.get("timestamp", 0.0))
        <= boundary_time + 1e-9
    ]
    preceding_free_flight = bool(
        preceding_motion and preceding_motion[-1].get("to") == "free_flight"
    )
    pre_boundary_contact = min(
        (
            float(event["timestamp"])
            for event in events
            if first_surface + 1e-9
            < float(event["timestamp"])
            < boundary_time - 1e-9
        ),
        default=math.inf,
    )
    pre_boundary_return = min(
        (
            float(row["timestamp"])
            for row in ordered
            if row.get("event_type") == "motion_mode_transition"
            and row.get("from") == "free_flight"
            and row.get("to") != "free_flight"
            and first_surface + 1e-9
            < float(row["timestamp"])
            < boundary_time - 1e-9
        ),
        default=math.inf,
    )
    post_boundary_contact = min(
        (
            float(event["timestamp"])
            for event in events
            if float(event["timestamp"]) > boundary_time + 1e-9
        ),
        default=math.inf,
    )
    post_boundary_mode_change = min(
        (
            float(row["timestamp"])
            for row in ordered
            if row.get("event_type") == "motion_mode_transition"
            and float(row["timestamp"]) > boundary_time + 1e-9
        ),
        default=math.inf,
    )
    dwell = max(
        0.0,
        min(
            float(terminal_time_s),
            post_boundary_contact,
            post_boundary_mode_change,
        )
        - boundary_time,
    )
    return {
        **base,
        "passed": bool(
            preceding_free_flight
            and not math.isfinite(pre_boundary_contact)
            and not math.isfinite(pre_boundary_return)
            and dwell + 1e-9 >= minimum_dwell
        ),
        "free_flight_start_time_s": boundary_time,
        "free_flight_dwell_s": dwell,
        "release_x_m": release_x,
        "intervening_contact_time_s": (
            post_boundary_contact if math.isfinite(post_boundary_contact) else None
        ),
        "intervening_motion_transition_time_s": (
            post_boundary_mode_change
            if math.isfinite(post_boundary_mode_change)
            else None
        ),
        "pre_boundary_contact_time_s": (
            pre_boundary_contact if math.isfinite(pre_boundary_contact) else None
        ),
        "pre_boundary_free_flight_return_time_s": (
            pre_boundary_return if math.isfinite(pre_boundary_return) else None
        ),
        "preceding_free_flight_transition": preceding_free_flight,
        "invalid_boundary_row_count": len(boundary_rows) - len(valid_boundaries),
    }


def _surface_to_free_flight_evidence(
    spec: ScenarioSpec,
    frames: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]] | None,
    *,
    surface: str,
) -> dict[str, Any]:
    """Measure an ordered, geometrically cleared, sustained flight transition."""

    surface_times = [
        float(event["timestamp"])
        for event in events
        if event.get("object_b") == surface
    ]
    release_x_raw = spec.extras.get("transition_release_x_m")
    direction = int(spec.extras.get("transition_direction", 1))
    minimum_dwell = float(
        spec.extras.get("transition_minimum_free_flight_dwell_s", 0.06)
    )
    if not surface_times or release_x_raw is None or direction not in {-1, 1}:
        return {
            "passed": False,
            "surface_contact_time_s": min(surface_times, default=None),
            "free_flight_start_time_s": None,
            "free_flight_dwell_s": 0.0,
            "release_x_m": release_x_raw,
            "direction": direction,
            "minimum_dwell_s": minimum_dwell,
            "evidence_source": (
                "persisted_transition_rows"
                if transitions is not None
                else "encoded_frame_fallback"
            ),
        }
    first_surface = min(surface_times)
    if transitions is not None:
        return _transition_surface_release_evidence(
            spec,
            events,
            transitions,
            surface=surface,
            terminal_time_s=float(frames[-1]["timestamp"]),
        )
    intervening_contact_time = min(
        (
            float(event["timestamp"])
            for event in events
            if float(event["timestamp"]) > first_surface + 1e-9
        ),
        default=math.inf,
    )
    release_x = float(release_x_raw)
    runs: list[tuple[float, float]] = []
    run_start: float | None = None
    run_end: float | None = None
    for row in frames:
        timestamp = float(row["timestamp"])
        mode = str(row.get("motion_mode") or row.get("object.motion_mode") or "")
        active_surface = str(
            row.get("active_surface") or row.get("object.active_surface") or "none"
        )
        position_x = float(_vector(row, "object.position")[0])
        eligible = bool(
            timestamp > first_surface
            and timestamp < intervening_contact_time
            and mode == "free_flight"
            and active_surface in {"", "none"}
            and direction * (position_x - release_x) >= 0.0
        )
        if eligible:
            if run_start is None:
                run_start = timestamp
            run_end = timestamp
        elif run_start is not None:
            runs.append((run_start, float(run_end)))
            run_start = run_end = None
    if run_start is not None:
        runs.append((run_start, float(run_end)))
    selected = runs[0] if runs else (None, None)
    best_start, best_end = selected
    dwell = (
        0.0
        if best_start is None or best_end is None
        else max(0.0, best_end - best_start)
    )
    return {
        "passed": bool(
            best_start is not None
            and best_end is not None
            and best_end - best_start + 1e-9 >= minimum_dwell
        ),
        "surface_contact_time_s": first_surface,
        "intervening_contact_time_s": (
            intervening_contact_time
            if math.isfinite(intervening_contact_time)
            else None
        ),
        "free_flight_start_time_s": best_start,
        "free_flight_dwell_s": dwell,
        "release_x_m": release_x,
        "direction": direction,
        "minimum_dwell_s": minimum_dwell,
        "evidence_source": "encoded_frame_fallback",
    }


def evaluate_saved_native_episode(
    spec: ScenarioSpec,
    frame_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]] | None = None,
) -> ObjectiveEvaluation:
    """Recompute labels without reading intended branch or generator outcomes."""

    spec.validate()
    if not frame_rows:
        raise ValueError("objective recomputation requires persisted frame rows")
    frames = sorted((dict(row) for row in frame_rows), key=lambda row: float(row["timestamp"]))
    events = sorted((dict(row) for row in event_rows), key=lambda row: float(row["timestamp"]))
    tool_events = [event for event in events if event.get("object_b") == "native_tool"]
    contacts = [str(event.get("object_b")) for event in events]

    def rebounded_from(surface: str) -> bool:
        return any(
            event.get("object_b") == surface
            and float(event.get("normal_velocity_pre_m_s", 0.0))
            * float(event.get("normal_velocity_post_m_s", 0.0))
            < 0.0
            and abs(float(event.get("normal_velocity_post_m_s", 0.0))) > 0.05
            for event in events
        )

    modes = [str(row.get("motion_mode") or row.get("object.motion_mode") or "unknown") for row in frames]
    chronological = [
        *[(float(event["timestamp"]), str(event.get("object_b"))) for event in events],
        *[(float(row["timestamp"]), mode) for row, mode in zip(frames, modes)],
    ]
    observed_sequence = [value for _timestamp, value in sorted(chronological)]
    expected_observed = _ordered(spec.expected_contact_sequence, observed_sequence)
    distances = [
        float(np.linalg.norm(_vector(row, "object.position") - _vector(row, "robot.tool_position")))
        for row in frames
    ]
    minimum_tool_distance = min(distances)
    final = frames[-1]
    final_velocity = _vector(final, "object.linear_velocity")
    final_speed = float(np.linalg.norm(final_velocity))
    enabled_values = [
        bool(
            row.get("action.command.enabled", row.get("command.enabled", False))
        )
        for row in frames
    ]
    command_vectors = [
        np.asarray(row.get("action.command.joint_position", ()), dtype=np.float64)
        for row in frames
    ]
    command_moved = bool(
        command_vectors
        and command_vectors[0].size
        and any(
            vector.shape == command_vectors[0].shape
            and float(np.linalg.norm(vector - command_vectors[0])) > 1e-5
            for vector in command_vectors[1:]
        )
    )
    passive_scenarios = {
        RigidScenario.STRAIGHT_ROLL,
        RigidScenario.STRAIGHT_SLIDE,
        RigidScenario.ROLLING_SLIDING_TRANSITION,
        RigidScenario.SMALL_SLOPE,
        RigidScenario.RAMP_TO_TABLE,
        RigidScenario.ROLL_OFF_EDGE,
        RigidScenario.TABLE_BOUNCE,
        RigidScenario.WALL_REBOUND,
        RigidScenario.ANGLED_BARRIER_REBOUND,
        RigidScenario.RAMP_LAUNCH,
        RigidScenario.PROJECTILE_ROLL_OFF_EDGE,
        RigidScenario.FLOOR_TO_WALL,
        RigidScenario.FLIGHT_TO_TABLE_BOUNCE,
    }
    robot_action_required = spec.scenario not in passive_scenarios
    joint_vectors = [
        np.asarray(row.get("robot.joint_position", ()), dtype=np.float64)
        for row in frames
    ]
    tool_vectors = [
        np.asarray(row.get("robot.tool_position", ()), dtype=np.float64)
        for row in frames
    ]
    maximum_joint_displacement = max(
        (
            float(np.max(np.abs(vector - joint_vectors[0])))
            for vector in joint_vectors[1:]
            if vector.shape == joint_vectors[0].shape and vector.size
        ),
        default=0.0,
    ) if joint_vectors and joint_vectors[0].size else math.inf
    maximum_tool_displacement = max(
        (
            float(np.linalg.norm(vector - tool_vectors[0]))
            for vector in tool_vectors[1:]
            if vector.shape == tool_vectors[0].shape and vector.size
        ),
        default=0.0,
    ) if tool_vectors and tool_vectors[0].size else math.inf
    measured_static_command = (
        not command_moved
        # Position actuators can settle slightly under gravity while holding
        # one constant setpoint.  Two-centimetre/radian bounds distinguish
        # this measured servo settling from a commanded task motion.
        and maximum_joint_displacement <= 2e-2
        and maximum_tool_displacement <= 2e-2
    )
    controller_no_op = robot_action_required and measured_static_command
    nominal_intercept_position = spec.extras.get("nominal_intercept_position_m")
    nominal_intercept_target: np.ndarray | None = None
    if nominal_intercept_position is not None:
        candidate = np.asarray(nominal_intercept_position, dtype=np.float64)
        if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
            raise ValueError(
                "nominal_intercept_position_m must contain three finite values"
            )
        nominal_intercept_target = candidate
    initial_nominal_intercept_error = (
        float(np.linalg.norm(tool_vectors[0] - nominal_intercept_target))
        if nominal_intercept_target is not None and tool_vectors
        else None
    )
    terminal_nominal_intercept_error = (
        float(np.linalg.norm(tool_vectors[-1] - nominal_intercept_target))
        if nominal_intercept_target is not None and tool_vectors
        else None
    )
    misdirected_action_measured = bool(
        robot_action_required
        and command_moved
        and maximum_tool_displacement
        >= OBJECTIVE_THRESHOLDS["wrong_action_minimum_tool_displacement_m"]
        and terminal_nominal_intercept_error is not None
        and terminal_nominal_intercept_error
        > OBJECTIVE_THRESHOLDS["wrong_action_target_error_m"]
    )
    metrics: dict[str, Any] = {
        "object_contacted_tool": bool(tool_events),
        "expected_contact_sequence": list(spec.expected_contact_sequence),
        "observed_contact_sequence": contacts,
        "expected_sequence_observed": expected_observed,
        "minimum_object_tool_distance_m": minimum_tool_distance,
        "final_object_speed_m_s": final_speed,
        "controller_no_op_measured": controller_no_op,
        "robot_action_required": robot_action_required,
        "static_command_measured": measured_static_command,
        "command_motion_measured": command_moved,
        "maximum_robot_joint_displacement_rad": maximum_joint_displacement,
        "maximum_robot_tool_displacement_m": maximum_tool_displacement,
        "controller_enabled_observation": any(enabled_values),
        "nominal_intercept_position_m": (
            nominal_intercept_target.tolist()
            if nominal_intercept_target is not None
            else None
        ),
        "initial_nominal_intercept_error_m": initial_nominal_intercept_error,
        "terminal_nominal_intercept_error_m": terminal_nominal_intercept_error,
        "misdirected_action_measured": misdirected_action_measured,
        "objective_evaluator_id": EVALUATOR_ID,
        "objective_evaluator_version": EVALUATOR_VERSION,
        "objective_threshold_set_hash": OBJECTIVE_THRESHOLD_SET_HASH,
    }
    success = False
    score = 0.0
    failure = "unexpected_failure"
    transition_failure = False
    if spec.family == NativeFamily.FALLING_CATCH:
        inside = [_inside_tool(spec, row) for row in frames]
        dwell = 0
        for value in reversed(inside):
            if not value:
                break
            dwell += 1
        dwell_s = dwell / spec.video_hz
        retained = bool(tool_events) and dwell_s >= OBJECTIVE_THRESHOLDS["retention_dwell_s"]
        phase_names = [
            str(row.get("task_phase") or row.get("task.phase") or "")
            for row in frames
        ]
        phases = set(phase_names)
        required_phase = {
            RigidScenario.CATCH_TRANSPORT: "transport",
            RigidScenario.CATCH_BRAKE: "abrupt_brake",
            RigidScenario.CATCH_TILT: "tilt",
            RigidScenario.CATCH_EDGE_RECOVERY: "edge_recovery",
        }.get(spec.scenario)
        phase_complete = required_phase is None or required_phase in phases
        task_specific_evidence = True
        task_specific_progress = 1.0
        relative_positions = [_tool_relative(row) for row in frames]
        if spec.scenario == RigidScenario.CATCH_TRANSPORT:
            terminal_target = np.asarray(
                spec.phases[-1].target_position_m, dtype=np.float64
            )
            terminal_position_error = float(
                np.linalg.norm(
                    _vector(final, "robot.tool_position") - terminal_target
                )
            )
            initial_target_error = float(
                np.linalg.norm(
                    _vector(frames[0], "robot.tool_position") - terminal_target
                )
            )
            task_specific_progress = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - terminal_position_error
                    / max(
                        initial_target_error,
                        OBJECTIVE_THRESHOLDS["transport_terminal_error_m"],
                    ),
                ),
            )
            task_specific_evidence = terminal_position_error <= OBJECTIVE_THRESHOLDS[
                "transport_terminal_error_m"
            ]
            metrics.update(
                transport_terminal_position_error_m=terminal_position_error,
                transport_initial_target_error_m=initial_target_error,
            )
        elif spec.scenario == RigidScenario.CATCH_BRAKE:
            brake_indices = [
                index
                for index, name in enumerate(phase_names)
                if name in {"abrupt_brake", "post_brake_hold"}
            ]
            brake_retention_fraction = (
                sum(inside[index] for index in brake_indices) / len(brake_indices)
                if brake_indices
                else 0.0
            )
            task_specific_progress = min(
                1.0,
                brake_retention_fraction
                / OBJECTIVE_THRESHOLDS["brake_retention_fraction"],
            )
            task_specific_evidence = brake_retention_fraction >= OBJECTIVE_THRESHOLDS[
                "brake_retention_fraction"
            ]
            metrics["brake_retention_fraction"] = brake_retention_fraction
        elif spec.scenario == RigidScenario.CATCH_TILT:
            initial_rotation = _rotation_wxyz(
                frames[0]["robot.tool_quaternion_wxyz"]
            )
            maximum_tilt = 0.0
            for row in frames:
                relative_rotation = initial_rotation.T @ _rotation_wxyz(
                    row["robot.tool_quaternion_wxyz"]
                )
                cosine = max(
                    -1.0,
                    min(1.0, (float(np.trace(relative_rotation)) - 1.0) / 2.0),
                )
                maximum_tilt = max(maximum_tilt, math.acos(cosine))
            task_specific_progress = min(
                1.0,
                maximum_tilt / OBJECTIVE_THRESHOLDS["tilt_minimum_rad"],
            )
            task_specific_evidence = maximum_tilt >= OBJECTIVE_THRESHOLDS[
                "tilt_minimum_rad"
            ]
            metrics["measured_maximum_tool_tilt_rad"] = maximum_tilt
        elif spec.scenario == RigidScenario.CATCH_EDGE_RECOVERY:
            edge_excursion = max(
                max(abs(float(position[0])), abs(float(position[1])))
                for position in relative_positions
            )
            final_offset = max(
                abs(float(relative_positions[-1][0])),
                abs(float(relative_positions[-1][1])),
            )
            task_specific_evidence = (
                edge_excursion
                >= OBJECTIVE_THRESHOLDS["edge_excursion_tool_fraction"]
                * min(spec.tool.half_extents_m[:2])
                and final_offset
                <= OBJECTIVE_THRESHOLDS["edge_recovery_tool_fraction"]
                * min(spec.tool.half_extents_m[:2])
            )
            excursion_threshold = (
                OBJECTIVE_THRESHOLDS["edge_excursion_tool_fraction"]
                * min(spec.tool.half_extents_m[:2])
            )
            recovery_threshold = (
                OBJECTIVE_THRESHOLDS["edge_recovery_tool_fraction"]
                * min(spec.tool.half_extents_m[:2])
            )
            excursion_progress = min(
                1.0, edge_excursion / max(excursion_threshold, 1e-9)
            )
            recovery_progress = min(
                1.0, recovery_threshold / max(final_offset, 1e-9)
            )
            task_specific_progress = 0.5 * (
                excursion_progress + recovery_progress
            )
            metrics.update(
                edge_excursion_m=edge_excursion,
                recovered_final_center_offset_m=final_offset,
            )
        phase_complete = phase_complete and task_specific_evidence
        success = retained and phase_complete
        retention_progress = (
            min(1.0, dwell_s / OBJECTIVE_THRESHOLDS["retention_dwell_s"])
            if tool_events
            else max(0.0, 1.0 - minimum_tool_distance / 0.35)
        )
        phase_progress = (
            1.0
            if required_phase is None
            else (
                task_specific_progress
                if required_phase in phases
                else 0.0
            )
        )
        if success:
            score = 1.0
        elif retained and required_phase is not None:
            # Retention is half of a post-contact task. Keep a failed second
            # phase strictly below one so the closed partial-success class and
            # its score agree.
            score = min(0.99, 0.5 + 0.49 * phase_progress)
        else:
            score = retention_progress
        metrics.update(
            object_entered_receptacle=any(inside),
            object_retained_until_end=retained,
            retention_dwell_s=dwell_s,
            post_contact_phase_completed=phase_complete,
            task_specific_post_contact_evidence=task_specific_evidence,
            retention_progress=retention_progress,
            post_contact_phase_progress=phase_progress,
        )
        if success:
            failure = "none"
        elif tool_events:
            failure = (
                "contact_without_completion"
                if retained
                else "object_not_retained"
            )
        elif controller_no_op:
            failure = "controller_no_op"
        elif minimum_tool_distance <= 0.25:
            failure = "receptacle_near_miss"
        else:
            failure = "receptacle_missed_object"
    elif spec.family == NativeFamily.ROLLING_INTERCEPTION:
        goal = np.asarray(spec.extras.get("goal_center_m", (0.16, 0.0, 0.0)), dtype=np.float64)
        goal_distances = [
            float(np.linalg.norm(_vector(row, "object.position")[:2] - goal[:2])) for row in frames
        ]
        entered_goal = min(goal_distances) <= OBJECTIVE_THRESHOLDS["goal_radius_m"]
        inside_goal = [
            distance <= OBJECTIVE_THRESHOLDS["goal_radius_m"]
            for distance in goal_distances
        ]
        terminal_goal_dwell = 0
        for value in reversed(inside_goal):
            if not value:
                break
            terminal_goal_dwell += 1
        terminal_goal_dwell_s = terminal_goal_dwell / spec.video_hz
        if spec.scenario == RigidScenario.RAMP_TO_TABLE:
            success = _ordered(("ramp_surface", "table_surface"), contacts)
            metrics["expected_sequence_observed"] = success
        elif spec.scenario == RigidScenario.ROLL_OFF_EDGE:
            transition = _surface_to_free_flight_evidence(
                spec,
                frames,
                events,
                transition_rows,
                surface="table_surface",
            )
            success = bool(transition["passed"])
            metrics["surface_to_free_flight"] = transition
            metrics["expected_sequence_observed"] = success
        elif spec.scenario == RigidScenario.STRAIGHT_ROLL:
            success = "rolling" in modes and abs(float(_vector(final, "object.position")[1])) < 0.5
        elif spec.scenario == RigidScenario.STRAIGHT_SLIDE:
            success = "sliding" in modes and abs(float(_vector(final, "object.position")[1])) < 0.5
        elif spec.scenario == RigidScenario.ROLLING_SLIDING_TRANSITION:
            success = _ordered(("sliding", "rolling"), modes)
        elif spec.scenario == RigidScenario.SMALL_SLOPE:
            success = "rolling" in modes and abs(float(_vector(final, "object.position")[1])) < 0.5
        elif spec.scenario == RigidScenario.PADDLE_BLOCK:
            success = bool(tool_events) and float(np.linalg.norm(final_velocity[:2])) < OBJECTIVE_THRESHOLDS["rolling_block_speed_m_s"]
        elif spec.scenario == RigidScenario.CONTAINER_RECEIVE:
            success = (
                bool(tool_events)
                and terminal_goal_dwell_s >= 0.25
                and float(np.linalg.norm(final_velocity[:2])) < 0.35
            )
        elif spec.scenario == RigidScenario.REDIRECT_TO_TARGET:
            contact_time = min(
                (float(event["timestamp"]) for event in tool_events),
                default=math.inf,
            )
            post_contact_distances = [
                distance
                for row, distance in zip(frames, goal_distances)
                if float(row["timestamp"]) >= contact_time
            ]
            directed_toward_goal = bool(
                len(post_contact_distances) >= 2
                and post_contact_distances[-1]
                < post_contact_distances[0] - 0.05
            )
            success = bool(tool_events) and entered_goal and directed_toward_goal
            metrics["post_contact_directed_toward_goal"] = directed_toward_goal
        elif spec.scenario == RigidScenario.OCCLUDED_INTERCEPTION:
            success = (
                bool(tool_events)
                and float(np.linalg.norm(final_velocity[:2]))
                < OBJECTIVE_THRESHOLDS["rolling_block_speed_m_s"]
            )
        else:
            success = bool(tool_events) and entered_goal
        score = max(0.0, min(1.0, 1.0 - goal_distances[-1] / 0.55))
        metrics.update(
            object_entered_goal=entered_goal,
            final_goal_distance_m=goal_distances[-1],
            rolling_observed="rolling" in modes,
            sliding_observed="sliding" in modes,
            free_flight_observed="free_flight" in modes,
            terminal_goal_dwell_s=terminal_goal_dwell_s,
        )
        if success:
            failure = "none"
        elif spec.scenario == RigidScenario.ROLL_OFF_EDGE:
            failure = "contact_without_completion"
            score = 0.0
            transition_failure = True
        elif tool_events:
            failure = "object_not_retained_in_goal"
        elif controller_no_op:
            failure = "controller_no_op"
        elif minimum_tool_distance <= 0.24:
            failure = "tool_near_miss"
        else:
            failure = "tool_missed_rolling_object"
    else:
        first_tool_contact_s = min(
            (float(event["timestamp"]) for event in tool_events),
            default=math.inf,
        )
        post_tool_contact_frames = [
            row
            for row in frames
            if float(row["timestamp"]) >= first_tool_contact_s
        ]
        most_negative_post_contact_x_velocity = min(
            (
                float(_vector(row, "object.linear_velocity")[0])
                for row in post_tool_contact_frames
            ),
            default=math.inf,
        )
        reverse_speed_quality = (
            max(
                0.0,
                min(
                    1.0,
                    -most_negative_post_contact_x_velocity
                    / OBJECTIVE_THRESHOLDS["projectile_reverse_speed_m_s"],
                ),
            )
            if tool_events
            else 0.0
        )
        if spec.scenario in {RigidScenario.TABLE_BOUNCE, RigidScenario.FLIGHT_TO_TABLE_BOUNCE}:
            success = rebounded_from("table_surface")
        elif spec.scenario in {RigidScenario.WALL_REBOUND, RigidScenario.ANGLED_BARRIER_REBOUND}:
            surface = "wall_surface" if spec.scenario == RigidScenario.WALL_REBOUND else "angled_barrier_surface"
            success = rebounded_from(surface)
        elif spec.scenario == RigidScenario.FLOOR_TO_WALL:
            success = _ordered(("table_surface", "wall_surface"), contacts)
        elif spec.scenario == RigidScenario.RAMP_LAUNCH:
            transition = _surface_to_free_flight_evidence(
                spec,
                frames,
                events,
                transition_rows,
                surface="ramp_surface",
            )
            success = bool(transition["passed"])
            metrics["surface_to_free_flight"] = transition
            metrics["expected_sequence_observed"] = success
        elif spec.scenario == RigidScenario.PROJECTILE_ROLL_OFF_EDGE:
            transition = _surface_to_free_flight_evidence(
                spec,
                frames,
                events,
                transition_rows,
                surface="table_surface",
            )
            success = bool(transition["passed"])
            metrics["surface_to_free_flight"] = transition
            metrics["expected_sequence_observed"] = success
        elif spec.scenario == RigidScenario.BOUNCE_TO_INTERCEPTION:
            success = _ordered(("table_surface", "native_tool"), contacts)
        else:
            success = bool(tool_events) and any(
                _vector(row, "object.linear_velocity")[0]
                < -OBJECTIVE_THRESHOLDS["projectile_reverse_speed_m_s"]
                for row in post_tool_contact_frames
            )
        if success:
            score = 1.0
        elif tool_events:
            # Contact alone is not partial success.  Direct deflection earns a
            # continuous score from the *measured post-contact* reverse speed;
            # a weak nearly-correct reversal can be partial, while unchanged
            # or wrong-direction motion is a contact failure.  In the
            # bounce-to-interception task, any non-successful tool contact has
            # the wrong event order by construction and receives zero credit.
            score = (
                0.0
                if spec.scenario == RigidScenario.BOUNCE_TO_INTERCEPTION
                else reverse_speed_quality
            )
        else:
            score = max(0.0, 1.0 - minimum_tool_distance / 0.45)
        metrics.update(
            correct_rebound_direction=success,
            ground_impact_count=sum(event.get("object_b") == "table_surface" for event in events),
            wall_impact_count=sum("wall" in str(event.get("object_b")) for event in events),
            first_tool_contact_s=(
                first_tool_contact_s if math.isfinite(first_tool_contact_s) else None
            ),
            most_negative_post_contact_x_velocity_m_s=(
                most_negative_post_contact_x_velocity
                if math.isfinite(most_negative_post_contact_x_velocity)
                else None
            ),
            measured_reverse_speed_quality=reverse_speed_quality,
        )
        if success:
            failure = "none"
        elif spec.scenario in {
            RigidScenario.RAMP_LAUNCH,
            RigidScenario.PROJECTILE_ROLL_OFF_EDGE,
        }:
            failure = "contact_without_completion"
            score = 0.0
            transition_failure = True
        elif tool_events:
            failure = "incorrect_rebound_direction"
        elif controller_no_op:
            failure = "controller_no_op"
        elif minimum_tool_distance <= OBJECTIVE_THRESHOLDS["near_tool_distance_m"]:
            failure = "paddle_near_miss"
        else:
            failure = "paddle_missed_projectile"
    if success:
        actual = "success"
    elif transition_failure:
        actual = "contact_failure"
    elif tool_events:
        actual = "partial_success" if 0.5 <= score < 1.0 else "contact_failure"
    elif controller_no_op:
        actual = "no_op"
    elif misdirected_action_measured:
        actual = "wrong_action"
        failure = "wrong_action"
    elif minimum_tool_distance <= OBJECTIVE_THRESHOLDS["near_tool_distance_m"]:
        actual = "near_miss"
    else:
        actual = "miss"
    metrics["objective_success"] = success
    evidence = {
        "stored_rows_recomputed": True,
        "frame_count": len(frames),
        "event_count": len(events),
        "transition_count": (
            None if transition_rows is None else len(transition_rows)
        ),
        "first_timestamp_s": float(frames[0]["timestamp"]),
        "last_timestamp_s": float(frames[-1]["timestamp"]),
    }
    outcome = OutcomeResult(
        task_success=success,
        partial_success_score=max(0.0, min(1.0, score)),
        failure_mode=failure,
        metrics=metrics,
        label_confidence=0.98,
        label_status="verified_objective",
    )
    return ObjectiveEvaluation(
        outcome=outcome,
        actual_outcome_class=actual,
        evaluator_id=EVALUATOR_ID,
        evaluator_version=EVALUATOR_VERSION,
        threshold_set_hash=OBJECTIVE_THRESHOLD_SET_HASH,
        evidence=evidence,
    )


OBJECTIVE_EVALUATORS: Mapping[str, Callable[..., ObjectiveEvaluation]] = {
    EVALUATOR_ID: evaluate_saved_native_episode,
}


def get_objective_evaluator(identifier: str) -> Callable[..., ObjectiveEvaluation]:
    try:
        return OBJECTIVE_EVALUATORS[identifier]
    except KeyError as error:
        raise KeyError(f"unknown native objective evaluator {identifier!r}") from error
