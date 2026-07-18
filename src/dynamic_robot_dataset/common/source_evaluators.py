"""Independent persisted-artifact evaluators for the owned rigid source backend."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contract_v2 import (
    DEFAULT_OBJECTIVE_EVALUATORS,
    ObjectiveRecomputeInput,
    ObjectiveRecomputeResult,
)
from .schema import ActualOutcomeClass


SOURCE_OBJECTIVE_EVALUATOR_VERSION = "1.1.0"
SOURCE_OBJECTIVE_EVALUATOR_IDS = (
    "passive_freeflight_v1",
    "passive_projectile_v1",
    "passive_rebound_v1",
    "passive_rolling_v1",
    "rigid_catch_v2",
    "rigid_projectile_interception_v2",
    "rigid_rebound_v2",
)


def _vector(row: Mapping[str, Any], name: str) -> tuple[float, ...] | None:
    raw = row.get(name)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    try:
        result = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return result if result and all(math.isfinite(value) for value in result) else None


def _invalid_penetration(events: Sequence[Mapping[str, Any]]) -> tuple[bool, float]:
    maximum = 0.0
    invalid = False
    for row in events:
        try:
            depth = float(row["penetration_depth_m"])
        except (KeyError, TypeError, ValueError):
            return True, math.inf
        if not math.isfinite(depth) or depth < 0:
            return True, depth
        maximum = max(maximum, depth)
        category = str(row.get("contact_category") or "")
        limit = 0.002 if category in {"gripper", "robot_arm"} else 0.003
        invalid = invalid or depth > limit
    return invalid, maximum


def _key_event(source_spec: Mapping[str, Any]) -> tuple[str, float]:
    physics = source_spec.get("physics")
    if not isinstance(physics, Mapping):
        raise ValueError("source evaluator lacks scenario physics")
    try:
        event_time = float(physics["key_event_time_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source evaluator lacks a planned key-event time") from error
    if not math.isfinite(event_time) or event_time < 0:
        raise ValueError("source evaluator key-event time is invalid")
    return str(physics.get("key_event_name") or "task_interaction"), event_time


def _contact_time_tolerance(source_spec: Mapping[str, Any]) -> float:
    physics = source_spec.get("physics")
    if not isinstance(physics, Mapping):
        return 1e-9
    try:
        simulation_hz = float(physics.get("simulation_hz", 0.0))
    except (TypeError, ValueError):
        return 1e-9
    return 1.0 / simulation_hz if math.isfinite(simulation_hz) and simulation_hz > 0 else 1e-9


def select_source_key_event(
    *,
    planned_key_event_name: str,
    planned_key_event_time_s: float,
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    passive: bool,
    contact_time_tolerance_s: float = 1e-9,
) -> dict[str, Any]:
    """Select the same persisted event for objective replay and visibility QC."""

    planned_name = str(planned_key_event_name or "task_interaction")
    planned_time = float(planned_key_event_time_s)
    if not planned_name.strip():
        raise ValueError("source key-event name cannot be empty")
    if not math.isfinite(planned_time) or planned_time < 0.0:
        raise ValueError("source key-event time is invalid")
    if passive:
        return {
            "key_event_name": planned_name,
            "key_event_time_s": planned_time,
            "key_event_source": "planned_source_scenario_event",
            "physical_contact_applicable": None,
            "contact_counterpart_geom_ids": [],
        }

    bilateral_times = [
        float(row["timestamp"])
        for row in state_rows
        if row.get("contact.bilateral") is True
        and isinstance(row.get("timestamp"), (int, float))
        and not isinstance(row.get("timestamp"), bool)
        and math.isfinite(float(row["timestamp"]))
    ]
    if bilateral_times:
        measured_time = min(bilateral_times)
        counterpart_ids = sorted(
            {
                int(row["counterpart_geom_id"])
                for row in event_rows
                if row.get("contact_category") == "gripper"
                and isinstance(row.get("timestamp"), (int, float))
                and not isinstance(row.get("timestamp"), bool)
                and abs(float(row["timestamp"]) - measured_time)
                <= float(contact_time_tolerance_s) + 1e-12
                and isinstance(row.get("counterpart_geom_id"), int)
                and not isinstance(row.get("counterpart_geom_id"), bool)
                and int(row["counterpart_geom_id"]) >= 0
            }
        )
        return {
            "key_event_name": "bilateral_grasp_onset",
            "key_event_time_s": measured_time,
            "key_event_source": "persisted_bilateral_contact",
            "physical_contact_applicable": True,
            "contact_counterpart_geom_ids": counterpart_ids,
        }

    tool_rows = [
        row
        for row in event_rows
        if row.get("contact_category") in {"gripper", "robot_arm"}
        and isinstance(row.get("timestamp"), (int, float))
        and not isinstance(row.get("timestamp"), bool)
        and math.isfinite(float(row["timestamp"]))
    ]
    if tool_rows:
        measured_time = min(float(row["timestamp"]) for row in tool_rows)
        counterpart_ids = sorted(
            {
                int(row["counterpart_geom_id"])
                for row in tool_rows
                if abs(float(row["timestamp"]) - measured_time) <= 1e-12
                and isinstance(row.get("counterpart_geom_id"), int)
                and not isinstance(row.get("counterpart_geom_id"), bool)
                and int(row["counterpart_geom_id"]) >= 0
            }
        )
        return {
            "key_event_name": "tool_contact_onset",
            "key_event_time_s": measured_time,
            "key_event_source": "persisted_contact_event",
            "physical_contact_applicable": True,
            "contact_counterpart_geom_ids": counterpart_ids,
        }
    return {
        "key_event_name": planned_name,
        "key_event_time_s": planned_time,
        "key_event_source": "planned_interception_for_measured_miss",
        "physical_contact_applicable": False,
        "contact_counterpart_geom_ids": [],
    }


def _invalid_result(
    *, evidence: dict[str, Any], key_event_name: str, key_event_time_s: float
) -> ObjectiveRecomputeResult:
    return ObjectiveRecomputeResult(
        task_success=False,
        actual_outcome_class=ActualOutcomeClass.INVALID,
        primary_failure_code="excessive_penetration",
        evidence=evidence,
        key_event_name=key_event_name,
        key_event_time_s=key_event_time_s,
    )


def evaluate_source_rows(
    *,
    evaluator_id: str,
    corpus_leaf_id: str,
    task_variant: str,
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> ObjectiveRecomputeResult:
    """Recompute a source outcome without branch intent or online labels."""

    if evaluator_id not in SOURCE_OBJECTIVE_EVALUATOR_IDS:
        raise ValueError(f"unsupported source objective evaluator {evaluator_id!r}")
    if not state_rows:
        raise ValueError("source objective replay requires persisted state rows")
    planned_key_event_name, planned_key_event_time_s = _key_event(source_spec)
    invalid_penetration, maximum_penetration = _invalid_penetration(event_rows)
    common_evidence: dict[str, Any] = {
        "evaluator_id": evaluator_id,
        "corpus_leaf_id": corpus_leaf_id,
        "task_variant": task_variant,
        "state_sample_count": len(state_rows),
        "contact_sample_count": len(event_rows),
        "maximum_penetration_depth_m": maximum_penetration,
        "branch_intent_read": False,
    }
    selected_key_event = select_source_key_event(
        planned_key_event_name=planned_key_event_name,
        planned_key_event_time_s=planned_key_event_time_s,
        state_rows=state_rows,
        event_rows=event_rows,
        passive=evaluator_id.startswith("passive_"),
        contact_time_tolerance_s=_contact_time_tolerance(source_spec),
    )
    if invalid_penetration:
        return _invalid_result(
            evidence={
                **common_evidence,
                "penetration_within_limits": False,
                "planned_key_event_name": planned_key_event_name,
                "planned_key_event_time_s": planned_key_event_time_s,
                "measured_key_event_source": selected_key_event[
                    "key_event_source"
                ],
            },
            key_event_name=str(selected_key_event["key_event_name"]),
            key_event_time_s=float(selected_key_event["key_event_time_s"]),
        )

    finite_state = all(
        _vector(row, "object.position") is not None
        and _vector(row, "object.linear_velocity") is not None
        for row in state_rows
    )
    if evaluator_id.startswith("passive_"):
        surface_contacts = [
            row for row in event_rows if row.get("contact_category") == "task_surface"
        ]
        free_flight_samples = sum(
            str(row.get("object.motion_mode") or row.get("motion_mode")) == "free_flight"
            for row in state_rows
        )
        if evaluator_id in {"passive_freeflight_v1", "passive_projectile_v1"}:
            objective_valid = finite_state and free_flight_samples >= 2
        elif evaluator_id == "passive_rebound_v1":
            objective_valid = finite_state and bool(surface_contacts)
        else:
            objective_valid = finite_state and bool(surface_contacts) and any(
                math.hypot(*(_vector(row, "object.linear_velocity") or (0.0, 0.0))[:2])
                > 1e-4
                for row in state_rows
            )
        evidence = {
            **common_evidence,
            "finite_state": finite_state,
            "free_flight_sample_count": free_flight_samples,
            "task_surface_contact_count": len(surface_contacts),
            "passive_observation_valid": objective_valid,
            "penetration_within_limits": True,
            "planned_key_event_name": planned_key_event_name,
            "planned_key_event_time_s": planned_key_event_time_s,
            "measured_key_event_source": selected_key_event[
                "key_event_source"
            ],
        }
        return ObjectiveRecomputeResult(
            task_success=objective_valid,
            actual_outcome_class=(
                ActualOutcomeClass.SUCCESS if objective_valid else ActualOutcomeClass.INVALID
            ),
            primary_failure_code="none" if objective_valid else "unstable_physics",
            evidence=evidence,
            key_event_name=str(selected_key_event["key_event_name"]),
            key_event_time_s=float(selected_key_event["key_event_time_s"]),
        )

    physics = source_spec.get("physics")
    if not isinstance(physics, Mapping):
        raise ValueError("source catch evaluator lacks scenario physics")
    simulation_hz = float(physics.get("simulation_hz", 0.0))
    if not math.isfinite(simulation_hz) or simulation_hz <= 0:
        raise ValueError("source catch evaluator lacks simulation_hz")
    maximum_run = run = 0
    relative_positions: list[tuple[float, ...]] = []
    for row in state_rows:
        bilateral = row.get("contact.bilateral") is True
        run = run + 1 if bilateral else 0
        maximum_run = max(maximum_run, run)
        if bilateral:
            object_position = _vector(row, "object.position")
            grasp_center = _vector(row, "grasp.center_position")
            if object_position is not None and grasp_center is not None:
                relative_positions.append(
                    tuple(left - right for left, right in zip(object_position, grasp_center))
                )
    sustained = maximum_run / simulation_hz >= 0.05
    stable = False
    relative_range = None
    retention_samples = int(round(0.10 * simulation_hz))
    if len(relative_positions) >= retention_samples > 0:
        recent = relative_positions[-retention_samples:]
        relative_range = max(
            max(row[axis] for row in recent) - min(row[axis] for row in recent)
            for axis in range(3)
        )
        stable = relative_range <= 0.01
    transport_supported = True
    if "transport" in task_variant:
        supported = [
            row
            for row in state_rows
            if row.get("contact.bilateral") is True
            and float(row.get("timestamp", -1.0)) >= planned_key_event_time_s
        ]
        start = _vector(supported[0], "object.position") if supported else None
        end = _vector(supported[-1], "object.position") if supported else None
        transport_supported = bool(
            start is not None and end is not None and math.dist(start, end) >= 0.06
        )
    success = bool(finite_state and sustained and stable and transport_supported)
    tool_contacts = sum(
        row.get("contact_category") in {"gripper", "robot_arm"} for row in event_rows
    )
    key_event_name = str(selected_key_event["key_event_name"])
    key_event_time_s = float(selected_key_event["key_event_time_s"])
    key_event_source = str(selected_key_event["key_event_source"])
    evidence = {
        **common_evidence,
        "finite_state": finite_state,
        "penetration_within_limits": True,
        "maximum_contiguous_bilateral_contact_s": maximum_run / simulation_hz,
        "sustained_opposing_bilateral_contacts": sustained,
        "stable_object_to_grasp_transform": stable,
        "maximum_object_to_grasp_relative_range_m": relative_range,
        "displacement_physically_supported_by_contacts": transport_supported,
        "tool_contact_count": tool_contacts,
        "planned_key_event_name": planned_key_event_name,
        "planned_key_event_time_s": planned_key_event_time_s,
        "measured_key_event_source": key_event_source,
    }
    if success:
        outcome_class = ActualOutcomeClass.SUCCESS
        failure_code = "none"
    elif tool_contacts:
        outcome_class = ActualOutcomeClass.CONTACT_FAILURE
        failure_code = "contact_without_completion"
    else:
        outcome_class = ActualOutcomeClass.MISS
        failure_code = "no_contact"
    return ObjectiveRecomputeResult(
        task_success=success,
        actual_outcome_class=outcome_class,
        primary_failure_code=failure_code,
        evidence=evidence,
        key_event_name=key_event_name,
        key_event_time_s=key_event_time_s,
    )


def evaluate_source_persisted(value: ObjectiveRecomputeInput) -> ObjectiveRecomputeResult:
    source_spec = value.record.extras.get("source_scenario_spec")
    if not isinstance(source_spec, Mapping):
        raise ValueError("source objective evaluator lacks SourceScenarioSpec")
    return evaluate_source_rows(
        evaluator_id=value.record.objective_evaluator_id,
        corpus_leaf_id=str(source_spec.get("corpus_leaf_id") or ""),
        task_variant=str(source_spec.get("task_variant") or value.record.variant),
        source_spec=source_spec,
        state_rows=value.object_state_rows or value.frame_rows,
        event_rows=value.event_rows,
    )


def register_source_objective_evaluators() -> None:
    for evaluator_id in SOURCE_OBJECTIVE_EVALUATOR_IDS:
        try:
            DEFAULT_OBJECTIVE_EVALUATORS.register(
                evaluator_id,
                SOURCE_OBJECTIVE_EVALUATOR_VERSION,
                evaluate_source_persisted,
            )
        except ValueError as error:
            if "already registered" not in str(error):
                raise


__all__ = [
    "SOURCE_OBJECTIVE_EVALUATOR_IDS",
    "SOURCE_OBJECTIVE_EVALUATOR_VERSION",
    "evaluate_source_persisted",
    "evaluate_source_rows",
    "register_source_objective_evaluators",
    "select_source_key_event",
]
