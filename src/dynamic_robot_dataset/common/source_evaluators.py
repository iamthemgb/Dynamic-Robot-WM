"""Independent persisted-artifact evaluators for the owned rigid source backend."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contract_v2 import (
    DEFAULT_OBJECTIVE_EVALUATORS,
    ObjectiveRecomputeInput,
    ObjectiveRecomputeResult,
)
from .grasp_retention import (
    DEFAULT_GRASP_RETENTION,
    GraspRetentionThresholds,
)
from .rebound import (
    DEFAULT_REBOUND_ACCEPTANCE,
    ReboundAcceptanceThresholds,
    measure_rebound_kinematics,
)
from .schema import ActualOutcomeClass


SOURCE_OBJECTIVE_EVALUATOR_VERSION = "1.5.0"
SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION = "1.4.0"
SOURCE_OBJECTIVE_EVALUATOR_IDS = (
    "passive_freeflight_v1",
    "passive_projectile_v1",
    "passive_rebound_v1",
    "passive_rolling_v1",
    "rigid_catch_v2",
    "rigid_projectile_interception_v2",
    "rigid_rebound_v2",
    "rigid_rolling_pickup_v1",
)

PASSIVE_EVENT_TASK_SURFACE_CONTACT = "task_surface_contact"
PASSIVE_EVENT_PROJECTILE_APEX = "projectile_apex"

_PASSIVE_EVENT_BY_EVALUATOR = {
    "passive_freeflight_v1": PASSIVE_EVENT_TASK_SURFACE_CONTACT,
    "passive_projectile_v1": PASSIVE_EVENT_PROJECTILE_APEX,
    "passive_rebound_v1": PASSIVE_EVENT_TASK_SURFACE_CONTACT,
}
_PASSIVE_EVENT_BY_MOTION_KIND = {
    "passive_freefall": PASSIVE_EVENT_TASK_SURFACE_CONTACT,
    "passive_projectile": PASSIVE_EVENT_PROJECTILE_APEX,
    "passive_table_bounce": PASSIVE_EVENT_TASK_SURFACE_CONTACT,
    "passive_wall_rebound": PASSIVE_EVENT_TASK_SURFACE_CONTACT,
}


def passive_event_semantics_for_evaluator(evaluator_id: str) -> str | None:
    """Return the persisted event definition owned by a passive evaluator."""

    return _PASSIVE_EVENT_BY_EVALUATOR.get(str(evaluator_id))


def passive_event_semantics_for_motion_kind(motion_kind: str) -> str | None:
    """Return the same persisted event definition from a compiled motion kind."""

    return _PASSIVE_EVENT_BY_MOTION_KIND.get(str(motion_kind))


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
    passive_event_semantics: str | None = None,
) -> dict[str, Any]:
    """Select the same persisted event for objective replay and visibility QC."""

    planned_name = str(planned_key_event_name or "task_interaction")
    planned_time = float(planned_key_event_time_s)
    if not planned_name.strip():
        raise ValueError("source key-event name cannot be empty")
    if not math.isfinite(planned_time) or planned_time < 0.0:
        raise ValueError("source key-event time is invalid")
    if passive and passive_event_semantics == PASSIVE_EVENT_TASK_SURFACE_CONTACT:
        surface_rows = [
            row
            for row in event_rows
            if row.get("contact_category") == "task_surface"
            and isinstance(row.get("timestamp"), (int, float))
            and not isinstance(row.get("timestamp"), bool)
            and math.isfinite(float(row["timestamp"]))
        ]
        if surface_rows:
            measured_time = min(float(row["timestamp"]) for row in surface_rows)
            counterpart_ids = sorted(
                {
                    int(row["counterpart_geom_id"])
                    for row in surface_rows
                    if abs(float(row["timestamp"]) - measured_time)
                    <= float(contact_time_tolerance_s) + 1e-12
                    and isinstance(row.get("counterpart_geom_id"), int)
                    and not isinstance(row.get("counterpart_geom_id"), bool)
                    and int(row["counterpart_geom_id"]) >= 0
                }
            )
            return {
                "key_event_name": "task_surface_contact_onset",
                "key_event_time_s": measured_time,
                "key_event_source": "persisted_task_surface_contact",
                "physical_contact_applicable": True,
                "contact_counterpart_geom_ids": counterpart_ids,
            }
        return {
            "key_event_name": planned_name,
            "key_event_time_s": planned_time,
            "key_event_source": "planned_source_scenario_event",
            "physical_contact_applicable": False,
            "contact_counterpart_geom_ids": [],
        }

    if passive and passive_event_semantics == PASSIVE_EVENT_PROJECTILE_APEX:
        persisted_samples: list[tuple[float, float, float, str]] = []
        for row in state_rows:
            timestamp = row.get("timestamp")
            position = _vector(row, "object.position")
            velocity = _vector(row, "object.linear_velocity")
            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not math.isfinite(float(timestamp))
                or position is None
                or len(position) < 3
                or velocity is None
                or len(velocity) < 3
            ):
                continue
            persisted_samples.append(
                (
                    float(timestamp),
                    float(position[2]),
                    float(velocity[2]),
                    str(row.get("object.motion_mode") or row.get("motion_mode") or ""),
                )
            )

        free_flight = [
            sample[:3] for sample in persisted_samples if sample[3] == "free_flight"
        ]
        candidates = [sample[:3] for sample in persisted_samples]
        samples = sorted(free_flight or candidates, key=lambda value: value[0])
        measured_time: float | None = None
        for first, second in zip(samples, samples[1:]):
            first_time, _, first_vz = first
            second_time, _, second_vz = second
            if second_time <= first_time:
                continue
            if first_vz == 0.0:
                measured_time = first_time
                break
            if first_vz > 0.0 and second_vz <= 0.0:
                fraction = first_vz / (first_vz - second_vz)
                measured_time = first_time + fraction * (second_time - first_time)
                break
        if measured_time is None and samples:
            measured_time = max(samples, key=lambda value: (value[1], -value[0]))[0]
        if measured_time is not None:
            return {
                "key_event_name": "projectile_apex",
                "key_event_time_s": measured_time,
                "key_event_source": "persisted_free_flight_apex",
                "physical_contact_applicable": False,
                "contact_counterpart_geom_ids": [],
            }
        return {
            "key_event_name": planned_name,
            "key_event_time_s": planned_time,
            "key_event_source": "planned_source_scenario_event",
            "physical_contact_applicable": False,
            "contact_counterpart_geom_ids": [],
        }

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


def _bound_rebound_acceptance(
    source_spec: Mapping[str, Any],
) -> tuple[ReboundAcceptanceThresholds, float]:
    """Validate the version-owned rebound contract before reading rollout rows.

    Scenario/config incompatibility is independent of whether a persisted
    rollout is complete.  Checking it up front keeps a missing or truncated
    state table from masking an unbound evaluator threshold change.
    """

    physics = source_spec.get("physics")
    if not isinstance(physics, Mapping):
        raise ValueError("persisted rebound evaluator lacks scenario physics")
    raw_thresholds = physics.get("rebound_acceptance")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError(
            "persisted rebound evaluator lacks bound acceptance thresholds"
        )
    thresholds = ReboundAcceptanceThresholds.from_dict(raw_thresholds)
    if thresholds != DEFAULT_REBOUND_ACCEPTANCE:
        raise ValueError(
            "persisted rebound acceptance differs from evaluator v1.3.0"
        )
    try:
        object_radius_m = float(physics["object_radius_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("persisted rebound evaluator lacks object radius") from error
    if not math.isfinite(object_radius_m) or object_radius_m <= 0.0:
        raise ValueError("persisted rebound evaluator has invalid object radius")
    return thresholds, object_radius_m


def _bound_grasp_retention(
    source_spec: Mapping[str, Any],
) -> tuple[GraspRetentionThresholds, float]:
    physics = source_spec.get("physics")
    if not isinstance(physics, Mapping):
        raise ValueError("source catch evaluator lacks scenario physics")
    raw_thresholds = physics.get("grasp_retention")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError("source catch evaluator lacks grasp-retention thresholds")
    thresholds = GraspRetentionThresholds.from_dict(raw_thresholds)
    if thresholds != DEFAULT_GRASP_RETENTION:
        raise ValueError(
            "persisted grasp retention differs from evaluator v1.5.0"
        )
    try:
        duration_s = float(source_spec["duration_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source catch evaluator lacks duration_s") from error
    if not math.isfinite(duration_s) or duration_s < thresholds.final_window_s:
        raise ValueError("source catch evaluator has invalid duration_s")
    return thresholds, duration_s


def evaluate_source_rows(
    *,
    evaluator_id: str,
    corpus_leaf_id: str,
    task_variant: str,
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    _evaluator_version: str = SOURCE_OBJECTIVE_EVALUATOR_VERSION,
) -> ObjectiveRecomputeResult:
    """Recompute a source outcome without branch intent or online labels."""

    if evaluator_id not in SOURCE_OBJECTIVE_EVALUATOR_IDS:
        raise ValueError(f"unsupported source objective evaluator {evaluator_id!r}")
    if _evaluator_version not in {
        SOURCE_OBJECTIVE_EVALUATOR_VERSION,
        SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION,
    }:
        raise ValueError(f"unsupported source objective version {_evaluator_version!r}")
    rebound_contract = (
        _bound_rebound_acceptance(source_spec)
        if evaluator_id == "passive_rebound_v1"
        else None
    )
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
        passive_event_semantics=passive_event_semantics_for_evaluator(
            evaluator_id
        ),
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
        rebound_evidence: Mapping[str, Any] = {}
        if evaluator_id in {"passive_freeflight_v1", "passive_projectile_v1"}:
            objective_valid = finite_state and free_flight_samples >= 2
        elif evaluator_id == "passive_rebound_v1":
            assert rebound_contract is not None
            thresholds, object_radius_m = rebound_contract
            rebound_evidence = measure_rebound_kinematics(
                state_rows,
                event_rows,
                object_radius_m=object_radius_m,
                thresholds=thresholds,
            )
            objective_valid = bool(
                finite_state
                and surface_contacts
                and rebound_evidence["rebound_acceptance_pass"]
            )
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
            **(
                {"rebound": dict(rebound_evidence)}
                if rebound_evidence
                else {}
            ),
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
    for row in state_rows:
        bilateral = row.get("contact.bilateral") is True
        run = run + 1 if bilateral else 0
        maximum_run = max(maximum_run, run)
    sustained = maximum_run / simulation_hz >= 0.05
    stable = False
    relative_range = None
    retention_evidence: dict[str, Any] = {}
    if _evaluator_version == SOURCE_OBJECTIVE_EVALUATOR_VERSION:
        retention, duration_s = _bound_grasp_retention(source_spec)
        final_rows = [
            row
            for row in state_rows
            if float(row.get("timestamp", -math.inf))
            >= duration_s - retention.final_window_s - 1e-12
        ]
        final_bilateral_rows = [
            row for row in final_rows if row.get("contact.bilateral") is True
        ]
        final_bilateral_fraction = (
            len(final_bilateral_rows) / len(final_rows) if final_rows else 0.0
        )
        final_bilateral_contact = bool(
            final_rows and final_rows[-1].get("contact.bilateral") is True
        )
        retained_through_final_state = bool(
            final_bilateral_contact
            and final_bilateral_fraction >= retention.minimum_bilateral_fraction
        )
        relative_positions = []
        for row in final_bilateral_rows:
            object_position = _vector(row, "object.position")
            grasp_center = _vector(row, "grasp.center_position")
            if object_position is not None and grasp_center is not None:
                relative_positions.append(
                    tuple(
                        left - right
                        for left, right in zip(object_position, grasp_center)
                    )
                )
        retention_samples = max(
            2,
            int(
                round(
                    retention.final_window_s
                    * simulation_hz
                    * retention.minimum_bilateral_fraction
                )
            ),
        )
        if (
            retained_through_final_state
            and len(relative_positions) >= retention_samples
        ):
            relative_range = max(
                max(row[axis] for row in relative_positions)
                - min(row[axis] for row in relative_positions)
                for axis in range(3)
            )
            stable = relative_range <= retention.maximum_relative_range_m
        retention_evidence = {
            "retained_through_final_state": retained_through_final_state,
            "final_bilateral_contact": final_bilateral_contact,
            "final_retention_window_s": retention.final_window_s,
            "final_retention_bilateral_fraction": final_bilateral_fraction,
        }
    else:
        # Exact v1.4 replay for already sealed artifacts.  This historical
        # evaluator intentionally used the last 100 ms of *bilateral rows*,
        # even when those rows occurred before the rollout ended.  It remains
        # registered for provenance replay only; new episodes always use v1.5.
        relative_positions = []
        for row in state_rows:
            if row.get("contact.bilateral") is not True:
                continue
            object_position = _vector(row, "object.position")
            grasp_center = _vector(row, "grasp.center_position")
            if object_position is not None and grasp_center is not None:
                relative_positions.append(
                    tuple(
                        left - right
                        for left, right in zip(object_position, grasp_center)
                    )
                )
        retention_samples = int(round(0.10 * simulation_hz))
        if len(relative_positions) >= retention_samples > 0:
            recent = relative_positions[-retention_samples:]
            relative_range = max(
                max(row[axis] for row in recent)
                - min(row[axis] for row in recent)
                for axis in range(3)
            )
            stable = relative_range <= 0.01
    transport_supported = True
    # A pickup is only complete when the grasp physically carries the object:
    # both rolling-pickup variants lift, so displacement evidence is required
    # for the pickup evaluator, not just for "transport" task variants.
    if "transport" in task_variant or evaluator_id == "rigid_rolling_pickup_v1":
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
        **retention_evidence,
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


def _evaluate_source_persisted_legacy_v1_4(
    value: ObjectiveRecomputeInput,
) -> ObjectiveRecomputeResult:
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
        _evaluator_version=SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION,
    )


def register_source_objective_evaluators() -> None:
    for evaluator_id in SOURCE_OBJECTIVE_EVALUATOR_IDS:
        for version, evaluator in (
            (SOURCE_OBJECTIVE_EVALUATOR_VERSION, evaluate_source_persisted),
            (
                SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION,
                _evaluate_source_persisted_legacy_v1_4,
            ),
        ):
            try:
                DEFAULT_OBJECTIVE_EVALUATORS.register(
                    evaluator_id,
                    version,
                    evaluator,
                )
            except ValueError as error:
                if "already registered" not in str(error):
                    raise


__all__ = [
    "PASSIVE_EVENT_PROJECTILE_APEX",
    "PASSIVE_EVENT_TASK_SURFACE_CONTACT",
    "SOURCE_OBJECTIVE_EVALUATOR_IDS",
    "SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION",
    "SOURCE_OBJECTIVE_EVALUATOR_VERSION",
    "evaluate_source_persisted",
    "evaluate_source_rows",
    "passive_event_semantics_for_evaluator",
    "passive_event_semantics_for_motion_kind",
    "register_source_objective_evaluators",
    "select_source_key_event",
]
