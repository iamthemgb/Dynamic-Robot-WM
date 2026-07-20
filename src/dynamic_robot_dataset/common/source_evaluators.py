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


SOURCE_OBJECTIVE_EVALUATOR_VERSION = "1.6.0"
SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION = "1.5.0"
SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION = "1.4.0"
SOURCE_OBJECTIVE_EVALUATOR_IDS = (
    "passive_freeflight_v1",
    "passive_projectile_v1",
    "passive_rebound_v1",
    "passive_rolling_v1",
    "rigid_catch_v2",
    "rigid_projectile_interception_v2",
    "rigid_ramp_launch_v1",
    "rigid_rebound_v2",
    "rigid_multi_rebound_v1",
    "rigid_arbitrary_rebound_v1",
    "rigid_dynamic_handoff_v1",
    "rigid_rolling_pickup_v1",
    "rigid_moving_base_catch_v1",
)
SOURCE_OBJECTIVE_EVALUATOR_LEGACY_IDS = (
    "passive_freeflight_v1",
    "passive_projectile_v1",
    "passive_rebound_v1",
    "passive_rolling_v1",
    "rigid_catch_v2",
    "rigid_projectile_interception_v2",
    "rigid_rebound_v2",
    "rigid_rolling_pickup_v1",
)

SURFACE_TRANSITION_SCHEMA_VERSION = "surface-to-free-flight/v1"
ORDERED_CONTACT_SCHEMA_VERSION = "ordered-surface-contacts/v1"
SAMPLED_SURFACE_SCHEMA_VERSION = "sampled-admitted-surface/v1"
ARBITRARY_SURFACE_CATALOG_VERSION = "rigid-arbitrary-surfaces/v1"

_V1_6_TASK_VARIANTS_BY_EVALUATOR: Mapping[str, frozenset[str]] = {
    "passive_freeflight_v1": frozenset({"nominal_freefall", "lateral_freefall"}),
    "passive_projectile_v1": frozenset({"ballistic_projectile", "angled_projectile"}),
    "passive_rebound_v1": frozenset({"table_bounce", "wall_rebound"}),
    "passive_rolling_v1": frozenset({"straight_roll", "slope_roll"}),
    "rigid_catch_v2": frozenset(
        {
            "catch_retain",
            "catch_transport",
            "off_center_catch",
            "off_center_near_miss",
            "drift_catch",
            "drift_near_miss",
            "mild_projectile_catch",
            "mild_projectile_near_miss",
        }
    ),
    "rigid_projectile_interception_v2": frozenset(
        {"direct_catch", "direct_deflection"}
    ),
    "rigid_ramp_launch_v1": frozenset(
        {"ramp_launch_catch", "ramp_launch_deflection"}
    ),
    "rigid_rebound_v2": frozenset(
        {"table_bounce", "floor_bounce", "wall_rebound", "angled_barrier_rebound"}
    ),
    "rigid_multi_rebound_v1": frozenset(
        {"floor_to_wall", "flight_to_table_bounce"}
    ),
    "rigid_arbitrary_rebound_v1": frozenset(
        {"random_plane_bounce", "random_barrier_bounce"}
    ),
    "rigid_dynamic_handoff_v1": frozenset({"platform_receive", "platform_handoff"}),
    "rigid_rolling_pickup_v1": frozenset(
        {"rolling_pickup", "rolling_pickup_transport"}
    ),
    "rigid_moving_base_catch_v1": frozenset(
        {"linear_base_catch", "oscillating_base_catch"}
    ),
}

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


def _evaluate_source_rows_v1_4_or_v1_5(
    *,
    evaluator_id: str,
    corpus_leaf_id: str,
    task_variant: str,
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    _evaluator_version: str,
) -> ObjectiveRecomputeResult:
    """Recompute a source outcome without branch intent or online labels."""

    if evaluator_id not in SOURCE_OBJECTIVE_EVALUATOR_IDS:
        raise ValueError(f"unsupported source objective evaluator {evaluator_id!r}")
    if _evaluator_version not in {
        SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION,
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
    if _evaluator_version == SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION:
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


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is outside its admitted range")
    return result


def _unit_vector_value(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must contain three finite numbers")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain three finite numbers") from error
    if len(vector) != 3 or any(not math.isfinite(item) for item in vector):
        raise ValueError(f"{label} must contain three finite numbers")
    norm = math.sqrt(sum(item * item for item in vector))
    if not 0.999 <= norm <= 1.001:
        raise ValueError(f"{label} must be a normalized world-frame vector")
    return tuple(item / norm for item in vector)  # type: ignore[return-value]


def _physics_mapping(source_spec: Mapping[str, Any]) -> Mapping[str, Any]:
    physics = source_spec.get("physics")
    if not isinstance(physics, Mapping):
        raise ValueError("source evaluator lacks scenario physics")
    return physics


def _surface_transition_contract(source_spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = _physics_mapping(source_spec).get("surface_transition_contract")
    if not isinstance(raw, Mapping):
        raise ValueError("ramp evaluator lacks surface_transition_contract")
    if raw.get("schema_version") != SURFACE_TRANSITION_SCHEMA_VERSION:
        raise ValueError("unsupported surface transition evaluator contract")
    surface_id = str(raw.get("support_surface_id") or "")
    if not surface_id:
        raise ValueError("surface transition contract lacks a stable support_surface_id")
    if raw.get("require_ordered_termination") is not True:
        raise ValueError("surface transition contract must require ordered termination")
    return {
        "schema_version": SURFACE_TRANSITION_SCHEMA_VERSION,
        "support_surface_id": surface_id,
        "support_normal_world_xyz": _unit_vector_value(
            raw.get("support_normal_world_xyz"),
            "surface transition support normal",
        ),
        "minimum_support_contact_s": _finite_number(
            raw.get("minimum_support_contact_s"),
            "minimum_support_contact_s",
            minimum=1e-9,
        ),
        "minimum_free_flight_s": _finite_number(
            raw.get("minimum_free_flight_s"),
            "minimum_free_flight_s",
            minimum=1e-9,
        ),
        "require_ordered_termination": True,
    }


def _ordered_contact_contract(source_spec: Mapping[str, Any]) -> dict[str, Any]:
    raw = _physics_mapping(source_spec).get("ordered_contact_contract")
    if not isinstance(raw, Mapping):
        raise ValueError("multi-contact evaluator lacks ordered_contact_contract")
    if raw.get("schema_version") != ORDERED_CONTACT_SCHEMA_VERSION:
        raise ValueError("unsupported ordered contact evaluator contract")
    raw_ids = raw.get("ordered_surface_ids")
    raw_normals = raw.get("ordered_surface_normals_world_xyz")
    if (
        not isinstance(raw_ids, Sequence)
        or isinstance(raw_ids, (str, bytes, bytearray))
        or not isinstance(raw_normals, Sequence)
        or isinstance(raw_normals, (str, bytes, bytearray))
    ):
        raise ValueError("ordered contact contract lacks IDs or normals")
    surface_ids = tuple(str(value) for value in raw_ids)
    if len(surface_ids) < 2 or any(not value for value in surface_ids):
        raise ValueError(
            "ordered multi-contact contract requires at least two surface IDs"
        )
    if len(surface_ids) != len(set(surface_ids)):
        raise ValueError("ordered contact contract repeats a surface ID")
    normals = tuple(
        _unit_vector_value(value, f"ordered surface normal {index}")
        for index, value in enumerate(raw_normals)
    )
    if len(normals) != len(surface_ids):
        raise ValueError("ordered contact IDs and normals differ in length")
    minimum_samples = raw.get("minimum_separated_pre_post_samples")
    if isinstance(minimum_samples, bool):
        raise ValueError("minimum separated sample count must be an integer")
    try:
        minimum_samples = int(minimum_samples)
    except (TypeError, ValueError) as error:
        raise ValueError("minimum separated sample count must be an integer") from error
    if minimum_samples < 2:
        raise ValueError("ordered contacts require at least two separated samples")
    if raw.get("reject_contact_chatter") is not True:
        raise ValueError("ordered contact contract must reject contact chatter")
    return {
        "schema_version": ORDERED_CONTACT_SCHEMA_VERSION,
        "ordered_surface_ids": surface_ids,
        "ordered_surface_normals_world_xyz": normals,
        "minimum_separated_pre_post_samples": minimum_samples,
        "minimum_inter_contact_free_flight_s": _finite_number(
            raw.get("minimum_inter_contact_free_flight_s"),
            "minimum_inter_contact_free_flight_s",
            minimum=1e-9,
        ),
        "reject_contact_chatter": True,
    }


def _sampled_surface_contract(
    source_spec: Mapping[str, Any],
    *,
    task_variant: str | None = None,
) -> dict[str, Any]:
    raw = _physics_mapping(source_spec).get("sampled_surface_contract")
    if not isinstance(raw, Mapping):
        raise ValueError("arbitrary-surface evaluator lacks sampled_surface_contract")
    if raw.get("schema_version") != SAMPLED_SURFACE_SCHEMA_VERSION:
        raise ValueError("unsupported sampled surface evaluator contract")
    if raw.get("catalog_version") != ARBITRARY_SURFACE_CATALOG_VERSION:
        raise ValueError("sampled surface uses an unadmitted geometry catalog")
    candidate_id = str(raw.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("sampled surface contract lacks candidate_id")
    source_seed = raw.get("source_seed")
    if (
        isinstance(source_seed, bool)
        or not isinstance(source_seed, int)
        or not 0 <= source_seed < 2**64
    ):
        raise ValueError("sampled surface source_seed must be a uint64")
    admission = raw.get("admission")
    admission_evidence = raw.get("admission_evidence_sha256")
    required_admission = (
        "grounded_supported",
        "reachability_checked",
        "swept_volume_clearance_checked",
        "background_clearance_checked",
        "calibrated_600_1200",
    )
    if (
        not isinstance(admission, Mapping)
        or set(admission) != set(required_admission)
        or any(not isinstance(admission.get(name), bool) for name in required_admission)
    ):
        raise ValueError("sampled surface admission evidence is malformed")
    admitted_names = {
        name for name in required_admission if admission.get(name) is True
    }
    if (
        not isinstance(admission_evidence, Mapping)
        or set(admission_evidence) != admitted_names
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in admission_evidence.values()
        )
    ):
        raise ValueError(
            "sampled surface admission is not hash-bound to calibration evidence"
        )
    def vector3(name: str, *, positive: bool = False) -> tuple[float, float, float]:
        value = raw.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError(f"sampled surface {name} must contain three values")
        result = tuple(_finite_number(item, f"sampled surface {name}") for item in value)
        if len(result) != 3 or (positive and any(item <= 0.0 for item in result)):
            raise ValueError(f"sampled surface {name} is invalid")
        return result  # type: ignore[return-value]
    contract_variant = str(raw.get("task_variant") or "")
    role = str(raw.get("role") or "")
    contact_profile = str(raw.get("contact_profile") or "")
    catalog_digest = str(raw.get("catalog_sha256") or "")
    result = {
        "schema_version": SAMPLED_SURFACE_SCHEMA_VERSION,
        "catalog_version": ARBITRARY_SURFACE_CATALOG_VERSION,
        "candidate_id": candidate_id,
        "task_variant": contract_variant,
        "role": role,
        "contact_profile": contact_profile,
        "catalog_sha256": catalog_digest,
        "source_seed": source_seed,
        "task_surface_id": f"owned_arbitrary_surface__{candidate_id}",
        "position_m": vector3("position_m"),
        "euler_rad": vector3("euler_rad"),
        "normal_world_xyz": _unit_vector_value(
            raw.get("normal_world_xyz"), "sampled surface normal"
        ),
        "half_size_m": vector3("half_size_m", positive=True),
        "admission": {name: admission[name] for name in required_admission},
        "admission_evidence_sha256": dict(admission_evidence),
    }
    variant = str(task_variant or source_spec.get("task_variant") or "")
    if variant not in {"random_plane_bounce", "random_barrier_bounce"}:
        raise ValueError("sampled surface contract lacks its exact task variant")
    # Replay the versioned PCG64 selection instead of trusting self-authored
    # candidate IDs, transforms, normals, or admission booleans in a scenario.
    # The import is intentionally local: the evaluator remains importable while
    # the source-MuJoCo package initializes, and resolution happens only during
    # validation/replay after module loading has completed.
    from ..backends.source_mujoco.rigid_breadth import (
        catalog_sha256,
        sample_surface_candidate,
    )

    expected = sample_surface_candidate(variant, source_seed=source_seed)
    expected_admission = {
        name: bool(getattr(expected.admission, name)) for name in required_admission
    }
    vectors_match = all(
        all(abs(float(left) - float(right)) <= 1e-12 for left, right in zip(actual, wanted))
        for actual, wanted in (
            (result["position_m"], expected.position_m),
            (result["euler_rad"], expected.euler_rad),
            (result["normal_world_xyz"], expected.normal_world_xyz),
            (result["half_size_m"], expected.half_size_m),
        )
    )
    if (
        result["candidate_id"] != expected.candidate_id
        or result["task_variant"] != variant
        or result["role"] != expected.role
        or result["contact_profile"] != expected.contact_profile
        or result["catalog_sha256"] != catalog_sha256()
        or not vectors_match
        or result["admission"] != expected_admission
        or result["admission_evidence_sha256"]
        != dict(expected.admission_evidence_sha256)
    ):
        raise ValueError(
            "sampled surface differs from deterministic catalog replay"
        )
    if not expected.admission.admitted:
        raise ValueError(
            f"sampled surface catalog candidate {expected.candidate_id} is not admitted"
        )
    source_hashes = source_spec.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or source_hashes.get("rigid_breadth_surface_catalog")
        != result["catalog_sha256"]
    ):
        raise ValueError("sampled surface catalog is not source-hash bound")
    return result


def _event_timestamp(row: Mapping[str, Any]) -> float | None:
    value = row.get("timestamp")
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _stable_surface_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("task_surface_id")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _surface_event_groups(
    event_rows: Sequence[Mapping[str, Any]],
    *,
    simulation_hz: float,
    before_time_s: float,
) -> list[dict[str, Any]]:
    rows = [
        (timestamp, row, _stable_surface_id(row))
        for row in event_rows
        if row.get("contact_category") == "task_surface"
        and (timestamp := _event_timestamp(row)) is not None
        and timestamp <= before_time_s + 1e-12
    ]
    rows.sort(key=lambda item: item[0])
    groups: list[dict[str, Any]] = []
    maximum_step_gap = 1.5 / simulation_hz
    for timestamp, row, surface_id in rows:
        if (
            not groups
            or groups[-1]["surface_id"] != surface_id
            or timestamp - groups[-1]["end_time_s"] > maximum_step_gap + 1e-12
        ):
            groups.append(
                {
                    "surface_id": surface_id,
                    "start_time_s": timestamp,
                    "end_time_s": timestamp,
                    "rows": [row],
                }
            )
        else:
            groups[-1]["end_time_s"] = timestamp
            groups[-1]["rows"].append(row)
    return groups


def _maximum_contiguous_free_flight_s(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    start_time_s: float,
    end_time_s: float,
    simulation_hz: float,
) -> float:
    times = sorted(
        timestamp
        for row in state_rows
        if (timestamp := _event_timestamp(row)) is not None
        and start_time_s - 1e-12 <= timestamp <= end_time_s + 1e-12
        and str(row.get("object.motion_mode") or row.get("motion_mode"))
        == "free_flight"
    )
    if not times:
        return 0.0
    maximum = run_start = previous = times[0]
    maximum_duration = 1.0 / simulation_hz
    for timestamp in times[1:]:
        if timestamp - previous > 1.5 / simulation_hz + 1e-12:
            run_start = timestamp
        previous = timestamp
        maximum_duration = max(
            maximum_duration,
            previous - run_start + 1.0 / simulation_hz,
        )
    return maximum_duration


def _normal_matches(
    rows: Sequence[Mapping[str, Any]], expected: Sequence[float]
) -> bool:
    measured: list[tuple[float, float, float]] = []
    for row in rows:
        value = row.get("normal_world")
        try:
            normal = _unit_vector_value(value, "persisted contact normal")
        except ValueError:
            continue
        measured.append(normal)
    return bool(
        measured
        and all(
            sum(a * b for a, b in zip(normal, expected)) >= 0.98
            for normal in measured
        )
    )


def _ramp_transition_evidence(
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _surface_transition_contract(source_spec)
    physics = _physics_mapping(source_spec)
    simulation_hz = _finite_number(
        physics.get("simulation_hz"), "simulation_hz", minimum=1e-9
    )
    _, planned_time = _key_event(source_spec)
    groups = _surface_event_groups(
        event_rows,
        simulation_hz=simulation_hz,
        before_time_s=planned_time,
    )
    expected_id = contract["support_surface_id"]
    ids = [group["surface_id"] for group in groups]
    support_groups = [group for group in groups if group["surface_id"] == expected_id]
    support = support_groups[0] if len(support_groups) == 1 else None
    contact_duration = (
        support["end_time_s"] - support["start_time_s"] + 1.0 / simulation_hz
        if support is not None
        else 0.0
    )
    free_flight_duration = (
        _maximum_contiguous_free_flight_s(
            state_rows,
            start_time_s=support["end_time_s"] + 0.5 / simulation_hz,
            end_time_s=planned_time,
            simulation_hz=simulation_hz,
        )
        if support is not None
        else 0.0
    )
    exact_surface_sequence = ids == [expected_id]
    normal_matches = bool(
        support
        and _normal_matches(
            support["rows"], contract["support_normal_world_xyz"]
        )
    )
    ordered_termination = bool(
        exact_surface_sequence
        and support is not None
        and contact_duration + 1e-12 >= contract["minimum_support_contact_s"]
        and free_flight_duration + 1e-12 >= contract["minimum_free_flight_s"]
    )
    robot_contact_guard_end_s = (
        support["end_time_s"] + contract["minimum_free_flight_s"]
        if support is not None
        else planned_time
    )
    premature_robot_contacts = [
        row
        for row in event_rows
        if row.get("contact_category") in {"gripper", "robot_arm"}
        and (timestamp := _event_timestamp(row)) is not None
        and timestamp <= robot_contact_guard_end_s + 1e-12
    ]
    no_premature_robot_contact = bool(
        support is not None and not premature_robot_contacts
    )
    return {
        "surface_transition_contract": {
            **contract,
            "support_normal_world_xyz": list(contract["support_normal_world_xyz"]),
        },
        "measured_surface_sequence": ids,
        "support_surface_identity_matches": exact_surface_sequence,
        "support_contact_normal_matches_contract": normal_matches,
        "support_contact_duration_s": contact_duration,
        "support_contact_duration_sufficient": bool(
            contact_duration + 1e-12 >= contract["minimum_support_contact_s"]
        ),
        "post_support_free_flight_duration_s": free_flight_duration,
        "post_support_ballistic_motion": bool(
            free_flight_duration + 1e-12 >= contract["minimum_free_flight_s"]
        ),
        "robot_contact_guard_end_s": robot_contact_guard_end_s,
        "premature_robot_contact_count": len(premature_robot_contacts),
        "no_robot_contact_before_ballistic_window": no_premature_robot_contact,
        "ordered_support_contact_termination": ordered_termination,
        "surface_transition_pass": bool(
            ordered_termination and normal_matches and no_premature_robot_contact
        ),
    }


def _windowed_rebound_measurement(
    state_rows: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
    *,
    window_start_s: float,
    window_end_s: float,
    object_radius_m: float,
    thresholds: ReboundAcceptanceThresholds,
) -> dict[str, Any]:
    windowed_states = [
        row
        for row in state_rows
        if (timestamp := _event_timestamp(row)) is not None
        and window_start_s - 1e-12 <= timestamp <= window_end_s + 1e-12
    ]
    return measure_rebound_kinematics(
        windowed_states,
        group["rows"],
        object_radius_m=object_radius_m,
        thresholds=thresholds,
    )


def _ordered_contact_evidence(
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _ordered_contact_contract(source_spec)
    physics = _physics_mapping(source_spec)
    simulation_hz = _finite_number(
        physics.get("simulation_hz"), "simulation_hz", minimum=1e-9
    )
    thresholds, object_radius_m = _bound_rebound_acceptance(source_spec)
    _, planned_time = _key_event(source_spec)
    groups = _surface_event_groups(
        event_rows,
        simulation_hz=simulation_hz,
        before_time_s=planned_time,
    )
    ids = [group["surface_id"] for group in groups]
    expected_ids = list(contract["ordered_surface_ids"])
    sequence_matches = ids == expected_ids
    chatter_rejected = bool(sequence_matches and len(ids) == len(set(ids)))
    normal_matches: list[bool] = []
    rebound_measurements: list[dict[str, Any]] = []
    separated_sample_counts: list[dict[str, int]] = []
    inter_contact_free_flight: list[float] = []
    for index, group in enumerate(groups[: len(expected_ids)]):
        expected_normal = contract["ordered_surface_normals_world_xyz"][index]
        normal_matches.append(_normal_matches(group["rows"], expected_normal))
        previous_end = groups[index - 1]["end_time_s"] if index else 0.0
        next_start = (
            groups[index + 1]["start_time_s"]
            if index + 1 < len(groups)
            else planned_time
        )
        measurement = _windowed_rebound_measurement(
            state_rows,
            group,
            window_start_s=previous_end,
            window_end_s=next_start,
            object_radius_m=object_radius_m,
            thresholds=thresholds,
        )
        rebound_measurements.append(measurement)
        before_count = sum(
            (timestamp := _event_timestamp(row)) is not None
            and previous_end - 1e-12 <= timestamp < group["start_time_s"]
            and int(row.get("contact.count", 0)) == 0
            for row in state_rows
        )
        after_count = sum(
            (timestamp := _event_timestamp(row)) is not None
            and group["end_time_s"] < timestamp <= next_start + 1e-12
            and int(row.get("contact.count", 0)) == 0
            for row in state_rows
        )
        separated_sample_counts.append({"pre": before_count, "post": after_count})
        if index + 1 < len(groups):
            inter_contact_free_flight.append(
                _maximum_contiguous_free_flight_s(
                    state_rows,
                    start_time_s=group["end_time_s"] + 0.5 / simulation_hz,
                    end_time_s=next_start - 0.5 / simulation_hz,
                    simulation_hz=simulation_hz,
                )
            )
    minimum_samples = contract["minimum_separated_pre_post_samples"]
    samples_pass = bool(
        len(separated_sample_counts) == len(expected_ids)
        and all(
            value["pre"] >= minimum_samples and value["post"] >= minimum_samples
            for value in separated_sample_counts
        )
    )
    free_flight_pass = bool(
        len(inter_contact_free_flight) == max(0, len(expected_ids) - 1)
        and all(
            value + 1e-12 >= contract["minimum_inter_contact_free_flight_s"]
            for value in inter_contact_free_flight
        )
    )
    rebounds_pass = bool(
        len(rebound_measurements) == len(expected_ids)
        and all(value.get("rebound_acceptance_pass") is True for value in rebound_measurements)
    )
    sequence_completion_time_s = (
        groups[-1]["end_time_s"] if sequence_matches and groups else None
    )
    tool_contact_guard_end_s = (
        sequence_completion_time_s
        + contract["minimum_separated_pre_post_samples"] / simulation_hz
        if sequence_completion_time_s is not None
        else planned_time
    )
    premature_tool_contacts = [
        row
        for row in event_rows
        if row.get("contact_category") in {"gripper", "robot_arm"}
        and (timestamp := _event_timestamp(row)) is not None
        and timestamp <= tool_contact_guard_end_s + 1e-12
    ]
    no_premature_tool_contact = bool(
        sequence_completion_time_s is not None and not premature_tool_contacts
    )
    return {
        "ordered_contact_contract": {
            **contract,
            "ordered_surface_ids": expected_ids,
            "ordered_surface_normals_world_xyz": [
                list(value)
                for value in contract["ordered_surface_normals_world_xyz"]
            ],
        },
        "measured_surface_sequence": ids,
        "ordered_surface_contact_sequence_matches": sequence_matches,
        "contact_chatter_rejected": chatter_rejected,
        "surface_contact_normals_match_contract": bool(
            len(normal_matches) == len(expected_ids) and all(normal_matches)
        ),
        "separated_sample_counts": separated_sample_counts,
        "separated_pre_post_contact_samples": samples_pass,
        "inter_contact_free_flight_s": inter_contact_free_flight,
        "inter_contact_free_flight_sufficient": free_flight_pass,
        "rebound_measurements": rebound_measurements,
        "ordered_rebound_kinematics_pass": rebounds_pass,
        "ordered_sequence_completion_time_s": sequence_completion_time_s,
        "tool_contact_guard_end_s": tool_contact_guard_end_s,
        "premature_tool_contact_count": len(premature_tool_contacts),
        "no_tool_contact_before_ordered_rebound_completion": (
            no_premature_tool_contact
        ),
        "ordered_contact_sequence_pass": bool(
            sequence_matches
            and chatter_rejected
            and len(normal_matches) == len(expected_ids)
            and all(normal_matches)
            and samples_pass
            and free_flight_pass
            and rebounds_pass
            and no_premature_tool_contact
        ),
    }


def _sampled_surface_evidence(
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    task_variant: str | None = None,
) -> dict[str, Any]:
    contract = _sampled_surface_contract(source_spec, task_variant=task_variant)
    thresholds, object_radius_m = _bound_rebound_acceptance(source_spec)
    _, planned_time = _key_event(source_spec)
    matching = [
        row
        for row in event_rows
        if row.get("contact_category") == "task_surface"
        and (timestamp := _event_timestamp(row)) is not None
        and timestamp <= planned_time + 1e-12
        and _stable_surface_id(row) == contract["task_surface_id"]
    ]
    all_surface_ids = {
        _stable_surface_id(row)
        for row in event_rows
        if row.get("contact_category") == "task_surface"
        and (timestamp := _event_timestamp(row)) is not None
        and timestamp <= planned_time + 1e-12
    }
    identity_matches = all_surface_ids == {contract["task_surface_id"]}
    normal_matches = _normal_matches(matching, contract["normal_world_xyz"])
    roll, pitch, yaw = contract["euler_rad"]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    face_tolerance_m = 0.006
    contact_points_on_sampled_face: list[bool] = []
    for row in matching:
        point = row.get("point_world_m")
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes, bytearray))
            or len(point) != 3
        ):
            contact_points_on_sampled_face.append(False)
            continue
        try:
            delta = tuple(
                float(point[index]) - contract["position_m"][index]
                for index in range(3)
            )
        except (TypeError, ValueError):
            contact_points_on_sampled_face.append(False)
            continue
        local = tuple(
            sum(rotation[world_axis][local_axis] * delta[world_axis] for world_axis in range(3))
            for local_axis in range(3)
        )
        half = contract["half_size_m"]
        if contract["role"] == "table":
            on_face = bool(
                abs(local[0]) <= half[0] + face_tolerance_m
                and abs(local[1]) <= half[1] + face_tolerance_m
                and abs(local[2] - half[2]) <= face_tolerance_m
            )
        else:
            on_face = bool(
                abs(local[0]) <= half[0] + face_tolerance_m
                and abs(local[2]) <= half[2] + face_tolerance_m
                and abs(local[1] + half[1]) <= face_tolerance_m
            )
        contact_points_on_sampled_face.append(on_face)
    points_match = bool(
        contact_points_on_sampled_face and all(contact_points_on_sampled_face)
    )
    rebound = measure_rebound_kinematics(
        state_rows,
        matching,
        object_radius_m=object_radius_m,
        thresholds=thresholds,
    )
    return {
        "sampled_surface_contract": {
            **contract,
            "position_m": list(contract["position_m"]),
            "euler_rad": list(contract["euler_rad"]),
            "normal_world_xyz": list(contract["normal_world_xyz"]),
            "half_size_m": list(contract["half_size_m"]),
        },
        "sampled_surface_identity_matches": identity_matches,
        "sampled_surface_contact_normal_matches": normal_matches,
        "sampled_surface_contact_point_count": len(contact_points_on_sampled_face),
        "sampled_surface_contact_points_within_face": points_match,
        "sampled_surface_rebound": rebound,
        "sampled_surface_rebound_pass": bool(
            identity_matches
            and normal_matches
            and points_match
            and rebound.get("rebound_acceptance_pass") is True
        ),
    }


def _deflection_result_v1_6(
    *,
    evaluator_id: str,
    corpus_leaf_id: str,
    task_variant: str,
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> ObjectiveRecomputeResult:
    if not state_rows:
        raise ValueError("source objective replay requires persisted state rows")
    physics = _physics_mapping(source_spec)
    simulation_hz = _finite_number(
        physics.get("simulation_hz"), "simulation_hz", minimum=1e-9
    )
    mass = _finite_number(physics.get("object_mass_kg"), "object_mass_kg", minimum=1e-9)
    gravity_raw = physics.get("gravity_m_s2", (0.0, 0.0, -9.81))
    if not isinstance(gravity_raw, Sequence) or isinstance(
        gravity_raw, (str, bytes, bytearray)
    ):
        raise ValueError("deflection evaluator lacks gravity_m_s2")
    gravity = tuple(_finite_number(value, "gravity_m_s2") for value in gravity_raw)
    if len(gravity) != 3:
        raise ValueError("deflection evaluator gravity_m_s2 must contain three values")
    planned_name, planned_time = _key_event(source_spec)
    selected = select_source_key_event(
        planned_key_event_name=planned_name,
        planned_key_event_time_s=planned_time,
        state_rows=state_rows,
        event_rows=event_rows,
        passive=False,
        contact_time_tolerance_s=1.0 / simulation_hz,
    )
    invalid_penetration, maximum_penetration = _invalid_penetration(event_rows)
    common = {
        "evaluator_id": evaluator_id,
        "evaluator_dispatch": "deflection",
        "corpus_leaf_id": corpus_leaf_id,
        "task_variant": task_variant,
        "state_sample_count": len(state_rows),
        "contact_sample_count": len(event_rows),
        "maximum_penetration_depth_m": maximum_penetration,
        "branch_intent_read": False,
    }
    if invalid_penetration:
        return _invalid_result(
            evidence={**common, "penetration_within_limits": False},
            key_event_name=str(selected["key_event_name"]),
            key_event_time_s=float(selected["key_event_time_s"]),
        )
    states = sorted(
        (
            (timestamp, row)
            for row in state_rows
            if (timestamp := _event_timestamp(row)) is not None
            and _vector(row, "object.position") is not None
            and _vector(row, "object.linear_velocity") is not None
        ),
        key=lambda item: item[0],
    )
    tool_events = [
        row
        for row in event_rows
        if row.get("contact_category") in {"gripper", "robot_arm"}
        and _event_timestamp(row) is not None
    ]
    tool_events.sort(key=lambda row: float(row["timestamp"]))
    contact_occurred = bool(tool_events)
    pre: tuple[float, Mapping[str, Any]] | None = None
    post: tuple[float, Mapping[str, Any]] | None = None
    if tool_events:
        contact_start = float(tool_events[0]["timestamp"])
        contact_end = float(tool_events[-1]["timestamp"])
        pre_candidates = [
            item
            for item in states
            if item[0] < contact_start and int(item[1].get("contact.count", 0)) == 0
        ]
        post_candidates = [
            item
            for item in states
            if item[0] > contact_end
            and int(item[1].get("contact.count", 0)) == 0
            and item[0] - contact_end >= 1.0 / simulation_hz - 1e-12
        ]
        pre = pre_candidates[-1] if pre_candidates else None
        post = post_candidates[0] if post_candidates else None
    delta_momentum: tuple[float, float, float] | None = None
    measured_impulse = [0.0, 0.0, 0.0]
    for row in tool_events:
        normal = row.get("normal_world")
        impulse = row.get("normal_impulse_n_s")
        try:
            unit = _unit_vector_value(normal, "deflection contact normal")
            magnitude = _finite_number(impulse, "deflection normal impulse", minimum=0.0)
        except ValueError:
            continue
        for axis in range(3):
            measured_impulse[axis] += unit[axis] * magnitude
    redirect_angle_deg: float | None = None
    momentum_impulse_ratio: float | None = None
    impulse_alignment: float | None = None
    redirected = False
    impulse_consistent = False
    if pre is not None and post is not None:
        pre_velocity = _vector(pre[1], "object.linear_velocity")
        post_velocity = _vector(post[1], "object.linear_velocity")
        assert pre_velocity is not None and post_velocity is not None
        pre_speed = math.sqrt(sum(value * value for value in pre_velocity))
        post_speed = math.sqrt(sum(value * value for value in post_velocity))
        if pre_speed > 1e-6 and post_speed >= 0.15:
            cosine = max(
                -1.0,
                min(
                    1.0,
                    sum(a * b for a, b in zip(pre_velocity, post_velocity))
                    / (pre_speed * post_speed),
                ),
            )
            redirect_angle_deg = math.degrees(math.acos(cosine))
            redirected = cosine < 0.7
        dt = post[0] - pre[0]
        delta_momentum = tuple(
            mass * (after - before - acceleration * dt)
            for before, after, acceleration in zip(pre_velocity, post_velocity, gravity)
        )
        delta_norm = math.sqrt(sum(value * value for value in delta_momentum))
        impulse_norm = math.sqrt(sum(value * value for value in measured_impulse))
        if delta_norm > 1e-12 and impulse_norm > 1e-12:
            impulse_alignment = sum(
                a * b for a, b in zip(delta_momentum, measured_impulse)
            ) / (delta_norm * impulse_norm)
            momentum_impulse_ratio = delta_norm / impulse_norm
            impulse_consistent = bool(
                impulse_alignment >= 0.5 and 0.25 <= momentum_impulse_ratio <= 4.0
            )
    finite_state = len(states) == len(state_rows)
    success = bool(finite_state and contact_occurred and redirected and impulse_consistent)
    evidence = {
        **common,
        "finite_state": finite_state,
        "penetration_within_limits": True,
        "deflection_contact_occurred": contact_occurred,
        "separated_pre_post_contact_samples": pre is not None and post is not None,
        "deflection_redirect_angle_deg": redirect_angle_deg,
        "object_redirected_by_hand_contact": redirected,
        "measured_contact_impulse_world_n_s": measured_impulse,
        "gravity_compensated_momentum_change_n_s": (
            None if delta_momentum is None else list(delta_momentum)
        ),
        "momentum_to_measured_impulse_ratio": momentum_impulse_ratio,
        "momentum_impulse_alignment_cosine": impulse_alignment,
        "velocity_change_matches_measured_contact_impulse": impulse_consistent,
        "planned_key_event_name": planned_name,
        "planned_key_event_time_s": planned_time,
        "measured_key_event_source": selected["key_event_source"],
    }
    if not finite_state:
        outcome = ActualOutcomeClass.INVALID
        failure = "unstable_physics"
    elif success:
        outcome = ActualOutcomeClass.SUCCESS
        failure = "none"
    elif contact_occurred:
        outcome = ActualOutcomeClass.CONTACT_FAILURE
        failure = "contact_without_completion"
    else:
        outcome = ActualOutcomeClass.MISS
        failure = "no_contact"
    return ObjectiveRecomputeResult(
        task_success=success,
        actual_outcome_class=outcome,
        primary_failure_code=failure,
        evidence=evidence,
        key_event_name=str(selected["key_event_name"]),
        key_event_time_s=float(selected["key_event_time_s"]),
    )


def _combine_prerequisite(
    base: ObjectiveRecomputeResult,
    *,
    dispatch: str,
    prerequisite_name: str,
    prerequisite_passed: bool,
    prerequisite_evidence: Mapping[str, Any],
) -> ObjectiveRecomputeResult:
    evidence = {
        **dict(base.evidence),
        "evaluator_dispatch": dispatch,
        prerequisite_name: prerequisite_passed,
        **dict(prerequisite_evidence),
    }
    if prerequisite_passed:
        return ObjectiveRecomputeResult(
            task_success=base.task_success,
            actual_outcome_class=base.actual_outcome_class,
            primary_failure_code=base.primary_failure_code,
            evidence=evidence,
            key_event_name=base.key_event_name,
            key_event_time_s=base.key_event_time_s,
        )
    return ObjectiveRecomputeResult(
        task_success=False,
        actual_outcome_class=ActualOutcomeClass.INVALID,
        primary_failure_code="unstable_physics",
        evidence=evidence,
        key_event_name=base.key_event_name,
        key_event_time_s=base.key_event_time_s,
    )


def _catch_result_v1_6(
    *,
    evaluator_id: str,
    corpus_leaf_id: str,
    task_variant: str,
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> ObjectiveRecomputeResult:
    result = _evaluate_source_rows_v1_4_or_v1_5(
        evaluator_id=evaluator_id,
        corpus_leaf_id=corpus_leaf_id,
        task_variant=task_variant,
        source_spec=source_spec,
        state_rows=state_rows,
        event_rows=event_rows,
        _evaluator_version=SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION,
    )
    return ObjectiveRecomputeResult(
        task_success=result.task_success,
        actual_outcome_class=result.actual_outcome_class,
        primary_failure_code=result.primary_failure_code,
        evidence={**dict(result.evidence), "evaluator_dispatch": "catch_retention"},
        key_event_name=result.key_event_name,
        key_event_time_s=result.key_event_time_s,
    )


def _evaluate_source_rows_v1_6(
    *,
    evaluator_id: str,
    corpus_leaf_id: str,
    task_variant: str,
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> ObjectiveRecomputeResult:
    admitted_variants = _V1_6_TASK_VARIANTS_BY_EVALUATOR.get(evaluator_id)
    if admitted_variants is None or task_variant not in admitted_variants:
        raise ValueError(
            f"{evaluator_id} has no v1.6 evaluator dispatch for task variant "
            f"{task_variant!r}"
        )
    if evaluator_id.startswith("passive_"):
        result = _evaluate_source_rows_v1_4_or_v1_5(
            evaluator_id=evaluator_id,
            corpus_leaf_id=corpus_leaf_id,
            task_variant=task_variant,
            source_spec=source_spec,
            state_rows=state_rows,
            event_rows=event_rows,
            _evaluator_version=SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION,
        )
        return ObjectiveRecomputeResult(
            task_success=result.task_success,
            actual_outcome_class=result.actual_outcome_class,
            primary_failure_code=result.primary_failure_code,
            evidence={**dict(result.evidence), "evaluator_dispatch": "passive_observation"},
            key_event_name=result.key_event_name,
            key_event_time_s=result.key_event_time_s,
        )
    if evaluator_id == "rigid_projectile_interception_v2":
        if task_variant == "direct_catch":
            return _catch_result_v1_6(
                evaluator_id=evaluator_id,
                corpus_leaf_id=corpus_leaf_id,
                task_variant=task_variant,
                source_spec=source_spec,
                state_rows=state_rows,
                event_rows=event_rows,
            )
        if task_variant == "direct_deflection":
            return _deflection_result_v1_6(
                evaluator_id=evaluator_id,
                corpus_leaf_id=corpus_leaf_id,
                task_variant=task_variant,
                source_spec=source_spec,
                state_rows=state_rows,
                event_rows=event_rows,
            )
        raise ValueError(
            "rigid_projectile_interception_v2 has no evaluator dispatch for "
            f"task variant {task_variant!r}"
        )
    if evaluator_id == "rigid_ramp_launch_v1":
        if task_variant not in {"ramp_launch_catch", "ramp_launch_deflection"}:
            raise ValueError(
                f"rigid_ramp_launch_v1 has no evaluator dispatch for {task_variant!r}"
            )
        base = (
            _deflection_result_v1_6(
                evaluator_id=evaluator_id,
                corpus_leaf_id=corpus_leaf_id,
                task_variant=task_variant,
                source_spec=source_spec,
                state_rows=state_rows,
                event_rows=event_rows,
            )
            if task_variant.endswith("deflection")
            else _catch_result_v1_6(
                evaluator_id=evaluator_id,
                corpus_leaf_id=corpus_leaf_id,
                task_variant=task_variant,
                source_spec=source_spec,
                state_rows=state_rows,
                event_rows=event_rows,
            )
        )
        transition = _ramp_transition_evidence(source_spec, state_rows, event_rows)
        return _combine_prerequisite(
            base,
            dispatch="ramp_transition_then_" + (
                "deflection" if task_variant.endswith("deflection") else "catch_retention"
            ),
            prerequisite_name="surface_transition_pass",
            prerequisite_passed=transition["surface_transition_pass"] is True,
            prerequisite_evidence=transition,
        )
    if evaluator_id == "rigid_rebound_v2":
        base = _catch_result_v1_6(
            evaluator_id=evaluator_id,
            corpus_leaf_id=corpus_leaf_id,
            task_variant=task_variant,
            source_spec=source_spec,
            state_rows=state_rows,
            event_rows=event_rows,
        )
        thresholds, object_radius_m = _bound_rebound_acceptance(source_spec)
        rebound = measure_rebound_kinematics(
            state_rows,
            event_rows,
            object_radius_m=object_radius_m,
            thresholds=thresholds,
        )
        return _combine_prerequisite(
            base,
            dispatch="rebound_then_catch_retention",
            prerequisite_name="rebound_acceptance_pass",
            prerequisite_passed=rebound["rebound_acceptance_pass"] is True,
            prerequisite_evidence={"rebound": rebound},
        )
    if evaluator_id == "rigid_multi_rebound_v1":
        if task_variant not in {"floor_to_wall", "flight_to_table_bounce"}:
            raise ValueError(
                f"rigid_multi_rebound_v1 has no evaluator dispatch for {task_variant!r}"
            )
        base = _catch_result_v1_6(
            evaluator_id=evaluator_id,
            corpus_leaf_id=corpus_leaf_id,
            task_variant=task_variant,
            source_spec=source_spec,
            state_rows=state_rows,
            event_rows=event_rows,
        )
        ordered = _ordered_contact_evidence(source_spec, state_rows, event_rows)
        return _combine_prerequisite(
            base,
            dispatch="ordered_multi_rebound_then_catch_retention",
            prerequisite_name="ordered_contact_sequence_pass",
            prerequisite_passed=ordered["ordered_contact_sequence_pass"] is True,
            prerequisite_evidence=ordered,
        )
    if evaluator_id == "rigid_arbitrary_rebound_v1":
        if task_variant not in {"random_plane_bounce", "random_barrier_bounce"}:
            raise ValueError(
                f"rigid_arbitrary_rebound_v1 has no evaluator dispatch for {task_variant!r}"
            )
        base = _catch_result_v1_6(
            evaluator_id=evaluator_id,
            corpus_leaf_id=corpus_leaf_id,
            task_variant=task_variant,
            source_spec=source_spec,
            state_rows=state_rows,
            event_rows=event_rows,
        )
        sampled = _sampled_surface_evidence(
            source_spec,
            state_rows,
            event_rows,
            task_variant=task_variant,
        )
        return _combine_prerequisite(
            base,
            dispatch="sampled_surface_rebound_then_catch_retention",
            prerequisite_name="sampled_surface_rebound_pass",
            prerequisite_passed=sampled["sampled_surface_rebound_pass"] is True,
            prerequisite_evidence=sampled,
        )
    catch_dispatch = {
        "rigid_catch_v2": "catch_retention",
        "rigid_rolling_pickup_v1": "rolling_pickup_then_catch_retention",
        "rigid_dynamic_handoff_v1": "dynamic_handoff_then_catch_retention",
        "rigid_moving_base_catch_v1": "moving_base_catch_retention",
    }
    if evaluator_id in catch_dispatch:
        result = _catch_result_v1_6(
            evaluator_id=evaluator_id,
            corpus_leaf_id=corpus_leaf_id,
            task_variant=task_variant,
            source_spec=source_spec,
            state_rows=state_rows,
            event_rows=event_rows,
        )
        return ObjectiveRecomputeResult(
            task_success=result.task_success,
            actual_outcome_class=result.actual_outcome_class,
            primary_failure_code=result.primary_failure_code,
            evidence={**dict(result.evidence), "evaluator_dispatch": catch_dispatch[evaluator_id]},
            key_event_name=result.key_event_name,
            key_event_time_s=result.key_event_time_s,
        )
    raise ValueError(
        f"source evaluator {evaluator_id!r} has no explicit v1.6 dispatch"
    )


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
    """Replay a versioned source objective with no generic task fallback."""

    if evaluator_id not in SOURCE_OBJECTIVE_EVALUATOR_IDS:
        raise ValueError(f"unsupported source objective evaluator {evaluator_id!r}")
    if _evaluator_version == SOURCE_OBJECTIVE_EVALUATOR_VERSION:
        return _evaluate_source_rows_v1_6(
            evaluator_id=evaluator_id,
            corpus_leaf_id=corpus_leaf_id,
            task_variant=task_variant,
            source_spec=source_spec,
            state_rows=state_rows,
            event_rows=event_rows,
        )
    if _evaluator_version in {
        SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION,
        SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION,
    }:
        if evaluator_id not in SOURCE_OBJECTIVE_EVALUATOR_LEGACY_IDS:
            raise ValueError(
                f"{evaluator_id!r} was not registered in source objective "
                f"version {_evaluator_version}"
            )
        return _evaluate_source_rows_v1_4_or_v1_5(
            evaluator_id=evaluator_id,
            corpus_leaf_id=corpus_leaf_id,
            task_variant=task_variant,
            source_spec=source_spec,
            state_rows=state_rows,
            event_rows=event_rows,
            _evaluator_version=_evaluator_version,
        )
    raise ValueError(f"unsupported source objective version {_evaluator_version!r}")


def validate_source_evaluator_contract(
    *,
    evaluator_id: str,
    task_variant: str,
    source_spec: Mapping[str, Any],
) -> None:
    """Fail closed when a v1.6 scenario omits its version-owned contract."""

    if evaluator_id not in SOURCE_OBJECTIVE_EVALUATOR_IDS:
        raise ValueError(f"unsupported source objective evaluator {evaluator_id!r}")
    if evaluator_id.startswith("passive_"):
        return
    if evaluator_id == "rigid_projectile_interception_v2" and task_variant not in {
        "direct_catch",
        "direct_deflection",
    }:
        raise ValueError(
            f"rigid_projectile_interception_v2 has no dispatch for {task_variant!r}"
        )
    if evaluator_id == "rigid_ramp_launch_v1":
        if task_variant not in {"ramp_launch_catch", "ramp_launch_deflection"}:
            raise ValueError(f"rigid_ramp_launch_v1 has no dispatch for {task_variant!r}")
        _surface_transition_contract(source_spec)
    elif evaluator_id == "rigid_rebound_v2":
        _bound_rebound_acceptance(source_spec)
    elif evaluator_id == "rigid_multi_rebound_v1":
        if task_variant not in {"floor_to_wall", "flight_to_table_bounce"}:
            raise ValueError(f"rigid_multi_rebound_v1 has no dispatch for {task_variant!r}")
        _bound_rebound_acceptance(source_spec)
        _ordered_contact_contract(source_spec)
    elif evaluator_id == "rigid_arbitrary_rebound_v1":
        if task_variant not in {"random_plane_bounce", "random_barrier_bounce"}:
            raise ValueError(
                f"rigid_arbitrary_rebound_v1 has no dispatch for {task_variant!r}"
            )
        _bound_rebound_acceptance(source_spec)
        _sampled_surface_contract(source_spec, task_variant=task_variant)
    _bound_grasp_retention(source_spec)


def source_contract_evidence(
    *,
    evaluator_id: str,
    source_spec: Mapping[str, Any],
    state_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure v1.6 prerequisite physics for online QC and persisted replay."""

    if evaluator_id == "rigid_ramp_launch_v1":
        return _ramp_transition_evidence(source_spec, state_rows, event_rows)
    if evaluator_id == "rigid_rebound_v2":
        thresholds, object_radius_m = _bound_rebound_acceptance(source_spec)
        return {
            "rebound": measure_rebound_kinematics(
                state_rows,
                event_rows,
                object_radius_m=object_radius_m,
                thresholds=thresholds,
            )
        }
    if evaluator_id == "rigid_multi_rebound_v1":
        return _ordered_contact_evidence(source_spec, state_rows, event_rows)
    if evaluator_id == "rigid_arbitrary_rebound_v1":
        return _sampled_surface_evidence(source_spec, state_rows, event_rows)
    return {}


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


def _evaluate_source_persisted_v1_5(
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
        _evaluator_version=SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION,
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
        registrations = [(SOURCE_OBJECTIVE_EVALUATOR_VERSION, evaluate_source_persisted)]
        if evaluator_id in SOURCE_OBJECTIVE_EVALUATOR_LEGACY_IDS:
            registrations.extend(
                (
                    (
                        SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION,
                        _evaluate_source_persisted_v1_5,
                    ),
                    (
                        SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION,
                        _evaluate_source_persisted_legacy_v1_4,
                    ),
                )
            )
        for version, evaluator in registrations:
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
    "SOURCE_OBJECTIVE_EVALUATOR_LEGACY_IDS",
    "SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION",
    "SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION",
    "SOURCE_OBJECTIVE_EVALUATOR_VERSION",
    "evaluate_source_persisted",
    "evaluate_source_rows",
    "passive_event_semantics_for_evaluator",
    "passive_event_semantics_for_motion_kind",
    "register_source_objective_evaluators",
    "select_source_key_event",
    "source_contract_evidence",
    "validate_source_evaluator_contract",
]
