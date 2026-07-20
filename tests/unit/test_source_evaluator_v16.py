from __future__ import annotations

from copy import deepcopy

import pytest

from dynamic_robot_dataset.common.contract_v2 import DEFAULT_OBJECTIVE_EVALUATORS
from dynamic_robot_dataset.common.corpus_registry import load_corpus_registry
from dynamic_robot_dataset.common.grasp_retention import DEFAULT_GRASP_RETENTION
from dynamic_robot_dataset.common.rebound import DEFAULT_REBOUND_ACCEPTANCE
from dynamic_robot_dataset.common.schema import ActualOutcomeClass
from dynamic_robot_dataset.common.source_evaluators import (
    SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION,
    SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION,
    SOURCE_OBJECTIVE_EVALUATOR_VERSION,
    evaluate_source_rows,
)


def _physics(*, duration_s: float, key_event_time_s: float) -> dict:
    return {
        "duration_s": duration_s,
        "physics": {
            "simulation_hz": 100,
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "object_mass_kg": 0.1,
            "object_radius_m": 0.0245,
            "key_event_name": "interception",
            "key_event_time_s": key_event_time_s,
            "grasp_retention": DEFAULT_GRASP_RETENTION.to_dict(),
            "rebound_acceptance": DEFAULT_REBOUND_ACCEPTANCE.to_dict(),
        },
    }


def _retained_rows(*, duration_s: float, catch_start_s: float) -> list[dict]:
    rows = []
    for index in range(round(duration_s * 100) + 1):
        timestamp = index / 100
        bilateral = timestamp >= catch_start_s
        rows.append(
            {
                "timestamp": timestamp,
                "object.position": [0.5, 0.0, 0.5],
                "object.linear_velocity": [0.0, 0.0, 0.0],
                "object.motion_mode": "gripper_contact" if bilateral else "free_flight",
                "contact.count": 2 if bilateral else 0,
                "contact.bilateral": bilateral,
                "grasp.center_position": [0.5, 0.0, 0.5],
            }
        )
    return rows


def _event(
    timestamp: float,
    *,
    category: str,
    normal: list[float],
    surface_id: str | None = None,
    impulse: float = 0.0,
) -> dict:
    result = {
        "timestamp": timestamp,
        "contact_category": category,
        "penetration_depth_m": 0.0005,
        "normal_world": normal,
        "normal_impulse_n_s": impulse,
    }
    if surface_id is not None:
        result["task_surface_id"] = surface_id
    return result


def test_v16_deflection_uses_measured_impulse_instead_of_catch_fallback() -> None:
    spec = _physics(duration_s=0.4, key_event_time_s=0.2)
    rows = [
        {
            "timestamp": timestamp,
            "object.position": [timestamp, 0.0, 0.5],
            "object.linear_velocity": velocity,
            "object.motion_mode": mode,
            "contact.count": count,
            "contact.bilateral": False,
        }
        for timestamp, velocity, mode, count in (
            (0.0, [-1.0, 0.0, 0.0], "free_flight", 0),
            (0.1, [-1.0, 0.0, 0.0], "free_flight", 0),
            (0.2, [0.0, 0.0, 0.0], "gripper_contact", 1),
            (0.3, [1.0, 0.0, 0.0], "free_flight", 0),
            (0.4, [1.0, 0.0, 0.0], "free_flight", 0),
        )
    ]
    events = [
        _event(
            0.2,
            category="robot_arm",
            normal=[1.0, 0.0, 0.0],
            impulse=0.2,
        )
    ]

    result = evaluate_source_rows(
        evaluator_id="rigid_projectile_interception_v2",
        corpus_leaf_id="F2a",
        task_variant="direct_deflection",
        source_spec=spec,
        state_rows=rows,
        event_rows=events,
    )

    assert SOURCE_OBJECTIVE_EVALUATOR_VERSION == "1.6.0"
    assert result.task_success is True
    assert result.evidence["evaluator_dispatch"] == "deflection"
    assert result.evidence["object_redirected_by_hand_contact"] is True
    assert result.evidence["velocity_change_matches_measured_contact_impulse"] is True

    no_impulse = deepcopy(events)
    no_impulse[0]["normal_impulse_n_s"] = 0.0
    failed = evaluate_source_rows(
        evaluator_id="rigid_projectile_interception_v2",
        corpus_leaf_id="F2a",
        task_variant="direct_deflection",
        source_spec=spec,
        state_rows=rows,
        event_rows=no_impulse,
    )
    assert failed.task_success is False
    assert failed.actual_outcome_class is ActualOutcomeClass.CONTACT_FAILURE
    assert failed.evidence["velocity_change_matches_measured_contact_impulse"] is False


def test_every_source_mujoco_leaf_evaluator_is_registered_at_v16() -> None:
    evaluator_ids = {
        leaf.evaluator
        for leaf in load_corpus_registry().leaves
        if leaf.backend == "source_mujoco"
    }
    assert evaluator_ids
    assert all(
        DEFAULT_OBJECTIVE_EVALUATORS.get(evaluator_id, "1.6.0") is not None
        for evaluator_id in evaluator_ids
    )


def test_v16_rejects_unknown_non_passive_dispatch() -> None:
    with pytest.raises(ValueError, match="no v1.6 evaluator dispatch"):
        evaluate_source_rows(
            evaluator_id="rigid_catch_v2",
            corpus_leaf_id="F1a",
            task_variant="invented_catch",
            source_spec=_physics(duration_s=0.2, key_event_time_s=0.1),
            state_rows=_retained_rows(duration_s=0.2, catch_start_s=0.1),
            event_rows=[],
        )


def test_ramp_transition_must_precede_retained_catch() -> None:
    spec = _physics(duration_s=0.5, key_event_time_s=0.3)
    spec["physics"]["surface_transition_contract"] = {
        "schema_version": "surface-to-free-flight/v1",
        "support_surface_id": "owned_ramp_launch_surface",
        "support_normal_world_xyz": [0.0, 0.0, 1.0],
        "minimum_support_contact_s": 0.05,
        "minimum_free_flight_s": 0.1,
        "require_ordered_termination": True,
    }
    rows = _retained_rows(duration_s=0.5, catch_start_s=0.31)
    for row in rows:
        timestamp = row["timestamp"]
        if timestamp <= 0.1:
            row["object.motion_mode"] = "surface_contact"
            row["contact.count"] = 1
        elif timestamp <= 0.3:
            row["object.motion_mode"] = "free_flight"
            row["contact.count"] = 0
    events = [
        _event(
            index / 100,
            category="task_surface",
            normal=[0.0, 0.0, 1.0],
            surface_id="owned_ramp_launch_surface",
        )
        for index in range(11)
    ]
    events.extend(
        _event(
            index / 100,
            category="gripper",
            normal=[1.0, 0.0, 0.0],
        )
        for index in range(31, 51)
    )

    result = evaluate_source_rows(
        evaluator_id="rigid_ramp_launch_v1",
        corpus_leaf_id="F2b",
        task_variant="ramp_launch_catch",
        source_spec=spec,
        state_rows=rows,
        event_rows=events,
    )
    assert result.task_success is True
    assert result.evidence["surface_transition_pass"] is True
    assert result.evidence["ordered_support_contact_termination"] is True
    assert result.evidence["post_support_ballistic_motion"] is True

    missing_identity = deepcopy(events)
    for row in missing_identity:
        row.pop("task_surface_id", None)
    failed = evaluate_source_rows(
        evaluator_id="rigid_ramp_launch_v1",
        corpus_leaf_id="F2b",
        task_variant="ramp_launch_catch",
        source_spec=spec,
        state_rows=rows,
        event_rows=missing_identity,
    )
    assert failed.actual_outcome_class is ActualOutcomeClass.INVALID
    assert failed.evidence["surface_transition_pass"] is False


def test_ramp_transition_rejects_robot_contact_before_ballistic_window() -> None:
    spec = _physics(duration_s=0.5, key_event_time_s=0.3)
    spec["physics"]["surface_transition_contract"] = {
        "schema_version": "surface-to-free-flight/v1",
        "support_surface_id": "owned_ramp_launch_surface",
        "support_normal_world_xyz": [0.0, 0.0, 1.0],
        "minimum_support_contact_s": 0.05,
        "minimum_free_flight_s": 0.1,
        "require_ordered_termination": True,
    }
    rows = _retained_rows(duration_s=0.5, catch_start_s=0.31)
    for row in rows:
        timestamp = row["timestamp"]
        if timestamp <= 0.1:
            row["object.motion_mode"] = "surface_contact"
            row["contact.count"] = 1
        elif timestamp <= 0.3:
            row["object.motion_mode"] = "free_flight"
            row["contact.count"] = 0
    events = [
        _event(
            index / 100,
            category="task_surface",
            normal=[0.0, 0.0, 1.0],
            surface_id="owned_ramp_launch_surface",
        )
        for index in range(11)
    ]
    # Ramp support ends at 0.10 s and the contract requires 0.10 s of clean
    # ballistic flight.  This 0.15 s arm hit is therefore premature even if a
    # later retained grasp happens to succeed.
    events.append(
        _event(
            0.15,
            category="robot_arm",
            normal=[1.0, 0.0, 0.0],
        )
    )
    events.extend(
        _event(
            index / 100,
            category="gripper",
            normal=[1.0, 0.0, 0.0],
        )
        for index in range(31, 51)
    )

    failed = evaluate_source_rows(
        evaluator_id="rigid_ramp_launch_v1",
        corpus_leaf_id="F2b",
        task_variant="ramp_launch_catch",
        source_spec=spec,
        state_rows=rows,
        event_rows=events,
    )

    assert failed.task_success is False
    assert failed.evidence["premature_robot_contact_count"] == 1
    assert failed.evidence["no_robot_contact_before_ballistic_window"] is False
    assert failed.evidence["surface_transition_pass"] is False


def test_multi_rebound_contract_requires_two_surfaces() -> None:
    spec = _physics(duration_s=0.5, key_event_time_s=0.4)
    spec["physics"]["ordered_contact_contract"] = {
        "schema_version": "ordered-surface-contacts/v1",
        "ordered_surface_ids": ["only_surface"],
        "ordered_surface_normals_world_xyz": [[0.0, 0.0, 1.0]],
        "minimum_separated_pre_post_samples": 2,
        "minimum_inter_contact_free_flight_s": 0.04,
        "reject_contact_chatter": True,
    }
    with pytest.raises(ValueError, match="at least two surface IDs"):
        evaluate_source_rows(
            evaluator_id="rigid_multi_rebound_v1",
            corpus_leaf_id="F2e",
            task_variant="floor_to_wall",
            source_spec=spec,
            state_rows=_retained_rows(duration_s=0.5, catch_start_s=0.4),
            event_rows=[],
        )

    spec["physics"]["ordered_contact_contract"].update(
        {
            "ordered_surface_ids": ["surface_a", "surface_b"],
            "ordered_surface_normals_world_xyz": [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            "minimum_separated_pre_post_samples": 1,
        }
    )
    with pytest.raises(ValueError, match="at least two separated samples"):
        evaluate_source_rows(
            evaluator_id="rigid_multi_rebound_v1",
            corpus_leaf_id="F2e",
            task_variant="floor_to_wall",
            source_spec=spec,
            state_rows=_retained_rows(duration_s=0.5, catch_start_s=0.4),
            event_rows=[],
        )
def _ordered_rows() -> list[dict]:
    rows = _retained_rows(duration_s=0.8, catch_start_s=0.61)
    for row in rows:
        timestamp = row["timestamp"]
        row["object.position"] = [0.0, 0.0, 0.0]
        row["contact.bilateral"] = timestamp >= 0.61
        row["grasp.center_position"] = row["object.position"]
        if timestamp < 0.1:
            row["object.linear_velocity"] = [0.0, 0.0, -1.0]
        elif timestamp == 0.1:
            row["object.linear_velocity"] = [0.0, 0.0, 0.0]
            row["object.motion_mode"] = "surface_contact"
            row["contact.count"] = 1
        elif timestamp < 0.3:
            row["object.position"] = [0.0, 0.0, 0.5 * (timestamp - 0.1)]
            row["object.linear_velocity"] = [-1.0, 0.0, 0.5]
            row["object.motion_mode"] = "free_flight"
            row["contact.count"] = 0
        elif timestamp == 0.3:
            row["object.position"] = [-0.2, 0.0, 0.1]
            row["object.linear_velocity"] = [0.0, 0.0, 0.0]
            row["object.motion_mode"] = "surface_contact"
            row["contact.count"] = 1
        elif timestamp < 0.61:
            row["object.position"] = [-0.2 + 0.5 * (timestamp - 0.3), 0.0, 0.1]
            row["object.linear_velocity"] = [0.5, 0.0, 0.0]
            row["object.motion_mode"] = "free_flight"
            row["contact.count"] = 0
    return rows


def test_ordered_multi_rebound_rejects_chatter_and_reversed_order() -> None:
    spec = _physics(duration_s=0.8, key_event_time_s=0.6)
    spec["physics"]["ordered_contact_contract"] = {
        "schema_version": "ordered-surface-contacts/v1",
        "ordered_surface_ids": ["owned_multi_floor", "owned_multi_wall"],
        "ordered_surface_normals_world_xyz": [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        "minimum_separated_pre_post_samples": 2,
        "minimum_inter_contact_free_flight_s": 0.05,
        "reject_contact_chatter": True,
    }
    events = [
        _event(
            0.1,
            category="task_surface",
            normal=[0.0, 0.0, 1.0],
            surface_id="owned_multi_floor",
        ),
        _event(
            0.3,
            category="task_surface",
            normal=[1.0, 0.0, 0.0],
            surface_id="owned_multi_wall",
        ),
    ]
    events.extend(
        _event(
            index / 100,
            category="gripper",
            normal=[1.0, 0.0, 0.0],
        )
        for index in range(61, 81)
    )
    rows = _ordered_rows()

    result = evaluate_source_rows(
        evaluator_id="rigid_multi_rebound_v1",
        corpus_leaf_id="F2e",
        task_variant="floor_to_wall",
        source_spec=spec,
        state_rows=rows,
        event_rows=events,
    )
    assert result.task_success is True
    assert result.evidence["ordered_contact_sequence_pass"] is True
    assert result.evidence["ordered_rebound_kinematics_pass"] is True

    chatter = deepcopy(events)
    chatter.insert(
        1,
        _event(
            0.2,
            category="task_surface",
            normal=[0.0, 0.0, 1.0],
            surface_id="owned_multi_floor",
        ),
    )
    failed = evaluate_source_rows(
        evaluator_id="rigid_multi_rebound_v1",
        corpus_leaf_id="F2e",
        task_variant="floor_to_wall",
        source_spec=spec,
        state_rows=rows,
        event_rows=chatter,
    )
    assert failed.actual_outcome_class is ActualOutcomeClass.INVALID
    assert failed.evidence["ordered_surface_contact_sequence_matches"] is False

    premature = deepcopy(events)
    premature.insert(
        0,
        _event(
            0.05,
            category="robot_arm",
            normal=[1.0, 0.0, 0.0],
            impulse=0.05,
        ),
    )
    failed = evaluate_source_rows(
        evaluator_id="rigid_multi_rebound_v1",
        corpus_leaf_id="F2e",
        task_variant="floor_to_wall",
        source_spec=spec,
        state_rows=rows,
        event_rows=premature,
    )
    assert failed.actual_outcome_class is ActualOutcomeClass.INVALID
    assert failed.evidence["premature_tool_contact_count"] == 1
    assert (
        failed.evidence["no_tool_contact_before_ordered_rebound_completion"]
        is False
    )


def test_sampled_surface_contract_replays_catalog_and_rejects_self_authored_geometry() -> None:
    spec = _physics(duration_s=0.5, key_event_time_s=0.3)
    spec["physics"]["sampled_surface_contract"] = {
        "schema_version": "sampled-admitted-surface/v1",
        "catalog_version": "rigid-arbitrary-surfaces/v1",
        "candidate_id": "plane_small",
        "source_seed": 17,
        "position_m": [0.0, 0.0, 0.0],
        "euler_rad": [0.0, 0.0, 0.0],
        "normal_world_xyz": [0.0, 0.0, 1.0],
        "half_size_m": [0.3, 0.3, 0.02],
        "admission": {
            "grounded_supported": True,
            "reachability_checked": True,
            "swept_volume_clearance_checked": True,
            "background_clearance_checked": True,
            "calibrated_600_1200": True,
        },
        "admission_evidence_sha256": {
            name: "a" * 64
            for name in (
                "grounded_supported",
                "reachability_checked",
                "swept_volume_clearance_checked",
                "background_clearance_checked",
                "calibrated_600_1200",
            )
        },
    }
    rows = _retained_rows(duration_s=0.5, catch_start_s=0.31)
    for row in rows:
        timestamp = row["timestamp"]
        if timestamp < 0.1:
            row["object.linear_velocity"] = [0.0, 0.0, -1.0]
        elif timestamp == 0.1:
            row["object.linear_velocity"] = [0.0, 0.0, 0.0]
            row["object.motion_mode"] = "surface_contact"
            row["contact.count"] = 1
        elif timestamp <= 0.3:
            row["object.position"] = [0.0, 0.0, 0.5 + 0.5 * (timestamp - 0.1)]
            row["object.linear_velocity"] = [0.0, 0.0, 0.5]
            row["object.motion_mode"] = "free_flight"
            row["contact.count"] = 0
    events = [
        _event(
            0.1,
            category="task_surface",
            normal=[0.0, 0.0, 1.0],
            surface_id="owned_arbitrary_surface__plane_small",
        )
    ]
    events.extend(
        _event(
            index / 100,
            category="gripper",
            normal=[1.0, 0.0, 0.0],
        )
        for index in range(31, 51)
    )

    with pytest.raises(ValueError, match="deterministic catalog replay"):
        evaluate_source_rows(
            evaluator_id="rigid_arbitrary_rebound_v1",
            corpus_leaf_id="F2f",
            task_variant="random_plane_bounce",
            source_spec=spec,
            state_rows=rows,
            event_rows=events,
        )

    from dynamic_robot_dataset.backends.source_mujoco.rigid_breadth import (
        sample_surface_candidate,
        sampled_surface_contract,
    )

    uncalibrated = deepcopy(spec)
    candidate = sample_surface_candidate("random_plane_bounce", source_seed=17)
    uncalibrated["physics"]["sampled_surface_contract"] = (
        sampled_surface_contract(candidate, source_seed=17).to_dict()
    )
    with pytest.raises(ValueError, match="is not admitted"):
        evaluate_source_rows(
            evaluator_id="rigid_arbitrary_rebound_v1",
            corpus_leaf_id="F2f",
            task_variant="random_plane_bounce",
            source_spec=uncalibrated,
            state_rows=rows,
            event_rows=events,
        )


def test_v14_and_v15_remain_distinct_replay_versions() -> None:
    assert SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION == "1.4.0"
    assert SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION == "1.5.0"
    spec = _physics(duration_s=0.3, key_event_time_s=0.05)
    rows = _retained_rows(duration_s=0.3, catch_start_s=0.05)
    for row in rows:
        if row["timestamp"] > 0.14:
            row["contact.bilateral"] = False
    old = evaluate_source_rows(
        evaluator_id="rigid_catch_v2",
        corpus_leaf_id="F1a",
        task_variant="catch_retain",
        source_spec=spec,
        state_rows=rows,
        event_rows=[],
        _evaluator_version=SOURCE_OBJECTIVE_EVALUATOR_LEGACY_VERSION,
    )
    previous = evaluate_source_rows(
        evaluator_id="rigid_catch_v2",
        corpus_leaf_id="F1a",
        task_variant="catch_retain",
        source_spec=spec,
        state_rows=rows,
        event_rows=[],
        _evaluator_version=SOURCE_OBJECTIVE_EVALUATOR_PREVIOUS_VERSION,
    )
    assert old.task_success is True
    assert previous.task_success is False
