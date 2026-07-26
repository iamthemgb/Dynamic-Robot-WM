from __future__ import annotations

import pytest

from dynamic_robot_dataset.common.grasp_retention import DEFAULT_GRASP_RETENTION
from dynamic_robot_dataset.common.rebound import DEFAULT_REBOUND_ACCEPTANCE
from dynamic_robot_dataset.common.schema import ActualOutcomeClass
from dynamic_robot_dataset.common.source_evaluators import (
    PASSIVE_EVENT_PROJECTILE_APEX,
    PASSIVE_EVENT_TASK_SURFACE_CONTACT,
    evaluate_source_rows,
    select_source_key_event,
)


def _spec() -> dict:
    return {
        "duration_s": 0.19,
        "physics": {
            "simulation_hz": 100,
            "key_event_name": "interception",
            "key_event_time_s": 0.5,
            "grasp_retention": DEFAULT_GRASP_RETENTION.to_dict(),
        }
    }


def _rebound_spec() -> dict:
    value = _spec()
    value["physics"].update(
        object_radius_m=0.0245,
        rebound_acceptance=DEFAULT_REBOUND_ACCEPTANCE.to_dict(),
    )
    return value


def test_persisted_catch_evaluator_accepts_only_sustained_stable_bilateral_grasp() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "timestamp": index / 100,
                "object.position": [0.5, 0.0, 0.5],
                "object.linear_velocity": [0.0, 0.0, 0.0],
                "contact.bilateral": index >= 5,
                "grasp.center_position": [0.5, 0.0, 0.5],
            }
        )
    result = evaluate_source_rows(
        evaluator_id="rigid_catch_v2",
        corpus_leaf_id="F1a",
        task_variant="catch_retain",
        source_spec=_spec(),
        state_rows=rows,
        event_rows=[],
    )
    assert result.task_success
    assert result.actual_outcome_class is ActualOutcomeClass.SUCCESS
    assert result.primary_failure_code == "none"
    assert result.key_event_name == "bilateral_grasp_onset"
    assert result.key_event_time_s == 0.05
    assert result.evidence["measured_key_event_source"] == (
        "persisted_bilateral_contact"
    )
    assert result.evidence["retained_through_final_state"] is True
    assert result.evidence["final_retention_bilateral_fraction"] == 1.0


def test_persisted_catch_rejects_transient_grasp_lost_before_final_window() -> None:
    source_spec = _spec()
    source_spec["duration_s"] = 0.3
    rows = [
        {
            "timestamp": index / 100,
            "object.position": [0.5, 0.0, 0.5],
            "object.linear_velocity": [0.0, 0.0, 0.0],
            "contact.bilateral": 2 <= index <= 11,
            "grasp.center_position": [0.5, 0.0, 0.5],
        }
        for index in range(31)
    ]

    result = evaluate_source_rows(
        evaluator_id="rigid_catch_v2",
        corpus_leaf_id="F1a",
        task_variant="catch_retain",
        source_spec=source_spec,
        state_rows=rows,
        event_rows=[
            {
                "timestamp": 0.02,
                "penetration_depth_m": 0.0005,
                "contact_category": "gripper",
            }
        ],
    )

    assert result.task_success is False
    assert result.actual_outcome_class is ActualOutcomeClass.CONTACT_FAILURE
    assert result.primary_failure_code == "contact_without_completion"
    assert result.evidence["sustained_opposing_bilateral_contacts"] is True
    assert result.evidence["retained_through_final_state"] is False
    assert result.evidence["final_bilateral_contact"] is False
    assert result.evidence["final_retention_bilateral_fraction"] == 0.0


def test_bilateral_selector_binds_both_saved_contact_geom_ids() -> None:
    selected = select_source_key_event(
        planned_key_event_name="interception",
        planned_key_event_time_s=0.5,
        state_rows=[{"timestamp": 0.12, "contact.bilateral": True}],
        event_rows=[
            {
                "timestamp": 0.12,
                "contact_category": "gripper",
                "counterpart_geom_id": geom_id,
            }
            for geom_id in (17, 23)
        ],
        passive=False,
        contact_time_tolerance_s=0.01,
    )

    assert selected["contact_counterpart_geom_ids"] == [17, 23]


@pytest.mark.parametrize("contact_category", ("gripper", "robot_arm"))
def test_persisted_contact_failure_uses_measured_tool_contact_time(
    contact_category: str,
) -> None:
    result = evaluate_source_rows(
        evaluator_id="rigid_catch_v2",
        corpus_leaf_id="F1a",
        task_variant="catch_retain",
        source_spec=_spec(),
        state_rows=[
            {
                "timestamp": index / 100,
                "object.position": [0.5, 0.0, 0.5],
                "object.linear_velocity": [0.0, 0.0, 0.0],
                "contact.bilateral": False,
            }
            for index in range(20)
        ],
        event_rows=[
            {
                "timestamp": 0.12,
                "penetration_depth_m": 0.0005,
                "contact_category": contact_category,
            }
        ],
    )

    assert not result.task_success
    assert result.actual_outcome_class is ActualOutcomeClass.CONTACT_FAILURE
    assert result.key_event_name == "tool_contact_onset"
    assert result.key_event_time_s == 0.12


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    (
        ("final_window_s", 0.11),
        ("minimum_bilateral_fraction", 0.96),
        ("maximum_relative_range_m", 0.011),
        ("require_bilateral_at_final_sample", False),
        ("schema_version", "dynamic-robot-grasp-retention/v2"),
    ),
)
def test_v15_catch_evaluator_rejects_every_default_threshold_mutation(
    field: str,
    mutated_value: object,
) -> None:
    source_spec = _spec()
    source_spec["physics"]["grasp_retention"][field] = mutated_value

    with pytest.raises(ValueError):
        evaluate_source_rows(
            evaluator_id="rigid_catch_v2",
            corpus_leaf_id="F1a",
            task_variant="catch_retain",
            source_spec=source_spec,
            state_rows=[
                {
                    "timestamp": 0.19,
                    "object.position": [0.5, 0.0, 0.5],
                    "object.linear_velocity": [0.0, 0.0, 0.0],
                    "contact.bilateral": False,
                }
            ],
            event_rows=[],
        )


def test_persisted_source_evaluator_marks_excessive_penetration_invalid() -> None:
    result = evaluate_source_rows(
        evaluator_id="rigid_catch_v2",
        corpus_leaf_id="F1a",
        task_variant="catch_retain",
        source_spec=_spec(),
        state_rows=[
            {
                "timestamp": 0.0,
                "object.position": [0.0, 0.0, 0.5],
                "object.linear_velocity": [0.0, 0.0, -1.0],
                "contact.bilateral": False,
            }
        ],
        event_rows=[
            {"penetration_depth_m": 0.003, "contact_category": "gripper"}
        ],
    )
    assert not result.task_success
    assert result.actual_outcome_class is ActualOutcomeClass.INVALID
    assert result.primary_failure_code == "excessive_penetration"


def test_persisted_passive_evaluator_treats_valid_observation_as_success() -> None:
    result = evaluate_source_rows(
        evaluator_id="passive_freeflight_v1",
        corpus_leaf_id="P0a",
        task_variant="nominal_freefall",
        source_spec=_spec(),
        state_rows=[
            {
                "object.position": [0.0, 0.0, 1.0 - index * 0.01],
                "object.linear_velocity": [0.0, 0.0, -index * 0.1],
                "object.motion_mode": "free_flight",
            }
            for index in range(3)
        ],
        event_rows=[],
    )
    assert result.task_success
    assert result.actual_outcome_class is ActualOutcomeClass.SUCCESS


def test_persisted_rebound_requires_visible_separation_and_lower_bound() -> None:
    result = evaluate_source_rows(
        evaluator_id="passive_rebound_v1",
        corpus_leaf_id="P0c",
        task_variant="table_bounce",
        source_spec=_rebound_spec(),
        state_rows=[
            {
                "timestamp": 0.09,
                "object.position": [0.0, 0.0, 0.03],
                "object.linear_velocity": [0.0, 0.0, -1.0],
                "object.motion_mode": "free_flight",
                "contact.count": 0,
            },
            {
                "timestamp": 0.10,
                "object.position": [0.0, 0.0, 0.0245],
                "object.linear_velocity": [0.0, 0.0, -0.4],
                "object.motion_mode": "surface_contact",
                "contact.count": 1,
            },
            {
                "timestamp": 0.11,
                "object.position": [0.0, 0.0, 0.0265],
                "object.linear_velocity": [0.0, 0.0, 0.2],
                "object.motion_mode": "free_flight",
                "contact.count": 0,
            },
            {
                "timestamp": 0.18,
                "object.position": [0.0, 0.0, 0.04],
                "object.linear_velocity": [0.0, 0.0, 0.18],
                "object.motion_mode": "free_flight",
                "contact.count": 0,
            },
        ],
        event_rows=[
            {
                "timestamp": 0.10,
                "contact_category": "task_surface",
                "penetration_depth_m": 0.001,
                "normal_world": [0.0, 0.0, 1.0],
            }
        ],
    )

    assert result.task_success is True
    rebound = result.evidence["rebound"]
    assert rebound["effective_restitution"] == pytest.approx(0.2)
    assert rebound["maximum_normal_separation_m"] == pytest.approx(0.0155)
    assert rebound["separation_duration_s"] == pytest.approx(0.08)
    assert rebound["rebound_acceptance_pass"] is True
    assert result.key_event_name == "task_surface_contact_onset"
    assert result.key_event_time_s == pytest.approx(0.10)
    assert result.evidence["measured_key_event_source"] == (
        "persisted_task_surface_contact"
    )


def test_persisted_rebound_rejects_contact_without_meaningful_bounce() -> None:
    result = evaluate_source_rows(
        evaluator_id="passive_rebound_v1",
        corpus_leaf_id="P0c",
        task_variant="table_bounce",
        source_spec=_rebound_spec(),
        state_rows=[
            {
                "timestamp": 0.09,
                "object.position": [0.0, 0.0, 0.03],
                "object.linear_velocity": [0.0, 0.0, -1.0],
                "contact.count": 0,
            },
            {
                "timestamp": 0.10,
                "object.position": [0.0, 0.0, 0.0245],
                "object.linear_velocity": [0.0, 0.0, -0.2],
                "contact.count": 1,
            },
            {
                "timestamp": 0.11,
                "object.position": [0.0, 0.0, 0.025],
                "object.linear_velocity": [0.0, 0.0, 0.05],
                "contact.count": 0,
            },
            {
                "timestamp": 0.12,
                "object.position": [0.0, 0.0, 0.025],
                "object.linear_velocity": [0.0, 0.0, 0.0],
                "contact.count": 1,
            },
        ],
        event_rows=[
            {
                "timestamp": 0.10,
                "contact_category": "task_surface",
                "penetration_depth_m": 0.001,
                "normal_world": [0.0, 0.0, 1.0],
            }
        ],
    )

    assert result.task_success is False
    assert result.actual_outcome_class is ActualOutcomeClass.INVALID
    rebound = result.evidence["rebound"]
    assert rebound["effective_restitution"] == pytest.approx(0.05)
    assert rebound["effective_restitution_within_limits"] is False
    assert rebound["outgoing_normal_speed_sufficient"] is False
    assert rebound["normal_separation_sufficient"] is False
    assert rebound["separation_duration_sufficient"] is False
    assert rebound["rebound_acceptance_pass"] is False


def test_persisted_rebound_rejects_unbound_threshold_contract() -> None:
    source_spec = _rebound_spec()
    source_spec["physics"]["rebound_acceptance"][
        "minimum_effective_restitution"
    ] = 0.01

    with pytest.raises(ValueError, match="differs from evaluator"):
        evaluate_source_rows(
            evaluator_id="passive_rebound_v1",
            corpus_leaf_id="P0c",
            task_variant="table_bounce",
            source_spec=source_spec,
            state_rows=[
                {
                    "timestamp": 0.0,
                    "object.position": [0.0, 0.0, 0.1],
                    "object.linear_velocity": [0.0, 0.0, -1.0],
                    "contact.count": 0,
                }
            ],
            event_rows=[],
        )


def test_passive_freefall_uses_first_persisted_task_surface_contact() -> None:
    event_rows = [
        {
            "timestamp": timestamp,
            "penetration_depth_m": 0.0005,
            "contact_category": "task_surface",
            "counterpart_geom_id": geom_id,
        }
        for timestamp, geom_id in ((0.31, 17), (0.31, 23), (0.32, 17))
    ]
    selected = select_source_key_event(
        planned_key_event_name="freefall",
        planned_key_event_time_s=0.3,
        state_rows=(),
        event_rows=event_rows,
        passive=True,
        contact_time_tolerance_s=0.01,
        passive_event_semantics=PASSIVE_EVENT_TASK_SURFACE_CONTACT,
    )

    assert selected == {
        "key_event_name": "task_surface_contact_onset",
        "key_event_time_s": 0.31,
        "key_event_source": "persisted_task_surface_contact",
        "physical_contact_applicable": True,
        "contact_counterpart_geom_ids": [17, 23],
    }

    result = evaluate_source_rows(
        evaluator_id="passive_freeflight_v1",
        corpus_leaf_id="P0a",
        task_variant="nominal_freefall",
        source_spec=_spec(),
        state_rows=[
            {
                "timestamp": index / 100,
                "object.position": [0.0, 0.0, 1.0 - index * 0.01],
                "object.linear_velocity": [0.0, 0.0, -index * 0.1],
                "object.motion_mode": "free_flight",
            }
            for index in range(3)
        ],
        event_rows=event_rows,
    )
    assert result.key_event_name == "task_surface_contact_onset"
    assert result.key_event_time_s == 0.31
    assert result.evidence["measured_key_event_source"] == (
        "persisted_task_surface_contact"
    )


def test_passive_projectile_interpolates_apex_from_persisted_free_flight() -> None:
    state_rows = [
        {
            "timestamp": 0.2,
            "object.position": [0.0, 0.0, 1.0],
            "object.linear_velocity": [1.0, 0.0, 1.0],
            "object.motion_mode": "free_flight",
        },
        {
            "timestamp": 0.4,
            "object.position": [0.2, 0.0, 1.0],
            "object.linear_velocity": [1.0, 0.0, -1.0],
            "object.motion_mode": "free_flight",
        },
        {
            "timestamp": 0.7,
            "object.position": [0.5, 0.0, 5.0],
            "object.linear_velocity": [0.0, 0.0, 0.0],
            "object.motion_mode": "surface_contact",
        },
    ]
    selected = select_source_key_event(
        planned_key_event_name="projectile",
        planned_key_event_time_s=0.4,
        state_rows=state_rows,
        event_rows=(),
        passive=True,
        passive_event_semantics=PASSIVE_EVENT_PROJECTILE_APEX,
    )

    assert selected["key_event_name"] == "projectile_apex"
    assert selected["key_event_time_s"] == pytest.approx(0.3)
    assert selected["key_event_source"] == "persisted_free_flight_apex"
    assert selected["physical_contact_applicable"] is False

    result = evaluate_source_rows(
        evaluator_id="passive_projectile_v1",
        corpus_leaf_id="P0b",
        task_variant="ballistic_projectile",
        source_spec=_spec(),
        state_rows=state_rows,
        event_rows=(),
    )
    assert result.key_event_name == "projectile_apex"
    assert result.key_event_time_s == pytest.approx(0.3)
    assert result.evidence["measured_key_event_source"] == (
        "persisted_free_flight_apex"
    )
