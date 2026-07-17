from __future__ import annotations

from dynamic_robot_dataset.common.schema import ActualOutcomeClass
from dynamic_robot_dataset.common.source_evaluators import evaluate_source_rows


def _spec() -> dict:
    return {
        "physics": {
            "simulation_hz": 100,
            "key_event_name": "interception",
            "key_event_time_s": 0.5,
        }
    }


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


def test_persisted_contact_failure_uses_measured_tool_contact_time() -> None:
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
                "contact_category": "gripper",
            }
        ],
    )

    assert not result.task_success
    assert result.actual_outcome_class is ActualOutcomeClass.CONTACT_FAILURE
    assert result.key_event_name == "tool_contact_onset"
    assert result.key_event_time_s == 0.12


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
