from __future__ import annotations

from pathlib import Path

from dynamic_robot_dataset.common.contract_v2 import (
    OBJECTIVE_EVIDENCE_VERSION,
    ObjectiveEvaluatorRegistry,
    ObjectiveRecomputeInput,
    ObjectiveRecomputeResult,
)
from dynamic_robot_dataset.common.hashing import sha256_json
from dynamic_robot_dataset.common.qc import QCValidator
from dynamic_robot_dataset.common.schema import ActualOutcomeClass, EpisodeRecord


EVALUATOR_ID = "freefall_trajectory_v1"
EVALUATOR_VERSION = "1"
EVIDENCE = {"terminal_height_m": 0.0, "derived_from": "frame_rows"}


def _source_record(**updates: object) -> EpisodeRecord:
    values: dict[str, object] = {
        "episode_uuid": "10000000-0000-4000-8000-000000000000",
        "episode_index": 0,
        "counterfactual_bundle_id": "P0a-review-0",
        "split_group_id": "P0a-review-0",
        "scene_seed": 10,
        "branch_seed": 20,
        "family": "passive_physics",
        "subfamily": "free_fall",
        # A success-seeking intent must not turn measured failure evidence into
        # success during persisted-artifact replay.
        "intended_branch": "success_seeking",
        "actual_outcome": "miss",
        "task_success": False,
        "failure_mode": "no_contact",
        "source_generator": "source_mujoco",
        "source_generator_version": "1",
        "config_hash": "a" * 64,
        "simulator_name": "mujoco",
        "simulator_version": "3.3.6",
        "renderer": "mujoco.Renderer",
        "event_time_s": 0.5,
        "key_event_name": "freefall_midpoint",
        "objective_evaluator_id": EVALUATOR_ID,
        "objective_evaluator_version": EVALUATOR_VERSION,
        "objective_evidence": {
            "independently_recomputed": True,
            "evidence_hash": sha256_json(EVIDENCE),
            "source": "persisted_frame_rows",
        },
        "objective_metrics": {"objective_success": False},
        # Keep this a bounded review record so generic release-claim behavior
        # cannot mask source_mujoco's backend-specific fail-closed checks.
        "quality_flags": ["review_only"],
        "extras": {
            "end_effector": "no_robot",
            "backend_provenance": {"backend": "source_mujoco"},
        },
    }
    values.update(updates)
    return EpisodeRecord(**values)  # type: ignore[arg-type]


def _registry(*, include_key_event: bool = True) -> ObjectiveEvaluatorRegistry:
    registry = ObjectiveEvaluatorRegistry()

    @registry.decorator(EVALUATOR_ID, EVALUATOR_VERSION)
    def evaluate(_input: ObjectiveRecomputeInput) -> ObjectiveRecomputeResult:
        return ObjectiveRecomputeResult(
            task_success=False,
            actual_outcome_class=ActualOutcomeClass.MISS,
            primary_failure_code="no_contact",
            evidence=dict(EVIDENCE),
            key_event_name="freefall_midpoint" if include_key_event else None,
            key_event_time_s=0.5 if include_key_event else None,
        )

    return registry


def test_source_qc_serializes_hash_bound_persisted_replay(tmp_path: Path) -> None:
    result, _ = QCValidator(
        tmp_path,
        deep_video_checks=False,
        objective_evaluators=_registry(),
    )._validate_episode(_source_record())

    serialized = result.to_dict()["metrics"]["objective_recompute"]
    assert serialized == {
        "evaluator_id": EVALUATOR_ID,
        "evaluator_version": EVALUATOR_VERSION,
        "evidence_version": OBJECTIVE_EVIDENCE_VERSION,
        "evidence_hash": sha256_json(EVIDENCE),
        "task_success": False,
        "actual_outcome_class": "miss",
        "primary_failure_code": "no_contact",
        "key_event_name": "freefall_midpoint",
        "key_event_time_s": 0.5,
        "replay_match": True,
    }
    assert not any("objective replay does not match" in value for value in result.hard_failures)


def test_source_qc_rejects_replay_without_measured_key_event(tmp_path: Path) -> None:
    result, _ = QCValidator(
        tmp_path,
        deep_video_checks=False,
        objective_evaluators=_registry(include_key_event=False),
    )._validate_episode(
        _source_record(event_time_s=None, key_event_name=None, key_event_time_s=None)
    )

    serialized = result.to_dict()["metrics"]["objective_recompute"]
    assert serialized["key_event_name"] is None
    assert serialized["key_event_time_s"] is None
    assert any("recomputation lacks key_event_name" in value for value in result.hard_failures)
    assert any(
        "recomputation lacks finite key_event_time_s" in value
        for value in result.hard_failures
    )


def test_source_qc_rejects_missing_stored_evidence_hash(tmp_path: Path) -> None:
    result, _ = QCValidator(
        tmp_path,
        deep_video_checks=False,
        objective_evaluators=_registry(),
    )._validate_episode(
        _source_record(
            objective_evidence={
                "independently_recomputed": True,
                "source": "persisted_frame_rows",
            }
        )
    )

    assert any(
        "record lacks a content-bound objective evidence hash" in value
        for value in result.hard_failures
    )


def test_source_qc_missing_evaluator_is_hard_failure_for_review_record(
    tmp_path: Path,
) -> None:
    result, _ = QCValidator(
        tmp_path,
        deep_video_checks=False,
        objective_evaluators=ObjectiveEvaluatorRegistry(),
    )._validate_episode(_source_record())

    assert "objective_recompute" not in result.metrics
    assert any(
        "source_mujoco episode lacks persisted-artifact objective recomputation"
        in value
        for value in result.hard_failures
    )
