from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dynamic_robot_dataset.common.contract_v2 import (
    CounterfactualRelation,
    MotionMode,
    ObjectiveEvaluatorRegistry,
    ObjectiveRecomputeInput,
    ObjectiveRecomputeResult,
    build_counterfactual_family_records,
    compare_recomputed_objective,
    validate_assistance_observations,
    validate_counterfactual_family_records,
    validate_v2_frame_semantics,
)
from dynamic_robot_dataset.common.hashing import sha256_json
from dynamic_robot_dataset.common.schema import (
    FAILURE_TAXONOMY_VERSION,
    FAILURE_CODES,
    FAILURE_TAGS,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ActualOutcomeClass,
    DynamicsMode,
    EpisodeRecord,
    PhysicsMetadata,
    PhysicsValue,
    ReleaseTier,
    SchemaValidationError,
    infer_actual_outcome_class,
)


def test_failure_taxonomy_config_matches_runtime_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = yaml.safe_load(
        (root / "configs/schema/failure_codes_v2.yaml").read_text(encoding="utf-8")
    )
    assert catalog["schema_version"] == FAILURE_TAXONOMY_VERSION
    assert set(catalog["codes"]) == FAILURE_CODES
    assert set(catalog["failure_tags"]) == FAILURE_TAGS


def _record(index: int = 0, **updates: object) -> EpisodeRecord:
    values: dict[str, object] = {
        "episode_uuid": f"10000000-0000-4000-8000-{index:012d}",
        "episode_index": index,
        "counterfactual_bundle_id": "action-family",
        "physics_counterfactual_family_id": "physics-family",
        "split_group_id": "scene-family",
        "scene_seed": 10,
        "branch_seed": index,
        "family": "falling_catch",
        "subfamily": "catch_retain",
        "intended_branch": "success_seeking",
        "actual_outcome": "success",
        "task_success": True,
        "failure_mode": "none",
        "source_generator": "test.native",
        "source_generator_version": "2",
        "config_hash": "a" * 64,
        "simulator_name": "mujoco",
        "simulator_version": "3.10.0",
        "renderer": "mujoco",
        "objective_metrics": {"objective_success": True},
        "extras": {
            "action_hash": sha256_json([0.1, 0.2]),
            "initial_state_hash": sha256_json({"qpos": [0.0]}),
        },
    }
    values.update(updates)
    return EpisodeRecord(**values)  # type: ignore[arg-type]


def test_v2_outcome_is_closed_and_failure_fields_agree() -> None:
    record = _record(
        task_success=False,
        actual_outcome="bad_action",
        failure_mode="wrong_action",
        objective_metrics={"objective_success": False},
    )
    record.validate()
    assert record.actual_outcome_class == ActualOutcomeClass.WRONG_ACTION
    assert record.primary_failure_code == "wrong_action"
    assert record.failure_taxonomy_version == FAILURE_TAXONOMY_VERSION
    assert "wrong_action" in record.failure_tags
    record.primary_failure_code = "no_contact"
    with pytest.raises(SchemaValidationError, match="primary_failure_code"):
        record.validate()


def test_compatibility_outcome_inference_prioritizes_invalid_then_partial() -> None:
    assert infer_actual_outcome_class(
        task_success=False,
        actual_outcome="partial_success",
        failure_code="contact_without_completion",
        partial_success_score=0.75,
    ) == ActualOutcomeClass.PARTIAL_SUCCESS
    assert infer_actual_outcome_class(
        task_success=False,
        actual_outcome="partial_success",
        failure_code="unstable_physics",
        partial_success_score=0.75,
    ) == ActualOutcomeClass.INVALID


def test_physics_counterfactual_family_is_optional_for_non_sweep_episode() -> None:
    record = _record(physics_counterfactual_family_id=None)
    record.validate()
    assert record.physics_counterfactual_family_id is None


def test_v1_episode_read_is_upgraded_but_not_silently_release_eligible() -> None:
    legacy = _record().to_dict()
    legacy["schema_version"] = LEGACY_SCHEMA_VERSION
    for name in (
        "actual_outcome_class",
        "primary_failure_code",
        "failure_tags",
        "failure_taxonomy_version",
        "objective_evaluator_id",
        "objective_evaluator_version",
        "objective_threshold_set_hash",
        "objective_evidence",
        "key_event_name",
        "key_event_time_s",
        "camera_stream_calibration_ids",
        "controller_profile",
        "robot_start_provenance",
        "tool_calibration_provenance",
    ):
        legacy.pop(name, None)
    legacy["physics"].pop("parameter_range_provenance", None)
    upgraded = EpisodeRecord.from_dict(legacy)
    assert upgraded.schema_version == SCHEMA_VERSION
    assert upgraded.extras["source_schema_version"] == LEGACY_SCHEMA_VERSION
    assert upgraded.actual_outcome_class == ActualOutcomeClass.SUCCESS
    assert not upgraded.release_eligible


def test_parameter_range_calibration_requires_artifact_hash() -> None:
    physics = PhysicsMetadata(
        parameter_range_provenance={
            "profile_id": "franka-rigid",
            "profile_version": "1",
            "partition": "train_id",
            "source": "calibration-suite",
            "calibrated": True,
            "calibration_artifact_hash": None,
        }
    )
    with pytest.raises(SchemaValidationError, match="SHA-256"):
        physics.validate()


def test_multiple_assistance_mechanisms_match_frame_intervals() -> None:
    mechanism = {
        "mechanism_id": "eq-grasp-1",
        "mechanism_type": "equality_constraint",
        "source": "simulator_observed",
        "constraint_ids": ["eq0"],
        "target_body_ids": ["rope_endpoint"],
        "target_element_ids": [],
        "activation_intervals": [{"start_time_s": 0.1, "end_time_s": 0.2}],
    }
    record = _record(
        dynamics_mode=DynamicsMode.ASSISTED_CONTACT,
        release_tier=ReleaseTier.ASSISTED_CONTACT,
        assistance={
            "assisted_grasp": False,
            "assisted_retention": False,
            "equality_constraint_active": True,
            "latch_active": False,
            "constraint_activation_time": 0.1,
            "constraint_deactivation_time": 0.2,
            "mechanisms": [mechanism],
        },
    )
    record.validate()
    rows = [
        {"timestamp": 0.0, "assistance.mechanism_ids": []},
        {"timestamp": 0.1, "assistance.mechanism_ids": ["eq-grasp-1"]},
        {"timestamp": 0.2, "assistance.mechanism_ids": ["eq-grasp-1"]},
        {"timestamp": 0.3, "assistance.mechanism_ids": []},
    ]
    assert validate_assistance_observations(record, rows, tolerance_s=0.0) == []
    rows[2]["assistance.mechanism_ids"] = []
    assert validate_assistance_observations(record, rows, tolerance_s=0.0)


def test_active_assistance_requires_named_simulator_mechanism() -> None:
    record = _record(
        dynamics_mode=DynamicsMode.ASSISTED_CONTACT,
        release_tier=ReleaseTier.ASSISTED_CONTACT,
        assistance={
            "assisted_grasp": True,
            "assisted_retention": False,
            "equality_constraint_active": False,
            "latch_active": False,
            "constraint_activation_time": 0.1,
            "constraint_deactivation_time": 0.2,
            "mechanisms": [],
        },
    )
    with pytest.raises(
        SchemaValidationError,
        match="simulator-observed mechanism records",
    ):
        record.validate()


def test_counterfactual_table_detects_missing_members_and_action_changes() -> None:
    records = [
        _record(0),
        _record(
            1,
            intended_branch="near_miss",
            branch_seed=1,
            extras={
                "action_hash": sha256_json([0.1, 0.3]),
                "initial_state_hash": sha256_json({"qpos": [0.0]}),
            },
        ),
    ]
    declarations = build_counterfactual_family_records(records)
    action = next(
        value for value in declarations if value.relation == CounterfactualRelation.ACTION
    )
    assert validate_counterfactual_family_records([action], records) == []
    missing = validate_counterfactual_family_records([action], records[:1])
    assert any("missing expected members" in problem for problem in missing)
    records[1].physics.parameters["mass_kg"] = PhysicsValue(
        "mass_kg", 0.2, "kg", True, True
    )
    changed = validate_counterfactual_family_records([action], records)
    assert any("changes" in problem and "physics" in problem for problem in changed)


def test_objective_registry_recomputes_only_from_persisted_rows() -> None:
    registry = ObjectiveEvaluatorRegistry()

    @registry.decorator("catch-retain", "1")
    def evaluator(evidence: ObjectiveRecomputeInput) -> ObjectiveRecomputeResult:
        retained = bool(evidence.frame_rows[-1]["retained"])
        return ObjectiveRecomputeResult(
            task_success=retained,
            actual_outcome_class=(
                ActualOutcomeClass.SUCCESS if retained else ActualOutcomeClass.CONTACT_FAILURE
            ),
            primary_failure_code="none" if retained else "object_not_retained",
            evidence={"terminal_retained": retained},
            key_event_name="retention_complete",
            key_event_time_s=0.5,
        )

    record = _record(
        objective_evaluator_id="catch-retain",
        objective_evaluator_version="1",
        event_time_s=0.5,
        key_event_name="retention_complete",
        objective_evidence={
            "independently_recomputed": True,
            "evidence_hash": sha256_json({"terminal_retained": True}),
        },
    )
    result = registry.get("catch-retain", "1")(
        ObjectiveRecomputeInput(record, [{"retained": True}], [], [])
    )
    assert compare_recomputed_objective(record, result) == []


def test_frame_semantics_use_closed_motion_and_contact_labels() -> None:
    rows = [
        {
            "task_phase": "approach",
            "motion_mode": MotionMode.FREE_FLIGHT.value,
            "active_surface": None,
            "contact_role": "none",
        }
    ]
    assert validate_v2_frame_semantics(rows) == []
    rows[0]["motion_mode"] = "teleporting"
    assert "unknown motion_mode" in validate_v2_frame_semantics(rows)[0]
