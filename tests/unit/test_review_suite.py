from __future__ import annotations

from dataclasses import replace
import json

import pytest

from dynamic_robot_dataset.backends.actuator_only import action_spec
from dynamic_robot_dataset.cli import main
from dynamic_robot_dataset.common.corpus_registry import (
    BackendCapabilityRegistry,
    load_backend_capability_registry,
    load_corpus_registry,
)
from dynamic_robot_dataset.common.paths import ExistingOutputError
from dynamic_robot_dataset.common.review import REVIEW_CHECKS, REVIEW_SCENE_SEQUENCE
from dynamic_robot_dataset.common.review_suite import (
    FIXED_REVIEW_MASTER_SEED,
    REVIEW_BUNDLE_FILE,
    REVIEW_PLAN_FILE,
    REVIEW_REQUEST_LEDGER_FILE,
    ReviewSuitePlan,
    ReviewSuiteValidationError,
    build_review_request_ledger,
    build_review_suite_plan,
    load_review_suite_bundle,
    load_review_robocasa_catalog,
    write_review_suite_bundle,
)


def _leaf_cases(plan: ReviewSuitePlan, leaf_id: str):
    return [case for case in plan.cases if case.corpus_leaf_id == leaf_id]


def test_fixed_review_plan_is_exact_twenty_by_six_matrix() -> None:
    plan = build_review_suite_plan()

    assert len(plan.cases) == 120
    assert len({case.corpus_leaf_id for case in plan.cases}) == 20
    assert len({case.case_id for case in plan.cases}) == 120
    assert len({case.episode_uuid for case in plan.cases}) == 120
    assert plan.fixed_master_seed == FIXED_REVIEW_MASTER_SEED
    for leaf_id in sorted({case.corpus_leaf_id for case in plan.cases}):
        cases = _leaf_cases(plan, leaf_id)
        assert [case.rollout_index for case in cases] == list(range(6))
        assert tuple(case.scene_profile for case in cases) == REVIEW_SCENE_SEQUENCE
        assert [case.randomization_level for case in cases] == [
            "R0",
            "R1",
            "R1",
            "R1",
            "R1",
            "R1",
        ]
        assert [case.requires_real_robocasa for case in cases] == [
            False,
            True,
            True,
            True,
            True,
            True,
        ]


def test_dual_embodiment_leaves_have_three_cases_each_and_real_negatives() -> None:
    plan = build_review_suite_plan()
    dual_leaf_ids = {
        case.corpus_leaf_id
        for case in plan.cases
        if case.embodiment == "robotiq_2f85_thick_pad"
    }

    assert dual_leaf_ids
    for leaf_id in dual_leaf_ids:
        cases = _leaf_cases(plan, leaf_id)
        for embodiment in ("franka_hand", "robotiq_2f85_thick_pad"):
            selected = [case for case in cases if case.embodiment == embodiment]
            assert len(selected) == 3
            assert [case.intended_outcome for case in selected].count("success") == 1
            assert [case.intended_outcome for case in selected].count("failure") == 2
            assert all(
                case.branch_role.startswith("deterministic_negative_")
                for case in selected
                if case.intended_outcome == "failure"
            )


def test_passive_and_single_embodiment_assignments_are_fail_closed() -> None:
    plan = build_review_suite_plan()

    for leaf_id in ("P0a", "P0b", "P0c", "P0d"):
        cases = _leaf_cases(plan, leaf_id)
        assert {case.embodiment for case in cases} == {"no_robot"}
        assert {case.intended_outcome for case in cases} == {"passive_observation"}
        assert len({case.passive_variation_profile for case in cases}) == 6
    for leaf_id in ("F3c", "D1", "D2"):
        cases = _leaf_cases(plan, leaf_id)
        assert {case.embodiment for case in cases} == {"franka_hand"}
        assert {case.intended_outcome for case in cases} == {"success", "failure"}
    assert {case.task_variant for case in _leaf_cases(plan, "D1")} == {
        "poke_cloth",
        "lift_corner_release",
        "fold_edge_fixed_line",
    }
    assert {case.task_variant for case in _leaf_cases(plan, "D2")} == {
        "drag_endpoint",
        "tug",
        "lift_and_drape",
        "wrap_around_post",
        "thread_through_ring",
    }


def test_default_review_quota_is_per_leaf_without_production_release() -> None:
    plan = build_review_suite_plan()
    review_leaves = {
        "P0a",
        "P0b",
        "P0c",
        "P0d",
        "F1a",
        "F1b",
        "F1c",
        "F1d",
    }

    # Content-bound candidates admit R1 attempts for review, while backend and
    # corpus release states remain independently blocked.
    assert plan.executable_case_count == 48
    assert plan.leaf_execution_quota() == {
        leaf_id: (6 if leaf_id in review_leaves else 0)
        for leaf_id in sorted({case.corpus_leaf_id for case in plan.cases})
    }
    assert {case.backend_release_state for case in plan.cases} == {"blocked"}
    assert {case.leaf_release_state for case in plan.cases} == {"blocked"}
    for case in plan.cases:
        expected = case.corpus_leaf_id in review_leaves
        assert case.execution_eligible is expected
        assert case.executable_quota == int(expected)
        if case.corpus_leaf_id not in review_leaves:
            assert case.support_execution_state == "blocked"
            assert any(
                blocker == "support_execution_state:blocked"
                for blocker in case.execution_blockers
            )
        else:
            assert case.support_execution_state == "review"
            assert case.execution_blockers == ()

    corpus = load_corpus_registry()
    registry = load_backend_capability_registry(corpus=corpus)
    original = registry.by_name["source_mujoco"]
    unpinned = replace(
        original,
        source_hashes={},
        blockers=(*original.blockers, "source_hashes_unpinned"),
    )
    modified = BackendCapabilityRegistry(
        registry_id=registry.registry_id,
        backends=tuple(
            unpinned if backend.name == original.name else backend
            for backend in registry.backends
        ),
    )
    modified.validate(corpus)

    unpinned_plan = build_review_suite_plan(
        corpus_registry=corpus,
        backend_registry=modified,
    )
    assert unpinned_plan.executable_case_count == 0
    assert all(
        "backend_source_hashes_unpinned" in case.execution_blockers
        for case in unpinned_plan.cases
        if case.backend == "source_mujoco"
    )

    unavailable_review_catalog = replace(
        load_review_robocasa_catalog(), review_ready=False
    )
    unavailable_plan = build_review_suite_plan(
        corpus_registry=corpus,
        robocasa_catalog=unavailable_review_catalog,
    )
    assert unavailable_plan.executable_case_count == 8
    assert unavailable_plan.leaf_execution_quota() == {
        leaf_id: (1 if leaf_id in review_leaves else 0)
        for leaf_id in sorted({case.corpus_leaf_id for case in unavailable_plan.cases})
    }
    assert all(
        "robocasa_catalog_not_review_ready" in case.execution_blockers
        for case in unavailable_plan.cases
        if case.corpus_leaf_id in review_leaves and case.requires_real_robocasa
    )


def test_review_request_ledger_requires_real_hashes_without_faking_them() -> None:
    plan = build_review_suite_plan()
    ledger = build_review_request_ledger(plan)

    assert len(ledger.requests) == 120
    assert len({request.request_sha256 for request in ledger.requests}) == 120
    for request, case in zip(ledger.requests, plan.cases):
        assert request.review_plan_sha256 == plan.plan_sha256
        assert request.case_sha256 == case.case_sha256
        assert request.status == "pending_artifacts"
        assert request.human_review_checks == REVIEW_CHECKS
        assert "review_plan_sha256" in request.required_artifact_hash_fields
        assert "review_case_sha256" in request.required_artifact_hash_fields
        assert "review_request_ledger_sha256" in request.required_artifact_hash_fields
        assert "review_request_sha256" in request.required_artifact_hash_fields
        assert "scenario_spec_sha256" in request.required_artifact_hash_fields
        assert "qc_report_sha256" in request.required_artifact_hash_fields
        assert "qc_episode_result_sha256" in request.required_artifact_hash_fields
        assert "frame_timestamps_sha256" in request.required_artifact_hash_fields
        assert "video_sha256.main" in request.required_artifact_hash_fields
        assert "video_sha256.secondary" in request.required_artifact_hash_fields
        assert "event_strip_sha256.main" in request.required_artifact_hash_fields
        assert "event_strip_sha256.secondary" in request.required_artifact_hash_fields
        assert "scenario_spec_sha256" not in request.source_scenario_identity
        embodiment = request.source_scenario_identity["embodiment"]
        if case.embodiment == "no_robot":
            assert embodiment == {
                "end_effector": "no_robot",
                "robot_model": "none",
                "action_names": [],
                "action_semantics": "no_actuators/v1",
            }
        else:
            specification = action_spec(case.embodiment)
            assert tuple(embodiment["action_names"]) == specification.actuator_names
            assert embodiment["action_semantics"] == "actual_actuator_command/v1"
        assert request.event_strip_request["views"] == ["main", "secondary"]
        assert request.event_strip_request["targets"] == [
            {"name": "pre_event", "offset_s": -0.1},
            {"name": "event", "offset_s": 0.0},
            {"name": "post_0p1_s", "offset_s": 0.1},
            {"name": "post_0p3_s", "offset_s": 0.3},
            {"name": "final", "frame": "final"},
        ]


def test_plan_and_request_hashes_are_stable_and_detect_tampering() -> None:
    left = build_review_suite_plan()
    right = build_review_suite_plan()

    assert left.plan_sha256 == right.plan_sha256
    assert [case.case_sha256 for case in left.cases] == [
        case.case_sha256 for case in right.cases
    ]
    restored = ReviewSuitePlan.from_dict(left.to_dict())
    assert restored == left

    changed = left.to_dict()
    changed["cases"][0]["scene_profile"] = "easier_replacement_scene"
    with pytest.raises(ReviewSuiteValidationError, match="case hash mismatch"):
        ReviewSuitePlan.from_dict(changed)


def test_bundle_write_is_idempotent_and_rejects_changed_immutable_output(tmp_path) -> None:
    output = tmp_path / "review"
    first = write_review_suite_bundle(output)
    second = write_review_suite_bundle(output)

    assert first.plan.plan_sha256 == second.plan.plan_sha256
    assert (output / REVIEW_PLAN_FILE).is_file()
    assert (output / REVIEW_REQUEST_LEDGER_FILE).is_file()
    assert (output / REVIEW_BUNDLE_FILE).is_file()
    loaded = load_review_suite_bundle(output)
    assert loaded.plan == first.plan
    assert loaded.requests.ledger_sha256 == first.requests.ledger_sha256

    requests = json.loads((output / REVIEW_REQUEST_LEDGER_FILE).read_text(encoding="utf-8"))
    requests["requests"][0]["status"] = "accepted_without_artifacts"
    (output / REVIEW_REQUEST_LEDGER_FILE).write_text(json.dumps(requests), encoding="utf-8")
    with pytest.raises(ExistingOutputError, match="immutable review artifact differs"):
        write_review_suite_bundle(output)
    with pytest.raises(ReviewSuiteValidationError, match="request hash mismatch"):
        load_review_suite_bundle(output)


def test_review_suite_cli_writes_then_validates_bundle(tmp_path, capsys) -> None:
    output = tmp_path / "review-cli"

    assert main(["review-suite", "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["planned_case_count"] == 120
    assert summary["required_video_count"] == 240
    assert summary["executable_case_count"] == 48
    assert summary["blocked_leaf_count"] == 12

    assert main(["review-suite", "--output", str(output), "--validate-only"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["plan_sha256"] == summary["plan_sha256"]
