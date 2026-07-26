"""Adversarial regression tests for release-critical artifact integrity.

These tests deliberately construct metadata that looks superficially valid but
breaks a counterfactual invariant or provenance binding.  Every case must fail
closed; claimed ``passed`` booleans are never evidence by themselves.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dynamic_robot_dataset.common.calibration import calibrate_physics_catalog
from dynamic_robot_dataset.common.contract_v2 import (
    CounterfactualRelation,
    build_counterfactual_family_records,
    validate_counterfactual_family_records,
)
from dynamic_robot_dataset.common.hashing import sha256_file, sha256_json
from dynamic_robot_dataset.common.native_suite import (
    build_planned_counterfactual_family_records,
    plan_suite_cases,
)
from dynamic_robot_dataset.common.readiness import (
    _acceptance_checks,
    _hard_qc_map,
    _model_checks,
)
from dynamic_robot_dataset.common.schema import EpisodeRecord, PhysicsMetadata, PhysicsValue
from dynamic_robot_dataset.common.suites import expand_suite
from dynamic_robot_dataset.common.episode_writer import (
    episode_record_to_table_row,
    write_parquet_atomic,
)


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/families/native_acceptance_160.yaml"


def _planned_subfamily(name: str):
    """Plan related cases independently so this test isolates one family."""

    cases = [case for case in expand_suite(SUITE) if case.subfamily == name]
    return [plan_suite_cases([case])[0] for case in cases]


def _episode(
    index: int,
    *,
    action_family: str = "action-family",
    physics_family: str | None = None,
    action_hash: str | None = None,
    gravity_m_s2: float = -9.81,
) -> EpisodeRecord:
    action_digest = action_hash or sha256_json(
        {"timestamp_s": [0.0, 0.1], "command": [0.0, float(index)]}
    )
    return EpisodeRecord(
        episode_uuid=f"90000000-0000-4000-8000-{index:012d}",
        episode_index=index,
        counterfactual_bundle_id=action_family,
        physics_counterfactual_family_id=physics_family,
        split_group_id="shared-split-group",
        scene_seed=17,
        branch_seed=index,
        family="falling_catch",
        subfamily="centered_vertical_drop",
        intended_branch="success_seeking" if index == 0 else "near_miss",
        actual_outcome="success" if index == 0 else "near_miss",
        task_success=index == 0,
        failure_mode="none" if index == 0 else "spatial_near_miss",
        source_generator="adversarial-test",
        source_generator_version="1",
        config_hash="a" * 64,
        simulator_name="mujoco",
        simulator_version="3.3.0",
        renderer="mujoco",
        physics=PhysicsMetadata(
            parameters={
                "gravity": PhysicsValue(
                    "gravity", (0.0, 0.0, gravity_m_s2), "m/s^2", True, True
                )
            },
            gravity_world_m_s2=(0.0, 0.0, gravity_m_s2),
            gravity_valid=True,
        ),
        extras={
            "derived_action_hash": action_digest,
            "action_hash": action_digest,
            "derived_initial_state_hash": sha256_json({"qpos": [0.0] * 7}),
            "initial_state_hash": sha256_json({"qpos": [0.0] * 7}),
        },
    )


def test_planned_action_family_rejects_identical_action_schedules() -> None:
    planned = _planned_subfamily("centered_vertical_drop")
    assert len(planned) > 1
    same_hash = planned[0].episode_plan.action_hash
    forged = [
        replace(item, episode_plan=replace(item.episode_plan, action_hash=same_hash))
        for item in planned
    ]

    with pytest.raises(ValueError, match="one action trajectory"):
        build_planned_counterfactual_family_records(forged)


def test_planned_physics_family_rejects_nonvarying_declared_intervention() -> None:
    planned = _planned_subfamily("gravity_sweep")
    assert len(planned) == 5
    shared_physics = planned[0].episode_plan.physics
    shared_physics_hash = planned[0].episode_plan.physics_hash
    forged = [
        replace(
            item,
            episode_plan=replace(
                item.episode_plan,
                physics=shared_physics,
                physics_hash=shared_physics_hash,
            ),
        )
        for item in planned
    ]

    with pytest.raises(ValueError, match="does not vary declared intervention gravity"):
        build_planned_counterfactual_family_records(forged)


def test_persisted_action_family_rejects_identical_schedules_and_extra_sibling() -> None:
    same_hash = sha256_json({"timestamp_s": [0.0, 0.1], "command": [0.0, 0.0]})
    identical = [_episode(0, action_hash=same_hash), _episode(1, action_hash=same_hash)]
    declaration = build_counterfactual_family_records(identical)[0]

    problems = validate_counterfactual_family_records([declaration], identical)
    assert any("does not vary the saved action trajectory" in problem for problem in problems)

    valid = [_episode(0), _episode(1)]
    valid_declaration = build_counterfactual_family_records(valid)[0]
    extra = _episode(2)
    problems = validate_counterfactual_family_records(
        [valid_declaration], [*valid, extra]
    )
    assert any("has undeclared extra members" in problem for problem in problems)


def test_persisted_physics_family_rejects_nonvarying_declared_intervention() -> None:
    records = [
        _episode(0, action_family="branch-0", physics_family="gravity-family"),
        _episode(1, action_family="branch-1", physics_family="gravity-family"),
    ]
    declarations = build_counterfactual_family_records(
        records, physics_interventions={"gravity-family": ("gravity",)}
    )
    declaration = next(
        item for item in declarations if item.relation == CounterfactualRelation.PHYSICS
    )

    problems = validate_counterfactual_family_records([declaration], records)
    assert any(
        "does not vary declared intervention gravity" in problem for problem in problems
    )


def _minimal_catalog() -> dict[str, object]:
    return {
        "schema_version": "dynamic-robot-physics-ranges/v1",
        "catalog_id": "adversarial-catalog",
        "fields": {
            "gravity_m_s2": {
                "unit": "m/s^2",
                "nominal": 9.81,
                "candidate": {"minimum": 9.0, "maximum": 10.0},
            }
        },
        "calibration": {
            "required_oracles": ["free_fall"],
            "required_admitted_support": [],
            "candidate_trials_per_value": 1,
        },
    }


def test_calibration_ignores_claimed_pass_and_requires_bound_raw_evidence(
    tmp_path: Path,
) -> None:
    catalog = _minimal_catalog()
    forged_claim = {
        "passed": True,
        "release_eligible": True,
        "oracles": {"free_fall": {"passed": True}},
    }
    report = calibrate_physics_catalog(catalog, observations=forged_claim)
    assert not report.passed and not report.release_eligible
    assert not next(check for check in report.checks if check.name == "observation_schema").passed
    assert not next(check for check in report.checks if check.name == "oracle.free_fall").passed

    episodes = tmp_path / "episodes.parquet"
    qc_report = tmp_path / "dataset_report.json"
    episodes.write_bytes(b"content-addressed episodes")
    qc_report.write_text('{"passed": true}', encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": "dynamic-robot-calibration-approval/v1",
                "approved": True,
                "reviewer": "reviewer",
                "catalog_hash": sha256_json(catalog),
                "observations_hash": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    timestamps = [0.0, 0.1, 0.2]
    evidence = {
        "schema_version": "dynamic-robot-native-calibration-observations/v1",
        "catalog_hash": sha256_json(catalog),
        "source_artifacts": [
            {"role": "episodes", "path": str(episodes), "sha256": sha256_file(episodes)},
            {"role": "qc_report", "path": str(qc_report), "sha256": sha256_file(qc_report)},
        ],
        "oracles": {
            "free_fall": {
                "measurements": {
                    "trials": [
                        {
                            "timestamps_s": timestamps,
                            "vertical_positions_m": [
                                1.0 - 0.5 * 9.81 * timestamp * timestamp
                                for timestamp in timestamps
                            ],
                            "gravity_m_s2": -9.81,
                        }
                    ]
                }
            }
        },
        "approval_artifact": {"path": str(approval), "sha256": sha256_file(approval)},
    }
    report = calibrate_physics_catalog(catalog, observations=evidence)
    assert not report.passed and not report.release_eligible
    assert next(check for check in report.checks if check.name == "oracle.free_fall").passed
    approval_check = next(
        check for check in report.checks if check.name == "release_approval_bound"
    )
    assert not approval_check.passed
    assert "does not bind" in approval_check.message


def test_readiness_rejects_unbound_qc_model_and_acceptance_claims(tmp_path: Path) -> None:
    record = _episode(0, physics_family=None)
    meta = tmp_path / "dataset" / "meta"
    qc = tmp_path / "dataset" / "qc"
    meta.mkdir(parents=True)
    qc.mkdir(parents=True)
    episodes_path = write_parquet_atomic(
        meta / "episodes.parquet", [episode_record_to_table_row(record)]
    )
    (qc / "dataset_report.json").write_text(
        json.dumps(
            {
                "schema_version": "dynamic-robot-qc-report/v2",
                "dataset_root": str(tmp_path / "dataset"),
                "dataset_episodes_sha256": "0" * 64,
                "global_failures": [],
                "episodes": [
                    {
                        "episode_uuid": record.episode_uuid,
                        "passed": True,
                        "release_eligible": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    passed_map, qc_failures, qc_hash = _hard_qc_map(
        (tmp_path / "dataset").resolve(), [record]
    )
    assert passed_map[record.episode_uuid]
    assert any("not bound to the current episodes.parquet" in value for value in qc_failures)
    assert qc_hash is not None and sha256_file(episodes_path) != "0" * 64

    model_checks = _model_checks(
        {"model_evaluation": {"trajectory_improvement": {"minimum": 0.10}}},
        {
            "metrics": {"trajectory_improvement": 1.0},
            "provenance": {
                "dataset_episodes_sha256": "0" * 64,
                "dataset_qc_report_sha256": "1" * 64,
                "model_artifact_sha256": "2" * 64,
                "evaluation_manifest_sha256": "3" * 64,
            },
        },
        dataset_episodes_sha256=sha256_file(episodes_path),
        dataset_qc_report_sha256=qc_hash,
    )
    assert next(
        check for check in model_checks if check.name == "model.trajectory_improvement"
    ).passed
    assert not next(
        check
        for check in model_checks
        if check.name == "model.provenance.dataset_episodes_sha256"
    ).passed
    assert not next(
        check
        for check in model_checks
        if check.name == "model.provenance.dataset_qc_report_sha256"
    ).passed

    forged_report = tmp_path / "claimed-passed-acceptance.json"
    forged_report.write_text(
        json.dumps(
            {
                "schema_version": "dynamic-robot-suite-execution/v1",
                "suite_name": "native_acceptance_160",
                "dataset_root": str(tmp_path / "dataset"),
                "passed_execution": True,
                "full_native": True,
                "planned_case_count": 160,
                "committed_episode_count": 160,
                "acceptance_gates": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    acceptance_checks, _ = _acceptance_checks(
        {"acceptance_suite": {"suite_name": "native_acceptance_160", "branch_count": 160}},
        forged_report.resolve(),
    )
    assert not next(
        check
        for check in acceptance_checks
        if check.name == "acceptance_suite.canonical_report_path"
    ).passed
    assert not next(
        check
        for check in acceptance_checks
        if check.name == "acceptance_suite.plan_present"
    ).passed
    assert not next(
        check
        for check in acceptance_checks
        if check.name == "acceptance_suite.exact_episode_membership"
    ).passed
    assert not next(
        check
        for check in acceptance_checks
        if check.name == "acceptance_suite.qc_v2_bound_and_passed"
    ).passed
