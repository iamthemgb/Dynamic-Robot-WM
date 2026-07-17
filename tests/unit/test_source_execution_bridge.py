from __future__ import annotations

from copy import copy
from pathlib import Path

import pytest

from dynamic_robot_dataset.backends.source_mujoco import SourceMujocoBackend
from dynamic_robot_dataset.common.episode_writer import (
    load_episode_records,
    read_parquet_rows,
)
from dynamic_robot_dataset.common.review_suite import (
    build_review_request_ledger,
    build_review_suite_plan,
)
from dynamic_robot_dataset.common.run_orchestration import (
    RunPlanEpisode,
    finalize_run,
    plan_run,
    run_shard,
)
from dynamic_robot_dataset.common.schema import DatasetInfo
from dynamic_robot_dataset.common.source_execution import (
    SourceExecutionBindingError,
    _semantic_fields,
    execute_source_mujoco_episode,
    materialize_source_mujoco_result,
    prepare_source_review_declaration,
    source_finalization_rows,
)


def _case(case_id: str):
    return next(
        case for case in build_review_suite_plan().cases if case.case_id == case_id
    )


def _entry(declaration: dict) -> RunPlanEpisode:
    return RunPlanEpisode(
        episode_uuid=declaration["episode_uuid"],
        episode_index=declaration["episode_index"],
        shard_id=0,
        declaration=declaration,
        source_scenario_spec_sha256=declaration[
            "source_scenario_spec_sha256"
        ],
    )


def test_frame_interval_contact_does_not_claim_sampled_active_contact() -> None:
    scenario = SourceMujocoBackend().compile_case(_case("P0c-review-03"))
    semantics = _semantic_fields(
        {
            "timestamp": scenario.key_event_time_s,
            "object.motion_mode": "free_flight",
            "contact.count": 0,
        },
        scenario=scenario,
        contacts=(),
        interval_contact=True,
    )
    assert semantics["contact.active"] is False
    assert semantics["event.contact"] is True


def test_declaration_remaps_only_local_index_and_matches_strict_request_identity() -> None:
    plan = build_review_suite_plan()
    case = next(value for value in plan.cases if value.case_id == "P0b-review-00")
    assert case.episode_index == 6
    declaration = prepare_source_review_declaration(case, episode_index=0)
    request = next(
        value
        for value in build_review_request_ledger(plan).requests
        if value.case_id == case.case_id
    )

    assert declaration["episode_index"] == 0
    assert declaration["review_suite_episode_index"] == 6
    assert declaration["review_case"]["episode_index"] == 6
    assert declaration["review_case_sha256"] == case.case_sha256
    assert declaration["source_scenario_spec"]["scenario_id"] == case.case_id
    assert (
        declaration["source_scenario_spec"]["scenario_id"]
        == request.source_scenario_identity["scenario_id"]
    )
    assert declaration["episode_uuid"] == request.episode_uuid


@pytest.fixture(scope="module")
def p0a_unrendered_runtime():
    case = _case("P0a-review-00")
    declaration = prepare_source_review_declaration(case, episode_index=0)
    result = SourceMujocoBackend().run(case, render=False)
    return _entry(declaration), result


@pytest.mark.integration
def test_materialization_labels_from_persisted_rows_not_online_summary(
    p0a_unrendered_runtime,
) -> None:
    entry, original = p0a_unrendered_runtime
    result = copy(original)
    result.outcome = {
        "task_success": False,
        "actual_outcome": "miss",
        "diagnostic_only": True,
    }
    materialization = materialize_source_mujoco_result(entry, result)

    assert materialization.record.task_success is True
    assert materialization.record.actual_outcome == "passive_observation"
    assert materialization.record.objective_evidence["independently_recomputed"] is True
    assert len(materialization.record.objective_evidence["evidence_hash"]) == 64
    assert "online_outcome_disagrees_with_independent_replay" in (
        materialization.record.quality_flags
    )
    assert materialization.record.key_event_name
    assert materialization.record.key_event_time_s is not None
    assert materialization.record.extras["review_suite_episode_index"] == 0
    assert all(
        row["action.actuator_command"]
        == row["simulator.applied_actuator_ctrl"]
        == []
        for row in materialization.high_rate_rows
    )


@pytest.mark.integration
def test_materialization_rejects_runtime_source_hash_change(
    p0a_unrendered_runtime,
) -> None:
    entry, original = p0a_unrendered_runtime
    result = copy(original)
    result.source_hashes = {
        **result.source_hashes,
        "compiled_scene_xml": "0" * 64,
    }
    with pytest.raises(SourceExecutionBindingError, match="source hashes differ"):
        materialize_source_mujoco_result(entry, result)


@pytest.mark.integration
def test_real_actuated_materialization_uses_eight_applied_controls() -> None:
    case = _case("F1a-review-00")
    declaration = prepare_source_review_declaration(case, episode_index=0)
    result = SourceMujocoBackend().run(case, render=False)
    materialization = materialize_source_mujoco_result(
        _entry(declaration), result
    )

    assert materialization.record.task_success is True
    assert materialization.record.key_event_name == "bilateral_grasp_onset"
    assert materialization.record.key_event_time_s is not None
    assert all(
        len(row["action.actuator_command"]) == 8
        and row["action.actuator_command"]
        == row["simulator.applied_actuator_ctrl"]
        for row in materialization.high_rate_rows
    )


@pytest.mark.integration
def test_real_p0a_writer_round_trip_and_finalization_metadata(tmp_path: Path) -> None:
    case = _case("P0a-review-00")
    declaration = prepare_source_review_declaration(case, episode_index=0)
    root = tmp_path / "p0a"
    plan_run(
        root,
        {
            "schema_version": "source-execution-bridge-test/v1",
            "backend": "source_mujoco",
            "generator_git_commit": "test",
        },
        [declaration],
        shard_count=1,
    )
    result = run_shard(
        root,
        0,
        lambda entry: execute_source_mujoco_episode(entry, render=True),
    )
    assert result.committed_count == 1
    record = load_episode_records(root)[0]
    frame_rows = read_parquet_rows(root / record.frame_data_path)
    high_rate_rows = read_parquet_rows(root / record.high_rate_path)
    object_rows = read_parquet_rows(root / record.object_states_path)
    assert len(frame_rows) == 24
    assert high_rate_rows
    assert object_rows
    assert all(row.get("grasp.center_position") is None for row in object_rows)
    assert record.extras["source_scenario_spec"]["scenario_id"] == case.case_id
    assert record.extras["review_suite_episode_index"] == case.episode_index

    finalization = source_finalization_rows([record])
    assert len(finalization.cameras) == 2
    assert len(finalization.provenance) == 1
    finalized = finalize_run(
        root,
        DatasetInfo(name="source-execution-p0a-test", generator_version="test"),
        tasks=[
            {
                "task_index": record.task_index,
                "family": record.family,
                "subfamily": record.subfamily,
            }
        ],
        cameras=finalization.cameras,
        provenance=finalization.provenance,
        counterfactual_families=[],
    )
    assert len(finalized) == 1
    assert (root / "meta" / "cameras.parquet").is_file()
    assert (root / "meta" / "provenance.parquet").is_file()
