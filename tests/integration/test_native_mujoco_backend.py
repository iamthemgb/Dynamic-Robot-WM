from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import pytest

from dynamic_robot_dataset.backends.mujoco_native import (
    NativeMuJoCoBackend,
    evaluate_saved_native_episode,
    make_scenario_spec,
    scenario_to_episode_plan,
)
from dynamic_robot_dataset.backends.base import PhysicsRangeProvenance
from dynamic_robot_dataset.common.contract_v2 import validate_v2_frame_semantics
from dynamic_robot_dataset.common.contract_v2 import (
    CounterfactualRelation,
    validate_counterfactual_family_records,
)
from dynamic_robot_dataset.common.native_suite import (
    build_planned_counterfactual_family_records,
    plan_suite_cases,
)
from dynamic_robot_dataset.common.suites import expand_suite
from dynamic_robot_dataset.cli import _episode_record
from dynamic_robot_dataset.smoke_runner import _record as _smoke_episode_record


pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]


def test_native_centered_drop_is_actuator_only_and_recomputable() -> None:
    pytest.importorskip("mujoco")
    spec = make_scenario_spec("centered_vertical_drop", seed=2)
    backend = NativeMuJoCoBackend(width=208, height=120)
    result = backend.run(scenario_to_episode_plan(spec), render=False)

    audit = result.backend_provenance["runtime_audit"]
    assert audit["initial_object_state_writes"] == 1
    assert audit["object_state_writes_after_initialization"] == 0
    assert audit["direct_robot_state_writes_after_initialization"] == 0
    assert audit["equality_constraint_count"] == 0
    assert audit["control_updates"] > 0
    assert result.simulation.physics_qc["physics_qc_pass"] is True
    physics_checks = result.simulation.physics_qc["checks"]
    assert physics_checks["momentum_impulse_accounting_consistent"] is True
    assert physics_checks["momentum_impulse_relative_error"] < 0.08
    assert physics_checks["joint_positions_within_limits"] is True
    assert physics_checks["actuator_forces_within_limits"] is True
    assert result.frame_rows[-1]["timestamp"] < spec.maximum_duration_s
    assert validate_v2_frame_semantics(result.frame_rows) == []
    assert result.backend_provenance["duration_policy"] == "event_adaptive"
    assert "visual_style_unvalidated" in result.quality_flags

    independent = evaluate_saved_native_episode(
        spec,
        result.frame_rows,
        result.event_rows,
        result.transition_events,
    )
    assert independent.outcome.task_success == result.simulation.outcome.task_success
    assert independent.outcome.failure_mode == result.simulation.outcome.failure_mode
    assert independent.actual_outcome_class == result.simulation.actual_outcome
    assert result.backend_provenance["native_scenario_spec"] == spec.to_dict()


def test_native_no_op_holds_home_command_and_is_measured_from_motion() -> None:
    pytest.importorskip("mujoco")
    spec = make_scenario_spec("centered_vertical_drop", seed=9, branch="no_op")
    result = NativeMuJoCoBackend(width=32, height=32).run(
        scenario_to_episode_plan(spec), render=False
    )

    commands = [row["action.command.joint_position"] for row in result.frame_rows]
    assert all(command == pytest.approx(commands[0], abs=1e-10) for command in commands)
    assert all(row["action.command.enabled"] is False for row in result.frame_rows)
    metrics = result.simulation.outcome.metrics
    assert metrics["command_motion_measured"] is False
    assert metrics["maximum_robot_joint_displacement_rad"] <= 2e-2
    assert metrics["maximum_robot_tool_displacement_m"] <= 2e-2
    assert metrics["controller_no_op_measured"] is True
    assert result.simulation.actual_outcome == "no_op"


def test_failed_transport_with_full_dwell_is_scored_as_partial_progress() -> None:
    pytest.importorskip("mujoco")
    item = next(
        value
        for value in plan_suite_cases(
            expand_suite(ROOT / "configs/families/native_acceptance_160.yaml")
        )
        if value.episode_uuid == "241e76f8-c583-546b-a1ea-4d9b39bfbb7b"
    )
    result = NativeMuJoCoBackend(width=32, height=32).run(
        item.episode_plan, render=False
    )

    assert result.simulation.outcome.task_success is False
    assert 0.5 <= result.simulation.outcome.partial_success_score < 1.0
    assert result.simulation.outcome.failure_mode == "contact_without_completion"
    assert result.simulation.actual_outcome == "partial_success"

    records = (
        _episode_record(
            result.simulation,
            item.case.case_index,
            "a" * 40,
            result.as_renderer_payload(),
        ),
        _smoke_episode_record(result.simulation, item.case.case_index, "a" * 40),
    )
    for record in records:
        record.validate()
        persisted = record.to_dict()
        assert persisted["actual_outcome"] == "partial_success"
        assert persisted["actual_outcome_class"] == "partial_success"
        assert persisted["primary_failure_code"] == "contact_without_completion"


@pytest.mark.parametrize(
    ("case_index", "initial_surface"),
    [
        (46, "ramp_surface"),
        (47, "table_surface"),
        (117, "ramp_surface"),
        (118, "table_surface"),
    ],
)
def test_native_mode_transitions_start_supported_and_complete_truthfully(
    case_index: int,
    initial_surface: str,
) -> None:
    pytest.importorskip("mujoco")
    item = next(
        value
        for value in plan_suite_cases(
            expand_suite(ROOT / "configs/families/native_acceptance_160.yaml")
        )
        if value.case.case_index == case_index
    )
    result = NativeMuJoCoBackend(width=32, height=32).run(
        item.episode_plan, render=False
    )

    first = result.high_rate_rows[0]
    assert first["object.active_surface"] == initial_surface
    assert first["object.motion_mode"] in {"rolling", "sliding"}
    if item.case.subfamily in {"ramp_launch", "roll_off_edge"}:
        transition = result.simulation.outcome.metrics["surface_to_free_flight"]
        assert transition["passed"] is True
        assert transition["free_flight_dwell_s"] >= transition["minimum_dwell_s"]
    assert result.simulation.actual_outcome == "success"
    assert result.simulation.physics_qc["physics_qc_pass"] is True
    motion_transitions = [
        event
        for event in result.transition_events
        if event["event_type"] == "motion_mode_transition"
    ]
    assert len(motion_transitions) < 20
    checks = result.simulation.physics_qc["checks"]
    if item.case.subfamily in {"ramp_launch", "roll_off_edge"}:
        assert checks["free_flight_measurement_available"] is True
        assert checks["free_flight_acceleration_consistent"] is True
        assert checks["free_flight_energy_consistent"] is True
    if case_index == 117:
        assert result.high_rate_rows[-1]["timestamp"] < 0.70
        assert (
            result.backend_provenance["runtime_audit"]["terminal_reason"]
            == "terminal_context_complete"
        )


def test_ramp_support_hysteresis_suppresses_prelaunch_solver_chatter() -> None:
    pytest.importorskip("mujoco")
    spec = make_scenario_spec("ramp_launch", seed=0)
    result = NativeMuJoCoBackend(width=32, height=32).run(
        scenario_to_episode_plan(spec), render=False
    )
    boundary = next(
        event
        for event in result.transition_events
        if event["event_type"] == "surface_release_boundary_crossing"
    )
    prelaunch_motion = [
        event
        for event in result.transition_events
        if event["event_type"] == "motion_mode_transition"
        and event["timestamp"] <= boundary["timestamp"] + 1e-9
    ]

    assert len(prelaunch_motion) == 1
    assert prelaunch_motion[0]["from"] in {"rolling", "sliding"}
    assert prelaunch_motion[0]["to"] == "free_flight"


@pytest.mark.parametrize(
    ("scenario", "fixture"),
    [
        ("table_bounce", "table_surface"),
        ("wall_rebound", "wall_surface"),
        ("angled_barrier_rebound", "angled_barrier_surface"),
    ],
)
def test_native_fixture_contact_onsets_are_labeled_as_impacts(
    scenario: str,
    fixture: str,
) -> None:
    pytest.importorskip("mujoco")
    spec = make_scenario_spec(scenario, seed=17)
    result = NativeMuJoCoBackend(width=32, height=32).run(
        scenario_to_episode_plan(spec), render=False
    )
    contact = next(event for event in result.event_rows if event["object_b"] == fixture)
    impact = next(
        event
        for event in result.transition_events
        if event["event_type"] == "motion_mode_transition"
        and event["to"] == "impact"
        and event["active_surface"] == fixture
    )

    assert contact["normal_velocity_pre_m_s"] <= -0.10
    assert impact["timestamp"] == pytest.approx(contact["timestamp"])
    assert impact["from"] == "free_flight"


def test_physics_counterfactuals_use_identical_fixed_action_horizons() -> None:
    pytest.importorskip("mujoco")
    base = make_scenario_spec("table_bounce", seed=41)
    base = replace(
        base,
        maximum_duration_s=0.4,
        minimum_terminal_context_s=0.1,
        physics_counterfactual_family_id="physics-family-test",
    )
    sibling = replace(
        base,
        gravity_m_s2=(0.0, 0.0, -6.0),
        object=replace(base.object, effective_restitution_target=0.75),
    )
    backend = NativeMuJoCoBackend(width=64, height=64)
    results = [
        backend.run(scenario_to_episode_plan(spec), render=False)
        for spec in (base, sibling)
    ]
    action_tables = [
        [
            {
                key: value
                for key, value in row.items()
                if key == "timestamp" or key.startswith("action.")
            }
            for row in result.frame_rows
        ]
        for result in results
    ]
    assert action_tables[0] == action_tables[1]
    assert results[0].frame_rows[-1]["timestamp"] == pytest.approx(0.4)
    assert all(
        result.backend_provenance["duration_policy"]
        == "fixed_physics_counterfactual_horizon"
        for result in results
    )
    assert all(
        result.backend_provenance["runtime_audit"]["terminal_reason"]
        == "physics_counterfactual_fixed_horizon_complete"
        for result in results
    )


def test_immutable_plan_counterfactual_hashes_match_final_record_contract() -> None:
    pytest.importorskip("mujoco")
    planned = plan_suite_cases(
        expand_suite(ROOT / "configs/families/native_acceptance_160.yaml")
    )
    by_bundle = {}
    for item in planned:
        if item.execution_backend == "native_mujoco":
            by_bundle.setdefault(item.case.counterfactual_bundle_id, []).append(item)
    siblings = next(values for values in by_bundle.values() if len(values) >= 3)
    sibling_uuids = {item.episode_uuid for item in siblings}
    declaration = next(
        value
        for value in build_planned_counterfactual_family_records(planned)
        if value.relation == CounterfactualRelation.ACTION
        and set(value.expected_episode_uuids) == sibling_uuids
    )
    backend = NativeMuJoCoBackend(width=32, height=32)
    records = []
    for episode_index, item in enumerate(siblings):
        run = backend.run(item.episode_plan, render=False)
        rendered = run.as_renderer_payload()
        rendered["backend_provenance"] = run.backend_provenance
        records.append(
            _episode_record(
                run.simulation,
                episode_index,
                "e" * 40,
                rendered,
            )
        )
    derived = {
        record.episode_uuid: {
            "derived_initial_state_hash": record.extras["derived_initial_state_hash"],
            "derived_action_hash": record.extras["derived_action_hash"],
        }
        for record in records
    }
    assert validate_counterfactual_family_records(
        [declaration], records, derived_by_uuid=derived
    ) == []


def test_native_restitution_persists_separated_sample_evidence() -> None:
    pytest.importorskip("mujoco")
    base = make_scenario_spec("table_bounce", seed=41)
    spec = replace(
        base,
        maximum_duration_s=0.8,
        minimum_terminal_context_s=0.1,
        physics_counterfactual_family_id="restitution-evidence-test",
        extras={**base.extras, "suite_subfamily": "restitution_sweep"},
    )
    result = NativeMuJoCoBackend(width=32, height=32).run(
        scenario_to_episode_plan(spec), render=False
    )

    checks = result.simulation.physics_qc["checks"]
    assert checks["restitution_sweep_measurement_available"] is True
    assert checks["rebound_measurement_count"] >= 1
    assert checks["restitution_measurement_sample_count"] >= 2
    measurement = checks["restitution_measurements"][0]
    assert measurement["incoming_sample_time_s"] < measurement["contact_event_time_s"]
    assert measurement["outgoing_sample_time_s"] > measurement["contact_event_time_s"]
    assert measurement["incoming_normal_velocity_m_s"] < 0.0
    assert measurement["outgoing_normal_velocity_m_s"] > 0.0
    assert "last_separated_incoming" in measurement["source"]
    event = result.event_rows[0]
    assert event["normal_world"][2] > 0.0
    assert event["normal_velocity_source"].endswith(
        "not_restitution_measurement"
    )
    assert event["restitution_measurement_status"] == "measured"


def test_restitution_sweep_fails_closed_before_outgoing_separation() -> None:
    pytest.importorskip("mujoco")
    base = make_scenario_spec("table_bounce", seed=41)
    spec = replace(
        base,
        maximum_duration_s=0.38,
        minimum_terminal_context_s=0.01,
        physics_counterfactual_family_id="restitution-no-separation-test",
        extras={**base.extras, "suite_subfamily": "restitution_sweep"},
    )
    result = NativeMuJoCoBackend(width=32, height=32).run(
        scenario_to_episode_plan(spec), render=False
    )

    checks = result.simulation.physics_qc["checks"]
    assert checks["restitution_sweep_measurement_available"] is False
    assert checks["rebound_measurement_count"] == 0
    assert checks["restitution_missing_outgoing_sample_count"] == 1
    assert result.simulation.physics_qc["physics_qc_pass"] is False
    assert result.event_rows[0]["restitution_measurement_status"] == (
        "no_separated_outgoing_sample"
    )


def test_calibrated_rebound_without_separated_measurement_fails_target_match(
    tmp_path: Path,
) -> None:
    """Calibration makes a missing rebound measurement a hard failure."""

    pytest.importorskip("mujoco")
    artifact = tmp_path / "calibration-approval.json"
    artifact.write_text('{"approved": true}\n', encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    base = make_scenario_spec("table_bounce", seed=41)
    spec = replace(
        base,
        maximum_duration_s=0.38,
        minimum_terminal_context_s=0.01,
        physics_provenance=PhysicsRangeProvenance(
            range_version="calibrated-test/v1",
            partition="train_id",
            calibrated=True,
            calibration_artifact=artifact_hash,
            calibration_artifact_path=str(artifact),
        ),
    )

    result = NativeMuJoCoBackend(width=32, height=32).run(
        scenario_to_episode_plan(spec), render=False
    )

    checks = result.simulation.physics_qc["checks"]
    assert checks["restitution_target_match_required"] == 1
    assert checks["restitution_target_fixture"] == "table_surface"
    assert checks["rebound_measurement_count"] == 0
    assert checks["measured_effective_restitution"] == "not_observed"
    assert checks["measured_restitution_matches_target"] is False
    assert result.simulation.physics_qc["physics_qc_pass"] is False


def test_bounce_interception_orders_contacts_and_preserves_honest_no_op() -> None:
    pytest.importorskip("mujoco")
    planned = [
        item
        for item in plan_suite_cases(
            expand_suite(ROOT / "configs/families/native_acceptance_160.yaml")
        )
        if item.scenario_spec is not None
        and item.case.subfamily == "bounce_to_robot_interception"
    ]
    active = next(
        item for item in planned if item.episode_plan.intended_branch == "success_seeking"
    )
    no_op = next(
        item
        for item in planned
        if item.case.counterfactual_bundle_id == active.case.counterfactual_bundle_id
        and item.episode_plan.intended_branch == "no_op"
    )
    backend = NativeMuJoCoBackend(width=32, height=32)

    active_result = backend.run(active.episode_plan, render=False)
    active_contacts = [event["object_b"] for event in active_result.event_rows]
    active_checks = active_result.simulation.physics_qc["checks"]
    assert active_result.simulation.actual_outcome == "success"
    assert active_contacts == ["table_surface", "native_tool"]
    assert active_checks["distinct_contact_count"] == len(active_contacts) == 2
    assert active_checks["task_contact_count"] == 1
    assert active_checks["distinct_contact_count_bounded"] is True
    assert active_result.simulation.physics_qc["physics_qc_pass"] is True

    no_op_result = backend.run(no_op.episode_plan, render=False)
    no_op_contacts = [event["object_b"] for event in no_op_result.event_rows]
    no_op_checks = no_op_result.simulation.physics_qc["checks"]
    assert no_op_result.simulation.actual_outcome == "no_op"
    assert no_op_result.simulation.outcome.metrics["controller_no_op_measured"] is True
    assert "native_tool" not in no_op_contacts
    assert len(no_op_contacts) <= 2
    assert no_op_checks["distinct_contact_count"] == len(no_op_contacts)
    assert no_op_checks["distinct_contact_count_bounded"] is True
    assert no_op_result.simulation.physics_qc["physics_qc_pass"] is True


def test_rolling_initial_support_is_persisted_but_not_a_dynamic_contact() -> None:
    pytest.importorskip("mujoco")
    planned = plan_suite_cases(
        expand_suite(ROOT / "configs/families/native_acceptance_160.yaml")
    )
    item = next(
        candidate
        for candidate in planned
        if candidate.scenario_spec is not None
        and candidate.case.subfamily == "straight_ball_left_to_right"
    )

    result = NativeMuJoCoBackend(width=32, height=32).run(
        item.episode_plan, render=False
    )

    assert len(result.event_rows) == 1
    support = result.event_rows[0]
    assert support["event_type"] == "contact_begin"
    assert support["object_b"] == "table_surface"
    assert support["contact_role"] == "initial_support"
    assert support["timestamp"] <= 2.0 / item.scenario_spec.sim_hz + 1e-12
    checks = result.simulation.physics_qc["checks"]
    assert checks["recorded_contact_event_count"] == 1
    assert checks["initial_support_event_count"] == 1
    assert checks["distinct_contact_count"] == 0
    assert checks["distinct_contact_count_bounded"] is True
    assert result.simulation.physics_qc["physics_qc_pass"] is True


def test_every_native_acceptance_subfamily_completes_the_backend_lifecycle() -> None:
    """Exercise every retained rigid scenario, not only its typed planner."""

    pytest.importorskip("mujoco")
    planned = plan_suite_cases(
        expand_suite(ROOT / "configs/families/native_acceptance_160.yaml")
    )
    by_subfamily = {}
    for item in planned:
        if item.scenario_spec is not None:
            by_subfamily.setdefault(
                (item.case.family, item.case.subfamily), item.scenario_spec
            )

    assert len(by_subfamily) == 35
    backend = NativeMuJoCoBackend(width=32, height=32)
    for scenario, spec in sorted(by_subfamily.items()):
        result = backend.run(scenario_to_episode_plan(spec), render=False)
        audit = result.backend_provenance["runtime_audit"]
        assert audit["initial_object_state_writes"] == 1, scenario
        assert audit["object_state_writes_after_initialization"] == 0, scenario
        assert audit["direct_robot_state_writes_after_initialization"] == 0, scenario
        assert audit["equality_constraint_count"] == 0, scenario
        assert audit["simulation_steps"] > 0, scenario
        assert validate_v2_frame_semantics(result.frame_rows) == [], scenario
        independently_recomputed = evaluate_saved_native_episode(
            spec,
            result.frame_rows,
            result.event_rows,
            result.transition_events,
        )
        assert (
            independently_recomputed.actual_outcome_class
            == result.simulation.actual_outcome
        ), scenario


def test_acceptance_edge_contacts_produce_measured_contact_failures() -> None:
    """Keep real failure coverage tied to native contact evidence."""

    pytest.importorskip("mujoco")
    planned = plan_suite_cases(
        expand_suite(ROOT / "configs/families/native_acceptance_160.yaml")
    )
    requested = {
        ("rolling_interception", "paddle_redirect"),
        ("projectile_rebound", "direct_interception"),
    }
    # Use stable, measured failures.  Intended outcome mismatches remain in
    # the acceptance confusion matrix and are never retried or relabeled.
    selected_case_indices = {
        ("rolling_interception", "paddle_redirect"): 75,
        ("projectile_rebound", "direct_interception"): 90,
    }
    selected = {}
    for item in planned:
        key = (item.case.family, item.case.subfamily)
        if (
            key in requested
            and item.case.case_index == selected_case_indices[key]
        ):
            selected[key] = item
    assert set(selected) == requested

    backend = NativeMuJoCoBackend(width=32, height=32)
    for key, item in selected.items():
        result = backend.run(item.episode_plan, render=False)
        tool_contacts = [
            event
            for event in result.event_rows
            if event["object_b"] == "native_tool"
        ]
        assert tool_contacts, key
        assert all(
            event["contact_role"] == "task_contact"
            and math.isfinite(float(event["normal_impulse_n_s"]))
            and float(event["normal_impulse_n_s"]) > 0.0
            for event in tool_contacts
        ), key
        assert result.simulation.actual_outcome == "contact_failure", key
        assert result.simulation.outcome.task_success is False, key
        assert result.simulation.physics_qc["physics_qc_pass"] is True, key
