from __future__ import annotations

from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

from dynamic_robot_dataset.backends.source_mujoco import (
    SourceMujocoBackend,
    compile_review_case,
)
from dynamic_robot_dataset.backends.source_mujoco.model import compile_source_model
from dynamic_robot_dataset.backends.source_mujoco.profiles import (
    RIGID_REVIEW_PROFILE,
    timestep_comparison_failures,
)
from dynamic_robot_dataset.backends.source_mujoco.rigid_breadth import (
    SurfaceAdmission,
    sample_admitted_surface,
    sample_surface_candidate,
)
from dynamic_robot_dataset.backends.source_mujoco.source_spec import (
    prepare_review_case,
)
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.common.source_evaluators import (
    validate_source_evaluator_contract,
)
from dynamic_robot_dataset.scenarios.f2e_multi_surface_rebound import (
    FLOOR_TO_WALL_REPAIR,
)


def _case(leaf: str, rollout: int):
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == leaf and case.rollout_index == rollout
    )


def _supports(scenario, fixture_id: str):
    return tuple(
        surface
        for surface in scenario.surfaces
        if surface.role == "structural_support"
        and surface.supports_fixture_id == fixture_id
    )


def test_f2b_compiles_a_grounded_ramp_and_transition_contract() -> None:
    scenario = compile_review_case(_case("F2b", 0))
    contract = scenario.surface_transition_contract

    assert contract is not None
    assert scenario.ordered_contact_contract is None
    assert scenario.sampled_surface_contract is None
    assert scenario.simulation_hz == 600
    assert contract.support_surface_id == "owned_ramp_launch_surface"
    assert contract.minimum_support_contact_s == pytest.approx(0.08)
    assert contract.minimum_free_flight_s == pytest.approx(0.08)

    ramp = next(
        surface
        for surface in scenario.surfaces
        if surface.name == contract.support_surface_id
    )
    supports = _supports(scenario, ramp.name)
    assert ramp.role == "ramp"
    assert ramp.expected_task_contact is True
    assert len(supports) == 4
    assert all(surface.grounded_plane_z_m == 0.0 for surface in supports)
    assert all(
        surface.support_interface_maximum_mismatch_m <= 0.002
        for surface in supports
    )


def test_f2b_robotiq_model_overrides_do_not_change_f2c_defaults() -> None:
    backend = SourceMujocoBackend()

    def compiled_xml(leaf_id: str) -> ET.Element:
        scenario = backend.compile_case(_case(leaf_id, 1))
        compiled = compile_source_model(
            scenario,
            source_dependency=backend.source_dependency,
            robocasa_dependency=(
                backend.robocasa_dependency
                if scenario.requires_real_robocasa
                else None
            ),
        )
        return ET.fromstring(compiled.xml)

    f2b = compiled_xml("F2b")
    f2c = compiled_xml("F2c")
    for name in (
        "rq_left_pad_thick_collision_pad",
        "rq_right_pad_thick_collision_pad",
    ):
        f2b_pad = f2b.find(f".//geom[@name='{name}']")
        f2c_pad = f2c.find(f".//geom[@name='{name}']")
        assert f2b_pad is not None
        assert f2c_pad is not None
        assert float(f2b_pad.get("size", "").split()[0]) == pytest.approx(
            0.0125
        )
        assert float(f2b_pad.get("margin", "nan")) == pytest.approx(0.0)
        assert float(f2c_pad.get("size", "").split()[0]) == pytest.approx(
            RIGID_REVIEW_PROFILE.robotiq_pad_half_depth_m
        )

    f2b_actuator = f2b.find(".//actuator/general[@name='rq_fingers_actuator']")
    f2c_actuator = f2c.find(".//actuator/general[@name='rq_fingers_actuator']")
    assert f2b_actuator is not None
    assert f2c_actuator is not None
    assert tuple(
        float(value) for value in f2b_actuator.get("forcerange", "").split()
    ) == pytest.approx((-0.085, 0.085))
    assert tuple(
        float(value) for value in f2c_actuator.get("forcerange", "").split()
    ) == pytest.approx((-0.16, 0.16))


def test_f2b_robotiq_nominal_success_is_physical_and_replayable() -> None:
    result = SourceMujocoBackend().run(_case("F2b", 1), render=False)
    penetration = result.physics_qc["penetration"]["metrics"]
    evidence = result.physics_qc["task_evidence"]

    assert result.outcome["task_success"] is True
    assert result.outcome["actual_outcome"] == "success"
    assert result.outcome["intended_outcome_match"] is True
    assert result.outcome["saved_artifact_objective_replay_matches"] is True
    assert result.physics_qc["physics_qc_pass"] is True
    assert penetration["maximum_gripper_penetration_m"] <= 0.002
    assert (
        result.physics_qc["maximum_passive_finger_acceleration_rad_s2"]
        <= result.physics_qc["passive_finger_acceleration_limit_rad_s2"]
    )
    assert evidence["sustained_opposing_bilateral_contacts"] is True
    assert evidence["stable_object_to_grasp_transform"] is True
    assert evidence["retained_through_final_state"] is True
    assert evidence["rolling_or_sliding_slip_within_limit"] is True
    assert evidence["friction_deceleration_consistent"] is True
    assert result.runtime_audit["mutation_boundary_violations"] == 0

    source_spec = prepare_review_case(_case("F2b", 1))
    controller_plan = source_spec.physics["controller_plan"]
    assert controller_plan["robotiq_controller_target_bias_m"] == pytest.approx(
        (0.019, 0.0, 0.0)
    )
    assert controller_plan["robotiq_tendon_target"] == pytest.approx(96.0)
    assert controller_plan["robotiq_actuator_force_limit_n"] == pytest.approx(0.085)
    assert controller_plan["robotiq_pad_half_depth_m"] == pytest.approx(0.0125)
    assert controller_plan["robotiq_pad_contact_margin_m"] == pytest.approx(0.0)


@pytest.mark.integration
def test_f2b_robotiq_nominal_passes_the_600_1200_timestep_gate() -> None:
    backend = SourceMujocoBackend()
    scenario = backend.compile_case(_case("F2b", 1))
    observations = []
    for simulation_hz in (600, 1200):
        result = backend.run(
            replace(scenario, simulation_hz=simulation_hz), render=False
        )
        event_time = float(result.outcome["key_event_time_s"])
        event = min(
            result.high_rate_rows,
            key=lambda row: abs(float(row["timestamp"]) - event_time),
        )
        penetration = result.physics_qc["penetration"]["metrics"]
        assert result.outcome["actual_outcome"] == "success"
        assert result.physics_qc["physics_qc_pass"] is True
        assert result.outcome["saved_artifact_objective_replay_matches"] is True
        assert penetration["maximum_gripper_penetration_m"] <= 0.002
        observations.append(
            {
                "outcome": result.outcome["actual_outcome"],
                "task_success": result.outcome["task_success"],
                "physics_qc_pass": result.physics_qc["physics_qc_pass"],
                "saved_artifact_objective_replay_matches": result.outcome[
                    "saved_artifact_objective_replay_matches"
                ],
                "key_event_time_s": event_time,
                "key_event_position_m": event["object.position"],
            }
        )
    assert timestep_comparison_failures(*observations) == ()


@pytest.mark.parametrize(
    ("rollout", "expected_outcome"),
    (
        (0, "success"),
        (1, "success"),
        (2, "miss"),
        (3, "miss"),
        (4, "contact_failure"),
        (5, "contact_failure"),
    ),
)
def test_f2b_fixed_six_passes_strict_physics_and_saved_replay(
    rollout: int,
    expected_outcome: str,
) -> None:
    result = SourceMujocoBackend().run(_case("F2b", rollout), render=False)
    penetration = result.physics_qc["penetration"]["metrics"]
    evidence = result.physics_qc["task_evidence"]

    assert result.outcome["actual_outcome"] == expected_outcome
    assert result.outcome["task_success"] is (rollout in {0, 1})
    assert result.outcome["intended_outcome_match"] is True
    assert result.outcome["saved_artifact_objective_replay_matches"] is True
    assert result.physics_qc["physics_qc_pass"] is True
    assert evidence["surface_transition_pass"] is True
    assert penetration["maximum_gripper_penetration_m"] <= 0.002
    assert penetration["maximum_task_surface_penetration_m"] <= 0.003
    assert result.runtime_audit["mutation_boundary_violations"] == 0


@pytest.mark.parametrize(
    ("rollout", "expected_variant", "expected_ids"),
    (
        (0, "floor_to_wall", ("owned_multi_floor", "owned_multi_wall")),
        (2, "flight_to_table_bounce", ("owned_multi_wall", "owned_multi_table")),
    ),
)
def test_f2e_binds_ordered_stable_surface_identities(
    rollout: int,
    expected_variant: str,
    expected_ids: tuple[str, str],
) -> None:
    scenario = compile_review_case(_case("F2e", rollout))
    contract = scenario.ordered_contact_contract

    assert scenario.task_variant == expected_variant
    assert contract is not None
    assert scenario.surface_transition_contract is None
    assert scenario.sampled_surface_contract is None
    assert contract.ordered_surface_ids == expected_ids
    assert len(contract.ordered_surface_normals_world_xyz) == len(expected_ids)
    assert contract.reject_contact_chatter is True

    fixtures = {surface.name: surface for surface in scenario.surfaces}
    assert set(expected_ids) <= fixtures.keys()
    for surface_id in expected_ids:
        fixture = fixtures[surface_id]
        if fixture.role in {"floor", "table", "slope", "ramp"}:
            assert len(_supports(scenario, surface_id)) == 4


def test_f2e_floor_to_wall_repair_is_variant_scoped_and_versioned() -> None:
    repaired = compile_review_case(_case("F2e", 0))
    unchanged = compile_review_case(_case("F2e", 2))
    repair = FLOOR_TO_WALL_REPAIR.to_dict()

    repaired_wall = next(
        surface for surface in repaired.surfaces if surface.name == "owned_multi_wall"
    )
    unchanged_wall = next(
        surface for surface in unchanged.surfaces if surface.name == "owned_multi_wall"
    )
    assert repair["schema_version"].endswith("/v1")
    assert repaired.ballistic_event_time_s == pytest.approx(
        repair["catch_event_time_s"]
    )
    assert repaired_wall.half_size_m[0] == pytest.approx(
        repair["wall_tangent_half_extent_m"]
    )
    assert unchanged_wall.half_size_m[0] == pytest.approx(0.32)
    assert unchanged.ballistic_event_time_s == pytest.approx(0.8631192660550459)


@pytest.mark.parametrize(
    ("rollout", "expected_outcome", "expected_rate"),
    (
        (4, "contact_failure", 600),
        (5, "miss", 1200),
    ),
)
def test_f2d_repaired_controller_negative_cases_pass_without_assistance(
    rollout: int,
    expected_outcome: str,
    expected_rate: int,
) -> None:
    result = SourceMujocoBackend().run(_case("F2d", rollout), render=False)
    penetration = result.physics_qc["penetration"]["metrics"]

    assert result.scenario.simulation_hz == expected_rate
    assert result.outcome["actual_outcome"] == expected_outcome
    assert result.outcome["intended_outcome_match"] is True
    assert result.outcome["saved_artifact_objective_replay_matches"] is True
    assert result.physics_qc["physics_qc_pass"] is True
    assert penetration["maximum_gripper_penetration_m"] <= 0.002
    assert penetration["maximum_task_surface_penetration_m"] <= 0.003
    assert result.runtime_audit["mutation_boundary_violations"] == 0


@pytest.mark.parametrize("rollout", range(6))
def test_f2e_fixed_six_passes_strict_physics_and_saved_replay(rollout: int) -> None:
    case = _case("F2e", rollout)
    result = SourceMujocoBackend().run(case, render=False)
    penetration = result.physics_qc["penetration"]["metrics"]
    evidence = result.physics_qc["task_evidence"]

    assert result.scenario.simulation_hz == FLOOR_TO_WALL_REPAIR.required_simulation_hz
    assert result.physics_qc["physics_qc_pass"] is True
    assert result.outcome["task_success"] is (rollout in {0, 1})
    assert result.outcome["intended_outcome_match"] is True
    assert result.outcome["saved_artifact_objective_replay_matches"] is True
    assert evidence["ordered_contact_sequence_pass"] is True
    assert penetration["maximum_gripper_penetration_m"] <= 0.002
    assert penetration["maximum_task_surface_penetration_m"] <= 0.003
    assert result.runtime_audit["mutation_boundary_violations"] == 0


def test_f2f_sampler_is_deterministic_and_fails_closed_for_rejected_candidates() -> None:
    first = sample_surface_candidate("random_plane_bounce", source_seed=0)
    second = sample_surface_candidate("random_plane_bounce", source_seed=0)

    assert first == second
    assert first.admission.admitted is True
    assert sample_admitted_surface("random_plane_bounce", source_seed=0) == first

    rejected = sample_surface_candidate("random_plane_bounce", source_seed=11)
    assert rejected.candidate_id == "plane_low_170"
    assert rejected.admission == SurfaceAdmission()
    assert rejected.admission.admitted is False
    with pytest.raises(ValueError, match="not fully admitted"):
        sample_admitted_surface("random_plane_bounce", source_seed=11)


@pytest.mark.parametrize("rollout", (0, 2))
def test_f2f_compilation_persists_exact_hash_bound_admitted_sample(
    rollout: int,
) -> None:
    case = _case("F2f", rollout)
    scenario = compile_review_case(case)
    contract = scenario.sampled_surface_contract

    assert contract is not None
    assert contract.source_seed == scenario.rng_subseeds["physics"]
    assert set(contract.admission.values()) == {True}
    assert set(contract.admission_evidence_sha256) == set(contract.admission)
    stable_id = f"owned_arbitrary_surface__{contract.candidate_id}"
    fixture = next(surface for surface in scenario.surfaces if surface.name == stable_id)
    assert fixture.position_m == contract.position_m
    assert fixture.euler_rad == contract.euler_rad
    assert fixture.half_size_m == contract.half_size_m

    source_spec = prepare_review_case(case).to_dict()
    validate_source_evaluator_contract(
        evaluator_id="rigid_arbitrary_rebound_v1",
        task_variant=scenario.task_variant,
        source_spec=source_spec,
    )


@pytest.mark.parametrize("rollout", range(6))
def test_f2f_fixed_six_passes_strict_physics_and_saved_replay(
    rollout: int,
) -> None:
    result = SourceMujocoBackend().run(_case("F2f", rollout), render=False)
    penetration = result.physics_qc["penetration"]["metrics"]

    assert result.physics_qc["physics_qc_pass"] is True
    assert result.outcome["task_success"] is (rollout in {0, 1})
    assert result.outcome["intended_outcome_match"] is True
    assert result.outcome["saved_artifact_objective_replay_matches"] is True
    assert penetration["maximum_gripper_penetration_m"] <= 0.002
    assert penetration["maximum_task_surface_penetration_m"] <= 0.003
    assert result.runtime_audit["mutation_boundary_violations"] == 0
