from __future__ import annotations

import pytest

from dynamic_robot_dataset.backends.source_mujoco import (
    RIGID_REVIEW_PROFILE,
    SourceMujocoBackend,
    compile_review_case,
)
from dynamic_robot_dataset.backends.source_mujoco.rigid_breadth import (
    SurfaceAdmission,
    sample_admitted_surface,
    sample_surface_candidate,
)
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.common.source_evaluators import (
    validate_source_evaluator_contract,
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
    assert scenario.simulation_hz == 1200
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


def test_f2b_fixed_robotiq_collision_is_invalid_not_an_online_success() -> None:
    result = SourceMujocoBackend().run(_case("F2b", 1), render=False)
    penetration = result.physics_qc["penetration"]["metrics"]
    evidence = result.physics_qc["task_evidence"]

    assert result.outcome["task_success"] is False
    assert result.outcome["actual_outcome"] == "invalid"
    assert result.outcome["saved_artifact_objective_replay_matches"] is False
    assert result.physics_qc["physics_qc_pass"] is False
    assert penetration["maximum_gripper_penetration_m"] > 0.002
    assert evidence["rolling_or_sliding_slip_within_limit"] is True
    assert evidence["friction_deceleration_consistent"] is True


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


def test_f2f_sampler_is_deterministic_but_unadmitted_until_calibration() -> None:
    seed = 0xD15EA5E
    first = sample_surface_candidate("random_plane_bounce", source_seed=seed)
    second = sample_surface_candidate("random_plane_bounce", source_seed=seed)

    assert first == second
    assert first.admission == SurfaceAdmission()
    assert first.admission.admitted is False
    with pytest.raises(ValueError, match="not fully admitted"):
        sample_admitted_surface("random_plane_bounce", source_seed=seed)


@pytest.mark.parametrize("rollout", (0, 2))
def test_f2f_compilation_persists_exact_sample_but_evaluator_fails_closed(
    rollout: int,
) -> None:
    scenario = compile_review_case(_case("F2f", rollout))
    contract = scenario.sampled_surface_contract

    assert contract is not None
    assert contract.source_seed == scenario.rng_subseeds["physics"]
    assert set(contract.admission.values()) == {False}
    stable_id = f"owned_arbitrary_surface__{contract.candidate_id}"
    fixture = next(surface for surface in scenario.surfaces if surface.name == stable_id)
    assert fixture.position_m == contract.position_m
    assert fixture.euler_rad == contract.euler_rad
    assert fixture.half_size_m == contract.half_size_m

    source_spec = {
        "duration_s": scenario.duration_s,
        "physics": {
            "simulation_hz": scenario.simulation_hz,
            "key_event_time_s": scenario.key_event_time_s,
            "object_radius_m": scenario.object_radius_m,
            "rebound_acceptance": (
                RIGID_REVIEW_PROFILE.rebound_acceptance().to_dict()
            ),
            "grasp_retention": RIGID_REVIEW_PROFILE.grasp_retention().to_dict(),
            "sampled_surface_contract": contract.to_dict(),
        },
    }
    with pytest.raises(ValueError, match="is not admitted"):
        validate_source_evaluator_contract(
            evaluator_id="rigid_arbitrary_rebound_v1",
            task_variant=scenario.task_variant,
            source_spec=source_spec,
        )
