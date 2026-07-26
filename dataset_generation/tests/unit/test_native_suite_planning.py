from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from dynamic_robot_dataset.backends import RigidScenario, RigidShape, ToolKind
from dynamic_robot_dataset.common.contract_v2 import CounterfactualRelation
from dynamic_robot_dataset.common.native_suite import (
    PHYSICS_SWEEP_VALUES,
    build_planned_counterfactual_family_records,
    plan_suite_cases,
)
from dynamic_robot_dataset.common.suites import expand_suite
from dynamic_robot_dataset.families.base import stable_hash


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "configs/families/native_acceptance_160.yaml"


@pytest.fixture(scope="module")
def cases():
    return expand_suite(SUITE)


@pytest.fixture(scope="module")
def planned(cases):
    return plan_suite_cases(cases)


def test_all_160_cases_plan_with_explicit_execution_tiers(cases, planned) -> None:
    assert len(planned) == len(cases) == 160
    assert Counter(item.execution_backend for item in planned) == {
        "native_mujoco": 128,
        "diagnostic_quarantine": 32,
    }
    assert [item.case.case_index for item in planned] == list(range(160))
    assert len({item.episode_uuid for item in planned}) == 160
    for item in planned:
        case, plan = item.case, item.episode_plan
        assert (plan.family, plan.subfamily) == (case.family, case.subfamily)
        assert plan.counterfactual_bundle_id == case.counterfactual_bundle_id
        assert plan.physics_counterfactual_family_id == case.physics_counterfactual_family_id
        assert plan.split_group_id == case.split_group_id
        assert plan.scene_seed == case.seed
        assert plan.options["suite_case"]["case_index"] == case.case_index
        if item.execution_backend == "native_mujoco":
            assert item.scenario_spec is not None
            assert item.scenario_spec.family.value == case.family
            assert item.scenario_spec.split_group_id == case.split_group_id
        else:
            assert item.scenario_spec is None
            assert plan.options["diagnostic_quarantine"] is True
            assert plan.options["production_eligible"] is False


def test_every_rigid_alias_is_typed_without_cross_family_coercion(planned) -> None:
    rigid = [item for item in planned if item.execution_backend == "native_mujoco"]
    assert len(rigid) == 128
    assert len({(item.case.family, item.case.subfamily) for item in rigid}) == 35
    scenarios_by_subfamily = {
        item.case.subfamily: item.scenario_spec.scenario for item in rigid
    }
    assert scenarios_by_subfamily["rolling_sliding_transition"] == RigidScenario.ROLLING_SLIDING_TRANSITION
    assert scenarios_by_subfamily["small_slope"] == RigidScenario.SMALL_SLOPE
    assert scenarios_by_subfamily["roll_off_edge"] == RigidScenario.PROJECTILE_ROLL_OFF_EDGE
    assert scenarios_by_subfamily["flight_to_table_bounce"] == RigidScenario.FLIGHT_TO_TABLE_BOUNCE


def test_shape_direction_and_tool_aliases_are_real_spec_fields(planned) -> None:
    by_subfamily = defaultdict(list)
    for item in planned:
        if item.scenario_spec is not None:
            by_subfamily[item.case.subfamily].append(item.scenario_spec)
    assert {spec.object.shape for spec in by_subfamily["sliding_puck"]} == {RigidShape.PUCK}
    assert {spec.object.shape for spec in by_subfamily["friction_sweep"]} == {RigidShape.PUCK}
    assert {spec.object.shape for spec in by_subfamily["sliding_cube"]} == {RigidShape.CUBE}
    assert all(spec.initial_state.linear_velocity_m_s[0] > 0 for spec in by_subfamily["straight_ball_left_to_right"])
    assert all(spec.initial_state.linear_velocity_m_s[0] < 0 for spec in by_subfamily["straight_ball_right_to_left"])
    assert {spec.tool.kind for spec in by_subfamily["shallow_tray"]} == {ToolKind.SHALLOW_TRAY}
    assert {spec.tool.kind for spec in by_subfamily["deep_tray"]} == {ToolKind.DEEP_TRAY}
    assert {spec.tool.kind for spec in by_subfamily["container_receive"]} == {ToolKind.SMALL_BIN}
    assert {spec.tool.kind for spec in by_subfamily["paddle_deflection"]} == {ToolKind.ANGLED_PADDLE}

    camera_latencies = {
        item.scenario_spec.camera_latency_s
        for item in planned
        if item.scenario_spec is not None
    }
    assert camera_latencies == {0.0, 1.0 / 30.0}


@pytest.mark.parametrize(
    ("subfamily", "field", "values"),
    [
        ("gravity_sweep", "gravity", PHYSICS_SWEEP_VALUES["gravity"]),
        ("friction_sweep", "surface_dynamic_friction", PHYSICS_SWEEP_VALUES["friction"]),
        (
            "restitution_sweep",
            "effective_restitution_target",
            PHYSICS_SWEEP_VALUES["restitution"],
        ),
    ],
)
def test_five_point_sweeps_apply_exact_physics_values(planned, subfamily, field, values) -> None:
    members = [item for item in planned if item.case.subfamily == subfamily]
    # Every sweep point replays one identical action trajectory.
    observed_by_variant = {}
    for item in members:
        raw = item.episode_plan.physics[field]["value"]
        observed_by_variant[item.case.physics_variant] = abs(raw[2]) if field == "gravity" else raw
    assert tuple(observed_by_variant[f"{subfamily.split('_')[0]}_{index}"] for index in range(5)) == values


def test_counterfactuals_are_symmetric_and_change_only_declared_fields(planned) -> None:
    action_groups = defaultdict(list)
    physics_groups = defaultdict(list)
    for item in planned:
        action_groups[item.case.counterfactual_bundle_id].append(item)
        if item.case.physics_counterfactual_family_id:
            physics_groups[item.case.physics_counterfactual_family_id].append(item)

    for siblings in action_groups.values():
        assert len({item.episode_plan.split_group_id for item in siblings}) == 1
        assert len({item.episode_plan.physics_hash for item in siblings}) == 1
        assert len({item.episode_plan.scene_seed for item in siblings}) == 1
        assert len({stable_hash(item.episode_plan.scene_parameters) for item in siblings}) == 1
        if siblings[0].scenario_spec is not None:
            assert len({stable_hash(item.scenario_spec.initial_state) for item in siblings}) == 1
            assert len({stable_hash(item.scenario_spec.cameras) for item in siblings}) == 1
            assert len({stable_hash(item.scenario_spec.object.rgba) for item in siblings}) == 1
        if len(siblings) > 1:
            assert len({item.episode_plan.action_hash for item in siblings}) > 1

    for siblings in physics_groups.values():
        fields = siblings[0].case.physics_intervention_fields
        assert fields
        assert len({item.episode_plan.split_group_id for item in siblings}) == 1
        assert len({item.episode_plan.action_hash for item in siblings}) == 1
        assert len({stable_hash(item.episode_plan.scene_parameters) for item in siblings}) == 1
        assert len({stable_hash(item.scenario_spec.initial_state) for item in siblings}) == 1
        assert len({stable_hash(item.scenario_spec.cameras) for item in siblings}) == 1
        nonintervened = [
            {
                key: value
                for key, value in item.episode_plan.physics.items()
                if key not in fields
            }
            for item in siblings
        ]
        assert len({stable_hash(value) for value in nonintervened}) == 1
        for field in fields:
            assert len({stable_hash(item.episode_plan.physics[field]) for item in siblings}) == 5


def test_planned_declarations_have_exact_expected_uuids(planned) -> None:
    declarations = build_planned_counterfactual_family_records(planned)
    assert Counter(value.relation for value in declarations) == {
        CounterfactualRelation.ACTION: 33,
        CounterfactualRelation.PHYSICS: 3,
    }
    action_groups = defaultdict(list)
    physics_groups = defaultdict(list)
    for item in planned:
        action_groups[item.case.counterfactual_bundle_id].append(item)
        if item.case.physics_counterfactual_family_id:
            physics_groups[item.case.physics_counterfactual_family_id].append(item)
    for declaration in declarations:
        declaration.validate()
        source = (
            action_groups[declaration.family_id]
            if declaration.relation == CounterfactualRelation.ACTION
            else physics_groups[declaration.family_id]
        )
        assert declaration.expected_episode_uuids == sorted(item.episode_uuid for item in source)
        assert declaration.expected_member_count == len(source)
        assert declaration.split_group_id == source[0].case.split_group_id
        if declaration.relation == CounterfactualRelation.ACTION:
            assert declaration.intervention_fields == ["action"]
        else:
            assert declaration.intervention_fields == list(
                source[0].case.physics_intervention_fields
            )
        assert all(len(value) == 64 for value in declaration.fixed_field_hashes.values())


def test_diagnostic_cases_keep_case_identity_and_first_class_no_op(planned) -> None:
    diagnostic = [item for item in planned if item.execution_backend == "diagnostic_quarantine"]
    assert Counter(item.case.family for item in diagnostic) == {
        "cloth": 12,
        "rope": 16,
        "legacy_proxy_quarantine": 4,
    }
    for item in diagnostic:
        plan = item.episode_plan
        assert plan.episode_uuid
        assert plan.counterfactual_bundle_id == item.case.counterfactual_bundle_id
        assert plan.physics_counterfactual_family_id is None
        assert plan.split_group_id == item.case.split_group_id
        if item.case.branch == "no_op":
            assert plan.intended_branch == "no_op"
            assert plan.branch_parameters.get("control_magnitude", 0.0) == 0.0
            assert not plan.branch_parameters.get("attachment_enabled", False)
            assert not plan.branch_parameters.get("endpoint_attachment", False)


def test_planning_is_bitwise_deterministic(cases, planned) -> None:
    repeated = plan_suite_cases(cases)
    assert [item.episode_plan.to_dict() for item in repeated] == [
        item.episode_plan.to_dict() for item in planned
    ]
    assert [item.scenario_spec.to_dict() if item.scenario_spec else None for item in repeated] == [
        item.scenario_spec.to_dict() if item.scenario_spec else None for item in planned
    ]
