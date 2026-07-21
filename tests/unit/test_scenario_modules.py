from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from dynamic_robot_dataset.backends.source_mujoco.compiler import compile_review_case
from dynamic_robot_dataset.common.corpus_registry import EXPECTED_CORPUS_LEAVES
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.scenarios import (
    ScenarioBlockedError,
    ScenarioBuildContext,
    list_scenario_definitions,
    load_scenario_definition,
    scenario_source_hashes,
)
from dynamic_robot_dataset.scenarios.__main__ import main as scenarios_main


_IMPLEMENTED = {
    "P0a",
    "P0b",
    "P0c",
    "P0d",
    "F1a",
    "F1b",
    "F1c",
    "F1d",
    "F2a",
    "F2b",
    "F2c",
    "F2d",
    "F2e",
    "F2f",
    "F3b",
}


def test_canonical_scenario_folder_contains_one_distinct_module_per_leaf() -> None:
    definitions = list_scenario_definitions()
    assert len(definitions) == 20
    assert {item.leaf_id for item in definitions} == EXPECTED_CORPUS_LEAVES
    assert len({item.module_path for item in definitions}) == 20
    assert {item.leaf_id for item in definitions if item.implemented} == _IMPLEMENTED
    for definition in definitions:
        module_name, attribute = definition.module_path.split(":")
        assert attribute == "SCENARIO"
        assert module_name.startswith("dynamic_robot_dataset.scenarios.")
        assert definition.module.controller_plan.actuator_only is True


def test_blocked_scenario_modules_expose_contract_and_fail_closed() -> None:
    for definition in list_scenario_definitions():
        if definition.implemented:
            continue
        with pytest.raises(ScenarioBlockedError, match=definition.leaf_id):
            definition.build(
                ScenarioBuildContext(
                    leaf_id=definition.leaf_id,
                    task_variant=definition.variants[0],
                    embodiment=definition.embodiments[0],
                    branch_role="nominal_success",
                    seed=1,
                    physics_seed=2,
                    tabletop_height_m=0.0,
                )
            )


def test_compiler_routes_each_implemented_leaf_through_its_definition() -> None:
    plan = build_review_suite_plan()
    first_r0 = {
        case.corpus_leaf_id: case
        for case in plan.cases
        if case.rollout_index == 0 and case.corpus_leaf_id in _IMPLEMENTED
    }
    assert set(first_r0) == _IMPLEMENTED
    for leaf_id, case in first_r0.items():
        definition = load_scenario_definition(leaf_id)
        compiled = compile_review_case(case)
        assert (compiled.corpus_leaf_id, compiled.family, compiled.subfamily) == (
            definition.leaf_id,
            definition.family,
            definition.subfamily,
        )
        assert compiled.evaluator == definition.evaluator_id


def test_compiler_no_longer_owns_the_leaf_recipe_dispatcher() -> None:
    compiler = (
        Path(__file__).resolve().parents[2]
        / "src/dynamic_robot_dataset/backends/source_mujoco/compiler.py"
    ).read_text(encoding="utf-8")
    assert "def _recipe_payload(" not in compiler
    assert "definition.build(" in compiler


def test_scenario_source_hashes_bind_leaf_and_shared_implementation() -> None:
    f2b = scenario_source_hashes("F2b")
    f2c = scenario_source_hashes("F2c")
    assert set(f2b) == {
        "scenario_module_py",
        "scenario_shared_recipe_py",
        "scenario_registry_py",
        "scenario_types_py",
    }
    assert f2b["scenario_module_py"] != f2c["scenario_module_py"]
    assert f2b["scenario_shared_recipe_py"] == f2c["scenario_shared_recipe_py"]
    assert all(len(value) == 64 for value in f2b.values())


def test_scenario_discovery_cli_lists_and_describes_modules(capsys) -> None:
    assert scenarios_main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "F2b\tramp_launch\tsource_mujoco\tyes" in listed
    assert "F3c\twater_current_pickup\tsource_genesis_fluid\tno" in listed

    assert scenarios_main(["show", "F2b"]) == 0
    shown = capsys.readouterr().out
    assert '"module_path": "dynamic_robot_dataset.scenarios.f2b_ramp_launch:SCENARIO"' in shown
    assert '"settled_aim_correction": false' in shown


@pytest.mark.parametrize("leaf_id", ("F1d", "F2a"))
def test_projectile_sampler_is_deterministic_bounded_and_ballistic(
    leaf_id: str,
) -> None:
    definition = load_scenario_definition(leaf_id)
    public = definition.module.randomization_contract
    assert public is not None
    assert public["rng_stream"] == "initial_state"
    assert public["outcome_conditioned_resampling"] is False
    ranges = public["ranges"]
    seen = set()
    for seed in range(32):
        context = ScenarioBuildContext(
            leaf_id=leaf_id,
            task_variant=definition.variants[0],
            embodiment="franka_hand",
            branch_role="nominal_success",
            seed=seed,
            physics_seed=10_000 + seed,
            tabletop_height_m=0.0,
            initial_state_mode="sampled_preview",
        )
        first = definition.build(context)
        second = definition.build(context)
        assert first == second
        contract = first["initial_state_sampling_contract"]
        assert contract["source_seed"] == seed
        for name, bounds in ranges.items():
            sampled_name = {
                "target_x_m": "sampled_target_position_m",
                "target_y_m": "sampled_target_position_m",
                "incoming_azimuth_deg": "sampled_incoming_azimuth_deg",
                "horizontal_distance_m": "sampled_horizontal_distance_m",
                "flight_time_s": "sampled_flight_time_s",
                "initial_vertical_velocity_m_s": (
                    "sampled_initial_vertical_velocity_m_s"
                ),
                "initial_spin_z_rad_s": (
                    "sampled_base_initial_angular_velocity_rad_s"
                ),
            }[name]
            sampled = contract[sampled_name]
            if name == "target_x_m":
                sampled = sampled[0]
            elif name == "target_y_m":
                sampled = sampled[1]
            elif name == "initial_spin_z_rad_s":
                sampled = sampled[2]
            assert bounds[0] <= sampled <= bounds[1]
        position = first["object_initial_position_m"]
        velocity = first["object_initial_linear_velocity_m_s"]
        target = first["physical_target_position_m"]
        event_time = first["ballistic_event_time_s"]
        arrival = (
            position[0] + velocity[0] * event_time,
            position[1] + velocity[1] * event_time,
            position[2]
            + velocity[2] * event_time
            - 0.5 * 9.81 * event_time**2,
        )
        assert arrival == pytest.approx(target, abs=1e-12)
        assert velocity[2] - 9.81 * event_time < 0.0
        seen.add((position, velocity, target))
    assert len(seen) == 32


def test_projectile_sampling_is_independent_of_controller_branch() -> None:
    definition = load_scenario_definition("F1d")

    def build(branch_role: str):
        return definition.build(
            ScenarioBuildContext(
                leaf_id="F1d",
                task_variant="mild_projectile_catch",
                embodiment="franka_hand",
                branch_role=branch_role,
                seed=123,
                physics_seed=456,
                tabletop_height_m=0.0,
                initial_state_mode="sampled_preview",
            )
        )

    nominal = build("nominal_success")
    controller_negative = build("deterministic_negative_controller_timing")
    assert (
        nominal["object_initial_position_m"]
        == controller_negative["object_initial_position_m"]
    )
    assert (
        nominal["object_initial_linear_velocity_m_s"]
        == controller_negative["object_initial_linear_velocity_m_s"]
    )
    assert nominal["controller_target_position_m"] != (
        controller_negative["controller_target_position_m"]
    )
    assert controller_negative["initial_state_sampling_contract"][
        "declared_intervention"
    ] == "controller_target_offset"


def test_f2d_negative_controller_offset_is_declarative_and_leaf_scoped() -> None:
    definition = load_scenario_definition("F2d")
    offset = definition.module.controller_plan.negative_controller_offset_m

    assert offset == pytest.approx((0.0, -0.10, 0.0))
    assert definition.to_dict()["controller_plan"][
        "negative_controller_offset_m"
    ] == pytest.approx(offset)
    for other in list_scenario_definitions():
        if other.leaf_id == "F2d":
            continue
        assert other.module.controller_plan.negative_controller_offset_m is None
        assert "negative_controller_offset_m" not in other.to_dict()[
            "controller_plan"
        ]

    cases = {
        case.rollout_index: compile_review_case(case)
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == "F2d"
    }
    panda_delta = tuple(
        negative - nominal
        for negative, nominal in zip(
            cases[4].controller_target_position_m,
            cases[0].controller_target_position_m,
            strict=True,
        )
    )
    robotiq_delta = tuple(
        negative - nominal
        for negative, nominal in zip(
            cases[5].controller_target_position_m,
            cases[1].controller_target_position_m,
            strict=True,
        )
    )
    assert panda_delta == pytest.approx(offset)
    assert robotiq_delta == pytest.approx(offset)
    assert cases[4].simulation_hz == 600
    assert cases[5].simulation_hz == 1200


def test_projectile_sample_cli_emits_reproducible_training_ineligible_specs(
    capsys,
) -> None:
    argv = ["sample", "F2a", "--count", "3", "--seed", "91"]
    assert scenarios_main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert scenarios_main(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["training_eligible"] is False
    assert first["sample_count"] == 3
    assert len({row["initial_state_seed"] for row in first["samples"]}) == 3
    assert all(
        math.isfinite(row["object_initial_speed_m_s"])
        for row in first["samples"]
    )


def test_compiler_persists_and_binds_sampled_projectile_contract() -> None:
    case = next(
        case
        for case in build_review_suite_plan().cases
        if case.case_id == "F2a-review-00"
    ).to_dict()
    case["initial_state_mode"] = "sampled_preview"
    scenario = compile_review_case(case)
    contract = scenario.initial_state_sampling_contract
    assert scenario.initial_state_mode == "sampled_preview"
    assert contract is not None
    assert tuple(contract["applied_initial_position_m"]) == (
        scenario.object_initial_position_m
    )
    assert tuple(contract["applied_initial_linear_velocity_m_s"]) == (
        scenario.object_initial_linear_velocity_m_s
    )
    assert contract["source_seed"] == scenario.rng_subseeds["initial_state"]

    tampered = dict(contract)
    tampered["applied_initial_position_m"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="differs from compiled state"):
        replace(scenario, initial_state_sampling_contract=tampered).validate()
