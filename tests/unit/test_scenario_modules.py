from __future__ import annotations

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
