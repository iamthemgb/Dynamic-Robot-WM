from __future__ import annotations

from pathlib import Path

import yaml

from dynamic_robot_dataset.common.schema import FAILURE_CODES


ROOT = Path(__file__).resolve().parents[2]


def _yaml(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def test_smoke_matrix_is_exactly_120_and_below_hard_cap() -> None:
    config = _yaml("configs/families/smoke_120.yaml")
    assert sum(config["counts"].values()) == 120
    assert config["max_episode_branches"] == 120
    assert config["max_episode_branches"] < 200


def test_splits_sum_to_one_and_are_group_aware() -> None:
    config = _yaml("configs/splits/default.yaml")
    assert abs(sum(config["fractions"].values()) - 1.0) < 1e-12
    groups = set(config["group_keys"])
    assert "counterfactual_bundle_id" in groups
    assert "physics_counterfactual_family_id" in groups
    assert "split_group_id" in groups


def test_physics_sweeps_are_five_point_counterfactuals() -> None:
    config = _yaml("configs/families/rigid_physics_sweeps.yaml")
    assert config["production_disabled"] is True
    assert all(len(values) == 5 for values in config["sweeps"].values())
    assert config["metadata_policy"]["do_not_invent_static_dynamic_split"] is True


def test_nonfluid_scope_is_explicit_in_wan_compatibility_schema() -> None:
    config = _yaml("configs/schema/wan_physics_fields_v1.yaml")
    assert "fluid_density_kg_m3" in config["fields"]
    assert "always masked" in config["notes"]["fluid_density_kg_m3"]


def test_failure_policy_forbids_empty_code_on_failure() -> None:
    config = _yaml("configs/schema/failure_codes_v1.yaml")
    forbidden = config["rules"]["task_success_false_forbids"]
    assert "none" in forbidden
    assert None in forbidden
    assert "" in forbidden
    assert set(config["codes"]) == set(FAILURE_CODES)
