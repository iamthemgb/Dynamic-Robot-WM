"""Bounded ``sampled_scale`` initial-state samplers for the scale fork."""

from dataclasses import replace

import pytest

from dynamic_robot_dataset.scenarios import (
    ScenarioBuildContext,
    load_scenario_definition,
)
from dynamic_robot_dataset.scenarios._rigid_shared import (
    SCALE_INITIAL_STATE_SAMPLER_VERSION,
    SCALE_SAMPLED_LEAVES,
    scale_parameter_ranges,
    scale_randomization_contract,
    scale_sampler_version,
)


def _context(**overrides) -> ScenarioBuildContext:
    value = dict(
        leaf_id="P0a",
        task_variant="lateral_freefall",
        embodiment="no_robot",
        branch_role="passive_observation",
        seed=123456789,
        physics_seed=987654321,
        tabletop_height_m=0.0,
        initial_state_mode="sampled_scale",
    )
    value.update(overrides)
    return ScenarioBuildContext(**value)


def _build(context: ScenarioBuildContext) -> dict:
    return load_scenario_definition(context.leaf_id).build(context)


def test_p0a_sampled_scale_contract_binds_applied_state() -> None:
    recipe = _build(_context())
    contract = recipe["initial_state_sampling_contract"]
    assert contract["schema_version"] == SCALE_INITIAL_STATE_SAMPLER_VERSION
    assert contract["source_seed"] == 123456789
    assert contract["applied_initial_position_m"] == list(
        recipe["object_initial_position_m"]
    )
    assert contract["applied_physical_target_position_m"] == []
    assert contract["declared_intervention"] == "none"


def test_sampled_scale_is_deterministic_and_seed_sensitive() -> None:
    first = _build(_context())
    second = _build(_context())
    different = _build(_context(seed=42))
    assert (
        first["object_initial_linear_velocity_m_s"]
        == second["object_initial_linear_velocity_m_s"]
    )
    assert (
        first["object_initial_linear_velocity_m_s"]
        != different["object_initial_linear_velocity_m_s"]
    )


def test_fixed_review_recipes_carry_no_scale_contract() -> None:
    recipe = _build(_context(initial_state_mode="fixed_review"))
    assert recipe.get("initial_state_sampling_contract") is None
    assert recipe["key_event_time_s"] == 0.3


def test_ballistic_scale_sampler_binds_controller_target() -> None:
    context = _context(
        leaf_id="F1a",
        task_variant="catch_retain",
        embodiment="franka_hand",
        branch_role="nominal_success",
    )
    recipe = _build(context)
    contract = recipe["initial_state_sampling_contract"]
    assert contract["declared_intervention"] == "none"
    assert contract["applied_controller_target_position_m"] == list(
        recipe["controller_target_position_m"]
    )
    # The sampled target must stay inside the declared IK envelope.
    ranges = contract["ranges"]
    target = recipe["physical_target_position_m"]
    assert ranges["target_x_m"][0] <= target[0] <= ranges["target_x_m"][1]
    assert ranges["target_y_m"][0] <= target[1] <= ranges["target_y_m"][1]


def test_negative_branch_records_declared_intervention() -> None:
    context = _context(
        leaf_id="F1a",
        task_variant="catch_retain",
        embodiment="franka_hand",
        branch_role="deterministic_negative_initial_state",
    )
    contract = _build(context)["initial_state_sampling_contract"]
    assert contract["declared_intervention"] == "initial_state_lateral_shift"


def test_f2c_scale_sampler_varies_lane_only() -> None:
    context = _context(
        leaf_id="F2c",
        task_variant="table_bounce",
        embodiment="franka_hand",
        branch_role="nominal_success",
    )
    recipe = _build(context)
    contract = recipe["initial_state_sampling_contract"]
    low, high = scale_parameter_ranges("F2c")["lane_speed_factor"]
    assert low <= contract["sampled_lane_speed_factor"] <= high
    assert contract["applied_initial_position_m"] == list(
        recipe["object_initial_position_m"]
    )
    assert recipe["object_initial_position_m"][1] == pytest.approx(
        contract["sampled_lane_y_m"]
    )


def test_f3b_scale_sampler_recomputes_start_from_speed() -> None:
    context = _context(
        leaf_id="F3b",
        task_variant="rolling_pickup",
        embodiment="franka_hand",
        branch_role="nominal_success",
    )
    recipe = _build(context)
    contract = recipe["initial_state_sampling_contract"]
    speed = contract["sampled_rolling_speed_m_s"]
    assert recipe["object_initial_linear_velocity_m_s"][0] == pytest.approx(-speed)
    start_x = recipe["object_initial_position_m"][0]
    target_x = recipe["physical_target_position_m"][0]
    assert start_x == pytest.approx(target_x + speed * recipe["key_event_time_s"])


@pytest.mark.parametrize(
    ("leaf_id", "task_variant"),
    (
        ("P0a", "nominal_freefall"),
        ("P0a", "lateral_freefall"),
        ("P0b", "ballistic_projectile"),
        ("P0b", "angled_projectile"),
        ("P0c", "table_bounce"),
        ("P0c", "wall_rebound"),
        ("P0d", "straight_roll"),
        ("P0d", "slope_roll"),
    ),
)
def test_passive_scale_events_stay_inside_review_window(
    leaf_id: str, task_variant: str
) -> None:
    for seed in (7, 123456789, 2**63 + 11):
        recipe = _build(
            _context(leaf_id=leaf_id, task_variant=task_variant, seed=seed)
        )
        contract = recipe["initial_state_sampling_contract"]
        assert contract is not None and contract["source_seed"] == seed
        assert 0.1 < recipe["key_event_time_s"] < recipe["duration_s"] - 0.3 + 1e-9


def test_scale_contract_helpers_cover_every_sampled_leaf() -> None:
    for leaf_id in SCALE_SAMPLED_LEAVES:
        contract = scale_randomization_contract(
            leaf_id,
            {
                "P0a": "nominal_freefall",
                "P0b": "ballistic_projectile",
                "P0c": "table_bounce",
                "P0d": "straight_roll",
            }.get(leaf_id),
        )
        assert contract["schema_version"] == scale_sampler_version(leaf_id)
        assert contract["rng_stream"] == "initial_state"
        assert contract["ranges"]


def test_leaf_sampler_versions_split_v3_and_v4() -> None:
    # The 2026-07-22 revision bumps only the leaves whose envelopes changed;
    # every other leaf must keep minting the v3 contract its planned blocks
    # were prepared under (shard execution re-prepares and compares hashes).
    for leaf_id in SCALE_SAMPLED_LEAVES:
        expected = "v4" if leaf_id in {"F1c", "F1d", "F2a"} else "v3"
        assert scale_sampler_version(leaf_id).endswith(expected)
