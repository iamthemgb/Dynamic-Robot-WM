from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml

from dynamic_robot_dataset.families import list_families
from dynamic_robot_dataset.families.smoke import EXPECTED_SMOKE_COUNTS, plan_smoke, simulate_smoke


def test_smoke_matrix_is_exact_bounded_and_complete() -> None:
    plans = plan_smoke(seed=123)
    assert len(plans) == 120
    assert len(plans) < 200
    assert Counter(plan.family for plan in plans) == Counter(EXPECTED_SMOKE_COUNTS)
    assert set(list_families()) == set(EXPECTED_SMOKE_COUNTS)
    assert len({plan.episode_uuid for plan in plans}) == 120
    assert {plan.scene_style for plan in plans} == {
        "clean_franka_lab",
        "robocasa_kitchen_tabletop",
        "robotwin_cluttered_tabletop",
    }


def test_smoke_counterfactual_invariants_and_grouping() -> None:
    plans = plan_smoke(seed=5)
    by_physics_family: dict[str, list] = defaultdict(list)
    by_action_bundle: dict[str, list] = defaultdict(list)
    for plan in plans:
        if plan.physics_counterfactual_family_id is not None:
            by_physics_family[plan.physics_counterfactual_family_id].append(plan)
        by_action_bundle[plan.counterfactual_bundle_id].append(plan)

    physics_families = [values for values in by_physics_family.values() if len(values) > 1]
    assert physics_families
    for values in physics_families:
        assert len({value.action_hash for value in values}) == 1
        assert len({value.invariant_hash for value in values}) == 1
        assert len({value.scene_seed for value in values}) == 1
        assert len({value.split_group_id for value in values}) == 1

    action_bundles = [values for values in by_action_bundle.values() if len(values) > 1]
    assert action_bundles
    for values in action_bundles:
        assert len({value.physics_hash for value in values}) == 1
        assert len({value.scene_seed for value in values}) == 1
        assert len({value.split_group_id for value in values}) == 1


def test_smoke_objective_outcomes_and_tiers_are_honest() -> None:
    results = simulate_smoke(seed=123)
    for family in ("falling_catch", "rolling_interception", "projectile_rebound"):
        family_results = [result for result in results if result.plan.family == family]
        assert any(result.outcome.task_success for result in family_results)
        assert any(not result.outcome.task_success for result in family_results)
        assert all(result.outcome.label_status == "verified_objective" for result in family_results)
        assert all(result.dynamics_mode == "free_contact" for result in family_results)
        assert all(result.physics_qc["physics_qc_pass"] for result in family_results)
        assert all(
            result.outcome.failure_mode != "none"
            for result in family_results
            if not result.outcome.task_success
        )

    cloth = [result for result in results if result.plan.family == "cloth"]
    rope = [result for result in results if result.plan.family == "rope"]
    assert any(result.outcome.task_success for result in cloth)
    assert any(not result.outcome.task_success for result in cloth)
    assert any(result.outcome.task_success for result in rope)
    assert any(not result.outcome.task_success for result in rope)
    assert all(result.outcome.label_status == "unverified" for result in cloth + rope)
    assert all(result.dynamics_mode == "scripted_motion" for result in cloth + rope)
    assert all(result.release_tier == "scripted_motion" for result in cloth + rope)
    assisted_results = [
        result
        for result in cloth + rope
        if any(
            result.assistance.get(name, False)
            for name in (
                "assisted_grasp",
                "assisted_retention",
                "equality_constraint_active",
                "latch_active",
            )
        )
    ]
    assert assisted_results
    assert all(result.assistance.get("mechanisms") for result in assisted_results)
    assert all(
        mechanism.get("target_body_ids") or mechanism.get("target_element_ids")
        for result in assisted_results
        for mechanism in result.assistance["mechanisms"]
    )

    legacy = [result for result in results if result.plan.family == "legacy_proxy_quarantine"]
    assert {result.dynamics_mode for result in legacy} == {"assisted_contact", "scripted_motion"}
    assert all(result.outcome.failure_mode == "legacy_proxy_quarantined" for result in legacy)
    scripted_legacy = [result for result in legacy if result.dynamics_mode == "scripted_motion"]
    assert scripted_legacy
    assert all(
        all(state.get("dynamics.scripted_active") is True for state in result.states)
        for result in scripted_legacy
    )


def test_every_smoke_failure_has_a_registered_concrete_code() -> None:
    root = Path(__file__).resolve().parents[2]
    taxonomy = yaml.safe_load((root / "configs/schema/failure_codes_v1.yaml").read_text(encoding="utf-8"))
    registered = set(taxonomy["codes"]) | {taxonomy["success_code"]}
    used = {result.outcome.failure_mode for result in simulate_smoke(seed=123)}
    assert used <= registered
