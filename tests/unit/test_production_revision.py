from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from dynamic_robot_dataset.cli import (
    _annotate_plans_with_counterfactual_hashes,
    _feature_unit,
    _native_plans_from_adapter,
    _planned_episode_declarations,
)
from dynamic_robot_dataset.common.calibration import (
    bounce_oracle,
    calibrate_physics_catalog,
    free_fall_oracle,
    rolling_slip_oracle,
)
from dynamic_robot_dataset.common.splits import SplitAssigner
from dynamic_robot_dataset.common.statistics import cramers_v
from dynamic_robot_dataset.common.visual_qc import NATIVE_VISUAL_THRESHOLDS
from dynamic_robot_dataset.common.suites import expand_suite, load_pilot_plan
from dynamic_robot_dataset.common.readiness import load_gate_config
from dynamic_robot_dataset.common.qc import (
    _ballistic_evidence_mode,
    _event_aware_free_fall_mask,
)
from dynamic_robot_dataset.families.base import normalize_branch
from dynamic_robot_dataset.families import get_family
from dynamic_robot_dataset.families.deformable.rope.native_contract import (
    grasp_equality_xml,
    validate_rope_grasp_target,
)


ROOT = Path(__file__).resolve().parents[2]


def test_encoded_gravity_mask_excludes_contact_between_video_frames() -> None:
    rows = [
        {
            "timestamp": index / 30.0,
            "free_fall": True,
            "object.motion_mode": "free_flight",
            "contact.role": "none",
        }
        for index in range(6)
    ]
    mask = _event_aware_free_fall_mask(
        rows,
        [{"timestamp": 2.5 / 30.0, "event_type": "contact_begin"}],
        "free_fall",
    )

    assert mask == [True, False, False, False, False, False]


def test_ballistic_qc_applicability_is_task_contract_specific() -> None:
    record = type("Record", (), {})()
    record.family = "falling_catch"
    record.extras = {"native_scenario_spec": {"scenario": "centered_drop"}}
    assert _ballistic_evidence_mode(record) == "precontact"
    record.family = "projectile_rebound"
    record.extras = {"native_scenario_spec": {"scenario": "ramp_launch"}}
    assert _ballistic_evidence_mode(record) == "post_release"
    record.family = "rolling_interception"
    record.extras = {"native_scenario_spec": {"scenario": "straight_roll"}}
    assert _ballistic_evidence_mode(record) == "not_required"
    record.extras = {}
    assert _ballistic_evidence_mode(record) == "not_required"


def test_post_release_gravity_mask_accepts_free_flight_after_surface_contact() -> None:
    rows = [
        {
            "timestamp": index / 30.0,
            "free_fall": index >= 4,
            "object.motion_mode": "free_flight" if index >= 4 else "rolling",
            "contact.role": "none" if index >= 4 else "support",
        }
        for index in range(8)
    ]
    mask = _event_aware_free_fall_mask(
        rows,
        [{"timestamp": 1.5 / 30.0, "event_type": "surface_contact"}],
        "free_fall",
        precontact_only=False,
    )
    assert mask[-2:] == [True, True]


def test_camera_config_and_enforced_visual_thresholds_are_identical() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/cameras/two_view_832x480.yaml").read_text(
            encoding="utf-8"
        )
    )["qc"]
    assert NATIVE_VISUAL_THRESHOLDS == {
        "minimum_target_visible_frame_fraction": config[
            "require_target_visible_fraction"
        ],
        "minimum_bbox_margin_px": config["minimum_crop_margin_px"],
        "minimum_key_event_object_area_px": config["minimum_target_area_px"],
        "maximum_underexposed_fraction": config[
            "reject_black_void_fraction_above"
        ],
        "maximum_overexposed_fraction": config[
            "reject_overexposed_fraction_above"
        ],
    }


def _record(index: int, group: int, *, outcome: str, style: str) -> dict:
    return {
        "episode_uuid": f"00000000-0000-4000-8000-{index:012d}",
        "episode_index": index,
        "counterfactual_bundle_id": f"bundle-{group}",
        "physics_counterfactual_family_id": f"physics-{group}",
        "split_group_id": f"scene-{group}",
        "scene_seed": group,
        "family": "rolling_interception",
        "subfamily": "straight_ball_left_to_right",
        "actual_outcome": outcome,
        "tool_type": "flat_paddle",
        "randomization": {"background_style": style, "object_asset_id": "ball-v1"},
        "extras": {"physics_bin": "id-mid"},
    }


def test_native_acceptance_suite_is_exact_and_deterministic() -> None:
    path = ROOT / "configs/families/native_acceptance_160.yaml"
    first = expand_suite(path)
    second = expand_suite(path)
    assert [value.to_dict() for value in first] == [value.to_dict() for value in second]
    assert len(first) == 160
    assert Counter(value.category_id for value in first) == {
        "falling_catch_retention": 40,
        "rolling_sliding_transitions": 48,
        "projectile_rebound_deflection": 40,
        "cloth_tiering": 12,
        "rope_repairs": 16,
        "negative_controls": 4,
    }
    assert {value.scene_style for value in first} == {
        "clean_franka_lab",
        "robocasa_kitchen_tabletop",
        "robotwin_cluttered_tabletop",
    }
    # Branch and physics interventions share the physical-scene seed. Action
    # bundles differ only by branch; physics families differ only by the named
    # sweep field. All relations share one leakage-prevention split group.
    request = [
        value
        for value in first
        if value.category_id == "rolling_sliding_transitions"
        and value.subfamily == "friction_sweep"
    ]
    assert len({value.seed for value in request}) == 1
    assert len({value.split_group_id for value in request}) == 1
    assert all(
        value.physics_intervention_fields
        == (
            "surface_dynamic_friction",
            "mujoco_surface_friction",
            "mujoco_object_friction",
        )
        for value in request
    )
    for physics_variant in {value.physics_variant for value in request}:
        siblings = [value for value in request if value.physics_variant == physics_variant]
        assert len({value.counterfactual_bundle_id for value in siblings}) == 1
    for branch in {value.branch for value in request}:
        siblings = [value for value in request if value.branch == branch]
        assert len({value.physics_counterfactual_family_id for value in siblings}) == 1


def test_general_native_generation_rebinds_real_actions_and_sweeps() -> None:
    config = {
        "family": "projectile_rebound",
        "subfamily": "gravity_sweep",
        "variant": "native_test",
        "robot_model": "franka_emika_panda",
        "tool_type": "flat_paddle",
        "num_bundles": 1,
        "branches": ["success_seeking"],
        "views": ["main", "secondary"],
        "seed": 4,
        "randomization_level": "R1",
        "scene_style": "clean_franka_lab",
        "sim_hz": 240,
        "control_hz": 60,
        "video_hz": 30,
        "duration_s": None,
        "physics_sweep": "gravity",
    }
    analytical = get_family("projectile_rebound").plan(config)
    native = _native_plans_from_adapter(analytical, config)
    assert len(native) == 5
    assert len({plan.physics_counterfactual_family_id for plan in native}) == 1
    assert None not in {plan.physics_counterfactual_family_id for plan in native}
    assert len({plan.action_hash for plan in native}) == 1
    assert [abs(plan.physics["gravity"]["value"][2]) for plan in native] == [
        4.905,
        7.3575,
        9.81,
        12.2625,
        14.715,
    ]
    for plan in native:
        assert plan.options["backend"] == "native_mujoco"
        assert plan.options["native_scenario_spec"]["branch"] == "success_seeking"
    declarations = _planned_episode_declarations(native)
    annotated = _annotate_plans_with_counterfactual_hashes(native, declarations)
    assert len(declarations) == 1
    assert declarations[0].relation.value == "physics"
    assert declarations[0].expected_member_count == 5
    assert declarations[0].intervention_fields == ["gravity"]
    assert set(declarations[0].expected_member_plan_hashes) == {
        plan.episode_uuid for plan in native
    }
    assert all(
        plan.options["planned_counterfactual_fixed_hashes"]["physics"]
        == declarations[0].fixed_field_hashes
        for plan in annotated
    )


def test_pilot_hours_are_unique_episode_allocations_and_not_submitted() -> None:
    ten = load_pilot_plan(ROOT / "configs/pilots/pilot_10h.yaml")
    hundred = load_pilot_plan(ROOT / "configs/pilots/pilot_100h.yaml")
    assert ten["submit"] is False and hundred["submit"] is False
    assert sum(value["unique_hours"] for value in ten["allocations"]) == 10.0
    assert sum(value["unique_hours"] for value in hundred["allocations"]) == 100.0
    deformable = next(value for value in ten["allocations"] if value["id"] == "free_contact_deformable")
    assert deformable["unique_hours"] == 1.5
    assert ten["policy"]["reallocate_blocked_family_quota"] is False


def test_scaling_gates_require_sparse_strata_and_prior_evidence() -> None:
    ten = load_gate_config(ROOT / "configs/release_gates/10h.yaml")
    hundred = load_gate_config(ROOT / "configs/release_gates/100h.yaml")
    three_hundred = load_gate_config(ROOT / "configs/release_gates/300h_blocked.yaml")
    thousand = load_gate_config(ROOT / "configs/release_gates/1000h_blocked.yaml")
    assert ten["require_no_sparse_strata"] is True
    assert hundred["prerequisite_gate"] == {"gate_id": "native_10h_v1", "passed": True}
    assert hundred["model_evaluation"]["required"] is True
    assert three_hundred["hard_blocked"] is True
    assert thousand["hard_blocked"] is True


def test_provisional_ranges_do_not_become_calibrated_by_loading_yaml() -> None:
    report = calibrate_physics_catalog(ROOT / "configs/physics/rigid_ranges_v1.yaml")
    assert not report.passed
    assert not report.release_eligible
    assert set(report.missing_oracles) == {
        "free_fall",
        "bounce",
        "sliding_deceleration",
        "rolling_slip",
        "contact_stability",
    }
    named_but_empty = {
        name: {}
        for name in (
            "free_fall",
            "bounce",
            "sliding_deceleration",
            "rolling_slip",
            "contact_stability",
        )
    }
    empty_report = calibrate_physics_catalog(
        ROOT / "configs/physics/rigid_ranges_v1.yaml",
        observations=named_but_empty,
    )
    assert not empty_report.passed
    assert all(
        not check.passed
        for check in empty_report.checks
        if check.name.startswith("oracle.")
    )


def test_physics_calibration_oracles() -> None:
    times = [0.0, 0.1, 0.2, 0.3]
    positions = [1.0 - 0.5 * 9.81 * time * time for time in times]
    assert free_fall_oracle(times, positions)["gravity_acceleration_rmse_m_s2"] < 1e-10
    assert bounce_oracle(-2.0, 1.0)["measured_effective_restitution"] == pytest.approx(0.5)
    rolling = rolling_slip_oracle([1.0, 0.5], [10.0, 5.0], 0.1)
    assert rolling["maximum_slip_speed_m_s"] == pytest.approx(0.0)


def test_canonical_feature_units_distinguish_joints_pose_and_categories() -> None:
    assert _feature_unit("robot.joint_position") == "rad"
    assert _feature_unit("action.command.joint_position") == "rad"
    assert _feature_unit("robot.joint_velocity") == "rad/s"
    assert _feature_unit("action.command.tool_target_position") == "m"
    assert _feature_unit("action.command.tool_target_rpy") == "rad"
    assert _feature_unit("robot.tool_quaternion_wxyz") == "1 (WXYZ)"
    assert _feature_unit("object.active_surface") == "categorical"
    assert _feature_unit("action.command.enabled") == "bool"


def test_grouped_split_is_80_10_10_and_reports_sparse_strata() -> None:
    records = []
    for group in range(30):
        outcome = "success" if group % 2 == 0 else "near_miss"
        style = ("clean_franka_lab", "robocasa_kitchen_tabletop", "robotwin_cluttered_tabletop")[group % 3]
        records.extend(_record(group * 2 + offset, group, outcome=outcome, style=style) for offset in range(2))
    assignments, diagnostics = SplitAssigner(seed=3).assign_with_diagnostics(records)
    counts = Counter(value.split for value in assignments)
    assert counts == {"train": 48, "validation": 6, "test": 6}
    assert diagnostics.requested_fractions == {"train": 0.8, "validation": 0.1, "test": 0.1}
    # Six five-group strata cannot each populate validation and test while
    # also retaining the exact global 80/10/10 count. The assignment keeps
    # connected groups intact and reports the resulting release blocker.
    assert diagnostics.sparse_strata
    assert not diagnostics.passed


def test_visual_outcome_association_reports_effect_size() -> None:
    independent = [
        {"style": style, "outcome": outcome}
        for style in ("a", "b")
        for outcome in ("success", "failure")
        for _ in range(20)
    ]
    confounded = ([{"style": "a", "outcome": "success"}] * 30) + ([{"style": "b", "outcome": "failure"}] * 30)
    assert cramers_v(independent, "style", "outcome") == pytest.approx(0.0)
    assert cramers_v(confounded, "style", "outcome") > 0.9


def test_no_op_is_first_class_and_rope_constraint_binds_requested_segment() -> None:
    assert normalize_branch("no-op") == "no_op"
    xml = grasp_equality_xml("RB_6")
    assert 'body1="RB_6"' in xml and "RB_first" not in xml
    validate_rope_grasp_target("RB_6", "RB_6")
    with pytest.raises(ValueError, match="RB_6.*RB_first"):
        validate_rope_grasp_target("RB_6", "RB_first")
