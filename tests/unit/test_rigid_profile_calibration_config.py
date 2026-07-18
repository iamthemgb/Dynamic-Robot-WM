from __future__ import annotations

import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "configs/physics/rigid_600_1200_calibration_v1.yaml"
CAPABILITIES_PATH = ROOT / "configs/backends/capabilities_v1.yaml"


def _mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_rigid_profile_evidence_is_versioned_source_bound_and_fail_closed() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    capabilities = _mapping(CAPABILITIES_PATH)
    source_mujoco = next(
        backend
        for backend in capabilities["backends"]
        if backend["name"] == "source_mujoco"
    )

    assert evidence["schema_version"] == "dynamic-robot-rigid-profile-calibration/v1"
    assert evidence["contract_version"] == "dynamic-robot-dataset/v2"
    assert evidence["calibration_id"].endswith("_v8")
    assert evidence["current_runtime_profile"].endswith("-v10")
    assert evidence["evidence_status"] == "exploratory_unbound"
    assert evidence["release_state"] == "blocked"
    assert evidence["release_eligible"] is False
    assert evidence["source_binding"]["compiled_scenario_schema"].endswith("/v3")
    assert evidence["source_binding"]["backend_version"] == "0.10.0-review"
    assert evidence["source_binding"]["objective_evaluator_version"] == "1.3.0"
    assert evidence["source_binding"]["visibility_qc_schema"].endswith("/v3")
    assert evidence["source_binding"]["background_clearance_schema"].endswith(
        "/v3"
    )
    assert evidence["source_binding"]["rebound_acceptance_schema"].endswith(
        "/v1"
    )
    assert evidence["admission"]["admitted"] is False
    assert evidence["admission"]["preferred_profile"] is None
    assert evidence["admission"]["cheaper_profile_admitted"] is False
    assert evidence["source_binding"]["evidence_artifacts_hash_bound"] is False
    assert evidence["source_binding"]["source_trees_read_only"] is True
    assert (
        evidence["source_binding"]["external_dependency_manifest_sha256"]
        == source_mujoco["source_hashes"]["external_dependency_manifest"]
    )
    assert all(not profile["admitted"] for profile in evidence["profiles"].values())


def test_v7_table_rebound_repair_is_same_seed_bounded_and_not_admitted() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    thresholds = evidence["acceptance_thresholds"]
    repair = evidence["measurements"]["p0c_table_rebound_v7_repair"]

    assert repair["runtime_profile"].endswith("-v7")
    assert repair["backend_version"] == "0.7.0-review"
    assert repair["fixed_review_case_ids"] == [
        "P0c-review-00",
        "P0c-review-02",
        "P0c-review-04",
    ]
    assert repair["fixed_seeds_preserved"] is True
    assert repair["table_contact"]["solref"] == [0.0045, 0.42]
    assert repair["table_contact"]["normal_profile_fixed_across_fixed_cases"] is True
    assert repair["table_contact"]["contact_parameter_counterfactual"] == (
        "tangential_friction_only"
    )
    assert repair["online_and_persisted_replay_share_thresholds"] is True
    assert repair["admission_claimed"] is False

    cases = repair["fixed_compiled_rate_measurements"]
    assert [case["review_case_id"] for case in cases] == repair[
        "fixed_review_case_ids"
    ]
    assert [case["simulation_hz"] for case in cases] == [600, 600, 600]
    required_separation = max(
        thresholds["minimum_rebound_normal_separation_m"],
        thresholds["minimum_rebound_normal_separation_radius_fraction"]
        * 0.0245,
    )
    for case in cases:
        assert (
            case["measured_effective_restitution"]
            >= thresholds["minimum_rebound_effective_restitution"]
        )
        assert (
            case["measured_effective_restitution"]
            <= thresholds["maximum_measured_effective_restitution"]
        )
        assert (
            case["outgoing_normal_velocity_m_s"]
            >= thresholds["minimum_rebound_outgoing_normal_speed_m_s"]
        )
        assert case["maximum_normal_separation_m"] >= required_separation
        assert (
            case["separation_duration_s"]
            >= thresholds["minimum_rebound_separation_duration_s"]
        )
        assert (
            case["maximum_object_task_surface_penetration_m"]
            <= thresholds["maximum_object_task_surface_penetration_m"]
        )
        assert case["rebound_acceptance_pass"] is True

    comparisons = repair["non_rendered_timestep_halving_measurements"]
    assert [case["review_case_id"] for case in comparisons] == repair[
        "fixed_review_case_ids"
    ]
    for case in comparisons:
        assert case["rigid_600hz_physics_qc_pass"] is True
        assert case["rigid_1200hz_physics_qc_pass"] is True
        assert case["rigid_600hz_saved_artifact_replay_pass"] is True
        assert case["rigid_1200hz_saved_artifact_replay_pass"] is True
        assert case["outcome_match"] is True
        assert case["absolute_event_time_shift_s"] <= 1.0 / 30.0
        assert case["key_event_position_shift_m"] <= 0.01


def test_rigid_profiles_are_exact_600_and_1200_hz_candidates() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    profiles = evidence["profiles"]
    assert set(profiles) == {
        "rigid_600hz_candidate_v4",
        "rigid_1200hz_reference_v4",
    }

    for name, expected_hz in (
        ("rigid_600hz_candidate_v4", 600),
        ("rigid_1200hz_reference_v4", 1200),
    ):
        profile = profiles[name]
        assert profile["simulation_hz"] == expected_hz
        assert math.isclose(
            profile["timestep_s"], 1.0 / expected_hz, rel_tol=0.0, abs_tol=1e-15
        )
        assert expected_hz % profile["control_hz"] == 0
        assert profile["video_hz"] == 30
        contact = profile["robotiq_thick_pad_contact"]
        assert contact["condim"] == 3
        assert contact["friction"] == [0.9, 0.005, 0.0001]
        assert contact["actuator_force_range_n"] == [-0.16, 0.16]
        assert contact["closure_duration_s"] > 0.0


def test_wall_rebound_measurements_are_physical_without_invented_fields() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    threshold = evidence["acceptance_thresholds"][
        "maximum_measured_effective_restitution"
    ]
    trials = evidence["measurements"]["wall_rebound"]["trials"]
    by_profile = {trial["profile"]: trial for trial in trials}
    low = by_profile["rigid_600hz_candidate_v4"]
    reference = by_profile["rigid_1200hz_reference_v4"]

    assert low["incoming_normal_velocity_m_s"] < 0.0
    assert low["outgoing_normal_velocity_m_s"] > 0.0
    assert math.isclose(low["measured_effective_restitution"], 0.18777696222292306)
    assert math.isclose(reference["measured_effective_restitution"], 0.24216913860953423)
    assert all(
        trial["measured_effective_restitution"] <= threshold for trial in trials
    )
    comparison = evidence["measurements"]["wall_rebound"][
        "timestep_halving_comparison"
    ]
    assert comparison["outcome_match"] is True
    assert comparison["within_one_video_frame"] is True
    assert comparison["within_one_centimeter"] is True


def test_robotiq_halving_evidence_meets_local_thresholds_but_not_admission() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    threshold = evidence["acceptance_thresholds"]
    measurement = evidence["measurements"]["robotiq_free_contact_catch"]
    assert measurement["review_case_id"] == "F1a-review-01"
    assert measurement["fixed_master_seed"] == 20260717
    assert measurement["controller_profile"].endswith("-v4")
    trials = measurement["trials"]
    comparison = measurement["timestep_halving_comparison"]

    measured_event_shift = abs(
        trials[0]["first_bilateral_contact_time_s"]
        - trials[1]["first_bilateral_contact_time_s"]
    )
    assert math.isclose(
        comparison["absolute_event_time_shift_s"],
        measured_event_shift,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert (
        comparison["absolute_event_time_shift_s"]
        <= threshold["timestep_halving_maximum_event_shift_s"]
    )
    assert (
        comparison["key_event_position_shift_m"]
        <= threshold["timestep_halving_maximum_key_position_shift_m"]
    )
    assert (
        max(trial["maximum_object_gripper_penetration_m"] for trial in trials)
        <= threshold["maximum_object_gripper_penetration_m"]
    )
    assert comparison["within_one_video_frame"] is True
    assert comparison["within_one_centimeter"] is True
    assert evidence["admission"]["admitted"] is False
    assert "full_leaf_timestep_halving_matrix_pending" in evidence["admission"][
        "blockers"
    ]
    assert "record_hash_bound_human_review_approval" in evidence[
        "required_before_admission"
    ]


def test_reference_rate_exceptions_preserve_strict_thresholds() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    measurement = evidence["measurements"]["reference_rate_required_cases"]
    assert measurement["policy"].endswith("rigid_1200hz_reference_v4")
    cases = {value["review_case_id"]: value for value in measurement["cases"]}
    assert set(cases) == {
        "P0c-review-01",
        "F1a-review-04",
        "F1b-review-04",
    }
    for case_id in ("P0c-review-01", "F1a-review-04", "F1b-review-04"):
        case = cases[case_id]
        assert case["rigid_600hz_observation"] > case["threshold"]
        assert case["rigid_1200hz_observation"] <= case["threshold"]


def test_persisted_artifact_repair_keeps_the_failed_seed() -> None:
    evidence = _mapping(EVIDENCE_PATH)["measurements"]["persisted_artifact_repairs"]
    assert evidence["review_case_id"] == "P0c-review-03"
    assert evidence["fixed_seed_preserved"] is True
    assert evidence["failed_profile"].endswith("-v3")
    assert evidence["repaired_profile"].endswith("-v4")
    assert evidence["failure_evidence"]["main_frozen_transition_fraction"] == 1.0
    repaired = evidence["repaired_evidence"]
    assert repaired["main_frozen_transition_fraction"] == 0.0
    assert repaired["finite_difference_velocity_rmse_m_s"] < repaired[
        "finite_difference_velocity_tolerance_m_s"
    ]
    assert repaired["strict_persisted_artifact_qc_pass"] is True


def test_wall_camera_repair_keeps_fixed_seeds_and_full_object_in_frame() -> None:
    evidence = _mapping(EVIDENCE_PATH)["measurements"]["camera_framing_repairs"]
    assert evidence["fixed_review_case_ids"] == [
        "P0c-review-01",
        "P0c-review-03",
        "P0c-review-05",
    ]
    assert evidence["fixed_seeds_preserved"] is True
    assert evidence["failed_profile"].endswith("-v4")
    assert evidence["repaired_profile"].endswith("-v5")
    assert evidence["repair"]["physics_or_seed_changed"] is False
    projection = evidence["full_trajectory_projection"]
    assert projection["includes_object_radius"] is True
    assert projection["evaluated_at_every_canonical_30hz_timestamp"] is True
    required_margin = projection["required_minimum_edge_margin_px"]
    assert required_margin == 8.0
    assert all(
        case["minimum_edge_margin_px"] >= required_margin
        for case in projection["cases"]
    )
    rendered = evidence["rendered_confirmation"]
    assert rendered["strict_persisted_artifact_qc_pass"] is True
    assert rendered["object_visible_in_all_main_event_strip_panels"] is True
    assert rendered["formal_human_approval_recorded"] is False


def test_v8_camera_repairs_preserve_fixed_seeds_and_bind_rendered_evidence() -> None:
    evidence = _mapping(EVIDENCE_PATH)["measurements"][
        "camera_framing_repairs_v8"
    ]
    assert evidence["runtime_profile"].endswith("-v8")
    assert evidence["backend_version"] == "0.8.0-review"
    assert evidence["fixed_seeds_preserved"] is True
    assert evidence["physics_or_initial_state_changed"] is False
    assert evidence["evidence_artifacts_hash_bound"] is False
    assert evidence["formal_human_approval_recorded"] is False
    assert evidence["admission_claimed"] is False

    projectile = evidence["p0b_main_projectile_envelope"]
    assert projectile["fixed_review_case_ids"] == [
        f"P0b-review-{index:02d}" for index in range(6)
    ]
    assert projectile["failure_evidence"]["review_case_id"] == "P0b-review-02"
    assert (
        projectile["failure_evidence"][
            "failed_minimum_projected_sphere_margin_px"
        ]
        < 0.0
    )
    assert projectile["repair"] == {
        "camera_position_m_relative_to_support": [1.2, -1.15, 1.15],
        "look_at_target_m_relative_to_support": [0.05, -0.1, 0.85],
        "vertical_fov_deg": 58.0,
        "required_minimum_edge_margin_px": 8.0,
    }
    projectile_runtime = projectile["rendered_runtime_measurements"]
    assert projectile_runtime["visibility_qc_schema_at_measurement"].endswith(
        "/v2"
    )
    assert projectile_runtime["main_all_canonical_checkpoints_visible"] is True
    assert projectile_runtime["both_views_all_canonical_checkpoints_visible"] is True
    assert projectile_runtime["both_views_target_visible_frame_fraction"] == 1.0
    assert projectile_runtime["physics_qc_pass"] is True
    assert projectile_runtime["intended_outcome_match"] is True
    assert projectile_runtime["background_clearance_pass"] is True
    required_margin = projectile["repair"]["required_minimum_edge_margin_px"]
    for case in projectile_runtime["cases"]:
        assert case["main_minimum_projected_sphere_margin_px"] >= required_margin
        assert case["main_minimum_segmentation_bbox_margin_px"] >= required_margin
        assert case["main_minimum_object_area_px"] >= 64
        assert case["main_key_event_object_area_px"] >= 64

    wall = evidence["p0c_wall_secondary_normal_view"]
    assert wall["fixed_review_case_ids"] == [
        "P0c-review-01",
        "P0c-review-03",
        "P0c-review-05",
    ]
    assert wall["table_rebound_secondary_serialization_unchanged"] is True
    assert wall["repair"] == {
        "camera_position_m_relative_to_R1_support": [-0.05, -1.7, 0.76],
        "look_at_target_m_relative_to_R1_support": [-0.05, 0.0, 0.76],
        "vertical_fov_deg": 58.0,
        "required_minimum_edge_margin_px": 8.0,
        "applied_motion_kind": "passive_wall_rebound",
    }
    wall_runtime = wall["rendered_runtime_measurements"]
    assert wall_runtime["visibility_qc_schema_at_measurement"].endswith("/v2")
    assert wall_runtime["physics_qc_pass"] is True
    assert wall_runtime["background_clearance_pass"] is True
    assert wall_runtime["strict_visibility_qc_pass"] is True
    required_margin = wall["repair"]["required_minimum_edge_margin_px"]
    for case in wall_runtime["cases"]:
        assert (
            case["secondary_minimum_projected_sphere_margin_px"]
            >= required_margin
        )
        assert (
            case["secondary_minimum_segmentation_bbox_margin_px"]
            >= required_margin
        )
        assert case["secondary_minimum_object_area_px"] >= 100
        assert case["secondary_key_event_object_area_px"] >= 100
        assert case["secondary_final_object_area_px"] >= 100
        assert case["event_to_post_0p3_pixel_displacement_px"] >= 32
        assert case["event_to_final_pixel_displacement_px"] >= 75


def test_v9_reaching_repair_records_defect_ctrl_only_plan_and_no_admission() -> None:
    evidence = _mapping(EVIDENCE_PATH)["measurements"][
        "reaching_controller_repair_v9"
    ]
    assert evidence["runtime_profile"].endswith("-v9")
    assert evidence["backend_version"] == "0.9.0-review"
    assert evidence["fixed_seeds_preserved"] is True
    assert evidence["object_physics_or_initial_state_changed"] is False

    defect = evidence["defect_evidence"]
    assert defect["runtime_profile_at_measurement"].endswith("-v8")
    assert defect["all_seven_arm_actuator_command_ranges_rad"] == 0.0
    assert defect["robot_initialized_at_intercept_pose"] is True

    repair = evidence["repair"]
    assert repair["ready_waypoint_hover_above_intercept_m"] == 0.045
    assert repair["initialize_once_at_ready_pose"] is True
    assert repair["minimum_jerk_reach_through_data_ctrl_only"] is True
    assert repair["reach_arrival_before_ballistic_event_s"] == 0.055
    assert repair["no_robot_or_object_state_rewrites_after_initialization"] is True
    assert repair["no_latch_or_weld_assistance"] is True

    checks = evidence["new_strict_qc_checks"]
    assert checks["arm_command_travel_present_minimum_rad"] == 0.01
    assert checks["arm_arrived_at_commanded_intercept_maximum_m"] == 0.025

    measured = evidence["non_rendered_fixed_case_measurements"]
    assert measured["leaves"] == ["F1a", "F1b", "F1c", "F1d"]
    assert measured["all_24_fixed_cases_strict_physics_qc_pass"] is True
    assert (
        measured["all_intended_outcome_labels_matched_without_seed_changes"]
        is True
    )
    assert measured["arm_command_travel_range_rad"][0] >= 0.01
    assert measured["reach_arrival_distance_range_m"][1] <= 0.025
    assert measured["maximum_gripper_penetration_m"] <= 0.002
    assert measured["maximum_measured_joint_acceleration_rad_s2"] <= 80.0

    divergence = evidence["robotiq_nominal_event_sampling_divergence"]
    assert (
        divergence["semantic_outcome_qc_and_replay_agree_at_both_rates"] is True
    )
    shifts = divergence["measured_first_bilateral_position_shift_mm"]
    added = divergence["added_reference_rate_classes"]
    for leaf, shift in shifts.items():
        entry = f"{leaf}/robotiq_2f85_thick_pad/nominal_success"
        assert (shift > 10.0) == (entry in added)
    assert evidence["admission_claimed"] is False
    assert evidence["formal_human_approval_recorded"] is False


def test_f1_r1_free_space_repair_records_failed_profile_and_required_reruns() -> None:
    evidence = _mapping(EVIDENCE_PATH)["measurements"][
        "f1_r1_free_space_scene_repair"
    ]
    assert evidence["fixed_review_case_ids"] == [
        "F1a-review-00",
        "F1a-review-02",
        "F1d-review-03",
    ]
    assert evidence["fixed_seeds_preserved"] is True
    assert evidence["failed_profile"].endswith("-v5")
    assert evidence["repaired_profile"].endswith("-v6")
    assert evidence["failure_evidence"]["failed_objects_crossed_visible_table_volume"] is True
    contract = evidence["repair_contract"]
    assert contract["local_f1_task_height_m"] == 0.0
    assert contract["robot_base_position_m"] == [0.0, 0.0, 0.0]
    assert contract["remove_all_named_elements_with_prefix"] == "robot_table_"
    assert contract["source_compiled_scenario_schema"].endswith("/v2")
    assert evidence["admission_claimed"] is False
    assert "full_F1_fixed_six_matrix" in evidence["required_reruns"]


def test_operator_guide_names_the_canonical_lifecycle_and_scale_gates() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/unified_generator_operator_guide.md").read_text(
        encoding="utf-8"
    )

    assert "docs/unified_generator_operator_guide.md" in readme
    for command in ("plan-run", "run-shard", "finalize-run", "generate", "review-suite"):
        assert f"dynamic-robot-dataset {command}" in guide
    for gate in ("100-episode pilot", "10-unique-hour gate", "100-unique-hour gate"):
        assert gate in guide
    assert "remain fail-closed historical gate" in guide


def test_v10_f2a_f3b_unblock_records_measured_design_evidence_and_no_admission() -> None:
    evidence = _mapping(EVIDENCE_PATH)["measurements"]["f2a_f3b_unblock_v10"]
    assert evidence["runtime_profile"].endswith("-v10")
    assert evidence["backend_version"] == "0.10.0-review"
    assert evidence["fixed_seeds_preserved"] is True
    assert evidence["f1_and_p0_constructions_unchanged"] is True

    design = evidence["measured_design_evidence"]
    assert (
        max(
            design["palm_up_catch_posture_infeasible_at_pickup_height"][
                "franka_wrist_link_z_below_floor_m"
            ]
        )
        < 0.0
    )
    assert (
        design["robotiq_knuckles_bottom_out_on_runway"]["shoulder_slam_rad_s2"]
        > 80.0
    )
    assert (
        design["wrist_servo_transients_during_transport"][
            "lift_0p15_default_window_rad_s2"
        ]
        > 80.0
    )
    assert (
        design["missed_ball_struck_robot_pedestal"]["unclassified_contact_count"]
        > 0
    )
    assert (
        design["rolling_evidence_restricted_to_first_sustained_surface_segment"]
        is True
    )

    from dynamic_robot_dataset.backends.source_mujoco import RIGID_REVIEW_PROFILE

    f3b = evidence["f3b_construction"]
    assert tuple(f3b["runway_half_xy_m"]) == (
        RIGID_REVIEW_PROFILE.rolling_pickup_runway_half_xy_m
    )
    assert f3b["runway_height_m"] == RIGID_REVIEW_PROFILE.rolling_pickup_runway_height_m
    assert f3b["rolling_speed_m_s"] == RIGID_REVIEW_PROFILE.rolling_pickup_speed_m_s
    assert f3b["event_time_s"] == RIGID_REVIEW_PROFILE.rolling_pickup_event_time_s
    assert f3b["pickup_lift_height_m"] == RIGID_REVIEW_PROFILE.pickup_lift_height_m

    f2a = evidence["f2a_construction"]
    assert f2a["deflection_gripper_never_closes"] is True
    assert f2a["deflection_outcome_requires_measured_contact_and_redirect"] is True

    measured = evidence["non_rendered_fixed_case_measurements"]
    assert measured["leaves"] == ["F2a", "F3b"]
    assert measured["all_12_fixed_cases_strict_physics_qc_pass"] is True
    assert (
        measured["all_intended_outcome_labels_matched_without_seed_changes"]
        is True
    )
    assert measured["reach_arrival_distance_range_m"][1] <= 0.025
    assert measured["maximum_gripper_penetration_m"] <= 0.002
    assert measured["maximum_measured_joint_acceleration_rad_s2"] <= 80.0

    halving = evidence["timestep_halving_calibration"]
    assert (
        halving["semantic_outcomes_agree_at_both_rates_for_all_12_cases"] is True
    )
    added = set(halving["added_reference_rate_classes"])
    assert added == {
        "F2a/robotiq_2f85_thick_pad/nominal_success",
        "F2a/franka_hand/deterministic_negative_controller_timing",
    }
    assert all(
        shift > 10.0
        for shift in halving["added_class_measured_shift_mm"].values()
    )
    assert halving["added_classes_pass_strict_qc_at_1200hz"] is True
    divergence = halving[
        "comparison_rate_divergence_without_contact_event_or_600hz_defect"
    ]
    assert divergence["compiled_rate_hz"] == 600
    assert divergence["compiled_rate_strict_qc_pass"] is True
    assert divergence["semantic_outcome_match_at_1200hz"] is True

    from dynamic_robot_dataset.backends.source_mujoco import (
        RIGID_REVIEW_PROFILE as PROFILE,
    )

    assert added <= set(PROFILE.reference_rate_required_case_classes)
    assert evidence["admission_claimed"] is False
    assert evidence["formal_human_approval_recorded"] is False
