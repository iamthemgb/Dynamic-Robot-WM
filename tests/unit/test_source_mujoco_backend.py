from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from dynamic_robot_dataset.backends.source_mujoco import (
    IMPLEMENTED_REVIEW_VARIANTS,
    PINNED_SOURCE_MANIFEST_SHA256,
    RIGID_REVIEW_PROFILE,
    SourceMujocoBackend,
    SourceMujocoUnsupported,
    compile_review_case,
    prepare_review_case,
    resolve_source_dependency,
    timestep_comparison_failures,
)
from dynamic_robot_dataset.backends.source_mujoco.compiler import (
    SOURCE_MUJOCO_BACKEND_VERSION,
    SOURCE_MUJOCO_COMPILED_SCHEMA,
)
from dynamic_robot_dataset.backends.source_mujoco.provenance import (
    SourceDependencyError,
    referenced_asset_manifest,
)
from dynamic_robot_dataset.backends.source_mujoco.backend import (
    _effective_restitution_within_limit,
    _energy_drift_within_limit,
    _restitution_evidence,
)
from dynamic_robot_dataset.common.review import event_strip_frame_indices
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.common.synchronization import (
    fixed_duration_frame_timestamps,
)


def _case(leaf: str, rollout: int = 0):
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == leaf and case.rollout_index == rollout
    )


def test_calibrated_source_manifest_and_rigid_profile_are_exact() -> None:
    dependency = resolve_source_dependency()
    assert dependency.manifest_sha256 == PINNED_SOURCE_MANIFEST_SHA256
    assert PINNED_SOURCE_MANIFEST_SHA256 == (
        "823463e7095fac9a0819cae2688d75df80a7a38ce6c93b1e72e1323fe469ae99"
    )
    assert RIGID_REVIEW_PROFILE.simulation_hz == 600
    assert RIGID_REVIEW_PROFILE.profile_id.endswith("-v6")
    assert RIGID_REVIEW_PROFILE.wall_solref == (0.012, 0.7)
    assert RIGID_REVIEW_PROFILE.robotiq_pad_friction == (0.9, 0.005, 0.0001)
    assert RIGID_REVIEW_PROFILE.robotiq_tendon_target == 115.0
    assert SOURCE_MUJOCO_BACKEND_VERSION == "0.6.0-review"
    assert SOURCE_MUJOCO_COMPILED_SCHEMA.endswith("/v2")


def test_f1_r0_r1_share_owned_physics_base_and_planned_robot_state() -> None:
    clean_case = _case("F1a", rollout=0)
    r1_case = replace(
        clean_case,
        scene_profile="robocasa_kitchen",
        randomization_level="R1",
        requires_real_robocasa=True,
    )
    clean = compile_review_case(clean_case)
    randomized = compile_review_case(r1_case)

    invariant_fields = (
        "object_initial_position_m",
        "object_initial_linear_velocity_m_s",
        "object_initial_angular_velocity_rad_s",
        "physical_target_position_m",
        "controller_target_position_m",
        "controller_transport_position_m",
        "robot_base_position_m",
        "robot_base_euler_rad",
    )
    for name in invariant_fields:
        assert getattr(randomized, name) == getattr(clean, name)
    assert clean.robot_base_position_m == (0.0, 0.0, 0.0)
    assert randomized.robot_base_position_m == (0.0, 0.0, 0.0)

    clean_spec = prepare_review_case(clean_case)
    randomized_spec = prepare_review_case(r1_case)
    assert clean_spec.initial_state["robot_base_position_m"] == (0.0, 0.0, 0.0)
    assert randomized_spec.initial_state["robot_base_position_m"] == (0.0, 0.0, 0.0)
    assert clean_spec.initial_state["robot_base_quaternion_wxyz"] == pytest.approx(
        (1.0, 0.0, 0.0, 0.0)
    )
    assert randomized_spec.initial_state["robot_base_pose_sha256"] == (
        clean_spec.initial_state["robot_base_pose_sha256"]
    )
    assert randomized_spec.initial_state["robot_initial_joint_qpos"] == (
        clean_spec.initial_state["robot_initial_joint_qpos"]
    )
    assert randomized_spec.initial_state["robot_initial_joint_qpos_sha256"] == (
        clean_spec.initial_state["robot_initial_joint_qpos_sha256"]
    )
    assert clean_spec.physics["robot_base_pose_source"] == (
        "owned_compiled_scenario/v2"
    )


def test_passive_source_spec_has_no_invented_robot_base_or_joint_state() -> None:
    scenario = compile_review_case(_case("P0d", rollout=2))
    spec = prepare_review_case(_case("P0d", rollout=2))
    assert scenario.robot_base_position_m is None
    assert scenario.robot_base_euler_rad is None
    assert spec.initial_state["robot_base_position_m"] is None
    assert spec.initial_state["robot_base_euler_rad"] is None
    assert spec.initial_state["robot_base_quaternion_wxyz"] is None
    assert spec.initial_state["robot_base_pose_sha256"] is None
    assert spec.initial_state["robot_initial_joint_qpos"] is None
    assert spec.initial_state["robot_initial_joint_qpos_sha256"] is None


def test_p0_supports_use_fixed_six_sweep_envelopes_and_normalized_r1_thickness() -> None:
    expected_xy = {
        "P0a": (0.28, 0.23),
        "P0b": (0.83, 0.28),
        "P0c": (0.58, 0.16),
    }
    for leaf, half_xy in expected_xy.items():
        scenario = compile_review_case(_case(leaf, rollout=2))
        horizontal = next(
            surface
            for surface in scenario.surfaces
            if surface.role in {"floor", "table"}
        )
        assert horizontal.half_size_m[:2] == half_xy
        assert horizontal.half_size_m[2] == 0.02
        assert horizontal.position_m[2] + horizontal.half_size_m[2] == 0.74


@pytest.mark.integration
@pytest.mark.parametrize(
    ("leaf", "rollout", "expected_outcome"),
    (("F1a", 0, "success"), ("F1a", 2, "miss"), ("F1d", 3, "miss")),
)
def test_repaired_f1_fixed_cases_render_physical_complete_outcomes(
    leaf: str,
    rollout: int,
    expected_outcome: str,
) -> None:
    result = SourceMujocoBackend().run(_case(leaf, rollout=rollout), render=True)
    visibility = result.visibility_qc

    assert result.physics_qc["physics_qc_pass"] is True
    assert result.outcome["actual_outcome"] == expected_outcome
    assert result.outcome["intended_outcome_match"] is True
    assert result.background_clearance["clearance_pass"] is True
    assert result.runtime_audit["compiled_robot_base_position_m"] == (0.0, 0.0, 0.0)
    assert result.runtime_audit["initialized_robot_joint_qpos_sha256"]
    assert visibility["target_visible_frame_fraction"] == 1.0
    assert visibility["initial_state_visible_in_any_view"] is True
    assert visibility["apex_visible_in_any_view"] is True
    assert visibility["key_event_visible_in_any_view"] is True
    assert visibility["final_state_visible_in_any_view"] is True
    if rollout:
        removed = result.backend_provenance["removed_task_volume_background_names"]
        assert "robot_table_top" in removed
        assert not any(
            str(row["stable_id"]).startswith("procedural:robot_table_")
            for row in result.background_clearance["background_rows"]
        )


def test_signed_energy_and_restitution_metrics_fail_closed() -> None:
    assert _energy_drift_within_limit(0.049)
    assert not _energy_drift_within_limit(-0.051)
    assert not _energy_drift_within_limit(math.inf)
    assert _effective_restitution_within_limit(0.0)
    assert _effective_restitution_within_limit(1.05)
    assert not _effective_restitution_within_limit(-0.01)
    assert not _effective_restitution_within_limit(1.051)


def test_initial_persistent_support_contact_is_zero_restitution_not_missing() -> None:
    rows = [
        {
            "timestamp": 0.0,
            "contact.count": 1,
            "object.position": [0.0, 0.0, 0.025],
            "object.linear_velocity": [0.5, 0.0, 0.0],
        },
        {
            "timestamp": 0.5,
            "contact.count": 1,
            "object.position": [0.2, 0.0, 0.025],
            "object.linear_velocity": [0.3, 0.0, 0.0],
        },
    ]
    contacts = [
        {
            "timestamp": 0.0,
            "contact_category": "task_surface",
            "normal_world": [0.0, 0.0, 1.0],
        }
    ]

    evidence = _restitution_evidence(rows, contacts)

    assert evidence["applicable"] is True
    assert evidence["effective_restitution"] == 0.0
    assert evidence["no_unexplained_contact_energy_gain"] is True


def test_all_executable_r0_recipes_have_distinct_complete_event_strips() -> None:
    for case in build_review_suite_plan().cases:
        if case.rollout_index != 0:
            continue
        if case.task_variant not in IMPLEMENTED_REVIEW_VARIANTS.get(
            case.corpus_leaf_id, ()
        ):
            continue
        scenario = compile_review_case(case)
        timestamps = fixed_duration_frame_timestamps(
            scenario.duration_s, scenario.video_hz
        )
        indices = event_strip_frame_indices(timestamps, scenario.key_event_time_s)
        assert len(indices) == 5
        assert len(set(indices.values())) == 5
        assert timestamps[-1] - scenario.key_event_time_s >= 0.3 - 1e-12


def test_wall_rebound_recipe_completes_an_airborne_arc_at_the_wall() -> None:
    scenario = compile_review_case(_case("P0c", rollout=1))
    assert scenario.task_variant == "wall_rebound"
    assert scenario.object_initial_linear_velocity_m_s[2] > 0.0
    apex_time = scenario.object_initial_linear_velocity_m_s[2] / abs(
        scenario.gravity_m_s2[2]
    )
    assert math.isclose(apex_time, scenario.key_event_time_s / 2.0)
    assert scenario.surfaces[-1].role == "wall"


def test_wall_rebound_main_camera_stays_on_visible_incoming_side() -> None:
    spec = prepare_review_case(_case("P0c", rollout=3))
    main = next(camera for camera in spec.cameras if camera.name == "main")
    assert main.pose.position_m == pytest.approx((-0.90, 0.95, 1.79))
    assert main.pose.position_m[0] < spec.initial_state["object_position_m"][0] + 0.1
    assert main.fovy_deg == 64.0


def test_owned_passive_secondary_overview_presets_cover_outcome_classes() -> None:
    projectile = prepare_review_case(_case("P0b", rollout=2))
    projectile_secondary = next(
        camera for camera in projectile.cameras if camera.name == "secondary"
    )
    assert projectile_secondary.pose.position_m == pytest.approx((0.20, -1.80, 1.94))
    assert projectile_secondary.fovy_deg == 62.0

    rolling = prepare_review_case(_case("P0d", rollout=2))
    rolling_secondary = next(
        camera for camera in rolling.cameras if camera.name == "secondary"
    )
    assert rolling_secondary.pose.position_m == pytest.approx((0.10, -1.80, 1.64))
    assert rolling_secondary.fovy_deg == 58.0


@pytest.mark.integration
def test_owned_passive_secondary_overviews_contain_complete_fixed_trajectories() -> None:
    backend = SourceMujocoBackend()
    minimum_edge_margin_px = 8.0
    fixed_cases = (
        *(("P0b", rollout) for rollout in range(6)),
        *(("P0d", rollout) for rollout in range(6)),
    )
    for leaf, rollout in fixed_cases:
        result = backend.run(_case(leaf, rollout=rollout), render=False)
        calibration = result.camera_calibrations["secondary"]
        focal_px = float(calibration.intrinsic_matrix[0])
        for row in result.frame_rows:
            pixel_x, pixel_y, depth_m = calibration.project_world(
                row["object.position"]
            )
            projected_radius_px = (
                focal_px * result.scenario.object_radius_m / depth_m
            )
            assert pixel_x - projected_radius_px >= minimum_edge_margin_px
            assert pixel_x + projected_radius_px <= (
                calibration.width - minimum_edge_margin_px
            )
            assert pixel_y - projected_radius_px >= minimum_edge_margin_px
            assert pixel_y + projected_radius_px <= (
                calibration.height - minimum_edge_margin_px
            )


@pytest.mark.integration
def test_wall_rebound_main_camera_contains_complete_fixed_trajectories() -> None:
    backend = SourceMujocoBackend()
    minimum_edge_margin_px = 8.0
    for rollout in (1, 3, 5):
        result = backend.run(_case("P0c", rollout=rollout), render=False)
        calibration = result.camera_calibrations["main"]
        focal_px = float(calibration.intrinsic_matrix[0])
        for row in result.frame_rows:
            pixel_x, pixel_y, depth_m = calibration.project_world(
                row["object.position"]
            )
            projected_radius_px = (
                focal_px * result.scenario.object_radius_m / depth_m
            )
            assert pixel_x - projected_radius_px >= minimum_edge_margin_px
            assert pixel_x + projected_radius_px <= (
                calibration.width - minimum_edge_margin_px
            )
            assert pixel_y - projected_radius_px >= minimum_edge_margin_px
            assert pixel_y + projected_radius_px <= (
                calibration.height - minimum_edge_margin_px
            )


def test_negative_initial_state_does_not_retarget_velocity_to_the_gripper() -> None:
    nominal = compile_review_case(_case("F1c", rollout=1))
    negative = compile_review_case(_case("F1c", rollout=3))
    nominal_arrival_y = (
        nominal.object_initial_position_m[1]
        + nominal.object_initial_linear_velocity_m_s[1]
        * nominal.ballistic_event_time_s
    )
    negative_arrival_y = (
        negative.object_initial_position_m[1]
        + negative.object_initial_linear_velocity_m_s[1]
        * negative.ballistic_event_time_s
    )
    assert math.isclose(
        nominal_arrival_y, nominal.physical_target_position_m[1], abs_tol=1e-12
    )
    assert abs(
        negative_arrival_y - negative.controller_target_position_m[1]
    ) >= 0.08


def test_p0_fixed_variation_profiles_change_physics_not_only_appearance() -> None:
    p0a = [compile_review_case(_case("P0a", rollout=index)) for index in range(6)]
    assert [scenario.passive_variation_profile for scenario in p0a] == [
        "nominal",
        "lower_initial_speed",
        "higher_initial_speed",
        "initial_spin",
        "lower_admitted_contact_parameter",
        "higher_admitted_contact_parameter",
    ]
    relative_height = lambda scenario: scenario.object_initial_position_m[2] - (
        0.74 if scenario.requires_real_robocasa else 0.0
    )
    assert relative_height(p0a[1]) < relative_height(p0a[0])
    assert relative_height(p0a[2]) > relative_height(p0a[0])
    assert p0a[3].object_initial_angular_velocity_rad_s[2] == 8.0
    assert p0a[4].surfaces[0].friction[0] < p0a[0].surfaces[0].friction[0]
    assert p0a[5].surfaces[0].friction[0] > p0a[0].surfaces[0].friction[0]

    slower_wall = compile_review_case(_case("P0c", rollout=1))
    assert slower_wall.object_initial_linear_velocity_m_s[0] < 1.10
    assert slower_wall.key_event_time_s > 0.686
    faster_bounce = compile_review_case(_case("P0c", rollout=2))
    assert faster_bounce.object_initial_linear_velocity_m_s[0] > 0.55
    assert slower_wall.simulation_hz == 1200
    assert faster_bounce.simulation_hz == 1200

    rolling_spin = compile_review_case(_case("P0d", rollout=3))
    nominal_spin = rolling_spin.object_initial_linear_velocity_m_s[0] / (
        rolling_spin.object_radius_m
    )
    assert rolling_spin.object_initial_angular_velocity_rad_s == pytest.approx(
        (0.0, 1.10 * nominal_spin, 0.0)
    )


def test_reference_rate_is_selected_only_for_failed_fixed_case_classes() -> None:
    assert compile_review_case(_case("F1a", rollout=4)).simulation_hz == 1200
    assert compile_review_case(_case("F1b", rollout=4)).simulation_hz == 1200
    assert compile_review_case(_case("F1c", rollout=4)).simulation_hz == 600
    assert compile_review_case(_case("F1a", rollout=5)).simulation_hz == 600


def test_unaccepted_f2c_and_all_f3_paths_fail_closed() -> None:
    with pytest.raises(SourceMujocoUnsupported):
        compile_review_case(_case("F2c"))
    value = _case("F1a").to_dict()
    value.update(corpus_leaf_id="F3a", task_variant="oscillating_handoff")
    with pytest.raises(SourceMujocoUnsupported):
        compile_review_case(value)


def test_referenced_assets_cannot_escape_verified_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    allowed_mesh = allowed / "mesh.obj"
    outside_mesh = outside / "mesh.obj"
    allowed_mesh.write_text("allowed", encoding="utf-8")
    outside_mesh.write_text("outside", encoding="utf-8")
    xml = f"<mujoco><asset><mesh file='{allowed_mesh}'/></asset></mujoco>"
    assert str(allowed_mesh.resolve()) in referenced_asset_manifest(
        xml, allowed_roots=(allowed,)
    )
    escaped = f"<mujoco><asset><mesh file='{outside_mesh}'/></asset></mujoco>"
    with pytest.raises(SourceDependencyError, match="escapes"):
        referenced_asset_manifest(escaped, allowed_roots=(allowed,))


def test_prepare_review_case_emits_valid_actual_actuator_endpoints() -> None:
    spec = prepare_review_case(_case("F1a"))
    spec.validate()
    assert spec.source_hashes["external_dependency_manifest"] == (
        PINNED_SOURCE_MANIFEST_SHA256
    )
    assert len(spec.embodiment.action_names) == 8
    assert spec.counterfactual.sibling_index == 0
    assert spec.counterfactual.physics_family_id is None
    assert spec.actuator_phases
    for phase in spec.actuator_phases:
        assert tuple(phase.commands) == spec.embodiment.action_names
        assert all(math.isfinite(value) for value in phase.commands.values())


@pytest.mark.integration
def test_rendered_p0a_uses_one_exact_two_view_persisted_trajectory() -> None:
    backend = SourceMujocoBackend()
    result = backend.run(_case("P0a"), render=True)
    expected = round(result.scenario.duration_s * result.scenario.video_hz)
    assert expected == 24
    assert len(result.frame_rows) == expected
    assert set(result.frames_by_camera) == {"main", "secondary"}
    assert all(len(frames) == expected for frames in result.frames_by_camera.values())
    assert all(frame.shape == (480, 832, 3) for frame in result.frames_by_camera["main"])
    assert result.outcome["task_success"] is True
    assert result.physics_qc["physics_qc_pass"] is True
    assert all(row["action.mode"] == "no_actuators/v1" for row in result.frame_rows)


@pytest.mark.integration
def test_fixed_f1a_is_strict_free_contact_success_with_exact_ctrl_echo() -> None:
    result = SourceMujocoBackend().run(_case("F1a"), render=False)
    assert result.outcome["task_success"] is True
    assert result.outcome["intended_outcome_match"] is True
    assert result.physics_qc["physics_qc_pass"] is True
    evidence = result.physics_qc["task_evidence"]
    assert evidence["sustained_opposing_bilateral_contacts"] is True
    assert evidence["stable_object_to_grasp_transform"] is True
    assert (
        result.physics_qc["penetration"]["metrics"][
            "maximum_gripper_penetration_m"
        ]
        <= 0.002
    )
    assert result.runtime_audit["action_ctrl_echo_exact"] is True
    assert all(
        row["action.actuator_command"]
        == row["simulator.applied_actuator_ctrl"]
        for row in result.high_rate_rows
    )
    assert all("point_world_m" in row for row in result.contact_rows)


@pytest.mark.integration
def test_fixed_robotiq_catch_passes_the_600_1200_timestep_gate() -> None:
    backend = SourceMujocoBackend()
    scenario = backend.compile_case(_case("F1a", rollout=1))
    observations = []
    for simulation_hz in (
        RIGID_REVIEW_PROFILE.simulation_hz,
        RIGID_REVIEW_PROFILE.comparison_simulation_hz,
    ):
        result = backend.run(
            replace(scenario, simulation_hz=simulation_hz), render=False
        )
        bilateral = [
            row for row in result.high_rate_rows if row["contact.bilateral"]
        ]
        assert bilateral
        assert result.outcome["actual_outcome"] == "success"
        assert result.physics_qc["physics_qc_pass"] is True
        event = bilateral[0]
        observations.append(
            {
                "outcome": result.outcome["actual_outcome"],
                "key_event_time_s": event["timestamp"],
                "key_event_position_m": event["object.position"],
            }
        )
    assert timestep_comparison_failures(*observations) == ()


@pytest.mark.integration
def test_uphill_roll_reversal_uses_signed_slope_physics_not_endpoint_speed() -> None:
    result = SourceMujocoBackend().run(_case("P0d", rollout=1), render=False)
    evidence = result.physics_qc["task_evidence"]
    contacted_vx = [
        row["object.linear_velocity"][0]
        for row in result.high_rate_rows
        if row["object.motion_mode"] == "surface_contact"
    ]
    assert contacted_vx[0] > 0.0 and contacted_vx[-1] < 0.0
    assert evidence["rolling_or_sliding_slip_within_limit"] is True
    assert evidence["friction_deceleration_consistent"] is True
    assert math.isclose(
        evidence["measured_tangent_acceleration_m_s2"],
        evidence["expected_rolling_tangent_acceleration_m_s2"],
        abs_tol=0.03,
    )
    assert evidence["maximum_mechanical_energy_gain_fraction"] <= 0.05
    assert result.physics_qc["physics_qc_pass"] is True


@pytest.mark.integration
def test_fixed_negative_is_label_consistent_without_success_evidence() -> None:
    value = _case("F1a").to_dict()
    value.update(
        branch_role="deterministic_negative_controller_timing",
        intended_outcome="failure",
    )
    result = SourceMujocoBackend().run(value, render=False)
    assert result.outcome["task_success"] is False
    assert result.outcome["intended_outcome_match"] is True
    assert result.outcome["measured_outcome_replay_matches"] is True
    evidence = result.physics_qc["task_evidence"]
    assert evidence["measured_failure_matches_persisted_label"] is True
    assert evidence["saved_artifact_objective_replay_matches"] is True
    assert result.physics_qc["task_evidence_failures"] == ()
