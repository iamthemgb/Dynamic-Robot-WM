from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from dynamic_robot_dataset.backends.source_mujoco import (
    IMPLEMENTED_REVIEW_VARIANTS,
    PINNED_ROLLING_ISLAND_MANIFEST_SHA256,
    PINNED_SOURCE_MANIFEST_SHA256,
    RIGID_REVIEW_PROFILE,
    SourceMujocoBackend,
    SourceMujocoUnsupported,
    compile_review_case,
    prepare_review_case,
    resolve_source_dependency,
    resolve_rolling_island_dependency,
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
from dynamic_robot_dataset.backends.source_mujoco.model import _add_secondary_camera
from dynamic_robot_dataset.backends.source_mujoco.backend import (
    _deflection_evidence,
    _effective_restitution_within_limit,
    _energy_drift_within_limit,
    _restitution_evidence,
    _rolling_evidence,
)
from dynamic_robot_dataset.common.review import event_strip_frame_indices
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.common.source_evaluators import evaluate_source_rows
from dynamic_robot_dataset.common.synchronization import (
    fixed_duration_frame_timestamps,
)


def _case(leaf: str, rollout: int = 0):
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == leaf and case.rollout_index == rollout
    )


def _camera_spec(leaf: str, rollout: int, name: str):
    spec = prepare_review_case(_case(leaf, rollout=rollout))
    return next(camera for camera in spec.cameras if camera.name == name)


def _minimum_projected_sphere_margin_px(result, view: str) -> float:
    calibration = result.camera_calibrations[view]
    focal_px = float(calibration.intrinsic_matrix[0])
    minimum = math.inf
    for row in result.frame_rows:
        pixel_x, pixel_y, depth_m = calibration.project_world(
            row["object.position"]
        )
        radius_px = focal_px * result.scenario.object_radius_m / depth_m
        minimum = min(
            minimum,
            pixel_x - radius_px,
            calibration.width - pixel_x - radius_px,
            pixel_y - radius_px,
            calibration.height - pixel_y - radius_px,
        )
    return minimum


def _secondary_camera_xml(scenario):
    root = ET.fromstring("<mujoco><worldbody /></mujoco>")
    _add_secondary_camera(root, scenario, height_offset=0.0)
    camera = root.find("./worldbody/camera[@name='secondary_camera']")
    assert camera is not None
    return camera


def _camera_forward(camera: ET.Element) -> tuple[float, float, float]:
    axes = tuple(float(value) for value in str(camera.get("xyaxes")).split())
    assert len(axes) == 6
    x_axis = axes[:3]
    y_axis = axes[3:]
    camera_z = (
        x_axis[1] * y_axis[2] - x_axis[2] * y_axis[1],
        x_axis[2] * y_axis[0] - x_axis[0] * y_axis[2],
        x_axis[0] * y_axis[1] - x_axis[1] * y_axis[0],
    )
    norm = math.sqrt(sum(value * value for value in camera_z))
    return tuple(-value / norm for value in camera_z)


def _normalized(vector) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    return tuple(float(value) / norm for value in vector)


def test_calibrated_source_manifest_and_rigid_profile_are_exact() -> None:
    dependency = resolve_source_dependency()
    assert dependency.manifest_sha256 == PINNED_SOURCE_MANIFEST_SHA256
    assert PINNED_SOURCE_MANIFEST_SHA256 == (
        "823463e7095fac9a0819cae2688d75df80a7a38ce6c93b1e72e1323fe469ae99"
    )
    assert RIGID_REVIEW_PROFILE.simulation_hz == 600
    assert RIGID_REVIEW_PROFILE.profile_id.endswith("-v15")
    assert RIGID_REVIEW_PROFILE.wall_solref == (0.012, 0.7)
    assert RIGID_REVIEW_PROFILE.table_rebound_solref == (0.0045, 0.42)
    assert RIGID_REVIEW_PROFILE.minimum_rebound_effective_restitution == 0.15
    assert RIGID_REVIEW_PROFILE.robotiq_pad_friction == (1.5, 0.005, 0.0001)
    assert RIGID_REVIEW_PROFILE.robotiq_pad_half_depth_m == 0.010
    assert RIGID_REVIEW_PROFILE.robotiq_passive_finger_acceleration_limit_rad_s2 == 200.0
    assert RIGID_REVIEW_PROFILE.robotiq_tendon_target == 115.0
    assert RIGID_REVIEW_PROFILE.ready_hover_above_intercept_m == 0.045
    assert RIGID_REVIEW_PROFILE.reach_arrival_before_ballistic_s == 0.055
    assert RIGID_REVIEW_PROFILE.minimum_reach_duration_s == 0.18
    assert SOURCE_MUJOCO_BACKEND_VERSION == "0.19.0-review"
    assert SOURCE_MUJOCO_COMPILED_SCHEMA.endswith("/v9")

    rolling = resolve_rolling_island_dependency()
    assert rolling.manifest_sha256 == PINNED_ROLLING_ISLAND_MANIFEST_SHA256
    assert rolling.controller_imported is False
    assert "controller.py" not in rolling.file_sha256
    assert "robocasa_assets.py" in rolling.file_sha256


def test_f3b_robocasa_style_is_deterministic_and_uses_only_the_asset_rng() -> None:
    case = _case("F3b", rollout=1)
    first = compile_review_case(case).rolling_island_scene
    assert first is not None
    assert first.asset_selection_policy.endswith("/v2")
    assert first.visual_model_xml_sha256
    assert all(path.endswith("/model.xml") for path in first.visual_model_xml_sha256)
    assert first.style_texture_sha256
    assert set(first.style_texture_path_by_material) == {
        "countertop",
        "cabinet_front",
        "scene_floor",
        "scene_wall",
    }
    admitted_families = ("sinks/", "dishwashers/", "fridges/")
    assert all(
        any(family in path for family in admitted_families)
        for path in first.visual_model_xml_sha256
    )
    assert not any("/stoves/" in path for path in first.visual_model_xml_sha256)

    next_assets = replace(case.rng_subseeds, assets=case.rng_subseeds.assets + 1)
    second = compile_review_case(replace(case, rng_subseeds=next_assets)).rolling_island_scene
    assert second is not None
    assert second.layout_id == first.layout_id
    assert second.style_id != first.style_id
    assert second.style_id == 11 + (first.style_id - 11 + 1) % 50
    assert second.task_frame_origin_world_xy_m == first.task_frame_origin_world_xy_m
    assert second.task_frame_yaw_world_rad == first.task_frame_yaw_world_rad
    assert second.counter_position_task_m == first.counter_position_task_m
    assert second.counter_half_size_m == first.counter_half_size_m
    assert second.visual_model_xml_sha256 != first.visual_model_xml_sha256
    assert second.style_texture_sha256 != first.style_texture_sha256


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


@pytest.mark.parametrize("leaf", ("P0c", "P0d"))
def test_randomized_elevated_p0_fixture_has_four_owned_grounded_supports(
    leaf: str,
) -> None:
    scenario = compile_review_case(_case(leaf, rollout=2))
    supports = [
        surface
        for surface in scenario.surfaces
        if surface.role == "structural_support"
    ]
    task_fixtures = {
        surface.name: surface
        for surface in scenario.surfaces
        if surface.expected_task_contact
    }

    assert len(supports) == 4
    assert len({support.name for support in supports}) == 4
    for support in supports:
        assert support.expected_task_contact is False
        assert support.supports_fixture_id in task_fixtures
        assert support.grounded_plane_z_m == 0.0
        assert support.position_m[2] - support.half_size_m[2] == pytest.approx(
            0.0, abs=1e-12
        )
        assert support.support_interface_maximum_mismatch_m is not None
        assert support.support_interface_maximum_mismatch_m <= 0.002


@pytest.mark.integration
@pytest.mark.parametrize(
    ("leaf", "rollout", "expected_outcome"),
    (
        ("F1a", 0, "success"),
        ("F1a", 2, "miss"),
        ("F1a", 4, "contact_failure"),
        ("F1b", 4, "contact_failure"),
        ("F1c", 4, "contact_failure"),
        ("F1d", 3, "miss"),
        ("F1d", 4, "contact_failure"),
    ),
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
    assert not _effective_restitution_within_limit(0.149, minimum=0.15)
    assert _effective_restitution_within_limit(0.15, minimum=0.15)
    assert _effective_restitution_within_limit(1.05)
    assert not _effective_restitution_within_limit(-0.01)
    assert not _effective_restitution_within_limit(1.051)


def test_timestep_gate_uses_passive_qc_and_replay_not_constant_display_label() -> None:
    common = {
        "outcome": "passive_observation",
        "key_event_time_s": 0.3,
        "key_event_position_m": [0.0, 0.0, 0.025],
    }
    failures = timestep_comparison_failures(
        {
            **common,
            "task_success": False,
            "physics_qc_pass": False,
            "saved_artifact_objective_replay_matches": False,
        },
        {
            **common,
            "task_success": True,
            "physics_qc_pass": True,
            "saved_artifact_objective_replay_matches": True,
        },
    )

    assert set(failures) == {
        "task_success_changed_at_1200_hz",
        "physics_qc_changed_at_1200_hz",
        "saved_artifact_replay_changed_at_1200_hz",
        "physics_qc_failed_at_comparison_rate",
        "saved_artifact_replay_failed_at_comparison_rate",
    }


def test_timestep_gate_fails_closed_without_semantic_evidence() -> None:
    failures = timestep_comparison_failures(
        {
            "outcome": "passive_observation",
            "key_event_time_s": 0.3,
            "key_event_position_m": [0.0, 0.0, 0.025],
        },
        {
            "outcome": "passive_observation",
            "key_event_time_s": 0.3,
            "key_event_position_m": [0.0, 0.0, 0.025],
        },
    )
    assert set(failures) == {
        "task_success_missing",
        "physics_qc_pass_missing",
        "saved_artifact_objective_replay_matches_missing",
    }


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


def test_p0c_table_uses_rebound_material_without_changing_p0d_support() -> None:
    bounce = compile_review_case(_case("P0c", rollout=0))
    rolling = compile_review_case(_case("P0d", rollout=0))

    assert bounce.surfaces[0].solref == RIGID_REVIEW_PROFILE.table_rebound_solref
    assert rolling.surfaces[0].solref == (0.003, 1.0)
    for rollout, expected_dynamic_friction in ((0, 0.9), (2, 0.9), (4, 0.72)):
        scenario = compile_review_case(_case("P0c", rollout=rollout))
        bounce_spec = prepare_review_case(_case("P0c", rollout=rollout))
        fixture = bounce_spec.fixtures[0]
        assert scenario.surfaces[0].solref == (
            RIGID_REVIEW_PROFILE.table_rebound_solref
        )
        assert fixture.parameters["solref"] == list(
            RIGID_REVIEW_PROFILE.table_rebound_solref
        )
        assert fixture.parameters["friction"][0] == pytest.approx(
            expected_dynamic_friction
        )
        assert fixture.parameters["contact_material_profile"] == (
            "p0c_table_rebound_v1"
        )
        assert bounce_spec.physics["rebound_acceptance"] == (
            RIGID_REVIEW_PROFILE.rebound_acceptance().to_dict()
        )


@pytest.mark.parametrize(
    ("rollout", "expected_variant", "expected_pad_top_m"),
    (
        (0, "table_bounce", RIGID_REVIEW_PROFILE.table_bounce_pad_top_z_m),
        (1, "table_bounce", RIGID_REVIEW_PROFILE.table_bounce_pad_top_z_m),
        (2, "floor_bounce", RIGID_REVIEW_PROFILE.floor_bounce_pad_top_z_m),
        (3, "floor_bounce", RIGID_REVIEW_PROFILE.floor_bounce_pad_top_z_m),
        (4, "table_bounce", RIGID_REVIEW_PROFILE.table_bounce_pad_top_z_m),
        (5, "table_bounce", RIGID_REVIEW_PROFILE.table_bounce_pad_top_z_m),
    ),
)
def test_f2c_uses_one_calibrated_bounce_pad_with_four_grounded_supports(
    rollout: int,
    expected_variant: str,
    expected_pad_top_m: float,
) -> None:
    scenario = compile_review_case(_case("F2c", rollout=rollout))
    pad = next(
        surface for surface in scenario.surfaces if surface.expected_task_contact
    )
    supports = tuple(
        surface
        for surface in scenario.surfaces
        if surface.role == "structural_support"
    )

    assert scenario.task_variant == expected_variant
    assert scenario.robot_base_position_m == (0.0, 0.0, 0.0)
    assert pad.name == "owned_bounce_pad"
    assert pad.role == "table"
    assert pad.contact_profile == "rebound_pad"
    assert pad.solref == RIGID_REVIEW_PROFILE.bounce_pad_solref
    assert pad.position_m[2] + pad.half_size_m[2] == pytest.approx(
        expected_pad_top_m
    )
    assert len(supports) == 4
    assert len({support.name for support in supports}) == 4
    for support in supports:
        assert support.expected_task_contact is False
        assert support.supports_fixture_id == pad.name
        assert support.grounded_plane_z_m == 0.0
        assert support.position_m[2] - support.half_size_m[2] == pytest.approx(
            0.0, abs=1e-12
        )
        assert support.position_m[2] + support.half_size_m[2] == pytest.approx(
            pad.position_m[2] - pad.half_size_m[2], abs=1e-12
        )
        assert support.support_interface_maximum_mismatch_m == 0.0


def test_f2c_source_spec_persists_the_calibrated_rebound_material() -> None:
    spec = prepare_review_case(_case("F2c", rollout=0))
    pad = next(
        fixture
        for fixture in spec.fixtures
        if fixture.fixture_id == "owned_bounce_pad"
    )

    assert pad.parameters["solref"] == list(RIGID_REVIEW_PROFILE.bounce_pad_solref)
    assert pad.parameters["contact_material_profile"] == "f2c_bounce_pad_v1"
    assert spec.physics["simulation_hz"] == 1200
    assert spec.physics["rebound_acceptance"] == (
        RIGID_REVIEW_PROFILE.rebound_acceptance().to_dict()
    )


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
    wall = next(
        surface
        for surface in scenario.surfaces
        if surface.expected_task_contact and surface.role == "wall"
    )
    assert wall.name == "supported_wall"


def test_wall_rebound_main_camera_stays_on_visible_incoming_side() -> None:
    spec = prepare_review_case(_case("P0c", rollout=3))
    main = next(camera for camera in spec.cameras if camera.name == "main")
    assert main.pose.position_m == pytest.approx((-0.90, 0.95, 1.79))
    assert main.pose.position_m[0] < spec.initial_state["object_position_m"][0] + 0.1
    assert main.fovy_deg == 64.0


@pytest.mark.parametrize("rollout", (2, 3))
def test_f2f_barrier_secondary_camera_frames_the_complete_miss_envelope(
    rollout: int,
) -> None:
    scenario = compile_review_case(_case("F2f", rollout=rollout))
    assert scenario.task_variant == "random_barrier_bounce"
    assert scenario.motion_kind == "arbitrary_surface_rebound_interception"
    anchor = scenario.physical_target_position_m
    assert anchor is not None

    camera = _secondary_camera_xml(scenario)
    position = tuple(float(value) for value in str(camera.get("pos")).split())
    expected_position = (anchor[0] - 0.10, anchor[1] - 1.60, anchor[2] + 0.60)
    expected_target = (anchor[0] + 0.05, anchor[1], anchor[2] - 0.14)

    assert position == pytest.approx(expected_position)
    assert float(str(camera.get("fovy"))) == 62.0
    assert _camera_forward(camera) == pytest.approx(
        _normalized(
            tuple(
                target_value - position_value
                for target_value, position_value in zip(
                    expected_target, expected_position
                )
            )
        )
    )


@pytest.mark.parametrize("rollout", (0, 2))
def test_f2d_wall_secondary_camera_keeps_its_existing_route(rollout: int) -> None:
    scenario = compile_review_case(_case("F2d", rollout=rollout))
    assert scenario.motion_kind == "wall_rebound_interception"
    anchor = scenario.physical_target_position_m
    assert anchor is not None

    camera = _secondary_camera_xml(scenario)
    position = tuple(float(value) for value in str(camera.get("pos")).split())
    expected_position = (anchor[0] - 0.10, anchor[1] - 1.60, anchor[2] + 0.60)
    expected_target = (anchor[0] + 0.05, anchor[1], anchor[2] + 0.45)

    assert position == pytest.approx(expected_position)
    assert float(str(camera.get("fovy"))) == 62.0
    assert _camera_forward(camera) == pytest.approx(
        _normalized(
            tuple(
                target_value - position_value
                for target_value, position_value in zip(
                    expected_target, expected_position
                )
            )
        )
    )


@pytest.mark.parametrize("rollout", (0, 4))
def test_f2f_plane_secondary_camera_keeps_its_existing_route(rollout: int) -> None:
    scenario = compile_review_case(_case("F2f", rollout=rollout))
    assert scenario.task_variant == "random_plane_bounce"
    assert scenario.motion_kind == "random_plane_bounce_pickup_interception"
    anchor = scenario.physical_target_position_m
    assert anchor is not None

    camera = _secondary_camera_xml(scenario)
    position = tuple(float(value) for value in str(camera.get("pos")).split())
    expected_position = (anchor[0] + 1.05, anchor[1] - 0.35, anchor[2] + 0.45)
    expected_target = (anchor[0], anchor[1], anchor[2] + 0.05)

    assert position == pytest.approx(expected_position)
    assert float(str(camera.get("fovy"))) == 55.0
    assert _camera_forward(camera) == pytest.approx(
        _normalized(
            tuple(
                target_value - position_value
                for target_value, position_value in zip(
                    expected_target, expected_position
                )
            )
        )
    )


def test_p0b_main_camera_has_exact_owned_projectile_serialization() -> None:
    expected_quaternion = (
        0.705512384115837,
        0.5825692730020547,
        0.25695922448486036,
        0.311186881094278,
    )
    for rollout in range(6):
        height_offset = 0.0 if rollout == 0 else 0.74
        main = _camera_spec("P0b", rollout, "main")

        assert main.role == "main_three_quarter_external"
        assert main.pose.position_m == pytest.approx(
            (1.20, -1.15, height_offset + 1.15)
        )
        assert main.pose.quaternion_wxyz == pytest.approx(expected_quaternion)
        assert main.look_at_m == pytest.approx(
            (
                0.4748483396525184,
                -0.487905005855077,
                height_offset + 0.9608300017880067,
            )
        )
        assert (main.width, main.height, main.fps) == (832, 480, 30)
        assert main.fovy_deg == 58.0


def test_p0c_wall_secondary_has_exact_side_on_serialization() -> None:
    for rollout in (1, 3, 5):
        secondary = _camera_spec("P0c", rollout, "secondary")

        assert secondary.role == "task_specific_secondary"
        assert secondary.pose.position_m == pytest.approx((-0.05, -1.70, 1.50))
        assert secondary.pose.quaternion_wxyz == pytest.approx(
            (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
        )
        assert secondary.look_at_m == pytest.approx((-0.05, -0.70, 1.50))
        assert (secondary.width, secondary.height, secondary.fps) == (
            832,
            480,
            30,
        )
        assert secondary.fovy_deg == 58.0


def test_p0c_table_secondary_serialization_is_unchanged() -> None:
    expected_quaternion = (
        0.8218820310890158,
        0.5009891177522585,
        -0.14112776965890958,
        -0.23152274941765338,
    )
    for rollout in (0, 2, 4):
        height_offset = 0.0 if rollout == 0 else 0.74
        secondary = _camera_spec("P0c", rollout, "secondary")

        assert secondary.pose.position_m == pytest.approx(
            (-0.72, -1.18, height_offset + 1.10)
        )
        assert secondary.pose.quaternion_wxyz == pytest.approx(expected_quaternion)
        assert secondary.look_at_m == pytest.approx(
            (
                -0.25603848811869045,
                -0.42184067119778335,
                height_offset + 0.6418142869501693,
            )
        )
        assert (secondary.width, secondary.height, secondary.fps) == (
            832,
            480,
            30,
        )
        assert secondary.fovy_deg == 52.0


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
def test_p0b_main_camera_contains_all_six_fixed_trajectories() -> None:
    backend = SourceMujocoBackend()
    robust_minimum_margin_px = 32.0

    for rollout in range(6):
        result = backend.run(_case("P0b", rollout=rollout), render=False)

        assert result.physics_qc["physics_qc_pass"] is True
        assert result.background_clearance["clearance_pass"] is True
        assert (
            _minimum_projected_sphere_margin_px(result, "main")
            >= robust_minimum_margin_px
        )


@pytest.mark.integration
def test_p0b_review_02_rendered_main_view_preserves_review_evidence() -> None:
    result = SourceMujocoBackend().run(_case("P0b", rollout=2), render=True)
    visibility = result.visibility_qc
    main = visibility["views"]["main"]

    assert result.physics_qc["physics_qc_pass"] is True
    assert result.outcome["intended_outcome_match"] is True
    assert result.outcome["key_event_name"] == "projectile_apex"
    assert result.outcome["key_event_source"] == "persisted_free_flight_apex"
    assert result.background_clearance["clearance_pass"] is True
    assert visibility["critically_cropped"] is False
    assert visibility["maximum_underexposed_fraction"] <= 0.10
    assert visibility["maximum_overexposed_fraction"] <= 0.10
    assert main["target_visible_frame_fraction"] == 1.0
    assert main["minimum_object_area_px"] >= 64
    assert main["minimum_segmentation_bbox_margin_px"] >= 48.0
    assert main["key_event_object_pixel_count"] >= 100
    for checkpoint in ("initial", "apex", "key_event", "final"):
        assert main["checkpoints"][checkpoint]["visible"] is True
        assert main["checkpoints"][checkpoint]["object_pixel_count"] >= 64


@pytest.mark.integration
def test_wall_rebound_views_preserve_framing_and_side_on_separation() -> None:
    backend = SourceMujocoBackend()
    main_minimum_edge_margin_px = 8.0
    secondary_minimum_edge_margin_px = 32.0
    minimum_post_event_separation_px = 32.0
    minimum_final_separation_px = 75.0
    for rollout in (1, 3, 5):
        result = backend.run(_case("P0c", rollout=rollout), render=False)
        assert result.physics_qc["physics_qc_pass"] is True
        assert result.background_clearance["clearance_pass"] is True
        assert (
            _minimum_projected_sphere_margin_px(result, "main")
            >= main_minimum_edge_margin_px
        )
        assert (
            _minimum_projected_sphere_margin_px(result, "secondary")
            >= secondary_minimum_edge_margin_px
        )

        calibration = result.camera_calibrations["secondary"]
        # CameraCalibration stores forward in the third column.  A side-on
        # view of this X-normal wall must look along Y, not along the rebound.
        view_forward = (
            calibration.camera_to_world[2],
            calibration.camera_to_world[6],
            calibration.camera_to_world[10],
        )
        assert abs(view_forward[0]) <= 0.05

        timestamps = [float(row["timestamp"]) for row in result.frame_rows]
        event_time_s = float(result.outcome["key_event_time_s"])
        event_index = min(
            range(len(timestamps)),
            key=lambda index: abs(timestamps[index] - event_time_s),
        )
        post_index = min(
            range(len(timestamps)),
            key=lambda index: abs(timestamps[index] - (event_time_s + 0.3)),
        )
        event_position = tuple(result.frame_rows[event_index]["object.position"])
        event_pixel = calibration.project_world(event_position)

        def wall_normal_separation_px(index: int) -> tuple[float, float]:
            position = result.frame_rows[index]["object.position"]
            normal_only_position = (
                position[0],
                event_position[1],
                event_position[2],
            )
            pixel = calibration.project_world(normal_only_position)
            return pixel[0] - event_pixel[0], pixel[1] - event_pixel[1]

        post_dx, post_dy = wall_normal_separation_px(post_index)
        final_dx, final_dy = wall_normal_separation_px(len(result.frame_rows) - 1)
        assert math.hypot(post_dx, post_dy) >= minimum_post_event_separation_px
        assert math.hypot(final_dx, final_dy) >= minimum_final_separation_px
        assert abs(post_dy) <= 1e-6
        assert abs(final_dy) <= 1e-6


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
    assert faster_bounce.simulation_hz == 600

    rolling_spin = compile_review_case(_case("P0d", rollout=3))
    nominal_spin = rolling_spin.object_initial_linear_velocity_m_s[0] / (
        rolling_spin.object_radius_m
    )
    assert rolling_spin.object_initial_angular_velocity_rad_s == pytest.approx(
        (0.0, 1.10 * nominal_spin, 0.0)
    )


def test_reference_rate_is_selected_only_for_failed_fixed_case_classes() -> None:
    assert compile_review_case(_case("P0c", rollout=1)).simulation_hz == 1200
    assert compile_review_case(_case("P0c", rollout=2)).simulation_hz == 600
    assert compile_review_case(_case("F1a", rollout=4)).simulation_hz == 1200
    assert compile_review_case(_case("F1b", rollout=4)).simulation_hz == 1200
    assert compile_review_case(_case("F1c", rollout=4)).simulation_hz == 600
    assert compile_review_case(_case("F1a", rollout=5)).simulation_hz == 600
    # v9 reaching-controller Robotiq nominal catches: the F1a/F1d
    # first-bilateral samples shifted past the 1 cm halving gate while
    # F1b/F1c stayed within it.
    assert compile_review_case(_case("F1a", rollout=1)).simulation_hz == 1200
    assert compile_review_case(_case("F1b", rollout=1)).simulation_hz == 600
    assert compile_review_case(_case("F1c", rollout=1)).simulation_hz == 600
    assert compile_review_case(_case("F1d", rollout=1)).simulation_hz == 1200
    assert compile_review_case(_case("F1a", rollout=3)).simulation_hz == 600
    # v10 measured classes: the F2a Robotiq nominal catch (21.8 mm) and the
    # F2a Panda negative-timing graze (32.7 mm) shift past the 1 cm gate
    # with full semantic and strict-QC agreement at both rates.  The F3b
    # rolling cases stay within the gate or have no contact event at all.
    assert compile_review_case(_case("F2a", rollout=1)).simulation_hz == 1200
    assert compile_review_case(_case("F2a", rollout=4)).simulation_hz == 1200
    assert compile_review_case(_case("F2a", rollout=0)).simulation_hz == 600
    assert compile_review_case(_case("F2a", rollout=2)).simulation_hz == 600
    # Every fixed F2c class uses the 1200 Hz reference profile: 600 Hz
    # under-resolves the calibrated stiff bounce-pad contact.
    for rollout in range(6):
        assert (
            compile_review_case(_case("F2c", rollout=rollout)).simulation_hz
            == 1200
        )
    for rollout in range(6):
        assert compile_review_case(_case("F3b", rollout=rollout)).simulation_hz == 600


def test_unimplemented_f3_paths_fail_closed() -> None:
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
    # The serialized phases must show the real reach: minimum-jerk arm motion
    # phases with nonzero commanded arm travel, not a pre-positioned hold.
    phase_names = [phase.name for phase in spec.actuator_phases]
    assert "minimum_jerk_reach" in phase_names
    assert "reach_with_bounded_closure" in phase_names
    arm_names = spec.embodiment.action_names[:7]
    reach = next(
        phase
        for phase in spec.actuator_phases
        if phase.name == "minimum_jerk_reach"
    )
    final = next(
        phase
        for phase in spec.actuator_phases
        if phase.name == "final_retention"
    )
    assert reach.interpolation == "jerk_limited"
    ready_arm = spec.initial_state["robot_initial_joint_qpos"][:7]
    travel = max(
        abs(final.commands[name] - initial)
        for name, initial in zip(arm_names, ready_arm)
    )
    assert travel >= 0.01


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
    # The v8 defect: the robot waited at the interception pose with exactly
    # zero commanded arm travel.  A genuine reaching catch must command real
    # arm motion, arrive at the intercept before the ballistic event, and
    # never rewrite robot/object state after initialization.
    checks = result.physics_qc["checks"]
    assert checks["arm_command_travel_present"] is True
    assert checks["arm_command_travel_rad"] >= 0.01
    assert checks["arm_arrived_at_commanded_intercept"] is True
    assert checks["reach_arrival_distance_m"] <= 0.025
    arm_commands = [row["action.actuator_command"][:7] for row in result.high_rate_rows]
    per_joint_travel = [
        max(values) - min(values) for values in zip(*arm_commands)
    ]
    assert max(per_joint_travel) >= 0.01
    assert result.runtime_audit["direct_robot_state_writes_after_initialization"] == 0
    assert result.runtime_audit["object_state_writes_after_initialization"] == 0
    assert result.runtime_audit["object_linked_equality_changes_after_initialization"] == 0


@pytest.mark.integration
def test_fixed_franka_reaching_catch_passes_the_600_1200_timestep_gate() -> None:
    backend = SourceMujocoBackend()
    scenario = backend.compile_case(_case("F1a", rollout=0))
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
        assert result.physics_qc["checks"]["arm_command_travel_present"] is True
        assert (
            result.physics_qc["checks"]["arm_arrived_at_commanded_intercept"]
            is True
        )
        event = bilateral[0]
        observations.append(
            {
                "outcome": result.outcome["actual_outcome"],
                "task_success": result.outcome["task_success"],
                "physics_qc_pass": result.physics_qc["physics_qc_pass"],
                "saved_artifact_objective_replay_matches": result.outcome[
                    "saved_artifact_objective_replay_matches"
                ],
                "key_event_time_s": event["timestamp"],
                "key_event_position_m": event["object.position"],
            }
        )
    assert timestep_comparison_failures(*observations) == ()


@pytest.mark.integration
def test_fixed_robotiq_catch_agrees_semantically_and_compiles_at_reference_rate() -> None:
    # Under the v9 reaching controller the Robotiq F1a nominal catch keeps
    # outcome, task-success, QC, and replay agreement at 600 and 1200 Hz, but
    # its first-bilateral-contact sample shifts more than 1 cm at the ball's
    # 4.3 m/s arrival speed.  The measured exception class therefore compiles
    # directly at the calibrated 1200 Hz reference instead of weakening the
    # halving gate.
    backend = SourceMujocoBackend()
    scenario = backend.compile_case(_case("F1a", rollout=1))
    assert scenario.simulation_hz == (
        RIGID_REVIEW_PROFILE.comparison_simulation_hz
    )
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
                "task_success": result.outcome["task_success"],
                "physics_qc_pass": result.physics_qc["physics_qc_pass"],
                "saved_artifact_objective_replay_matches": result.outcome[
                    "saved_artifact_objective_replay_matches"
                ],
                "key_event_time_s": event["timestamp"],
                "key_event_position_m": event["object.position"],
            }
        )
    failures = timestep_comparison_failures(*observations)
    assert set(failures) <= {"key_event_position_shift_exceeds_1cm"}


@pytest.mark.integration
def test_fixed_p0c_table_cases_pass_600_1200_semantic_gate() -> None:
    backend = SourceMujocoBackend()
    for rollout in (0, 2, 4):
        case = _case("P0c", rollout=rollout)
        scenario = backend.compile_case(case)
        assert scenario.simulation_hz == 600
        assert scenario.surfaces[0].solref == RIGID_REVIEW_PROFILE.table_rebound_solref
        observations = []
        for simulation_hz in (
            RIGID_REVIEW_PROFILE.simulation_hz,
            RIGID_REVIEW_PROFILE.comparison_simulation_hz,
        ):
            result = backend.run(
                replace(scenario, simulation_hz=simulation_hz), render=False
            )
            rebound = result.physics_qc["restitution"]
            source_spec = prepare_review_case(case).to_dict()
            source_spec["physics"]["simulation_hz"] = simulation_hz
            replay = evaluate_source_rows(
                evaluator_id=case.evaluator,
                corpus_leaf_id="P0c",
                task_variant=case.task_variant,
                source_spec=source_spec,
                state_rows=result.high_rate_rows,
                event_rows=result.contact_rows,
            )
            assert result.physics_qc["physics_qc_pass"] is True
            assert result.outcome["task_success"] is True
            assert replay.task_success is True
            observations.append(
                {
                    "outcome": result.outcome["actual_outcome"],
                    "task_success": result.outcome["task_success"],
                    "physics_qc_pass": result.physics_qc["physics_qc_pass"],
                    "saved_artifact_objective_replay_matches": (
                        replay.task_success == result.outcome["task_success"]
                    ),
                    "key_event_time_s": rebound["event_time_s"],
                    "key_event_position_m": rebound["event_position_m"],
                }
            )
        assert timestep_comparison_failures(*observations) == ()


@pytest.mark.integration
def test_fixed_p0c_wall_cases_select_rate_from_semantic_timestep_gate() -> None:
    backend = SourceMujocoBackend()
    for rollout in (1, 3, 5):
        case = _case("P0c", rollout=rollout)
        scenario = backend.compile_case(case)
        observations = []
        for simulation_hz in (
            RIGID_REVIEW_PROFILE.simulation_hz,
            RIGID_REVIEW_PROFILE.comparison_simulation_hz,
        ):
            result = backend.run(
                replace(scenario, simulation_hz=simulation_hz), render=False
            )
            rebound = result.physics_qc["restitution"]
            source_spec = prepare_review_case(case).to_dict()
            source_spec["physics"]["simulation_hz"] = simulation_hz
            replay = evaluate_source_rows(
                evaluator_id=case.evaluator,
                corpus_leaf_id="P0c",
                task_variant=case.task_variant,
                source_spec=source_spec,
                state_rows=result.high_rate_rows,
                event_rows=result.contact_rows,
            )
            observations.append(
                {
                    "outcome": result.outcome["actual_outcome"],
                    "task_success": result.outcome["task_success"],
                    "physics_qc_pass": result.physics_qc["physics_qc_pass"],
                    "saved_artifact_objective_replay_matches": (
                        replay.task_success == result.outcome["task_success"]
                    ),
                    "key_event_time_s": rebound["event_time_s"],
                    "key_event_position_m": rebound["event_position_m"],
                }
            )
        failures = timestep_comparison_failures(*observations)
        if rollout == 1:
            assert scenario.simulation_hz == 1200
            assert observations[0]["physics_qc_pass"] is False
            assert observations[1]["physics_qc_pass"] is True
            assert set(failures) == {
                "task_success_changed_at_1200_hz",
                "physics_qc_changed_at_1200_hz",
                "physics_qc_failed_at_comparison_rate",
            }
        else:
            assert scenario.simulation_hz == 600
            assert failures == ()


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


def test_f3b_recipe_rolls_without_slip_on_the_real_counter_without_a_backstop() -> None:
    for rollout in range(6):
        scenario = compile_review_case(_case("F3b", rollout=rollout))
        assert scenario.motion_kind == "rolling_pickup_interception"
        vx, vy, vz = scenario.object_initial_linear_velocity_m_s
        wx, wy, wz = scenario.object_initial_angular_velocity_rad_s
        radius = scenario.object_radius_m
        assert math.hypot(vx - wy * radius, vy + wx * radius) < 1e-9
        assert vz == 0.0
        assert [surface.role for surface in scenario.surfaces] == ["table"]
        table = scenario.surfaces[0]
        assert table.name == "supported_rolling_pickup_island_top"
        table_top = table.position_m[2] + table.half_size_m[2]
        assert scenario.object_initial_position_m[2] == pytest.approx(
            table_top + radius
        )
        assert scenario.physical_target_position_m[2] == pytest.approx(
            table_top + radius
        )
        if scenario.branch_role == "nominal_success":
            assert scenario.controller_transport_position_m is not None
            assert scenario.controller_transport_position_m[2] == pytest.approx(
                scenario.physical_target_position_m[2]
                + RIGID_REVIEW_PROFILE.pickup_lift_height_m
            )
        else:
            assert scenario.controller_transport_position_m is None
        standoff = (
            RIGID_REVIEW_PROFILE.robotiq_pickup_standoff_m
            if scenario.embodiment == "robotiq_2f85_thick_pad"
            else 0.0
        )
        assert scenario.controller_target_position_m[2] == pytest.approx(
            scenario.physical_target_position_m[2] + standoff
        )
        if rollout == 0:
            assert scenario.rolling_island_scene is None
        else:
            assert scenario.rolling_island_scene is not None


def test_f2a_deflection_shares_the_direct_catch_ballistic_construction() -> None:
    catch = compile_review_case(_case("F2a", rollout=0))
    deflection = compile_review_case(_case("F2a", rollout=2))
    assert catch.task_variant == "direct_catch"
    assert deflection.task_variant == "direct_deflection"
    assert (
        catch.motion_kind
        == deflection.motion_kind
        == "direct_free_contact_interception"
    )
    assert deflection.controller_transport_position_m is None


def test_deflection_evidence_requires_measured_contact_for_impulse() -> None:
    scenario = compile_review_case(_case("F2a", rollout=2))
    dt = 1.0 / scenario.simulation_hz

    def _row(index, velocity, mode="free_flight", contacts=0):
        return {
            "timestamp": index * dt,
            "object.linear_velocity": list(velocity),
            "object.motion_mode": mode,
            "contact.count": contacts,
        }

    falling = [_row(i, (0.0, 0.0, -9.81 * i * dt)) for i in range(200)]
    clean = _deflection_evidence(falling, scenario)
    assert clean["deflection_contact_occurred"] is False
    assert clean["object_redirected_by_hand_contact"] is False
    assert clean["velocity_change_matches_measured_contact_impulse"] is True

    jumped = list(falling)
    jumped[100] = _row(100, (0.9, 0.0, -9.81 * 100 * dt))
    tampered = _deflection_evidence(jumped, scenario)
    assert tampered["velocity_change_matches_measured_contact_impulse"] is False

    rows = [_row(i, (0.0, 0.0, -2.0)) for i in range(50)]
    for i in range(50, 80):
        blend = (i - 50) / 30.0
        rows.append(
            _row(i, (0.0, 0.0, -2.0 + 3.5 * blend), mode="gripper_contact", contacts=1)
        )
    rows.extend(_row(i, (0.0, 0.0, 1.5)) for i in range(80, 200))
    deflected = _deflection_evidence(rows, scenario)
    assert deflected["deflection_contact_occurred"] is True
    assert deflected["object_redirected_by_hand_contact"] is True
    assert deflected["velocity_change_matches_measured_contact_impulse"] is True
    assert deflected["deflection_redirect_angle_deg"] > 90.0


def test_rolling_evidence_uses_only_the_first_sustained_surface_segment() -> None:
    scenario = compile_review_case(_case("F3b", rollout=2))
    radius = scenario.object_radius_m
    dt = 1.0 / scenario.simulation_hz

    def _row(index, x, vx, mode):
        return {
            "timestamp": index * dt,
            "object.position": [x, 0.0, radius],
            "object.linear_velocity": [vx, 0.0, 0.0],
            "object.angular_velocity": [0.0, vx / radius, 0.0],
            "object.motion_mode": mode,
        }

    rows = [_row(i, 0.7 - 0.55 * i * dt, -0.55, "surface_contact") for i in range(300)]
    rows.extend(_row(300 + i, 0.05, -3.0, "free_flight") for i in range(60))
    # A wild second surface segment (post-fall floor skid) that would break
    # both the deceleration fit and the slip limit if it were included.
    rows.extend(_row(360 + i, 0.0, 3.0, "gripper_contact") for i in range(200))
    evidence = _rolling_evidence(rows, scenario)
    assert evidence["rolling_or_sliding_slip_within_limit"] is True
    assert evidence["friction_deceleration_consistent"] is True
    assert abs(evidence["measured_tangent_acceleration_m_s2"]) < 0.05


def test_fixed_f2a_deflection_negative_is_a_clean_declared_miss() -> None:
    result = SourceMujocoBackend().run(_case("F2a", rollout=2), render=False)
    assert result.outcome["task_success"] is False
    assert result.outcome["intended_outcome_match"] is True
    assert result.physics_qc["physics_qc_pass"] is True
    evidence = result.physics_qc["task_evidence"]
    assert evidence["deflection_contact_occurred"] is False
    assert evidence["object_redirected_by_hand_contact"] is False
    assert evidence["velocity_change_matches_measured_contact_impulse"] is True
    gripper_commands = {
        round(float(row["action.actuator_command"][7]), 9)
        for row in result.high_rate_rows
    }
    assert len(gripper_commands) == 1


@pytest.mark.integration
def test_fixed_f2c_matrix_passes_rebound_qc_without_state_assistance() -> None:
    backend = SourceMujocoBackend()
    expected_outcomes = (
        "success",
        "success",
        "miss",
        "miss",
        "contact_failure",
        "contact_failure",
    )
    forbidden_runtime_mutations = (
        "object_state_writes_after_initialization",
        "direct_robot_state_writes_after_initialization",
        "mocap_writes_after_initialization",
        "applied_force_writes_after_initialization",
        "object_linked_equality_changes_after_initialization",
        "model_physics_mutations_after_initialization",
    )

    for rollout, expected_outcome in enumerate(expected_outcomes):
        case = _case("F2c", rollout=rollout)
        result = backend.run(case, render=False)
        rebound = result.physics_qc["restitution"]
        replay = evaluate_source_rows(
            evaluator_id=case.evaluator,
            corpus_leaf_id=case.corpus_leaf_id,
            task_variant=case.task_variant,
            source_spec=prepare_review_case(case).to_dict(),
            state_rows=result.high_rate_rows,
            event_rows=result.contact_rows,
        )

        assert result.scenario.simulation_hz == 1200
        assert result.outcome["actual_outcome"] == expected_outcome
        assert result.outcome["intended_outcome_match"] is True
        assert result.outcome["saved_artifact_objective_replay_matches"] is True
        assert replay.task_success is result.outcome["task_success"]
        assert result.physics_qc["physics_qc_pass"] is True
        assert rebound["applicable"] is True
        assert rebound["separated_pre_post_contact_samples"] is True
        assert rebound["measured_contact_normal"] is True
        assert rebound["rebound_acceptance_pass"] is True
        assert result.background_clearance["clearance_pass"] is True
        assert any(
            row["contact_category"] == "task_surface"
            for row in result.contact_rows
        )
        assert all(
            result.runtime_audit[name] == 0
            for name in forbidden_runtime_mutations
        )

        evidence = result.physics_qc["task_evidence"]
        assert replay.evidence["retained_through_final_state"] is evidence[
            "retained_through_final_state"
        ]
        if rollout == 5:
            contact_categories = {
                row["contact_category"] for row in result.contact_rows
            }
            assert "robot_arm" in contact_categories
            assert "gripper" not in contact_categories
            assert result.outcome["actual_outcome"] == "contact_failure"
            assert replay.actual_outcome_class.value == "contact_failure"
        if rollout < 2:
            assert evidence["sustained_opposing_bilateral_contacts"] is True
            assert evidence["stable_object_to_grasp_transform"] is True
            assert evidence["retained_through_final_state"] is True
            assert evidence["final_retention_bilateral_fraction"] >= 0.95
            assert evidence["displacement_physically_supported_by_contacts"] is True
            assert replay.evidence["retained_through_final_state"] is True
            assert replay.evidence["final_retention_bilateral_fraction"] >= 0.95
        else:
            assert evidence["retained_through_final_state"] is False
            assert replay.evidence["retained_through_final_state"] is False


@pytest.mark.integration
def test_f2c_robotiq_nominal_rejects_600_hz_candidate_and_keeps_reference_rate() -> None:
    case = _case("F2c", rollout=1)
    backend = SourceMujocoBackend()
    scenario = backend.compile_case(case)
    assert scenario.simulation_hz == 1200
    observations = []
    results = []
    for simulation_hz in (600, 1200):
        result = backend.run(
            replace(scenario, simulation_hz=simulation_hz), render=False
        )
        source_spec = prepare_review_case(case).to_dict()
        source_spec["physics"]["simulation_hz"] = simulation_hz
        replay = evaluate_source_rows(
            evaluator_id=case.evaluator,
            corpus_leaf_id="F2c",
            task_variant=case.task_variant,
            source_spec=source_spec,
            state_rows=result.high_rate_rows,
            event_rows=result.contact_rows,
        )
        rebound = result.physics_qc["restitution"]
        observations.append(
            {
                "outcome": result.outcome["actual_outcome"],
                "task_success": result.outcome["task_success"],
                "physics_qc_pass": result.physics_qc["physics_qc_pass"],
                "saved_artifact_objective_replay_matches": (
                    replay.task_success == result.outcome["task_success"]
                ),
                "key_event_time_s": rebound["event_time_s"],
                "key_event_position_m": rebound["event_position_m"],
            }
        )
        results.append(result)

    assert results[0].outcome["actual_outcome"] == "contact_failure"
    assert results[0].physics_qc["physics_qc_pass"] is False
    assert results[1].outcome["actual_outcome"] == "success"
    assert results[1].physics_qc["physics_qc_pass"] is True
    assert set(timestep_comparison_failures(*observations)) == {
        "outcome_changed_at_1200_hz",
        "task_success_changed_at_1200_hz",
        "physics_qc_changed_at_1200_hz",
        "physics_qc_failed_at_comparison_rate",
    }


def test_fixed_f3b_rolling_pickup_is_a_strict_lifted_free_contact_success() -> None:
    result = SourceMujocoBackend().run(_case("F3b", rollout=0), render=False)
    assert result.outcome["task_success"] is True
    assert result.outcome["intended_outcome_match"] is True
    assert result.physics_qc["physics_qc_pass"] is True
    evidence = result.physics_qc["task_evidence"]
    assert evidence["sustained_opposing_bilateral_contacts"] is True
    assert evidence["stable_object_to_grasp_transform"] is True
    assert evidence["displacement_physically_supported_by_contacts"] is True
    assert evidence["rolling_or_sliding_slip_within_limit"] is True
    assert evidence["friction_deceleration_consistent"] is True
    checks = result.physics_qc["checks"]
    assert checks["arm_command_travel_present"] is True
    assert checks["arm_arrived_at_commanded_intercept"] is True
    assert result.physics_qc["maximum_joint_acceleration_rad_s2"] <= 80.0
    assert result.runtime_audit["object_state_writes_after_initialization"] == 0
    assert result.runtime_audit["direct_robot_state_writes_after_initialization"] == 0


@pytest.mark.integration
def test_fixed_f3b_matrix_passes_the_v13_600_1200_timestep_gate() -> None:
    backend = SourceMujocoBackend()
    for rollout in range(6):
        scenario = backend.compile_case(_case("F3b", rollout=rollout))
        observations = []
        for simulation_hz in (
            RIGID_REVIEW_PROFILE.simulation_hz,
            RIGID_REVIEW_PROFILE.comparison_simulation_hz,
        ):
            result = backend.run(
                replace(scenario, simulation_hz=simulation_hz), render=False
            )
            event_time_s = float(result.outcome["key_event_time_s"])
            event_row = min(
                result.high_rate_rows,
                key=lambda row: abs(float(row["timestamp"]) - event_time_s),
            )
            observations.append(
                {
                    "outcome": result.outcome["actual_outcome"],
                    "task_success": result.outcome["task_success"],
                    "physics_qc_pass": result.physics_qc["physics_qc_pass"],
                    "saved_artifact_objective_replay_matches": result.outcome[
                        "saved_artifact_objective_replay_matches"
                    ],
                    "key_event_time_s": event_time_s,
                    "key_event_position_m": event_row["object.position"],
                }
            )
        assert timestep_comparison_failures(*observations) == ()
