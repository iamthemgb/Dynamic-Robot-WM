from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from dynamic_robot_dataset.backends.actuator_only import (
    ACTION_FIELD,
    ACTION_SEMANTICS,
    ActuatorOnlyViolation,
    ActuatorTrajectoryGuard,
    ActuatorTrajectoryLimits,
    ControlObservation,
    action_spec,
    apply_actuator_only_callback,
    validate_action_rows,
)
from dynamic_robot_dataset.common.assets import (
    AxisAlignedBoundingBox,
    inspect_robocasa_background_asset,
    load_robocasa_catalog_policy,
    validate_robocasa_asset_manifest,
)
from dynamic_robot_dataset.common.embodiments import (
    FRANKA_HAND,
    ROBOTIQ_2F85_THICK_PAD,
)
from dynamic_robot_dataset.common.physics_contract import (
    STRICT_RIGID_THRESHOLDS,
    deformable_release_failures,
    fluid_release_failures,
    rigid_task_evidence_failures,
    strict_contact_penetration_check,
    strict_persisted_physics_failures,
    strict_runtime_audit_failures,
    timestep_halving_failures,
)
from dynamic_robot_dataset.common.qc import DatasetQCReport, EpisodeQC
from dynamic_robot_dataset.common.randomization import (
    RandomizationCatalog,
    RandomizationPlanner,
    assert_bundle_randomization_invariant,
    load_randomization_policy,
    scene_attempt_seed,
    validate_randomization_admission,
    validate_scene_attempt_history,
)
from dynamic_robot_dataset.common.review import (
    HumanReview,
    HumanReviewLedger,
    REVIEW_CHECKS,
    REVIEW_SCENE_SEQUENCE,
    ReviewArtifactManifest,
    ReviewItem,
    bind_review_artifacts,
    event_strip_frame_indices,
)


ROOT = Path(__file__).resolve().parents[2]
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _observation() -> ControlObservation:
    return ControlObservation(
        timestamp_s=0.1,
        robot_joint_position=(0.0,) * 7,
        robot_joint_velocity=(0.0,) * 7,
        object_position_m=(0.0, 0.0, 0.5),
        object_linear_velocity_m_s=(0.0, 0.0, -0.2),
    )


def _trajectory_guard() -> ActuatorTrajectoryGuard:
    return ActuatorTrajectoryGuard(
        FRANKA_HAND,
        ActuatorTrajectoryLimits(
            maximum_velocity_per_s=(100.0,) * 8,
            maximum_acceleration_per_s2=(1000.0,) * 8,
            maximum_jerk_per_s3=(10000.0,) * 8,
        ),
    )


def _fake_model_data() -> tuple[SimpleNamespace, SimpleNamespace]:
    model = SimpleNamespace(
        body_mass=np.array([1.0]),
        body_inertia=np.ones((1, 3)),
        dof_damping=np.zeros(8),
        geom_friction=np.ones((2, 3)),
        geom_solref=np.ones((2, 2)),
        geom_solimp=np.ones((2, 5)),
        geom_contype=np.ones(2, dtype=np.int32),
        geom_conaffinity=np.ones(2, dtype=np.int32),
        actuator_ctrlrange=np.tile([-2.0, 2.0], (8, 1)),
        actuator_ctrllimited=np.ones(8, dtype=np.int32),
        actuator_forcerange=np.tile([-100.0, 100.0], (8, 1)),
        eq_type=np.array([0], dtype=np.int32),
        eq_obj1id=np.array([1], dtype=np.int32),
        eq_obj2id=np.array([2], dtype=np.int32),
        eq_data=np.zeros((1, 11)),
        opt=SimpleNamespace(
            gravity=np.array([0.0, 0.0, -9.81]),
            timestep=1.0 / 1200.0,
            integrator=0,
            solver=2,
            iterations=100,
        ),
    )
    data = SimpleNamespace(
        qpos=np.zeros(15),
        qvel=np.zeros(14),
        mocap_pos=np.zeros((1, 3)),
        mocap_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        xfrc_applied=np.zeros((3, 6)),
        qfrc_applied=np.zeros(14),
        eq_active=np.zeros(1, dtype=np.int32),
        ctrl=np.zeros(8),
    )
    return model, data


def test_both_real_grippers_use_eight_actual_actuator_commands() -> None:
    for embodiment in (FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD):
        spec = action_spec(embodiment)
        assert spec.width == 8
        assert spec.semantics == ACTION_SEMANTICS
        assert spec.field_name == ACTION_FIELD
    assert action_spec(FRANKA_HAND).actuator_names[-1] == "panda_finger_ctrl"
    assert action_spec(ROBOTIQ_2F85_THICK_PAD).actuator_names[-1] == "robotiq_tendon_ctrl"


def test_action_rows_reject_desired_kinematic_targets_and_wrong_width() -> None:
    problems = validate_action_rows(
        [
            {
                ACTION_FIELD: [0.0] * 7,
                "action.command.joint_position": [0.0] * 7,
            }
        ],
        embodiment=FRANKA_HAND,
        action_semantics="desired_joint_target",
    )
    assert any("semantics" in value for value in problems)
    assert any("desired kinematic targets" in value for value in problems)
    assert any("shape (8,)" in value for value in problems)


def test_actuator_only_callback_applies_and_returns_exact_command() -> None:
    model, data = _fake_model_data()
    command = [0.1 * index for index in range(8)]
    applied = apply_actuator_only_callback(
        lambda _observation: command,
        _observation(),
        embodiment=FRANKA_HAND,
        model=model,
        data=data,
        actuator_ids=tuple(range(8)),
        trajectory_guard=_trajectory_guard(),
    )
    assert applied.tolist() == pytest.approx(command)
    assert data.ctrl.tolist() == pytest.approx(command)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, data: data.qpos.__setitem__(0, 1.0),
        lambda model, data: data.qvel.__setitem__(0, 1.0),
        lambda model, data: data.mocap_pos.__setitem__((0, 0), 1.0),
        lambda model, data: data.eq_active.__setitem__(0, 1),
        lambda model, data: model.geom_friction.__setitem__((0, 0), 9.0),
        lambda model, data: setattr(model.opt, "timestep", 0.1),
    ),
)
def test_actuator_only_callback_rejects_hidden_state_or_model_mutation(mutation) -> None:
    model, data = _fake_model_data()

    def callback(_observation):
        mutation(model, data)
        return [0.0] * 8

    with pytest.raises(ActuatorOnlyViolation, match="forbidden simulator state"):
        apply_actuator_only_callback(
            callback,
            _observation(),
            embodiment=FRANKA_HAND,
            model=model,
            data=data,
            actuator_ids=tuple(range(8)),
            trajectory_guard=_trajectory_guard(),
        )


def test_actuator_only_callback_rejects_callback_side_ctrl_mutation() -> None:
    model, data = _fake_model_data()
    observation = _observation()

    def callback(_observation):
        data.ctrl[3] = 0.25
        return np.zeros(8)

    with pytest.raises(ActuatorOnlyViolation, match="data.ctrl"):
        apply_actuator_only_callback(
            callback,
            observation,
            embodiment=FRANKA_HAND,
            model=model,
            data=data,
            actuator_ids=tuple(range(8)),
            trajectory_guard=_trajectory_guard(),
        )


def test_actuator_only_callback_rejects_preexisting_forces_and_object_welds() -> None:
    model, data = _fake_model_data()
    data.xfrc_applied[1, 2] = 2.0
    with pytest.raises(ActuatorOnlyViolation, match="applied forces"):
        apply_actuator_only_callback(
            lambda _observation: [0.0] * 8,
            _observation(),
            embodiment=FRANKA_HAND,
            model=model,
            data=data,
            actuator_ids=tuple(range(8)),
            trajectory_guard=_trajectory_guard(),
        )
    data.xfrc_applied[:] = 0.0
    data.eq_active[0] = 1
    with pytest.raises(ActuatorOnlyViolation, match="equality/weld/latch"):
        apply_actuator_only_callback(
            lambda _observation: [0.0] * 8,
            _observation(),
            embodiment=FRANKA_HAND,
            model=model,
            data=data,
            actuator_ids=tuple(range(8)),
            trajectory_guard=_trajectory_guard(),
            forbidden_equality_ids=(0,),
        )


def test_actuator_only_callback_rejects_out_of_range_command_without_clipping() -> None:
    model, data = _fake_model_data()
    with pytest.raises(ActuatorOnlyViolation, match="outside"):
        apply_actuator_only_callback(
            lambda _observation: [3.0, *([0.0] * 7)],
            _observation(),
            embodiment=FRANKA_HAND,
            model=model,
            data=data,
            actuator_ids=tuple(range(8)),
            trajectory_guard=_trajectory_guard(),
        )
    assert np.all(data.ctrl == 0.0)


def test_trajectory_guard_rejects_instantaneous_gripper_closure() -> None:
    limits = ActuatorTrajectoryLimits(
        maximum_velocity_per_s=(1.0,) * 8,
        maximum_acceleration_per_s2=(10.0,) * 8,
        maximum_jerk_per_s3=(100.0,) * 8,
    )
    guard = ActuatorTrajectoryGuard(FRANKA_HAND, limits)
    guard.validate_next(0.0, [0.0] * 8)
    with pytest.raises(ActuatorOnlyViolation, match="velocity/closure"):
        guard.validate_next(0.01, [*([0.0] * 7), 1.0])


def test_strict_penetration_uses_separate_gripper_and_surface_limits() -> None:
    passing = strict_contact_penetration_check(
        [
            {
                "object_a": "ball",
                "object_b": "left_fingertip_pad",
                "penetration_depth_m": 0.002,
            },
            {
                "object_a": "ball",
                "object_b": "ramp_surface",
                "penetration_depth_m": 0.003,
            },
        ]
    )
    assert passing.passed
    gripper_failure = strict_contact_penetration_check(
        [
            {
                "contact_category": "gripper",
                "penetration_depth_m": 0.002001,
            }
        ]
    )
    surface_failure = strict_contact_penetration_check(
        [
            {
                "contact_category": "task_surface",
                "penetration_depth_m": 0.003001,
            }
        ]
    )
    assert any("object-gripper" in value for value in gripper_failure.failures)
    assert any("task-surface" in value for value in surface_failure.failures)


def test_strict_penetration_rejects_background_and_unclassified_contacts() -> None:
    result = strict_contact_penetration_check(
        [
            {"contact_category": "background", "penetration_depth_m": 0.0},
            {"object_a": "ball", "object_b": "mystery", "penetration_depth_m": 0.0},
        ]
    )
    assert not result.passed
    assert any("visual-only background" in value for value in result.failures)
    assert any("lack gripper/task-surface classification" in value for value in result.failures)

    missing_depth = strict_contact_penetration_check(
        [{"contact_category": "gripper"}]
    )
    assert any("lacks measured penetration" in value for value in missing_depth.failures)


def _strict_physics_qc(*, energy: float = 0.05, restitution: float = 1.05) -> dict:
    return {
        "physics_qc_pass": True,
        "checks": {
            "finite_state": True,
            "no_solver_warnings": True,
            "no_tunneling": True,
            "no_unexplained_velocity_discontinuity": True,
            "no_mutation_boundary_violation": True,
            "no_applied_forces": True,
            "no_object_linked_equality_or_latch_assistance": True,
            "actuator_forces_within_model_limits": True,
            "joint_motion_within_model_limits": True,
            "free_flight_energy_applicable": True,
            "static_restitution_applicable": True,
            "free_flight_relative_energy_drift": energy,
            "maximum_measured_effective_restitution": restitution,
        },
    }


def test_strict_physics_limits_energy_and_restitution_and_fails_on_missing_audit() -> None:
    assert strict_persisted_physics_failures(_strict_physics_qc()) == []
    inapplicable = _strict_physics_qc()
    inapplicable["checks"]["free_flight_energy_applicable"] = False
    inapplicable["checks"]["static_restitution_applicable"] = False
    del inapplicable["checks"]["free_flight_relative_energy_drift"]
    del inapplicable["checks"]["maximum_measured_effective_restitution"]
    assert strict_persisted_physics_failures(inapplicable) == []
    failures = strict_persisted_physics_failures(
        _strict_physics_qc(energy=0.05001, restitution=1.05001)
    )
    assert any("energy drift" in value for value in failures)
    assert any("restitution" in value for value in failures)
    negative_failures = strict_persisted_physics_failures(
        _strict_physics_qc(energy=-0.05001, restitution=-0.01)
    )
    assert any("energy drift" in value for value in negative_failures)
    assert any("restitution is negative" in value for value in negative_failures)
    assert len(strict_runtime_audit_failures({})) > 10


def test_passive_runtime_audit_does_not_require_actuator_updates() -> None:
    audit = {
        "object_state_writes_after_initialization": 0,
        "direct_robot_state_writes_after_initialization": 0,
        "mocap_writes_after_initialization": 0,
        "applied_force_writes_after_initialization": 0,
        "object_linked_equality_changes_after_initialization": 0,
        "model_physics_mutations_after_initialization": 0,
        "mutation_boundary_violations": 0,
        "solver_warning_count": 0,
        "non_finite_state_count": 0,
        "tunneling_event_count": 0,
        "unexplained_velocity_discontinuity_count": 0,
        "control_updates": 0,
    }
    assert strict_runtime_audit_failures(
        audit, require_control_updates=False
    ) == []


def test_timestep_halving_requires_same_outcome_event_frame_and_centimeter() -> None:
    coarse = {
        "outcome": "success",
        "key_event_time_s": 0.5,
        "key_event_position_m": [0.0, 0.0, 0.5],
    }
    fine = {
        "outcome": "success",
        "key_event_time_s": 0.5 + 1.0 / 30.0,
        "key_event_position_m": [0.01, 0.0, 0.5],
    }
    assert timestep_halving_failures(coarse, fine) == []
    bad = dict(fine, outcome="failure", key_event_time_s=0.55)
    bad["key_event_position_m"] = [0.02, 0.0, 0.5]
    failures = timestep_halving_failures(coarse, bad)
    assert len(failures) == 3


def test_family_specific_rigid_evidence_is_fail_closed() -> None:
    catch_required = {
        "sustained_opposing_bilateral_contacts": True,
        "stable_object_to_grasp_transform": True,
        "displacement_physically_supported_by_contacts": True,
    }
    assert rigid_task_evidence_failures(
        family="falling_catch",
        subfamily="centered_vertical_drop",
        task_variant="catch_transport",
        evidence=catch_required,
    ) == []
    missing = rigid_task_evidence_failures(
        family="falling_catch",
        subfamily="centered_vertical_drop",
        task_variant="catch_transport",
        evidence={},
    )
    assert len(missing) == 3
    negative_evidence = {
        "measured_failure_matches_persisted_label": True,
        "saved_artifact_objective_replay_matches": True,
    }
    assert rigid_task_evidence_failures(
        family="falling_catch",
        subfamily="centered_vertical_drop",
        task_variant="catch_transport",
        evidence=negative_evidence,
        task_success=False,
    ) == []
    deflection = rigid_task_evidence_failures(
        family="projectile_rebound",
        subfamily="direct_interception",
        task_variant="direct_deflection",
        evidence={},
    )
    assert deflection == [
        "rigid task evidence is absent or false: velocity_change_matches_measured_contact_impulse"
    ]


def test_randomization_policy_matches_config_and_independent_streams() -> None:
    policy = load_randomization_policy(ROOT / "configs/randomization/default.yaml")
    assert policy.varies("R0", "lighting") is False
    assert policy.varies("R1", "lighting") is True
    assert policy.varies("R1", "camera_pose") is False
    assert policy.varies("R2", "camera_pose") is True
    planner = RandomizationPlanner(seed=13, policy=policy)
    r0 = planner.plan("bundle", randomization_level="R0")
    assert r0.background_style == "clean_franka_lab"
    assert r0.varied_fields == ()
    assert len(set(r0.rng_subseeds.values())) == len(r0.rng_subseeds)
    assert planner.plan("bundle", randomization_level="R1") == planner.plan(
        "bundle", randomization_level="R1"
    )


def test_randomization_admission_requires_bound_robocasa_asset_and_r1_gate() -> None:
    policy = load_randomization_policy(ROOT / "configs/randomization/default.yaml")
    catalog = RandomizationCatalog(
        scene_asset_ids=("cabinet-01",),
        scene_asset_manifest_sha256={"cabinet-01": HEX_A},
    )
    planner = RandomizationPlanner(seed=1, catalog=catalog, policy=policy)
    value = planner.plan("bundle", randomization_level="R2").to_dict()
    value["background_style"] = "robocasa_lab"
    with pytest.raises(ValueError, match="before.*R1"):
        validate_randomization_admission(value, r1_accepted=False)
    validate_randomization_admission(value, r1_accepted=True)
    value["scene_asset_manifest_sha256"] = None
    with pytest.raises(ValueError, match="manifest hash"):
        validate_randomization_admission(value, r1_accepted=True)


def test_counterfactual_randomization_includes_rng_and_asset_hash_invariants() -> None:
    base = RandomizationPlanner(seed=2).plan("bundle", randomization_level="R1").to_dict()
    records = [
        {"counterfactual_bundle_id": "bundle", "randomization": dict(base)},
        {"counterfactual_bundle_id": "bundle", "randomization": dict(base)},
    ]
    assert_bundle_randomization_invariant(records)
    records[1]["randomization"]["rng_subseeds"] = {
        **base["rng_subseeds"],
        "camera": 999,
    }
    with pytest.raises(ValueError, match="Visual randomization changed"):
        assert_bundle_randomization_invariant(records)


def test_scene_resampling_allows_construction_failures_only() -> None:
    attempts = [
        {
            "attempt_index": 0,
            "attempt_seed": scene_attempt_seed("bundle", 0),
            "status": "construction_failed",
            "failure_code": "swept_volume_intersection",
        },
        {
            "attempt_index": 1,
            "attempt_seed": scene_attempt_seed("bundle", 1),
            "status": "accepted",
        },
    ]
    validate_scene_attempt_history(attempts)
    attempts[0]["failure_code"] = "outcome_mismatch"
    with pytest.raises(ValueError, match="outcome failures"):
        validate_scene_attempt_history(attempts)


def test_robocasa_asset_admission_hashes_license_and_enforces_clearance(tmp_path: Path) -> None:
    descriptor = tmp_path / "asset.xml"
    mesh = tmp_path / "mesh.stl"
    license_path = tmp_path / "LICENSE"
    descriptor.write_text("<mujoco/>", encoding="utf-8")
    mesh.write_bytes(b"mesh")
    license_path.write_text("test license", encoding="utf-8")
    asset_bounds = AxisAlignedBoundingBox((2.0, 2.0, 0.0), (2.4, 2.4, 1.0))
    task_bounds = AxisAlignedBoundingBox((-1.0, -1.0, -0.2), (1.0, 1.0, 2.0))
    admitted = inspect_robocasa_background_asset(
        "asset-1",
        descriptor,
        source_root=tmp_path,
        catalog_version="robocasa-v1",
        referenced_files=(mesh,),
        license_notice_path=license_path,
        scale_to_meters=1.0,
        world_aabb=asset_bounds,
        transform_row_major_4x4=(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        collision_enabled=False,
        visual_only=True,
        task_swept_volume=task_bounds,
        occlusion_validated=True,
    )
    assert admitted.admitted
    assert descriptor.name in admitted.referenced_file_sha256
    assert mesh.name in admitted.referenced_file_sha256
    assert license_path.name in admitted.referenced_file_sha256
    validate_robocasa_asset_manifest([admitted], required_asset_ids=("asset-1",))
    colliding = inspect_robocasa_background_asset(
        "asset-2",
        descriptor,
        source_root=tmp_path,
        catalog_version="robocasa-v1",
        referenced_files=(mesh,),
        license_notice_path=license_path,
        scale_to_meters=1.0,
        world_aabb=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.2, 0.2, 0.2)),
        transform_row_major_4x4=(1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
        collision_enabled=True,
        visual_only=False,
        task_swept_volume=task_bounds,
        occlusion_validated=False,
    )
    assert not colliding.admitted
    assert "background_collision_not_disabled" in colliding.blockers
    assert "background_intersects_task_swept_volume" in colliding.blockers


def test_versioned_robocasa_catalog_binds_real_read_only_root_but_stays_blocked() -> None:
    policy = load_robocasa_catalog_policy(
        ROOT / "configs/assets/robocasa_catalog_v1.yaml"
    )
    assert policy.catalog_version == "robocasa-backgrounds/v1"
    assert policy.configured_asset_count == 4
    assert policy.review_candidate_asset_count == 4
    assert policy.admitted_asset_count == 0
    assert policy.review_ready
    assert not policy.release_ready
    assert set(policy.review_ready_profiles) == {
        "lab",
        "kitchen",
        "workbench",
        "storage",
        "tabletop",
    }
    assert policy.release_ready_profiles == ()
    assert all(asset.visual_only for asset in policy.assets)
    assert not any(asset.collision_enabled for asset in policy.assets)
    assert all(not asset.admitted for asset in policy.assets)
    assert all(
        set(asset.blockers)
        == {
            "scene_specific_clearance_pending",
            "fixture_intersection_clearance_pending",
            "rendered_occlusion_review_pending",
        }
        for asset in policy.assets
    )
    assert len(policy.catalog_sha256) == 64
    assert len(policy.license_notice_sha256) == 64


def _review_artifact(leaf_id: str, index: int) -> ReviewArtifactManifest:
    return ReviewArtifactManifest(
        leaf_id=leaf_id,
        episode_uuid=f"00000000-0000-4000-8000-{index:012d}",
        rollout_index=index,
        fixed_master_seed=20260717,
        review_plan_sha256=HEX_A,
        review_case_sha256=HEX_B,
        review_request_ledger_sha256=HEX_C,
        review_request_sha256=HEX_A,
        scenario_spec_sha256=HEX_A,
        qc_report_sha256=HEX_B,
        qc_report_schema="dynamic-robot-qc-report/v2",
        qc_episode_result_sha256=HEX_C,
        qc_strict_all=True,
        automated_qc_passed=True,
        source_manifest_sha256=HEX_C,
        frame_timestamps_sha256=HEX_A,
        key_event_time_s=0.2,
        video_paths={
            "main": f"videos/main/{index}.mp4",
            "secondary": f"videos/secondary/{index}.mp4",
        },
        video_sha256={"main": HEX_A, "secondary": HEX_B},
        event_strip_paths={
            "main": f"review/main/{index}.png",
            "secondary": f"review/secondary/{index}.png",
        },
        event_strip_sha256={"main": HEX_B, "secondary": HEX_C},
        event_strip_indices={
            "pre_event": 3,
            "event": 6,
            "post_0p1_s": 9,
            "post_0p3_s": 15,
            "final": 29,
        },
    )


def test_event_strip_is_event_aligned_and_includes_final() -> None:
    indices = event_strip_frame_indices([index / 30 for index in range(31)], 0.5)
    assert indices == {
        "pre_event": 12,
        "event": 15,
        "post_0p1_s": 18,
        "post_0p3_s": 24,
        "final": 30,
    }


def test_review_artifact_binding_derives_strict_qc_and_event_indices(
    tmp_path: Path,
) -> None:
    episode_uuid = "00000000-0000-4000-8000-000000000000"
    for relative in (
        "videos/main.mp4",
        "videos/secondary.mp4",
        "review/main.png",
        "review/secondary.png",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    qc = tmp_path / "qc/dataset_report.json"
    qc.parent.mkdir(parents=True)
    qc.write_text(
        json.dumps(
            {
                "schema_version": "dynamic-robot-qc-report/v2",
                "strict_all": True,
                "passed": True,
                "episodes": [{"episode_uuid": episode_uuid, "passed": True}],
            }
        ),
        encoding="utf-8",
    )
    binding_arguments = {
        "leaf_id": "F1a",
        "episode_uuid": episode_uuid,
        "rollout_index": 0,
        "review_plan_sha256": HEX_A,
        "review_case_sha256": HEX_B,
        "review_request_ledger_sha256": HEX_C,
        "review_request_sha256": HEX_A,
        "scenario_spec_sha256": HEX_B,
        "qc_report_path": "qc/dataset_report.json",
        "source_manifest_sha256": HEX_C,
        "video_paths": {
            "main": "videos/main.mp4",
            "secondary": "videos/secondary.mp4",
        },
        "event_strip_paths": {
            "main": "review/main.png",
            "secondary": "review/secondary.png",
        },
        "frame_timestamps_s": [index / 30 for index in range(31)],
        "key_event_time_s": 0.5,
    }
    with pytest.raises(ValueError, match="explicit acknowledgement"):
        bind_review_artifacts(tmp_path, **binding_arguments)
    artifact = bind_review_artifacts(
        tmp_path,
        **binding_arguments,
        unsafe_compatibility_acknowledged=True,
    )
    assert artifact.automated_qc_passed
    assert artifact.event_strip_indices["event"] == 15
    assert artifact.event_strip_indices["final"] == 30

    value = json.loads(qc.read_text(encoding="utf-8"))
    value["strict_all"] = False
    qc.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="strict_all"):
        bind_review_artifacts(
            tmp_path,
            **binding_arguments,
            unsafe_compatibility_acknowledged=True,
        )


def test_six_rollout_review_ledger_is_hash_bound_and_blocks_any_failure() -> None:
    items: list[ReviewItem] = []
    for index, scene in enumerate(REVIEW_SCENE_SEQUENCE):
        artifact = _review_artifact("F1a", index)
        review = HumanReview(
            artifact_binding_sha256=artifact.binding_sha256,
            reviewer="reviewer@example.org",
            reviewed_at="2026-07-17T12:00:00+00:00",
            checks={name: True for name in REVIEW_CHECKS},
        )
        items.append(ReviewItem(index, scene, artifact, True, review))
    ledger = HumanReviewLedger(tuple(items), HEX_A, HEX_C)
    ledger.validate(required_leaf_ids=("F1a",), require_complete_reviews=True)
    assert ledger.activation_failures("F1a") == []
    assert len(ledger.ledger_sha256) == 64
    failed_checks = dict(items[2].review.checks)
    failed_checks["penetration"] = False
    failed_review = replace(items[2].review, checks=failed_checks)
    failed_item = replace(items[2], review=failed_review)
    failed = HumanReviewLedger(
        tuple([*items[:2], failed_item, *items[3:]]), HEX_A, HEX_C
    )
    assert failed.activation_failures("F1a") == ["rollout 2 failed human review"]


def test_review_rejects_signature_for_different_artifacts() -> None:
    artifact = _review_artifact("F1a", 0)
    review = HumanReview(
        artifact_binding_sha256=HEX_A,
        reviewer="reviewer",
        reviewed_at="2026-07-17T12:00:00+00:00",
        checks={name: True for name in REVIEW_CHECKS},
    )
    with pytest.raises(ValueError, match="does not match"):
        ReviewItem(0, "clean_R0", artifact, True, review).validate()


def test_strict_all_report_counts_nonrelease_failures() -> None:
    failed_nonrelease = EpisodeQC("episode", 0, False)
    failed_nonrelease.fail("physics failure")
    assert DatasetQCReport("dataset", [failed_nonrelease]).passed
    report = DatasetQCReport("dataset", [failed_nonrelease], strict_all=True)
    assert not report.passed
    assert report.to_dict()["failed_episode_count"] == 1


def test_unfinished_deformable_and_fluid_backends_fail_closed() -> None:
    assert len(deformable_release_failures({})) == 8
    assert len(fluid_release_failures({})) == 8
    deformable = {
        name.split(": ")[-1]: True
        for name in deformable_release_failures({})
    }
    fluid = {name.split(": ")[-1]: True for name in fluid_release_failures({})}
    assert deformable_release_failures(deformable) == []
    assert fluid_release_failures(fluid) == []


def test_review_config_matches_python_contract() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/review/acceptance_v1.yaml").read_text(encoding="utf-8")
    )
    assert tuple(config["scene_sequence"]) == REVIEW_SCENE_SEQUENCE
    assert tuple(config["human_checks"]) == REVIEW_CHECKS
    assert config["rollouts_per_leaf"] == 6
    assert STRICT_RIGID_THRESHOLDS.maximum_gripper_penetration_m == 0.002
    assert STRICT_RIGID_THRESHOLDS.maximum_task_surface_penetration_m == 0.003
    assert STRICT_RIGID_THRESHOLDS.maximum_effective_restitution == 1.05
    assert STRICT_RIGID_THRESHOLDS.maximum_free_flight_energy_drift_fraction == 0.05
