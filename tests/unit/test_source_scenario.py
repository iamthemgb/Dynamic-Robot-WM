from __future__ import annotations

from dataclasses import replace

import pytest

from dynamic_robot_dataset.common.corpus_registry import (
    BackendNotReleasedError,
    load_backend_capability_registry,
)
from dynamic_robot_dataset.common.source_scenario import (
    ActuatorPhaseSpec,
    CounterfactualIdentity,
    EmbodimentSpec,
    FixtureSpec,
    PoseSpec,
    RNGSubseeds,
    RoboCasaAssetManifest,
    RoboCasaAssetSpec,
    SourceCameraSpec,
    SourceScenarioSpec,
    SourceScenarioValidationError,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
PANDA_ACTIONS = tuple(
    [*(f"panda_joint_{index}_ctrl" for index in range(1, 8)), "panda_finger_ctrl"]
)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> PoseSpec:
    return PoseSpec(position_m=(x, y, z))


def _valid_spec() -> SourceScenarioSpec:
    return SourceScenarioSpec(
        scenario_id="F1a-review-000",
        corpus_leaf_id="F1a",
        task_variant="catch_retain",
        backend="source_mujoco",
        duration_s=2.0,
        physics={
            "gravity_m_s2": [0.0, 0.0, -9.81],
            "simulation_hz": 1200,
            "contact_profile": "rigid_contact_candidate_v2",
        },
        initial_state={
            "object_position_m": [0.5, 0.0, 1.2],
            "object_linear_velocity_m_s": [0.0, 0.0, 0.0],
            "robot_joint_position_rad": [0.0] * 7,
        },
        embodiment=EmbodimentSpec(
            end_effector="franka_hand",
            robot_model="franka_panda",
            action_names=PANDA_ACTIONS,
        ),
        fixtures=(
            FixtureSpec(
                fixture_id="task_table",
                fixture_type="supported_table",
                pose=_pose(z=0.4),
                parameters={"half_extents_m": [0.7, 0.7, 0.04]},
            ),
        ),
        actuator_phases=(
            ActuatorPhaseSpec(
                name="intercept",
                start_s=0.0,
                end_s=1.0,
                commands={name: 0.0 for name in PANDA_ACTIONS},
            ),
            ActuatorPhaseSpec(
                name="close_and_hold",
                start_s=1.0,
                end_s=2.0,
                commands={**{name: 0.0 for name in PANDA_ACTIONS[:-1]}, PANDA_ACTIONS[-1]: 0.5},
            ),
        ),
        cameras=(
            SourceCameraSpec(
                name="main",
                role="main_three_quarter",
                pose=_pose(2.0, -2.0, 1.6),
                look_at_m=(0.5, 0.0, 0.7),
            ),
            SourceCameraSpec(
                name="secondary",
                role="secondary_side",
                pose=_pose(0.5, 2.2, 1.3),
                look_at_m=(0.5, 0.0, 0.7),
            ),
        ),
        rng_subseeds=RNGSubseeds(
            physics=11,
            initial_state=12,
            camera=13,
            assets=14,
            controller=15,
            scene_construction=16,
        ),
        robocasa_manifest=RoboCasaAssetManifest(
            catalog_id="robocasa_visual_assets",
            catalog_version="v1",
            asset_root_id="robocasa_read_only",
            catalog_sha256=HASH_A,
            license_manifest_sha256=HASH_B,
            assets=(),
        ),
        counterfactual=CounterfactualIdentity(
            bundle_id="bundle-F1a-000",
            split_group_id="split-F1a-000",
            branch_id="success",
            sibling_index=0,
        ),
        source_hashes={
            **load_backend_capability_registry().by_name["source_mujoco"].source_hashes,
            "source_generator": HASH_C,
            "robocasa_catalog": HASH_A,
        },
    )


def test_source_scenario_round_trips_with_stable_hash() -> None:
    spec = _valid_spec()
    spec.validate()

    restored = SourceScenarioSpec.from_json(spec.to_json())

    assert restored == spec
    assert restored.to_dict() == spec.to_dict()
    assert restored.spec_hash == spec.spec_hash
    assert len(restored.embodiment.action_names) == 8


def test_source_scenario_rejects_backend_pin_substitution() -> None:
    spec = _valid_spec()
    changed = dict(spec.source_hashes)
    changed["external_dependency_manifest"] = HASH_B
    with pytest.raises(SourceScenarioValidationError, match="capability pins"):
        replace(spec, source_hashes=changed).validate()


def test_scenario_uses_registry_to_reject_unsupported_combination() -> None:
    spec = _valid_spec()
    robotiq = replace(
        spec.embodiment,
        end_effector="robotiq_2f85_thick_pad",
        robot_model="franka_panda_nohand_plus_robotiq_2f85",
    )
    invalid = replace(
        spec,
        corpus_leaf_id="D1",
        task_variant="poke_cloth",
        backend="source_mujoco_deformable",
        embodiment=robotiq,
    )

    with pytest.raises(ValueError, match="does not support embodiment"):
        invalid.validate()


def test_canonical_actions_reject_undeclared_kinematic_joint_targets() -> None:
    spec = _valid_spec()
    invalid_phase = replace(
        spec.actuator_phases[0],
        commands={**spec.actuator_phases[0].commands, "desired_gripper_joint_08": 0.1},
    )

    with pytest.raises(SourceScenarioValidationError, match="undeclared actuators"):
        replace(spec, actuator_phases=(invalid_phase, spec.actuator_phases[1])).validate()


def test_actuator_phase_requires_an_explicit_command_for_every_actuator() -> None:
    spec = _valid_spec()
    incomplete = replace(
        spec.actuator_phases[0],
        commands={name: 0.0 for name in PANDA_ACTIONS[:-1]},
    )

    with pytest.raises(SourceScenarioValidationError, match="omits actuator commands"):
        replace(spec, actuator_phases=(incomplete, spec.actuator_phases[1])).validate()


def test_embodiment_requires_exact_applied_actuator_layout_and_semantics() -> None:
    spec = _valid_spec()

    with pytest.raises(SourceScenarioValidationError, match="exact seven Panda"):
        replace(
            spec,
            embodiment=replace(
                spec.embodiment,
                action_names=(*PANDA_ACTIONS[:-1], "desired_finger_joint_position"),
            ),
        ).validate()

    with pytest.raises(SourceScenarioValidationError, match="actual_actuator_command/v1"):
        replace(
            spec,
            embodiment=replace(
                spec.embodiment,
                action_semantics="desired_joint_position/v1",
            ),
        ).validate()


def test_task_fixtures_must_be_physical_and_supported() -> None:
    spec = _valid_spec()
    floating = replace(spec.fixtures[0], anchored=False)

    with pytest.raises(SourceScenarioValidationError, match="physical and anchored"):
        replace(spec, fixtures=(floating,)).validate()


def test_robocasa_background_collision_is_rejected() -> None:
    spec = _valid_spec()
    asset = RoboCasaAssetSpec(
        asset_id="cabinet-001",
        asset_type="cabinet",
        source_xml="fixtures/cabinets/cabinet.xml",
        xml_sha256=HASH_C,
        mesh_sha256={"meshes/cabinet.obj": HASH_A},
        texture_sha256={"textures/wood.png": HASH_B},
        scale_xyz=(1.0, 1.0, 1.0),
        aabb_min_m=(-0.5, -0.3, 0.0),
        aabb_max_m=(0.5, 0.3, 1.0),
        pose=_pose(1.5, 0.0, 0.0),
        collision_enabled=True,
    )
    manifest = replace(spec.robocasa_manifest, assets=(asset,))

    with pytest.raises(SourceScenarioValidationError, match="visual-only"):
        replace(spec, robocasa_manifest=manifest).validate()


def test_rng_streams_must_be_independent() -> None:
    spec = _valid_spec()
    duplicated = replace(spec.rng_subseeds, assets=spec.rng_subseeds.camera)

    with pytest.raises(SourceScenarioValidationError, match="must be independent"):
        replace(spec, rng_subseeds=duplicated).validate()


def test_robocasa_catalog_hash_is_bound_into_source_hashes() -> None:
    spec = _valid_spec()

    with pytest.raises(SourceScenarioValidationError, match="must bind"):
        replace(
            spec,
            source_hashes={**spec.source_hashes, "robocasa_catalog": HASH_B},
        ).validate()


def test_blocked_leaf_cannot_validate_for_release() -> None:
    with pytest.raises(BackendNotReleasedError, match="F1a is blocked"):
        _valid_spec().validate(require_released=True)
