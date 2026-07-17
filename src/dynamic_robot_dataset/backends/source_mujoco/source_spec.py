"""Public bridge from a fixed review case to the canonical source contract."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from ..actuator_only import action_spec
from ...common.embodiments import FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD
from ...common.hashing import combined_manifest_hash, sha256_file
from ...common.source_scenario import (
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
)
from .backend import (
    SourceMujocoBackend,
    _controller_for_scenario,
    _require_runtime_dependencies,
    _robocasa_manifest,
)
from .compiler import SourceMujocoCompiledScenario
from .model import CompiledSourceModel, compile_source_model
from .profiles import RIGID_REVIEW_PROFILE
from .provenance import PINNED_SOURCE_FILES


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_ROBOCASA_CATALOG = _REPOSITORY_ROOT / "configs/assets/robocasa_catalog_v1.yaml"


def _case_value(case: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(case, Mapping):
        return case
    value = case.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("review case to_dict() must return a mapping")
    return value


def _pose_from_matrix(mujoco: Any, position: Any, matrix: Any) -> PoseSpec:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=np.float64).reshape(-1))
    return PoseSpec(
        position_m=tuple(float(item) for item in position),
        quaternion_wxyz=tuple(float(item) for item in quaternion),
    )


def _embodiment(value: str) -> EmbodimentSpec:
    if value == "no_robot":
        return EmbodimentSpec(
            end_effector="no_robot",
            robot_model="none",
            action_names=(),
            action_semantics="no_actuators/v1",
        )
    specification = action_spec(value)
    robot_model = (
        "franka_panda"
        if value == FRANKA_HAND
        else "franka_panda_nohand_plus_robotiq_2f85"
    )
    return EmbodimentSpec(
        end_effector=value,
        robot_model=robot_model,
        action_names=specification.actuator_names,
        action_semantics=specification.semantics,
    )


def _command(names: tuple[str, ...], arm: Any, gripper: float) -> dict[str, float]:
    values = [*(float(item) for item in arm), float(gripper)]
    return dict(zip(names, values, strict=True))


def _actuator_phases(
    mujoco: Any,
    least_squares: Any,
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
    names: tuple[str, ...],
) -> tuple[ActuatorPhaseSpec, ...]:
    if scenario.embodiment == "no_robot":
        return ()
    controller, _, _ = _controller_for_scenario(
        mujoco, least_squares, compiled, scenario
    )
    assert controller is not None
    plan = controller.plan
    phases: list[ActuatorPhaseSpec] = []

    def append(name: str, start: float, end: float, arm: Any, gripper: float, interpolation: str) -> None:
        if end - start <= 1e-12:
            return
        phases.append(
            ActuatorPhaseSpec(
                name=name,
                start_s=start,
                end_s=end,
                commands=_command(names, arm, gripper),
                interpolation=interpolation,
            )
        )

    append(
        "pre_intercept_hold",
        0.0,
        plan.closure_start_s,
        plan.intercept_arm_command,
        plan.open_gripper_command,
        "hold",
    )
    append(
        "bounded_closure",
        plan.closure_start_s,
        plan.closure_end_s,
        plan.intercept_arm_command,
        plan.closed_gripper_command,
        "jerk_limited",
    )
    cursor = plan.closure_end_s
    if plan.transport_arm_command is not None:
        append(
            "retention_hold",
            cursor,
            plan.transport_start_s,
            plan.intercept_arm_command,
            plan.closed_gripper_command,
            "hold",
        )
        append(
            "supported_transport",
            plan.transport_start_s,
            plan.transport_end_s,
            plan.transport_arm_command,
            plan.closed_gripper_command,
            "jerk_limited",
        )
        cursor = plan.transport_end_s
        final_arm = plan.transport_arm_command
    else:
        final_arm = plan.intercept_arm_command
    append(
        "final_retention",
        cursor,
        scenario.duration_s,
        final_arm,
        plan.closed_gripper_command,
        "hold",
    )
    return tuple(phases)


def _fixtures(mujoco: Any, compiled: CompiledSourceModel, scenario: SourceMujocoCompiledScenario) -> tuple[FixtureSpec, ...]:
    result = []
    for surface in scenario.surfaces:
        geom_id = compiled.ids.surface_geom_ids[surface.name]
        pose = _pose_from_matrix(
            mujoco, compiled.data.geom_xpos[geom_id], compiled.data.geom_xmat[geom_id]
        )
        result.append(
            FixtureSpec(
                fixture_id=surface.name,
                fixture_type=surface.role,
                pose=pose,
                parameters={
                    "half_size_m": list(surface.half_size_m),
                    "friction": list(surface.friction),
                    "solref": list(surface.solref),
                    "collision_enabled": True,
                },
            )
        )
    return tuple(result)


def _cameras(mujoco: Any, compiled: CompiledSourceModel) -> tuple[SourceCameraSpec, SourceCameraSpec]:
    result = []
    roles = {"main": "main_three_quarter_external", "secondary": "task_specific_secondary"}
    for name in ("main", "secondary"):
        camera_id = int(
            mujoco.mj_name2id(
                compiled.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                compiled.camera_names[name],
            )
        )
        rotation = np.asarray(compiled.data.cam_xmat[camera_id]).reshape(3, 3)
        position = np.asarray(compiled.data.cam_xpos[camera_id])
        look_at = position - rotation[:, 2]
        result.append(
            SourceCameraSpec(
                name=name,
                role=roles[name],
                pose=_pose_from_matrix(mujoco, position, rotation),
                look_at_m=tuple(float(item) for item in look_at),
                fovy_deg=float(compiled.model.cam_fovy[camera_id]),
            )
        )
    return result[0], result[1]


def _robocasa_assets(
    mujoco: Any,
    backend: SourceMujocoBackend,
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
) -> tuple[RoboCasaAssetSpec, ...]:
    rows = _robocasa_manifest(
        mujoco, compiled, scenario, backend.robocasa_dependency
    )
    assets = []
    for row in rows:
        referenced = dict(row["referenced_file_sha256"])
        textures = {
            path: digest
            for path, digest in referenced.items()
            if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        }
        meshes = {path: digest for path, digest in referenced.items() if path not in textures}
        transform = row["transform_row_major_4x4"]
        yaw = math.atan2(float(transform[4]), float(transform[0]))
        pose = PoseSpec(
            position_m=(float(transform[3]), float(transform[7]), float(transform[11])),
            quaternion_wxyz=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
        )
        bounds = row["world_aabb"]
        assets.append(
            RoboCasaAssetSpec(
                asset_id=str(row["asset_id"]),
                asset_type="visual_background",
                source_xml=str(row["descriptor_path"]),
                xml_sha256=str(row["descriptor_sha256"]),
                mesh_sha256=meshes,
                texture_sha256=textures,
                scale_xyz=(1.0, 1.0, 1.0),
                aabb_min_m=tuple(float(item) for item in bounds["minimum_m"]),
                aabb_max_m=tuple(float(item) for item in bounds["maximum_m"]),
                pose=pose,
                collision_enabled=False,
            )
        )
    return tuple(assets)


def prepare_review_case(
    case: Mapping[str, Any] | Any,
    *,
    backend: SourceMujocoBackend | None = None,
) -> SourceScenarioSpec:
    """Compile and validate the exact immutable scenario used by ``run``."""

    source_backend = backend or SourceMujocoBackend()
    value = _case_value(case)
    scenario = source_backend.compile_case(value)
    mujoco, least_squares = _require_runtime_dependencies()
    compiled = compile_source_model(
        scenario,
        source_dependency=source_backend.source_dependency,
        robocasa_dependency=(
            source_backend.robocasa_dependency
            if scenario.requires_real_robocasa
            else None
        ),
    )
    mujoco.mj_forward(compiled.model, compiled.data)
    catalog_raw = yaml.safe_load(_ROBOCASA_CATALOG.read_text(encoding="utf-8"))
    source_hashes = {
        "external_dependency_manifest": source_backend.source_dependency.manifest_sha256,
        "scene_builder_py": PINNED_SOURCE_FILES["scripts_mujoco/scene_builder.py"],
        "variants_py": PINNED_SOURCE_FILES["scripts_mujoco/variants.py"],
        "robocasa_importer_py": PINNED_SOURCE_FILES["scripts_mujoco/robocasa_assets.py"],
        "panda_xml": PINNED_SOURCE_FILES["third_party/mujoco_menagerie/franka_emika_panda/panda.xml"],
        "panda_nohand_xml": PINNED_SOURCE_FILES["third_party/mujoco_menagerie/franka_emika_panda/panda_nohand.xml"],
        "robotiq_2f85_xml": PINNED_SOURCE_FILES["third_party/mujoco_menagerie/robotiq_2f85/2f85.xml"],
        "robocasa_catalog": sha256_file(_ROBOCASA_CATALOG),
        "compiled_scene_xml": compiled.xml_sha256,
        "compiled_asset_manifest": combined_manifest_hash(compiled.source_asset_sha256),
    }
    embodiment = _embodiment(scenario.embodiment)
    spec = SourceScenarioSpec(
        # ReviewArtifactRequest binds SourceScenarioSpec to the fixed logical
        # case ID.  The storage UUID is a separate run-plan identity and must
        # not replace this scenario identity.
        scenario_id=scenario.case_id,
        corpus_leaf_id=scenario.corpus_leaf_id,
        task_variant=scenario.task_variant,
        backend="source_mujoco",
        duration_s=scenario.duration_s,
        physics={
            "profile": RIGID_REVIEW_PROFILE.profile_id,
            "simulation_hz": scenario.simulation_hz,
            "control_hz": scenario.control_hz,
            "video_hz": scenario.video_hz,
            "gravity_m_s2": list(scenario.gravity_m_s2),
            "object_mass_kg": scenario.object_mass_kg,
            "object_radius_m": scenario.object_radius_m,
            "key_event_time_s": scenario.key_event_time_s,
            "ballistic_event_time_s": scenario.ballistic_event_time_s,
            "scene_profile": scenario.scene_profile,
            "scene_variant": scenario.scene_variant,
            "randomization_level": scenario.randomization_level,
            "requires_real_robocasa": scenario.requires_real_robocasa,
            "passive_variation_profile": scenario.passive_variation_profile,
            "evaluator": scenario.evaluator,
            "release_state": "blocked",
        },
        initial_state={
            "object_position_m": list(scenario.object_initial_position_m),
            "object_linear_velocity_m_s": list(
                scenario.object_initial_linear_velocity_m_s
            ),
            "object_angular_velocity_rad_s": list(
                scenario.object_initial_angular_velocity_rad_s
            ),
            "motion_kind": scenario.motion_kind,
            "physical_target_position_m": scenario.physical_target_position_m,
            "controller_target_position_m": scenario.controller_target_position_m,
            "controller_transport_position_m": (
                scenario.controller_transport_position_m
            ),
            "branch_role": scenario.branch_role,
            "intended_outcome": scenario.intended_outcome,
            "passive_variation_profile": scenario.passive_variation_profile,
        },
        embodiment=embodiment,
        fixtures=_fixtures(mujoco, compiled, scenario),
        actuator_phases=_actuator_phases(
            mujoco,
            least_squares,
            compiled,
            scenario,
            embodiment.action_names,
        ),
        cameras=_cameras(mujoco, compiled),
        rng_subseeds=RNGSubseeds.from_dict(scenario.rng_subseeds),
        robocasa_manifest=RoboCasaAssetManifest(
            catalog_id="robocasa-backgrounds",
            catalog_version=str(catalog_raw["catalog_version"]),
            asset_root_id="mzl7/robocasa/robocasa/models/assets",
            catalog_sha256=source_hashes["robocasa_catalog"],
            license_manifest_sha256=source_backend.robocasa_dependency.license_sha256,
            assets=_robocasa_assets(
                mujoco, source_backend, compiled, scenario
            ),
        ),
        counterfactual=CounterfactualIdentity(
            bundle_id=str(value.get("counterfactual_bundle_id") or scenario.case_id),
            split_group_id=str(value.get("counterfactual_bundle_id") or scenario.case_id),
            branch_id=str(value.get("counterfactual_branch_id") or scenario.branch_role),
            sibling_index=int(value.get("counterfactual_sibling_index", 0)),
            # Every fixed acceptance rollout is a standalone bundle (sibling
            # index zero) with intentionally different scene/physics/state.
            # Grouping cases merely because they share a task variant creates
            # a false counterfactual-family claim and correctly fails global
            # invariant QC.  Real R2 counterfactual siblings will declare a
            # shared family explicitly when that planner is activated.
            physics_family_id=None,
        ),
        source_hashes=source_hashes,
    )
    spec.validate()
    return spec


__all__ = ["prepare_review_case"]
