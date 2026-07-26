"""Canonical orchestration bridge for the owned ``source_mujoco`` backend.

The review-suite planner, source scenario compiler, simulator, and episode
writer deliberately have different contracts.  This module is the only bridge
between them.  In particular, it never treats a requested branch or the
backend's online outcome summary as a label: the canonical label is recomputed
from the state/contact rows that will be persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from ..backends import get_backend
from ..backends.source_mujoco import (
    SourceMujocoBackend,
    SourceMujocoRunResult,
    prepare_review_case,
)
from ..backends.source_mujoco.compiler import SourceMujocoCompiledScenario
from .cameras import (
    CameraCalibration,
    invert_rigid_transform,
    quaternion_to_rotation_matrix,
)
from .contract_v2 import MotionMode, ContactRole
from .corpus_registry import load_corpus_registry
from .episode_writer import canonical_camera_name
from .hashing import canonical_json_bytes, sha256_json
from .randomization import default_randomization_policy
from .review_suite import ReviewSuiteCase
from .run_orchestration import EpisodeMaterialization, RunPlanEpisode
from .schema import (
    ActualOutcomeClass,
    DynamicsMode,
    EpisodeRecord,
    LabelStatus,
    ParameterRangePartition,
    PhysicsMetadata,
    PhysicsValue,
    PhysicsValueKind,
    ReleaseTier,
    default_failure_tags,
)
from .source_evaluators import (
    SOURCE_OBJECTIVE_EVALUATOR_VERSION,
    evaluate_source_rows,
)
from .source_scenario import SourceScenarioSpec
from .visual_qc import source_mujoco_visibility_media_binding


SOURCE_EXECUTION_BRIDGE_SCHEMA = "dynamic-robot-source-execution-bridge/v1"
SOURCE_FINALIZATION_PROVENANCE_SCHEMA = (
    "dynamic-robot-source-finalization-provenance/v3"
)


class SourceExecutionBindingError(RuntimeError):
    """A runtime artifact differs from its immutable source review plan."""


def _canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _require_executable_case(case: ReviewSuiteCase) -> None:
    if not isinstance(case, ReviewSuiteCase):
        raise TypeError("source review planning requires an immutable ReviewSuiteCase")
    if case.backend != "source_mujoco":
        raise ValueError(
            f"review case {case.case_id} uses {case.backend}, not source_mujoco"
        )
    if not case.execution_eligible or case.executable_quota != 1:
        raise ValueError(
            f"review case {case.case_id} is not admitted for execution: "
            + ", ".join(case.execution_blockers)
        )


def _review_backend(case: ReviewSuiteCase) -> SourceMujocoBackend:
    """Resolve an owned backend through the capability gate, never by import alone."""

    backend = get_backend(
        "source_mujoco",
        purpose="review",
        corpus_leaf_id=case.corpus_leaf_id,
        embodiment=case.embodiment,
        task_variant=case.task_variant,
    )
    if not isinstance(backend, SourceMujocoBackend):
        raise TypeError("capability factory returned a non-owned source_mujoco backend")
    return backend


def _task_index(corpus_leaf_id: str) -> int:
    leaves = load_corpus_registry().leaves
    for index, leaf in enumerate(leaves):
        if leaf.corpus_id == corpus_leaf_id:
            return index
    raise ValueError(f"unknown corpus leaf {corpus_leaf_id!r}")


def _metadata_seed(value: int) -> int:
    """Project a uint64 RNG identity into Arrow's non-negative int64 field.

    The complete uint64 streams remain losslessly bound in SourceScenarioSpec;
    these two compatibility columns cannot represent the upper half of uint64.
    """

    return int(value) & ((1 << 63) - 1)


def prepare_source_review_declaration(
    case: ReviewSuiteCase,
    *,
    episode_index: int,
    generator_git_commit: str = "unknown",
) -> dict[str, Any]:
    """Prepare one immutable source declaration for :func:`plan_run`.

    ``episode_index`` is the contiguous index in the selected preview dataset.
    The fixed suite's global index and complete case hash remain embedded under
    ``review_case`` and cannot be changed by this remapping.
    """

    _require_executable_case(case)
    if isinstance(episode_index, bool) or episode_index < 0:
        raise ValueError("episode_index must be a non-negative contiguous plan index")
    backend = _review_backend(case)
    scenario_spec = prepare_review_case(case, backend=backend)
    compiled = backend.compile_case(case)
    scenario_spec.validate()
    compiled.validate()
    if scenario_spec.scenario_id != case.case_id:
        raise SourceExecutionBindingError(
            "prepared SourceScenarioSpec does not preserve the logical review case ID"
        )
    if compiled.episode_uuid != case.episode_uuid:
        raise SourceExecutionBindingError(
            "compiled source scenario does not preserve the review episode UUID"
        )
    runtime_source_hashes = {
        **dict(scenario_spec.source_hashes),
        "robocasa_license": backend.robocasa_dependency.license_sha256,
    }
    review_case = case.to_dict()
    declaration = {
        "schema_version": SOURCE_EXECUTION_BRIDGE_SCHEMA,
        "episode_uuid": case.episode_uuid,
        "episode_index": int(episode_index),
        "review_suite_episode_index": case.episode_index,
        "review_case": review_case,
        "review_case_sha256": case.case_sha256,
        "backend": "source_mujoco",
        "backend_version": backend.version,
        "corpus_leaf_id": case.corpus_leaf_id,
        "family": case.family,
        "subfamily": case.subfamily,
        "task_variant": case.task_variant,
        "variant": case.task_variant,
        "embodiment": case.embodiment,
        "robot_model": scenario_spec.embodiment.robot_model,
        "tool_type": scenario_spec.embodiment.end_effector,
        "duration_s": scenario_spec.duration_s,
        "task_index": _task_index(case.corpus_leaf_id),
        "counterfactual_bundle_id": case.counterfactual_bundle_id,
        "counterfactual_branch_id": case.counterfactual_branch_id,
        "counterfactual_sibling_index": case.counterfactual_sibling_index,
        "physics_counterfactual_family_id": (
            scenario_spec.counterfactual.physics_family_id
        ),
        "split_group_id": scenario_spec.counterfactual.split_group_id,
        "scene_seed": _metadata_seed(
            scenario_spec.rng_subseeds.scene_construction
        ),
        "branch_seed": _metadata_seed(scenario_spec.rng_subseeds.controller),
        "intended_branch": case.branch_role,
        "intended_outcome": case.intended_outcome,
        "generator_git_commit": str(generator_git_commit),
        "source_scenario_spec": scenario_spec.to_dict(),
        "source_scenario_spec_sha256": scenario_spec.spec_hash,
        "source_compiled_scenario": compiled.to_dict(),
        "source_compiled_scenario_sha256": compiled.scenario_sha256,
        "runtime_source_hashes": dict(sorted(runtime_source_hashes.items())),
        "runtime_source_hashes_sha256": sha256_json(runtime_source_hashes),
    }
    return _canonical_copy(declaration)


def _planned_inputs(
    entry: RunPlanEpisode,
) -> tuple[ReviewSuiteCase, SourceScenarioSpec]:
    declaration = entry.declaration
    if declaration.get("schema_version") != SOURCE_EXECUTION_BRIDGE_SCHEMA:
        raise SourceExecutionBindingError("run-plan entry is not a canonical source bridge request")
    if int(declaration.get("episode_index", -1)) != entry.episode_index:
        raise SourceExecutionBindingError("declaration episode index differs from its run plan")
    if str(declaration.get("episode_uuid") or "") != entry.episode_uuid:
        raise SourceExecutionBindingError("declaration episode UUID differs from its run plan")
    raw_case = declaration.get("review_case")
    raw_spec = declaration.get("source_scenario_spec")
    raw_compiled = declaration.get("source_compiled_scenario")
    if not isinstance(raw_case, Mapping):
        raise SourceExecutionBindingError("source declaration lacks its immutable review case")
    if not isinstance(raw_spec, Mapping):
        raise SourceExecutionBindingError("source declaration lacks SourceScenarioSpec")
    if not isinstance(raw_compiled, Mapping):
        raise SourceExecutionBindingError("source declaration lacks compiled scenario identity")
    case = ReviewSuiteCase.from_dict(raw_case)
    _require_executable_case(case)
    spec = SourceScenarioSpec.from_dict(raw_spec)
    if case.episode_uuid != entry.episode_uuid:
        raise SourceExecutionBindingError("review case episode UUID differs from the run plan")
    if int(declaration.get("review_suite_episode_index", -1)) != case.episode_index:
        raise SourceExecutionBindingError("review-suite global episode index binding changed")
    if declaration.get("review_case_sha256") != case.case_sha256:
        raise SourceExecutionBindingError("review-case hash binding changed")
    if entry.source_scenario_spec_sha256 != spec.spec_hash or declaration.get(
        "source_scenario_spec_sha256"
    ) != spec.spec_hash:
        raise SourceExecutionBindingError("SourceScenarioSpec hash binding changed")
    if spec.scenario_id != case.case_id:
        raise SourceExecutionBindingError("SourceScenarioSpec logical case identity changed")
    expected_outer = {
        "backend": spec.backend,
        "corpus_leaf_id": spec.corpus_leaf_id,
        "task_variant": spec.task_variant,
        "embodiment": spec.embodiment.end_effector,
        "duration_s": spec.duration_s,
        "family": case.family,
        "subfamily": case.subfamily,
        "variant": case.task_variant,
        "task_index": _task_index(case.corpus_leaf_id),
        "counterfactual_bundle_id": case.counterfactual_bundle_id,
        "counterfactual_branch_id": case.counterfactual_branch_id,
        "counterfactual_sibling_index": case.counterfactual_sibling_index,
        "physics_counterfactual_family_id": spec.counterfactual.physics_family_id,
        "split_group_id": spec.counterfactual.split_group_id,
    }
    for name, expected in expected_outer.items():
        if declaration.get(name) != expected:
            raise SourceExecutionBindingError(
                f"source declaration {name} differs from its immutable inputs"
            )
    return case, spec


def _verify_prepared_identity(
    entry: RunPlanEpisode,
    backend: SourceMujocoBackend,
    case: ReviewSuiteCase,
    planned_spec: SourceScenarioSpec,
) -> SourceMujocoCompiledScenario:
    freshly_prepared = prepare_review_case(case, backend=backend)
    if freshly_prepared.spec_hash != planned_spec.spec_hash:
        raise SourceExecutionBindingError(
            "fresh SourceScenarioSpec differs from the immutable run plan"
        )
    compiled = backend.compile_case(case)
    expected_compiled_hash = entry.declaration.get(
        "source_compiled_scenario_sha256"
    )
    if compiled.scenario_sha256 != expected_compiled_hash:
        raise SourceExecutionBindingError(
            "fresh compiled source scenario differs from the immutable run plan"
        )
    if entry.declaration.get("source_compiled_scenario") != _canonical_copy(
        compiled.to_dict()
    ):
        raise SourceExecutionBindingError("compiled source scenario payload changed")
    return compiled


def _verify_runtime_identity(
    entry: RunPlanEpisode,
    spec: SourceScenarioSpec,
    result: SourceMujocoRunResult,
) -> None:
    scenario = result.scenario
    expected_compiled_hash = entry.declaration[
        "source_compiled_scenario_sha256"
    ]
    if scenario.scenario_sha256 != expected_compiled_hash:
        raise SourceExecutionBindingError("runtime scenario differs from the run plan")
    if (
        scenario.case_id != spec.scenario_id
        or scenario.episode_uuid != entry.episode_uuid
    ):
        raise SourceExecutionBindingError(
            "runtime logical-case or episode UUID identity differs from the run plan"
        )
    provenance = result.backend_provenance
    if provenance.get("backend") != "source_mujoco":
        raise SourceExecutionBindingError("runtime provenance names the wrong backend")
    if provenance.get("backend_version") != entry.declaration.get("backend_version"):
        raise SourceExecutionBindingError("runtime backend version differs from the run plan")
    if provenance.get("scenario_hash") != expected_compiled_hash:
        raise SourceExecutionBindingError("runtime provenance scenario hash changed")
    expected_source_hashes = entry.declaration.get("runtime_source_hashes")
    if not isinstance(expected_source_hashes, Mapping):
        raise SourceExecutionBindingError("run plan lacks its runtime source-hash manifest")
    actual_source_hashes = dict(result.source_hashes)
    if actual_source_hashes != dict(expected_source_hashes):
        raise SourceExecutionBindingError(
            "runtime source hashes differ from the immutable run plan"
        )
    if sha256_json(actual_source_hashes) != entry.declaration.get(
        "runtime_source_hashes_sha256"
    ):
        raise SourceExecutionBindingError("runtime source-hash manifest binding changed")
    if dict(provenance.get("source_hashes") or {}) != actual_source_hashes:
        raise SourceExecutionBindingError("backend provenance source hashes disagree")
    for name, digest in spec.source_hashes.items():
        if actual_source_hashes.get(name) != digest:
            raise SourceExecutionBindingError(
                f"runtime source hash {name!r} differs from SourceScenarioSpec"
            )
    if result.runtime_audit != provenance.get("runtime_audit"):
        raise SourceExecutionBindingError("runtime audit and backend provenance disagree")
    if result.runtime_audit.get("action_ctrl_echo_exact") is not True:
        raise SourceExecutionBindingError("runtime did not preserve exact action/data.ctrl echoes")
    planned_robot_qpos = spec.initial_state.get("robot_initial_joint_qpos")
    planned_robot_qpos_sha256 = spec.initial_state.get(
        "robot_initial_joint_qpos_sha256"
    )
    actual_robot_qpos = result.runtime_audit.get("initialized_robot_joint_qpos")
    actual_robot_qpos_sha256 = result.runtime_audit.get(
        "initialized_robot_joint_qpos_sha256"
    )
    if (
        _canonical_copy(actual_robot_qpos) != _canonical_copy(planned_robot_qpos)
        or actual_robot_qpos_sha256 != planned_robot_qpos_sha256
        or (
            planned_robot_qpos is not None
            and sha256_json(planned_robot_qpos) != planned_robot_qpos_sha256
        )
    ):
        raise SourceExecutionBindingError(
            "initialized robot joint qpos differs from SourceScenarioSpec"
        )
    planned_base_position = spec.initial_state.get("robot_base_position_m")
    planned_base_quaternion = spec.initial_state.get(
        "robot_base_quaternion_wxyz"
    )
    planned_base_pose_sha256 = spec.initial_state.get("robot_base_pose_sha256")
    if (
        _canonical_copy(result.runtime_audit.get("compiled_robot_base_position_m"))
        != _canonical_copy(planned_base_position)
        or _canonical_copy(
            result.runtime_audit.get("compiled_robot_base_quaternion_wxyz")
        )
        != _canonical_copy(planned_base_quaternion)
        or result.runtime_audit.get("compiled_robot_base_pose_sha256")
        != planned_base_pose_sha256
        or (
            planned_base_position is not None
            and sha256_json(
                {
                    "position_m": planned_base_position,
                    "euler_rad": spec.initial_state.get("robot_base_euler_rad"),
                    "quaternion_wxyz": planned_base_quaternion,
                }
            )
            != planned_base_pose_sha256
        )
    ):
        raise SourceExecutionBindingError(
            "compiled robot base pose differs from SourceScenarioSpec"
        )
    expected_asset_ids = [asset.asset_id for asset in spec.robocasa_manifest.assets]
    actual_asset_ids = [str(row.get("asset_id") or "") for row in result.robocasa_asset_manifest]
    if actual_asset_ids != expected_asset_ids:
        raise SourceExecutionBindingError(
            "runtime RoboCasa assets differ from SourceScenarioSpec"
        )
    by_id = {asset.asset_id: asset for asset in spec.robocasa_manifest.assets}
    for row in result.robocasa_asset_manifest:
        asset = by_id[str(row["asset_id"])]
        aabb = row.get("world_aabb")
        references = row.get("referenced_file_sha256")
        if not isinstance(aabb, Mapping):
            raise SourceExecutionBindingError("runtime RoboCasa asset lacks its world AABB")
        if not isinstance(references, Mapping):
            raise SourceExecutionBindingError(
                "runtime RoboCasa asset lacks its file-hash manifest"
            )
        planned_references = {
            **dict(asset.mesh_sha256),
            **dict(asset.texture_sha256),
        }
        if (
            str(row.get("descriptor_path")) != asset.source_xml
            or str(row.get("descriptor_sha256")) != asset.xml_sha256
            or dict(references) != planned_references
            or tuple(float(value) for value in aabb.get("minimum_m", ()))
            != asset.aabb_min_m
            or tuple(float(value) for value in aabb.get("maximum_m", ()))
            != asset.aabb_max_m
            or bool(row.get("collision_enabled")) != asset.collision_enabled
        ):
            raise SourceExecutionBindingError(
                f"runtime RoboCasa admission differs for {asset.asset_id}"
            )
    if set(result.camera_calibrations) != {"main", "secondary"}:
        raise SourceExecutionBindingError("runtime lacks the exact two planned cameras")
    planned_cameras = {camera.name: camera for camera in spec.cameras}
    canonical_roles = {
        "main": "main_three_quarter_external",
        "secondary": "task_specific_secondary",
    }
    for name, calibration in result.camera_calibrations.items():
        calibration.validate()
        planned = planned_cameras[name]
        raw_rotation = quaternion_to_rotation_matrix(
            planned.pose.quaternion_wxyz
        )
        position = tuple(float(value) for value in planned.pose.position_m)
        # SourceCameraSpec stores MuJoCo's camera rotation (right, up,
        # backward). CameraCalibration uses the canonical image convention
        # (right, down, forward), so flip its second and third axes.
        expected_camera_to_world = (
            raw_rotation[0], -raw_rotation[1], -raw_rotation[2], position[0],
            raw_rotation[3], -raw_rotation[4], -raw_rotation[5], position[1],
            raw_rotation[6], -raw_rotation[7], -raw_rotation[8], position[2],
            0.0, 0.0, 0.0, 1.0,
        )
        expected_world_to_camera = invert_rigid_transform(
            expected_camera_to_world
        )
        focal_px = 0.5 * planned.height / math.tan(
            math.radians(planned.fovy_deg) / 2.0
        )
        expected_intrinsic = (
            focal_px,
            0.0,
            planned.width / 2.0,
            0.0,
            focal_px,
            planned.height / 2.0,
            0.0,
            0.0,
            1.0,
        )
        planned_look_direction = tuple(
            float(planned.look_at_m[index]) - position[index]
            for index in range(3)
        )
        look_norm = math.sqrt(
            sum(value * value for value in planned_look_direction)
        )
        planned_look_direction = tuple(
            value / look_norm for value in planned_look_direction
        )
        runtime_look_direction = (
            float(calibration.camera_to_world[2]),
            float(calibration.camera_to_world[6]),
            float(calibration.camera_to_world[10]),
        )

        def differs(
            actual: Sequence[float], expected: Sequence[float]
        ) -> bool:
            return len(actual) != len(expected) or any(
                abs(float(left) - float(right)) > 1e-9
                for left, right in zip(actual, expected)
            )

        if (
            calibration.camera_name != name
            or planned.role != canonical_roles[name]
            or (calibration.width, calibration.height, calibration.fps)
            != (planned.width, planned.height, float(planned.fps))
            or differs(calibration.camera_to_world, expected_camera_to_world)
            or differs(calibration.world_to_camera, expected_world_to_camera)
            or differs(calibration.intrinsic_matrix, expected_intrinsic)
            or differs(runtime_look_direction, planned_look_direction)
        ):
            raise SourceExecutionBindingError(
                f"runtime camera {name} differs from SourceScenarioSpec"
            )


def _events_by_timestamp(
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[float, list[Mapping[str, Any]]]:
    result: dict[float, list[Mapping[str, Any]]] = {}
    for row in event_rows:
        result.setdefault(round(float(row["timestamp"]), 12), []).append(row)
    return result


def _semantic_fields(
    row: Mapping[str, Any],
    *,
    scenario: SourceMujocoCompiledScenario,
    contacts: Sequence[Mapping[str, Any]],
    interval_contact: bool = False,
) -> dict[str, Any]:
    raw_mode = str(row.get("object.motion_mode") or "")
    bilateral = row.get("contact.bilateral") is True
    categories = {str(value.get("contact_category") or "") for value in contacts}
    if raw_mode == "free_flight":
        motion_mode = MotionMode.FREE_FLIGHT
    elif raw_mode == "gripper_contact":
        motion_mode = MotionMode.RETAINED if bilateral else MotionMode.IMPACT
    elif raw_mode == "surface_contact" and "roll" in scenario.motion_kind:
        motion_mode = MotionMode.ROLLING
    elif raw_mode == "surface_contact":
        motion_mode = MotionMode.IMPACT
    else:
        motion_mode = MotionMode.UNKNOWN
    if categories & {"gripper", "robot_arm"}:
        contact_role = ContactRole.ROBOT_TOOL
        active_surface = "robot_tool"
    elif "task_surface" in categories:
        contact_role = ContactRole.SUPPORT
        surfaces = sorted(
            str(
                value.get("task_surface_id")
                or value.get("object_b")
                or "task_surface"
            )
            for value in contacts
            if value.get("contact_category") == "task_surface"
        )
        active_surface = surfaces[0] if surfaces else "task_surface"
    elif contacts:
        contact_role = ContactRole.UNKNOWN
        active_surface = "unknown"
    else:
        contact_role = ContactRole.NONE
        active_surface = "none"
    timestamp = float(row["timestamp"])
    event_window = 0.5 / scenario.video_hz
    if timestamp < scenario.key_event_time_s - event_window:
        task_phase = "pre_event"
    elif timestamp <= scenario.key_event_time_s + event_window:
        task_phase = "key_event"
    else:
        task_phase = "post_event"
    return {
        "task_phase": task_phase,
        "motion_mode": motion_mode.value,
        "active_surface": active_surface,
        "contact_role": contact_role.value,
        "contact.active": bool(contacts) or int(row.get("contact.count", 0)) > 0,
        # A contact impulse can occur between adjacent 30 Hz frame samples.
        # ``contact.active`` remains the exact sampled state; ``event.contact``
        # marks an impulse inside this frame's exposure bin so persisted
        # finite-difference QC does not mistake a rebound discontinuity for
        # corrupt velocity data.
        "event.contact": (
            bool(contacts)
            or int(row.get("contact.count", 0)) > 0
            or interval_contact
        ),
        "free_fall": motion_mode is MotionMode.FREE_FLIGHT,
        "assistance.active": False,
        "assistance.assisted_grasp": False,
        "assistance.assisted_retention": False,
        "assistance.equality_constraint_active": False,
        "assistance.latch_active": False,
        "assistance.mechanism_ids": [],
    }


def _state_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep replay-relevant source state while omitting online-only force summaries."""

    names = (
        "object.position",
        "object.quaternion_wxyz",
        "object.linear_velocity",
        "object.angular_velocity",
        "object.motion_mode",
        "robot.joint_position",
        "robot.joint_velocity",
        "grasp.center_position",
        "contact.count",
        "contact.left_gripper",
        "contact.right_gripper",
        "contact.bilateral",
        "action.actuator_command",
        "simulator.applied_actuator_ctrl",
        "action.mode",
    )
    return {
        name: _canonical_copy(row[name])
        for name in names
        if name in row and row[name] is not None
    }


def _validate_actions(
    rows: Sequence[Mapping[str, Any]], spec: SourceScenarioSpec, *, label: str
) -> None:
    expected_width = len(spec.embodiment.action_names)
    for index, row in enumerate(rows):
        action = row.get("action.actuator_command")
        applied = row.get("simulator.applied_actuator_ctrl")
        if not isinstance(action, Sequence) or isinstance(action, (str, bytes)):
            raise SourceExecutionBindingError(f"{label} row {index} lacks actuator commands")
        if not isinstance(applied, Sequence) or isinstance(applied, (str, bytes)):
            raise SourceExecutionBindingError(f"{label} row {index} lacks data.ctrl echo")
        action_tuple = tuple(float(value) for value in action)
        applied_tuple = tuple(float(value) for value in applied)
        if len(action_tuple) != expected_width or applied_tuple != action_tuple:
            raise SourceExecutionBindingError(
                f"{label} row {index} does not contain the exact {expected_width}-control echo"
            )
        if str(row.get("action.mode") or "") != spec.embodiment.action_semantics:
            raise SourceExecutionBindingError(
                f"{label} row {index} action mode differs from SourceScenarioSpec"
            )


def _normalize_rows(
    entry: RunPlanEpisode,
    spec: SourceScenarioSpec,
    result: SourceMujocoRunResult,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_frames = [dict(row) for row in result.frame_rows]
    source_high_rate = [dict(row) for row in result.high_rate_rows]
    source_events = [dict(row) for row in result.contact_rows]
    _validate_actions(source_frames, spec, label="frame")
    _validate_actions(source_high_rate, spec, label="high-rate")
    event_map = _events_by_timestamp(source_events)
    task_index = int(entry.declaration["task_index"])
    frames: list[dict[str, Any]] = []
    for frame_index, source in enumerate(source_frames):
        timestamp = float(source["timestamp"])
        contacts = event_map.get(round(timestamp, 12), ())
        interval_contact = any(
            abs(float(event["timestamp"]) - timestamp)
            <= 0.5 / result.scenario.video_hz + 1e-12
            for event in source_events
        )
        frames.append(
            {
                "episode_index": entry.episode_index,
                "task_index": task_index,
                "frame_index": frame_index,
                "video_frame_index": frame_index,
                "timestamp": timestamp,
                "simulation_timestamp": float(source["simulation_timestamp"]),
                "synchronization_error_s": float(source["synchronization_error_s"]),
                **_state_payload(source),
                **_semantic_fields(
                    source,
                    scenario=result.scenario,
                    contacts=contacts,
                    interval_contact=interval_contact,
                ),
            }
        )
    high_rate: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for source in source_high_rate:
        timestamp = float(source["timestamp"])
        contacts = event_map.get(round(timestamp, 12), ())
        semantics = _semantic_fields(
            source, scenario=result.scenario, contacts=contacts
        )
        high_rate_semantics = {
            name: value
            for name, value in semantics.items()
            if not name.startswith("assistance.")
        }
        payload = _state_payload(source)
        high_rate.append(
            {
                "episode_index": entry.episode_index,
                "timestamp": timestamp,
                **payload,
                **high_rate_semantics,
            }
        )
        objects.append(
            {
                "episode_index": entry.episode_index,
                "timestamp": timestamp,
                "object_id": "catch_ball",
                **{
                    name: value
                    for name, value in payload.items()
                    if name.startswith(("object.", "grasp.", "contact."))
                },
                "motion_mode": semantics["motion_mode"],
                "active_surface": semantics["active_surface"],
                "contact_role": semantics["contact_role"],
            }
        )
        semantic_rows.append({"timestamp": timestamp, **semantics})
    events = [
        {
            "episode_index": entry.episode_index,
            **{
                name: _canonical_copy(source.get(name))
                for name in (
                    "timestamp",
                    "object_a",
                    "object_b",
                    "point_world_m",
                    "normal_world",
                    "penetration_depth_m",
                    "contact_category",
                    "task_surface_id",
                    "counterpart_geom_id",
                    "normal_force_n",
                    "normal_impulse_n_s",
                    "relative_velocity_world_m_s",
                    "expected_fixture_contact",
                    "snag",
                )
            },
        }
        for source in source_events
    ]
    transitions: list[dict[str, Any]] = []
    for previous, current in zip(semantic_rows, semantic_rows[1:]):
        if current["motion_mode"] == previous["motion_mode"]:
            continue
        transitions.append(
            {
                "episode_index": entry.episode_index,
                "timestamp": current["timestamp"],
                "event_type": "motion_mode_transition",
                "from": previous["motion_mode"],
                "to": current["motion_mode"],
                "active_surface": current["active_surface"],
            }
        )
    return frames, high_rate, events, transitions, objects


def _camera_rows(
    calibrations: Mapping[str, CameraCalibration],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    mapping: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for short_name in ("main", "secondary"):
        calibration = calibrations[short_name]
        value = calibration.to_dict()
        stream = canonical_camera_name(short_name)
        value["camera_name"] = stream
        CameraCalibration.from_dict(value)
        identifier = f"{stream}@{sha256_json(value)[:16]}"
        mapping[stream] = identifier
        rows.append({"camera_id": identifier, **value})
    return mapping, rows


def _randomization(
    case: ReviewSuiteCase,
    spec: SourceScenarioSpec,
    runtime_assets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    first_asset = runtime_assets[0] if runtime_assets else None
    return {
        "counterfactual_bundle_id": case.counterfactual_bundle_id,
        "scene_asset_id": None if first_asset is None else str(first_asset["asset_id"]),
        "background_style": (
            "clean_franka_lab" if case.randomization_level == "R0" else case.scene_profile
        ),
        "lighting_id": "neutral",
        "object_asset_id": "catch_ball_sphere_v1",
        "object_color_id": "review_fixed",
        "tool_asset_id": (
            None if case.embodiment == "no_robot" else case.embodiment
        ),
        "camera_preset_id": "source_scenario_main_secondary_v1",
        "randomization_level": case.randomization_level,
        "randomization_seed": spec.rng_subseeds.assets,
        "rng_subseeds": _canonical_copy(spec.rng_subseeds.__dict__ if hasattr(spec.rng_subseeds, "__dict__") else {
            "physics": spec.rng_subseeds.physics,
            "initial_state": spec.rng_subseeds.initial_state,
            "camera": spec.rng_subseeds.camera,
            "assets": spec.rng_subseeds.assets,
            "controller": spec.rng_subseeds.controller,
            "scene_construction": spec.rng_subseeds.scene_construction,
        }),
        "varied_fields": (
            []
            if case.randomization_level == "R0"
            else ["scene_asset", "background_style", "lighting", "texture", "object_color"]
        ),
        "randomization_policy_sha256": default_randomization_policy().policy_hash,
        "scene_asset_manifest_sha256": (
            None if first_asset is None else first_asset.get("manifest_sha256")
        ),
        "training_eligible": False,
    }


def _physics_metadata(
    spec: SourceScenarioSpec, result: SourceMujocoRunResult
) -> PhysicsMetadata:
    physics = spec.physics
    gravity = tuple(float(value) for value in physics["gravity_m_s2"])
    profile_id = str(physics["profile"])
    return PhysicsMetadata(
        parameters={
            "object_mass": PhysicsValue(
                name="object_mass",
                value=float(physics["object_mass_kg"]),
                unit="kg",
                valid=True,
                implemented=True,
                kind=PhysicsValueKind.PHYSICAL,
                source="compiled_mujoco_model",
            ),
            "object_radius": PhysicsValue(
                name="object_radius",
                value=float(physics["object_radius_m"]),
                unit="m",
                valid=True,
                implemented=True,
                kind=PhysicsValueKind.PHYSICAL,
                source="compiled_mujoco_model",
            ),
            "gravity": PhysicsValue(
                name="gravity",
                value=list(gravity),
                unit="m/s^2",
                valid=True,
                implemented=True,
                kind=PhysicsValueKind.PHYSICAL,
                source="compiled_mujoco_model",
            ),
            "simulation_timestep": PhysicsValue(
                name="simulation_timestep",
                value=1.0 / float(physics["simulation_hz"]),
                unit="s",
                valid=True,
                implemented=True,
                kind=PhysicsValueKind.PHYSICAL,
                source="compiled_mujoco_model",
            ),
        },
        gravity_world_m_s2=gravity,  # type: ignore[arg-type]
        gravity_valid=True,
        simulation_timestep_s=1.0 / float(physics["simulation_hz"]),
        substeps=1,
        solver_settings={
            "profile_id": profile_id,
            "compiled_model_sha256": result.source_hashes["compiled_scene_xml"],
            "simulation_hz": int(physics["simulation_hz"]),
            "control_hz": int(physics["control_hz"]),
            "backend_physics_qc_schema": result.physics_qc.get("schema_version"),
        },
        external_impulses=[],
        parameter_range_provenance={
            "profile_id": profile_id,
            "profile_version": profile_id,
            "partition": ParameterRangePartition.UNCALIBRATED.value,
            "source": "owned_source_mujoco_review_profile",
            "calibrated": False,
            "calibration_verified": False,
            "calibration_artifact_hash": None,
        },
    )


def _simulator_version() -> str:
    try:
        return package_version("mujoco")
    except PackageNotFoundError:
        return "runtime-version-unavailable"


def materialize_source_mujoco_result(
    entry: RunPlanEpisode,
    result: SourceMujocoRunResult,
) -> EpisodeMaterialization:
    """Normalize one verified source rollout into the canonical writer contract."""

    case, spec = _planned_inputs(entry)
    return _materialize_verified(entry, result, case, spec)


def _materialize_verified(
    entry: RunPlanEpisode,
    result: SourceMujocoRunResult,
    case: Any,
    spec: SourceScenarioSpec,
    *,
    case_payload_key: str = "review_case",
) -> EpisodeMaterialization:
    """Shared normalization for one planned case its own bridge already verified.

    ``case`` is a ``ReviewSuiteCase`` on the fixed-review path and a
    ``ScaleSuiteCase`` on the diagnostic scale path; both carry the identical
    attribute surface this normalization consumes.  Behavior for review
    callers is byte-identical to the pre-split implementation.
    """

    _verify_runtime_identity(entry, spec, result)
    frames, high_rate, events, transitions, objects = _normalize_rows(
        entry, spec, result
    )
    measurement = evaluate_source_rows(
        evaluator_id=case.evaluator,
        corpus_leaf_id=case.corpus_leaf_id,
        task_variant=case.task_variant,
        source_spec=spec.to_dict(),
        state_rows=objects,
        event_rows=events,
    )
    measurement.validate()
    camera_mapping, camera_rows = _camera_rows(result.camera_calibrations)
    quality_flags = {
        *result.quality_flags,
        "review_only_unreleased_backend",
        "uncalibrated_physics_profile",
        "human_review_pending",
    }
    if not any(result.frames_by_camera.values()):
        quality_flags.add("unrendered_diagnostic_only")
    online_success = result.outcome.get("task_success")
    online_outcome = str(result.outcome.get("actual_outcome") or "")
    measured_outcome = (
        "passive_observation"
        if measurement.task_success and case.embodiment == "no_robot"
        else measurement.actual_outcome_class.value
    )
    online_replay_matches = bool(
        isinstance(online_success, bool)
        and online_success == measurement.task_success
        and online_outcome == measured_outcome
    )
    if not online_replay_matches:
        quality_flags.add("online_outcome_disagrees_with_independent_replay")
    evaluator_invalid = measurement.actual_outcome_class is ActualOutcomeClass.INVALID
    physics_qc_pass = bool(
        result.physics_qc.get("physics_qc_pass") is True
        and not evaluator_invalid
        and online_replay_matches
    )
    actual_outcome = measured_outcome
    evidence_hash = sha256_json(measurement.evidence)
    source_manifest_sha256 = sha256_json(result.source_hashes)
    backend_provenance_sha256 = sha256_json(result.backend_provenance)
    runtime_audit_sha256 = sha256_json(result.runtime_audit)
    visibility_qc_sha256 = sha256_json(result.visibility_qc)
    if (
        result.backend_provenance.get("visibility_qc_sha256")
        != visibility_qc_sha256
    ):
        raise SourceExecutionBindingError(
            "runtime visibility QC differs from its backend provenance hash"
        )
    background_clearance_sha256 = sha256_json(result.background_clearance)
    if (
        result.backend_provenance.get("background_clearance_sha256")
        != background_clearance_sha256
        or result.runtime_audit.get("background_clearance_sha256")
        != background_clearance_sha256
    ):
        raise SourceExecutionBindingError(
            "runtime background clearance differs from provenance/audit hash"
        )
    case_hash_key = case_payload_key + "_sha256"
    suite_index_key = (
        "review_suite_episode_index"
        if case_payload_key == "review_case"
        else "scale_suite_episode_index"
    )
    record = EpisodeRecord(
        episode_uuid=entry.episode_uuid,
        episode_index=entry.episode_index,
        counterfactual_bundle_id=case.counterfactual_bundle_id,
        physics_counterfactual_family_id=spec.counterfactual.physics_family_id,
        split_group_id=spec.counterfactual.split_group_id,
        scene_seed=_metadata_seed(spec.rng_subseeds.scene_construction),
        branch_seed=_metadata_seed(spec.rng_subseeds.controller),
        family=case.family,
        subfamily=case.subfamily,
        intended_branch=case.branch_role,
        actual_outcome=actual_outcome,
        actual_outcome_class=measurement.actual_outcome_class,
        task_success=measurement.task_success,
        failure_mode=measurement.primary_failure_code,
        primary_failure_code=measurement.primary_failure_code,
        failure_tags=default_failure_tags(measurement.primary_failure_code),
        variant=case.task_variant,
        robot_model=spec.embodiment.robot_model,
        tool_type=spec.embodiment.end_effector,
        action_mode=spec.embodiment.action_semantics,
        partial_success_score=measurement.partial_success_score,
        label_confidence=1.0,
        label_status=LabelStatus.VERIFIED,
        dynamics_mode=DynamicsMode.FREE_CONTACT,
        release_tier=ReleaseTier.FREE_CONTACT,
        physics_qc_pass=physics_qc_pass,
        source_generator="dynamic_robot_dataset.source_execution",
        source_generator_version=SOURCE_EXECUTION_BRIDGE_SCHEMA,
        generator_git_commit=str(
            entry.declaration.get("generator_git_commit") or "unknown"
        ),
        config_hash="pending-episode-writer-binding",
        asset_ids=[
            "catch_ball_sphere_v1",
            *[str(row["asset_id"]) for row in result.robocasa_asset_manifest],
        ],
        asset_hashes={
            **dict(result.source_hashes),
            **{
                f"robocasa:{row['asset_id']}": str(row["manifest_sha256"])
                for row in result.robocasa_asset_manifest
            },
        },
        simulator_name="mujoco",
        simulator_version=_simulator_version(),
        renderer="mujoco.Renderer",
        task_index=int(entry.declaration["task_index"]),
        duration_s=spec.duration_s,
        event_time_s=measurement.key_event_time_s,
        key_event_name=measurement.key_event_name,
        key_event_time_s=measurement.key_event_time_s,
        objective_evaluator_id=case.evaluator,
        objective_evaluator_version=SOURCE_OBJECTIVE_EVALUATOR_VERSION,
        objective_threshold_set_hash=sha256_json(
            {
                "evaluator_id": case.evaluator,
                "evaluator_version": SOURCE_OBJECTIVE_EVALUATOR_VERSION,
                "physics_thresholds": result.physics_qc.get("thresholds"),
            }
        ),
        objective_evidence={
            "stored_objective_success": measurement.task_success,
            "independently_recomputed": True,
            "source": "source_persisted_state_event/v1",
            "evidence_hash": evidence_hash,
            "evidence_version": measurement.evidence_version,
        },
        physics=_physics_metadata(spec, result),
        assistance={
            "assisted_grasp": False,
            "assisted_retention": False,
            "equality_constraint_active": False,
            "latch_active": False,
            "constraint_activation_time": None,
            "constraint_deactivation_time": None,
            "mechanisms": [],
        },
        objective_metrics={
            "objective_success": measurement.task_success,
            "independent_evidence_hash": evidence_hash,
        },
        controller_profile={
            "profile_id": str(spec.physics["profile"]),
            "profile_version": str(spec.physics["profile"]),
            "control_latency_s": 0.0,
            "camera_latency_s": 0.0,
        },
        robot_start_provenance={
            "source": "compiled_source_scenario_initialization",
            "rng_subseed": spec.rng_subseeds.controller,
            "state_writes_after_initialization": int(
                result.runtime_audit.get(
                    "direct_robot_state_writes_after_initialization", -1
                )
            ),
        },
        tool_calibration_provenance={
            "calibrated": False,
            "source": "review_profile_pending_acceptance",
            "end_effector": spec.embodiment.end_effector,
        },
        randomization=_randomization(
            case, spec, result.robocasa_asset_manifest
        ),
        quality_flags=sorted(quality_flags),
        extras={
            "bridge_schema_version": str(entry.declaration["schema_version"]),
            case_payload_key: case.to_dict(),
            case_hash_key: case.case_sha256,
            suite_index_key: case.episode_index,
            "metadata_seed_projection": {
                "method": "uint64_bitmask_to_nonnegative_int64/v1",
                "scene_construction_uint64": spec.rng_subseeds.scene_construction,
                "controller_uint64": spec.rng_subseeds.controller,
                "scene_seed_int64": _metadata_seed(
                    spec.rng_subseeds.scene_construction
                ),
                "branch_seed_int64": _metadata_seed(spec.rng_subseeds.controller),
            },
            "source_scenario_spec": spec.to_dict(),
            "source_scenario_spec_sha256": spec.spec_hash,
            "source_compiled_scenario_sha256": entry.declaration[
                "source_compiled_scenario_sha256"
            ],
            "source_hashes": dict(result.source_hashes),
            "source_manifest_sha256": source_manifest_sha256,
            "backend_provenance": dict(result.backend_provenance),
            "backend_provenance_sha256": backend_provenance_sha256,
            "runtime_audit": dict(result.runtime_audit),
            "runtime_audit_sha256": runtime_audit_sha256,
            "physics_qc": dict(result.physics_qc),
            "visibility_qc": dict(result.visibility_qc),
            "visibility_qc_sha256": visibility_qc_sha256,
            "background_clearance": dict(result.background_clearance),
            "background_clearance_sha256": background_clearance_sha256,
            "online_outcome_diagnostic": dict(result.outcome),
            "independent_objective_evidence": dict(measurement.evidence),
            "robocasa_asset_manifest": list(result.robocasa_asset_manifest),
            "robocasa_planned_manifest": spec.to_dict()["robocasa_manifest"],
            "camera_calibrations": camera_rows,
            "end_effector": spec.embodiment.end_effector,
            "control_hz": int(spec.physics["control_hz"]),
            "production_eligible": False,
            "release_blockers": [
                "backend_not_released",
                "physics_profile_not_calibrated",
                "human_review_pending",
            ],
            "r1_accepted": False,
        },
    )
    record.validate()
    return EpisodeMaterialization(
        record=record,
        frame_rows=frames,
        videos={
            canonical_camera_name(name): result.frames_by_camera[name]
            for name in ("main", "secondary")
        },
        high_rate_rows=high_rate,
        event_rows=events,
        transition_rows=transitions,
        object_state_rows=objects,
        camera_calibration_ids=camera_mapping,
    )


def execute_source_mujoco_episode(
    entry: RunPlanEpisode,
    *,
    render: bool = True,
) -> EpisodeMaterialization:
    """Execute an immutable run-plan entry through the capability-gated backend."""

    case, spec = _planned_inputs(entry)
    backend = _review_backend(case)
    compiled = _verify_prepared_identity(entry, backend, case, spec)
    result = backend.run(compiled, render=render)
    return materialize_source_mujoco_result(entry, result)


@dataclass(frozen=True, slots=True)
class SourceFinalizationRows:
    """Dataset-level camera and provenance rows recovered from committed records."""

    cameras: tuple[dict[str, Any], ...]
    provenance: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cameras": [_canonical_copy(row) for row in self.cameras],
            "provenance": [_canonical_copy(row) for row in self.provenance],
        }


def source_finalization_rows(
    records: Iterable[EpisodeRecord],
) -> SourceFinalizationRows:
    """Recover exact camera/provenance metadata without rerunning simulation."""

    camera_by_id: dict[str, dict[str, Any]] = {}
    provenance_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        raw_cameras = record.extras.get("camera_calibrations")
        if not isinstance(raw_cameras, Sequence) or isinstance(
            raw_cameras, (str, bytes, bytearray)
        ):
            raise SourceExecutionBindingError(
                f"committed source record lacks camera calibrations: {record.episode_uuid}"
            )
        provided_camera_ids: set[str] = set()
        for raw in raw_cameras:
            if not isinstance(raw, Mapping):
                raise SourceExecutionBindingError("camera calibration row is not a mapping")
            row = dict(raw)
            identifier = str(row.get("camera_id") or "")
            calibration = dict(row)
            calibration.pop("camera_id", None)
            CameraCalibration.from_dict(calibration)
            stream = str(calibration.get("camera_name") or "")
            expected_identifier = f"{stream}@{sha256_json(calibration)[:16]}"
            if identifier != expected_identifier:
                raise SourceExecutionBindingError(
                    f"camera calibration content hash changed: {identifier}"
                )
            if identifier not in set(record.camera_stream_calibration_ids.values()):
                raise SourceExecutionBindingError(
                    f"unreferenced camera calibration {identifier} in {record.episode_uuid}"
                )
            existing = camera_by_id.get(identifier)
            if existing is not None and existing != row:
                raise SourceExecutionBindingError(
                    f"camera calibration ID collision: {identifier}"
                )
            camera_by_id[identifier] = _canonical_copy(row)
            provided_camera_ids.add(identifier)
        referenced_camera_ids = set(record.camera_stream_calibration_ids.values())
        if provided_camera_ids != referenced_camera_ids:
            raise SourceExecutionBindingError(
                "source record camera rows do not exactly match its calibration references: "
                f"provided={sorted(provided_camera_ids)}, "
                f"referenced={sorted(referenced_camera_ids)}"
            )
        backend_provenance = record.extras.get("backend_provenance")
        runtime_audit = record.extras.get("runtime_audit")
        source_hashes = record.extras.get("source_hashes")
        physics_qc = record.extras.get("physics_qc")
        visibility_qc = record.extras.get("visibility_qc")
        background_clearance = record.extras.get("background_clearance")
        source_spec = record.extras.get("source_scenario_spec")
        if not all(
            isinstance(value, Mapping)
            for value in (
                backend_provenance,
                runtime_audit,
                source_hashes,
                physics_qc,
                visibility_qc,
                background_clearance,
                source_spec,
            )
        ):
            raise SourceExecutionBindingError(
                f"committed source record lacks provenance manifests: {record.episode_uuid}"
            )
        scenario = SourceScenarioSpec.from_dict(source_spec)
        expected_hash_bindings = {
            "source_scenario_spec_sha256": scenario.spec_hash,
            "source_manifest_sha256": sha256_json(source_hashes),
            "backend_provenance_sha256": sha256_json(backend_provenance),
            "runtime_audit_sha256": sha256_json(runtime_audit),
            "visibility_qc_sha256": sha256_json(visibility_qc),
            "background_clearance_sha256": sha256_json(
                background_clearance
            ),
        }
        visibility_media_binding = record.extras.get(
            "visibility_media_binding"
        )
        if not isinstance(visibility_media_binding, Mapping):
            raise SourceExecutionBindingError(
                "committed source record lacks visibility media binding"
            )
        try:
            recomputed_visibility_media_binding = (
                source_mujoco_visibility_media_binding(
                    visibility_qc_sha256=sha256_json(visibility_qc),
                    camera_rows=raw_cameras,
                    camera_stream_calibration_ids=(
                        record.camera_stream_calibration_ids
                    ),
                    video_paths=record.video_paths,
                    content_hashes=record.content_hashes,
                )
            )
        except (TypeError, ValueError) as error:
            raise SourceExecutionBindingError(
                "committed source visibility media binding cannot be "
                f"recomputed: {error}"
            ) from error
        if _canonical_copy(visibility_media_binding) != (
            recomputed_visibility_media_binding
        ):
            raise SourceExecutionBindingError(
                "committed source visibility media binding differs from "
                "camera rows or encoded media"
            )
        visibility_media_binding = recomputed_visibility_media_binding
        expected_hash_bindings["visibility_media_binding_sha256"] = (
            sha256_json(visibility_media_binding)
        )
        for name, expected in expected_hash_bindings.items():
            if record.extras.get(name) != expected:
                raise SourceExecutionBindingError(
                    f"committed source provenance hash changed: {name}"
                )
        background_clearance_sha256 = sha256_json(background_clearance)
        if (
            backend_provenance.get("background_clearance_sha256")
            != background_clearance_sha256
            or backend_provenance.get("background_clearance")
            != background_clearance
            or runtime_audit.get("background_clearance_sha256")
            != background_clearance_sha256
        ):
            raise SourceExecutionBindingError(
                "committed source background clearance binding changed"
            )
        provenance = {
            "schema_version": SOURCE_FINALIZATION_PROVENANCE_SCHEMA,
            "episode_uuid": record.episode_uuid,
            "episode_index": record.episode_index,
            "source_generator": record.source_generator,
            "source_generator_version": record.source_generator_version,
            "generator_git_commit": record.generator_git_commit,
            "config_hash": record.config_hash,
            "simulator_name": record.simulator_name,
            "simulator_version": record.simulator_version,
            "renderer": record.renderer,
            "backend": str(backend_provenance.get("backend") or ""),
            "backend_version": str(
                backend_provenance.get("backend_version") or ""
            ),
            "source_scenario_spec_sha256": str(
                record.extras.get("source_scenario_spec_sha256") or ""
            ),
            "source_manifest_sha256": sha256_json(source_hashes),
            "backend_provenance_sha256": sha256_json(backend_provenance),
            "runtime_audit_sha256": sha256_json(runtime_audit),
            "physics_qc_sha256": sha256_json(physics_qc),
            "visibility_qc_sha256": sha256_json(visibility_qc),
            "background_clearance_sha256": sha256_json(
                background_clearance
            ),
            "visibility_media_binding_sha256": sha256_json(
                visibility_media_binding
            ),
            "review_only": True,
            "production_eligible": False,
        }
        provenance_id = (
            f"source_mujoco:{record.episode_uuid}:"
            f"{sha256_json(provenance)[:16]}"
        )
        provenance["provenance_id"] = provenance_id
        provenance_by_id[provenance_id] = _canonical_copy(provenance)
    return SourceFinalizationRows(
        cameras=tuple(camera_by_id[key] for key in sorted(camera_by_id)),
        provenance=tuple(
            provenance_by_id[key] for key in sorted(provenance_by_id)
        ),
    )


__all__ = [
    "SOURCE_EXECUTION_BRIDGE_SCHEMA",
    "SOURCE_FINALIZATION_PROVENANCE_SCHEMA",
    "SourceExecutionBindingError",
    "SourceFinalizationRows",
    "execute_source_mujoco_episode",
    "materialize_source_mujoco_result",
    "prepare_source_review_declaration",
    "source_finalization_rows",
]
