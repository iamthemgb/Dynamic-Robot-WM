"""Executable, review-only MuJoCo backend with strict physical evidence.

The external demo repository contributes scene/model assets only.  IK,
trajectory construction, callbacks, runtime mutation enforcement, state
sampling, contact evidence, synchronization, and QC are owned here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..actuator_only import ACTION_FIELD, ACTION_SEMANTICS, ControlObservation
from ...common.cameras import CameraCalibration
from ...common.embodiments import FRANKA_HAND, ROBOTIQ_2F85_THICK_PAD
from ...common.hashing import combined_manifest_hash, sha256_file, sha256_json
from ...common.source_evaluators import select_source_key_event
from ...common.physics_contract import (
    STRICT_RIGID_QC_SCHEMA,
    rigid_task_evidence_failures,
    strict_contact_penetration_check,
)
from ...common.synchronization import (
    fixed_duration_frame_timestamps,
    validate_persisted_render_schedule,
)
from ...common.visual_qc import (
    SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA,
    SOURCE_MUJOCO_VISUAL_THRESHOLDS,
)
from .compiler import (
    SOURCE_MUJOCO_BACKEND_VERSION,
    SourceMujocoCompiledScenario,
    compile_review_case,
)
from .controller import OwnedActuatorController, OwnedControllerPlan
from .model import CompiledSourceModel, compile_source_model
from .profiles import RIGID_REVIEW_PROFILE
from .provenance import (
    PINNED_SOURCE_MANIFEST_SHA256,
    PINNED_SOURCE_FILES,
    RoboCasaDependency,
    SourceDependencyManifest,
    resolve_robocasa_dependency,
    resolve_source_dependency,
)


PANDA_HOME_Q = np.array(
    (0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.78), dtype=np.float64
)
PANDA_FINGERTIP_LOCAL_CENTER = np.array(
    (0.0, 0.0055, 0.0445), dtype=np.float64
)
ROBOTIQ_OPEN_Q = np.array(
    (
        0.002273890874386094,
        0.0001364909715520716,
        0.0024731211244206548,
        -0.00267025473687781,
        0.002273890874386094,
        0.0001364909715520716,
        0.0024731211244206548,
        -0.00267025473687781,
    ),
    dtype=np.float64,
)

SOURCE_MUJOCO_BACKGROUND_CLEARANCE_SCHEMA = (
    "source-mujoco-background-clearance/v2"
)
SOURCE_MUJOCO_TOOL_VISIBILITY_TOPOLOGY_SCHEMA = (
    "source-mujoco-tool-visibility-topology/v1"
)


@dataclass(frozen=True, slots=True)
class IKDiagnostics:
    success: bool
    solver_success: bool
    target_position_m: tuple[float, float, float]
    achieved_grasp_center_m: tuple[float, float, float]
    position_error_m: float
    measured_hand_to_fingertip_offset_m: tuple[float, float, float] | None
    correction_pass_count: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceMujocoRunResult:
    scenario: SourceMujocoCompiledScenario
    frames_by_camera: Mapping[str, Sequence[np.ndarray]]
    camera_calibrations: Mapping[str, CameraCalibration]
    frame_rows: Sequence[Mapping[str, Any]]
    high_rate_rows: Sequence[Mapping[str, Any]]
    contact_rows: Sequence[Mapping[str, Any]]
    outcome: Mapping[str, Any]
    physics_qc: Mapping[str, Any]
    visibility_qc: Mapping[str, Any]
    background_clearance: Mapping[str, Any]
    runtime_audit: Mapping[str, Any]
    backend_provenance: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    robocasa_asset_manifest: Sequence[Mapping[str, Any]]
    quality_flags: tuple[str, ...]

    @property
    def production_eligible(self) -> bool:
        return False

    def as_renderer_payload(self) -> dict[str, Any]:
        return {
            "videos": dict(self.frames_by_camera),
            "cameras": dict(self.camera_calibrations),
            "frame_rows": list(self.frame_rows),
            "high_rate_rows": list(self.high_rate_rows),
            "event_rows": list(self.contact_rows),
            "object_state_rows": list(self.high_rate_rows),
            "transition_rows": [],
            "renderer": "mujoco.Renderer",
            "backend_provenance": dict(self.backend_provenance),
            "production_eligible": False,
            "quality_flags": list(self.quality_flags),
            "physics_qc": dict(self.physics_qc),
            "visibility_qc": dict(self.visibility_qc),
            "background_clearance": dict(self.background_clearance),
            "runtime_audit": dict(self.runtime_audit),
            "source_hashes": dict(self.source_hashes),
            "robocasa_asset_manifest": list(self.robocasa_asset_manifest),
            "objective_evidence": dict(self.outcome),
            "action_mode": (
                "no_actuators/v1"
                if self.scenario.embodiment == "no_robot"
                else ACTION_SEMANTICS
            ),
        }


def _require_runtime_dependencies() -> tuple[Any, Any]:
    if not os.environ.get("DISPLAY"):
        # Headless cluster workers require an explicit EGL backend before the
        # first mujoco import; otherwise rendering silently selects GLFW/X11.
        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError(
            "source_mujoco requires the project mujoco/scipy optional dependencies"
        ) from error
    return mujoco, least_squares


def _quat_error(mujoco: Any, current: np.ndarray, target: np.ndarray) -> np.ndarray:
    inverse = np.array((current[0], -current[1], -current[2], -current[3]))
    value = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(value, target, inverse)
    if value[0] < 0:
        value *= -1
    return value[1:]


def _panda_fingertip_center(model: Any, data: Any, ids: Any) -> np.ndarray:
    assert ids.left_gripper_body is not None and ids.right_gripper_body is not None
    left = (
        data.xpos[ids.left_gripper_body]
        + data.xmat[ids.left_gripper_body].reshape(3, 3)
        @ PANDA_FINGERTIP_LOCAL_CENTER
    )
    right = (
        data.xpos[ids.right_gripper_body]
        + data.xmat[ids.right_gripper_body].reshape(3, 3)
        @ PANDA_FINGERTIP_LOCAL_CENTER
    )
    return 0.5 * (left + right)


def _robotiq_thick_pad_center(data: Any, ids: Any) -> np.ndarray:
    """Return the physical midpoint of the two calibrated thick pads.

    The source scene's historical ``robotiq_grasp_center_site`` is attached to
    the gripper base.  It is about a centimetre behind the actual thick-pad
    contact corridor and therefore is not a valid IK or retention reference.
    The owned backend instead measures the collision geometry that can
    physically support the object.
    """

    if not ids.left_gripper_geom_ids or not ids.right_gripper_geom_ids:
        raise RuntimeError("Robotiq thick-pad collision geometry is unavailable")
    left = np.asarray(
        data.geom_xpos[ids.left_gripper_geom_ids[-1]], dtype=np.float64
    )
    right = np.asarray(
        data.geom_xpos[ids.right_gripper_geom_ids[-1]], dtype=np.float64
    )
    return 0.5 * (left + right)


def _grasp_center(
    model: Any,
    data: Any,
    ids: Any,
    embodiment: str,
) -> np.ndarray | None:
    if embodiment == "no_robot":
        return None
    if embodiment == FRANKA_HAND:
        return _panda_fingertip_center(model, data, ids)
    return _robotiq_thick_pad_center(data, ids)


def _joint_ranges(model: Any, joint_ids: Sequence[int]) -> np.ndarray:
    ranges = np.asarray(model.jnt_range[list(joint_ids)], dtype=np.float64).copy()
    for index, joint_id in enumerate(joint_ids):
        if not bool(model.jnt_limited[int(joint_id)]):
            ranges[index] = (-3.0, 3.0)
    return ranges


def _solve_arm_ik(
    mujoco: Any,
    least_squares: Any,
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
    target_position: Sequence[float],
    *,
    initial_arm_q: np.ndarray | None = None,
) -> tuple[np.ndarray, IKDiagnostics]:
    """Solve and reject failed IK; Panda uses measured fingertip correction."""

    model, data, ids = compiled.model, compiled.data, compiled.ids
    target = np.asarray(target_position, dtype=np.float64)
    arm_joint_ids = ids.robot_joint_ids[:7]
    arm_qpos = np.asarray(ids.robot_qpos_adrs[:7], dtype=np.int32)
    ranges = _joint_ranges(model, arm_joint_ids)
    initial = PANDA_HOME_Q.copy() if initial_arm_q is None else initial_arm_q.copy()
    data.qpos[arm_qpos] = initial
    if scenario.embodiment == FRANKA_HAND:
        data.qpos[list(ids.robot_qpos_adrs[7:9])] = 0.04
    else:
        data.qpos[list(ids.robot_qpos_adrs[7:])] = ROBOTIQ_OPEN_Q
    data.qvel[list(ids.robot_qvel_adrs)] = 0.0
    mujoco.mj_forward(model, data)
    assert ids.hand_body is not None
    target_quaternion = np.asarray(data.xquat[ids.hand_body], dtype=np.float64).copy()
    if scenario.embodiment == FRANKA_HAND:
        # Upward-facing catch posture: the Panda fingertip center is +Z in the
        # hand frame.  Preserving the arbitrary home orientation instead put
        # the palm/link6 above the grasp point and directly in the falling
        # object's path.
        target_quaternion = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    measured_offset: np.ndarray | None = None
    hand_target = target.copy()
    correction_passes = 0
    if scenario.embodiment == FRANKA_HAND:
        measured_offset = _panda_fingertip_center(model, data, ids) - data.xpos[ids.hand_body]
        hand_target = target - measured_offset

    result = None
    candidate = initial.copy()
    for pass_index in range(2 if scenario.embodiment == FRANKA_HAND else 1):
        correction_passes = pass_index + 1

        def residual(q: np.ndarray) -> np.ndarray:
            data.qpos[arm_qpos] = q
            if scenario.embodiment == FRANKA_HAND:
                data.qpos[list(ids.robot_qpos_adrs[7:9])] = 0.04
            else:
                data.qpos[list(ids.robot_qpos_adrs[7:])] = ROBOTIQ_OPEN_Q
            data.qvel[list(ids.robot_qvel_adrs)] = 0.0
            mujoco.mj_forward(model, data)
            if scenario.embodiment == FRANKA_HAND:
                current_position = np.asarray(data.xpos[ids.hand_body], dtype=np.float64)
            else:
                current_position = _robotiq_thick_pad_center(data, ids)
            position_error = current_position - hand_target
            orientation_error = _quat_error(
                mujoco,
                np.asarray(data.xquat[ids.hand_body], dtype=np.float64),
                target_quaternion,
            )
            orientation_weight = 0.8 if scenario.embodiment == FRANKA_HAND else 0.25
            regularization_weight = 0.005 if scenario.embodiment == FRANKA_HAND else 0.02
            return np.r_[
                4.0 * position_error,
                orientation_weight * orientation_error,
                regularization_weight * (q - initial),
            ]

        result = least_squares(
            residual,
            candidate,
            bounds=(ranges[:, 0], ranges[:, 1]),
            max_nfev=600,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        candidate = np.asarray(result.x, dtype=np.float64)
        data.qpos[arm_qpos] = candidate
        mujoco.mj_forward(model, data)
        if scenario.embodiment == FRANKA_HAND:
            measured_offset = _panda_fingertip_center(model, data, ids) - data.xpos[ids.hand_body]
            hand_target = target - measured_offset

    data.qpos[arm_qpos] = candidate
    mujoco.mj_forward(model, data)
    achieved = _grasp_center(model, data, ids, scenario.embodiment)
    assert achieved is not None and result is not None
    error = float(np.linalg.norm(achieved - target))
    success = bool(result.success and np.isfinite(candidate).all() and error <= 0.003)
    diagnostics = IKDiagnostics(
        success=success,
        solver_success=bool(result.success),
        target_position_m=tuple(float(value) for value in target),
        achieved_grasp_center_m=tuple(float(value) for value in achieved),
        position_error_m=error,
        measured_hand_to_fingertip_offset_m=(
            None
            if measured_offset is None
            else tuple(float(value) for value in measured_offset)
        ),
        correction_pass_count=correction_passes,
        message=str(result.message),
    )
    if not success:
        raise RuntimeError(
            f"IK rejected for {scenario.case_id}: solver={result.success}, "
            f"grasp-center error={error:.6f} m"
        )
    return candidate, diagnostics


def _controller_for_scenario(
    mujoco: Any,
    least_squares: Any,
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
) -> tuple[OwnedActuatorController | None, np.ndarray, tuple[IKDiagnostics, ...]]:
    if scenario.embodiment == "no_robot":
        return None, np.empty(0, dtype=np.float64), ()
    assert scenario.controller_target_position_m is not None
    intercept, intercept_diagnostics = _solve_arm_ik(
        mujoco,
        least_squares,
        compiled,
        scenario,
        scenario.controller_target_position_m,
    )
    transport: np.ndarray | None = None
    diagnostics = [intercept_diagnostics]
    if scenario.controller_transport_position_m is not None:
        transport, transport_diagnostics = _solve_arm_ik(
            mujoco,
            least_squares,
            compiled,
            scenario,
            scenario.controller_transport_position_m,
            initial_arm_q=intercept,
        )
        diagnostics.append(transport_diagnostics)
    event_time = float(scenario.ballistic_event_time_s)
    closure_start = max(
        0.0,
        event_time - RIGID_REVIEW_PROFILE.closure_start_before_ballistic_s,
    )
    if scenario.embodiment == ROBOTIQ_2F85_THICK_PAD:
        open_gripper = 0.0
        closed_gripper = RIGID_REVIEW_PROFILE.robotiq_tendon_target
        initial_robot_q = np.r_[intercept, ROBOTIQ_OPEN_Q]
    else:
        open_gripper = 255.0
        desired_finger_q = 0.85 * scenario.object_radius_m
        closed_gripper = 255.0 * desired_finger_q / 0.04
        initial_robot_q = np.r_[intercept, (0.04, 0.04)]
    plan = OwnedControllerPlan(
        embodiment=scenario.embodiment,
        initial_arm_command=tuple(float(value) for value in intercept),
        intercept_arm_command=tuple(float(value) for value in intercept),
        open_gripper_command=open_gripper,
        closed_gripper_command=closed_gripper,
        arm_motion_start_s=0.0,
        arm_motion_end_s=max(0.05, closure_start),
        closure_start_s=closure_start,
        closure_end_s=closure_start + RIGID_REVIEW_PROFILE.closure_duration_s,
        transport_arm_command=(
            None if transport is None else tuple(float(value) for value in transport)
        ),
    )
    return OwnedActuatorController(plan), initial_robot_q, tuple(diagnostics)


def _initialize_state(
    mujoco: Any,
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
    initial_robot_q: np.ndarray,
) -> None:
    data, ids = compiled.data, compiled.ids
    qpos = ids.object_qpos_adr
    qvel = ids.object_qvel_adr
    data.qpos[qpos : qpos + 7] = (
        *scenario.object_initial_position_m,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    data.qvel[qvel : qvel + 6] = (
        *scenario.object_initial_linear_velocity_m_s,
        *scenario.object_initial_angular_velocity_rad_s,
    )
    if scenario.embodiment != "no_robot":
        data.qpos[list(ids.robot_qpos_adrs)] = initial_robot_q
        data.qvel[list(ids.robot_qvel_adrs)] = 0.0
    data.time = 0.0
    mujoco.mj_forward(compiled.model, data)


def _control_observation(compiled: CompiledSourceModel) -> ControlObservation:
    data, ids = compiled.data, compiled.ids
    qpos = ids.object_qpos_adr
    qvel = ids.object_qvel_adr
    return ControlObservation(
        timestamp_s=float(data.time),
        robot_joint_position=tuple(float(data.qpos[index]) for index in ids.robot_qpos_adrs),
        robot_joint_velocity=tuple(float(data.qvel[index]) for index in ids.robot_qvel_adrs),
        object_position_m=tuple(float(value) for value in data.qpos[qpos : qpos + 3]),
        object_linear_velocity_m_s=tuple(
            float(value) for value in data.qvel[qvel : qvel + 3]
        ),
    )


def _geom_name(mujoco: Any, model: Any, geom_id: int) -> str:
    return str(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        or f"unnamed_geom_{geom_id}"
    )


def _contact_category(other: int, name: str, ids: Any) -> str:
    if other in ids.left_gripper_geom_ids or other in ids.right_gripper_geom_ids:
        return "gripper"
    if other in ids.surface_geom_ids.values() or any(
        token in name.lower() for token in ("floor", "table", "wall", "ramp", "surface")
    ):
        return "task_surface"
    if name.startswith(("rc_", "rw_", "robocasa_")):
        return "background"
    return "unknown"


def _contacts_at_state(
    mujoco: Any,
    compiled: CompiledSourceModel,
) -> tuple[list[dict[str, Any]], bool, bool]:
    model, data, ids = compiled.model, compiled.data, compiled.ids
    rows: list[dict[str, Any]] = []
    left = right = False
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if int(contact.geom1) != ids.object_geom and int(contact.geom2) != ids.object_geom:
            continue
        object_is_geom1 = int(contact.geom1) == ids.object_geom
        other = int(contact.geom2 if object_is_geom1 else contact.geom1)
        left = left or other in ids.left_gripper_geom_ids
        right = right or other in ids.right_gripper_geom_ids
        normal = np.asarray(contact.frame[:3], dtype=np.float64).copy()
        # MuJoCo points from geom1 to geom2; canonical normal points from the
        # counterpart surface/gripper toward the task object.
        if object_is_geom1:
            normal *= -1.0
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        point = np.asarray(contact.pos, dtype=np.float64).copy()
        object_velocity = np.zeros(6, dtype=np.float64)
        other_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_GEOM,
            ids.object_geom,
            object_velocity,
            0,
        )
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_GEOM,
            other,
            other_velocity,
            0,
        )
        object_point_velocity = object_velocity[3:] + np.cross(
            object_velocity[:3], point - data.geom_xpos[ids.object_geom]
        )
        other_point_velocity = other_velocity[3:] + np.cross(
            other_velocity[:3], point - data.geom_xpos[other]
        )
        relative_velocity = object_point_velocity - other_point_velocity
        other_name = _geom_name(mujoco, model, other)
        other_body_id = int(model.geom_bodyid[other])
        other_body_name = str(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body_id)
            or f"unnamed_body_{other_body_id}"
        )
        category = _contact_category(other, other_name, ids)
        if category == "unknown" and other_body_name == "hand":
            category = "gripper"
        elif category == "unknown" and other_body_name.startswith(
            ("link", "rq_")
        ):
            category = "robot_arm"
        rows.append(
            {
                "timestamp": float(data.time),
                "object_a": "catch_ball",
                "object_b": other_name,
                "geom_a": "catch_ball_geom",
                "geom_b": other_name,
                "counterpart": other_name,
                "counterpart_geom_id": other,
                "contact_category": category,
                "counterpart_body": other_body_name,
                "penetration_depth_m": max(0.0, -float(contact.dist)),
                "distance_m": float(contact.dist),
                "point_world_m": [float(value) for value in point],
                "normal_world": [float(value) for value in normal],
                "normal_force_n": max(0.0, float(force[0])),
                "normal_impulse_n_s": max(
                    0.0, float(force[0]) * float(model.opt.timestep)
                ),
                "relative_velocity_world_m_s": [
                    float(value) for value in relative_velocity
                ],
                "expected_fixture_contact": category == "task_surface",
                "snag": False,
                "contact_role": "task_contact",
            }
        )
    return rows, left, right


def _solver_warning_count(data: Any) -> int:
    try:
        return int(sum(int(value.number) for value in data.warning))
    except (AttributeError, TypeError):
        return 0


def _state_row(
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
    action: np.ndarray,
    contacts: Sequence[Mapping[str, Any]],
    left: bool,
    right: bool,
) -> dict[str, Any]:
    data, ids = compiled.data, compiled.ids
    qpos, qvel = ids.object_qpos_adr, ids.object_qvel_adr
    grasp_center = _grasp_center(compiled.model, data, ids, scenario.embodiment)
    categories = {str(row["contact_category"]) for row in contacts}
    if "gripper" in categories or "robot_arm" in categories:
        motion_mode = "gripper_contact"
    elif "task_surface" in categories:
        motion_mode = "surface_contact"
    else:
        motion_mode = "free_flight"
    actuator_force = (
        []
        if not ids.actuator_ids
        else [float(data.actuator_force[index]) for index in ids.actuator_ids]
    )
    return {
        "timestamp": float(data.time),
        "object.position": [float(value) for value in data.qpos[qpos : qpos + 3]],
        "object.quaternion_wxyz": [float(value) for value in data.qpos[qpos + 3 : qpos + 7]],
        "object.linear_velocity": [float(value) for value in data.qvel[qvel : qvel + 3]],
        "object.angular_velocity": [float(value) for value in data.qvel[qvel + 3 : qvel + 6]],
        "object.motion_mode": motion_mode,
        "robot.joint_position": [float(data.qpos[index]) for index in ids.robot_qpos_adrs],
        "robot.joint_velocity": [float(data.qvel[index]) for index in ids.robot_qvel_adrs],
        "robot.actuator_force": actuator_force,
        ACTION_FIELD: [float(value) for value in action],
        "simulator.applied_actuator_ctrl": (
            []
            if not ids.actuator_ids
            else [float(data.ctrl[index]) for index in ids.actuator_ids]
        ),
        "action.mode": (
            "no_actuators/v1"
            if scenario.embodiment == "no_robot"
            else ACTION_SEMANTICS
        ),
        "contact.count": len(contacts),
        "contact.left_gripper": bool(left),
        "contact.right_gripper": bool(right),
        "contact.bilateral": bool(left and right),
        "grasp.center_position": (
            None if grasp_center is None else [float(value) for value in grasp_center]
        ),
    }


def _camera_calibration(
    model: Any,
    data: Any,
    camera_id: int,
    camera_name: str,
) -> CameraCalibration:
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    position = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
    right = rotation[:, 0]
    up = rotation[:, 1]
    forward = -rotation[:, 2]
    down = -up
    world_to_camera = (
        *right,
        -float(np.dot(right, position)),
        *down,
        -float(np.dot(down, position)),
        *forward,
        -float(np.dot(forward, position)),
        0.0,
        0.0,
        0.0,
        1.0,
    )
    camera_to_world = (
        right[0], down[0], forward[0], position[0],
        right[1], down[1], forward[1], position[1],
        right[2], down[2], forward[2], position[2],
        0.0, 0.0, 0.0, 1.0,
    )
    fovy = float(model.cam_fovy[camera_id])
    focal = 0.5 * RIGID_REVIEW_PROFILE.height / math.tan(math.radians(fovy) / 2.0)
    calibration = CameraCalibration(
        camera_name=camera_name,
        intrinsic_matrix=(
            focal,
            0.0,
            RIGID_REVIEW_PROFILE.width / 2.0,
            0.0,
            focal,
            RIGID_REVIEW_PROFILE.height / 2.0,
            0.0,
            0.0,
            1.0,
        ),
        world_to_camera=tuple(float(value) for value in world_to_camera),
        camera_to_world=tuple(float(value) for value in camera_to_world),
        width=RIGID_REVIEW_PROFILE.width,
        height=RIGID_REVIEW_PROFILE.height,
        fps=float(RIGID_REVIEW_PROFILE.video_hz),
        near_m=0.03,
        far_m=20.0,
        renderer="mujoco.Renderer",
    )
    calibration.validate()
    return calibration


def _segmentation_mask_summary(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"pixel_count": 0, "bbox_xyxy": None}
    return {
        "pixel_count": int(len(xs)),
        "bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ],
    }


def _render_segmentation_observation(
    mujoco: Any,
    renderer: Any,
    rgb: np.ndarray,
    *,
    object_geom_id: int,
    left_tool_geom_ids: Sequence[int],
    right_tool_geom_ids: Sequence[int],
    fixture_geom_ids: Sequence[int],
) -> dict[str, Any]:
    """Render target/counterpart IDs from the exact persisted RGB scene."""

    renderer.enable_segmentation_rendering()
    try:
        segmentation = np.asarray(renderer.render(), dtype=np.int32).copy()
    finally:
        renderer.disable_segmentation_rendering()
    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise RuntimeError(
            "MuJoCo segmentation rendering did not return HxWx2 object IDs"
        )
    geom_ids = segmentation[:, :, 0]
    geom_type = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    object_mask = geom_type & (geom_ids == int(object_geom_id))
    left_tool_ids = tuple(int(value) for value in left_tool_geom_ids)
    right_tool_ids = tuple(int(value) for value in right_tool_geom_ids)
    tool_ids = (*left_tool_ids, *right_tool_ids)
    left_tool_mask = geom_type & np.isin(geom_ids, left_tool_ids)
    right_tool_mask = geom_type & np.isin(geom_ids, right_tool_ids)
    tool_mask = geom_type & np.isin(geom_ids, tool_ids)
    fixture_mask = geom_type & np.isin(
        geom_ids, tuple(int(value) for value in fixture_geom_ids)
    )
    visible_geom_ids, visible_geom_counts = np.unique(
        geom_ids[geom_type], return_counts=True
    )
    luminance = rgb.astype(np.float32).mean(axis=2)
    return {
        "object": _segmentation_mask_summary(object_mask),
        "tool": _segmentation_mask_summary(tool_mask),
        "tool_left": _segmentation_mask_summary(left_tool_mask),
        "tool_right": _segmentation_mask_summary(right_tool_mask),
        "fixture": _segmentation_mask_summary(fixture_mask),
        "geom_pixel_counts": {
            str(int(geom_id)): int(count)
            for geom_id, count in zip(
                visible_geom_ids.tolist(), visible_geom_counts.tolist()
            )
        },
        "mean_luminance": float(luminance.mean()),
        "underexposed_fraction": float(np.mean(luminance <= 3.0)),
        "overexposed_fraction": float(np.mean(luminance >= 252.0)),
    }


def _bbox_margin_px(
    summary: Mapping[str, Any], calibration: CameraCalibration
) -> float:
    bbox = summary.get("bbox_xyxy")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return -1.0
    return float(
        min(
            int(bbox[0]),
            int(bbox[1]),
            calibration.width - int(bbox[2]),
            calibration.height - int(bbox[3]),
        )
    )


def _projected_sphere(
    calibration: CameraCalibration,
    position_m: Sequence[float],
    radius_m: float,
) -> tuple[float, float, float, float]:
    """Return ``x, y, depth, edge_margin`` for the complete target sphere."""

    pixel_x, pixel_y, depth_m = calibration.project_world(position_m)
    focal_px = float(calibration.intrinsic_matrix[0])
    projected_radius_px = focal_px * float(radius_m) / depth_m
    margin = min(
        pixel_x - projected_radius_px,
        pixel_y - projected_radius_px,
        calibration.width - pixel_x - projected_radius_px,
        calibration.height - pixel_y - projected_radius_px,
    )
    return pixel_x, pixel_y, depth_m, float(margin)


def _compiled_tool_visibility_topology(
    compiled: CompiledSourceModel,
) -> dict[str, Any]:
    """Return the canonical rendered/contact topology from the compiled model."""

    left_geom_ids = sorted(int(value) for value in compiled.ids.left_gripper_geom_ids)
    right_geom_ids = sorted(int(value) for value in compiled.ids.right_gripper_geom_ids)
    tool_body_ids = {
        int(compiled.model.geom_bodyid[geom_id])
        for geom_id in (*left_geom_ids, *right_geom_ids)
    }
    if compiled.ids.hand_body is not None:
        tool_body_ids.add(int(compiled.ids.hand_body))
    geom_body_ids = {
        str(geom_id): int(compiled.model.geom_bodyid[geom_id])
        for geom_id in range(int(compiled.model.ngeom))
        if int(compiled.model.geom_bodyid[geom_id]) in tool_body_ids
    }
    return {
        "schema_version": SOURCE_MUJOCO_TOOL_VISIBILITY_TOPOLOGY_SCHEMA,
        "tool_geom_body_ids": geom_body_ids,
        "left_tool_geom_ids": left_geom_ids,
        "right_tool_geom_ids": right_geom_ids,
        "left_tool_body_id": (
            None
            if compiled.ids.left_gripper_body is None
            else int(compiled.ids.left_gripper_body)
        ),
        "right_tool_body_id": (
            None
            if compiled.ids.right_gripper_body is None
            else int(compiled.ids.right_gripper_body)
        ),
    }


def _measured_visibility_key_event(
    scenario: SourceMujocoCompiledScenario,
    state_rows: Sequence[Mapping[str, Any]],
    contact_rows: Sequence[Mapping[str, Any]],
    *,
    left_tool_geom_ids: Sequence[int] = (),
    right_tool_geom_ids: Sequence[int] = (),
    tool_geom_body_ids: Mapping[int, int] | None = None,
    left_tool_body_id: int | None = None,
    right_tool_body_id: int | None = None,
    tool_visibility_topology_sha256: str | None = None,
) -> dict[str, Any]:
    """Select the same measured F1 event used by persisted objective replay."""

    planned_time_s = float(scenario.key_event_time_s)
    selected = select_source_key_event(
        planned_key_event_name="task_interaction",
        planned_key_event_time_s=planned_time_s,
        state_rows=state_rows,
        event_rows=contact_rows,
        passive=scenario.embodiment == "no_robot",
        contact_time_tolerance_s=1.0 / scenario.simulation_hz,
    )
    physical_contact_applicable = selected["physical_contact_applicable"]
    counterpart_geom_ids = list(selected["contact_counterpart_geom_ids"])
    if scenario.embodiment == "no_robot":
        nearby_fixture_contacts = [
            row
            for row in contact_rows
            if row.get("contact_category") == "task_surface"
            and isinstance(row.get("timestamp"), (int, float))
            and not isinstance(row.get("timestamp"), bool)
            and abs(float(row["timestamp"]) - planned_time_s)
            <= 1.0 / scenario.video_hz + 1e-12
        ]
        physical_contact_applicable = bool(nearby_fixture_contacts)
        counterpart_geom_ids = sorted(
            {
                int(row["counterpart_geom_id"])
                for row in nearby_fixture_contacts
                if isinstance(row.get("counterpart_geom_id"), int)
                and not isinstance(row.get("counterpart_geom_id"), bool)
            }
        )
    return {
        "planned_key_event_time_s": planned_time_s,
        "planned_key_event_name": "task_interaction",
        "actual_key_event_time_s": selected["key_event_time_s"],
        "actual_key_event_name": selected["key_event_name"],
        "actual_key_event_source": selected["key_event_source"],
        "physical_contact_applicable": physical_contact_applicable,
        "contact_counterpart_geom_ids": counterpart_geom_ids,
        "left_tool_geom_ids": [int(value) for value in left_tool_geom_ids],
        "right_tool_geom_ids": [int(value) for value in right_tool_geom_ids],
        "tool_geom_body_ids": {
            str(int(key)): int(value)
            for key, value in (tool_geom_body_ids or {}).items()
        },
        "left_tool_body_id": left_tool_body_id,
        "right_tool_body_id": right_tool_body_id,
        "tool_visibility_topology_sha256": tool_visibility_topology_sha256,
    }


def _contact_body_proxy_visibility(
    *,
    counterpart_geom_ids: Sequence[Any],
    tool_geom_body_ids: Mapping[Any, Any],
    key_geom_pixels: Mapping[str, Mapping[str, Any]],
    key_target_visible: Mapping[str, bool],
    required_views: Sequence[str],
    minimum_area_px: int,
) -> dict[str, Any]:
    """Resolve collision geoms to rendered proxies on the exact same tool body.

    Collision-only geoms are often not present in MuJoCo's segmentation
    output.  A contact is nevertheless reviewable when another rendered geom
    rigidly attached to that *same* tool body is visible.  Resolution is kept
    fail-closed: every contacted geom must be in the persisted tool-body map,
    and pixels on a different body never count.
    """

    normalized_body_map: dict[int, int] = {}
    for raw_geom_id, raw_body_id in tool_geom_body_ids.items():
        try:
            if isinstance(raw_geom_id, bool) or isinstance(raw_body_id, bool):
                continue
            geom_id = int(raw_geom_id)
            body_id = int(raw_body_id)
        except (TypeError, ValueError):
            continue
        normalized_body_map[geom_id] = body_id

    contacted_ids: list[int] = []
    invalid_contact_id = False
    for raw_geom_id in counterpart_geom_ids:
        try:
            if isinstance(raw_geom_id, bool):
                raise ValueError
            contacted_ids.append(int(raw_geom_id))
        except (TypeError, ValueError):
            invalid_contact_id = True
    contacted_ids = sorted(set(contacted_ids))
    unresolved_ids = sorted(
        geom_id for geom_id in contacted_ids if geom_id not in normalized_body_map
    )
    resolution_complete = bool(contacted_ids) and not (
        invalid_contact_id or unresolved_ids
    )
    contacted_body_ids = (
        sorted({normalized_body_map[geom_id] for geom_id in contacted_ids})
        if resolution_complete
        else []
    )
    proxy_geom_ids = (
        sorted(
            geom_id
            for geom_id, body_id in normalized_body_map.items()
            if body_id in contacted_body_ids
        )
        if resolution_complete
        else []
    )

    proxy_pixels_by_view: dict[str, int] = {}
    visible_proxy_geom_ids_by_view: dict[str, list[int]] = {}
    for view in required_views:
        pixels = key_geom_pixels.get(view, {})
        visible_proxy_ids = [
            geom_id
            for geom_id in proxy_geom_ids
            if int(pixels.get(str(geom_id), 0)) > 0
        ]
        visible_proxy_geom_ids_by_view[view] = visible_proxy_ids
        proxy_pixels_by_view[view] = sum(
            int(pixels.get(str(geom_id), 0)) for geom_id in proxy_geom_ids
        )

    visible = resolution_complete and any(
        key_target_visible.get(view) is True
        and proxy_pixels_by_view[view] >= minimum_area_px
        for view in required_views
    )
    return {
        "resolution_complete": resolution_complete,
        "unresolved_counterpart_geom_ids": unresolved_ids,
        "counterpart_body_ids": contacted_body_ids,
        "proxy_geom_ids": proxy_geom_ids,
        "visible_proxy_geom_ids_by_view": visible_proxy_geom_ids_by_view,
        "proxy_pixel_counts_by_view": proxy_pixels_by_view,
        "visible": visible,
    }


def _source_visibility_qc(
    scenario: SourceMujocoCompiledScenario,
    frame_rows: Sequence[Mapping[str, Any]],
    calibrations: Mapping[str, CameraCalibration],
    frames: Mapping[str, Sequence[np.ndarray]],
    segmentation_observations: Mapping[str, Sequence[Mapping[str, Any]]],
    key_event: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate rendered target and applicable counterpart visibility.

    Whole-trajectory coverage uses a small segmentation-presence floor across
    either view.  The 64-pixel footprint is reserved for the key event.  Apex
    and final checkpoints are explicit so aggregate coverage cannot hide a
    missing endpoint.
    """

    thresholds = dict(SOURCE_MUJOCO_VISUAL_THRESHOLDS)
    expected_count = len(frame_rows)
    expected_views = ("main", "secondary")
    tool_applicable = scenario.embodiment != "no_robot"
    fixture_applicable = not tool_applicable
    physical_contact_applicable = bool(
        key_event.get("physical_contact_applicable") is True
    )
    streams_complete = bool(expected_count) and all(
        len(frames.get(name, ())) == expected_count
        and len(segmentation_observations.get(name, ())) == expected_count
        and name in calibrations
        for name in expected_views
    )
    result: dict[str, Any] = {
        "schema_version": SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA,
        "evaluated": streams_complete,
        "rendered_streams_complete": streams_complete,
        "rendered_frame_count": expected_count if streams_complete else 0,
        "expected_frame_count": expected_count,
        "required_views": list(expected_views),
        "required_checkpoint_names": ["initial", "apex", "key_event", "final"],
        "target_visible_frame_fraction": 0.0,
        "key_event_visible_in_any_view": False,
        "apex_visible_in_any_view": False,
        "final_state_visible_in_any_view": False,
        "initial_state_visible_in_any_view": False,
        "critically_cropped": True,
        "physical_contact_applicable": physical_contact_applicable,
        "contact_occluded_both_views": None,
        "actual_contact_counterpart_visible_at_key_event": None,
        "contact_body_proxy_resolution_complete": None,
        "contact_counterpart_body_ids": [],
        "contact_proxy_geom_ids": [],
        "contact_visible_proxy_geom_ids_by_view": {},
        "contact_proxy_pixel_counts_by_view": {},
        "unresolved_contact_counterpart_geom_ids": [],
        "contact_exact_fixture_geom_ids": [],
        "contact_visible_fixture_geom_ids_by_view": {},
        "contact_fixture_pixel_counts_by_view": {},
        "bilateral_tool_sides_visible_at_key_event": None,
        "planned_checkpoint_covisible_in_any_view": False,
        "contact_counterpart_geom_ids": list(
            key_event.get("contact_counterpart_geom_ids") or ()
        ),
        "left_tool_geom_ids": list(key_event.get("left_tool_geom_ids") or ()),
        "right_tool_geom_ids": list(key_event.get("right_tool_geom_ids") or ()),
        "tool_geom_body_ids": dict(key_event.get("tool_geom_body_ids") or {}),
        "left_tool_body_id": key_event.get("left_tool_body_id"),
        "right_tool_body_id": key_event.get("right_tool_body_id"),
        "tool_visibility_topology_sha256": key_event.get(
            "tool_visibility_topology_sha256"
        ),
        "minimum_bbox_margin_px": -1.0,
        "key_event_object_area_px": 0,
        "tool_visibility_applicable": tool_applicable,
        "tool_visible_at_key_event": None,
        "fixture_visibility_applicable": fixture_applicable,
        "fixture_visible_at_key_event": None,
        "counterpart_visible_at_key_event": False,
        "camera_roles_correct": set(calibrations) == set(expected_views),
        "maximum_underexposed_fraction": 1.0,
        "maximum_overexposed_fraction": 1.0,
        "thresholds": thresholds,
        "checkpoints": {},
        "views": {},
    }
    if not streams_complete:
        return result

    timestamps = [float(row["timestamp"]) for row in frame_rows]
    planned_key_event_time_s = float(key_event["planned_key_event_time_s"])
    actual_key_event_time_s = float(key_event["actual_key_event_time_s"])
    planned_key_event_index = min(
        range(expected_count),
        key=lambda index: abs(timestamps[index] - planned_key_event_time_s),
    )
    actual_key_event_index = min(
        range(expected_count),
        key=lambda index: abs(timestamps[index] - actual_key_event_time_s),
    )
    checkpoint_indices = {
        "initial": 0,
        "apex": max(
            range(expected_count),
            key=lambda index: float(frame_rows[index]["object.position"][2]),
        ),
        "key_event": actual_key_event_index,
        "final": expected_count - 1,
    }
    result.update(
        {
            "planned_key_event_name": str(
                key_event.get("planned_key_event_name") or "task_interaction"
            ),
            "planned_key_event_time_s": planned_key_event_time_s,
            "planned_key_event_frame_index": planned_key_event_index,
            "planned_key_event_frame_timestamp_s": timestamps[
                planned_key_event_index
            ],
            "actual_key_event_time_s": actual_key_event_time_s,
            "actual_key_event_frame_index": actual_key_event_index,
            "actual_key_event_frame_timestamp_s": timestamps[
                actual_key_event_index
            ],
            "actual_key_event_name": str(key_event["actual_key_event_name"]),
            "actual_key_event_source": str(key_event["actual_key_event_source"]),
        }
    )

    per_view_presence: dict[str, list[bool]] = {}
    checkpoint_visibility = {name: [] for name in checkpoint_indices}
    key_target_visible: dict[str, bool] = {}
    key_tool_pixels: dict[str, int] = {}
    key_left_pixels: dict[str, int] = {}
    key_right_pixels: dict[str, int] = {}
    key_fixture_pixels: dict[str, int] = {}
    key_geom_pixels: dict[str, Mapping[str, Any]] = {}
    planned_target_visible: dict[str, bool] = {}
    planned_counterpart_pixels: dict[str, int] = {}
    key_margins: list[float] = []
    key_areas: list[int] = []
    underexposed: list[float] = []
    overexposed: list[float] = []
    trajectory_margins_by_frame: list[list[float]] = [
        [] for _ in range(expected_count)
    ]

    for name in expected_views:
        calibration = calibrations[name]
        observations = segmentation_observations[name]
        presence: list[bool] = []
        frame_metrics: list[dict[str, Any]] = []
        for index, (row, observation) in enumerate(zip(frame_rows, observations)):
            summaries = {
                key: (
                    observation.get(key)
                    if isinstance(observation.get(key), Mapping)
                    else {}
                )
                for key in ("object", "tool", "tool_left", "tool_right", "fixture")
            }
            object_pixels = int(summaries["object"].get("pixel_count", 0))
            segmentation_margin = _bbox_margin_px(
                summaries["object"], calibration
            )
            try:
                pixel_x, pixel_y, _depth, projected_margin = _projected_sphere(
                    calibration, row["object.position"], scenario.object_radius_m
                )
                projected_center_visible = bool(
                    0.0 <= pixel_x < calibration.width
                    and 0.0 <= pixel_y < calibration.height
                )
            except ValueError:
                projected_margin = -1.0
                projected_center_visible = False
            target_present = bool(
                object_pixels
                >= int(thresholds["minimum_trajectory_object_area_px"])
                and projected_center_visible
            )
            geom_pixel_counts = observation.get("geom_pixel_counts")
            if not isinstance(geom_pixel_counts, Mapping):
                geom_pixel_counts = {}
            metric = {
                "frame_index": index,
                "timestamp_s": timestamps[index],
                "object_pixel_count": object_pixels,
                "segmentation_bbox_margin_px": segmentation_margin,
                "projected_sphere_margin_px": projected_margin,
                "projected_center_visible": projected_center_visible,
                "target_present": target_present,
                "tool_pixel_count": int(summaries["tool"].get("pixel_count", 0)),
                "left_tool_pixel_count": int(
                    summaries["tool_left"].get("pixel_count", 0)
                ),
                "right_tool_pixel_count": int(
                    summaries["tool_right"].get("pixel_count", 0)
                ),
                "fixture_pixel_count": int(
                    summaries["fixture"].get("pixel_count", 0)
                ),
                "geom_pixel_counts": {
                    str(key): int(value) for key, value in geom_pixel_counts.items()
                },
                "underexposed_fraction": float(
                    observation.get("underexposed_fraction", 1.0)
                ),
                "overexposed_fraction": float(
                    observation.get("overexposed_fraction", 1.0)
                ),
            }
            frame_metrics.append(metric)
            presence.append(target_present)
            trajectory_margins_by_frame[index].append(
                min(projected_margin, segmentation_margin)
            )
            underexposed.append(metric["underexposed_fraction"])
            overexposed.append(metric["overexposed_fraction"])

        per_view_presence[name] = presence
        key_metric = frame_metrics[actual_key_event_index]
        key_margin = min(
            key_metric["segmentation_bbox_margin_px"],
            key_metric["projected_sphere_margin_px"],
        )
        key_visible = bool(
            key_metric["object_pixel_count"]
            >= int(thresholds["minimum_key_event_object_area_px"])
            and key_margin >= float(thresholds["minimum_bbox_margin_px"])
        )
        key_target_visible[name] = key_visible
        key_tool_pixels[name] = key_metric["tool_pixel_count"]
        key_left_pixels[name] = key_metric["left_tool_pixel_count"]
        key_right_pixels[name] = key_metric["right_tool_pixel_count"]
        key_fixture_pixels[name] = key_metric["fixture_pixel_count"]
        key_geom_pixels[name] = key_metric["geom_pixel_counts"]
        key_margins.append(key_margin)
        key_areas.append(key_metric["object_pixel_count"])

        planned_metric = frame_metrics[planned_key_event_index]
        planned_margin = min(
            planned_metric["segmentation_bbox_margin_px"],
            planned_metric["projected_sphere_margin_px"],
        )
        planned_target_visible[name] = bool(
            planned_metric["object_pixel_count"]
            >= int(thresholds["minimum_key_event_object_area_px"])
            and planned_margin >= float(thresholds["minimum_bbox_margin_px"])
        )
        planned_counterpart_pixels[name] = int(
            planned_metric[
                "tool_pixel_count" if tool_applicable else "fixture_pixel_count"
            ]
        )

        view_checkpoints: dict[str, Any] = {}
        for checkpoint_name, checkpoint_index in checkpoint_indices.items():
            metric = frame_metrics[checkpoint_index]
            minimum_area = int(
                thresholds["minimum_key_event_object_area_px"]
                if checkpoint_name == "key_event"
                else thresholds["minimum_trajectory_object_area_px"]
            )
            checkpoint_visible = bool(
                metric["object_pixel_count"] >= minimum_area
                and min(
                    metric["projected_sphere_margin_px"],
                    metric["segmentation_bbox_margin_px"],
                )
                >= float(thresholds["minimum_bbox_margin_px"])
            )
            checkpoint_visibility[checkpoint_name].append(checkpoint_visible)
            view_checkpoints[checkpoint_name] = {
                **metric,
                "visible": checkpoint_visible,
            }
        result["views"][name] = {
            "target_visible_frame_fraction": sum(presence) / expected_count,
            "minimum_object_area_px": min(
                (row["object_pixel_count"] for row in frame_metrics), default=0
            ),
            "minimum_segmentation_bbox_margin_px": min(
                (row["segmentation_bbox_margin_px"] for row in frame_metrics),
                default=-1.0,
            ),
            "minimum_projected_sphere_margin_px": min(
                (row["projected_sphere_margin_px"] for row in frame_metrics),
                default=-1.0,
            ),
            "key_event_object_pixel_count": key_metric["object_pixel_count"],
            "key_event_bbox_margin_px": key_margin,
            "key_event_tool_pixel_count": key_tool_pixels[name],
            "key_event_left_tool_pixel_count": key_left_pixels[name],
            "key_event_right_tool_pixel_count": key_right_pixels[name],
            "key_event_fixture_pixel_count": key_fixture_pixels[name],
            "frames": frame_metrics,
            "checkpoints": view_checkpoints,
        }

    visible_any_view_by_frame = [
        any(per_view_presence[name][index] for name in expected_views)
        for index in range(expected_count)
    ]
    result["target_visible_frame_fraction"] = (
        sum(visible_any_view_by_frame) / expected_count
    )
    result["initial_state_visible_in_any_view"] = any(
        checkpoint_visibility["initial"]
    )
    result["apex_visible_in_any_view"] = any(checkpoint_visibility["apex"])
    result["key_event_visible_in_any_view"] = any(
        checkpoint_visibility["key_event"]
    )
    result["final_state_visible_in_any_view"] = any(
        checkpoint_visibility["final"]
    )
    result["critically_cropped"] = not result["key_event_visible_in_any_view"]
    result["minimum_bbox_margin_px"] = max(key_margins, default=-1.0)
    result["key_event_object_area_px"] = max(key_areas, default=0)

    counterpart_threshold = int(thresholds["minimum_counterpart_area_px"])
    planned_counterpart_threshold = int(
        thresholds["minimum_planned_counterpart_area_px"]
    )
    if tool_applicable:
        counterpart_pixels = key_tool_pixels
        applicable_counterpart_threshold = (
            counterpart_threshold
            if physical_contact_applicable
            else planned_counterpart_threshold
        )
        tool_visible = any(
            value >= applicable_counterpart_threshold
            for value in counterpart_pixels.values()
        )
        result["tool_visible_at_key_event"] = tool_visible
        result["counterpart_visible_at_key_event"] = tool_visible
    else:
        counterpart_pixels = key_fixture_pixels
        fixture_visible = any(
            value >= counterpart_threshold for value in counterpart_pixels.values()
        )
        result["fixture_visible_at_key_event"] = fixture_visible
        result["counterpart_visible_at_key_event"] = fixture_visible

    result["planned_checkpoint_covisible_in_any_view"] = any(
        planned_target_visible[name]
        and planned_counterpart_pixels[name] >= planned_counterpart_threshold
        for name in expected_views
    )
    if physical_contact_applicable:
        event_source = str(key_event["actual_key_event_source"])
        if not tool_applicable:
            raw_counterpart_ids = result["contact_counterpart_geom_ids"]
            fixture_geom_ids = sorted(
                {
                    int(geom_id)
                    for geom_id in raw_counterpart_ids
                    if isinstance(geom_id, int) and not isinstance(geom_id, bool)
                }
            )
            identity_complete = bool(fixture_geom_ids) and len(
                fixture_geom_ids
            ) == len(raw_counterpart_ids)
            fixture_pixels_by_view = {
                name: sum(
                    int(key_geom_pixels[name].get(str(geom_id), 0))
                    for geom_id in fixture_geom_ids
                )
                for name in expected_views
            }
            visible_fixture_ids_by_view = {
                name: [
                    geom_id
                    for geom_id in fixture_geom_ids
                    if int(key_geom_pixels[name].get(str(geom_id), 0)) > 0
                ]
                for name in expected_views
            }
            contact_visible = identity_complete and any(
                key_target_visible[name]
                and fixture_pixels_by_view[name] >= counterpart_threshold
                for name in expected_views
            )
            result.update(
                {
                    "contact_exact_fixture_geom_ids": fixture_geom_ids,
                    "contact_visible_fixture_geom_ids_by_view": (
                        visible_fixture_ids_by_view
                    ),
                    "contact_fixture_pixel_counts_by_view": (
                        fixture_pixels_by_view
                    ),
                }
            )
        elif event_source == "persisted_bilateral_contact":
            counterpart_ids = set(result["contact_counterpart_geom_ids"])
            contacted_left_ids = counterpart_ids.intersection(
                result["left_tool_geom_ids"]
            )
            contacted_right_ids = counterpart_ids.intersection(
                result["right_tool_geom_ids"]
            )
            contact_visible = any(
                key_target_visible[name]
                and key_left_pixels[name] >= counterpart_threshold
                and key_right_pixels[name] >= counterpart_threshold
                for name in expected_views
            ) if contacted_left_ids and contacted_right_ids else False
            result["bilateral_tool_sides_visible_at_key_event"] = contact_visible
        else:
            proxy_visibility = _contact_body_proxy_visibility(
                counterpart_geom_ids=result["contact_counterpart_geom_ids"],
                tool_geom_body_ids=result["tool_geom_body_ids"],
                key_geom_pixels=key_geom_pixels,
                key_target_visible=key_target_visible,
                required_views=expected_views,
                minimum_area_px=counterpart_threshold,
            )
            contact_visible = bool(proxy_visibility["visible"])
            result.update(
                {
                    "contact_body_proxy_resolution_complete": proxy_visibility[
                        "resolution_complete"
                    ],
                    "contact_counterpart_body_ids": proxy_visibility[
                        "counterpart_body_ids"
                    ],
                    "contact_proxy_geom_ids": proxy_visibility["proxy_geom_ids"],
                    "contact_visible_proxy_geom_ids_by_view": proxy_visibility[
                        "visible_proxy_geom_ids_by_view"
                    ],
                    "contact_proxy_pixel_counts_by_view": proxy_visibility[
                        "proxy_pixel_counts_by_view"
                    ],
                    "unresolved_contact_counterpart_geom_ids": proxy_visibility[
                        "unresolved_counterpart_geom_ids"
                    ],
                }
            )
        result["actual_contact_counterpart_visible_at_key_event"] = contact_visible
        result["contact_occluded_both_views"] = not contact_visible

    result["maximum_underexposed_fraction"] = max(underexposed, default=1.0)
    result["maximum_overexposed_fraction"] = max(overexposed, default=1.0)
    result["minimum_trajectory_bbox_margin_px_any_view"] = min(
        (max(values) for values in trajectory_margins_by_frame), default=-1.0
    )
    result["checkpoints"] = {
        checkpoint_name: {
            "frame_index": checkpoint_index,
            "timestamp_s": timestamps[checkpoint_index],
            "visible_in_any_view": any(checkpoint_visibility[checkpoint_name]),
        }
        for checkpoint_name, checkpoint_index in checkpoint_indices.items()
    }
    return result


def _free_flight_energy_drift(
    rows: Sequence[Mapping[str, Any]],
    scenario: SourceMujocoCompiledScenario,
) -> tuple[bool, float]:
    # Contact can legitimately change mechanical energy.  Measure only the
    # initial contiguous free-flight interval, never a post-impact segment.
    free: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("object.motion_mode") != "free_flight":
            break
        free.append(row)
    if len(free) < 3:
        return False, 0.0
    inertia = 0.4 * scenario.object_mass_kg * scenario.object_radius_m**2
    gravity = np.asarray(scenario.gravity_m_s2, dtype=np.float64)
    values = []
    for row in free:
        position = np.asarray(row["object.position"], dtype=np.float64)
        velocity = np.asarray(row["object.linear_velocity"], dtype=np.float64)
        angular = np.asarray(row["object.angular_velocity"], dtype=np.float64)
        values.append(
            0.5 * scenario.object_mass_kg * float(np.dot(velocity, velocity))
            + 0.5 * inertia * float(np.dot(angular, angular))
            - scenario.object_mass_kg * float(np.dot(gravity, position))
        )
    scale = max(abs(float(np.mean(values))), 1e-9)
    return True, float((max(values) - min(values)) / scale)


def _restitution_evidence(
    rows: Sequence[Mapping[str, Any]],
    contacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    surface = [row for row in contacts if row.get("contact_category") == "task_surface"]
    if not surface:
        return {
            "applicable": False,
            "separated_pre_post_contact_samples": False,
            "measured_contact_normal": False,
            "effective_restitution": None,
            "no_unexplained_contact_energy_gain": False,
        }
    event_time = min(float(row["timestamp"]) for row in surface)
    event_rows = [row for row in surface if abs(float(row["timestamp"]) - event_time) < 1e-9]
    normal = np.mean(
        np.asarray([row["normal_world"] for row in event_rows], dtype=np.float64), axis=0
    )
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return {
            "applicable": True,
            "separated_pre_post_contact_samples": False,
            "measured_contact_normal": False,
            "effective_restitution": None,
            "no_unexplained_contact_energy_gain": False,
        }
    normal /= norm
    incoming = next(
        (
            row
            for row in reversed(rows)
            if float(row["timestamp"]) < event_time
            and int(row["contact.count"]) == 0
        ),
        None,
    )
    outgoing = next(
        (
            row
            for row in rows
            if float(row["timestamp"]) > event_time
            and int(row["contact.count"]) == 0
        ),
        None,
    )
    if incoming is None or outgoing is None:
        final_velocity = np.asarray(rows[-1]["object.linear_velocity"], dtype=np.float64)
        initial_supported_contact = bool(
            incoming is None
            and rows
            and abs(event_time - float(rows[0]["timestamp"])) <= 1e-9
            and int(rows[0]["contact.count"]) > 0
        )
        settled_static_contact = bool(
            (incoming is not None or initial_supported_contact)
            and int(rows[-1]["contact.count"]) > 0
            and abs(float(np.dot(final_velocity, normal))) <= 0.1
        )
        return {
            "applicable": True,
            "separated_pre_post_contact_samples": False,
            "measured_contact_normal": True,
            "effective_restitution": 0.0 if settled_static_contact else None,
            "no_unexplained_contact_energy_gain": settled_static_contact,
        }
    incoming_velocity = np.asarray(incoming["object.linear_velocity"], dtype=np.float64)
    outgoing_velocity = np.asarray(outgoing["object.linear_velocity"], dtype=np.float64)
    vn_in = float(np.dot(incoming_velocity, normal))
    vn_out = float(np.dot(outgoing_velocity, normal))
    effective = vn_out / -vn_in if vn_in < -1e-4 and vn_out > 0 else math.inf
    return {
        "applicable": True,
        "separated_pre_post_contact_samples": True,
        "measured_contact_normal": True,
        "event_time_s": event_time,
        "event_position_m": list(incoming["object.position"]),
        "incoming_normal_velocity_m_s": vn_in,
        "outgoing_normal_velocity_m_s": vn_out,
        "effective_restitution": effective,
        "no_unexplained_contact_energy_gain": bool(math.isfinite(effective) and effective <= 1.05),
    }


def _catch_evidence(
    rows: Sequence[Mapping[str, Any]],
    scenario: SourceMujocoCompiledScenario,
) -> dict[str, Any]:
    bilateral = [row for row in rows if row.get("contact.bilateral") is True]
    maximum_run = current_run = 0
    for row in rows:
        current_run = current_run + 1 if row.get("contact.bilateral") is True else 0
        maximum_run = max(maximum_run, current_run)
    sustained = maximum_run / scenario.simulation_hz >= 0.05
    relative = []
    for row in bilateral:
        center = row.get("grasp.center_position")
        if center is None:
            continue
        relative.append(
            np.asarray(row["object.position"], dtype=np.float64)
            - np.asarray(center, dtype=np.float64)
        )
    stable = False
    maximum_relative_range = math.inf
    if len(relative) >= int(round(0.10 * scenario.simulation_hz)):
        recent = np.asarray(relative[-int(round(0.10 * scenario.simulation_hz)) :])
        maximum_relative_range = float(np.max(np.ptp(recent, axis=0)))
        stable = maximum_relative_range <= 0.01
    transport_supported = True
    if scenario.controller_transport_position_m is not None:
        transport_rows = [
            row
            for row in rows
            if 1.0 <= float(row["timestamp"]) <= 1.6
        ]
        bilateral_fraction = (
            sum(row.get("contact.bilateral") is True for row in transport_rows)
            / len(transport_rows)
            if transport_rows
            else 0.0
        )
        displacement = (
            math.dist(
                transport_rows[0]["object.position"],
                transport_rows[-1]["object.position"],
            )
            if len(transport_rows) >= 2
            else 0.0
        )
        transport_supported = bilateral_fraction >= 0.75 and displacement >= 0.06
    return {
        "sustained_opposing_bilateral_contacts": sustained,
        "stable_object_to_grasp_transform": stable,
        "displacement_physically_supported_by_contacts": transport_supported,
        "maximum_contiguous_bilateral_contact_s": maximum_run / scenario.simulation_hz,
        "maximum_object_to_grasp_relative_range_m": (
            None if not math.isfinite(maximum_relative_range) else maximum_relative_range
        ),
    }


def _rolling_evidence(
    rows: Sequence[Mapping[str, Any]],
    scenario: SourceMujocoCompiledScenario,
) -> dict[str, Any]:
    contacted = [row for row in rows if row.get("object.motion_mode") == "surface_contact"]
    if len(contacted) < 3 or not scenario.surfaces:
        return {
            "rolling_or_sliding_slip_within_limit": False,
            "friction_deceleration_consistent": False,
            "maximum_slip_m_s": None,
            "measured_tangent_acceleration_m_s2": None,
            "expected_rolling_tangent_acceleration_m_s2": None,
            "maximum_mechanical_energy_gain_fraction": None,
        }
    slips = []
    for row in contacted:
        vx, vy, _ = (float(value) for value in row["object.linear_velocity"])
        wx, wy, _ = (float(value) for value in row["object.angular_velocity"])
        slips.append(
            math.hypot(
                vx - wy * scenario.object_radius_m,
                vy + wx * scenario.object_radius_m,
            )
        )
    maximum_slip = max(slips)

    # Evaluate signed motion along the physical surface tangent.  A sphere
    # rolling uphill can slow, reverse, and finish faster downhill without any
    # non-physical energy gain; comparing only endpoint speed rejects that
    # valid trajectory.  MuJoCo's Y Euler rotation maps local +X to this world
    # tangent for the owned straight/slope fixtures.
    pitch = float(scenario.surfaces[0].euler_rad[1])
    tangent = np.asarray((math.cos(pitch), 0.0, -math.sin(pitch)), dtype=np.float64)
    timestamps = np.asarray(
        [float(row["timestamp"]) for row in contacted], dtype=np.float64
    )
    tangent_velocity = np.asarray(
        [
            float(
                np.dot(
                    np.asarray(row["object.linear_velocity"], dtype=np.float64),
                    tangent,
                )
            )
            for row in contacted
        ],
        dtype=np.float64,
    )
    measured_acceleration = float(
        np.polyfit(timestamps - timestamps[0], tangent_velocity, 1)[0]
    )
    expected_acceleration = float(
        (5.0 / 7.0)
        * np.dot(np.asarray(scenario.gravity_m_s2, dtype=np.float64), tangent)
    )
    acceleration_tolerance = max(0.12, 0.15 * abs(expected_acceleration))
    acceleration_consistent = bool(
        math.isfinite(measured_acceleration)
        and abs(measured_acceleration - expected_acceleration)
        <= acceleration_tolerance
    )

    mass = scenario.object_mass_kg
    inertia = 0.4 * mass * scenario.object_radius_m**2
    gravity = np.asarray(scenario.gravity_m_s2, dtype=np.float64)
    energies = []
    for row in contacted:
        position = np.asarray(row["object.position"], dtype=np.float64)
        linear = np.asarray(row["object.linear_velocity"], dtype=np.float64)
        angular = np.asarray(row["object.angular_velocity"], dtype=np.float64)
        energies.append(
            0.5 * mass * float(np.dot(linear, linear))
            + 0.5 * inertia * float(np.dot(angular, angular))
            - mass * float(np.dot(gravity, position))
        )
    energy_scale = max(abs(float(energies[0])), 1e-9)
    maximum_energy_gain = max(
        0.0, (max(energies) - float(energies[0])) / energy_scale
    )
    energy_consistent = bool(
        math.isfinite(maximum_energy_gain)
        and maximum_energy_gain
        <= RIGID_REVIEW_PROFILE.maximum_free_flight_energy_drift_fraction
    )
    return {
        "rolling_or_sliding_slip_within_limit": maximum_slip <= 0.12,
        "friction_deceleration_consistent": acceleration_consistent
        and energy_consistent,
        "maximum_slip_m_s": maximum_slip,
        "measured_tangent_acceleration_m_s2": measured_acceleration,
        "expected_rolling_tangent_acceleration_m_s2": expected_acceleration,
        "maximum_mechanical_energy_gain_fraction": maximum_energy_gain,
    }


def _energy_drift_within_limit(value: float) -> bool:
    return bool(
        math.isfinite(value)
        and abs(value)
        <= RIGID_REVIEW_PROFILE.maximum_free_flight_energy_drift_fraction
    )


def _effective_restitution_within_limit(value: Any) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(number)
        and 0.0 <= number <= RIGID_REVIEW_PROFILE.maximum_effective_restitution
    )


def _physics_qc(
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
    rows: Sequence[Mapping[str, Any]],
    contacts: Sequence[Mapping[str, Any]],
    runtime_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    penetration = strict_contact_penetration_check(contacts, require_classification=True)
    energy_applicable, energy_drift = _free_flight_energy_drift(rows, scenario)
    restitution = _restitution_evidence(rows, contacts)
    catch = _catch_evidence(rows, scenario) if scenario.embodiment != "no_robot" else {}
    rolling = (
        _rolling_evidence(rows, scenario)
        if "roll" in scenario.motion_kind
        else {}
    )
    task_evidence: dict[str, Any] = {}
    if scenario.embodiment != "no_robot":
        task_evidence.update(catch)
    if "rebound" in scenario.motion_kind or "bounce" in scenario.motion_kind:
        task_evidence.update(
            {
                "separated_pre_post_contact_samples": restitution[
                    "separated_pre_post_contact_samples"
                ],
                "measured_contact_normal": restitution["measured_contact_normal"],
                "effective_restitution_within_limit": bool(
                    _effective_restitution_within_limit(
                        restitution.get("effective_restitution")
                    )
                ),
                "no_unexplained_contact_energy_gain": restitution[
                    "no_unexplained_contact_energy_gain"
                ],
            }
        )
    elif "roll" in scenario.motion_kind:
        task_evidence.update(rolling)
    elif scenario.embodiment == "no_robot":
        task_evidence["family_specific_physics_evaluator_passed"] = bool(
            energy_applicable
            and _energy_drift_within_limit(energy_drift)
        )

    finite = all(
        np.isfinite(
            np.asarray(
                [
                    *row["object.position"],
                    *row["object.linear_velocity"],
                    *row["object.angular_velocity"],
                    *row["robot.joint_position"],
                    *row["robot.joint_velocity"],
                    *row[ACTION_FIELD],
                ],
                dtype=np.float64,
            )
        ).all()
        for row in rows
    )
    joint_velocities = [
        abs(float(value)) for row in rows for value in row["robot.joint_velocity"]
    ]
    maximum_joint_velocity = max(joint_velocities, default=0.0)
    maximum_joint_acceleration = 0.0
    for left, right in zip(rows, rows[1:]):
        if not left["robot.joint_velocity"]:
            continue
        dt = float(right["timestamp"]) - float(left["timestamp"])
        maximum_joint_acceleration = max(
            maximum_joint_acceleration,
            max(
                abs(float(b) - float(a)) / dt
                for a, b in zip(
                    left["robot.joint_velocity"], right["robot.joint_velocity"]
                )
            ),
        )
    force_ok = True
    for row in rows:
        for actuator_id, force in zip(
            compiled.ids.actuator_ids, row["robot.actuator_force"]
        ):
            if bool(compiled.model.actuator_forcelimited[actuator_id]):
                low, high = compiled.model.actuator_forcerange[actuator_id]
                if not float(low) - 1e-8 <= float(force) <= float(high) + 1e-8:
                    force_ok = False
    joint_ok = maximum_joint_velocity <= 3.5 and maximum_joint_acceleration <= 80.0
    checks = {
        "finite_state": finite,
        "no_solver_warnings": int(runtime_audit["solver_warning_count"]) == 0,
        "no_tunneling": int(runtime_audit["tunneling_event_count"]) == 0,
        "no_unexplained_velocity_discontinuity": int(
            runtime_audit["unexplained_velocity_discontinuity_count"]
        )
        == 0,
        "no_mutation_boundary_violation": int(
            runtime_audit["mutation_boundary_violations"]
        )
        == 0,
        "no_applied_forces": int(runtime_audit["applied_force_writes_after_initialization"])
        == 0,
        "no_object_linked_equality_or_latch_assistance": int(
            runtime_audit["object_linked_equality_changes_after_initialization"]
        )
        == 0,
        "actuator_forces_within_model_limits": force_ok,
        "joint_motion_within_model_limits": joint_ok,
        "free_flight_energy_applicable": bool(energy_applicable),
        "free_flight_relative_energy_drift": energy_drift,
        "static_restitution_applicable": bool(restitution["applicable"]),
        "maximum_measured_effective_restitution": (
            0.0
            if restitution.get("effective_restitution") is None
            else float(restitution["effective_restitution"])
        ),
    }
    common_pass = all(
        bool(checks[name])
        for name in (
            "finite_state",
            "no_solver_warnings",
            "no_tunneling",
            "no_unexplained_velocity_discontinuity",
            "no_mutation_boundary_violation",
            "no_applied_forces",
            "no_object_linked_equality_or_latch_assistance",
            "actuator_forces_within_model_limits",
            "joint_motion_within_model_limits",
        )
    )
    energy_pass = (
        not energy_applicable
        or _energy_drift_within_limit(energy_drift)
    )
    restitution_pass = (
        not restitution["applicable"]
        or _effective_restitution_within_limit(
            restitution.get("effective_restitution")
        )
    )
    if scenario.embodiment == "no_robot":
        passive_task_booleans = [
            value for value in task_evidence.values() if isinstance(value, bool)
        ]
        outcome_success = bool(
            passive_task_booleans
            and all(passive_task_booleans)
            and common_pass
            and penetration.passed
            and energy_pass
            and restitution_pass
        )
    else:
        outcome_success = bool(
            catch.get("sustained_opposing_bilateral_contacts")
            and catch.get("stable_object_to_grasp_transform")
            and catch.get("displacement_physically_supported_by_contacts", True)
        )
    replayed_catch = (
        _catch_evidence(tuple(rows), scenario)
        if scenario.embodiment != "no_robot"
        else {}
    )
    replayed_success = (
        outcome_success
        if scenario.embodiment == "no_robot"
        else bool(
            replayed_catch.get("sustained_opposing_bilateral_contacts")
            and replayed_catch.get("stable_object_to_grasp_transform")
            and replayed_catch.get("displacement_physically_supported_by_contacts", True)
        )
    )
    intended_outcome_match = (
        scenario.intended_outcome == "passive_observation"
        if scenario.embodiment == "no_robot"
        else outcome_success == (scenario.intended_outcome == "success")
    )
    task_evidence.update(
        {
            # These fields bind the objective label to a second evaluation of
            # the persisted state/contact rows.  A physically valid negative
            # is therefore accepted as a negative; it is never forced to look
            # like a successful grasp and is never regenerated to change its
            # label.
            "measured_outcome_replay_matches": replayed_success == outcome_success,
            "saved_artifact_objective_replay_matches": replayed_success
            == outcome_success,
            "measured_failure_matches_persisted_label": bool(
                not outcome_success and not replayed_success
            ),
            "measured_outcome_matches_branch_label": intended_outcome_match,
        }
    )
    task_failures = rigid_task_evidence_failures(
        family=scenario.family,
        subfamily=scenario.subfamily,
        task_variant=scenario.task_variant,
        evidence=task_evidence,
        task_success=outcome_success,
    )
    # Projectile/rebound leaves are still interception tasks even when their
    # taxonomy variant names only the preceding bounce.  Successful examples
    # must retain free-contact grasp evidence; failures use the replay-bound
    # negative evidence selected above.
    if scenario.embodiment != "no_robot" and outcome_success:
        for name in (
            "sustained_opposing_bilateral_contacts",
            "stable_object_to_grasp_transform",
        ):
            if task_evidence.get(name) is not True:
                task_failures.append(
                    f"rigid task evidence is absent or false: {name}"
                )
    if not intended_outcome_match:
        task_failures.append("measured outcome does not match fixed branch label")
    task_pass = not task_failures
    passed = common_pass and penetration.passed and energy_pass and restitution_pass and task_pass
    qc = {
        "schema_version": STRICT_RIGID_QC_SCHEMA,
        "physics_qc_pass": passed,
        "checks": checks,
        "penetration": penetration.to_dict(),
        "restitution": restitution,
        "task_evidence": task_evidence,
        "task_evidence_failures": tuple(task_failures),
        "maximum_joint_velocity_rad_s": maximum_joint_velocity,
        "maximum_joint_acceleration_rad_s2": maximum_joint_acceleration,
        "thresholds": RIGID_REVIEW_PROFILE.to_dict(),
    }
    outcome = {
        "task_success": outcome_success,
        "actual_outcome": (
            "passive_observation"
            if scenario.embodiment == "no_robot"
            else "success"
            if outcome_success
            else "contact_failure"
            if any(row.get("contact_category") == "gripper" for row in contacts)
            else "miss"
        ),
        "intended_outcome": scenario.intended_outcome,
        "intended_outcome_match": intended_outcome_match,
        "measured_outcome_replay_matches": replayed_success == outcome_success,
        "saved_artifact_objective_replay_matches": replayed_success
        == outcome_success,
        "evaluator": scenario.evaluator,
        "key_event_time_s": scenario.key_event_time_s,
        "task_evidence": task_evidence,
    }
    failures: list[str] = []
    failures.extend(penetration.failures)
    if not common_pass:
        failures.append("strict_common_physics_check_failed")
    if not energy_pass:
        failures.append("free_flight_energy_drift_exceeded")
    if not restitution_pass:
        failures.append("effective_restitution_exceeded")
    if not task_pass:
        failures.extend(task_failures)
    return qc, outcome, tuple(sorted(set(failures)))


def _aabb_intersects(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(
        float(left[index]) <= float(right[index + 3])
        and float(right[index]) <= float(left[index + 3])
        for index in range(3)
    )


def _aabb_has_positive_overlap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance_m: float = 1e-6,
) -> bool:
    """Return true only for volumetric overlap, not supported face contact."""

    return all(
        min(float(left[index + 3]), float(right[index + 3]))
        - max(float(left[index]), float(right[index]))
        > tolerance_m
        for index in range(3)
    )


def _world_geom_aabb(model: Any, data: Any, geom_id: int) -> dict[str, Any]:
    """Transform MuJoCo's compiled local geom AABB into exact world axes."""

    local = np.asarray(model.geom_aabb[int(geom_id)], dtype=np.float64)
    if local.shape != (6,) or not np.isfinite(local).all():
        raise RuntimeError(f"geom {geom_id} has no finite compiled AABB")
    half_size = local[3:]
    if np.any(half_size < 0.0):
        raise RuntimeError(f"geom {geom_id} has a negative compiled AABB extent")
    rotation = np.asarray(data.geom_xmat[int(geom_id)], dtype=np.float64).reshape(
        3, 3
    )
    geom_origin = np.asarray(data.geom_xpos[int(geom_id)], dtype=np.float64)
    world_center = geom_origin + rotation @ local[:3]
    world_half_size = np.abs(rotation) @ half_size
    minimum = world_center - world_half_size
    maximum = world_center + world_half_size
    if not np.isfinite((*minimum, *maximum)).all():
        raise RuntimeError(f"geom {geom_id} produced a non-finite world AABB")
    return {
        "minimum_m": [float(value) for value in minimum],
        "maximum_m": [float(value) for value in maximum],
        "method": "mujoco_compiled_local_aabb_transformed/v1",
    }


def _flatten_aabb(value: Mapping[str, Any]) -> tuple[float, ...]:
    minimum = value.get("minimum_m")
    maximum = value.get("maximum_m")
    if (
        not isinstance(minimum, Sequence)
        or isinstance(minimum, (str, bytes))
        or not isinstance(maximum, Sequence)
        or isinstance(maximum, (str, bytes))
        or len(minimum) != 3
        or len(maximum) != 3
    ):
        raise RuntimeError("runtime clearance AABB must contain XYZ bounds")
    result = tuple(float(item) for item in (*minimum, *maximum))
    if not np.isfinite(result).all() or any(
        result[index] > result[index + 3] for index in range(3)
    ):
        raise RuntimeError("runtime clearance AABB is invalid")
    return result


def _evaluate_background_clearance_rows(
    *,
    background_rows: Sequence[Mapping[str, Any]],
    fixture_rows: Sequence[Mapping[str, Any]],
    high_rate_rows: Sequence[Mapping[str, Any]],
    object_radius_m: float,
) -> dict[str, Any]:
    """Bind every background geom to the complete persisted object sweep."""

    radius = float(object_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise RuntimeError("background clearance requires a positive object radius")
    if not high_rate_rows:
        raise RuntimeError("background clearance requires high-rate rollout rows")
    timestamps: list[float] = []
    positions: list[tuple[float, float, float]] = []
    sweep_hash_rows: list[dict[str, Any]] = []
    for index, row in enumerate(high_rate_rows):
        timestamp = float(row["timestamp"])
        position_raw = row["object.position"]
        if (
            not isinstance(position_raw, Sequence)
            or isinstance(position_raw, (str, bytes))
            or len(position_raw) != 3
        ):
            raise RuntimeError("background clearance object position is malformed")
        position = tuple(float(value) for value in position_raw)
        if not math.isfinite(timestamp) or not np.isfinite(position).all():
            raise RuntimeError("background clearance sweep contains non-finite state")
        timestamps.append(timestamp)
        positions.append(position)  # type: ignore[arg-type]
        sweep_hash_rows.append(
            {
                "sample_index": index,
                "timestamp_s": timestamp,
                "object_position_m": list(position),
            }
        )
    position_array = np.asarray(positions, dtype=np.float64)
    aggregate_minimum = position_array.min(axis=0) - radius
    aggregate_maximum = position_array.max(axis=0) + radius

    normalized_fixtures: list[dict[str, Any]] = []
    for raw in fixture_rows:
        fixture_id = str(raw.get("fixture_id") or "")
        if not fixture_id:
            raise RuntimeError("runtime clearance fixture lacks a stable ID")
        aabb = raw.get("world_aabb")
        if not isinstance(aabb, Mapping):
            raise RuntimeError(f"runtime clearance fixture {fixture_id} lacks an AABB")
        _flatten_aabb(aabb)
        normalized_fixtures.append(
            {
                "fixture_id": fixture_id,
                "role": str(raw.get("role") or ""),
                "geom_id": int(raw.get("geom_id", -1)),
                "world_aabb": dict(aabb),
            }
        )
    normalized_fixtures.sort(key=lambda value: value["fixture_id"])

    evaluated_backgrounds: list[dict[str, Any]] = []
    for raw in background_rows:
        stable_id = str(raw.get("stable_id") or "")
        if not stable_id:
            raise RuntimeError("runtime clearance background lacks a stable ID")
        aabb_value = raw.get("world_aabb")
        if not isinstance(aabb_value, Mapping):
            raise RuntimeError(f"runtime clearance background {stable_id} lacks an AABB")
        aabb = _flatten_aabb(aabb_value)
        background_minimum = np.asarray(aabb[:3], dtype=np.float64)
        background_maximum = np.asarray(aabb[3:], dtype=np.float64)
        intersects_sweep = np.all(
            (position_array - radius) <= background_maximum,
            axis=1,
        ) & np.all(
            background_minimum <= (position_array + radius),
            axis=1,
        )
        intersection_indices = np.flatnonzero(intersects_sweep)
        object_overlap_by_axis = np.zeros(3, dtype=np.float64)
        maximum_object_penetration = 0.0
        for sample_index in intersection_indices:
            sphere_minimum = position_array[sample_index] - radius
            sphere_maximum = position_array[sample_index] + radius
            overlap = np.minimum(background_maximum, sphere_maximum) - np.maximum(
                background_minimum, sphere_minimum
            )
            object_overlap_by_axis = np.maximum(object_overlap_by_axis, overlap)
            maximum_object_penetration = max(
                maximum_object_penetration,
                float(np.min(overlap)),
            )
        fixture_intersection_rows = []
        for fixture in normalized_fixtures:
            fixture_aabb = _flatten_aabb(fixture["world_aabb"])
            overlap = [
                min(aabb[index + 3], fixture_aabb[index + 3])
                - max(aabb[index], fixture_aabb[index])
                for index in range(3)
            ]
            if all(value > 1e-6 for value in overlap):
                fixture_intersection_rows.append(
                    {
                        "fixture_id": fixture["fixture_id"],
                        "overlap_m_by_axis": overlap,
                        "penetration_depth_m": min(overlap),
                    }
                )
        fixture_intersections = [
            value["fixture_id"] for value in fixture_intersection_rows
        ]
        contype = int(raw.get("contype", -1))
        conaffinity = int(raw.get("conaffinity", -1))
        body_weld_id = int(raw.get("body_weld_id", -1))
        # These admission claims are derived from persisted primitive model
        # fields.  Never trust self-authored booleans that can be changed and
        # rehashed together with the rest of the clearance payload.
        collision_disabled = contype == 0 and conaffinity == 0
        anchored = body_weld_id == 0
        first_index = (
            int(intersection_indices[0]) if len(intersection_indices) else None
        )
        evaluated_backgrounds.append(
            {
                "stable_id": stable_id,
                "geom_id": int(raw.get("geom_id", -1)),
                "classification": str(raw.get("classification") or ""),
                "source_name": raw.get("source_name"),
                "catalog_slot": raw.get("catalog_slot"),
                "body_id": int(raw.get("body_id", -1)),
                "body_name": raw.get("body_name"),
                "body_weld_id": body_weld_id,
                "world_aabb": dict(aabb_value),
                "contype": contype,
                "conaffinity": conaffinity,
                "collision_disabled": collision_disabled,
                "anchored": anchored,
                "object_swept_clear": first_index is None,
                "object_intersection_sample_count": int(len(intersection_indices)),
                "maximum_object_overlap_m_by_axis": [
                    float(value) for value in object_overlap_by_axis
                ],
                "maximum_object_penetration_depth_m": (
                    maximum_object_penetration
                ),
                "first_object_intersection": (
                    None
                    if first_index is None
                    else {
                        "sample_index": first_index,
                        "timestamp_s": timestamps[first_index],
                        "object_position_m": list(positions[first_index]),
                    }
                ),
                "fixture_intersection_clear": not fixture_intersections,
                "intersecting_fixture_ids": fixture_intersections,
                "fixture_intersections": fixture_intersection_rows,
            }
        )
    evaluated_backgrounds.sort(key=lambda value: value["stable_id"])
    object_failures = [
        row["stable_id"]
        for row in evaluated_backgrounds
        if row["object_swept_clear"] is not True
    ]
    fixture_failures = [
        row["stable_id"]
        for row in evaluated_backgrounds
        if row["fixture_intersection_clear"] is not True
    ]
    collision_failures = [
        row["stable_id"]
        for row in evaluated_backgrounds
        if row["collision_disabled"] is not True
    ]
    anchoring_failures = [
        row["stable_id"]
        for row in evaluated_backgrounds
        if row["anchored"] is not True
    ]
    result = {
        "schema_version": SOURCE_MUJOCO_BACKGROUND_CLEARANCE_SCHEMA,
        "evaluated": True,
        "object_sweep": {
            "source": "every_high_rate_object_position_expanded_by_radius/v1",
            "sample_count": len(sweep_hash_rows),
            "object_radius_m": radius,
            "aggregate_world_aabb": {
                "minimum_m": [float(value) for value in aggregate_minimum],
                "maximum_m": [float(value) for value in aggregate_maximum],
            },
            "exact_rows_sha256": sha256_json(sweep_hash_rows),
        },
        "fixture_rows": normalized_fixtures,
        "fixture_rows_sha256": sha256_json(normalized_fixtures),
        "background_rows": evaluated_backgrounds,
        "background_rows_sha256": sha256_json(evaluated_backgrounds),
        "background_geom_count": len(evaluated_backgrounds),
        "all_background_collision_disabled": not collision_failures,
        "all_background_anchored": not anchoring_failures,
        "object_swept_volume_clear": not object_failures,
        "fixture_intersection_clear": not fixture_failures,
        "collision_failure_ids": collision_failures,
        "anchoring_failure_ids": anchoring_failures,
        "object_sweep_failure_ids": object_failures,
        "fixture_intersection_failure_ids": fixture_failures,
    }
    result["clearance_pass"] = bool(
        result["all_background_collision_disabled"]
        and result["all_background_anchored"]
        and result["object_swept_volume_clear"]
        and result["fixture_intersection_clear"]
    )
    return result


def _fixture_clearance_static_rows(
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
) -> list[dict[str, Any]]:
    """Return the immutable compiled fixture geometry used by replay QC."""

    rows = []
    for surface in scenario.surfaces:
        geom_id = int(compiled.ids.surface_geom_ids[surface.name])
        rows.append(
            {
                "fixture_id": surface.name,
                "role": surface.role,
                "geom_id": geom_id,
                "world_aabb": _world_geom_aabb(
                    compiled.model, compiled.data, geom_id
                ),
            }
        )
    rows.sort(key=lambda value: value["fixture_id"])
    return rows


def _background_clearance_static_rows(
    mujoco: Any,
    compiled: CompiledSourceModel,
) -> list[dict[str, Any]]:
    """Return hash-bound primitive background geometry, without QC claims."""

    model, data = compiled.model, compiled.data
    rows = []
    for descriptor in compiled.background_geom_descriptors:
        geom_id = int(descriptor["geom_id"])
        body_id = int(model.geom_bodyid[geom_id])
        rows.append(
            {
                **dict(descriptor),
                "body_id": body_id,
                "body_name": str(
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    )
                    or "world"
                ),
                "body_weld_id": int(model.body_weldid[body_id]),
                "world_aabb": _world_geom_aabb(model, data, geom_id),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
            }
        )
    rows.sort(key=lambda value: value["stable_id"])
    return rows


def _runtime_background_clearance(
    mujoco: Any,
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
    high_rate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve classified compiled geoms into hashable runtime evidence."""

    fixture_rows = _fixture_clearance_static_rows(compiled, scenario)
    background_rows = _background_clearance_static_rows(mujoco, compiled)
    result = _evaluate_background_clearance_rows(
        background_rows=background_rows,
        fixture_rows=fixture_rows,
        high_rate_rows=high_rate_rows,
        object_radius_m=scenario.object_radius_m,
    )
    result["classification"] = {
        "source": "CompiledSourceModel.background_geom_descriptors/v1",
        "descriptor_count": len(compiled.background_geom_descriptors),
        "descriptors_sha256": sha256_json(compiled.background_geom_descriptors),
        "background_static_rows_sha256": sha256_json(background_rows),
        "fixture_static_rows_sha256": sha256_json(fixture_rows),
        "exclusions": dict(compiled.background_geom_exclusions),
        "exclusions_sha256": sha256_json(compiled.background_geom_exclusions),
    }
    return result


def _robocasa_manifest(
    mujoco: Any,
    compiled: CompiledSourceModel,
    scenario: SourceMujocoCompiledScenario,
    dependency: RoboCasaDependency,
    *,
    background_clearance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not scenario.requires_real_robocasa:
        return []
    model, data = compiled.model, compiled.data
    runtime_rows: tuple[Mapping[str, Any], ...] = ()
    clearance_sha256: str | None = None
    clearance_evaluated = background_clearance is not None
    if background_clearance is not None:
        if (
            background_clearance.get("schema_version")
            != SOURCE_MUJOCO_BACKGROUND_CLEARANCE_SCHEMA
            or background_clearance.get("evaluated") is not True
        ):
            raise RuntimeError("RoboCasa runtime clearance evidence is malformed")
        raw_rows = background_clearance.get("background_rows")
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise RuntimeError("RoboCasa runtime clearance lacks background rows")
        runtime_rows = tuple(
            row for row in raw_rows if isinstance(row, Mapping)
        )
        if len(runtime_rows) != len(raw_rows):
            raise RuntimeError("RoboCasa runtime clearance contains malformed rows")
        clearance_sha256 = sha256_json(background_clearance)
    result: list[dict[str, Any]] = []
    asset_root = Path(dependency.asset_root).resolve(strict=True)
    source_root = Path(dependency.source_root).resolve(strict=True)
    for item in compiled.robocasa_assets:
        slot = str(item["slot"])
        relative = str(item["model_xml"])
        descriptor = asset_root / relative
        if not descriptor.is_file() and relative.startswith("lightwheel/"):
            descriptor = asset_root / "objects" / relative
        descriptor = descriptor.resolve(strict=True)
        descriptor_sha256 = sha256_file(descriptor)
        if descriptor_sha256 != item.get("descriptor_sha256"):
            raise RuntimeError(f"RoboCasa catalog descriptor hash changed: {slot}")
        if item.get("license_notice_sha256") != dependency.license_sha256:
            raise RuntimeError(f"RoboCasa catalog license binding changed: {slot}")
        expected_hashes = item.get("referenced_file_sha256")
        if not isinstance(expected_hashes, Mapping) or not expected_hashes:
            raise RuntimeError(f"RoboCasa catalog content manifest is absent: {slot}")
        runtime_hashes: dict[str, str] = {}
        for raw_path, expected_sha256 in expected_hashes.items():
            content_path = (source_root / str(raw_path)).resolve(strict=True)
            try:
                content_path.relative_to(source_root)
            except ValueError as error:
                raise RuntimeError(
                    f"RoboCasa catalog content escapes its dependency root: {slot}"
                ) from error
            actual_sha256 = sha256_file(content_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"RoboCasa catalog content hash changed: {slot}")
            # The source descriptor itself is compiled away by the visual-only
            # importer; every mesh and texture must remain an explicit MJCF
            # file reference in the compiled manifest.
            if (
                content_path != descriptor
                and compiled.source_asset_sha256.get(str(content_path))
                != actual_sha256
            ):
                raise RuntimeError(
                    f"compiled RoboCasa content differs from its catalog manifest: {slot}"
                )
            runtime_hashes[str(content_path)] = actual_sha256
        if combined_manifest_hash(expected_hashes) != item.get("manifest_sha256"):
            raise RuntimeError(f"RoboCasa catalog manifest hash changed: {slot}")
        body_prefix = f"rc_{slot}_"
        owned_body_ids = {
            body_id
            for body_id in range(model.nbody)
            if str(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            ).startswith(body_prefix)
        }
        # Imported RoboCasa geoms are often unnamed.  Resolve membership from
        # the namespaced body hierarchy rather than silently dropping them.
        for body_id in range(model.nbody):
            if body_id in owned_body_ids:
                continue
            parent = int(model.body_parentid[body_id])
            while parent > 0:
                if parent in owned_body_ids:
                    owned_body_ids.add(body_id)
                    break
                parent = int(model.body_parentid[parent])
        geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in owned_body_ids
        ]
        if not geom_ids:
            raise RuntimeError(f"selected RoboCasa asset has no compiled geoms: {slot}")
        geom_aabbs = [
            _flatten_aabb(_world_geom_aabb(model, data, index))
            for index in geom_ids
        ]
        minimum = np.min(
            np.asarray([value[:3] for value in geom_aabbs], dtype=np.float64),
            axis=0,
        )
        maximum = np.max(
            np.asarray([value[3:] for value in geom_aabbs], dtype=np.float64),
            axis=0,
        )
        aabb = (*tuple(float(value) for value in minimum), *tuple(float(value) for value in maximum))
        collisions_disabled = all(
            int(model.geom_contype[index]) == 0 and int(model.geom_conaffinity[index]) == 0
            for index in geom_ids
        )
        if not collisions_disabled:
            raise RuntimeError(f"RoboCasa background collision is enabled: {slot}")
        slot_runtime_rows = tuple(
            row
            for row in runtime_rows
            if row.get("classification") == "catalog"
            and str(row.get("catalog_slot") or "") == slot
        )
        if clearance_evaluated and not slot_runtime_rows:
            raise RuntimeError(
                f"RoboCasa runtime clearance omitted catalog slot: {slot}"
            )
        swept_clear = (
            all(row.get("object_swept_clear") is True for row in slot_runtime_rows)
            if clearance_evaluated
            else None
        )
        fixture_clear = (
            all(
                row.get("fixture_intersection_clear") is True
                for row in slot_runtime_rows
            )
            if clearance_evaluated
            else None
        )
        anchored = (
            all(row.get("anchored") is True for row in slot_runtime_rows)
            if clearance_evaluated
            else None
        )
        blockers = ["rendered_occlusion_review_pending"]
        if not clearance_evaluated:
            blockers.append("runtime_clearance_pending")
        else:
            if swept_clear is not True:
                blockers.append("background_intersects_task_swept_volume")
            if fixture_clear is not True:
                blockers.append("background_intersects_physical_fixture")
            if anchored is not True:
                blockers.append("background_not_anchored")
        yaw = float(item.get("yaw", 0.0))
        position = tuple(float(value) for value in item.get("position", (0, 0, 0)))
        c, s = math.cos(yaw), math.sin(yaw)
        transform = (
            c, -s, 0.0, position[0],
            s, c, 0.0, position[1],
            0.0, 0.0, 1.0, position[2],
            0.0, 0.0, 0.0, 1.0,
        )
        result.append(
            {
                "schema_version": "dynamic-robot-robocasa-admission/v1",
                "catalog_version": str(item["catalog_version"]),
                "catalog_sha256": str(item["catalog_sha256"]),
                "asset_id": str(item["catalog_asset_id"]),
                "selection_source": str(item["selection_source"]),
                "source_root": dependency.source_root,
                "descriptor_path": str(descriptor),
                "descriptor_sha256": descriptor_sha256,
                "referenced_file_sha256": dict(sorted(runtime_hashes.items())),
                "license_notice_path": dependency.license_path,
                "license_notice_sha256": dependency.license_sha256,
                "scale_to_meters": float(item["scale_to_meters"]),
                "world_aabb": {
                    "minimum_m": list(aabb[:3]),
                    "maximum_m": list(aabb[3:]),
                },
                "transform_row_major_4x4": list(transform),
                "visual_only": True,
                "collision_enabled": False,
                "runtime_clearance_evaluated": clearance_evaluated,
                "runtime_clearance_sha256": clearance_sha256,
                "catalog_geom_clearance_sha256": (
                    sha256_json(slot_runtime_rows)
                    if clearance_evaluated
                    else None
                ),
                "catalog_geom_count": len(slot_runtime_rows),
                "background_anchored": anchored,
                "swept_volume_clear": swept_clear,
                "fixture_intersection_clear": fixture_clear,
                # Occlusion is deliberately left for rendered automated/human
                # review; therefore an R1 preview is not an admitted release row.
                "occlusion_validated": False,
                "admitted": False,
                "blockers": sorted(blockers),
            }
        )
    for row in result:
        row["manifest_sha256"] = combined_manifest_hash(row["referenced_file_sha256"])
    return result


class SourceMujocoBackend:
    """Owned review backend; it never changes registry or leaf release state."""

    name = "source_mujoco"
    version = SOURCE_MUJOCO_BACKEND_VERSION

    def __init__(
        self,
        *,
        source_root: str | Path | None = None,
        robocasa_root: str | Path | None = None,
    ) -> None:
        self.source_dependency: SourceDependencyManifest = resolve_source_dependency(
            source_root
        )
        if self.source_dependency.manifest_sha256 != PINNED_SOURCE_MANIFEST_SHA256:
            raise RuntimeError("source dependency differs from the canonical ten-file pin")
        self.robocasa_dependency: RoboCasaDependency = resolve_robocasa_dependency(
            robocasa_root
        )

    def compile_case(
        self, case: Mapping[str, Any] | Any
    ) -> SourceMujocoCompiledScenario:
        return compile_review_case(case)

    def run(
        self,
        scenario: SourceMujocoCompiledScenario | Mapping[str, Any] | Any,
        *,
        render: bool = True,
    ) -> SourceMujocoRunResult:
        if not isinstance(scenario, SourceMujocoCompiledScenario):
            scenario = self.compile_case(scenario)
        scenario.validate()
        mujoco, least_squares = _require_runtime_dependencies()
        compiled = compile_source_model(
            scenario,
            source_dependency=self.source_dependency,
            robocasa_dependency=(
                self.robocasa_dependency if scenario.requires_real_robocasa else None
            ),
        )
        controller, initial_robot_q, ik_diagnostics = _controller_for_scenario(
            mujoco, least_squares, compiled, scenario
        )
        _initialize_state(mujoco, compiled, scenario, initial_robot_q)
        initialized_robot_joint_qpos: tuple[float, ...] | None = None
        if scenario.embodiment != "no_robot":
            initialized_robot_joint_qpos = tuple(
                float(compiled.data.qpos[index])
                for index in compiled.ids.robot_qpos_adrs
            )
            if not np.allclose(
                initialized_robot_joint_qpos,
                initial_robot_q,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "initialized robot qpos differs from the owned controller plan"
                )
        initialized_robot_joint_qpos_sha256 = (
            None
            if initialized_robot_joint_qpos is None
            else sha256_json(initialized_robot_joint_qpos)
        )
        compiled_robot_base_pose = (
            None
            if compiled.robot_base_position_m is None
            else {
                "position_m": compiled.robot_base_position_m,
                "euler_rad": scenario.robot_base_euler_rad,
                "quaternion_wxyz": compiled.robot_base_quaternion_wxyz,
            }
        )
        compiled_robot_base_pose_sha256 = (
            None
            if compiled_robot_base_pose is None
            else sha256_json(compiled_robot_base_pose)
        )

        renderers: dict[str, Any] = {}
        frames: dict[str, list[np.ndarray]] = {"main": [], "secondary": []}
        segmentation_observations: dict[str, list[dict[str, Any]]] = {
            "main": [],
            "secondary": [],
        }
        if render:
            renderers = {
                name: mujoco.Renderer(
                    compiled.model,
                    height=RIGID_REVIEW_PROFILE.height,
                    width=RIGID_REVIEW_PROFILE.width,
                )
                for name in ("main", "secondary")
            }
        camera_ids = {
            name: int(
                mujoco.mj_name2id(
                    compiled.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )
            )
            for name, camera_name in compiled.camera_names.items()
        }
        if any(index < 0 for index in camera_ids.values()):
            raise RuntimeError("compiled source model lacks a canonical camera")
        camera_calibrations = {
            name: _camera_calibration(
                compiled.model, compiled.data, camera_id, name
            )
            for name, camera_id in camera_ids.items()
        }

        target_frame_times = fixed_duration_frame_timestamps(
            scenario.duration_s, scenario.video_hz
        )
        frame_steps = {
            int(round(timestamp * scenario.simulation_hz)): (index, timestamp)
            for index, timestamp in enumerate(target_frame_times)
        }
        total_steps = int(round(scenario.duration_s * scenario.simulation_hz))
        control_stride = scenario.simulation_hz // scenario.control_hz
        current_action = np.empty(0, dtype=np.float64)
        high_rate_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        contact_rows: list[dict[str, Any]] = []
        control_updates = 0
        mutation_violations = 0
        tunneling = 0
        discontinuities = 0
        previous_position: np.ndarray | None = None
        previous_velocity: np.ndarray | None = None
        previous_had_contact = False
        maximum_penetration = 0.0

        try:
            for step in range(total_steps + 1):
                if controller is not None and step % control_stride == 0:
                    try:
                        current_action = controller.apply(
                            _control_observation(compiled),
                            model=compiled.model,
                            data=compiled.data,
                            actuator_ids=compiled.ids.actuator_ids,
                            forbidden_equality_ids=(
                                compiled.ids.object_linked_equality_ids
                            ),
                        )
                    except Exception:
                        mutation_violations += 1
                        raise
                    control_updates += 1
                contacts, left, right = _contacts_at_state(mujoco, compiled)
                for row in contacts:
                    maximum_penetration = max(
                        maximum_penetration, float(row["penetration_depth_m"])
                    )
                contact_rows.extend(contacts)
                state = _state_row(
                    compiled,
                    scenario,
                    current_action,
                    contacts,
                    left,
                    right,
                )
                high_rate_rows.append(state)
                position = np.asarray(state["object.position"], dtype=np.float64)
                velocity = np.asarray(state["object.linear_velocity"], dtype=np.float64)
                if previous_position is not None:
                    if float(np.linalg.norm(position - previous_position)) > max(
                        0.08, 2.0 * scenario.object_radius_m
                    ):
                        tunneling += 1
                    expected_delta = np.asarray(scenario.gravity_m_s2) / scenario.simulation_hz
                    residual = float(
                        np.linalg.norm(velocity - previous_velocity - expected_delta)
                    )
                    if residual > 0.35 and not contacts and not previous_had_contact:
                        discontinuities += 1
                previous_position = position
                previous_velocity = velocity
                previous_had_contact = bool(contacts)

                if step in frame_steps:
                    _, target_timestamp = frame_steps[step]
                    simulation_timestamp = float(compiled.data.time)
                    frame_rows.append(
                        {
                            **state,
                            "timestamp": float(target_timestamp),
                            "simulation_timestamp": simulation_timestamp,
                            "synchronization_error_s": abs(
                                simulation_timestamp - float(target_timestamp)
                            ),
                        }
                    )
                    for view, renderer in renderers.items():
                        renderer.update_scene(
                            compiled.data, camera=compiled.camera_names[view]
                        )
                        rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
                        frames[view].append(rgb)
                        segmentation_observations[view].append(
                            _render_segmentation_observation(
                                mujoco,
                                renderer,
                                rgb,
                                object_geom_id=compiled.ids.object_geom,
                                left_tool_geom_ids=compiled.ids.left_gripper_geom_ids,
                                right_tool_geom_ids=compiled.ids.right_gripper_geom_ids,
                                fixture_geom_ids=tuple(
                                    compiled.ids.surface_geom_ids.values()
                                ),
                            )
                        )
                if step < total_steps:
                    mujoco.mj_step(compiled.model, compiled.data)
        finally:
            for renderer in renderers.values():
                renderer.close()

        validate_persisted_render_schedule(
            frame_rows,
            duration_s=scenario.duration_s,
            fps_num=scenario.video_hz,
            maximum_sample_error_s=1.0 / scenario.simulation_hz,
        )
        if render and any(len(value) != len(target_frame_times) for value in frames.values()):
            raise RuntimeError("rendered views differ from the persisted frame schedule")

        runtime_audit = {
            "initial_object_state_writes": 1,
            "initial_robot_state_writes": int(scenario.embodiment != "no_robot"),
            "initialized_robot_joint_qpos": initialized_robot_joint_qpos,
            "initialized_robot_joint_qpos_sha256": (
                initialized_robot_joint_qpos_sha256
            ),
            "compiled_robot_base_position_m": compiled.robot_base_position_m,
            "compiled_robot_base_quaternion_wxyz": (
                compiled.robot_base_quaternion_wxyz
            ),
            "compiled_robot_base_pose_sha256": compiled_robot_base_pose_sha256,
            "object_state_writes_after_initialization": 0,
            "direct_robot_state_writes_after_initialization": 0,
            "mocap_writes_after_initialization": 0,
            "applied_force_writes_after_initialization": 0,
            "object_linked_equality_changes_after_initialization": 0,
            "model_physics_mutations_after_initialization": 0,
            "mutation_boundary_violations": mutation_violations,
            "solver_warning_count": _solver_warning_count(compiled.data),
            "non_finite_state_count": int(
                any(
                    not np.isfinite(
                        np.asarray(
                            [
                                *row["object.position"],
                                *row["object.linear_velocity"],
                                *row["robot.joint_position"],
                                *row[ACTION_FIELD],
                            ],
                            dtype=np.float64,
                        )
                    ).all()
                    for row in high_rate_rows
                )
            ),
            "tunneling_event_count": tunneling,
            "unexplained_velocity_discontinuity_count": discontinuities,
            "control_updates": control_updates,
            "actuator_control_applicable": scenario.embodiment != "no_robot",
            "simulation_steps": total_steps,
            "rendered_frame_count": len(frame_rows),
            "maximum_penetration_m": maximum_penetration,
            "action_ctrl_echo_exact": all(
                row[ACTION_FIELD] == row["simulator.applied_actuator_ctrl"]
                for row in high_rate_rows
            ),
            "object_linked_equality_ids": list(
                compiled.ids.object_linked_equality_ids
            ),
        }
        physics_qc, outcome, qc_flags = _physics_qc(
            compiled,
            scenario,
            high_rate_rows,
            contact_rows,
            runtime_audit,
        )
        tool_visibility_topology = _compiled_tool_visibility_topology(compiled)
        measured_key_event = _measured_visibility_key_event(
            scenario,
            high_rate_rows,
            contact_rows,
            left_tool_geom_ids=tool_visibility_topology["left_tool_geom_ids"],
            right_tool_geom_ids=tool_visibility_topology["right_tool_geom_ids"],
            tool_geom_body_ids=tool_visibility_topology["tool_geom_body_ids"],
            left_tool_body_id=tool_visibility_topology["left_tool_body_id"],
            right_tool_body_id=tool_visibility_topology["right_tool_body_id"],
            tool_visibility_topology_sha256=sha256_json(
                tool_visibility_topology
            ),
        )
        outcome = {
            **outcome,
            "planned_key_event_time_s": measured_key_event[
                "planned_key_event_time_s"
            ],
            "key_event_time_s": measured_key_event["actual_key_event_time_s"],
            "key_event_name": measured_key_event["actual_key_event_name"],
            "key_event_source": measured_key_event["actual_key_event_source"],
        }
        visibility_qc = _source_visibility_qc(
            scenario,
            frame_rows,
            camera_calibrations,
            frames,
            segmentation_observations,
            measured_key_event,
        )
        background_clearance = _runtime_background_clearance(
            mujoco,
            compiled,
            scenario,
            high_rate_rows,
        )
        background_clearance_sha256 = sha256_json(background_clearance)
        runtime_audit = {
            **runtime_audit,
            "background_clearance_schema": (
                SOURCE_MUJOCO_BACKGROUND_CLEARANCE_SCHEMA
            ),
            "background_clearance_sha256": background_clearance_sha256,
            "background_clearance_pass": background_clearance[
                "clearance_pass"
            ],
        }
        robocasa_manifest = _robocasa_manifest(
            mujoco,
            compiled,
            scenario,
            self.robocasa_dependency,
            background_clearance=background_clearance,
        )
        source_hashes = {
            "external_dependency_manifest": self.source_dependency.manifest_sha256,
            "scene_builder_py": PINNED_SOURCE_FILES[
                "scripts_mujoco/scene_builder.py"
            ],
            "variants_py": PINNED_SOURCE_FILES["scripts_mujoco/variants.py"],
            "robocasa_importer_py": PINNED_SOURCE_FILES[
                "scripts_mujoco/robocasa_assets.py"
            ],
            "panda_xml": PINNED_SOURCE_FILES[
                "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
            ],
            "panda_nohand_xml": PINNED_SOURCE_FILES[
                "third_party/mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
            ],
            "robotiq_2f85_xml": PINNED_SOURCE_FILES[
                "third_party/mujoco_menagerie/robotiq_2f85/2f85.xml"
            ],
            "robocasa_catalog": sha256_file(
                Path(__file__).resolve().parents[4]
                / "configs/assets/robocasa_catalog_v1.yaml"
            ),
            "compiled_scene_xml": compiled.xml_sha256,
            "compiled_asset_manifest": combined_manifest_hash(
                compiled.source_asset_sha256
            ),
            "robocasa_license": self.robocasa_dependency.license_sha256,
        }
        quality_flags = list(qc_flags)
        if not outcome["intended_outcome_match"]:
            quality_flags.append("intended_outcome_mismatch_preserved")
        if scenario.requires_real_robocasa:
            quality_flags.append("rendered_robocasa_occlusion_review_pending")
        if background_clearance.get("clearance_pass") is not True:
            quality_flags.append("background_runtime_clearance_failed")
        if render:
            if visibility_qc.get("evaluated") is not True:
                quality_flags.append("rendered_visibility_qc_not_evaluated")
            if visibility_qc.get("key_event_visible_in_any_view") is not True:
                quality_flags.append("key_event_visibility_failed")
            if visibility_qc.get("apex_visible_in_any_view") is not True:
                quality_flags.append("apex_visibility_failed")
            if visibility_qc.get("final_state_visible_in_any_view") is not True:
                quality_flags.append("final_state_visibility_failed")
            if float(
                visibility_qc.get("target_visible_frame_fraction", 0.0)
            ) < float(
                SOURCE_MUJOCO_VISUAL_THRESHOLDS[
                    "minimum_target_visible_frame_fraction"
                ]
            ):
                quality_flags.append("whole_episode_target_visibility_failed")
            if visibility_qc.get("contact_occluded_both_views") is True:
                quality_flags.append("critical_counterpart_visibility_failed")
        else:
            quality_flags.append("rendered_visibility_qc_not_evaluated")
        backend_provenance = {
            "backend": self.name,
            "backend_version": self.version,
            "release_state": "blocked",
            "production_eligible": False,
            "review_only": True,
            "scenario_hash": scenario.scenario_sha256,
            "source_dependency": self.source_dependency.to_dict(),
            "robocasa_dependency": self.robocasa_dependency.to_dict(),
            "source_hashes": source_hashes,
            "compiled_model_sha256": compiled.xml_sha256,
            "compiled_asset_sha256": dict(compiled.source_asset_sha256),
            "model_calibration": RIGID_REVIEW_PROFILE.to_dict(),
            "robot_base_pose": {
                "position_m": compiled.robot_base_position_m,
                "quaternion_wxyz": compiled.robot_base_quaternion_wxyz,
                "pose_sha256": compiled_robot_base_pose_sha256,
                "source": "owned_compiled_scenario/v2",
            },
            "removed_visual_work_surface_names": list(
                compiled.removed_visual_work_surface_names
            ),
            "removed_task_volume_background_names": list(
                compiled.removed_task_volume_background_names
            ),
            "removed_fixture_intersection_background_names": list(
                compiled.removed_fixture_intersection_background_names
            ),
            "relocated_visual_backgrounds": [
                dict(value) for value in compiled.relocated_visual_backgrounds
            ],
            "controller_profile_id": RIGID_REVIEW_PROFILE.profile_id,
            "controller_profile_version": RIGID_REVIEW_PROFILE.profile_id,
            "controller_boundary": "apply_actuator_only_callback",
            "action_mode": (
                "no_actuators/v1"
                if scenario.embodiment == "no_robot"
                else ACTION_SEMANTICS
            ),
            "actual_action_width": (0 if scenario.embodiment == "no_robot" else 8),
            "runtime_audit": runtime_audit,
            "ik_diagnostics": [value.to_dict() for value in ik_diagnostics],
            "external_scene_controller_imported": False,
            "external_source_usage": "read_only_scene_and_assets_only",
            "stripped_unlicensed_robotwin_asset_count": compiled.stripped_robotwin_asset_count,
            "renderer": "mujoco.Renderer",
            "frame_schedule": {
                "clock": "half_open_k_over_30",
                "frame_count": len(frame_rows),
                "maximum_synchronization_error_s": max(
                    (float(row["synchronization_error_s"]) for row in frame_rows),
                    default=0.0,
                ),
            },
            "planned_key_event_time_s": scenario.key_event_time_s,
            "key_event_time_s": measured_key_event["actual_key_event_time_s"],
            "key_event_name": measured_key_event["actual_key_event_name"],
            "key_event_source": measured_key_event["actual_key_event_source"],
            "visibility_qc_schema": SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA,
            "visibility_qc_sha256": sha256_json(visibility_qc),
            "background_clearance_schema": (
                SOURCE_MUJOCO_BACKGROUND_CLEARANCE_SCHEMA
            ),
            "background_clearance_sha256": background_clearance_sha256,
            "background_clearance": background_clearance,
        }
        return SourceMujocoRunResult(
            scenario=scenario,
            frames_by_camera=frames,
            camera_calibrations=camera_calibrations,
            frame_rows=frame_rows,
            high_rate_rows=high_rate_rows,
            contact_rows=contact_rows,
            outcome=outcome,
            physics_qc=physics_qc,
            visibility_qc=visibility_qc,
            background_clearance=background_clearance,
            runtime_audit=runtime_audit,
            backend_provenance=backend_provenance,
            source_hashes=source_hashes,
            robocasa_asset_manifest=robocasa_manifest,
            quality_flags=tuple(sorted(set(quality_flags))),
        )


__all__ = [
    "IKDiagnostics",
    "SourceMujocoBackend",
    "SourceMujocoRunResult",
]
