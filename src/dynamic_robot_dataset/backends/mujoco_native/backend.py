"""Retired custom-attachment MuJoCo simulation regression implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..base import (
    BackendRunResult,
    IntendedBranch,
    NativeBackendError,
    NativeEpisodePlan,
    NativeFamily,
    RigidScenario,
    RigidShape,
    ScenarioSpec,
    SimulationBackend,
    ToolKind,
)
from ...common.cameras import CameraCalibration
from ...common.hashing import sha256_file
from ...common.visual_qc import (
    NATIVE_VISUAL_QC_SCHEMA,
    NATIVE_VISUAL_THRESHOLDS,
)
from ...families.base import (
    EpisodePlan,
    OutcomeResult,
    SimulationResult,
    assistance_record,
    default_physics_qc,
    stable_hash,
)
from .model import (
    OBJECT_BODY,
    OBJECT_GEOM,
    OBJECT_JOINT,
    TOOL_BODY,
    TOOL_SITE,
    CompiledModelDescription,
    build_model_description,
    camera_axes,
    resolve_franka_asset,
)
from .planning import compile_native_plan


@dataclass
class _Sample:
    timestamp: float
    object_position: np.ndarray
    object_quaternion: np.ndarray
    object_linear_velocity: np.ndarray
    object_angular_velocity: np.ndarray
    tool_position: np.ndarray
    tool_quaternion: np.ndarray
    tool_rotation: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_command: np.ndarray
    tool_target_position: np.ndarray
    tool_target_rpy: np.ndarray
    phase: str
    motion_mode: str
    active_surface: str
    contact_role: str
    controller_enabled: bool


_RESTITUTION_MEASUREMENT_SOURCE = (
    "native_sim_samples:last_separated_incoming/first_separated_outgoing:"
    "static_fixture_contact_normal"
)


def _verify_content_bound_artifact(
    path_value: Any, expected_sha256: Any
) -> tuple[bool, str | None, str | None]:
    """Resolve and hash an admission artifact instead of trusting a flag."""

    expected = str(expected_sha256 or "")
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        return False, None, None
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        return False, None, None
    try:
        path = Path(path_value).resolve(strict=True)
    except (FileNotFoundError, OSError):
        return False, str(path_value), None
    if not path.is_file():
        return False, str(path), None
    actual = sha256_file(path)
    return actual == expected, str(path), actual


def _measure_static_fixture_restitution(
    sim_samples: Sequence[_Sample],
    events: Sequence[Mapping[str, Any]],
    *,
    minimum_incoming_speed_m_s: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Measure rebound only from separated native-simulation samples.

    A contact-solver step is not a valid post-impact sample: its velocity can
    still change over subsequent constraint iterations/steps.  For each
    static-fixture contact this routine instead selects the last fully
    separated sample before contact and the first fully separated sample
    after contact.  Both velocities are projected onto the persisted
    fixture-to-object normal.

    Contacts without a separated incoming/outgoing pair are deliberately not
    measurements.  In particular, a restitution sweep therefore fails closed
    when the object never separates after impact.
    """

    diagnostics = {
        "eligible_fixture_contact_count": 0,
        "missing_incoming_sample_count": 0,
        "missing_outgoing_sample_count": 0,
        "invalid_normal_count": 0,
        "invalid_direction_count": 0,
    }
    measurements: list[dict[str, Any]] = []
    eligible_events = [
        event
        for event in events
        if bool(event.get("expected_fixture_contact", False))
        and event.get("object_b") != "native_tool"
    ]
    eligible_events.sort(key=lambda event: float(event["timestamp"]))

    for event_index, event in enumerate(eligible_events):
        diagnostics["eligible_fixture_contact_count"] += 1
        contact_time = float(event["timestamp"])
        previous_contact_time = (
            float(eligible_events[event_index - 1]["timestamp"])
            if event_index > 0
            else -math.inf
        )
        next_contact_time = (
            float(eligible_events[event_index + 1]["timestamp"])
            if event_index + 1 < len(eligible_events)
            else math.inf
        )
        normal = np.asarray(event.get("normal_world", ()), dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            diagnostics["invalid_normal_count"] += 1
            if isinstance(event, dict):
                event["restitution_measurement_status"] = "invalid_contact_normal"
                event["restitution_measurement_source"] = _RESTITUTION_MEASUREMENT_SOURCE
            continue
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            diagnostics["invalid_normal_count"] += 1
            if isinstance(event, dict):
                event["restitution_measurement_status"] = "invalid_contact_normal"
                event["restitution_measurement_source"] = _RESTITUTION_MEASUREMENT_SOURCE
            continue
        normal /= norm

        incoming_candidates = [
            sample
            for sample in sim_samples
            if previous_contact_time < sample.timestamp < contact_time
            and sample.active_surface == "none"
            and sample.contact_role == "none"
        ]
        if not incoming_candidates:
            diagnostics["missing_incoming_sample_count"] += 1
            if isinstance(event, dict):
                event["restitution_measurement_status"] = "no_separated_incoming_sample"
                event["restitution_measurement_source"] = _RESTITUTION_MEASUREMENT_SOURCE
            continue
        incoming = incoming_candidates[-1]
        incoming_normal_velocity = float(
            np.dot(incoming.object_linear_velocity, normal)
        )
        if incoming_normal_velocity >= -minimum_incoming_speed_m_s:
            diagnostics["invalid_direction_count"] += 1
            if isinstance(event, dict):
                event["restitution_measurement_status"] = "incoming_sample_not_approaching"
                event["restitution_measurement_source"] = _RESTITUTION_MEASUREMENT_SOURCE
                event["restitution_incoming_sample_time_s"] = incoming.timestamp
                event["restitution_incoming_normal_velocity_m_s"] = incoming_normal_velocity
            continue

        outgoing = next(
            (
                sample
                for sample in sim_samples
                if contact_time < sample.timestamp < next_contact_time
                and sample.active_surface == "none"
                and sample.contact_role == "none"
            ),
            None,
        )
        if outgoing is None:
            diagnostics["missing_outgoing_sample_count"] += 1
            if isinstance(event, dict):
                event["restitution_measurement_status"] = "no_separated_outgoing_sample"
                event["restitution_measurement_source"] = _RESTITUTION_MEASUREMENT_SOURCE
                event["restitution_incoming_sample_time_s"] = incoming.timestamp
                event["restitution_incoming_normal_velocity_m_s"] = incoming_normal_velocity
            continue
        outgoing_normal_velocity = float(
            np.dot(outgoing.object_linear_velocity, normal)
        )
        if outgoing_normal_velocity <= 0.0:
            diagnostics["invalid_direction_count"] += 1
            if isinstance(event, dict):
                event["restitution_measurement_status"] = "outgoing_sample_not_separating"
                event["restitution_measurement_source"] = _RESTITUTION_MEASUREMENT_SOURCE
                event["restitution_incoming_sample_time_s"] = incoming.timestamp
                event["restitution_outgoing_sample_time_s"] = outgoing.timestamp
                event["restitution_incoming_normal_velocity_m_s"] = incoming_normal_velocity
                event["restitution_outgoing_normal_velocity_m_s"] = outgoing_normal_velocity
            continue

        measured = outgoing_normal_velocity / -incoming_normal_velocity
        evidence = {
            "fixture": str(event.get("object_b") or "unknown_fixture"),
            "contact_event_time_s": contact_time,
            "contact_normal_world": [float(value) for value in normal],
            "incoming_sample_time_s": incoming.timestamp,
            "outgoing_sample_time_s": outgoing.timestamp,
            "incoming_normal_velocity_m_s": incoming_normal_velocity,
            "outgoing_normal_velocity_m_s": outgoing_normal_velocity,
            "measured_effective_restitution": measured,
            "sample_count": 2,
            "source": _RESTITUTION_MEASUREMENT_SOURCE,
        }
        measurements.append(evidence)
        if isinstance(event, dict):
            event.update(
                {
                    "restitution_measurement_status": "measured",
                    "restitution_measurement_source": _RESTITUTION_MEASUREMENT_SOURCE,
                    "restitution_measurement_sample_count": 2,
                    "restitution_incoming_sample_time_s": incoming.timestamp,
                    "restitution_outgoing_sample_time_s": outgoing.timestamp,
                    "restitution_incoming_normal_velocity_m_s": incoming_normal_velocity,
                    "restitution_outgoing_normal_velocity_m_s": outgoing_normal_velocity,
                    "measured_effective_restitution": measured,
                }
            )
    return measurements, diagnostics


@dataclass
class _RuntimeAudit:
    initial_object_state_writes: int = 0
    initial_robot_state_writes: int = 0
    object_state_writes_after_initialization: int = 0
    direct_robot_state_writes_after_initialization: int = 0
    control_updates: int = 0
    simulation_steps: int = 0
    rendered_frame_count: int = 0
    equality_constraint_count: int = 0
    terminal_reason: str = "maximum_duration"
    maximum_penetration_m: float = 0.0
    maximum_ctrl_range_violation: float = 0.0
    maximum_actuator_force_range_violation: float = 0.0
    maximum_actuator_force_abs: float = 0.0
    integrated_contact_impulse_n_s: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    contact_impulse_sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_object_state_writes": self.initial_object_state_writes,
            "initial_robot_state_writes": self.initial_robot_state_writes,
            "object_state_writes_after_initialization": self.object_state_writes_after_initialization,
            "direct_robot_state_writes_after_initialization": self.direct_robot_state_writes_after_initialization,
            "control_updates": self.control_updates,
            "simulation_steps": self.simulation_steps,
            "rendered_frame_count": self.rendered_frame_count,
            "equality_constraint_count": self.equality_constraint_count,
            "terminal_reason": self.terminal_reason,
            "maximum_penetration_m": self.maximum_penetration_m,
            "maximum_ctrl_range_violation": self.maximum_ctrl_range_violation,
            "maximum_actuator_force_range_violation": self.maximum_actuator_force_range_violation,
            "maximum_actuator_force_abs": self.maximum_actuator_force_abs,
            "integrated_contact_impulse_n_s": list(self.integrated_contact_impulse_n_s),
            "contact_impulse_sample_count": self.contact_impulse_sample_count,
        }


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as error:
        raise NativeBackendError(
            "native_mujoco requires the project mujoco extra (mujoco==3.10.0)"
        ) from error
    return mujoco


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    return 0.5 * sum(
        (np.cross(current[:, axis], desired[:, axis]) for axis in range(3)),
        start=np.zeros(3, dtype=np.float64),
    )


def _ordered_subsequence(expected: Sequence[str], observed: Sequence[str]) -> bool:
    iterator = iter(observed)
    return all(any(item == target for item in iterator) for target in expected)


def _phase_target(
    spec: ScenarioSpec,
    timestamp: float,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    delayed = max(0.0, timestamp - spec.controller_latency_s)
    if not spec.phases:
        return (
            np.asarray(spec.tool.start_position_m, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            "uncommanded",
            False,
        )
    previous_position = np.asarray(spec.tool.start_position_m, dtype=np.float64)
    previous_rpy = np.zeros(3, dtype=np.float64)
    for phase in spec.phases:
        target_position = np.asarray(phase.target_position_m, dtype=np.float64)
        target_rpy = np.asarray(phase.target_rpy_rad, dtype=np.float64)
        if delayed < phase.start_s:
            return previous_position, previous_rpy, "controller_delay", spec.branch != IntendedBranch.NO_OP
        if delayed <= phase.end_s:
            fraction = (delayed - phase.start_s) / (phase.end_s - phase.start_s)
            if phase.interpolation == "hold":
                fraction = 1.0
            elif phase.interpolation == "smoothstep":
                fraction = _smoothstep(fraction)
            return (
                previous_position + fraction * (target_position - previous_position),
                previous_rpy + fraction * (target_rpy - previous_rpy),
                phase.name,
                spec.branch != IntendedBranch.NO_OP,
            )
        previous_position = target_position
        previous_rpy = target_rpy
    return previous_position, previous_rpy, spec.phases[-1].name, spec.branch != IntendedBranch.NO_OP


def _camera_calibration(spec: Any, width: int, height: int) -> CameraCalibration:
    right, up, forward = camera_axes(spec)
    down = tuple(-value for value in up)
    position = spec.position_m

    def dot(left: Sequence[float], right_values: Sequence[float]) -> float:
        return sum(left[index] * right_values[index] for index in range(3))

    world_to_camera = (
        *right,
        -dot(right, position),
        *down,
        -dot(down, position),
        *forward,
        -dot(forward, position),
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
    focal = 0.5 * height / math.tan(math.radians(spec.fovy_deg) / 2.0)
    return CameraCalibration(
        camera_name=spec.name,
        intrinsic_matrix=(focal, 0.0, width / 2.0, 0.0, focal, height / 2.0, 0.0, 0.0, 1.0),
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        width=width,
        height=height,
        fps=30.0,
        near_m=0.02,
        far_m=12.0,
        renderer="mujoco.Renderer",
    )


class NativeMuJoCoBackend(SimulationBackend):
    """Native rigid contact using an actuator-controlled Menagerie Panda."""

    name = "native_mujoco"
    version = "1.0.0"

    def __init__(
        self,
        *,
        franka_mjcf_path: str | Path | None = None,
        width: int = 832,
        height: int = 480,
        render_device_id: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("render dimensions must be positive")
        self._mujoco = _require_mujoco()
        self.asset = resolve_franka_asset(franka_mjcf_path)
        self.width = int(width)
        self.height = int(height)
        self.render_device_id = render_device_id

    def compile(self, plan: EpisodePlan | NativeEpisodePlan) -> NativeEpisodePlan:
        return plan if isinstance(plan, NativeEpisodePlan) else compile_native_plan(plan)

    def _compile_model(
        self, native_plan: NativeEpisodePlan
    ) -> tuple[Any, Any, CompiledModelDescription]:
        description = build_model_description(native_plan.scenario, self.asset)
        try:
            model = self._mujoco.MjModel.from_xml_string(description.xml)
        except Exception as error:
            raise NativeBackendError(
                f"MuJoCo could not compile scenario {native_plan.scenario.scenario_id}: {error}"
            ) from error
        data = self._mujoco.MjData(model)
        return model, data, description

    def _initialize_state(
        self,
        model: Any,
        data: Any,
        spec: ScenarioSpec,
        audit: _RuntimeAudit,
    ) -> None:
        """The only routine allowed to write qpos/qvel directly.

        All later robot motion is produced by actuator controls and all later
        object motion is produced by MuJoCo dynamics/contact.
        """

        if model.nkey:
            self._mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            self._mujoco.mj_resetData(model, data)
        for index, offset in enumerate(spec.robot_start_joint_offsets_rad, start=1):
            joint_id = self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}"
            )
            actuator_id = self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{index}"
            )
            if joint_id < 0 or actuator_id < 0:
                raise NativeBackendError(
                    f"compiled model is missing Franka joint/actuator {index}"
                )
            address = int(model.jnt_qposadr[joint_id])
            position = float(data.qpos[address] + offset)
            if bool(model.jnt_limited[joint_id]):
                position = float(np.clip(position, *model.jnt_range[joint_id]))
            data.qpos[address] = position
            data.ctrl[actuator_id] = position
        audit.initial_robot_state_writes = 1
        joint_id = self._mujoco.mj_name2id(
            model, self._mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT
        )
        if joint_id < 0:
            raise NativeBackendError("compiled model is missing the task object free joint")
        qpos_address = int(model.jnt_qposadr[joint_id])
        qvel_address = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_address : qpos_address + 3] = spec.initial_state.position_m
        data.qpos[qpos_address + 3 : qpos_address + 7] = spec.initial_state.quaternion_wxyz
        data.qvel[qvel_address : qvel_address + 3] = spec.initial_state.linear_velocity_m_s
        data.qvel[qvel_address + 3 : qvel_address + 6] = spec.initial_state.angular_velocity_rad_s
        audit.initial_object_state_writes += 1
        self._mujoco.mj_forward(model, data)

    def _ids(self, model: Any) -> dict[str, Any]:
        object_joint = self._mujoco.mj_name2id(
            model, self._mujoco.mjtObj.mjOBJ_JOINT, OBJECT_JOINT
        )
        ids = {
            "object_joint": object_joint,
            "object_qpos": int(model.jnt_qposadr[object_joint]),
            "object_dof": int(model.jnt_dofadr[object_joint]),
            "object_body": self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY
            ),
            "object_geom": self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_GEOM, OBJECT_GEOM
            ),
            "tool_body": self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_BODY, TOOL_BODY
            ),
            "tool_site": self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE
            ),
            "arm_joints": [
                self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
                for index in range(1, 8)
            ],
            "arm_actuators": [
                self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{index}")
                for index in range(1, 8)
            ],
            "finger_actuator": self._mujoco.mj_name2id(
                model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
            ),
        }
        if any(value < 0 for key, value in ids.items() if isinstance(value, int)):
            raise NativeBackendError(f"compiled model has missing named elements: {ids}")
        if any(value < 0 for value in (*ids["arm_joints"], *ids["arm_actuators"])):
            raise NativeBackendError("compiled model does not expose all seven Franka joints/actuators")
        return ids

    def _update_controls(
        self,
        model: Any,
        data: Any,
        ids: Mapping[str, Any],
        command: np.ndarray,
        audit: _RuntimeAudit,
    ) -> np.ndarray:
        for index, (joint, actuator) in enumerate(zip(ids["arm_joints"], ids["arm_actuators"])):
            if bool(model.jnt_limited[joint]):
                low, high = model.jnt_range[joint]
                safety_margin = min(0.025, 0.1 * float(high - low))
                command[index] = np.clip(
                    command[index], low + safety_margin, high - safety_margin
                )
            if bool(model.actuator_ctrllimited[actuator]):
                low, high = model.actuator_ctrlrange[actuator]
                clipped = float(np.clip(command[index], low, high))
                audit.maximum_ctrl_range_violation = max(
                    audit.maximum_ctrl_range_violation, abs(float(command[index]) - clipped)
                )
                command[index] = clipped
            data.ctrl[actuator] = command[index]
        if ids["finger_actuator"] >= 0:
            data.ctrl[int(ids["finger_actuator"])] = 255.0
        audit.control_updates += 1
        return command

    @staticmethod
    def _rate_limit_joint_command(
        requested: np.ndarray,
        previous_command: np.ndarray,
        previous_velocity: np.ndarray,
        *,
        dt: float,
        maximum_velocity_rad_s: float = 1.5,
        maximum_acceleration_rad_s2: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply hardware-plausible velocity and acceleration limits."""

        desired_velocity = (requested - previous_command) / dt
        desired_velocity = np.clip(
            desired_velocity, -maximum_velocity_rad_s, maximum_velocity_rad_s
        )
        velocity_delta = np.clip(
            desired_velocity - previous_velocity,
            -maximum_acceleration_rad_s2 * dt,
            maximum_acceleration_rad_s2 * dt,
        )
        velocity = previous_velocity + velocity_delta
        command = previous_command + velocity * dt
        return command, velocity

    def _solve_ik_waypoints(
        self,
        model: Any,
        ids: Mapping[str, Any],
        spec: ScenarioSpec,
        home_command: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Solve Cartesian phase endpoints on scratch data before rollout.

        Scratch qpos updates are planning computations, never rollout state
        mutations.  During the actual episode only the resulting actuator
        setpoints are applied through ``data.ctrl``.
        """

        scratch = self._mujoco.MjData(model)
        if model.nkey:
            self._mujoco.mj_resetDataKeyframe(model, scratch, 0)
        else:
            self._mujoco.mj_resetData(model, scratch)
        qpos_addresses = [int(model.jnt_qposadr[joint]) for joint in ids["arm_joints"]]
        dofs = [int(model.jnt_dofadr[joint]) for joint in ids["arm_joints"]]
        for address, value in zip(qpos_addresses, home_command):
            scratch.qpos[address] = value
        self._mujoco.mj_forward(model, scratch)
        reference_rotation = np.asarray(
            scratch.site_xmat[int(ids["tool_site"])], dtype=np.float64
        ).reshape(3, 3).copy()
        results: list[np.ndarray] = []
        for phase in spec.phases:
            desired_position = np.asarray(phase.target_position_m, dtype=np.float64)
            desired_rotation = reference_rotation @ _rpy_matrix(phase.target_rpy_rad)
            for _iteration in range(240):
                self._mujoco.mj_forward(model, scratch)
                current_position = np.asarray(
                    scratch.site_xpos[int(ids["tool_site"])], dtype=np.float64
                )
                current_rotation = np.asarray(
                    scratch.site_xmat[int(ids["tool_site"])], dtype=np.float64
                ).reshape(3, 3)
                position_error = desired_position - current_position
                rotation_error = _orientation_error(current_rotation, desired_rotation)
                if np.linalg.norm(position_error) <= 0.0025 and np.linalg.norm(rotation_error) <= 0.025:
                    break
                jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
                jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
                self._mujoco.mj_jacSite(
                    model,
                    scratch,
                    jacobian_position,
                    jacobian_rotation,
                    int(ids["tool_site"]),
                )
                jacobian = np.vstack(
                    (jacobian_position[:, dofs], 0.45 * jacobian_rotation[:, dofs])
                )
                error = np.concatenate((position_error, 0.45 * rotation_error))
                damping = 0.025
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping**2 * np.eye(6), error
                )
                delta = np.clip(delta, -0.08, 0.08)
                for index, (joint, address) in enumerate(zip(ids["arm_joints"], qpos_addresses)):
                    value = float(scratch.qpos[address] + delta[index])
                    if bool(model.jnt_limited[joint]):
                        value = float(np.clip(value, *model.jnt_range[joint]))
                    scratch.qpos[address] = value
            results.append(
                np.asarray([scratch.qpos[address] for address in qpos_addresses], dtype=np.float64)
            )
        return tuple(results)

    @staticmethod
    def _joint_phase_command(
        spec: ScenarioSpec,
        timestamp: float,
        home_command: np.ndarray,
        waypoint_commands: Sequence[np.ndarray],
    ) -> np.ndarray:
        if spec.branch == IntendedBranch.NO_OP:
            # A no-op is a measured stationary-control branch.  Never solve or
            # apply its descriptive phase waypoint as an actuator target.
            return home_command.copy()
        delayed = max(0.0, timestamp - spec.controller_latency_s)
        previous = home_command
        for phase, target in zip(spec.phases, waypoint_commands):
            if delayed < phase.start_s:
                return previous.copy()
            if delayed <= phase.end_s:
                fraction = (delayed - phase.start_s) / (phase.end_s - phase.start_s)
                if phase.interpolation == "hold":
                    fraction = 1.0
                elif phase.interpolation == "smoothstep":
                    fraction = _smoothstep(fraction)
                return previous + fraction * (target - previous)
            previous = target
        return previous.copy()

    @staticmethod
    def _object_state(data: Any, ids: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
        qpos = int(ids["object_qpos"])
        dof = int(ids["object_dof"])
        return (
            np.asarray(data.qpos[qpos : qpos + 3], dtype=np.float64).copy(),
            np.asarray(data.qpos[qpos + 3 : qpos + 7], dtype=np.float64).copy(),
            np.asarray(data.qvel[dof : dof + 3], dtype=np.float64).copy(),
            np.asarray(data.qvel[dof + 3 : dof + 6], dtype=np.float64).copy(),
        )

    def _active_object_contacts(
        self,
        model: Any,
        data: Any,
        ids: Mapping[str, Any],
    ) -> list[tuple[int, str, np.ndarray, float]]:
        object_geom = int(ids["object_geom"])
        active: list[tuple[int, str, np.ndarray, float]] = []
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if object_geom not in {int(contact.geom1), int(contact.geom2)}:
                continue
            other_id = int(contact.geom2) if int(contact.geom1) == object_geom else int(contact.geom1)
            other_name = self._mujoco.mj_id2name(
                model, self._mujoco.mjtObj.mjOBJ_GEOM, other_id
            ) or f"geom_{other_id}"
            if other_name.startswith("native_tool_"):
                role = "native_tool"
            else:
                role = other_name
            # MuJoCo's contact normal points from geom1 toward geom2.  Persist
            # fixture-to-object normals so an approaching object has a
            # negative projected velocity and a separating object a positive
            # one, independent of MuJoCo's geom ordering.
            normal = np.asarray(contact.frame[:3], dtype=np.float64).copy()
            if int(contact.geom1) == object_geom:
                normal *= -1.0
            active.append((contact_index, role, normal, max(0.0, -float(contact.dist))))
        return active

    def _contact_events(
        self,
        model: Any,
        data: Any,
        spec: ScenarioSpec,
        active: Sequence[tuple[int, str, np.ndarray, float]],
        armed_roles: set[str],
        previous_velocity: np.ndarray,
        current_velocity: np.ndarray,
        audit: _RuntimeAudit,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for contact_index, role, normal, penetration in active:
            audit.maximum_penetration_m = max(audit.maximum_penetration_m, penetration)
            if role not in armed_roles:
                continue
            contact = data.contact[contact_index]
            force = np.zeros(6, dtype=np.float64)
            self._mujoco.mj_contactForce(model, data, contact_index, force)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-12:
                continue
            normal /= normal_norm
            initial_support = bool(
                spec.family == NativeFamily.ROLLING_INTERCEPTION
                and role != "native_tool"
                and float(data.time) <= 2.0 * float(model.opt.timestep) + 1e-12
            )
            events.append(
                {
                    "timestamp": float(data.time),
                    "object_a": OBJECT_BODY,
                    "object_b": role,
                    "point_world_m": [float(value) for value in contact.pos],
                    "normal_world": [float(value) for value in normal],
                    "penetration_depth_m": penetration,
                    "normal_force_n": abs(float(force[0])),
                    "normal_impulse_n_s": abs(float(force[0])) * float(model.opt.timestep),
                    "relative_velocity_world_m_s": [float(value) for value in previous_velocity],
                    "relative_velocity_post_world_m_s": [float(value) for value in current_velocity],
                    "normal_velocity_pre_m_s": float(np.dot(previous_velocity, normal)),
                    "normal_velocity_post_m_s": float(np.dot(current_velocity, normal)),
                    "normal_velocity_source": "adjacent_solver_step_diagnostic_not_restitution_measurement",
                    "expected_fixture_contact": role != "native_tool",
                    "snag": False,
                    "contact_role": (
                        "initial_support"
                        if initial_support
                        else (
                            "task_contact"
                            if role == "native_tool"
                            else "physics_context"
                        )
                    ),
                    "event_type": "contact_begin",
                }
            )
            armed_roles.discard(role)
        return events

    @staticmethod
    def _motion_mode(
        spec: ScenarioSpec,
        position: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        active_roles: set[str],
        impact_roles: set[str],
    ) -> tuple[str, str, str]:
        if impact_roles:
            role = (
                "native_tool"
                if "native_tool" in impact_roles
                else sorted(impact_roles)[0]
            )
            return (
                "impact",
                role,
                "robot_tool" if role == "native_tool" else "fixture",
            )
        if "native_tool" in active_roles:
            speed = float(np.linalg.norm(linear_velocity))
            return (
                "impact" if speed >= 0.10 else "retained",
                "native_tool",
                "robot_tool",
            )
        surfaces = sorted(role for role in active_roles if role != "native_tool")
        if not surfaces:
            return "free_flight", "none", "none"
        surface = surfaces[0]
        speed = float(np.linalg.norm(linear_velocity))
        if speed < 0.035 and float(np.linalg.norm(angular_velocity)) < 0.7:
            role = "fixture" if any(token in surface for token in ("wall", "barrier", "container")) else "support"
            return "stationary", surface, role
        if spec.object.shape == RigidShape.SPHERE:
            contact_offset = np.array((0.0, 0.0, -spec.object.radius_m))
            slip = linear_velocity + np.cross(angular_velocity, contact_offset)
            if float(np.linalg.norm(slip[:2])) <= 0.10:
                return "rolling", surface, "support"
        role = "fixture" if any(token in surface for token in ("wall", "barrier", "container")) else "support"
        return "sliding", surface, role

    def _capture_sample(
        self,
        model: Any,
        data: Any,
        ids: Mapping[str, Any],
        spec: ScenarioSpec,
        command: np.ndarray,
        target_position: np.ndarray,
        target_rpy: np.ndarray,
        phase: str,
        enabled: bool,
        active_roles: set[str],
        impact_roles: set[str] | None = None,
    ) -> _Sample:
        position, quaternion, velocity, angular_velocity = self._object_state(data, ids)
        mode, active_surface, contact_role = self._motion_mode(
            spec,
            position,
            velocity,
            angular_velocity,
            active_roles,
            impact_roles or set(),
        )
        tool_body = int(ids["tool_body"])
        qpos_addresses = [int(model.jnt_qposadr[joint]) for joint in ids["arm_joints"]]
        dof_addresses = [int(model.jnt_dofadr[joint]) for joint in ids["arm_joints"]]
        return _Sample(
            timestamp=float(data.time),
            object_position=position,
            object_quaternion=quaternion,
            object_linear_velocity=velocity,
            object_angular_velocity=angular_velocity,
            tool_position=np.asarray(data.xpos[tool_body], dtype=np.float64).copy(),
            tool_quaternion=np.asarray(data.xquat[tool_body], dtype=np.float64).copy(),
            tool_rotation=np.asarray(data.xmat[tool_body], dtype=np.float64).reshape(3, 3).copy(),
            joint_position=np.asarray([data.qpos[address] for address in qpos_addresses], dtype=np.float64),
            joint_velocity=np.asarray([data.qvel[address] for address in dof_addresses], dtype=np.float64),
            joint_command=command.copy(),
            tool_target_position=target_position.copy(),
            tool_target_rpy=target_rpy.copy(),
            phase=phase,
            motion_mode=mode,
            active_surface=active_surface,
            contact_role=contact_role,
            controller_enabled=enabled,
        )

    def _decisive_condition(
        self,
        spec: ScenarioSpec,
        sample: _Sample,
        events: Sequence[Mapping[str, Any]],
        transitions: Sequence[Mapping[str, Any]],
    ) -> bool:
        contacts = [str(event.get("object_b")) for event in events]
        chronological = [
            *[
                (float(event.get("timestamp", 0.0)), str(event.get("object_b")))
                for event in events
            ],
            *[
                (float(event.get("timestamp", 0.0)), str(event.get("to")))
                for event in transitions
                if event.get("event_type") == "motion_mode_transition"
            ],
        ]
        observed = [value for _timestamp, value in sorted(chronological)]

        def stable_surface_release(surface: str) -> bool:
            from .evaluators import _transition_surface_release_evidence

            evidence = _transition_surface_release_evidence(
                spec,
                events,
                transitions,
                surface=surface,
                terminal_time_s=sample.timestamp,
            )
            return bool(evidence["passed"])

        if spec.family == NativeFamily.FALLING_CATCH:
            inside = self._inside_tool(spec, sample)
            if spec.scenario in {
                RigidScenario.CATCH_TRANSPORT,
                RigidScenario.CATCH_BRAKE,
                RigidScenario.CATCH_TILT,
                RigidScenario.CATCH_EDGE_RECOVERY,
            }:
                return inside and sample.phase == "terminal_hold"
            return inside and "native_tool" in contacts and sample.timestamp >= 0.70
        if spec.family == NativeFamily.ROLLING_INTERCEPTION:
            if spec.scenario == RigidScenario.ROLL_OFF_EDGE:
                return stable_surface_release("table_surface")
            if spec.scenario == RigidScenario.RAMP_TO_TABLE:
                return _ordered_subsequence(("ramp_surface", "table_surface"), contacts)
            if "native_tool" in contacts:
                return True
            return sample.timestamp >= 1.4 and sample.motion_mode == "stationary"
        if spec.scenario == RigidScenario.RAMP_LAUNCH:
            return stable_surface_release("ramp_surface")
        if spec.scenario == RigidScenario.PROJECTILE_ROLL_OFF_EDGE:
            return stable_surface_release("table_surface")
        if spec.expected_contact_sequence:
            return _ordered_subsequence(spec.expected_contact_sequence, observed)
        return bool(events) and sample.timestamp >= 1.0

    @staticmethod
    def _terminal_reason(
        spec: ScenarioSpec,
        sample: _Sample,
        decisive_since: float | None,
    ) -> str | None:
        # Preserve pre-event context, then let the per-scenario terminal dwell
        # determine how much outcome context is required.  A blanket 0.75 s
        # floor forced short projectile interactions to record avoidable
        # secondary ground bounces after their outcome was already observable.
        if sample.timestamp < 0.35:
            return None
        if sample.object_position[2] < -0.38:
            return "object_below_workspace"
        if float(np.linalg.norm(sample.object_position[:2])) > 2.6:
            return "object_outside_workspace"
        if spec.family == NativeFamily.ROLLING_INTERCEPTION:
            if abs(float(sample.object_position[1])) > 0.78:
                return "object_left_table_laterally"
        if (
            decisive_since is not None
            and sample.timestamp - decisive_since >= spec.minimum_terminal_context_s
        ):
            return "terminal_context_complete"
        return None

    def _render_frames(
        self,
        model: Any,
        data: Any,
        renderer: Any | None,
        frames: dict[str, list[np.ndarray]],
        segmentation_observations: dict[str, list[dict[str, Any]]],
        *,
        object_geom_id: int,
        tool_geom_ids: frozenset[int],
        fixture_geom_ids: frozenset[int],
    ) -> None:
        if renderer is None:
            return

        def mask_summary(mask: np.ndarray) -> dict[str, Any]:
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

        for camera_name in ("main", "secondary"):
            renderer.update_scene(data, camera=camera_name)
            rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
            frames[camera_name].append(rgb)
            renderer.enable_segmentation_rendering()
            try:
                segmentation = np.asarray(renderer.render(), dtype=np.int32).copy()
            finally:
                renderer.disable_segmentation_rendering()
            if segmentation.ndim != 3 or segmentation.shape[2] != 2:
                raise NativeBackendError(
                    "MuJoCo segmentation rendering did not return HxWx2 object IDs"
                )
            geom_ids = segmentation[:, :, 0]
            geom_type = segmentation[:, :, 1] == int(
                self._mujoco.mjtObj.mjOBJ_GEOM
            )
            object_mask = geom_type & (geom_ids == object_geom_id)
            tool_mask = geom_type & np.isin(geom_ids, tuple(tool_geom_ids))
            fixture_mask = geom_type & np.isin(geom_ids, tuple(fixture_geom_ids))
            luminance = rgb.astype(np.float32).mean(axis=2)
            segmentation_observations[camera_name].append(
                {
                    "object": mask_summary(object_mask),
                    "tool": mask_summary(tool_mask),
                    "fixture": mask_summary(fixture_mask),
                    "mean_luminance": float(luminance.mean()),
                    "underexposed_fraction": float(np.mean(luminance <= 3.0)),
                    "overexposed_fraction": float(np.mean(luminance >= 252.0)),
                }
            )

    @staticmethod
    def _sample_to_state(sample: _Sample) -> dict[str, Any]:
        return {
            "timestamp": sample.timestamp,
            "object.position": sample.object_position.tolist(),
            "object.quaternion_wxyz": sample.object_quaternion.tolist(),
            "object.linear_velocity": sample.object_linear_velocity.tolist(),
            "object.angular_velocity": sample.object_angular_velocity.tolist(),
            "robot.joint_position": sample.joint_position.tolist(),
            "robot.joint_velocity": sample.joint_velocity.tolist(),
            "robot.tool_position": sample.tool_position.tolist(),
            "robot.tool_quaternion_wxyz": sample.tool_quaternion.tolist(),
            "task.phase": sample.phase,
            "object.motion_mode": sample.motion_mode,
            "object.active_surface": sample.active_surface,
            "contact.role": sample.contact_role,
            "assistance.active": False,
        }

    @staticmethod
    def _sample_to_action(sample: _Sample) -> dict[str, Any]:
        return {
            "timestamp": sample.timestamp,
            "command.joint_position": sample.joint_command.tolist(),
            "command.tool_target_position": sample.tool_target_position.tolist(),
            "command.tool_target_rpy": sample.tool_target_rpy.tolist(),
            "command.enabled": sample.controller_enabled,
        }

    @staticmethod
    def _inside_tool(spec: ScenarioSpec, sample: _Sample) -> bool:
        if spec.tool.kind in {ToolKind.FLAT_PADDLE, ToolKind.ANGLED_PADDLE}:
            return False
        relative_world = sample.object_position - sample.tool_position
        relative_local = sample.tool_rotation.T @ relative_world
        hx, hy, _ = spec.tool.half_extents_m
        radial_inside = abs(float(relative_local[0])) <= hy and abs(float(relative_local[1])) <= hx
        vertical_inside = -(
            spec.tool.wall_height_m + 2.0 * spec.object.radius_m
        ) <= float(relative_local[2]) <= spec.object.radius_m
        return radial_inside and vertical_inside

    def _evaluate_outcome(
        self,
        spec: ScenarioSpec,
        samples: Sequence[_Sample],
        events: Sequence[Mapping[str, Any]],
        transition_events: Sequence[Mapping[str, Any]],
    ) -> tuple[OutcomeResult, str]:
        # The persisted-table evaluator is the source of truth.  Running the
        # same function before writing guarantees that offline recomputation is
        # byte-for-byte label equivalent and cannot read intended branch names.
        from .evaluators import evaluate_saved_native_episode

        persisted_rows = []
        for sample in samples:
            persisted_rows.append(
                {
                    **self._sample_to_state(sample),
                    "task_phase": sample.phase,
                    "motion_mode": sample.motion_mode,
                    "active_surface": sample.active_surface,
                    "contact_role": sample.contact_role,
                    "action.command.enabled": sample.controller_enabled,
                    "action.command.joint_position": sample.joint_command.tolist(),
                    "action.command.tool_target_position": sample.tool_target_position.tolist(),
                }
            )
        recomputed = evaluate_saved_native_episode(
            spec,
            persisted_rows,
            events,
            transition_events,
        )
        return recomputed.outcome, recomputed.actual_outcome_class

    def _physics_qc(
        self,
        model: Any,
        spec: ScenarioSpec,
        sim_samples: Sequence[_Sample],
        events: Sequence[Mapping[str, Any]],
        audit: _RuntimeAudit,
    ) -> dict[str, Any]:
        finite = all(
            np.isfinite(values).all()
            for sample in sim_samples
            for values in (
                sample.object_position,
                sample.object_quaternion,
                sample.object_linear_velocity,
                sample.object_angular_velocity,
                sample.joint_position,
                sample.joint_velocity,
            )
        )
        gravity_errors: list[float] = []
        measured_accelerations: list[np.ndarray] = []
        velocity_errors: list[float] = []
        energy_values: list[float] = []
        ballistic_energy_reference_time: float | None = None
        gravity = np.asarray(spec.gravity_m_s2, dtype=np.float64)
        first_contact_time = min(
            (float(event["timestamp"]) for event in events), default=math.inf
        )
        next_contact_time = min(
            (
                float(event["timestamp"])
                for event in events
                if float(event["timestamp"])
                > first_contact_time + float(model.opt.timestep)
            ),
            default=math.inf,
        )
        post_release_ballistic = spec.scenario in {
            RigidScenario.ROLL_OFF_EDGE,
            RigidScenario.RAMP_LAUNCH,
            RigidScenario.PROJECTILE_ROLL_OFF_EDGE,
        }
        release_x = spec.extras.get("transition_release_x_m")
        release_direction = int(spec.extras.get("transition_direction", 1))
        for previous, current in zip(sim_samples, sim_samples[1:]):
            dt = current.timestamp - previous.timestamp
            if dt <= 0:
                continue
            # Contact-mode classification can flicker for a few solver steps
            # during rebound.  Physics-law checks therefore use the unambiguous
            # pre-first-contact free-flight interval only.
            free_pair = previous.motion_mode == current.motion_mode == "free_flight"
            if post_release_ballistic:
                ballistic_interval = bool(
                    free_pair
                    and release_x is not None
                    and previous.timestamp > first_contact_time + model.opt.timestep
                    and current.timestamp < next_contact_time - model.opt.timestep
                    and release_direction
                    * (float(previous.object_position[0]) - float(release_x))
                    >= 0.0
                    and release_direction
                    * (float(current.object_position[0]) - float(release_x))
                    >= 0.0
                )
            else:
                ballistic_interval = bool(
                    free_pair
                    and current.timestamp < first_contact_time - model.opt.timestep
                )
            if ballistic_interval:
                measured_acceleration = (
                    current.object_linear_velocity - previous.object_linear_velocity
                ) / dt
                measured_accelerations.append(measured_acceleration)
                gravity_errors.append(float(np.linalg.norm(measured_acceleration - gravity)))
                finite_difference = (current.object_position - previous.object_position) / dt
                reference_velocity = 0.5 * (
                    current.object_linear_velocity + previous.object_linear_velocity
                )
                velocity_errors.append(float(np.linalg.norm(finite_difference - reference_velocity)))
                if ballistic_energy_reference_time is None:
                    ballistic_energy_reference_time = current.timestamp
                # MuJoCo's semi-implicit Euler update has a known
                # -0.5*m*|g|^2*dt^2 energy bias per gravity-only step. Remove
                # that deterministic integrator term before assessing drift.
                integrator_correction = (
                    0.5
                    * spec.object.mass_kg
                    * float(np.dot(gravity, gravity))
                    * float(model.opt.timestep)
                    * (current.timestamp - ballistic_energy_reference_time)
                )
                energy_values.append(
                    0.5
                    * spec.object.mass_kg
                    * float(
                        np.dot(
                            current.object_linear_velocity,
                            current.object_linear_velocity,
                        )
                    )
                    - spec.object.mass_kg
                    * float(np.dot(gravity, current.object_position))
                    + integrator_correction
                )
        gravity_rmse = math.sqrt(sum(value * value for value in gravity_errors) / len(gravity_errors)) if gravity_errors else 0.0
        measured_gravity = (
            np.mean(np.asarray(measured_accelerations), axis=0)
            if measured_accelerations
            else None
        )
        measured_gravity_magnitude = (
            float(np.linalg.norm(measured_gravity))
            if measured_gravity is not None
            else None
        )
        velocity_rmse = math.sqrt(sum(value * value for value in velocity_errors) / len(velocity_errors)) if velocity_errors else 0.0
        energy_drift = (
            (max(energy_values) - min(energy_values)) / max(abs(np.mean(energy_values)), 1e-9)
            if len(energy_values) >= 2
            else 0.0
        )
        maximum_joint_velocity = max(
            (float(np.max(np.abs(sample.joint_velocity))) for sample in sim_samples),
            default=0.0,
        )
        maximum_joint_position_violation = 0.0
        for sample in sim_samples:
            for index, position in enumerate(sample.joint_position, start=1):
                joint_id = self._mujoco.mj_name2id(
                    model, self._mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}"
                )
                if joint_id >= 0 and bool(model.jnt_limited[joint_id]):
                    low, high = model.jnt_range[joint_id]
                    maximum_joint_position_violation = max(
                        maximum_joint_position_violation,
                        float(low) - float(position),
                        float(position) - float(high),
                        0.0,
                    )
        maximum_joint_acceleration = 0.0
        for previous, current in zip(sim_samples, sim_samples[1:]):
            dt = current.timestamp - previous.timestamp
            if dt > 0:
                maximum_joint_acceleration = max(
                    maximum_joint_acceleration,
                    float(np.max(np.abs(current.joint_velocity - previous.joint_velocity))) / dt,
                )
        restitution_measurements: list[dict[str, Any]] = []
        restitution_diagnostics = {
            "eligible_fixture_contact_count": 0,
            "missing_incoming_sample_count": 0,
            "missing_outgoing_sample_count": 0,
            "invalid_normal_count": 0,
            "invalid_direction_count": 0,
        }
        if spec.family != NativeFamily.ROLLING_INTERCEPTION:
            restitution_measurements, restitution_diagnostics = (
                _measure_static_fixture_restitution(sim_samples, events)
            )
        subfamily = str(spec.extras.get("suite_subfamily") or "")
        gravity_sweep = subfamily == "gravity_sweep"
        restitution_sweep = subfamily == "restitution_sweep"
        friction_sweep = subfamily == "friction_sweep"
        expected_static_fixtures = tuple(
            role
            for role in spec.expected_contact_sequence
            if role.endswith("_surface")
        )
        target_restitution_fixture = (
            expected_static_fixtures[0] if expected_static_fixtures else None
        )
        target_restitution_measurement = next(
            (
                measurement
                for measurement in restitution_measurements
                if measurement.get("fixture") == target_restitution_fixture
            ),
            restitution_measurements[0] if restitution_measurements else None,
        )
        measured_effective_restitution = (
            float(target_restitution_measurement["measured_effective_restitution"])
            if target_restitution_measurement is not None
            else None
        )
        restitution_target_match_required = bool(
            restitution_sweep
            or (
                spec.physics_provenance.calibrated
                and target_restitution_fixture is not None
            )
        )

        # A sliding-friction intervention must create a measured response, not
        # merely a different XML value.  The dedicated sweep uses a horizontal
        # sliding puck, so work-energy gives an effective coefficient from its
        # observed stopping distance.  All values come from native sim states.
        friction_response_sample_count = 0
        friction_stop_observed = False
        friction_stopping_distance_m: float | None = None
        measured_effective_dynamic_friction: float | None = None
        if friction_sweep and sim_samples:
            initial_horizontal_velocity = np.asarray(
                sim_samples[0].object_linear_velocity[:2], dtype=np.float64
            )
            initial_horizontal_speed = float(np.linalg.norm(initial_horizontal_velocity))
            if initial_horizontal_speed > 1e-9:
                direction = initial_horizontal_velocity / initial_horizontal_speed
                initial_position = np.asarray(
                    sim_samples[0].object_position[:2], dtype=np.float64
                )
                stop_index: int | None = None
                for index, sample in enumerate(sim_samples[1:], start=1):
                    horizontal_speed = float(
                        np.linalg.norm(sample.object_linear_velocity[:2])
                    )
                    if (
                        sample.timestamp >= 2.0 * float(model.opt.timestep)
                        and horizontal_speed <= 0.02
                    ):
                        stop_index = index
                        break
                response_samples = (
                    sim_samples[: stop_index + 1]
                    if stop_index is not None
                    else sim_samples
                )
                friction_response_sample_count = len(response_samples)
                friction_stop_observed = stop_index is not None
                signed_travel = [
                    float(
                        np.dot(
                            np.asarray(sample.object_position[:2], dtype=np.float64)
                            - initial_position,
                            direction,
                        )
                    )
                    for sample in response_samples
                ]
                distance = max(signed_travel, default=0.0)
                if distance > 1e-6:
                    friction_stopping_distance_m = distance
                    gravity_magnitude = float(np.linalg.norm(gravity))
                    if friction_stop_observed and gravity_magnitude > 1e-9:
                        measured_effective_dynamic_friction = (
                            initial_horizontal_speed**2
                            / (2.0 * gravity_magnitude * distance)
                        )
        rolling_slip_speeds: list[float] = []
        for sample in sim_samples:
            if sample.motion_mode != "rolling" or spec.object.shape != RigidShape.SPHERE:
                continue
            bottom_velocity = sample.object_linear_velocity + np.cross(
                sample.object_angular_velocity,
                np.array((0.0, 0.0, -spec.object.radius_m)),
            )
            rolling_slip_speeds.append(float(np.linalg.norm(bottom_velocity[:2])))
        maximum_rolling_slip = max(rolling_slip_speeds, default=0.0)
        task_contact_count = sum(event.get("object_b") == "native_tool" for event in events)
        initial_support_event_count = sum(
            event.get("contact_role") == "initial_support" for event in events
        )
        distinct_contact_count = sum(
            event.get("contact_role") != "initial_support" for event in events
        )
        forces_finite = all(
            math.isfinite(float(event.get("normal_force_n", 0.0)))
            and float(event.get("normal_force_n", 0.0)) >= 0.0
            for event in events
        )
        momentum_delta = np.zeros(3, dtype=np.float64)
        gravity_impulse = np.zeros(3, dtype=np.float64)
        contact_impulse = np.asarray(
            audit.integrated_contact_impulse_n_s, dtype=np.float64
        )
        momentum_residual = 0.0
        momentum_relative_error = 0.0
        if len(sim_samples) >= 2:
            elapsed = sim_samples[-1].timestamp - sim_samples[0].timestamp
            momentum_delta = spec.object.mass_kg * (
                sim_samples[-1].object_linear_velocity
                - sim_samples[0].object_linear_velocity
            )
            gravity_impulse = spec.object.mass_kg * gravity * elapsed
            momentum_residual = float(
                np.linalg.norm(momentum_delta - gravity_impulse - contact_impulse)
            )
            momentum_scale = max(
                float(np.linalg.norm(momentum_delta)),
                float(np.linalg.norm(gravity_impulse)),
                float(np.linalg.norm(contact_impulse)),
                1e-6,
            )
            momentum_relative_error = momentum_residual / momentum_scale
        ballistic_evidence_required = bool(
            spec.family in {
                NativeFamily.FALLING_CATCH,
                NativeFamily.PROJECTILE_REBOUND,
            }
            or spec.scenario == RigidScenario.ROLL_OFF_EDGE
        )
        checks = {
            "finite_state": bool(finite),
            "no_post_initialization_object_state_writes": bool(audit.object_state_writes_after_initialization == 0),
            "no_direct_robot_state_writes_after_initialization": bool(audit.direct_robot_state_writes_after_initialization == 0),
            "no_equality_or_latch_assistance": bool(model.neq == 0),
            "contact_penetration_bounded": bool(audit.maximum_penetration_m <= 0.02),
            "control_within_declared_ranges": bool(audit.maximum_ctrl_range_violation <= 1e-12),
            "free_flight_measurement_available": bool(
                not ballistic_evidence_required
                or len(measured_accelerations) >= 2
            ),
            "free_flight_acceleration_consistent": bool(
                (not ballistic_evidence_required and not gravity_errors)
                or (bool(gravity_errors) and gravity_rmse <= 1.2)
            ),
            "gravity_sweep_measurement_available": bool(
                not gravity_sweep
                or (
                    len(measured_accelerations) >= 5
                    and measured_gravity_magnitude is not None
                )
            ),
            "position_velocity_consistent": bool(not velocity_errors or velocity_rmse <= 0.12),
            "free_flight_energy_consistent": bool(len(energy_values) < 2 or energy_drift <= 0.08),
            "joint_velocity_bounded": bool(maximum_joint_velocity <= 3.5),
            "joint_acceleration_bounded": bool(maximum_joint_acceleration <= 80.0),
            "joint_positions_within_limits": bool(
                maximum_joint_position_violation <= 1e-8
            ),
            "actuator_forces_within_limits": bool(
                audit.maximum_actuator_force_range_violation <= 1e-8
            ),
            "contact_forces_finite": bool(forces_finite),
            "momentum_impulse_accounting_consistent": bool(
                len(sim_samples) >= 2
                and np.isfinite(contact_impulse).all()
                and momentum_relative_error <= 0.08
            ),
            "task_contact_count_bounded": bool(task_contact_count <= spec.max_task_contacts),
            "distinct_contact_count_bounded": bool(distinct_contact_count <= spec.max_task_contacts),
            "no_measured_contact_energy_gain": bool(
                all(
                    float(measurement["measured_effective_restitution"]) <= 1.05
                    for measurement in restitution_measurements
                )
            ),
            "measured_restitution_matches_target": bool(
                not restitution_target_match_required
                or (
                    measured_effective_restitution is not None
                    and abs(
                        measured_effective_restitution
                        - spec.object.effective_restitution_target
                    )
                    <= 0.25
                )
            ),
            "restitution_sweep_measurement_available": bool(
                not restitution_sweep or bool(restitution_measurements)
            ),
            "friction_sweep_measurement_available": bool(
                not friction_sweep
                or (
                    friction_response_sample_count >= 5
                    and friction_stop_observed
                    and measured_effective_dynamic_friction is not None
                )
            ),
            "rolling_slip_bounded": bool(not rolling_slip_speeds or maximum_rolling_slip <= 0.12),
        }
        return default_physics_qc(
            **checks,
            gravity_acceleration_rmse_m_s2=gravity_rmse,
            free_flight_pair_count=len(measured_accelerations),
            measured_gravity_vector_m_s2=(
                [float(value) for value in measured_gravity]
                if measured_gravity is not None
                else "not_observed"
            ),
            measured_gravity_magnitude_m_s2=(
                measured_gravity_magnitude
                if measured_gravity_magnitude is not None
                else "not_observed"
            ),
            position_velocity_rmse_m_s=velocity_rmse,
            free_flight_relative_energy_drift=energy_drift,
            maximum_contact_penetration_m=audit.maximum_penetration_m,
            maximum_joint_velocity_rad_s=maximum_joint_velocity,
            maximum_joint_acceleration_rad_s2=maximum_joint_acceleration,
            maximum_joint_position_limit_violation_rad=maximum_joint_position_violation,
            maximum_actuator_force_range_violation_n=audit.maximum_actuator_force_range_violation,
            maximum_actuator_force_abs_n=audit.maximum_actuator_force_abs,
            object_momentum_delta_n_s=[float(value) for value in momentum_delta],
            gravity_impulse_n_s=[float(value) for value in gravity_impulse],
            integrated_contact_impulse_n_s=[float(value) for value in contact_impulse],
            momentum_impulse_residual_n_s=momentum_residual,
            momentum_impulse_relative_error=momentum_relative_error,
            momentum_impulse_sample_count=audit.contact_impulse_sample_count,
            effective_restitution_target=spec.object.effective_restitution_target,
            restitution_target_match_required=int(
                restitution_target_match_required
            ),
            restitution_target_fixture=(
                target_restitution_fixture or "not_declared"
            ),
            measured_effective_restitution=(
                measured_effective_restitution
                if measured_effective_restitution is not None
                else "not_observed"
            ),
            rebound_measurement_count=len(restitution_measurements),
            restitution_measurement_sample_count=sum(
                int(measurement["sample_count"])
                for measurement in restitution_measurements
            ),
            restitution_measurement_source=_RESTITUTION_MEASUREMENT_SOURCE,
            restitution_measurements=restitution_measurements,
            restitution_eligible_fixture_contact_count=restitution_diagnostics[
                "eligible_fixture_contact_count"
            ],
            restitution_missing_incoming_sample_count=restitution_diagnostics[
                "missing_incoming_sample_count"
            ],
            restitution_missing_outgoing_sample_count=restitution_diagnostics[
                "missing_outgoing_sample_count"
            ],
            restitution_invalid_normal_count=restitution_diagnostics[
                "invalid_normal_count"
            ],
            restitution_invalid_direction_count=restitution_diagnostics[
                "invalid_direction_count"
            ],
            friction_response_sample_count=friction_response_sample_count,
            friction_stop_observed_count=int(friction_stop_observed),
            friction_stopping_distance_m=(
                friction_stopping_distance_m
                if friction_stopping_distance_m is not None
                else "not_observed"
            ),
            measured_effective_dynamic_friction=(
                measured_effective_dynamic_friction
                if measured_effective_dynamic_friction is not None
                else "not_observed"
            ),
            task_contact_count=task_contact_count,
            distinct_contact_count=distinct_contact_count,
            recorded_contact_event_count=len(events),
            initial_support_event_count=initial_support_event_count,
            maximum_rolling_slip_m_s=maximum_rolling_slip,
            audit=audit.to_dict(),
        )

    def _visibility_qc(
        self,
        samples: Sequence[_Sample],
        calibrations: Mapping[str, CameraCalibration],
        frames: Mapping[str, Sequence[np.ndarray]],
        events: Sequence[Mapping[str, Any]],
        segmentation_observations: Mapping[str, Sequence[Mapping[str, Any]]],
        camera_latency_frames: int = 0,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": NATIVE_VISUAL_QC_SCHEMA,
            "evaluated": bool(samples)
            and all(bool(frames.get(name)) for name in calibrations),
            "key_event_visible_in_any_view": False,
            "critically_cropped": True,
            "contact_occluded_both_views": True,
            "minimum_visible_fraction": 0.0,
            "target_visible_frame_fraction": 0.0,
            "minimum_bbox_margin_px": 0.0,
            "key_event_object_area_px": 0,
            "tool_visible_at_key_event": False,
            "fixture_visible_at_key_event": False,
            "maximum_underexposed_fraction": 1.0,
            "maximum_overexposed_fraction": 1.0,
            "thresholds": dict(NATIVE_VISUAL_THRESHOLDS),
            "views": {},
        }
        if not samples:
            return result
        key_event = next(
            (event for event in events if event.get("object_b") == "native_tool"),
            events[0] if events else None,
        )
        key_time = (
            float(key_event["timestamp"])
            if key_event is not None
            else samples[-1].timestamp
        )
        result["key_event_counterpart"] = (
            str(key_event.get("object_b")) if key_event is not None else None
        )
        key_index = min(
            range(len(samples)), key=lambda index: abs(samples[index].timestamp - key_time)
        )
        encoded_key_index = key_index + camera_latency_frames
        visible_ratios_by_frame: list[float] = []
        visible_pixels_by_frame: list[int] = []
        for sample_index in range(len(samples)):
            encoded_index = sample_index + camera_latency_frames
            per_view_pixels: list[int] = []
            per_view_ratios: list[float] = []
            for name in calibrations:
                observations = segmentation_observations.get(name, ())
                if not observations:
                    continue
                observation = observations[min(encoded_index, len(observations) - 1)]
                object_summary = observation.get("object")
                if not isinstance(object_summary, Mapping):
                    continue
                pixels = int(object_summary.get("pixel_count", 0))
                bbox = object_summary.get("bbox_xyxy")
                bbox_area = 0
                if isinstance(bbox, Sequence) and len(bbox) == 4:
                    bbox_area = max(1, int(bbox[2]) - int(bbox[0])) * max(
                        1, int(bbox[3]) - int(bbox[1])
                    )
                per_view_pixels.append(pixels)
                per_view_ratios.append(min(1.0, pixels / max(1, bbox_area)))
            visible_pixels_by_frame.append(max(per_view_pixels, default=0))
            visible_ratios_by_frame.append(max(per_view_ratios, default=0.0))
        result["minimum_visible_fraction"] = min(
            visible_ratios_by_frame, default=0.0
        )
        result["target_visible_frame_fraction"] = sum(
            pixels >= 64 for pixels in visible_pixels_by_frame
        ) / max(1, len(visible_pixels_by_frame))

        key_object_visible: list[bool] = []
        key_contact_visible: list[bool] = []
        key_margins: list[float] = []
        key_areas: list[int] = []
        underexposed: list[float] = []
        overexposed: list[float] = []
        for name, calibration in calibrations.items():
            visible = 0
            for sample in samples:
                try:
                    x, y, _depth = calibration.project_world(sample.object_position)
                except ValueError:
                    continue
                if 0 <= x < calibration.width and 0 <= y < calibration.height:
                    visible += 1
            camera_frames = frames.get(name, ())
            nonempty = bool(camera_frames) and any(float(np.std(frame)) > 2.0 for frame in camera_frames)
            fraction = visible / max(1, len(samples))
            observations = segmentation_observations.get(name, ())
            observation = (
                observations[min(encoded_key_index, len(observations) - 1)]
                if observations
                else {}
            )
            object_summary = observation.get("object", {})
            tool_summary = observation.get("tool", {})
            fixture_summary = observation.get("fixture", {})
            object_pixels = int(object_summary.get("pixel_count", 0))
            tool_pixels = int(tool_summary.get("pixel_count", 0))
            fixture_pixels = int(fixture_summary.get("pixel_count", 0))
            object_bbox = object_summary.get("bbox_xyxy")
            bbox_margin = 0.0
            if isinstance(object_bbox, Sequence) and len(object_bbox) == 4:
                bbox_margin = float(
                    min(
                        int(object_bbox[0]),
                        int(object_bbox[1]),
                        calibration.width - int(object_bbox[2]),
                        calibration.height - int(object_bbox[3]),
                    )
                )
            key_margins.append(bbox_margin)
            key_areas.append(object_pixels)
            counterpart_pixels = (
                tool_pixels
                if key_event is not None and key_event.get("object_b") == "native_tool"
                else fixture_pixels
            )
            segmentation_key_visible = object_pixels >= 64 and bbox_margin >= 8
            segmentation_contact_visible = (
                segmentation_key_visible and counterpart_pixels >= 16
            )
            key_object_visible.append(segmentation_key_visible)
            key_contact_visible.append(segmentation_contact_visible)
            if observation:
                underexposed.append(float(observation.get("underexposed_fraction", 1.0)))
                overexposed.append(float(observation.get("overexposed_fraction", 1.0)))
            key_crop_contrast = 0.0
            try:
                key_x, key_y, _key_depth = calibration.project_world(
                    samples[key_index].object_position
                )
                margin_x = 0.05 * calibration.width
                margin_y = 0.05 * calibration.height
                projected_key_visible = (
                    margin_x <= key_x < calibration.width - margin_x
                    and margin_y <= key_y < calibration.height - margin_y
                )
                if projected_key_visible and camera_frames:
                    frame = camera_frames[
                        min(
                            key_index + camera_latency_frames,
                            len(camera_frames) - 1,
                        )
                    ]
                    x0, x1 = max(0, int(key_x) - 8), min(calibration.width, int(key_x) + 9)
                    y0, y1 = max(0, int(key_y) - 8), min(calibration.height, int(key_y) + 9)
                    key_crop_contrast = float(np.std(frame[y0:y1, x0:x1]))
                    projected_key_visible = key_crop_contrast > 1.5
            except ValueError:
                projected_key_visible = False
            result["views"][name] = {
                "projected_object_visible_fraction": fraction,
                "nonempty_render": nonempty,
                "key_event_visible": bool(
                    projected_key_visible and segmentation_key_visible and nonempty
                ),
                "key_event_crop_contrast": key_crop_contrast,
                "key_event_object_pixel_count": object_pixels,
                "key_event_object_bbox_xyxy": object_bbox,
                "key_event_bbox_margin_px": bbox_margin,
                "key_event_tool_pixel_count": tool_pixels,
                "key_event_fixture_pixel_count": fixture_pixels,
                "key_event_contact_visible": segmentation_contact_visible,
                "mean_luminance": observation.get("mean_luminance"),
                "underexposed_fraction": observation.get("underexposed_fraction"),
                "overexposed_fraction": observation.get("overexposed_fraction"),
            }
            result["key_event_visible_in_any_view"] = bool(
                result["key_event_visible_in_any_view"]
                or (projected_key_visible and segmentation_key_visible and nonempty)
            )
        result["critically_cropped"] = not any(key_object_visible)
        result["contact_occluded_both_views"] = not any(key_contact_visible)
        result["minimum_bbox_margin_px"] = max(key_margins, default=0.0)
        result["key_event_object_area_px"] = max(key_areas, default=0)
        result["tool_visible_at_key_event"] = any(
            int(value.get("key_event_tool_pixel_count", 0)) >= 16
            for value in result["views"].values()
        )
        result["fixture_visible_at_key_event"] = any(
            int(value.get("key_event_fixture_pixel_count", 0)) >= 16
            for value in result["views"].values()
        )
        result["maximum_underexposed_fraction"] = max(underexposed, default=1.0)
        result["maximum_overexposed_fraction"] = max(overexposed, default=1.0)
        return result

    def run(
        self,
        plan: EpisodePlan | NativeEpisodePlan,
        *,
        render: bool = True,
    ) -> BackendRunResult:
        native_plan = self.compile(plan)
        spec = native_plan.scenario
        model, data, description = self._compile_model(native_plan)
        ids = self._ids(model)
        audit = _RuntimeAudit(equality_constraint_count=int(model.neq))
        self._initialize_state(model, data, spec, audit)
        home_command = np.asarray(
            [data.ctrl[actuator] for actuator in ids["arm_actuators"]], dtype=np.float64
        )
        waypoint_commands = self._solve_ik_waypoints(
            model, ids, spec, home_command
        )
        target_position, target_rpy, phase, enabled = _phase_target(spec, 0.0)
        requested_command = self._joint_phase_command(
            spec, 0.0, home_command, waypoint_commands
        )
        command = self._update_controls(
            model, data, ids, requested_command, audit
        )
        command_velocity = np.zeros_like(command)
        control_stride = spec.sim_hz // spec.control_hz
        frame_stride = spec.sim_hz // spec.video_hz
        maximum_steps = int(round(spec.maximum_duration_s * spec.sim_hz))
        frames: dict[str, list[np.ndarray]] = {"main": [], "secondary": []}
        segmentation_observations: dict[str, list[dict[str, Any]]] = {
            "main": [],
            "secondary": [],
        }
        tool_geom_ids = frozenset(
            int(self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in description.task_tool_geoms
        )
        fixture_geom_ids = frozenset(
            int(self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_GEOM, name))
            for name in description.surface_geoms
        )
        if any(value < 0 for value in (*tool_geom_ids, *fixture_geom_ids)):
            raise NativeBackendError("compiled model is missing declared tool/fixture geometry")
        renderer = None
        if render:
            try:
                # MuJoCo 3.10 selects EGL devices through the process-level
                # MUJOCO_EGL_DEVICE_ID setting; Renderer itself intentionally
                # has no device argument.
                renderer = self._mujoco.Renderer(
                    model,
                    height=self.height,
                    width=self.width,
                )
            except Exception as error:
                raise NativeBackendError(
                    "MuJoCo rendering failed. Use an EGL allocation (MUJOCO_GL=egl); "
                    "the native backend does not silently substitute diagnostic frames."
                ) from error
        frame_samples: list[_Sample] = []
        sim_samples: list[_Sample] = []
        high_rate_samples: list[_Sample] = []
        events: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        armed_roles: set[str] = set()
        absent_steps: dict[str, int] = {}
        rearm_steps = max(1, int(round(0.12 * spec.sim_hz)))
        # MuJoCo contacts can disappear for a few solver steps while a
        # rigid body remains geometrically supported.  Smooth only the
        # semantic mode classifier; raw contacts, forces, and event rows stay
        # untouched.  A 20 ms support-only carry remains below one 30 Hz video
        # period and prevents state rows/transition sidecars from alternating
        # between support and free flight because of solver chatter.
        classification_hysteresis_steps = max(
            1, int(math.ceil(0.020 * spec.sim_hz))
        )
        active = self._active_object_contacts(model, data, ids)
        active_roles = {role for _index, role, _normal, _penetration in active}
        last_role_normals = {
            role: np.asarray(normal, dtype=np.float64).copy()
            for _index, role, normal, _penetration in active
        }
        initial_sample = self._capture_sample(
            model,
            data,
            ids,
            spec,
            command,
            target_position,
            target_rpy,
            phase,
            enabled,
            active_roles,
        )
        sim_samples.append(initial_sample)
        high_rate_samples.append(initial_sample)
        frame_samples.append(initial_sample)
        self._render_frames(
            model,
            data,
            renderer,
            frames,
            segmentation_observations,
            object_geom_id=int(ids["object_geom"]),
            tool_geom_ids=tool_geom_ids,
            fixture_geom_ids=fixture_geom_ids,
        )
        audit.rendered_frame_count += int(render)
        previous_velocity = initial_sample.object_linear_velocity.copy()
        previous_mode = initial_sample.motion_mode
        previous_phase = initial_sample.phase
        decisive_since: float | None = None
        fixed_counterfactual_horizon = bool(spec.physics_counterfactual_family_id)
        for step in range(1, maximum_steps + 1):
            # The policy/controller target is sampled into the persisted table
            # at ``control_hz``, but a real Franka position controller does not
            # apply one discontinuous setpoint step every 1/control_hz seconds.
            # Interpolate the same actuator-only command at the native servo
            # rate so the official high-gain Panda actuator model is not driven
            # by artificial zero-order-hold torque impulses.
            target_position, target_rpy, phase, enabled = _phase_target(
                spec, float(data.time)
            )
            requested_command = self._joint_phase_command(
                spec, float(data.time), home_command, waypoint_commands
            )
            requested_command, command_velocity = self._rate_limit_joint_command(
                requested_command,
                command,
                command_velocity,
                dt=1.0 / spec.sim_hz,
            )
            command = self._update_controls(
                model, data, ids, requested_command, audit
            )
            self._mujoco.mj_step(model, data)
            audit.simulation_steps += 1
            object_constraint_force = np.asarray(
                data.qfrc_constraint[
                    int(ids["object_dof"]) : int(ids["object_dof"]) + 3
                ],
                dtype=np.float64,
            )
            impulse_increment = object_constraint_force * float(model.opt.timestep)
            audit.integrated_contact_impulse_n_s = [
                float(value)
                for value in (
                    np.asarray(audit.integrated_contact_impulse_n_s)
                    + impulse_increment
                )
            ]
            if float(np.linalg.norm(object_constraint_force)) > 1e-9:
                audit.contact_impulse_sample_count += 1
            for actuator in range(model.nu):
                force = float(data.actuator_force[actuator])
                audit.maximum_actuator_force_abs = max(
                    audit.maximum_actuator_force_abs, abs(force)
                )
                if bool(model.actuator_forcelimited[actuator]):
                    low, high = model.actuator_forcerange[actuator]
                    violation = max(float(low) - force, force - float(high), 0.0)
                    audit.maximum_actuator_force_range_violation = max(
                        audit.maximum_actuator_force_range_violation, violation
                    )
            active = self._active_object_contacts(model, data, ids)
            active_roles = {role for _index, role, _normal, _penetration in active}
            _position, _quaternion, current_velocity, _angular = self._object_state(
                data, ids
            )
            for _index, role, normal, _penetration in active:
                last_role_normals[role] = np.asarray(
                    normal, dtype=np.float64
                ).copy()
            known_roles = set(absent_steps) | set(active_roles)
            for role in known_roles:
                if role in active_roles:
                    if role not in absent_steps:
                        armed_roles.add(role)
                    absent_steps[role] = 0
                else:
                    absent_steps[role] = absent_steps.get(role, 0) + 1
                    if absent_steps[role] >= rearm_steps:
                        armed_roles.add(role)
            classification_roles = set(active_roles)
            active_fixture_roles = {
                role for role in active_roles if role != "native_tool"
            }
            if not active_fixture_roles and "native_tool" not in active_roles:
                classification_roles.update(
                    role
                    for role, count in absent_steps.items()
                    if role in {"table_surface", "ramp_surface"}
                    and 0 < count <= classification_hysteresis_steps
                    and role in last_role_normals
                    and float(
                        np.dot(current_velocity, last_role_normals[role])
                    )
                    <= 0.10
                )
            new_contact_events = self._contact_events(
                model,
                data,
                spec,
                active,
                armed_roles,
                previous_velocity,
                current_velocity,
                audit,
            )
            events.extend(new_contact_events)
            impact_roles = {
                str(event["object_b"])
                for event in new_contact_events
                if event.get("contact_role") != "initial_support"
                and float(event.get("normal_velocity_pre_m_s", 0.0)) <= -0.10
            }
            sample = self._capture_sample(
                model,
                data,
                ids,
                spec,
                command,
                target_position,
                target_rpy,
                phase,
                enabled,
                classification_roles,
                impact_roles,
            )
            previous_sample = sim_samples[-1]
            sim_samples.append(sample)
            if sample.motion_mode != previous_mode:
                transitions.append(
                    {
                        "timestamp": sample.timestamp,
                        "event_type": "motion_mode_transition",
                        "from": previous_mode,
                        "to": sample.motion_mode,
                        "active_surface": sample.active_surface,
                        "object_position_m": [
                            float(value) for value in sample.object_position
                        ],
                    }
                )
                previous_mode = sample.motion_mode
            release_surface = {
                RigidScenario.ROLL_OFF_EDGE: "table_surface",
                RigidScenario.RAMP_LAUNCH: "ramp_surface",
                RigidScenario.PROJECTILE_ROLL_OFF_EDGE: "table_surface",
            }.get(spec.scenario)
            release_x_raw = spec.extras.get("transition_release_x_m")
            release_direction = int(spec.extras.get("transition_direction", 1))
            if (
                release_surface is not None
                and release_x_raw is not None
                and release_direction in {-1, 1}
                and sample.motion_mode == "free_flight"
                and sample.active_surface == "none"
            ):
                release_x = float(release_x_raw)
                previous_side = release_direction * (
                    float(previous_sample.object_position[0]) - release_x
                )
                current_side = release_direction * (
                    float(sample.object_position[0]) - release_x
                )
                crossed_boundary = previous_side < 0.0 <= current_side
                became_free_beyond_boundary = bool(
                    previous_sample.motion_mode != "free_flight"
                    and current_side >= 0.0
                )
                boundary_already_recorded = any(
                    event.get("event_type")
                    == "surface_release_boundary_crossing"
                    and event.get("surface") == release_surface
                    for event in transitions
                )
                if (
                    (crossed_boundary or became_free_beyond_boundary)
                    and not boundary_already_recorded
                ):
                    transitions.append(
                        {
                            "timestamp": sample.timestamp,
                            "event_type": "surface_release_boundary_crossing",
                            "from": "pre_release_region",
                            "to": "post_release_region",
                            "surface": release_surface,
                            "release_x_m": release_x,
                            "direction": release_direction,
                            "object_position_m": [
                                float(value) for value in sample.object_position
                            ],
                        }
                    )
            if sample.phase != previous_phase:
                transitions.append(
                    {
                        "timestamp": sample.timestamp,
                        "event_type": "task_phase_transition",
                        "from": previous_phase,
                        "to": sample.phase,
                    }
                )
                previous_phase = sample.phase
            if step % control_stride == 0:
                high_rate_samples.append(sample)
            if step % frame_stride == 0:
                frame_samples.append(sample)
                self._render_frames(
                    model,
                    data,
                    renderer,
                    frames,
                    segmentation_observations,
                    object_geom_id=int(ids["object_geom"]),
                    tool_geom_ids=tool_geom_ids,
                    fixture_geom_ids=fixture_geom_ids,
                )
                audit.rendered_frame_count += int(render)
                decisive = self._decisive_condition(spec, sample, events, transitions)
                if decisive and decisive_since is None:
                    decisive_since = sample.timestamp
                elif not decisive and spec.family == NativeFamily.FALLING_CATCH:
                    # Retention must remain continuous; a spill resets the
                    # terminal-context dwell.  In contrast, a verified
                    # surface departure is an irreversible task event.  A
                    # later landing must not erase a launch/edge transition
                    # that already satisfied its minimum free-flight dwell.
                    decisive_since = None
                terminal = self._terminal_reason(spec, sample, decisive_since)
                if terminal is not None and not fixed_counterfactual_horizon:
                    audit.terminal_reason = terminal
                    break
            previous_velocity = current_velocity
        if fixed_counterfactual_horizon:
            # Physics siblings must persist the same timestamped action table.
            # Outcome-dependent early stopping would otherwise create an
            # undeclared action intervention and invalidate the family.
            audit.terminal_reason = "physics_counterfactual_fixed_horizon_complete"
        if renderer is not None:
            renderer.close()
        camera_latency_frames = int(round(spec.camera_latency_s * spec.video_hz))
        if camera_latency_frames:
            for camera_name, camera_frames in frames.items():
                if camera_frames:
                    frames[camera_name] = (
                        [
                            camera_frames[0].copy()
                            for _ in range(camera_latency_frames)
                        ]
                        + camera_frames[:-camera_latency_frames]
                    )
                observations = segmentation_observations[camera_name]
                if observations:
                    segmentation_observations[camera_name] = (
                        [dict(observations[0]) for _ in range(camera_latency_frames)]
                        + observations[:-camera_latency_frames]
                    )
        outcome, actual_outcome = self._evaluate_outcome(spec, frame_samples, events, transitions)
        physics_qc = self._physics_qc(model, spec, sim_samples, events, audit)
        states = [self._sample_to_state(sample) for sample in frame_samples]
        actions = [self._sample_to_action(sample) for sample in frame_samples]
        visual_style_artifact = str(
            spec.extras.get("visual_style_validation_artifact_hash") or ""
        )
        visual_style_artifact_ok, visual_style_artifact_path, visual_style_actual_hash = (
            _verify_content_bound_artifact(
                spec.extras.get("visual_style_validation_artifact_path"),
                visual_style_artifact,
            )
        )
        visual_style_validated = bool(
            spec.extras.get("visual_style_validated", False)
            and visual_style_artifact_ok
        )
        tool_calibration_artifact = str(
            spec.extras.get("tool_calibration_artifact_hash") or ""
        )
        tool_artifact_ok, tool_artifact_path, tool_actual_hash = (
            _verify_content_bound_artifact(
                spec.extras.get("tool_calibration_artifact_path"),
                tool_calibration_artifact,
            )
        )
        tool_calibrated = bool(
            spec.extras.get("tool_calibrated", False) and tool_artifact_ok
        )
        physics_artifact_ok, physics_artifact_path, physics_actual_hash = (
            _verify_content_bound_artifact(
                spec.physics_provenance.calibration_artifact_path,
                spec.physics_provenance.calibration_artifact,
            )
        )
        physics_range_calibrated = bool(
            spec.physics_provenance.calibrated and physics_artifact_ok
        )
        # This backend mounts synthetic tray/paddle/bin geometry.  Even if all
        # historical calibration fields are supplied, it is not the required
        # Panda-hand or Robotiq embodiment and can never become production
        # eligible.  Keep the other checks for regression diagnostics only.
        simulator_production_eligible = False
        simulation = SimulationResult(
            plan=native_plan.episode_plan,
            frame_times_s=[sample.timestamp for sample in frame_samples],
            states=states,
            actions=actions,
            high_rate_states=[
                {
                    **self._sample_to_state(sample),
                    **{
                        f"action.{key}": value
                        for key, value in self._sample_to_action(sample).items()
                        if key != "timestamp"
                    },
                }
                for sample in high_rate_samples
            ],
            contacts=events,
            outcome=outcome,
            actual_outcome=actual_outcome,
            dynamics_mode="free_contact",
            release_tier="free_contact",
            assistance=assistance_record(),
            physics_qc=physics_qc,
            simulator={
                "name": "MuJoCo",
                "version": self._mujoco.__version__,
                "backend": self.name,
                "backend_version": self.version,
                "native_mujoco": True,
                "production_eligible": simulator_production_eligible,
                "contact_resolution": "native_mujoco_constraint_solver",
                "object_state_writes_after_initialization": 0,
                "direct_robot_state_writes_after_initialization": 0,
                "coordinate_frame": "world_right_handed_z_up",
                "quaternion_order": "wxyz",
                "model_hash": description.model_hash,
            },
            notes=(
                "RETIRED: synthetic task attachment; not Panda-hand or Robotiq production data.",
                "Object qpos/qvel are written only during initialization; all subsequent motion uses native MuJoCo dynamics.",
                "Franka motion uses seven joint position actuators driven by Cartesian damped-least-squares control.",
                "MuJoCo contact parameters are solver settings; effective restitution is measured from the rollout when observable.",
            ),
        )
        calibrations = {
            camera.name: _camera_calibration(camera, self.width, self.height)
            for camera in spec.cameras
        }
        visibility = self._visibility_qc(
            frame_samples,
            calibrations,
            frames,
            events,
            segmentation_observations,
            camera_latency_frames=camera_latency_frames,
        )
        camera_roles = {camera.name: camera.role for camera in spec.cameras}
        expected_secondary_role = (
            "top_oblique"
            if spec.family == NativeFamily.ROLLING_INTERCEPTION
            else "side"
        )
        visibility["camera_roles"] = camera_roles
        visibility["camera_roles_correct"] = (
            camera_roles.get("main") == "main_three_quarter"
            and camera_roles.get("secondary") == expected_secondary_role
        )
        thresholds = visibility["thresholds"]
        quality_flags: list[str] = ["retired_custom_attachment_backend"]
        if not self.asset.verified_menagerie_layout:
            quality_flags.append("unverified_franka_asset")
        if not physics_range_calibrated:
            quality_flags.append("physics_range_uncalibrated")
        if not tool_calibrated:
            quality_flags.append("tool_calibration_unverified")
        if not visual_style_validated:
            quality_flags.append("visual_style_unvalidated")
        if not physics_qc.get("physics_qc_pass", False):
            quality_flags.append("native_physics_qc_failed")
        if render:
            if not visibility.get("key_event_visible_in_any_view", False):
                quality_flags.append("key_event_visibility_failed")
            if visibility.get("critically_cropped", True):
                quality_flags.append("critical_crop_failed")
            if visibility.get("contact_occluded_both_views", True):
                quality_flags.append("critical_contact_visibility_failed")
            if float(visibility.get("target_visible_frame_fraction", 0.0)) < float(
                thresholds["minimum_target_visible_frame_fraction"]
            ):
                quality_flags.append("whole_episode_target_visibility_failed")
            if float(visibility.get("minimum_bbox_margin_px", 0.0)) < float(
                thresholds["minimum_bbox_margin_px"]
            ):
                quality_flags.append("target_bbox_margin_failed")
            if int(visibility.get("key_event_object_area_px", 0)) < int(
                thresholds["minimum_key_event_object_area_px"]
            ):
                quality_flags.append("target_pixel_footprint_failed")
            if not visibility.get("tool_visible_at_key_event", False):
                quality_flags.append("tool_visibility_failed")
            if not visibility.get("fixture_visible_at_key_event", False):
                quality_flags.append("fixture_visibility_failed")
            if float(visibility.get("maximum_underexposed_fraction", 1.0)) > float(
                thresholds["maximum_underexposed_fraction"]
            ):
                quality_flags.append("underexposure_failed")
            if float(visibility.get("maximum_overexposed_fraction", 1.0)) > float(
                thresholds["maximum_overexposed_fraction"]
            ):
                quality_flags.append("overexposure_failed")
            if not visibility.get("camera_roles_correct", False):
                quality_flags.append("camera_role_failed")
        if not render:
            quality_flags.append("render_disabled")
        frame_rows = []
        object_rows = []
        for frame_index, (state, action) in enumerate(zip(states, actions)):
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "video_frame_index": frame_index,
                    "timestamp": state["timestamp"],
                    "contact.active": state["contact.role"] != "none",
                    "event.contact": any(
                        abs(float(event["timestamp"]) - float(state["timestamp"]))
                        <= 0.5 / spec.video_hz
                        for event in events
                    ),
                    "task_phase": state["task.phase"],
                    "motion_mode": state["object.motion_mode"],
                    "active_surface": state["object.active_surface"],
                    "contact_role": state["contact.role"],
                    "object.free_fall": state["object.motion_mode"] == "free_flight",
                    **{key: value for key, value in state.items() if key != "timestamp"},
                    **{
                        f"action.{key}": value
                        for key, value in action.items()
                        if key != "timestamp"
                    },
                }
            )
            object_rows.append(
                {
                    "timestamp": state["timestamp"],
                    "object_id": OBJECT_BODY,
                    "position_m": state["object.position"],
                    "quaternion_wxyz": state["object.quaternion_wxyz"],
                    "linear_velocity_m_s": state["object.linear_velocity"],
                    "angular_velocity_rad_s": state["object.angular_velocity"],
                    "motion_mode": state["object.motion_mode"],
                    "active_surface": state["object.active_surface"],
                }
            )
        backend_provenance = {
            "backend": self.name,
            "backend_version": self.version,
            "simulator": "MuJoCo",
            "simulator_version": self._mujoco.__version__,
            "renderer": "mujoco.Renderer",
            "production_eligible": simulator_production_eligible,
            "retired_custom_attachment_backend": True,
            "allowed_production_end_effector": False,
            "model_hash": description.model_hash,
            "scenario_hash": native_plan.compiled_scenario_hash,
            "fixed_field_hash": native_plan.fixed_field_hash,
            "action_replay_hash": native_plan.action_replay_hash,
            "initial_state_hash": stable_hash(spec.initial_state),
            "robot_start_joint_offsets_rad": list(
                spec.robot_start_joint_offsets_rad
            ),
            "franka_asset_source": self.asset.source,
            "franka_xml_path": str(self.asset.xml_path),
            "franka_xml_sha256": self.asset.xml_sha256,
            "franka_license_sha256": self.asset.license_sha256,
            "franka_asset_verified": self.asset.verified_menagerie_layout,
            "physics_range_provenance": {
                **spec.physics_provenance.__dict__,
                "content_verified": physics_artifact_ok,
                "resolved_artifact_path": physics_artifact_path,
                "actual_artifact_sha256": physics_actual_hash,
            },
            "tool_calibration_id": spec.tool_calibration_id,
            "tool_calibrated": tool_calibrated,
            "tool_calibration_artifact_path": tool_artifact_path,
            "tool_calibration_artifact_sha256": tool_calibration_artifact or None,
            "tool_calibration_actual_sha256": tool_actual_hash,
            "controller_profile_id": "franka_cartesian_ik_joint_position",
            "controller_profile_version": "1.0.0",
            "low_level_servo_rate_hz": spec.sim_hz,
            "persisted_controller_rate_hz": spec.control_hz,
            "command_velocity_limit_rad_s": 1.5,
            "command_acceleration_limit_rad_s2": 10.0,
            "control_latency_s": spec.controller_latency_s,
            "camera_latency_s": spec.camera_latency_s,
            "camera_latency_frames": camera_latency_frames,
            "camera_stream_to_calibration": {
                "observation.images.main": "main",
                "observation.images.secondary": "secondary",
            },
            "camera_calibration_ids": {
                camera.name: f"{native_plan.fixed_field_hash[:16]}-{camera.name}"
                for camera in spec.cameras
            },
            "runtime_audit": audit.to_dict(),
            "native_scenario_spec": spec.to_dict(),
            "duration_policy": (
                "fixed_physics_counterfactual_horizon"
                if fixed_counterfactual_horizon
                else "event_adaptive"
            ),
            "planned_horizon_s": spec.maximum_duration_s,
            "realized_duration_s": frame_samples[-1].timestamp,
            "visual_style_requested": spec.scene_style,
            "visual_style_implementation": "procedural_native_material_alias",
            "visual_style_validated": visual_style_validated,
            "visual_style_validation_artifact_hash": visual_style_artifact or None,
            "visual_style_validation_artifact_path": visual_style_artifact_path,
            "visual_style_validation_actual_sha256": visual_style_actual_hash,
        }
        return BackendRunResult(
            simulation=simulation,
            frames_by_camera=frames,
            camera_calibrations=calibrations,
            frame_rows=frame_rows,
            high_rate_rows=simulation.high_rate_states,
            event_rows=events,
            object_state_rows=object_rows,
            transition_events=transitions,
            backend_provenance=backend_provenance,
            quality_flags=tuple(sorted(set(quality_flags))),
            visibility_qc=visibility,
        )
