from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .scene_builder import EpisodeSample
from .utils import as_float_list, smoothstep


FINGERTIP_PAD_LOCAL_CENTER = np.array([0.0, 0.0055, 0.0445], dtype=np.float64)
ROLLING_INTERCEPT_CLEARANCE = 0.006
ROLLING_GRASP_XY_TOL = 0.035
ROLLING_GRASP_Z_TOL = 0.045
ROLLING_CLOSE_Z_TOL = 0.025
ROLLING_CLOSE_XY_TOL = 0.035
ROLLING_CLOSE_GRACE_S = 0.3
ROLLING_ARM_MAX_JOINT_SPEED_RAD_S = 4.0
ROLLING_ARM_MAX_JOINT_ACCEL_RAD_S2 = 35.0
ROLLING_ARM_CORRECTION_GAIN_S = 15.0
ROLLING_IK_UPDATE_INTERVAL_STEPS = 4
ROLLING_LIFT_HEIGHT = 0.12
ROLLING_LIFT_DURATION_S = 0.32
DEFAULT_HAND_TARGET_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class JointIds:
    joint_names: list[str]
    qpos_adrs: np.ndarray
    qvel_adrs: np.ndarray
    joint_ids: np.ndarray
    ball_qpos_adr: int
    ball_qvel_adr: int
    hand_body: int
    left_finger_body: int
    right_finger_body: int
    ball_body: int


class MujocoCatchController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, sample: EpisodeSample):
        self.model = model
        self.data = data
        self.sample = sample
        self.ids = self._resolve_ids()
        self.home_q = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.78, 0.04, 0.04], dtype=np.float64)
        self.open_q: np.ndarray | None = None
        self.closed_q: np.ndarray | None = None
        self.intercept_position = np.array(
            [sample.ball_initial_position[0], sample.ball_initial_position[1], sample.catch_center_z],
            dtype=np.float64,
        )
        self.closed = False
        self.close_time_s: float | None = None
        self.close_ball_pos: list[float] | None = None
        self.first_contact_time_s: float | None = None
        self.max_held_steps = 0
        self.held_steps = 0
        self.contact_frames = 0
        self.success = False
        self.predicted_close_time_s: float | None = None
        # Once the ball is genuinely captured between the closed fingers, hold
        # its captured offset. This avoids a visual snap to an artificial target.
        self.grasped = False
        self.grasp_offset: np.ndarray | None = None
        self.capture_steps = 0
        self.grasp_time_s: float | None = None
        self.hand_to_fingertip_center_local: np.ndarray | None = None
        self.current_target_hand_pos: np.ndarray | None = None
        self.current_target_hand_quat: np.ndarray | None = None
        self.current_target_arm_q: np.ndarray | None = None
        self.latest_target_update_time_s: float | None = None
        self.rolling_target_quat: np.ndarray | None = None
        self.lift_start_time_s: float | None = None
        self.lift_start_fingertip_center: np.ndarray | None = None
        self.rolling_arm_qvel: np.ndarray | None = None
        self.rolling_target_qvel_estimate: np.ndarray | None = None
        self._arm_joint_ranges_cache: np.ndarray | None = None

    def _name2id(self, objtype, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, objtype, name)
        if idx < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return idx

    def _resolve_ids(self) -> JointIds:
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "finger_joint2"]
        joint_ids = np.array([self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names], dtype=np.int32)
        ball_joint = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        return JointIds(
            joint_names=joint_names,
            qpos_adrs=np.array([self.model.jnt_qposadr[j] for j in joint_ids], dtype=np.int32),
            qvel_adrs=np.array([self.model.jnt_dofadr[j] for j in joint_ids], dtype=np.int32),
            joint_ids=joint_ids,
            ball_qpos_adr=int(self.model.jnt_qposadr[ball_joint]),
            ball_qvel_adr=int(self.model.jnt_dofadr[ball_joint]),
            hand_body=self._name2id(mujoco.mjtObj.mjOBJ_BODY, "hand"),
            left_finger_body=self._name2id(mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
            right_finger_body=self._name2id(mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
            ball_body=self._name2id(mujoco.mjtObj.mjOBJ_BODY, "catch_ball"),
        )

    def prepare(self) -> None:
        arm_q = self._solve_ik(
            self.intercept_position + np.array([0.0, 0.0, -0.075]),
            DEFAULT_HAND_TARGET_QUAT,
        )
        self.open_q = np.r_[arm_q, [0.04, 0.04]]
        closed_finger = float(np.clip(0.85 * self.sample.ball_radius, 0.002, 0.04))
        self.closed_q = np.r_[arm_q, [closed_finger, closed_finger]]
        close_z = self.sample.catch_center_z + 0.055
        time_to_close_z = self._time_until_ball_reaches_z(close_z, prefer_descending=True)
        if time_to_close_z is None:
            time_to_close_z = self._time_until_ball_reaches_z(close_z) or 0.30
        self.predicted_close_time_s = self.sample.release_time_s + time_to_close_z - 0.025
        self.data.qpos[self.ids.qpos_adrs] = self.home_q
        self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)
        mujoco.mj_forward(self.model, self.data)
        self._cache_hand_to_fingertip_center_local()

    def _arm_joint_ranges(self) -> np.ndarray:
        if self._arm_joint_ranges_cache is None:
            arm_jids = self.ids.joint_ids[:7]
            ranges = self.model.jnt_range[arm_jids].copy()
            for i, jid in enumerate(arm_jids):
                if not self.model.jnt_limited[jid]:
                    ranges[i] = [-3.0, 3.0]
            self._arm_joint_ranges_cache = ranges
        return self._arm_joint_ranges_cache

    def _solve_ik(
        self,
        target_hand_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        init_q: np.ndarray | None = None,
    ) -> np.ndarray:
        arm_adrs = self.ids.qpos_adrs[:7]
        ranges = self._arm_joint_ranges()
        init = np.asarray(init_q, dtype=np.float64).copy() if init_q is not None else self.home_q[:7].copy()
        init = np.clip(init, ranges[:, 0], ranges[:, 1])
        target_quat = np.asarray(target_quat, dtype=np.float64)
        target_quat /= np.linalg.norm(target_quat)

        def quat_error(current_quat):
            inv = np.array([current_quat[0], -current_quat[1], -current_quat[2], -current_quat[3]])
            out = np.empty(4)
            mujoco.mju_mulQuat(out, target_quat, inv)
            if out[0] < 0:
                out *= -1
            return out[1:]

        def residual(q):
            self.data.qpos[arm_adrs] = q
            self.data.qpos[self.ids.qpos_adrs[7:9]] = [0.04, 0.04]
            mujoco.mj_forward(self.model, self.data)
            pos_err = self.data.xpos[self.ids.hand_body] - target_hand_pos
            rot_err = quat_error(self.data.xquat[self.ids.hand_body].copy())
            regularizer = q - init
            return np.r_[3.0 * pos_err, 0.4 * rot_err, 0.02 * regularizer]

        result = least_squares(
            residual,
            init,
            bounds=(ranges[:, 0], ranges[:, 1]),
            max_nfev=400,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        return result.x

    def _cache_hand_to_fingertip_center_local(self) -> None:
        finger_center = self._fingertip_pad_center()
        hand_pos = self.data.xpos[self.ids.hand_body]
        hand_mat = self.data.xmat[self.ids.hand_body].reshape(3, 3)
        self.hand_to_fingertip_center_local = hand_mat.T @ (finger_center - hand_pos)

    def _rolling_target_quat(self, planar_velocity: np.ndarray) -> np.ndarray:
        path_dir = np.asarray(planar_velocity, dtype=np.float64).reshape(-1)
        if path_dir.size == 2:
            path_dir = np.array([path_dir[0], path_dir[1], 0.0], dtype=np.float64)
        else:
            path_dir = path_dir.copy()
            path_dir[2] = 0.0
        norm = float(np.linalg.norm(path_dir))
        if norm <= 1e-9:
            path_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            path_dir /= norm
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        x_axis = path_dir
        y_axis = np.cross(z_axis, x_axis)
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm <= 1e-9:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            y_axis /= y_norm
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        target_mat = np.column_stack((x_axis, y_axis, z_axis))
        quat_xyzw = Rotation.from_matrix(target_mat).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    def _rolling_hand_target_position(
        self,
        fingertip_center_xy: np.ndarray,
        *,
        fingertip_center_z: float,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        if self.hand_to_fingertip_center_local is None:
            raise RuntimeError("Hand-to-fingertip offset has not been initialized.")
        quat_xyzw = np.array([target_quat[1], target_quat[2], target_quat[3], target_quat[0]], dtype=np.float64)
        hand_mat = Rotation.from_quat(quat_xyzw).as_matrix()
        fingertip_center = np.array(
            [float(fingertip_center_xy[0]), float(fingertip_center_xy[1]), float(fingertip_center_z)],
            dtype=np.float64,
        )
        return fingertip_center - hand_mat @ self.hand_to_fingertip_center_local

    def _ball_position_at_time(self, t: float) -> np.ndarray:
        initial_position = np.asarray(self.sample.ball_initial_position, dtype=np.float64)
        initial_velocity = np.asarray(self.sample.ball_initial_velocity, dtype=np.float64)
        dt = max(0.0, t - self.sample.release_time_s)
        pos = initial_position + initial_velocity * dt
        if self.sample.controller_mode == "rolling_surface_intercept":
            pos[2] = initial_position[2]
        return pos

    def _current_arm_q(self) -> np.ndarray:
        return self.data.qpos[self.ids.qpos_adrs[:7]].copy()

    def _update_rolling_target(self, t: float, *, force: bool = False) -> None:
        if self.sample.controller_mode != "rolling_surface_intercept":
            return
        step_idx = int(round(t / self.sample.timestep))
        if (
            not force
            and self.current_target_arm_q is not None
            and step_idx % ROLLING_IK_UPDATE_INTERVAL_STEPS != 0
        ):
            return
        planar_velocity = np.asarray(self.sample.ball_initial_velocity[:2], dtype=np.float64)
        target_quat = self.rolling_target_quat if self.rolling_target_quat is not None else self._rolling_target_quat(planar_velocity)
        if self.grasped:
            base_center = (
                self.lift_start_fingertip_center.copy()
                if self.lift_start_fingertip_center is not None
                else self._fingertip_pad_center().copy()
            )
            lift_start = self.lift_start_time_s if self.lift_start_time_s is not None else t
            lift_alpha = smoothstep((t - lift_start) / ROLLING_LIFT_DURATION_S)
            lifted_center = base_center.copy()
            lifted_center[2] += ROLLING_LIFT_HEIGHT * lift_alpha
            target_hand_pos = self._rolling_hand_target_position(
                lifted_center[:2],
                fingertip_center_z=lifted_center[2],
                target_quat=target_quat,
            )
        else:
            ball_pos = self._ball_position_at_time(t)
            target_hand_pos = self._rolling_hand_target_position(
                ball_pos[:2],
                fingertip_center_z=ball_pos[2] + ROLLING_INTERCEPT_CLEARANCE,
                target_quat=target_quat,
            )
        target_arm_q = self._solve_ik(
            target_hand_pos,
            target_quat,
            init_q=self._current_arm_q(),
        )
        # Feedforward: the ball (and thus the IK target) moves at a roughly
        # constant rate while rolling, so estimate that rate from consecutive
        # target solves and hand it to the tracking law. Without this, the
        # tracker only reacts to positional error, and a faster-rolling ball
        # settles into a larger steady-state tracking lag (lag grows with
        # ball speed) since the feedback-only law needs a bigger error to
        # sustain a higher matching speed.
        if self.current_target_arm_q is not None and self.latest_target_update_time_s is not None:
            dt_since_update = float(t) - self.latest_target_update_time_s
            if dt_since_update > 1e-9:
                self.rolling_target_qvel_estimate = (target_arm_q - self.current_target_arm_q) / dt_since_update
        self.current_target_hand_pos = target_hand_pos
        self.current_target_hand_quat = target_quat
        self.current_target_arm_q = target_arm_q
        self.latest_target_update_time_s = float(t)

    def _set_ball_pose(self, pos, vel) -> None:
        qadr = self.ids.ball_qpos_adr
        vadr = self.ids.ball_qvel_adr
        self.data.qpos[qadr : qadr + 7] = [pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[vadr : vadr + 6] = 0.0
        # Freejoint qvel layout is [linear(3), angular(3)].
        self.data.qvel[vadr : vadr + 3] = vel
        self.data.qvel[vadr + 3 : vadr + 6] = getattr(
            self.sample,
            "ball_initial_angular_velocity",
            (0.0, 0.0, 0.0),
        )

    def _time_until_ball_reaches_z(self, target_z: float, *, prefer_descending: bool = False) -> float | None:
        z0 = float(self.sample.ball_initial_position[2])
        vz0 = float(self.sample.ball_initial_velocity[2])
        gz = float(self.sample.gravity[2])
        a = 0.5 * gz
        b = vz0
        c = z0 - float(target_z)
        eps = 1e-9

        roots: list[float] = []
        if abs(a) < eps:
            if abs(b) >= eps:
                roots.append(-c / b)
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                sqrt_discriminant = float(np.sqrt(discriminant))
                roots.append((-b - sqrt_discriminant) / (2.0 * a))
                roots.append((-b + sqrt_discriminant) / (2.0 * a))
        positive_roots = sorted(float(root) for root in roots if root >= -eps)
        if not positive_roots:
            return None
        if prefer_descending:
            descending_roots = [root for root in positive_roots if vz0 + gz * root <= 0.0]
            if descending_roots:
                return max(0.0, descending_roots[0])
        return max(0.0, positive_roots[0])

    def _fingertip_pad_center(self) -> np.ndarray:
        left_center = (
            self.data.xpos[self.ids.left_finger_body]
            + self.data.xmat[self.ids.left_finger_body].reshape(3, 3) @ FINGERTIP_PAD_LOCAL_CENTER
        )
        right_center = (
            self.data.xpos[self.ids.right_finger_body]
            + self.data.xmat[self.ids.right_finger_body].reshape(3, 3) @ FINGERTIP_PAD_LOCAL_CENTER
        )
        return 0.5 * (left_center + right_center)

    def before_step(self, t: float) -> None:
        assert self.open_q is not None and self.closed_q is not None
        if t < self.sample.release_time_s:
            self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)

        ball_pos = self.data.qpos[self.ids.ball_qpos_adr : self.ids.ball_qpos_adr + 3].copy()
        ball_vz = float(self.data.qvel[self.ids.ball_qvel_adr + 2])
        should_close = (
            t >= self.sample.release_time_s
            and (
                (self.predicted_close_time_s is not None and t >= self.predicted_close_time_s)
                or (ball_vz <= 0.0 and ball_pos[2] <= self.sample.catch_center_z + 0.08)
            )
        )
        if self.sample.controller_mode != "hold_home" and not self.closed and should_close:
            self.closed = True
            self.close_time_s = float(t)
            self.close_ball_pos = as_float_list(ball_pos)

        alpha = smoothstep(t / 0.45)
        q_cmd = (1.0 - alpha) * self.home_q + alpha * self.open_q
        if self.closed and self.sample.controller_mode != "hold_home":
            q_cmd = self.closed_q
        self.data.qpos[self.ids.qpos_adrs] = q_cmd
        self.data.qvel[self.ids.qvel_adrs] = 0.0

        if self.grasped and self.grasp_offset is not None:
            finger_center = self._fingertip_pad_center()
            self.grasp_offset *= 0.94
            qadr = self.ids.ball_qpos_adr
            vadr = self.ids.ball_qvel_adr
            self.data.qpos[qadr : qadr + 3] = finger_center + self.grasp_offset
            self.data.qvel[vadr : vadr + 6] = 0.0

    def after_step(self, t: float) -> None:
        contacts_now = 0
        ball_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "catch_ball_geom")
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == ball_geom or con.geom2 == ball_geom:
                contacts_now += 1
        if contacts_now:
            self.contact_frames += 1
            if self.first_contact_time_s is None:
                self.first_contact_time_s = float(t)

        ball_pos = self.data.xpos[self.ids.ball_body]
        finger_center = self._fingertip_pad_center()

        if self.closed and not self.grasped:
            if self.sample.controller_mode == "rolling_surface_intercept":
                captured_now = (
                    np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= ROLLING_GRASP_XY_TOL
                    and abs(ball_pos[2] - finger_center[2]) <= ROLLING_GRASP_Z_TOL
                )
            else:
                captured_now = (
                    contacts_now > 0
                    and np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.045
                    and abs(ball_pos[2] - finger_center[2]) <= 0.05
                )
            self.capture_steps = self.capture_steps + 1 if captured_now else 0
            required_capture_time = (
                0.008 if self.sample.controller_mode == "rolling_surface_intercept" else 0.02
            )
            if self.sample.enable_grasp_capture and self.capture_steps * self.sample.timestep >= required_capture_time:
                self.grasped = True
                self.grasp_offset = (ball_pos - finger_center).copy()
                self.grasp_time_s = float(t)
                if self.sample.controller_mode == "rolling_surface_intercept":
                    self.lift_start_time_s = float(t)
                    self.lift_start_fingertip_center = finger_center.copy()

        retained = (
            np.linalg.norm(ball_pos[:2] - finger_center[:2])
            <= (0.10 if self.sample.controller_mode == "rolling_surface_intercept" else 0.04)
            and abs(ball_pos[2] - finger_center[2])
            <= (0.12 if self.sample.controller_mode == "rolling_surface_intercept" else 0.05)
        )
        if retained:
            self.held_steps += 1
        else:
            self.held_steps = 0
        self.max_held_steps = max(self.max_held_steps, self.held_steps)
        self.success = self.held_steps * self.sample.timestep >= 0.40

    def result(self) -> dict:
        final_ball_position = self.data.xpos[self.ids.ball_body].copy()
        final_fingertip_center = self._fingertip_pad_center().copy()
        return {
            "success": bool(self.success),
            "intercept_position": as_float_list(self.intercept_position),
            "predicted_close_time_s": self.predicted_close_time_s,
            "gripper_close_time": self.close_time_s,
            "close_ball_position": self.close_ball_pos,
            "first_contact_time_s": self.first_contact_time_s,
            "contact_frames": int(self.contact_frames),
            "grasped": bool(self.grasped),
            "grasp_time_s": self.grasp_time_s,
            "max_retained_time_s": float(self.max_held_steps * self.sample.timestep),
            "final_ball_position": as_float_list(final_ball_position),
            "final_fingertip_center_position": as_float_list(final_fingertip_center),
            "final_fingertip_error": float(np.linalg.norm(final_ball_position - final_fingertip_center)),
            "final_ball_qpos": as_float_list(self.data.qpos[self.ids.ball_qpos_adr : self.ids.ball_qpos_adr + 7]),
            "closed_finger_joint_position": float(self.closed_q[-1]) if self.closed_q is not None else None,
            "controller": {
                "type": "scripted_ik_direct_joint_position",
                "uses_native_gripper_joints": ["finger_joint1", "finger_joint2"],
                "fallback_note": "Robot joints are set directly; after real contact capture, the ball is held at its captured offset.",
            },
        }


class MujocoInterceptionController(MujocoCatchController):
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sample: EpisodeSample,
        *,
        reach_lead_time_s: float = 0.040,
        close_lead_time_s: float = 0.024,
    ):
        super().__init__(model, data, sample)
        self.reach_lead_time_s = float(reach_lead_time_s)
        self.close_lead_time_s = float(close_lead_time_s)
        self.ballistic_intercept_time_s: float | None = None
        self.arm_arrival_time_s: float | None = None
        self.arm_reaction_time_s: float = 0.0
        self._arm_reaction_started = False

    def prepare(self) -> None:
        if self.sample.controller_mode == "rolling_surface_intercept":
            self._prepare_rolling_surface_intercept()
            return
        close_z = self.sample.catch_center_z + 0.055
        time_to_close_z = self._time_until_ball_reaches_z(close_z, prefer_descending=True)
        if time_to_close_z is None:
            time_to_close_z = self._time_until_ball_reaches_z(close_z) or 0.45

        self.ballistic_intercept_time_s = self.sample.release_time_s + time_to_close_z
        initial_position = np.asarray(self.sample.ball_initial_position, dtype=np.float64)
        initial_velocity = np.asarray(self.sample.ball_initial_velocity, dtype=np.float64)
        gravity = np.asarray(self.sample.gravity, dtype=np.float64)
        predicted_ball_position = initial_position + initial_velocity * time_to_close_z + 0.5 * gravity * time_to_close_z**2
        self.intercept_position = np.array(
            [predicted_ball_position[0], predicted_ball_position[1], self.sample.catch_center_z],
            dtype=np.float64,
        )
        self.intercept_position += np.asarray(self.sample.controller_target_offset, dtype=np.float64)

        arm_q = self._solve_ik(
            self.intercept_position + np.array([0.0, 0.0, -0.075]),
            DEFAULT_HAND_TARGET_QUAT,
        )
        self.open_q = np.r_[arm_q, [0.04, 0.04]]
        closed_finger = float(np.clip(0.85 * self.sample.ball_radius, 0.002, 0.04))
        self.closed_q = np.r_[arm_q, [closed_finger, closed_finger]]
        self.predicted_close_time_s = max(
            self.sample.release_time_s,
            self.ballistic_intercept_time_s - self.close_lead_time_s + float(self.sample.close_lead_time_offset_s),
        )
        self.arm_arrival_time_s = max(
            0.12,
            self.predicted_close_time_s - self.reach_lead_time_s + float(self.sample.reach_lead_time_offset_s),
        )
        self.data.qpos[self.ids.qpos_adrs] = self.home_q
        self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)
        mujoco.mj_forward(self.model, self.data)

    def _prepare_rolling_surface_intercept(self) -> None:
        # Absolute sim time (from episode start, t=0), not relative to
        # release_time_s: the default 0.0 must reproduce the original
        # behavior of reacting from t=0 (before the ball even starts
        # rolling) so every other rolling_surface_intercept sample --
        # dataset_generation.py's randomized episodes included -- is
        # unaffected. Callers that want "let the ball roll for a bit first"
        # need to pass a value greater than release_time_s.
        self.arm_reaction_time_s = float(self.sample.arm_reaction_delay_s)
        self._arm_reaction_started = False
        initial_position = np.asarray(self.sample.ball_initial_position, dtype=np.float64)
        initial_velocity = np.asarray(self.sample.ball_initial_velocity, dtype=np.float64)
        planar_velocity = initial_velocity[:2]
        speed = float(np.linalg.norm(planar_velocity))
        if speed <= 1e-6:
            raise ValueError("Rolling surface intercept requires nonzero planar ball velocity.")
        target_xy = np.asarray(
            (self.sample.visual_settings or {}).get("catch_position_xyz", self.sample.ball_initial_position)[:2],
            dtype=np.float64,
        )
        time_to_target = max(0.18, float(np.dot(target_xy - initial_position[:2], planar_velocity) / (speed * speed)))
        self.ballistic_intercept_time_s = self.sample.release_time_s + time_to_target
        predicted_xy = initial_position[:2] + planar_velocity * time_to_target
        self.intercept_position = np.array(
            [predicted_xy[0], predicted_xy[1], self.sample.catch_center_z],
            dtype=np.float64,
        )
        self.intercept_position += np.asarray(self.sample.controller_target_offset, dtype=np.float64)
        self.data.qpos[self.ids.qpos_adrs] = self.home_q
        self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)
        mujoco.mj_forward(self.model, self.data)
        self._cache_hand_to_fingertip_center_local()

        planar_velocity = np.asarray(self.sample.ball_initial_velocity[:2], dtype=np.float64)
        self.rolling_target_quat = self._rolling_target_quat(planar_velocity)
        self._update_rolling_target(0.0, force=True)
        if self.current_target_arm_q is None:
            raise RuntimeError("Failed to initialize rolling target arm configuration.")
        self.open_q = np.r_[self.current_target_arm_q, [0.04, 0.04]]
        closed_finger = float(np.clip(0.85 * self.sample.ball_radius, 0.002, 0.04))
        self.closed_q = np.r_[self.current_target_arm_q, [closed_finger, closed_finger]]
        self.predicted_close_time_s = max(
            self.sample.release_time_s,
            self.ballistic_intercept_time_s - 0.005 + float(self.sample.close_lead_time_offset_s),
        )
        self.arm_arrival_time_s = max(
            0.12,
            self.predicted_close_time_s - self.reach_lead_time_s + float(self.sample.reach_lead_time_offset_s),
        )
        self.data.qpos[self.ids.qpos_adrs] = self.home_q
        self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)
        mujoco.mj_forward(self.model, self.data)

    def _set_rolling_ball_state(self, t: float) -> None:
        if self.grasped:
            return
        initial_position = np.asarray(self.sample.ball_initial_position, dtype=np.float64)
        initial_velocity = np.asarray(self.sample.ball_initial_velocity, dtype=np.float64)
        dt = max(0.0, t - self.sample.release_time_s)
        pos = initial_position + initial_velocity * dt
        pos[2] = initial_position[2]
        self._set_ball_pose(pos, initial_velocity)

    def before_step(self, t: float) -> None:
        assert self.open_q is not None and self.closed_q is not None
        # Snapshot the true current joint state before _update_rolling_target
        # (via _solve_ik) mutates self.data.qpos as a side effect of its FK
        # evaluations. Reading "current" arm position after that call would
        # see the IK solver's leftover (~target) pose instead of where the
        # arm actually is, making any servo law converge instantly.
        current_arm_q = self._current_arm_q()
        if self.sample.controller_mode == "rolling_surface_intercept":
            current_full_q = self.data.qpos[self.ids.qpos_adrs].copy()
            self._set_rolling_ball_state(t)
            # Let the ball roll on its own, untracked, until arm_reaction_time_s
            # -- the arm stays parked at home_q below. Only start solving/
            # following an IK target once the reaction delay has elapsed, so
            # the demo reads as "ball moves, then the arm reacts" instead of
            # the arm already hovering over the ball's spawn point at t=0.
            if t >= self.arm_reaction_time_s:
                if not self._arm_reaction_started:
                    # First target solve after the wait: discard the stale
                    # feedforward estimate from prepare()'s seed call at
                    # t=0, or the huge elapsed dt would produce a bogus
                    # average-velocity feedforward term for this step.
                    self._arm_reaction_started = True
                    self.latest_target_update_time_s = None
                    self.rolling_target_qvel_estimate = None
                    self._update_rolling_target(t, force=True)
                else:
                    # Keep tracking through every phase (chasing the ball, closing
                    # the fingers around it, lifting once grasped) -- the ball keeps
                    # rolling under its own scripted kinematics until grasped, so
                    # freezing the target while the fingers close let it roll out
                    # from under the (now slower-converging, no-teleport) hand.
                    self._update_rolling_target(t)
            # _update_rolling_target's IK solve leaves qpos/xpos at its last
            # trial (the target pose), not the arm's true current pose.
            # Restore the true pose before reading finger_center below, or
            # the close/grasp gate checks where the arm is about to be
            # instead of where it actually is -- invisible when convergence
            # was near-instant, but now that motion is smoothly rate-limited
            # the arm can lag the target long enough for this to matter.
            self.data.qpos[self.ids.qpos_adrs] = current_full_q
            mujoco.mj_forward(self.model, self.data)
        elif t < self.sample.release_time_s:
            self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)

        ball_pos = self.data.qpos[self.ids.ball_qpos_adr : self.ids.ball_qpos_adr + 3].copy()
        ball_vz = float(self.data.qvel[self.ids.ball_qvel_adr + 2])
        finger_center = self._fingertip_pad_center()
        hand_ready = (
            self.sample.controller_mode != "rolling_surface_intercept"
            or (
                np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= max(ROLLING_CLOSE_XY_TOL, 0.75 * self.sample.ball_radius)
                and abs(ball_pos[2] - finger_center[2]) <= ROLLING_CLOSE_Z_TOL
            )
        )
        should_close = (
            t >= self.sample.release_time_s
            and (
                # The rolling-intercept hand tracks the ball continuously from
                # before release, so there's no separate "arrival" milestone
                # to wait for (arm_arrival_time_s was sized for the old
                # jump-then-wait approach). But predicted_close_time_s is a
                # property of the ball/scene, not the arm -- it's when the
                # ball reaches the designated catch point -- so it still must
                # gate closing, or hand_ready (now nearly satisfied from the
                # start, since the hand begins right on top of the ball)
                # lets the gripper snap shut the instant the ball is
                # released, before it has rolled anywhere. The tracking law
                # settles into a small residual oscillation rather than a
                # perfect lock (unlike the old instant-snap convergence), so
                # hand_ready may not land exactly at predicted_close_time_s --
                # the grace deadline is a safety net against waiting for an
                # unlucky oscillation phase.
                (
                    self.sample.controller_mode == "rolling_surface_intercept"
                    and self.predicted_close_time_s is not None
                    and t >= self.predicted_close_time_s
                    and (hand_ready or t >= self.predicted_close_time_s + ROLLING_CLOSE_GRACE_S)
                )
                or (
                    self.sample.controller_mode != "rolling_surface_intercept"
                    and self.predicted_close_time_s is not None
                    and t >= self.predicted_close_time_s
                )
                or (
                    self.sample.controller_mode != "rolling_surface_intercept"
                    and ball_vz <= 0.0
                    and ball_pos[2] <= self.sample.catch_center_z + 0.08
                )
            )
        )
        if not self.closed and should_close:
            self.closed = True
            self.close_time_s = float(t)
            self.close_ball_pos = as_float_list(ball_pos)
            if self.current_target_arm_q is not None:
                self.closed_q = np.r_[self.current_target_arm_q, self.closed_q[-2:]]

        arrival_time = self.arm_arrival_time_s if self.arm_arrival_time_s is not None else 0.45
        if self.sample.controller_mode == "hold_home":
            q_cmd = self.home_q
        elif self.sample.controller_mode == "rolling_surface_intercept" and t < self.arm_reaction_time_s:
            # Parked, waiting for the reaction delay to elapse -- the ball
            # rolls on unattended (see _set_rolling_ball_state above).
            q_cmd = self.home_q
            self.rolling_arm_qvel = np.zeros(7, dtype=np.float64)
        elif self.sample.controller_mode == "rolling_surface_intercept":
            target_arm_q = self.current_target_arm_q if self.current_target_arm_q is not None else self.open_q[:7]
            dt = self.sample.timestep
            # Bound joint speed *and* acceleration (instead of a fixed
            # per-step fractional gain, which converges ~94% of any gap
            # within a single rendered frame here and looked like a
            # teleport). Bounding speed alone still snaps velocity from 0 to
            # max instantly and back to 0 the moment the target is reached.
            # Bounding acceleration too gives every velocity change -- the
            # initial start, the jump when the ball is released, and the
            # final approach -- a smooth ramp instead of a step function.
            # The correction term is a saturated *linear* P term (bounded,
            # constant gain everywhere), not error/dt (deadbeat -- saturates
            # the speed cap for almost any non-tiny error) and not a
            # stopping-distance sqrt(2*a*|error|) law either: sqrt's gain
            # (d/de) is unbounded as error -> 0, and once the ball is tracked
            # continuously (feedforward keeps error small essentially the
            # whole time, not just at final convergence) that singularity
            # sits right where the system spends most of its time, turning
            # any small IK re-solve jitter into a sustained ~0.4-0.5s-period
            # oscillation instead of a settled lock.
            error = target_arm_q - current_arm_q
            correction_qvel = np.clip(
                ROLLING_ARM_CORRECTION_GAIN_S * error,
                -ROLLING_ARM_MAX_JOINT_SPEED_RAD_S,
                ROLLING_ARM_MAX_JOINT_SPEED_RAD_S,
            )
            feedforward_qvel = (
                self.rolling_target_qvel_estimate
                if self.rolling_target_qvel_estimate is not None
                else np.zeros_like(current_arm_q)
            )
            desired_qvel = np.clip(
                feedforward_qvel + correction_qvel,
                -ROLLING_ARM_MAX_JOINT_SPEED_RAD_S,
                ROLLING_ARM_MAX_JOINT_SPEED_RAD_S,
            )
            if self.rolling_arm_qvel is None:
                self.rolling_arm_qvel = np.zeros_like(current_arm_q)
            max_dqvel = ROLLING_ARM_MAX_JOINT_ACCEL_RAD_S2 * dt
            qvel = np.clip(desired_qvel, self.rolling_arm_qvel - max_dqvel, self.rolling_arm_qvel + max_dqvel)
            arm_ranges = self._arm_joint_ranges()
            arm_q = np.clip(current_arm_q + qvel * dt, arm_ranges[:, 0], arm_ranges[:, 1])
            self.rolling_arm_qvel = qvel
            finger_q = self.closed_q[-2:] if self.closed else self.open_q[-2:]
            q_cmd = np.r_[arm_q, finger_q]
        else:
            alpha = smoothstep(t / arrival_time)
            q_cmd = (1.0 - alpha) * self.home_q + alpha * self.open_q
        if self.closed and self.sample.controller_mode not in ("rolling_surface_intercept", "hold_home"):
            q_cmd = self.closed_q
        self.data.qpos[self.ids.qpos_adrs] = q_cmd
        self.data.qvel[self.ids.qvel_adrs] = 0.0

        if self.grasped and self.grasp_offset is not None:
            finger_center = self._fingertip_pad_center()
            self.grasp_offset *= 0.94
            qadr = self.ids.ball_qpos_adr
            vadr = self.ids.ball_qvel_adr
            self.data.qpos[qadr : qadr + 3] = finger_center + self.grasp_offset
            self.data.qvel[vadr : vadr + 6] = 0.0

    def result(self) -> dict:
        result = super().result()
        result["ballistic_intercept_time_s"] = self.ballistic_intercept_time_s
        result["arm_arrival_time_s"] = self.arm_arrival_time_s
        result["controller"] = {
            "type": (
                "scripted_rolling_surface_interception_ik_direct_joint_position"
                if self.sample.controller_mode == "rolling_surface_intercept"
                else "scripted_ballistic_interception_ik_direct_joint_position"
            ),
            "uses_native_gripper_joints": ["finger_joint1", "finger_joint2"],
            "fallback_note": "Robot joints are set directly; after real contact capture, the ball is held at its captured offset.",
            "interception_note": (
                "The arm starts tucked near the robot, extends smoothly to the moving ball, then lifts after grasp."
                if self.sample.controller_mode == "rolling_surface_intercept"
                else "The arm starts from home while the ball waits on the table, then reaches the predicted ballistic intercept point for the launched arc."
            ),
        }
        return result
