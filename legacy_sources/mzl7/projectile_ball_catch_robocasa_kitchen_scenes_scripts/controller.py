from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .scene_builder import EpisodeSample
from .utils import as_float_list, smoothstep


FINGERTIP_PAD_LOCAL_CENTER = np.array([0.0, 0.0055, 0.0445], dtype=np.float64)


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
        arm_q = self._solve_ik(self.intercept_position + np.array([0.0, 0.0, -0.075]))
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

    def _solve_ik(self, target_hand_pos: np.ndarray) -> np.ndarray:
        arm_adrs = self.ids.qpos_adrs[:7]
        arm_jids = self.ids.joint_ids[:7]
        ranges = self.model.jnt_range[arm_jids].copy()
        for i, jid in enumerate(arm_jids):
            if not self.model.jnt_limited[jid]:
                ranges[i] = [-3.0, 3.0]
        init = self.home_q[:7].copy()
        target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

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

    def _set_ball_pose(self, pos, vel) -> None:
        qadr = self.ids.ball_qpos_adr
        vadr = self.ids.ball_qvel_adr
        self.data.qpos[qadr : qadr + 7] = [pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[vadr : vadr + 6] = 0.0
        # Freejoint qvel layout is [linear(3), angular(3)].
        self.data.qvel[vadr : vadr + 3] = vel

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
            captured_now = (
                contacts_now > 0
                and np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.045
                and abs(ball_pos[2] - finger_center[2]) <= 0.05
            )
            self.capture_steps = self.capture_steps + 1 if captured_now else 0
            if self.sample.enable_grasp_capture and self.capture_steps * self.sample.timestep >= 0.02:
                self.grasped = True
                self.grasp_offset = (ball_pos - finger_center).copy()
                self.grasp_time_s = float(t)

        retained = np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.04 and abs(ball_pos[2] - finger_center[2]) <= 0.05
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

    def prepare(self) -> None:
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

        arm_q = self._solve_ik(self.intercept_position + np.array([0.0, 0.0, -0.075]))
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
        if not self.closed and should_close:
            self.closed = True
            self.close_time_s = float(t)
            self.close_ball_pos = as_float_list(ball_pos)

        arrival_time = self.arm_arrival_time_s if self.arm_arrival_time_s is not None else 0.45
        if self.sample.controller_mode == "hold_home":
            q_cmd = self.home_q
        else:
            alpha = smoothstep(t / arrival_time)
            q_cmd = (1.0 - alpha) * self.home_q + alpha * self.open_q
        if self.closed:
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
            "type": "scripted_ballistic_interception_ik_direct_joint_position",
            "uses_native_gripper_joints": ["finger_joint1", "finger_joint2"],
            "fallback_note": "Robot joints are set directly; after real contact capture, the ball is held at its captured offset.",
            "interception_note": "The arm starts from home while the ball waits on the table, then reaches the predicted ballistic intercept point for the launched arc.",
        }
        return result
