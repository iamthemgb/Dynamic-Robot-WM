from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .scene_builder import EpisodeSample
from .utils import as_float_list, smoothstep


FINGERTIP_PAD_LOCAL_CENTER = np.array([0.0, 0.0055, 0.0445], dtype=np.float64)
ROBOTIQ_OPEN_Q = np.array(
    [
        0.002273890874386094,
        0.0001364909715520716,
        0.0024731211244206548,
        -0.00267025473687781,
        0.002273890874386094,
        0.0001364909715520716,
        0.0024731211244206548,
        -0.00267025473687781,
    ],
    dtype=np.float64,
)
ROBOTIQ_THICK_PAD_CLOSED_Q = np.array(
    [
        0.3314823523217108,
        0.0001046492628712871,
        0.3307642905111326,
        -0.3176014596959107,
        0.3314823523217108,
        0.0001046492628712871,
        0.3307642905111326,
        -0.3176014596959107,
    ],
    dtype=np.float64,
)

# --- Franka Panda arm limits (the Robotiq gripper rides on a Panda) + min-jerk
# retrieve, mirroring franka_catch. The reach is small (the home pose already sits
# near the catch height), so the post-catch LIFT to a raised present/hold pose is
# what gives every clip a large, clearly-visible, hardware-feasible motion.
FRANKA_VMAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=np.float64)
FRANKA_AMAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=np.float64)
LIMIT_SAFETY = 0.8
LIFT_UP = 0.26              # m: hand rises this far retrieving a caught ball
LIFT_BACK = 0.15           # m: and pulls back toward the base (-x)
RESET_UP = 0.20            # m: on a miss, hand pulls up ...
RESET_BACK = 0.10          # m: ... and back to a neutral reset pose
DWELL_AFTER_CATCH_S = 0.25 # s: hold at the catch before lifting (latch the grasp)
RETURN_FLOOR_S = 0.55      # s: floor on the lift/reset duration (always slow & smooth)


def min_jerk(x: float) -> float:
    """Quintic minimum-jerk time-scaling (zero vel & accel at both endpoints)."""
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def feasible_reach_time(dq) -> float:
    """Shortest min-jerk rest-to-rest duration for joint move ``dq`` (rad) keeping
    peak velocity and acceleration under the Panda limits (times LIMIT_SAFETY)."""
    dq = np.abs(np.asarray(dq, dtype=np.float64))
    if dq.size == 0:
        return 0.0
    t_vel = 1.875 * dq / (LIMIT_SAFETY * FRANKA_VMAX)
    t_acc = np.sqrt(5.7735 * dq / (LIMIT_SAFETY * FRANKA_AMAX))
    return float(np.max(np.maximum(t_vel, t_acc)))


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


@dataclass
class RobotiqJointIds:
    joint_names: list[str]
    qpos_adrs: np.ndarray
    qvel_adrs: np.ndarray
    joint_ids: np.ndarray
    ball_qpos_adr: int
    ball_qvel_adr: int
    tool_body: int
    left_pad_site: int
    right_pad_site: int
    grasp_site: int
    left_pad_geoms: tuple[int, ...]
    right_pad_geoms: tuple[int, ...]
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
        drop = max(0.0, float(self.sample.ball_initial_position[2] - close_z))
        self.predicted_close_time_s = self.sample.release_time_s + np.sqrt(2.0 * drop / abs(self.sample.gravity[2])) - 0.025
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
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])

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

    def _time_until_ball_reaches_z(self, target_z: float) -> float | None:
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
        positive_roots = [float(root) for root in roots if root >= -eps]
        if not positive_roots:
            return None
        return max(0.0, min(positive_roots))

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
        should_close = (
            t >= self.sample.release_time_s
            and (
                (self.predicted_close_time_s is not None and t >= self.predicted_close_time_s)
                or ball_pos[2] <= self.sample.catch_center_z + 0.08
            )
        )
        if not self.closed and should_close:
            self.closed = True
            self.close_time_s = float(t)
            self.close_ball_pos = as_float_list(ball_pos)

        alpha = smoothstep(t / 0.45)
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
                and np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.035
                and abs(ball_pos[2] - finger_center[2]) <= 0.04
            )
            self.capture_steps = self.capture_steps + 1 if captured_now else 0
            if self.capture_steps * self.sample.timestep >= 0.05:
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
        time_to_close_z = self._time_until_ball_reaches_z(close_z)
        if time_to_close_z is None:
            time_to_close_z = 0.45

        self.ballistic_intercept_time_s = self.sample.release_time_s + time_to_close_z
        initial_position = np.asarray(self.sample.ball_initial_position, dtype=np.float64)
        initial_velocity = np.asarray(self.sample.ball_initial_velocity, dtype=np.float64)
        gravity = np.asarray(self.sample.gravity, dtype=np.float64)
        predicted_ball_position = initial_position + initial_velocity * time_to_close_z + 0.5 * gravity * time_to_close_z**2
        self.intercept_position = np.array(
            [predicted_ball_position[0], predicted_ball_position[1], self.sample.catch_center_z],
            dtype=np.float64,
        )

        arm_q = self._solve_ik(self.intercept_position + np.array([0.0, 0.0, -0.075]))
        self.open_q = np.r_[arm_q, [0.04, 0.04]]
        closed_finger = float(np.clip(0.85 * self.sample.ball_radius, 0.002, 0.04))
        self.closed_q = np.r_[arm_q, [closed_finger, closed_finger]]
        self.predicted_close_time_s = max(
            self.sample.release_time_s,
            self.ballistic_intercept_time_s - self.close_lead_time_s,
        )
        self.arm_arrival_time_s = max(
            0.12,
            self.predicted_close_time_s - self.reach_lead_time_s,
        )
        self.data.qpos[self.ids.qpos_adrs] = self.home_q
        self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)
        mujoco.mj_forward(self.model, self.data)

    def before_step(self, t: float) -> None:
        assert self.open_q is not None and self.closed_q is not None
        if t < self.sample.release_time_s:
            self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)

        ball_pos = self.data.qpos[self.ids.ball_qpos_adr : self.ids.ball_qpos_adr + 3].copy()
        should_close = (
            t >= self.sample.release_time_s
            and (
                (self.predicted_close_time_s is not None and t >= self.predicted_close_time_s)
                or ball_pos[2] <= self.sample.catch_center_z + 0.08
            )
        )
        if not self.closed and should_close:
            self.closed = True
            self.close_time_s = float(t)
            self.close_ball_pos = as_float_list(ball_pos)

        arrival_time = self.arm_arrival_time_s if self.arm_arrival_time_s is not None else 0.45
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
            "interception_note": "The arm starts from home while the ball is already falling and reaches the predicted ballistic intercept point.",
        }
        return result


class MujocoRobotiqThickPadInterceptionController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sample: EpisodeSample,
        *,
        reach_lead_time_s: float = 0.020,
        close_lead_time_s: float = 0.015,
        target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.model = model
        self.data = data
        self.sample = sample
        self.ids = self._resolve_ids()
        self.home_q = np.r_[
            np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.78], dtype=np.float64),
            ROBOTIQ_OPEN_Q,
        ]
        self.open_q: np.ndarray | None = None
        self.closed_q: np.ndarray | None = None
        # post-catch retrieve/reset (the big visible motion; see _apply_return)
        self.lift_arm_q: np.ndarray | None = None
        self.reset_arm_q: np.ndarray | None = None
        self.lift_start_s: float | None = None
        self._return_arm_q: np.ndarray | None = None
        self._return_dur: float | None = None
        self.reach_start_s: float | None = None
        self.reach_dur: float | None = None
        self.intercept_position = np.array(
            [sample.ball_initial_position[0], sample.ball_initial_position[1], sample.catch_center_z],
            dtype=np.float64,
        )
        self.target_offset = np.asarray(target_offset, dtype=np.float64)
        self.reach_lead_time_s = float(reach_lead_time_s)
        self.close_lead_time_s = float(close_lead_time_s)
        self.closed = False
        self.close_time_s: float | None = None
        self.close_ball_pos: list[float] | None = None
        self.first_contact_time_s: float | None = None
        self.max_held_steps = 0
        self.held_steps = 0
        self.contact_frames = 0
        self.success = False
        self.predicted_close_time_s: float | None = None
        self.ballistic_intercept_time_s: float | None = None
        self.arm_arrival_time_s: float | None = None
        self.grasped = False
        self.grasp_offset: np.ndarray | None = None
        self.capture_steps = 0
        self.grasp_time_s: float | None = None
        self.left_pad_contact_frames = 0
        self.right_pad_contact_frames = 0
        self.two_pad_contact_frames = 0
        self.first_pad_contact_time_s: float | None = None
        self.first_two_pad_contact_time_s: float | None = None
        self.surface_contact_geom_ids: dict[str, set[int]] = {}
        for group in getattr(sample, "surface_contact_groups", ()) or ():
            group_name = str(group.get("name", "surface"))
            geom_ids: set[int] = set()
            for geom_name in group.get("geoms", ()):
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, str(geom_name))
                if geom_id >= 0:
                    geom_ids.add(int(geom_id))
            if geom_ids:
                self.surface_contact_geom_ids[group_name] = geom_ids
        self.surface_contact_frames = {
            name: 0 for name in self.surface_contact_geom_ids
        }
        self.surface_first_contact_time_s = {
            name: None for name in self.surface_contact_geom_ids
        }

    def _name2id(self, objtype, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, objtype, name)
        if idx < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return idx

    def _resolve_ids(self) -> RobotiqJointIds:
        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "rq_right_driver_joint",
            "rq_right_coupler_joint",
            "rq_right_spring_link_joint",
            "rq_right_follower_joint",
            "rq_left_driver_joint",
            "rq_left_coupler_joint",
            "rq_left_spring_link_joint",
            "rq_left_follower_joint",
        ]
        joint_ids = np.array([self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names], dtype=np.int32)
        ball_joint = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        left_pad_geoms = tuple(
            self._name2id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "rq_left_pad1",
                "rq_left_pad2",
                "rq_left_pad_thick_collision_pad",
            )
        )
        right_pad_geoms = tuple(
            self._name2id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "rq_right_pad1",
                "rq_right_pad2",
                "rq_right_pad_thick_collision_pad",
            )
        )
        return RobotiqJointIds(
            joint_names=joint_names,
            qpos_adrs=np.array([self.model.jnt_qposadr[j] for j in joint_ids], dtype=np.int32),
            qvel_adrs=np.array([self.model.jnt_dofadr[j] for j in joint_ids], dtype=np.int32),
            joint_ids=joint_ids,
            ball_qpos_adr=int(self.model.jnt_qposadr[ball_joint]),
            ball_qvel_adr=int(self.model.jnt_dofadr[ball_joint]),
            tool_body=self._name2id(mujoco.mjtObj.mjOBJ_BODY, "rq_base"),
            left_pad_site=self._name2id(mujoco.mjtObj.mjOBJ_SITE, "robotiq_left_pad_site"),
            right_pad_site=self._name2id(mujoco.mjtObj.mjOBJ_SITE, "robotiq_right_pad_site"),
            grasp_site=self._name2id(mujoco.mjtObj.mjOBJ_SITE, "robotiq_grasp_center_site"),
            left_pad_geoms=left_pad_geoms,
            right_pad_geoms=right_pad_geoms,
            ball_body=self._name2id(mujoco.mjtObj.mjOBJ_BODY, "catch_ball"),
        )

    def _set_ball_pose(self, pos, vel) -> None:
        qadr = self.ids.ball_qpos_adr
        vadr = self.ids.ball_qvel_adr
        self.data.qpos[qadr : qadr + 7] = [pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[vadr : vadr + 6] = 0.0
        self.data.qvel[vadr : vadr + 3] = vel

    def _time_until_ball_reaches_z(self, target_z: float) -> float | None:
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
        positive_roots = [float(root) for root in roots if root >= -eps]
        if not positive_roots:
            return None
        return max(0.0, min(positive_roots))

    def _pad_center(self) -> np.ndarray:
        return 0.5 * (
            self.data.site_xpos[self.ids.left_pad_site]
            + self.data.site_xpos[self.ids.right_pad_site]
        )

    def _apply_qpos_command(self, q_cmd: np.ndarray, *, closed: bool) -> None:
        self.data.qpos[self.ids.qpos_adrs] = q_cmd
        self.data.qvel[self.ids.qvel_adrs] = 0.0
        if self.model.nu >= 7:
            self.data.ctrl[:7] = q_cmd[:7]
        if self.model.nu > 7:
            self.data.ctrl[7] = 105.0 if closed else 0.0

    def _solve_ik(self, target_grasp_pos: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
        arm_adrs = self.ids.qpos_adrs[:7]
        arm_jids = self.ids.joint_ids[:7]
        ranges = self.model.jnt_range[arm_jids].copy()
        for i, jid in enumerate(arm_jids):
            if not self.model.jnt_limited[jid]:
                ranges[i] = [-3.0, 3.0]

        init = (self.home_q[:7].copy() if init is None
                else np.clip(np.asarray(init, dtype=np.float64)[:7], ranges[:, 0], ranges[:, 1]))
        self.data.qpos[self.ids.qpos_adrs] = self.home_q
        self.data.qvel[self.ids.qvel_adrs] = 0.0
        mujoco.mj_forward(self.model, self.data)
        target_quat = self.data.xquat[self.ids.tool_body].copy()

        def quat_error(current_quat):
            inv = np.array([current_quat[0], -current_quat[1], -current_quat[2], -current_quat[3]])
            out = np.empty(4)
            mujoco.mju_mulQuat(out, target_quat, inv)
            if out[0] < 0:
                out *= -1
            return out[1:]

        def residual(q):
            self.data.qpos[arm_adrs] = q
            self.data.qpos[self.ids.qpos_adrs[7:]] = ROBOTIQ_OPEN_Q
            self.data.qvel[self.ids.qvel_adrs] = 0.0
            mujoco.mj_forward(self.model, self.data)
            pos_err = self.data.site_xpos[self.ids.grasp_site] - target_grasp_pos
            rot_err = quat_error(self.data.xquat[self.ids.tool_body].copy())
            regularizer = q - init
            return np.r_[3.4 * pos_err, 0.22 * rot_err, 0.025 * regularizer]

        result = least_squares(
            residual,
            init,
            bounds=(ranges[:, 0], ranges[:, 1]),
            max_nfev=550,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        return result.x

    def prepare(self) -> None:
        if (
            getattr(self.sample, "planned_intercept_time_s", None) is not None
            and getattr(self.sample, "planned_intercept_position", None) is not None
        ):
            self.ballistic_intercept_time_s = float(self.sample.planned_intercept_time_s)
            self.intercept_position = (
                np.asarray(self.sample.planned_intercept_position, dtype=np.float64)
                + self.target_offset
            )
        else:
            close_z = self.sample.catch_center_z + 0.055
            time_to_close_z = self._time_until_ball_reaches_z(close_z)
            if time_to_close_z is None:
                time_to_close_z = 0.45

            self.ballistic_intercept_time_s = self.sample.release_time_s + time_to_close_z
            initial_position = np.asarray(self.sample.ball_initial_position, dtype=np.float64)
            initial_velocity = np.asarray(self.sample.ball_initial_velocity, dtype=np.float64)
            gravity = np.asarray(self.sample.gravity, dtype=np.float64)
            predicted_ball_position = initial_position + initial_velocity * time_to_close_z + 0.5 * gravity * time_to_close_z**2
            self.intercept_position = np.array(
                [predicted_ball_position[0], predicted_ball_position[1], self.sample.catch_center_z],
                dtype=np.float64,
            ) + self.target_offset
        arm_q = self._solve_ik(self.intercept_position)
        self.open_q = np.r_[arm_q, ROBOTIQ_OPEN_Q]
        self.closed_q = np.r_[arm_q, ROBOTIQ_THICK_PAD_CLOSED_Q]
        self.predicted_close_time_s = max(
            self.sample.release_time_s,
            self.ballistic_intercept_time_s - self.close_lead_time_s,
        )
        self.arm_arrival_time_s = max(
            0.12,
            self.predicted_close_time_s - self.reach_lead_time_s,
        )
        # Size the reach with the Panda-feasible min-jerk duration so even the
        # larger reaches stay within joint accel limits; hold home, then reach,
        # arriving by arm_arrival (a hair late only on the rare big reach).
        self.reach_dur = float(max(0.12, feasible_reach_time(self.open_q[:7] - self.home_q[:7])))
        self.reach_start_s = float(max(0.0, self.arm_arrival_time_s - self.reach_dur))
        # Post-catch retrieve/reset poses (raised present/hold, or pull-back on a
        # miss), seeded from the catch config so they stay in one joint basin.
        lift_target = self.intercept_position + np.array([-LIFT_BACK, 0.0, LIFT_UP])
        reset_target = self.intercept_position + np.array([-RESET_BACK, 0.0, RESET_UP])
        self.lift_arm_q = self._solve_ik(lift_target, init=arm_q)
        self.reset_arm_q = self._solve_ik(reset_target, init=arm_q)
        self.lift_start_s = max(self.arm_arrival_time_s + DWELL_AFTER_CATCH_S,
                                (self.ballistic_intercept_time_s or 0.0) + 0.10)
        self._return_arm_q = None
        self._return_dur = None
        self._apply_qpos_command(self.home_q, closed=False)
        self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)
        mujoco.mj_forward(self.model, self.data)

    def _apply_return(self, t: float, q_cmd: np.ndarray) -> np.ndarray:
        """After ``lift_start_s`` blend the arm from the catch config to the raised
        retrieve pose (ball grasped) or reset pose (missed), on a slow min-jerk
        profile. Only the 7 arm joints are overwritten; the gripper slots (open or
        closed) pass through, so a caught ball stays pinched during the lift."""
        if self.lift_start_s is None or t < self.lift_start_s or self.open_q is None:
            return q_cmd
        if self._return_arm_q is None:
            if not (self.grasped or t > self.lift_start_s + 0.15):
                return q_cmd
            target = self.lift_arm_q if (self.grasped and self.lift_arm_q is not None) else self.reset_arm_q
            if target is None:
                return q_cmd
            self._return_arm_q = np.asarray(target, dtype=np.float64)
            self._return_dur = float(max(RETURN_FLOOR_S,
                                         feasible_reach_time(self._return_arm_q - self.open_q[:7])))
            self.lift_start_s = float(t)
        beta = min_jerk((t - self.lift_start_s) / max(self._return_dur or RETURN_FLOOR_S, 1e-3))
        out = np.asarray(q_cmd, dtype=np.float64).copy()
        out[:7] = (1.0 - beta) * self.open_q[:7] + beta * self._return_arm_q
        return out

    def before_step(self, t: float) -> None:
        assert self.open_q is not None and self.closed_q is not None
        if t < self.sample.release_time_s:
            self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)

        ball_pos = self.data.qpos[self.ids.ball_qpos_adr : self.ids.ball_qpos_adr + 3].copy()
        should_close = (
            t >= self.sample.release_time_s
            and (
                (self.predicted_close_time_s is not None and t >= self.predicted_close_time_s)
                or ball_pos[2] <= self.sample.catch_center_z + 0.08
            )
        )
        if not self.closed and should_close:
            self.closed = True
            self.close_time_s = float(t)
            self.close_ball_pos = as_float_list(ball_pos)

        # min-jerk reach (zero accel at endpoints), held at home until reach_start
        # then feasibility-sized, so the recorded stream is smooth and hardware-
        # replayable (smoothstep + a fixed window had accel steps on big reaches).
        reach_start = self.reach_start_s if self.reach_start_s is not None else 0.0
        reach_dur = self.reach_dur if self.reach_dur is not None else (self.arm_arrival_time_s or 0.45)
        alpha = min_jerk((t - reach_start) / max(reach_dur, 1e-3))
        q_cmd = (1.0 - alpha) * self.home_q + alpha * self.open_q
        if self.closed:
            # Close only the gripper joints; keep the arm on its smooth min-jerk
            # path (overriding the arm here would teleport it if the pads shut
            # before the reach finished -> a big accel spike).
            q_cmd = np.asarray(q_cmd, dtype=np.float64).copy()
            q_cmd[7:] = self.closed_q[7:]
        # post-catch retrieve/reset: the large, clearly-visible lift/pull-back.
        q_cmd = self._apply_return(t, q_cmd)
        self._apply_qpos_command(q_cmd, closed=self.closed)

        if self.grasped and self.grasp_offset is not None:
            finger_center = self._pad_center()
            self.grasp_offset *= 0.94
            qadr = self.ids.ball_qpos_adr
            vadr = self.ids.ball_qvel_adr
            self.data.qpos[qadr : qadr + 3] = finger_center + self.grasp_offset
            self.data.qvel[vadr : vadr + 6] = 0.0

    def after_step(self, t: float) -> None:
        contacts_now = 0
        left_pad_contacts_now = 0
        right_pad_contacts_now = 0
        surface_contacts_now = {name: False for name in self.surface_contact_geom_ids}
        ball_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "catch_ball_geom")
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 == ball_geom or con.geom2 == ball_geom:
                contacts_now += 1
                other_geom = con.geom2 if con.geom1 == ball_geom else con.geom1
                if other_geom in self.ids.left_pad_geoms:
                    left_pad_contacts_now += 1
                if other_geom in self.ids.right_pad_geoms:
                    right_pad_contacts_now += 1
                for group_name, geom_ids in self.surface_contact_geom_ids.items():
                    if int(other_geom) in geom_ids:
                        surface_contacts_now[group_name] = True
        if contacts_now:
            self.contact_frames += 1
            if self.first_contact_time_s is None:
                self.first_contact_time_s = float(t)
        if left_pad_contacts_now or right_pad_contacts_now:
            if self.first_pad_contact_time_s is None:
                self.first_pad_contact_time_s = float(t)
        if left_pad_contacts_now:
            self.left_pad_contact_frames += 1
        if right_pad_contacts_now:
            self.right_pad_contact_frames += 1
        if left_pad_contacts_now and right_pad_contacts_now:
            self.two_pad_contact_frames += 1
            if self.first_two_pad_contact_time_s is None:
                self.first_two_pad_contact_time_s = float(t)
        for group_name, contacted in surface_contacts_now.items():
            if contacted:
                self.surface_contact_frames[group_name] += 1
                if self.surface_first_contact_time_s[group_name] is None:
                    self.surface_first_contact_time_s[group_name] = float(t)

        ball_pos = self.data.xpos[self.ids.ball_body]
        finger_center = self._pad_center()

        if self.closed and not self.grasped:
            captured_now = (
                left_pad_contacts_now > 0
                and right_pad_contacts_now > 0
                and np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.070
                and abs(ball_pos[2] - finger_center[2]) <= 0.070
            )
            self.capture_steps = self.capture_steps + 1 if captured_now else 0
            if self.capture_steps * self.sample.timestep >= 0.05:
                self.grasped = True
                self.grasp_offset = (ball_pos - finger_center).copy()
                self.grasp_time_s = float(t)

        retained = np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.075 and abs(ball_pos[2] - finger_center[2]) <= 0.075
        if retained:
            self.held_steps += 1
        else:
            self.held_steps = 0
        self.max_held_steps = max(self.max_held_steps, self.held_steps)
        self.success = self.grasped and self.held_steps * self.sample.timestep >= 0.40

    def result(self) -> dict:
        final_ball_position = self.data.xpos[self.ids.ball_body].copy()
        final_fingertip_center = self._pad_center().copy()
        return {
            "success": bool(self.success),
            "intercept_position": as_float_list(self.intercept_position),
            "predicted_close_time_s": self.predicted_close_time_s,
            "gripper_close_time": self.close_time_s,
            "close_ball_position": self.close_ball_pos,
            "first_contact_time_s": self.first_contact_time_s,
            "first_pad_contact_time_s": self.first_pad_contact_time_s,
            "first_two_pad_contact_time_s": self.first_two_pad_contact_time_s,
            "contact_frames": int(self.contact_frames),
            "left_pad_contact_frames": int(self.left_pad_contact_frames),
            "right_pad_contact_frames": int(self.right_pad_contact_frames),
            "two_pad_contact_frames": int(self.two_pad_contact_frames),
            "grasped": bool(self.grasped),
            "grasp_time_s": self.grasp_time_s,
            "max_retained_time_s": float(self.max_held_steps * self.sample.timestep),
            "final_ball_position": as_float_list(final_ball_position),
            "final_fingertip_center_position": as_float_list(final_fingertip_center),
            "final_fingertip_error": float(np.linalg.norm(final_ball_position - final_fingertip_center)),
            "final_ball_qpos": as_float_list(self.data.qpos[self.ids.ball_qpos_adr : self.ids.ball_qpos_adr + 7]),
            "closed_finger_joint_position": float(ROBOTIQ_THICK_PAD_CLOSED_Q[0]),
            "target_offset": as_float_list(self.target_offset),
            "interception_subfamily": getattr(
                self.sample,
                "interception_subfamily",
                "direct_projectile_interception",
            ),
            "expected_contact_sequence": list(getattr(self.sample, "expected_contact_sequence", ()) or ()),
            "surface_contact_frames": {
                name: int(value) for name, value in self.surface_contact_frames.items()
            },
            "surface_first_contact_time_s": {
                name: value for name, value in self.surface_first_contact_time_s.items()
            },
            "ballistic_intercept_time_s": self.ballistic_intercept_time_s,
            "arm_arrival_time_s": self.arm_arrival_time_s,
            "controller": {
                "type": "scripted_ballistic_interception_franka_robotiq_2f85_thick_pad",
                "uses_native_gripper_joints": [
                    "rq_right_driver_joint",
                    "rq_right_coupler_joint",
                    "rq_right_spring_link_joint",
                    "rq_right_follower_joint",
                    "rq_left_driver_joint",
                    "rq_left_coupler_joint",
                    "rq_left_spring_link_joint",
                    "rq_left_follower_joint",
                ],
                "fallback_note": "Robot joints are set directly; after real contact capture, the ball is held at its captured offset.",
                "interception_note": "The Franka arm starts from home while the ball is already falling, using a Robotiq 2F-85 with added thick pad geoms.",
            },
        }
