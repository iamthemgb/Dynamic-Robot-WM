"""Cartesian control of attached Panda arms.

Control mode: resolved-rate (damped-least-squares differential IK) tracking
scripted Cartesian TCP waypoints, feeding the Menagerie Panda's joint
position-servo actuators (actuator1..7). Gripper via actuator8 (0..255).

Orientation is held fixed per episode: gripper pointing straight down with
the yaw the arm had at its home configuration (avoids wrist flips).
"""

from typing import List

import mujoco
import numpy as np

from .scene_builder import PANDA_HOME_QPOS, TCP_OFFSET
from .tasks import Waypoint


def interp_waypoints(waypoints: List[Waypoint], t: float):
    """Smoothstep interpolation between waypoints -> (pos(3,), grip float)."""
    if t <= waypoints[0].t:
        return waypoints[0].pos.copy(), waypoints[0].grip
    if t >= waypoints[-1].t:
        return waypoints[-1].pos.copy(), waypoints[-1].grip
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        if a.t <= t <= b.t:
            u = (t - a.t) / max(b.t - a.t, 1e-9)
            s = 3 * u * u - 2 * u * u * u
            pos = (1 - s) * a.pos + s * b.pos
            grip = (1 - s) * a.grip + s * b.grip
            return pos, grip
    return waypoints[-1].pos.copy(), waypoints[-1].grip


class PandaArm:
    """Wraps one attached Panda (by prefix) with diff-IK Cartesian control."""

    N_ARM = 7

    def __init__(self, model: mujoco.MjModel, prefix: str, name: str):
        self.model = model
        self.prefix = prefix
        self.name = name
        m = model
        self.joint_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint{i+1}")
            for i in range(self.N_ARM)]
        assert all(j >= 0 for j in self.joint_ids), f"missing joints for {prefix}"
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([m.jnt_dofadr[j] for j in self.joint_ids])
        self.jnt_range = m.jnt_range[self.joint_ids].copy()
        self.act_ids = np.array([
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}actuator{i+1}")
            for i in range(self.N_ARM)])
        self.grip_act_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}actuator8")
        self.hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}hand")
        self.finger_qpos_adr = [
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                            f"{prefix}finger_joint{i+1}")]
            for i in range(2)]
        assert self.hand_bid >= 0 and self.grip_act_id >= 0
        self.q_target = PANDA_HOME_QPOS.copy()
        self.grip_cmd = 1.0        # normalized 0=closed, 1=open
        self.target_pos = None
        self.target_quat = None    # fixed downward orientation, set in reset()

    # ------------------------------------------------------------- state

    def reset_home(self, data: mujoco.MjData):
        data.qpos[self.qpos_adr] = PANDA_HOME_QPOS
        for adr in self.finger_qpos_adr:
            data.qpos[adr] = 0.04
        data.ctrl[self.act_ids] = PANDA_HOME_QPOS
        data.ctrl[self.grip_act_id] = 255.0
        self.q_target = PANDA_HOME_QPOS.copy()

    def capture_down_orientation(self, data: mujoco.MjData):
        """Fix target orientation: tool axis straight down, yaw from current pose."""
        R = data.xmat[self.hand_bid].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        cz, sz = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
        Rx = np.diag([1.0, -1.0, -1.0])          # 180 deg about x: z -> down
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (Rz @ Rx).flatten())
        self.target_quat = quat

    def tcp_pose(self, data):
        R = data.xmat[self.hand_bid].reshape(3, 3)
        pos = data.xpos[self.hand_bid] + R @ np.array([0, 0, TCP_OFFSET])
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        return pos, quat

    def tcp_velocity(self, data):
        """(6,) [lin, ang] world velocity of the TCP point."""
        jacp, jacr = self._jacobians(data)
        dq = data.qvel[self.dof_adr]
        return np.concatenate([jacp @ dq, jacr @ dq])

    def gripper_width(self, data) -> float:
        return float(data.qpos[self.finger_qpos_adr[0]] +
                     data.qpos[self.finger_qpos_adr[1]])

    def tau_cmd(self, data) -> np.ndarray:
        return data.actuator_force[self.act_ids].copy()

    def gripper_force_cmd(self, data) -> float:
        return float(data.actuator_force[self.grip_act_id])

    # ----------------------------------------------------------- control

    def _jacobians(self, data):
        pos, _ = self.tcp_pose(data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jacp, jacr, pos, self.hand_bid)
        return jacp[:, self.dof_adr], jacr[:, self.dof_adr]

    def set_targets(self, pos, grip: float):
        self.target_pos = np.asarray(pos, dtype=float)
        self.grip_cmd = float(np.clip(grip, 0.0, 1.0))

    def update(self, data, dt_ctrl: float):
        """One resolved-rate IK step; writes position-servo ctrl targets."""
        pos, quat = self.tcp_pose(data)
        err_p = self.target_pos - pos
        # orientation error as world axis-angle
        dq = np.empty(4)
        conj = np.empty(4)
        mujoco.mju_negQuat(conj, quat)
        mujoco.mju_mulQuat(dq, self.target_quat, conj)
        err_r = np.empty(3)
        mujoco.mju_quat2Vel(err_r, dq, 1.0)

        v = np.clip(6.0 * err_p, -1.2, 1.2)      # m/s
        w = np.clip(6.0 * err_r, -3.0, 3.0)      # rad/s
        jacp, jacr = self._jacobians(data)
        J = np.vstack([jacp, jacr])
        twist = np.concatenate([v, w])
        lam = 0.05
        JJt = J @ J.T + lam * lam * np.eye(6)
        dq_vel = J.T @ np.linalg.solve(JJt, twist)
        dq_vel = np.clip(dq_vel, -2.5, 2.5)

        self.q_target = self.q_target + dq_vel * dt_ctrl
        margin = 0.02
        self.q_target = np.clip(self.q_target,
                                self.jnt_range[:, 0] + margin,
                                self.jnt_range[:, 1] - margin)
        data.ctrl[self.act_ids] = self.q_target
        data.ctrl[self.grip_act_id] = self.grip_cmd * 255.0


class RopeGrasp:
    """Runtime proxy attachment: toggles the pre-declared connect equality
    welding the rope endpoint body to the hand frame.

    On activation the connect's hand-side anchor is set to the endpoint
    body's current position expressed in the hand frame, so the endpoint is
    held at its instantaneous offset from the gripper (a "pinch").
    endpoint_attachment_method = "equality_constraint".

    TODO(preview->dataset): replace with real friction-based pinch grasping.
    """

    def __init__(self, model, arm: PandaArm, grasp_event, eq_name="grasp_endpoint"):
        self.model = model
        self.arm = arm
        self.eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        assert self.eq_id >= 0, f"equality {eq_name} not found"
        self.endpoint_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, grasp_event.endpoint_body)
        assert self.endpoint_bid >= 0, f"body {grasp_event.endpoint_body} not found"
        self.t_on = grasp_event.t_on
        self.t_off = grasp_event.t_off
        self.on = False
        self.off_done = False
        self.anchor_local = None   # endpoint position in hand frame at t_on

    def active(self) -> bool:
        return self.on and not self.off_done

    def update(self, data, t: float, log: list):
        if not self.on and t >= self.t_on:
            epos = data.xpos[self.endpoint_bid].copy()
            hand_pos = data.xpos[self.arm.hand_bid]
            hand_R = data.xmat[self.arm.hand_bid].reshape(3, 3)
            self.anchor_local = hand_R.T @ (epos - hand_pos)
            self.model.eq_data[self.eq_id, 0:3] = 0.0
            self.model.eq_data[self.eq_id, 3:6] = self.anchor_local
            data.eq_active[self.eq_id] = 1
            self.on = True
            log.append({"event": "grasp_on", "t": t,
                        "endpoint_body_id": int(self.endpoint_bid),
                        "endpoint_pos_world": epos.tolist()})
        if self.on and not self.off_done and self.t_off is not None and t >= self.t_off:
            data.eq_active[self.eq_id] = 0
            self.off_done = True
            log.append({"event": "grasp_off", "t": t})

    def anchor_error(self, data) -> float:
        """Distance between the welded hand-frame anchor point and the actual
        endpoint body position (constraint stretch -> slip indicator)."""
        if not self.active() or self.anchor_local is None:
            return 0.0
        hand_pos = data.xpos[self.arm.hand_bid]
        hand_R = data.xmat[self.arm.hand_bid].reshape(3, 3)
        expected = hand_pos + hand_R @ self.anchor_local
        return float(np.linalg.norm(expected - data.xpos[self.endpoint_bid]))
