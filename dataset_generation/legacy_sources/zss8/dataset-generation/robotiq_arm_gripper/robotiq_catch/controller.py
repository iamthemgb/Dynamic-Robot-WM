from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from scripts_mujoco.controller import (
    ROBOTIQ_OPEN_Q,
    ROBOTIQ_THICK_PAD_CLOSED_Q,
    MujocoRobotiqThickPadInterceptionController,
)
from scripts_mujoco.utils import as_float_list, smoothstep


@dataclass
class RobotiqCatchPlan:
    branch: str = "success"
    branch_mode: str = ""
    lateral_miss: tuple[float, float] = (0.0, 0.0)
    target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    close_lead_time_s: float = 0.015
    reach_lead_time_s: float = 0.020
    close_fraction: float = 1.0
    disable_latch: bool = False
    random_joint_delta: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 7)


class PlannedRobotiqCatchController(MujocoRobotiqThickPadInterceptionController):
    """Robotiq thick-pad controller with scripted success/failure branches."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, sample, plan: RobotiqCatchPlan):
        offset = np.asarray(
            (
                plan.lateral_miss[0] + plan.target_offset[0],
                plan.lateral_miss[1] + plan.target_offset[1],
                plan.target_offset[2],
            ),
            dtype=np.float64,
        )
        super().__init__(
            model,
            data,
            sample,
            reach_lead_time_s=plan.reach_lead_time_s,
            close_lead_time_s=plan.close_lead_time_s,
            target_offset=tuple(float(v) for v in offset),
        )
        self.plan = plan
        self.last_cmd = self.home_q.copy()
        self.random_target_q: np.ndarray | None = None
        self.min_dist = float("inf")

    def _apply_qpos_command(self, q_cmd: np.ndarray, *, closed: bool) -> None:
        self.last_cmd = np.asarray(q_cmd, dtype=np.float64).copy()
        super()._apply_qpos_command(q_cmd, closed=closed)

    def prepare(self) -> None:
        super().prepare()
        if self.closed_q is not None and self.plan.close_fraction < 0.999:
            frac = float(np.clip(self.plan.close_fraction, 0.0, 1.0))
            self.closed_q[7:] = ROBOTIQ_OPEN_Q + frac * (ROBOTIQ_THICK_PAD_CLOSED_Q - ROBOTIQ_OPEN_Q)
        if self.plan.branch == "wrong_action" and self.plan.branch_mode == "random":
            target = self.home_q.copy()
            delta = np.asarray(self.plan.random_joint_delta, dtype=np.float64)
            target[:7] = target[:7] + delta[:7]
            target[7:] = ROBOTIQ_OPEN_Q
            self.random_target_q = target
        self.last_cmd = self.home_q.copy()

    def before_step(self, t: float) -> None:
        if self.plan.branch == "wrong_action" and self.plan.branch_mode in {"idle", "random"}:
            if t < self.sample.release_time_s:
                self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity)
            if self.plan.branch_mode == "idle" or self.random_target_q is None:
                q_cmd = self.home_q.copy()
            else:
                arrive = max(0.35, self.arm_arrival_time_s or 0.35)
                alpha = smoothstep(t / arrive)
                q_cmd = (1.0 - alpha) * self.home_q + alpha * self.random_target_q
            self._apply_qpos_command(q_cmd, closed=False)
            return
        super().before_step(t)

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
        self.min_dist = min(self.min_dist, float(np.linalg.norm(ball_pos - finger_center)))

        if self.closed and not self.grasped and not self.plan.disable_latch:
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

        retained = (
            np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.075
            and abs(ball_pos[2] - finger_center[2]) <= 0.075
        )
        if retained:
            self.held_steps += 1
        else:
            self.held_steps = 0
        self.max_held_steps = max(self.max_held_steps, self.held_steps)
        self.success = self.grasped and self.held_steps * self.sample.timestep >= 0.40

    def result(self) -> dict:
        result = super().result()
        result.update(
            {
                "branch": self.plan.branch,
                "branch_mode": self.plan.branch_mode,
                "min_dist": None if self.min_dist == float("inf") else float(self.min_dist),
                "final_grasp_center_position": as_float_list(self._pad_center()),
                "planned_disable_latch": bool(self.plan.disable_latch),
                "planned_close_fraction": float(self.plan.close_fraction),
            }
        )
        return result

