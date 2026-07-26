"""Reactive ground-bounce interception controller (subfamily F1_B).

The ball is released at t=0 and falls; the arm waits at home. When the ball
**bounces** off the tabletop, the controller reads the *actual* post-bounce
velocity from the sim, predicts the rebound apex, and drives the arm (direct
joint positions, like the PI's controller) to snatch the ball there, arriving
just before it. Because the intercept is planned from the measured rebound (not
an analytic restitution), the catch is robust to the sim's true bounciness.

Reuses ``franka_catch.controller.MujocoCatchController`` for id resolution, IK,
contact/latch bookkeeping (``after_step``) and the result schema.
"""
from __future__ import annotations

import numpy as np

from franka_catch.controller import (
    REACH_FLOOR_S,
    CatchPlan,
    MujocoCatchController,
    feasible_reach_time,
)
from franka_catch.scene_builder import EpisodeSample
from franka_catch.utils import as_float_list, min_jerk

from .scene_builder import BOUNCE_SURFACE_Z, G

# Horizontal clearance (m) between the catch-ready hand and the ball's vertical
# column. The Franka gripper points up to receive a descending ball, so during the
# fall/rise it must sit BESIDE the column -- a hand hovering in the column would be
# batted by the ball on its way up. After the bounce the hand reaches sideways into
# the descending ball. 0.10 m clears the ~0.025 m ball plus the open pads while
# keeping the reach small enough to complete inside the post-bounce window. The
# ready pose sits at the catch *height* (no vertical hover) so the reach is nearly
# pure lateral -- vertical drop would load the low-acceleration shoulder joint and
# make the dip too slow to arrive before the ball.
SIDE_OFFSET = 0.10
HAND_BALL_OFFSET = 0.075  # hand centre this far below the ball centre at the catch


class BounceCatchController(MujocoCatchController):
    def __init__(self, model, data, sample: EpisodeSample, plan: CatchPlan | None = None):
        super().__init__(model, data, sample, plan)
        self.z_contact = BOUNCE_SURFACE_Z + float(sample.ball_radius)

        # Natural-reach vs secure-catch trade-off (see ``_plan_catch``):
        #   catch_drop   -- how far below the apex to receive the ball. Deeper =>
        #                   a wider reach window (slower, more natural arm) but a
        #                   faster ball at the pads (harder to secure).
        #   close_z_lead -- start closing the pads this far ABOVE the nestle point
        #                   so a fast descending ball is trapped as it enters the
        #                   gap rather than zipping through before the pads shut.
        # Tuned by sweep: drop=0.14 gives an ~0.22 s reach (was ~0.12 s "teleport")
        # while keeping the success-branch catch rate high (~89%); deeper is
        # slower still but the faster ball at the pads costs reliability.
        self.catch_drop = 0.14
        self.close_z_lead = 0.035

        # Nominal (pre-bounce) prediction, used only for timing references/metadata.
        g = abs(float(sample.gravity[2]))
        p0 = np.asarray(sample.ball_initial_position, dtype=np.float64)
        v0 = np.asarray(sample.ball_initial_velocity, dtype=np.float64)
        fall = max(1e-4, p0[2] - self.z_contact)
        self.t_fall_nom = float((-v0[2] + np.sqrt(v0[2] ** 2 + 2.0 * g * fall)) / g)
        v_impact = abs(v0[2]) + g * self.t_fall_nom
        self.v_up_nom = float(sample.ball_restitution) * v_impact
        self.t_up_nom = self.v_up_nom / g
        self.t_reach_nom = self.t_fall_nom + self.t_up_nom

        # Bounce / catch state (filled in reactively).
        self.armed = False               # ball has entered the near-surface band
        self.bounced = False             # post-bounce velocity captured
        self.bounce_time_s: float | None = None
        self.bounce_position: list[float] | None = None
        self.pre_impact_vz: float = 0.0  # most-negative vz seen while armed
        self.post_bounce_velocity: list[float] | None = None
        self.measured_restitution: float | None = None
        self.measured_apex_z: float | None = None
        self.catch_time_s: float | None = None
        self._prev_vz = 0.0

        # Arm targets: home until bounce, then reach the measured apex.
        self.catch_target: np.ndarray | None = None
        self.reach_start_s: float | None = None

        closed_finger = float(np.clip(sample.ball_radius - 0.001 + self.plan.gap_slip, 0.002, 0.04))
        # after_step reads closed_q[7] as the finger half-gap; arm slots unused there.
        self.closed_q = np.r_[self.home_q[:7], [closed_finger, closed_finger]]
        self.open_q = self.home_q.copy()

    # -- setup ---------------------------------------------------------------
    def prepare(self) -> None:
        # Catch-ready pose: hand hovering at ~catch height but offset SIDE_OFFSET
        # back from the ball's vertical column, pads open and up, so it is clear of
        # the falling and rebounding ball. The exact catch is planned reactively on
        # the bounce (``_plan_catch``); this is a nominal ready stance the arm holds
        # until then. Starting here (rather than a far neutral home) is what makes
        # the post-bounce reach a small, Franka-feasible dip instead of a teleport.
        p0 = np.asarray(self.sample.ball_initial_position, dtype=np.float64)
        catch_z_nom = float(np.clip(self.sample.catch_center_z - self.catch_drop,
                                    self.z_contact + 0.20, 1.22))
        # Ready pose at the catch hand-height, offset laterally out of the column.
        ready_target = np.array([p0[0] - SIDE_OFFSET, p0[1], catch_z_nom - HAND_BALL_OFFSET])
        ready_arm_q = self._solve_ik(ready_target)
        ready_arm_q = ready_arm_q + np.asarray(self.plan.start_jitter, dtype=np.float64)
        self.ready_arm_q = ready_arm_q
        self.ready_q = np.r_[ready_arm_q, [0.04, 0.04]]
        # "Hold home" before the bounce == hold this clear side-ready pose.
        self.home_q = self.ready_q.copy()
        self.open_q = self.ready_q.copy()  # overwritten with the catch target in _plan_catch
        closed_finger = float(np.clip(self.sample.ball_radius - 0.001 + self.plan.gap_slip, 0.002, 0.04))
        self.closed_q = np.r_[ready_arm_q, [closed_finger, closed_finger]]
        self.last_cmd = self.ready_q.copy()  # first recorded action == start pose (no frame-1 jump)

        self.data.qpos[self.ids.qpos_adrs] = self.ready_q
        self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity,
                            self.sample.ball_initial_angular_velocity)
        import mujoco
        mujoco.mj_forward(self.model, self.data)

    def _plan_catch(self, t: float, p: np.ndarray, v: np.ndarray) -> None:
        """Plan the post-bounce reach from the measured ball state (p, v)."""
        g = G
        vz = max(0.05, float(v[2]))
        apex_z = float(p[2]) + vz * vz / (2.0 * g)
        t_apex = vz / g
        self.measured_apex_z = apex_z

        # The Franka gripper points UP (finger gap opens upward, palm below), so it
        # can only receive a ball descending into it from above -- a ball rising
        # from below just hits the palm. So we catch on the DESCENT, a margin below
        # the apex, exactly like the proven F1_A vertical-drop capture. The hand
        # waits at home until the ball has apexed (see reach_start below), then
        # reaches down into the descending ball -- it never crosses the ball on the
        # way up, so it can't bat it away.
        # Catch well below the apex (not a shallow 0.12 m nip): a deeper catch
        # makes the ball spend a longer window descending into the open pads,
        # which in turn gives the arm a wide, natural reach window instead of the
        # ~0.12 s "teleport" snap the shallow catch used to force.
        CATCH_DROP = self.catch_drop
        catch_z = float(np.clip(apex_z - CATCH_DROP, self.z_contact + 0.20, 1.22))
        drop_below = max(0.04, apex_z - catch_z)
        t_descent = float(np.sqrt(2.0 * drop_below / g))
        t_catch = t_apex + t_descent                          # descending through catch_z
        catch_xy = p[:2] + v[:2] * t_catch

        target = np.array([catch_xy[0], catch_xy[1], catch_z], dtype=np.float64)
        target[0] += self.plan.lateral_miss[0]
        target[1] += self.plan.lateral_miss[1]
        target += np.asarray(self.plan.wrong_target_offset, dtype=np.float64)
        self.intercept_position = target.copy()
        self.ballistic_intercept = np.array([catch_xy[0], catch_xy[1], catch_z], dtype=np.float64)
        self.catch_target = target

        self.catch_time_s = float(t + t_catch)

        # Solve IK for the reach target (hand 7.5 cm below the ball centre so it
        # nestles between the pads), seeded from the side-ready pose so the reach
        # stays in one joint basin and is small.
        arm_q = self._solve_ik(target + np.array([0.0, 0.0, -0.075]),
                               init=self.ready_arm_q if self.ready_arm_q is not None else None)
        self.arm_target_q = arm_q
        self.open_q = np.r_[arm_q, [0.04, 0.04]]

        # Franka-feasible min-jerk reach side-ready -> catch. Sized so peak joint
        # velocity/acceleration stay under the arm's limits (never shortened below
        # the feasible duration). Timed to arrive just before the ball reaches the
        # catch height. Because the reach only completes near the end (min-jerk
        # starts slow) and arm_arrival is after the apex, the hand sweeps into the
        # column only once the ball is descending -- it never bats the rising ball.
        arrive_margin = 0.09
        ready_arm = self.ready_arm_q if self.ready_arm_q is not None else self.home_q[:7]
        self.reach_dur = float(max(REACH_FLOOR_S, feasible_reach_time(arm_q - ready_arm)))
        arm_arrival = t + t_catch - arrive_margin
        reach_start = max(t, arm_arrival - self.reach_dur)   # hold ready, then dip in
        arm_arrival = max(arm_arrival, reach_start + self.reach_dur)  # ensure a full reach
        self.reach_start_s = float(reach_start)
        self.arm_arrival_time_s = float(arm_arrival)
        self.predicted_close_time_s = float(t + t_catch - self.plan.close_lead)

        # Schedule the post-catch retrieve/reset (the big visible motion). The catch
        # hand target is 7.5 cm below the ball centre (as used for the reach IK above).
        self._plan_return(target + np.array([0.0, 0.0, -0.075]), arm_q,
                          self.arm_arrival_time_s, ready_arm, catch_time_s=self.catch_time_s)

    def _maybe_bounce(self, t: float) -> None:
        """Detect the descent onto the surface and inject the analytic rebound.

        The natural contact is inelastic, so at the impact instant we set the ball
        just above the surface and give it an upward velocity ``e * |v_impact|``
        (tangential velocity lightly damped). This makes the rebound apex land in
        the reachable band by construction and gives an exact restitution label.
        """
        qadr = self.ids.ball_qpos_adr
        vadr = self.ids.ball_qvel_adr
        p = self.data.qpos[qadr:qadr + 3].copy()
        vz = float(self.data.qvel[vadr + 2])
        if vz < 0.0:
            self.pre_impact_vz = min(self.pre_impact_vz, vz)  # deepest downward speed
        if self.bounced:
            return
        # Impact: ball centre has descended to the contact height while falling.
        if p[2] <= self.z_contact + 0.006 and vz < -0.05:
            v = self.data.qvel[vadr:vadr + 3].copy()
            v_pre = max(abs(vz), abs(self.pre_impact_vz), 0.1)
            e = float(self.sample.ball_restitution)
            v_post_z = e * v_pre
            # Lift just clear of the surface and rebound; damp tangential a touch.
            self.data.qpos[qadr + 2] = self.z_contact + 0.002
            self.data.qvel[vadr + 0] = float(v[0]) * 0.92
            self.data.qvel[vadr + 1] = float(v[1]) * 0.92
            self.data.qvel[vadr + 2] = v_post_z
            v_out = self.data.qvel[vadr:vadr + 3].copy()
            self.bounced = True
            self.armed = True
            self.bounce_time_s = float(t)
            self.bounce_position = as_float_list([p[0], p[1], self.z_contact])
            self.post_bounce_velocity = as_float_list(v_out)
            self.pre_impact_vz = -v_pre
            self.measured_restitution = float(v_post_z / v_pre)
            self._plan_catch(t, np.array([p[0], p[1], self.z_contact + 0.002]), v_out)

    # -- per-step ------------------------------------------------------------
    def before_step(self, t: float) -> None:
        if t < self.sample.release_time_s:
            self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity,
                                self.sample.ball_initial_angular_velocity)

        self._maybe_bounce(t)
        q_cmd = self._arm_command(t)

        ball_pos = self.data.qpos[self.ids.ball_qpos_adr:self.ids.ball_qpos_adr + 3].copy()
        wants_gripper = self.plan.branch != "wrong_action"
        ball_vz = float(self.data.qvel[self.ids.ball_qvel_adr + 2])
        # Close only once the ball has actually descended INTO the pad gap (its
        # centre at/near catch_z). Closing earlier -- while the ball is still above
        # the pads -- makes it perch on the closed fingertips instead of nestling
        # between them. A late time-based fallback guarantees the pads do shut.
        should_close = (
            self.bounced and wants_gripper and self.catch_target is not None
            and (
                (ball_vz < 0.0 and ball_pos[2] <= self.catch_target[2] + self.close_z_lead)
                or (self.catch_time_s is not None and t >= self.catch_time_s + 0.06)
            )
        )
        if not self.closed and should_close:
            self.closed = True
            self.close_time_s = float(t)
            self.close_ball_pos = as_float_list(ball_pos)

        if self.closed and wants_gripper:
            q_cmd = np.r_[q_cmd[:7], self.closed_q[7:9]]

        self.data.qpos[self.ids.qpos_adrs] = q_cmd
        self.data.qvel[self.ids.qvel_adrs] = 0.0
        self.last_cmd = np.asarray(q_cmd, dtype=np.float64).copy()
        self.gripper_cmd = 1.0 if (self.closed and wants_gripper) else 0.0

        if self.grasped and self.grasp_offset is not None:
            finger_center = 0.5 * (
                self.data.xpos[self.ids.left_finger_body] + self.data.xpos[self.ids.right_finger_body]
            )
            qadr = self.ids.ball_qpos_adr
            vadr = self.ids.ball_qvel_adr
            self.data.qpos[qadr:qadr + 3] = finger_center + self.grasp_offset
            self.data.qvel[vadr:vadr + 6] = 0.0

    def _arm_command(self, t: float) -> np.ndarray:
        branch, mode = self.plan.branch, self.plan.branch_mode
        # wrong_action: never mount a real catch -- hold the clear side-ready pose.
        if branch == "wrong_action" and mode == "idle":
            return self.ready_q.copy()
        if branch == "wrong_action" and mode == "random":
            if not self.bounced:
                return self.ready_q.copy()
            target = self.ready_q.copy()
            target[:7] = self.ready_arm_q + np.asarray(self.plan.random_joint_delta, dtype=np.float64)
            dur = max(REACH_FLOOR_S, feasible_reach_time(target[:7] - self.ready_arm_q))
            alpha = min_jerk((t - (self.reach_start_s or t)) / max(dur, 1e-3))
            return (1.0 - alpha) * self.ready_q + alpha * target

        # Before the bounce, hold the side-ready pose so the ball falls and rebounds
        # unobstructed.
        if not self.bounced or self.arm_target_q is None:
            return self.ready_q.copy()

        # After the bounce: min-jerk reach from side-ready to the measured intercept
        # over the planned window, then hold (min_jerk clamps at 1). The duration is
        # Franka-feasible, so the recorded joint stream stays within the arm's limits.
        start = self.reach_start_s if self.reach_start_s is not None else t
        dur = self.reach_dur if self.reach_dur is not None else 0.2
        alpha = min_jerk((t - start) / max(dur, 1e-3))
        reach_pose = (1.0 - alpha) * self.ready_q + alpha * self.open_q
        return self._apply_return(t, reach_pose)

    # -- result --------------------------------------------------------------
    def result(self) -> dict:
        res = super().result()
        res.update({
            "subfamily_kind": "ground_bounce_catch",
            "bounce_detected": bool(self.bounced),
            "bounce_time_s": self.bounce_time_s,
            "bounce_position": self.bounce_position,
            "pre_impact_vz": float(self.pre_impact_vz),
            "post_bounce_velocity": self.post_bounce_velocity,
            "nominal_restitution": float(self.sample.ball_restitution),
            "measured_restitution": self.measured_restitution,
            "measured_apex_z": self.measured_apex_z,
            "catch_time_s": self.catch_time_s,
            "surface_z": BOUNCE_SURFACE_Z,
            "nominal_fall_time_s": self.t_fall_nom,
            "nominal_reach_time_s": self.t_reach_nom,
        })
        res["controller"]["type"] = "reactive_ground_bounce_interception_ik_direct_joint_position"
        res["controller"]["interception_note"] = (
            "Arm waits at home; on the ball's ground bounce the controller reads "
            "the measured post-bounce velocity, predicts the rebound apex, and "
            "snatches the ball there."
        )
        return res
