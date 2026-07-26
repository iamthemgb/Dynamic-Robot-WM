from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .scene_builder import EpisodeSample
from .utils import as_float_list, min_jerk


# --- Franka Panda joint limits (used to keep commanded motion executable) ---
# Rated max joint velocity (rad/s) and acceleration (rad/s^2) for the 7 arm
# joints. Commanded trajectories are sized so their peaks stay under these times
# LIMIT_SAFETY, so the recorded actions are replayable on a real Franka.
FRANKA_VMAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=np.float64)
FRANKA_AMAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=np.float64)
LIMIT_SAFETY = 0.8          # only ever use 80% of the rated limits (headroom)
READY_HOVER = 0.045         # m: hand hovers this far above the intercept, pads up
REACH_FLOOR_S = 0.18        # never dip faster than this even if limits would allow

# --- post-catch retrieve / retract (the big, clearly-visible motion) ---------
# The reactive dip-catch itself is physically tiny: a freely-falling ball only
# gives ~0.5 s, and a real Franka cannot make a large fast reach in that window
# (see ``feasible_reach_time``). So after the catch the arm performs a slow
# minimum-jerk LIFT to a raised present/hold pose (up + back toward the base) when
# the ball is secured, or a matching pull-back RESET when it is not -- giving every
# clip a large, unmistakably-robotic, Franka-feasible motion, and (for the lift) a
# genuinely useful secure-and-retrieve training signal.
LIFT_UP = 0.26              # m: hand rises this far while retrieving a caught ball
LIFT_BACK = 0.15           # m: and pulls back toward the base (-x)
RESET_UP = 0.20            # m: on a miss, the hand pulls up ...
RESET_BACK = 0.10          # m: ... and back to a neutral reset pose
DWELL_AFTER_CATCH_S = 0.25 # s: hold at the catch point before lifting (let the grasp latch)
RETURN_FLOOR_S = 0.55      # s: floor on the lift/reset duration so it always reads slow & smooth


def feasible_reach_time(dq) -> float:
    """Shortest min-jerk rest-to-rest duration for a joint move ``dq`` (rad) that
    keeps every joint's peak velocity and acceleration under the Franka limits.

    Inverts the min-jerk peak forms: peak vel = 1.875*dq/T (=> T_v) and peak
    accel = 5.7735*dq/T^2 (=> T_a); the binding duration is the max over joints of
    max(T_v, T_a).
    """
    dq = np.abs(np.asarray(dq, dtype=np.float64))
    if dq.size == 0:
        return 0.0
    t_vel = 1.875 * dq / (LIMIT_SAFETY * FRANKA_VMAX)
    t_acc = np.sqrt(5.7735 * dq / (LIMIT_SAFETY * FRANKA_AMAX))
    return float(np.max(np.maximum(t_vel, t_acc)))


# ---------------------------------------------------------------------------
# Branch taxonomy (what the arm is *told* to do). The measured ``outcome`` is
# read back from the physics after the rollout.
#   success        : reach the predicted ballistic intercept, pinch, hold to end.
#   spatial_near_miss : reach an intercept target offset laterally by a margin so
#                    the ball falls just beside the hand (no contact / grazing).
#   contact_failure: reach the ball's path but fail to secure it (gap too wide,
#                    close too late, or shallow) -> touch then slip/deflect.
#   wrong_action   : arm does the wrong thing (idle / random / wrong-way reach).
BRANCHES = ("success", "spatial_near_miss", "contact_failure", "wrong_action")


@dataclass
class CatchPlan:
    """Per-episode arm behaviour, layered on top of the physical scene sample."""

    branch: str = "success"
    branch_mode: str = ""  # sub-mode (e.g. wrong_action: idle/random/wrong_way)
    # spatial near-miss lateral offset applied to the intercept target (m).
    lateral_miss: tuple[float, float] = (0.0, 0.0)
    # extra finger half-gap for contact failures (m); >0 means pads too wide.
    gap_slip: float = 0.0
    # gripper close timing lead relative to ballistic arrival (s). Negative =
    # close late (ball deflects off the fingers before they shut).
    close_lead: float = 0.020
    # arm arrives this many seconds before the ball reaches the catch height.
    arrive_margin: float = 0.055
    # minimum arm travel duration so the reach is always visibly reactive (s).
    min_travel_s: float = 0.16
    # random wrong-way / random target offset (m) and joint deltas.
    wrong_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    random_joint_delta: tuple = field(default_factory=lambda: (0.0,) * 7)
    disable_latch: bool = False
    # small per-episode perturbation of the catch-ready starting configuration (rad),
    # for arm-pose variety without changing where the hand ends up.
    start_jitter: tuple = field(default_factory=lambda: (0.0,) * 7)


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
    ball_geom: int
    gripper_geoms: set


class MujocoCatchController:
    """Scripted, reactive ballistic interception -- Franka-executable motion.

    The ball is released at ``release_time_s`` (0) from a high drop point and is
    already falling. The arm begins in a **catch-ready pose** (hand hovering
    ``READY_HOVER`` above the predicted intercept, pads open and up) rather than a
    far neutral home, because a real Franka cannot traverse the full home->catch
    reach (~2.4 rad) inside the ~0.5 s fall without exceeding its velocity limit by
    ~4x. From the ready pose the reactive catch is a small dip onto the ball,
    time-scaled by a quintic **minimum-jerk** profile whose duration is chosen so
    peak joint velocity and acceleration stay under the Franka limits (see
    ``feasible_reach_time``). The result is smooth, replayable-on-hardware motion.

    Contact/capture dynamics stay full MuJoCo; a genuine pinch is latched so the
    kinematic gripper can sustain the sphere.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, sample: EpisodeSample, plan: CatchPlan | None = None):
        self.model = model
        self.data = data
        self.sample = sample
        self.plan = plan or CatchPlan()
        self.ids = self._resolve_ids()
        self.home_q = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.78, 0.04, 0.04], dtype=np.float64)

        g = abs(float(sample.gravity[2]))
        p0 = np.asarray(sample.ball_initial_position, dtype=np.float64)
        v0 = np.asarray(sample.ball_initial_velocity, dtype=np.float64)
        z_catch = float(sample.catch_center_z)
        drop = max(1e-4, p0[2] - z_catch)
        # time for the ball to fall to the catch height (positive root).
        self.t_reach = float((v0[2] + np.sqrt(v0[2] ** 2 + 2.0 * g * drop)) / g)
        # predicted ball XY when it reaches the catch height.
        pred_xy = p0[:2] + v0[:2] * self.t_reach
        self.ballistic_intercept = np.array([pred_xy[0], pred_xy[1], z_catch], dtype=np.float64)

        # Where the *hand* should go. Near-miss / wrong-way shift this target.
        target = self.ballistic_intercept.copy()
        target[0] += self.plan.lateral_miss[0]
        target[1] += self.plan.lateral_miss[1]
        target[:3] += np.asarray(self.plan.wrong_target_offset, dtype=np.float64)
        self.intercept_position = target

        self.open_q: np.ndarray | None = None
        self.closed_q: np.ndarray | None = None
        self.ready_q: np.ndarray | None = None       # catch-ready starting config (9,)
        self.ready_arm_q: np.ndarray | None = None    # arm slots of the ready pose (7,)
        self.arm_target_q: np.ndarray | None = None
        self.reach_start_s: float | None = None       # when the dip onto the ball begins
        self.reach_dur: float | None = None           # feasible min-jerk reach duration
        self.arm_arrival_time_s: float | None = None
        # post-catch retrieve/reset (the big visible motion; see _plan_return)
        self.lift_arm_q: np.ndarray | None = None      # raised present/hold pose (grasp)
        self.reset_arm_q: np.ndarray | None = None     # raised reset pose (miss)
        self.lift_start_s: float | None = None         # when the retrieve/reset begins
        self._return_arm_q: np.ndarray | None = None   # latched target once phase B commits
        self._return_dur: float | None = None
        self.closed = False
        self.close_time_s: float | None = None
        self.close_ball_pos: list[float] | None = None
        self.predicted_close_time_s: float | None = None
        self.first_contact_time_s: float | None = None
        self.first_contact_frame: int | None = None
        self.grasp_time_s: float | None = None
        self.exit_time_s: float | None = None
        self.contact_frames = 0
        self.held_steps = 0
        self.max_held_steps = 0
        self.min_dist = float("inf")
        self.success = False
        self.grasped = False
        self.grasp_offset: np.ndarray | None = None
        self.capture_steps = 0
        self._step_idx = 0
        self._frame_count = 0
        self.last_cmd = self.home_q.copy()  # last applied joint command (9,)
        self.gripper_cmd = 0.0  # 0 open, 1 closing/closed

    # -- setup ---------------------------------------------------------------
    def _name2id(self, objtype, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, objtype, name)
        if idx < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return idx

    def _resolve_ids(self) -> JointIds:
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "finger_joint2"]
        joint_ids = np.array([self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names], dtype=np.int32)
        ball_joint = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        hand_body = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "hand")
        left_finger_body = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_finger_body = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        grip_bodies = {hand_body, left_finger_body, right_finger_body}
        gripper_geoms = {g for g in range(self.model.ngeom)
                         if int(self.model.geom_bodyid[g]) in grip_bodies}
        return JointIds(
            joint_names=joint_names,
            qpos_adrs=np.array([self.model.jnt_qposadr[j] for j in joint_ids], dtype=np.int32),
            qvel_adrs=np.array([self.model.jnt_dofadr[j] for j in joint_ids], dtype=np.int32),
            joint_ids=joint_ids,
            ball_qpos_adr=int(self.model.jnt_qposadr[ball_joint]),
            ball_qvel_adr=int(self.model.jnt_dofadr[ball_joint]),
            hand_body=hand_body,
            left_finger_body=left_finger_body,
            right_finger_body=right_finger_body,
            ball_body=self._name2id(mujoco.mjtObj.mjOBJ_BODY, "catch_ball"),
            ball_geom=self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "catch_ball_geom"),
            gripper_geoms=gripper_geoms,
        )

    def prepare(self) -> None:
        # Reactive catch target: hand 7.5 cm below the ball centre so the ball
        # nestles between the finger pads, not at the tips.
        intercept_target = self.intercept_position + np.array([0.0, 0.0, -0.075])

        # Solve the intercept (catch) configuration first, seeded from the neutral
        # home. The catch-ready pose is then the SAME configuration with the hand
        # lifted READY_HOVER, solved by seeding the IK *from the intercept config* so
        # both stay in one joint basin -> the dip is a small, consistent correction
        # (no elbow flips that would blow up the reach and make the arm arrive late).
        arm_q = self._solve_ik(intercept_target)
        ready_target = intercept_target + np.array([0.0, 0.0, READY_HOVER])
        ready_arm_q = self._solve_ik(ready_target, init=arm_q)
        # A small per-episode perturbation of the *starting* pose only (the reach
        # still ends at the exact intercept), kept tiny so it never inflates the dip.
        ready_arm_q = ready_arm_q + np.asarray(self.plan.start_jitter, dtype=np.float64)

        self.arm_target_q = arm_q
        self.ready_arm_q = ready_arm_q
        self.open_q = np.r_[arm_q, [0.04, 0.04]]
        self.ready_q = np.r_[ready_arm_q, [0.04, 0.04]]
        closed_finger = float(np.clip(self.sample.ball_radius - 0.001 + self.plan.gap_slip, 0.002, 0.04))
        self.closed_q = np.r_[arm_q, [closed_finger, closed_finger]]

        # Min-jerk dip ready->intercept. ``reach_dur`` is the Franka-feasible
        # duration and is NEVER shortened below it -- respecting the arm's limits
        # takes priority over arriving on time. When the feasible dip fits inside
        # the fall window the hand holds the ready pose until ``reach_start_s`` and
        # arrives just before the ball; on the rare episode whose dip needs longer,
        # the hand starts at t=0 and simply arrives a touch later (a clean miss)
        # rather than moving faster than a real Franka could.
        window = float(max(REACH_FLOOR_S, self.t_reach - self.plan.arrive_margin))
        self.reach_dur = float(max(REACH_FLOOR_S, feasible_reach_time(arm_q - ready_arm_q)))
        self.reach_start_s = float(max(0.0, window - self.reach_dur))
        self.arm_arrival_time_s = float(self.reach_start_s + self.reach_dur)
        self.predicted_close_time_s = float(
            self.sample.release_time_s + self.t_reach - self.plan.close_lead
        )

        # Schedule the post-catch retrieve/reset (planned statically here for F1_A).
        self._plan_return(intercept_target, arm_q, self.arm_arrival_time_s, ready_arm_q,
                          catch_time_s=self.t_reach)

        # From now on ``home_q`` is the ready pose (idle branches hold it) and the
        # first recorded command matches the initial qpos -- no frame-1 teleport.
        self.home_q = self.ready_q.copy()
        self.last_cmd = self.ready_q.copy()
        self.data.qpos[self.ids.qpos_adrs] = self.ready_q
        self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity,
                            self.sample.ball_initial_angular_velocity)
        mujoco.mj_forward(self.model, self.data)

    def _solve_ik(self, target_hand_pos: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
        arm_adrs = self.ids.qpos_adrs[:7]
        arm_jids = self.ids.joint_ids[:7]
        ranges = self.model.jnt_range[arm_jids].copy()
        for i, jid in enumerate(arm_jids):
            if not self.model.jnt_limited[jid]:
                ranges[i] = [-3.0, 3.0]
        init = self.home_q[:7].copy() if init is None else np.clip(np.asarray(init, dtype=np.float64), ranges[:, 0], ranges[:, 1])
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
            residual, init, bounds=(ranges[:, 0], ranges[:, 1]),
            max_nfev=400, xtol=1e-8, ftol=1e-8, gtol=1e-8,
        )
        return result.x

    def _set_ball_pose(self, pos, vel, avel=(0.0, 0.0, 0.0)) -> None:
        qadr = self.ids.ball_qpos_adr
        vadr = self.ids.ball_qvel_adr
        self.data.qpos[qadr : qadr + 7] = [pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[vadr : vadr + 6] = 0.0
        self.data.qvel[vadr : vadr + 3] = vel  # linear (freejoint qvel is [lin, ang])
        self.data.qvel[vadr + 3 : vadr + 6] = avel  # spin

    # -- per-step ------------------------------------------------------------
    def before_step(self, t: float) -> None:
        assert self.open_q is not None and self.closed_q is not None
        if t < self.sample.release_time_s:
            self._set_ball_pose(self.sample.ball_initial_position, self.sample.ball_initial_velocity,
                                self.sample.ball_initial_angular_velocity)

        q_cmd = self._arm_command(t)

        ball_pos = self.data.qpos[self.ids.ball_qpos_adr : self.ids.ball_qpos_adr + 3].copy()
        wants_gripper = self.plan.branch != "wrong_action" or self.plan.branch_mode == "random_close"
        should_close = (
            wants_gripper
            and t >= self.sample.release_time_s
            and (
                (self.predicted_close_time_s is not None and t >= self.predicted_close_time_s)
                or ball_pos[2] <= self.sample.catch_center_z + 0.06
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
            self.data.qpos[qadr : qadr + 3] = finger_center + self.grasp_offset
            self.data.qvel[vadr : vadr + 6] = 0.0

    def _plan_return(self, catch_hand_target, arm_catch_q, arm_arrival_time_s,
                     ready_seed_q, catch_time_s: float | None = None) -> None:
        """Solve the raised retrieve (grasp) and reset (miss) arm poses and schedule
        the post-catch return. Which one is used is chosen at run time in
        ``_apply_return`` (lift if the ball is secured, reset otherwise). Called from
        ``prepare`` (F1_A, static) or from the reactive catch plan (F1_B, at bounce)."""
        catch_hand_target = np.asarray(catch_hand_target, dtype=np.float64)
        lift_target = catch_hand_target + np.array([-LIFT_BACK, 0.0, LIFT_UP])
        reset_target = catch_hand_target + np.array([-RESET_BACK, 0.0, RESET_UP])
        self.lift_arm_q = self._solve_ik(lift_target, init=arm_catch_q)
        self.reset_arm_q = self._solve_ik(reset_target, init=ready_seed_q)
        start = float(arm_arrival_time_s) + DWELL_AFTER_CATCH_S
        if catch_time_s is not None:
            start = max(start, float(catch_time_s) + 0.10)
        self.lift_start_s = start
        self._return_arm_q = None
        self._return_dur = None

    def _apply_return(self, t: float, pose9: np.ndarray) -> np.ndarray:
        """After ``lift_start_s`` blend the arm from the catch configuration to the
        raised retrieve pose (ball grasped) or the reset pose (missed), on a slow
        Franka-feasible min-jerk profile. The finger slots are left as commanded --
        the close logic in ``before_step`` keeps a caught ball pinched during the
        lift. Idle/random ``wrong_action`` sub-branches never reach here."""
        if self.lift_start_s is None or t < self.lift_start_s or self.arm_target_q is None:
            return pose9
        if self._return_arm_q is None:
            # Hold the catch pose up to 0.15 s longer while the grasp latch resolves,
            # then commit to lift (grasped) or reset (not), measuring the blend from
            # this instant so it always starts smoothly from rest.
            if not (self.grasped or t > self.lift_start_s + 0.15):
                return pose9
            target = self.lift_arm_q if (self.grasped and self.lift_arm_q is not None) else self.reset_arm_q
            if target is None:
                return pose9
            self._return_arm_q = np.asarray(target, dtype=np.float64)
            self._return_dur = float(max(RETURN_FLOOR_S,
                                         feasible_reach_time(self._return_arm_q - self.arm_target_q)))
            self.lift_start_s = float(t)
        beta = min_jerk((t - self.lift_start_s) / max(self._return_dur or RETURN_FLOOR_S, 1e-3))
        arm = (1.0 - beta) * self.arm_target_q + beta * self._return_arm_q
        return np.r_[arm, np.asarray(pose9, dtype=np.float64)[7:9]]

    def _arm_command(self, t: float) -> np.ndarray:
        """Direct joint-position command for the arm at time t (7 arm + 2 finger).

        Every branch drives from the catch-ready pose with a quintic min-jerk
        profile whose duration is Franka-feasible, so the recorded joint stream
        never exceeds the arm's velocity/acceleration limits. After the catch the
        arm lifts/retracts via ``_apply_return`` (the large, clearly-visible motion).
        """
        branch, mode = self.plan.branch, self.plan.branch_mode
        reach_start = self.reach_start_s or 0.0
        if branch == "wrong_action" and mode == "idle":
            return self.ready_q.copy()
        if branch == "wrong_action" and mode == "random":
            target = self.ready_q.copy()
            target[:7] = self.ready_arm_q + np.asarray(self.plan.random_joint_delta, dtype=np.float64)
            dur = max(REACH_FLOOR_S, feasible_reach_time(target[:7] - self.ready_arm_q))
            alpha = min_jerk((t - reach_start) / max(dur, 1e-3))
            return (1.0 - alpha) * self.ready_q + alpha * target
        # success / near-miss / contact-failure / wrong-way: hold the ready pose,
        # then dip onto the (possibly offset) intercept, arriving just before the
        # ball, then lift/retract (the big visible motion, via _apply_return).
        alpha = min_jerk((t - reach_start) / max(self.reach_dur or REACH_FLOOR_S, 1e-3))
        reach_pose = (1.0 - alpha) * self.ready_q + alpha * self.open_q
        return self._apply_return(t, reach_pose)

    def after_step(self, t: float) -> None:
        self._step_idx += 1
        ball_geom = self.ids.ball_geom
        gg = self.ids.gripper_geoms
        contacts_now = 0  # ball <-> gripper contacts only
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == ball_geom and g2 in gg:
                contacts_now += 1
            elif g2 == ball_geom and g1 in gg:
                contacts_now += 1
        if contacts_now:
            self.contact_frames += 1
            if self.first_contact_time_s is None:
                self.first_contact_time_s = float(t)

        ball_pos = self.data.xpos[self.ids.ball_body]
        finger_center = 0.5 * (self.data.xpos[self.ids.left_finger_body] + self.data.xpos[self.ids.right_finger_body])
        dist = float(np.linalg.norm(ball_pos - finger_center))
        self.min_dist = min(self.min_dist, dist)

        pinch = float(self.closed_q[7]) < (self.sample.ball_radius - 0.0005) if self.closed_q is not None else False
        if self.closed and not self.grasped and not self.plan.disable_latch and pinch:
            captured_now = (
                contacts_now > 0
                and np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.04
                and abs(ball_pos[2] - finger_center[2]) <= 0.05
            )
            self.capture_steps = self.capture_steps + 1 if captured_now else 0
            if self.capture_steps * self.sample.timestep >= 0.05:
                self.grasped = True
                self.grasp_offset = (ball_pos - finger_center).copy()
                self.grasp_time_s = float(t)

        retained = np.linalg.norm(ball_pos[:2] - finger_center[:2]) <= 0.05 and abs(ball_pos[2] - finger_center[2]) <= 0.07
        if retained:
            self.held_steps += 1
        else:
            if self.held_steps > 0 and self.first_contact_time_s is not None and self.exit_time_s is None:
                self.exit_time_s = float(t)
            self.held_steps = 0
        self.max_held_steps = max(self.max_held_steps, self.held_steps)
        self.success = self.held_steps * self.sample.timestep >= 0.40

    # -- result --------------------------------------------------------------
    def _time_to_frame(self, t: float | None) -> int | None:
        if t is None:
            return None
        return int(round(t * self.sample.fps))

    def result(self) -> dict:
        return {
            "success": bool(self.success),
            "branch": self.plan.branch,
            "branch_mode": self.plan.branch_mode,
            "ballistic_intercept": as_float_list(self.ballistic_intercept),
            "intercept_position": as_float_list(self.intercept_position),
            "ballistic_intercept_time_s": self.t_reach,
            "arm_arrival_time_s": self.arm_arrival_time_s,
            "reach_start_time_s": self.reach_start_s,
            "reach_duration_s": self.reach_dur,
            "ready_qpos": None if self.ready_q is None else as_float_list(self.ready_q),
            "predicted_close_time_s": self.predicted_close_time_s,
            "gripper_close_time": self.close_time_s,
            "close_ball_position": self.close_ball_pos,
            "first_contact_time_s": self.first_contact_time_s,
            "grasp_time_s": self.grasp_time_s,
            "exit_time_s": self.exit_time_s,
            "contact_frames": int(self.contact_frames),
            "grasped": bool(self.grasped),
            "min_dist": None if self.min_dist == float("inf") else self.min_dist,
            "max_retained_time_s": float(self.max_held_steps * self.sample.timestep),
            "final_ball_position": as_float_list(self.data.xpos[self.ids.ball_body]),
            "final_ball_qpos": as_float_list(self.data.qpos[self.ids.ball_qpos_adr : self.ids.ball_qpos_adr + 7]),
            "controller": {
                "type": "scripted_ballistic_interception_ik_direct_joint_position",
                "uses_native_gripper_joints": ["finger_joint1", "finger_joint2"],
                "uses_native_gravity": True,
                "motion_profile": "quintic_min_jerk",
                "franka_limited": True,
                "limit_safety_factor": LIMIT_SAFETY,
                "interception_note": "The arm begins in a catch-ready pose hovering above the predicted intercept and dips onto the ball with a Franka-limit-respecting minimum-jerk reach as it falls.",
                "fallback_note": "Robot joints are set directly; after a real pinch capture, the ball is held at its captured offset.",
            },
        }
