"""Task definitions: scene layout, scripted EE waypoints, grasp events.

All trajectories are deterministic scripted Cartesian waypoints (no grasp
planning). Grasping is a *proxy*: an equality-connect constraint between the
gripper hand body and the nearest cloth vertex body, activated/deactivated at
scripted times, synchronized with gripper open/close commands.

Initial conditions are randomized per seed (cloth size/color/texture, robot
base pose, approach hover heights); the same seed across physics variants
forms a counterfactual bundle with identical scripted actions.

TODO(preview->dataset): replace the equality-connect proxy grasp with real
contact grasping (close fingers on the cloth, rely on friction), and replace
scripted release with gripper opening only.
"""

import colorsys
import zlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Waypoint:
    t: float
    pos: np.ndarray      # (3,) world TCP target
    grip: float          # 0 = closed, 1 = open (normalized command)


@dataclass
class GraspEvent:
    """Proxy grasp: weld cloth vertex nearest `point` to the arm's hand body."""
    arm: str             # arm name
    point: np.ndarray    # (3,) desired grasp point, world
    t_on: float
    t_off: Optional[float]   # None = never released


@dataclass
class ArmSpec:
    name: str            # "arm0" | "left" | "right"
    prefix: str          # MJCF attach prefix, e.g. "A0_"
    base_pos: np.ndarray
    base_yaw: float      # rad, rotation of base about z
    waypoints: List[Waypoint] = field(default_factory=list)


@dataclass
class ClothSpec:
    count: tuple         # (nx, ny) grid vertices
    spacing: tuple       # (sx, sy) m between vertices per axis
    center: np.ndarray   # (3,) world position of cloth center
    rgba: tuple
    # texture randomization ("flat" = solid rgba, no texture asset)
    texture_kind: str = "flat"      # "flat" | "checker" | "gradient"
    rgba2: tuple = (0.5, 0.5, 0.5, 1.0)   # secondary texture color
    texrepeat: float = 4.0

    @property
    def size(self):
        return ((self.count[0] - 1) * self.spacing[0],
                (self.count[1] - 1) * self.spacing[1])

    @property
    def area(self):
        return self.size[0] * self.size[1]


@dataclass
class TaskSpec:
    task_id: str
    subfamily: str
    skill_label: str
    embodiment: str              # "single_franka" | "dual_franka"
    arms: List[ArmSpec]
    cloth: ClothSpec
    grasps: List[GraspEvent]
    workspace_center: np.ndarray  # for camera aiming
    outcome_branch: str = "success"
    branch_note: str = ""
    extra_worldbody_xml: str = ""
    release_time: Optional[float] = None
    fold_line_world: Optional[list] = None   # [[x,y,z],[x,y,z]] two points
    box_pose_world: Optional[dict] = None
    tshirt_proxy: bool = False
    dual_franka_proxy: bool = False          # always False: full dual Franka works
    notes: str = ""


TASK_IDS = [
    "poke_cloth",
    "lift_corner_release",
    "fold_edge_fixed_line",
    "dual_franka_tshirt_fold_box",
]

# Outcome-branch schedule (matches the ball-roll family design):
# 50% success, 20% near-miss, 20% execution failure, 10% wrong action.
# Branch is a function of the SEED (not variant), so all physics variants of
# a counterfactual bundle share the same scripted outcome branch.
BRANCHES = ("success", "near_miss", "execution_failure", "wrong_action")
_BRANCH_PATTERN = ("success", "success", "success", "success", "success",
                   "near_miss", "near_miss",
                   "execution_failure", "execution_failure",
                   "wrong_action")


def branch_for_seed(seed: int) -> str:
    return _BRANCH_PATTERN[seed % len(_BRANCH_PATTERN)]

TABLE_Z = 0.0           # table top surface height
CLOTH_REST_Z = 0.008    # cloth spawn height above table
GRASP_Z = 0.010         # TCP height when "pinching" cloth on the table


def _cloth_corner(cloth: ClothSpec, sx: int, sy: int) -> np.ndarray:
    """World position of cloth corner, sx/sy in {-1,+1} (cloth axis-aligned)."""
    w, h = cloth.size
    return cloth.center + np.array([sx * w / 2, sy * h / 2, 0.0])


def _edge_mid(cloth: ClothSpec, axis: str, sign: int) -> np.ndarray:
    w, h = cloth.size
    if axis == "x":
        return cloth.center + np.array([sign * w / 2, 0.0, 0.0])
    return cloth.center + np.array([0.0, sign * h / 2, 0.0])


def build_task_spec(task_id: str, seed: int, duration: float,
                    cloth_rgba: tuple, branch: str = None) -> TaskSpec:
    """Deterministic task spec. `seed` randomizes initial conditions only, so
    the same seed across physics variants forms a counterfactual bundle with
    identical scripted actions. `cloth_rgba` (material color) is superseded
    by a seed-random color/texture. `branch` (default: from the seed
    schedule) selects the scripted outcome: success or a failure mode."""
    if branch is None:
        branch = branch_for_seed(seed)
    assert branch in BRANCHES, f"unknown branch {branch!r}"
    rng = np.random.default_rng([zlib.crc32(task_id.encode()), seed])
    if task_id == "poke_cloth":
        return _poke_cloth(rng, duration, cloth_rgba, branch)
    if task_id == "lift_corner_release":
        return _lift_corner_release(rng, duration, cloth_rgba, branch)
    if task_id == "fold_edge_fixed_line":
        return _fold_edge_fixed_line(rng, duration, cloth_rgba, branch)
    if task_id == "dual_franka_tshirt_fold_box":
        return _dual_tshirt_fold_box(rng, duration, cloth_rgba, branch)
    raise ValueError(f"unknown task_id {task_id!r}")


# ------------------------------------------------- randomized initial state

BASE_POS_JITTER = 0.03      # m, per-arm base xy jitter
BASE_YAW_JITTER = 0.06      # rad, per-arm base yaw jitter
HOVER_Z_JITTER = 0.03       # m, approach/retreat hover height jitter
APPROACH_XY_JITTER = 0.02   # m, initial approach xy jitter


def _random_cloth_rgba(rng) -> tuple:
    """Saturated-ish random cloth color (avoids marble-countertop grays)."""
    h = rng.uniform(0.0, 1.0)
    s = rng.uniform(0.45, 0.85)
    v = rng.uniform(0.35, 0.90)
    return (*colorsys.hsv_to_rgb(h, s, v), 1.0)


def _random_cloth_appearance(rng) -> dict:
    """Random color + texture fields for a ClothSpec."""
    rgba = _random_cloth_rgba(rng)
    kind = rng.choice(["flat", "checker", "gradient"], p=[0.4, 0.35, 0.25])
    if kind == "checker":
        rgba2 = _random_cloth_rgba(rng)
    else:  # gradient: darkened primary; unused for flat
        rgba2 = (*(0.45 * np.asarray(rgba[:3])), 1.0)
    return {"rgba": rgba, "texture_kind": str(kind),
            "rgba2": tuple(float(v) for v in rgba2),
            "texrepeat": float(rng.uniform(2.0, 8.0))}


def _jitter_base(rng, base_pos, base_yaw):
    pos = np.asarray(base_pos, dtype=float).copy()
    pos[:2] += rng.uniform(-BASE_POS_JITTER, BASE_POS_JITTER, size=2)
    return pos, base_yaw + rng.uniform(-BASE_YAW_JITTER, BASE_YAW_JITTER)


def _jitter_endpoints(rng, wps):
    """Jitter the first/last (non-contact) waypoints: approach xy + hover z.
    Grasp/contact waypoints in between stay exact so the task still succeeds."""
    axy = rng.uniform(-APPROACH_XY_JITTER, APPROACH_XY_JITTER, size=2)
    dz = max(0.0, rng.uniform(-HOVER_Z_JITTER, HOVER_Z_JITTER))
    wps[0] = Waypoint(wps[0].t, wps[0].pos + np.array([axy[0], axy[1], dz]),
                      wps[0].grip)
    wps[-1] = Waypoint(wps[-1].t, wps[-1].pos + np.array([0.0, 0.0, dz]),
                       wps[-1].grip)
    return wps


# ---------------------------------------------------------------- tasks 1-3

def _single_arm(rng, waypoints=None) -> ArmSpec:
    base_pos, base_yaw = _jitter_base(rng, [0.0, 0.0, 0.0], 0.0)
    return ArmSpec(name="arm0", prefix="A0_",
                   base_pos=base_pos, base_yaw=base_yaw,
                   waypoints=waypoints or [])


def _single_cloth(rng, rgba, count=(9, 9)) -> ClothSpec:
    """Randomized single-arm cloth: per-axis spacing (possibly non-square),
    random color/texture, jittered center. `rgba` (material color) is ignored
    in favor of seed-random appearance; kept for signature compatibility."""
    spacing = tuple(rng.uniform(0.028, 0.040, size=2))
    jitter = rng.uniform(-0.02, 0.02, size=2)
    center = np.array([0.55 + jitter[0], 0.0 + jitter[1],
                       TABLE_Z + CLOTH_REST_Z])
    return ClothSpec(count=count, spacing=spacing, center=center,
                     **_random_cloth_appearance(rng))


def _poke_cloth(rng, T, rgba, branch="success") -> TaskSpec:
    cloth = _single_cloth(rng, rgba)
    poke = cloth.center + np.array([rng.uniform(-0.06, 0.06),
                                    rng.uniform(-0.06, 0.06), 0.0])
    poke[2] = TABLE_Z
    branch_note = ""
    if branch == "near_miss":
        # poke just off the cloth edge, onto the countertop beside it
        axis = int(rng.integers(0, 2))
        sign = float(rng.choice([-1.0, 1.0]))
        off = np.zeros(3)
        off[axis] = sign * (cloth.size[axis] / 2 + rng.uniform(0.03, 0.07))
        poke = cloth.center + off
        poke[2] = TABLE_Z
        branch_note = "poke lands just beside the cloth (near miss)"
    elif branch == "execution_failure":
        branch_note = "poke stops short above the cloth (no real contact)"
    elif branch == "wrong_action":
        # poke a clearly wrong spot, back toward the robot base
        ang = rng.uniform(0.75 * np.pi, 1.25 * np.pi)
        r = rng.uniform(0.14, 0.20)
        poke = cloth.center + np.array([r * np.cos(ang), r * np.sin(ang), 0.0])
        poke[2] = TABLE_Z
        branch_note = "poke aimed far from the cloth (wrong action)"
    hover = poke + np.array([0, 0, 0.20])
    deep = poke + np.array([0, 0, 0.012])   # fingertips press cloth into table
    if branch == "execution_failure":
        deep = poke + np.array([0, 0, rng.uniform(0.05, 0.08)])  # stops short
    wps = [
        Waypoint(0.00 * T, hover + np.array([0, 0, 0.05]), 0.0),
        Waypoint(0.25 * T, hover, 0.0),
        Waypoint(0.40 * T, deep, 0.0),
        Waypoint(0.52 * T, deep, 0.0),                       # hold the poke
        Waypoint(0.70 * T, hover, 0.0),
        Waypoint(1.00 * T, hover + np.array([0, 0, 0.05]), 0.0),
    ]
    wps = _jitter_endpoints(rng, wps)
    arm = _single_arm(rng, wps)
    return TaskSpec(
        task_id="poke_cloth", subfamily="poke_cloth",
        skill_label="poke_retract", embodiment="single_franka",
        arms=[arm], cloth=cloth, grasps=[],
        workspace_center=cloth.center.copy(),
        outcome_branch=branch, branch_note=branch_note,
        notes="closed-gripper fingertip poke at a randomized point, then retract",
    )


def _lift_corner_release(rng, T, rgba, branch="success") -> TaskSpec:
    cloth = _single_cloth(rng, rgba)
    # pick a random corner; keep it on the far side (+x half) for reach comfort
    sx, sy = 1, int(rng.choice([-1, 1]))
    corner = _cloth_corner(cloth, sx, sy)
    corner[2] = TABLE_Z
    branch_note = ""
    grasp_effective = True
    t_grasp = 0.32 * T
    t_release = 0.76 * T
    if branch == "near_miss":
        # pinch just outside the corner: gripper closes on air, no grasp
        diag = np.array([sx, sy, 0.0]) / np.sqrt(2.0)
        corner = corner + diag * rng.uniform(0.04, 0.08)
        grasp_effective = False
        branch_note = "pinch closes just beside the cloth corner (near miss)"
    elif branch == "execution_failure":
        # grasp succeeds but slips mid-lift (weld released early)
        t_release = rng.uniform(0.45, 0.56) * T
        branch_note = "grasp slips mid-lift; cloth falls back early"
    hover = corner + np.array([0, 0, 0.15])
    pinch = corner + np.array([0, 0, GRASP_Z])
    lift_to = corner + np.array([-0.10, 0.0, 0.28])  # lift up and slightly inward
    if branch == "wrong_action":
        # carry the grasped corner far sideways/inward instead of lifting in
        # place (angle kept in the -x half-plane so the target stays reachable)
        ang = rng.uniform(0.75 * np.pi, 1.25 * np.pi)
        r = rng.uniform(0.15, 0.22)
        lift_to = corner + np.array([-0.10 + r * np.cos(ang),
                                     r * np.sin(ang), 0.24])
        branch_note = "cloth carried to a wrong location before release"
    wps = [
        Waypoint(0.00 * T, hover + np.array([0, 0, 0.08]), 1.0),
        Waypoint(0.18 * T, hover, 1.0),
        Waypoint(0.30 * T, pinch, 1.0),
        Waypoint(0.34 * T, pinch, 0.0),                      # close gripper
        Waypoint(0.62 * T, lift_to, 0.0),
        Waypoint(0.75 * T, lift_to, 0.0),                    # hold in air
        Waypoint(0.78 * T, lift_to, 1.0),                    # open / release
        Waypoint(1.00 * T, lift_to + np.array([0, 0, 0.06]), 1.0),
    ]
    wps = _jitter_endpoints(rng, wps)
    arm = _single_arm(rng, wps)
    grasps = ([GraspEvent("arm0", corner, t_on=t_grasp, t_off=t_release)]
              if grasp_effective else [])
    return TaskSpec(
        task_id="lift_corner_release", subfamily="lift_corner_release",
        skill_label="grasp_lift_release", embodiment="single_franka",
        arms=[arm], cloth=cloth, grasps=grasps,
        workspace_center=cloth.center.copy(),
        release_time=t_release,
        outcome_branch=branch, branch_note=branch_note,
        notes="proxy grasp (equality connect) on one corner; lift; release mid-air",
    )


def _fold_edge_fixed_line(rng, T, rgba, branch="success") -> TaskSpec:
    cloth = _single_cloth(rng, rgba)
    w, _h = cloth.size
    # fixed fold line: parallel to y, through the cloth center x
    fold_x = cloth.center[0]
    fold_line = [[fold_x, cloth.center[1] - 0.30, TABLE_Z],
                 [fold_x, cloth.center[1] + 0.30, TABLE_Z]]
    # grasp midpoint of the -x edge, carry it over the line to the mirrored x
    grasp_pt = _edge_mid(cloth, "x", -1)
    grasp_pt[2] = TABLE_Z
    branch_note = ""
    grasp_effective = True
    t_grasp = 0.30 * T
    t_release = 0.80 * T
    if branch == "near_miss":
        # pinch just off the -x edge: gripper closes on air, cloth unmoved
        grasp_pt = grasp_pt + np.array([-rng.uniform(0.04, 0.08),
                                        rng.uniform(-0.03, 0.03), 0.0])
        grasp_effective = False
        branch_note = "pinch closes just off the cloth edge (near miss)"
    elif branch == "execution_failure":
        # grasp slips at the apex: fold left incomplete
        t_release = rng.uniform(0.50, 0.62) * T
        branch_note = "grasp slips mid-fold; edge dropped before crossing the line"
    target = grasp_pt.copy()
    target[0] = fold_x + (fold_x - grasp_pt[0]) - 0.01  # mirrored across line
    if branch == "wrong_action":
        # drag the edge AWAY from the fold line instead of across it
        target = grasp_pt + np.array([-rng.uniform(0.10, 0.16),
                                      rng.uniform(-0.05, 0.05), 0.0])
        branch_note = "edge dragged away from the fold line (wrong direction)"
    apex = 0.5 * (grasp_pt + target) + np.array([0, 0, 0.16])
    wps = [
        Waypoint(0.00 * T, grasp_pt + np.array([0, 0, 0.18]), 1.0),
        Waypoint(0.16 * T, grasp_pt + np.array([0, 0, 0.12]), 1.0),
        Waypoint(0.28 * T, grasp_pt + np.array([0, 0, GRASP_Z]), 1.0),
        Waypoint(0.32 * T, grasp_pt + np.array([0, 0, GRASP_Z]), 0.0),  # close
        Waypoint(0.55 * T, apex, 0.0),
        Waypoint(0.76 * T, target + np.array([0, 0, 0.035]), 0.0),
        Waypoint(0.82 * T, target + np.array([0, 0, 0.035]), 1.0),      # open
        Waypoint(1.00 * T, target + np.array([0, 0, 0.16]), 1.0),
    ]
    wps = _jitter_endpoints(rng, wps)
    arm = _single_arm(rng, wps)
    # visual marker for the fold line (non-colliding thin box)
    mid = [(fold_line[0][i] + fold_line[1][i]) / 2 for i in range(3)]
    extra = (f'<geom name="fold_line_marker" type="box" '
             f'size="0.002 0.30 0.0008" pos="{mid[0]} {mid[1]} 0.001" '
             f'rgba="0.9 0.1 0.1 0.8" contype="0" conaffinity="0"/>')
    grasps = ([GraspEvent("arm0", grasp_pt, t_on=t_grasp, t_off=t_release)]
              if grasp_effective else [])
    return TaskSpec(
        task_id="fold_edge_fixed_line", subfamily="fold_edge_fixed_line",
        skill_label="grasp_fold_over_line_release", embodiment="single_franka",
        arms=[arm], cloth=cloth, grasps=grasps,
        workspace_center=cloth.center.copy(),
        release_time=t_release,
        fold_line_world=fold_line,
        extra_worldbody_xml=extra,
        outcome_branch=branch, branch_note=branch_note,
        notes="fold -x edge midpoint over fixed line x=cloth_center_x via arc",
    )


# ------------------------------------------------------------------ task 4

def _dual_tshirt_fold_box(rng, T, rgba, branch="success") -> TaskSpec:
    # T-shirt proxy: rectangular cloth with a distinct color; a true T-shaped
    # flex mesh is left as a TODO (tshirt_proxy=True in metadata).
    spacing = tuple(rng.uniform(0.036, 0.046, size=2))
    center_jitter = rng.uniform(-0.03, 0.03, size=2)
    cloth = ClothSpec(count=(9, 9), spacing=spacing,
                      center=np.array([0.0 + center_jitter[0],
                                       0.44 + center_jitter[1],
                                       TABLE_Z + CLOTH_REST_Z]),
                      **_random_cloth_appearance(rng))
    w, h = cloth.size                       # ~0.29-0.37 per side
    hem_y = cloth.center[1] - h / 2         # near edge ("hem")
    shoulder_y = cloth.center[1] + h / 2    # far edge ("shoulders")

    # shallow box beyond the shoulder edge (position tracks the cloth)
    box_jitter = rng.uniform(-0.03, 0.03, size=2)
    box_center = np.array([cloth.center[0] + box_jitter[0],
                           shoulder_y + 0.25 + box_jitter[1], TABLE_Z])
    box_inner = (0.36, 0.26)                # inner x,y
    box_wall_h = 0.05
    box_wall_t = 0.008

    l_pos, l_yaw = _jitter_base(rng, [-0.52, 0.50, 0.0], 0.0)
    r_pos, r_yaw = _jitter_base(rng, [0.52, 0.50, 0.0], np.pi)
    arms = [
        ArmSpec(name="left", prefix="L_", base_pos=l_pos, base_yaw=l_yaw),
        ArmSpec(name="right", prefix="R_", base_pos=r_pos, base_yaw=r_yaw),
    ]

    branch_note = ""
    t_grasp_off = 0.85 * T
    miss_arm = None
    box_miss = np.zeros(3)
    if branch == "near_miss":
        # one arm's pinch closes just off its hem corner: single-sided carry
        miss_arm = str(rng.choice(["left", "right"]))
        branch_note = f"{miss_arm} gripper misses its hem corner (near miss)"
    elif branch == "execution_failure":
        # both grasps slip mid-transport, before reaching the box
        t_grasp_off = rng.uniform(0.55, 0.68) * T
        branch_note = "both grasps slip mid-transport; cloth dropped short of the box"
    elif branch == "wrong_action":
        # place the cloth beside/beyond the box instead of inside it
        if rng.random() < 0.5:
            box_miss = np.array([float(rng.choice([-1.0, 1.0])) *
                                 rng.uniform(0.18, 0.26), 0.0, 0.0])
        else:
            box_miss = np.array([0.0, rng.uniform(0.20, 0.28), 0.0])
        branch_note = "cloth placed outside the box (wrong placement)"

    grasps = []
    for arm, sx in ((arms[0], -1), (arms[1], 1)):
        hem_corner = np.array([sx * w / 2 * 0.94, hem_y + 0.01, TABLE_Z])
        if arm.name == miss_arm:
            # pinch point pushed outward past the cloth corner: closes on air
            hem_corner = hem_corner + np.array([sx * rng.uniform(0.05, 0.09),
                                                -rng.uniform(0.02, 0.05), 0.0])
        # fold hem over the shoulders, then carry into the box
        over_shoulder = np.array([sx * w / 2 * 0.7, shoulder_y, 0.0])
        in_box = np.array([sx * box_inner[0] / 4, box_center[1] + 0.01, 0.0]) \
            + box_miss
        wps = [
            Waypoint(0.00 * T, hem_corner + np.array([0, 0, 0.22]), 1.0),
            Waypoint(0.16 * T, hem_corner + np.array([0, 0, 0.12]), 1.0),
            Waypoint(0.28 * T, hem_corner + np.array([0, 0, GRASP_Z]), 1.0),
            Waypoint(0.32 * T, hem_corner + np.array([0, 0, GRASP_Z]), 0.0),
            Waypoint(0.50 * T, hem_corner * [1, 0, 0] +
                     np.array([0, cloth.center[1], 0.20]), 0.0),  # lift
            Waypoint(0.66 * T, over_shoulder + np.array([0, 0, 0.16]), 0.0),
            Waypoint(0.82 * T, in_box + np.array([0, 0, 0.11]), 0.0),
            Waypoint(0.86 * T, in_box + np.array([0, 0, 0.11]), 1.0),  # open
            Waypoint(1.00 * T, in_box + np.array([0, 0, 0.24]), 1.0),
        ]
        arm.waypoints = _jitter_endpoints(rng, wps)
        if arm.name != miss_arm:
            grasps.append(GraspEvent(arm.name, hem_corner,
                                     t_on=0.30 * T, t_off=t_grasp_off))

    # shallow open box: bottom + 4 walls
    bx, by = box_inner[0] / 2 + box_wall_t, box_inner[1] / 2 + box_wall_t
    cx, cy = box_center[0], box_center[1]
    extra = f"""
    <body name="shallow_box" pos="{cx} {cy} 0">
      <geom name="box_bottom" type="box" size="{bx} {by} 0.004"
            pos="0 0 0.004" rgba="0.55 0.38 0.22 1"/>
      <geom name="box_wall_xm" type="box" size="{box_wall_t} {by} {box_wall_h/2}"
            pos="{-bx} 0 {box_wall_h/2 + 0.008}" rgba="0.55 0.38 0.22 1"/>
      <geom name="box_wall_xp" type="box" size="{box_wall_t} {by} {box_wall_h/2}"
            pos="{bx} 0 {box_wall_h/2 + 0.008}" rgba="0.55 0.38 0.22 1"/>
      <geom name="box_wall_ym" type="box" size="{bx} {box_wall_t} {box_wall_h/2}"
            pos="0 {-by} {box_wall_h/2 + 0.008}" rgba="0.55 0.38 0.22 1"/>
      <geom name="box_wall_yp" type="box" size="{bx} {box_wall_t} {box_wall_h/2}"
            pos="0 {by} {box_wall_h/2 + 0.008}" rgba="0.55 0.38 0.22 1"/>
    </body>
    <geom name="collar_marker" type="box" size="0.05 0.008 0.0008"
          pos="{cloth.center[0]} {shoulder_y - 0.02} 0.012"
          rgba="0.2 0.2 0.7 0.9" contype="0" conaffinity="0"/>
    <geom name="sleeve_marker_l" type="box" size="0.03 0.015 0.0008"
          pos="{cloth.center[0] - w/2 + 0.03} {shoulder_y - 0.06} 0.012"
          rgba="0.2 0.2 0.7 0.9" contype="0" conaffinity="0"/>
    <geom name="sleeve_marker_r" type="box" size="0.03 0.015 0.0008"
          pos="{cloth.center[0] + w/2 - 0.03} {shoulder_y - 0.06} 0.012"
          rgba="0.2 0.2 0.7 0.9" contype="0" conaffinity="0"/>
    """
    return TaskSpec(
        task_id="dual_franka_tshirt_fold_box",
        subfamily="dual_franka_tshirt_fold_box",
        skill_label="dual_grasp_fold_place_in_box",
        embodiment="dual_franka",
        arms=arms, cloth=cloth, grasps=grasps,
        workspace_center=np.array([cloth.center[0],
                                   0.5 * (cloth.center[1] + box_center[1]),
                                   0.0]),
        extra_worldbody_xml=extra,
        release_time=t_grasp_off,
        outcome_branch=branch, branch_note=branch_note,
        box_pose_world={"pos": box_center.tolist(), "quat": [1, 0, 0, 0],
                        "inner_size_xy": list(box_inner),
                        "wall_height": box_wall_h},
        tshirt_proxy=True,
        dual_franka_proxy=False,
        notes=("full dual Franka via namespaced <attach>; rectangular cloth "
               "with sleeve/collar visual markers as T-shirt proxy "
               "(TODO: true T-shaped flex mesh)"),
    )
