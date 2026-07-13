"""Soft-body task definitions: object/fixture layout + scripted EE waypoints.

All trajectories are deterministic scripted Cartesian waypoints. Per-seed
layout jitter is drawn from a (task, seed) rng only, so all physics variants
of a counterfactual bundle share object geometry, initial pose, box pose and
the scripted action exactly.

All four tasks use the full Franka (embodiment = "single_franka"); the
gripper squeeze and the box lid are real contact interactions (squeeze_proxy
= false, lid_proxy = false). The lid is a hinged dynamic body driven by a
force-bounded position actuator, so a stiff object can hold it open.
"""

import zlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Waypoint:
    t: float
    pos: np.ndarray      # (3,) world TCP target
    grip: float          # 0 = closed, 1 = fully open (0.08 m); continuous


@dataclass
class ObjectSpec:
    object_id: str
    count: tuple         # flexcomp grid vertex count per axis
    spacing: tuple       # m
    radius: float        # m, collision/visual shell around vertices
    pos: np.ndarray      # (3,) world center at t=0
    dt_scale: float = 1.0   # multiplies material dt_sim (lighter vertices
                            # from finer/smaller grids need smaller steps)

    @property
    def inner_size(self):
        return tuple(s * (c - 1) for s, c in zip(self.spacing, self.count))

    @property
    def outer_size(self):
        return tuple(i + 2 * self.radius for i in self.inner_size)

    @property
    def volume(self) -> float:
        return float(np.prod(self.outer_size))

    @property
    def num_vertices(self) -> int:
        return int(np.prod(self.count))

    @property
    def top_z(self) -> float:
        return float(self.pos[2] + self.inner_size[2] / 2 + self.radius)


@dataclass
class TaskSpec:
    task_id: str
    subfamily: str
    skill_label: str
    waypoints: List[Waypoint]
    obj: ObjectSpec
    workspace_center: np.ndarray
    extra_worldbody_xml: str = ""
    extra_actuator_xml: str = ""
    press_depth: Optional[float] = None
    squeeze_width_target: Optional[float] = None
    squeeze_band: Optional[dict] = None    # pad-footprint band for the local
                                           # squeeze metric {y_half, z_min}
    push_distance: Optional[float] = None
    push_direction_world: Optional[list] = None
    box_pose_world: Optional[dict] = None
    box_opening_pose_world: Optional[dict] = None
    box_interior: Optional[dict] = None    # AABB {x: [lo,hi], y: [...], z: [...]}
    lid: Optional[dict] = None             # {open_angle, angle_trajectory, ...}
    notes: str = ""


TASK_IDS = [
    "press_release",
    "squeeze_with_gripper",
    "push_toward_box_opening",
    "compress_and_close_lid",
]

TABLE_Z = 0.0
DROP_GAP = 0.0015        # m above table at t=0 (settles during preroll)
FLEX_RADIUS = 0.003

# default sponge, outer size ~ [0.098, 0.080, 0.061] m (spec: ~[0.10,0.08,0.06]).
# Vertex spacing (~11 mm) is kept below the Panda fingertip pad width (17 mm):
# with the original 19 mm grid the pads wedge between vertex rows and pass
# through the flex surface instead of denting it.
DEFAULT_COUNT = (9, 8, 6)
DEFAULT_SPACING = (0.0115, 0.0105, 0.011)
# squeeze sponge: pinch-scale object, outer ~ [0.048, 0.039, 0.041] m, short
# axis along world x = the finger-travel axis at the home wrist yaw. Two hard
# constraints found empirically: (a) vertex spacing (~11 mm) must stay below
# the fingertip pad width (17 mm) or the pads wedge between vertex rows and
# slide through the surface with near-zero force; (b) the object cross
# section must be comparable to the pad face or the pads knife local dents
# into a big sponge instead of squeezing the bulk.
SQUEEZE_COUNT = (5, 4, 4)
SQUEEZE_SPACING = (0.0105, 0.011, 0.0115)


def _make_obj(object_id, count, spacing, xy, dt_scale=1.0) -> ObjectSpec:
    half_z = spacing[2] * (count[2] - 1) / 2
    pos = np.array([xy[0], xy[1], TABLE_Z + half_z + FLEX_RADIUS + DROP_GAP])
    return ObjectSpec(object_id, count, spacing, FLEX_RADIUS, pos, dt_scale)


def build_task_spec(task_id: str, seed: int, duration: float) -> TaskSpec:
    """Deterministic per-(task, seed); physics variants share it exactly."""
    rng = np.random.default_rng([zlib.crc32(task_id.encode()), seed])
    if task_id == "press_release":
        return _press(rng, duration)
    if task_id == "squeeze_with_gripper":
        return _squeeze(rng, duration)
    if task_id == "push_toward_box_opening":
        return _push(rng, duration)
    if task_id == "compress_and_close_lid":
        return _compress_lid(rng, duration)
    raise ValueError(f"unknown task_id {task_id!r}")


# ------------------------------------------------------------------ fixtures

def _open_front_box_xml(x_lo, x_hi, y_half, wall_h, t):
    """3-walled box (open front at x_lo, open top) sitting on the table."""
    zc, zh = wall_h / 2, wall_h / 2
    parts = [
        # back wall
        f'<geom name="box_back" type="box" size="{t/2} {y_half + t} {zh}" '
        f'pos="{x_hi + t/2} 0 {zc}" material="box_mat"/>',
        # side walls span the full interior depth plus the back wall
        f'<geom name="box_side_left" type="box" '
        f'size="{(x_hi + t - x_lo)/2} {t/2} {zh}" '
        f'pos="{(x_lo + x_hi + t)/2} {-(y_half + t/2)} {zc}" material="box_mat"/>',
        f'<geom name="box_side_right" type="box" '
        f'size="{(x_hi + t - x_lo)/2} {t/2} {zh}" '
        f'pos="{(x_lo + x_hi + t)/2} {y_half + t/2} {zc}" material="box_mat"/>',
    ]
    return "\n".join(parts)


def _lidded_box_xml(x_lo, x_hi, y_half, wall_h, t, lid_open_angle):
    """4-walled box + hinged lid (hinge along +y at the back top edge)."""
    zc, zh = wall_h / 2, wall_h / 2
    span = (x_hi + t - (x_lo - t)) / 2
    lid_len = x_hi + t - (x_lo - t)          # covers both walls
    parts = [
        f'<geom name="box_back" type="box" size="{t/2} {y_half + t} {zh}" '
        f'pos="{x_hi + t/2} 0 {zc}" material="box_mat"/>',
        f'<geom name="box_front" type="box" size="{t/2} {y_half + t} {zh}" '
        f'pos="{x_lo - t/2} 0 {zc}" material="box_mat"/>',
        f'<geom name="box_side_left" type="box" size="{span} {t/2} {zh}" '
        f'pos="{(x_lo - t + x_hi + t)/2} {-(y_half + t/2)} {zc}" material="box_mat"/>',
        f'<geom name="box_side_right" type="box" size="{span} {t/2} {zh}" '
        f'pos="{(x_lo - t + x_hi + t)/2} {y_half + t/2} {zc}" material="box_mat"/>',
        # lid: hinge at the outer back top edge, panel extends -x when closed
        f'<body name="lid" pos="{x_hi + t} 0 {wall_h}">'
        f'<joint name="lid_hinge" type="hinge" axis="0 1 0" '
        f'range="-0.05 {lid_open_angle + 0.15}" damping="0.8"/>'
        f'<geom name="lid_panel" type="box" '
        f'size="{lid_len/2} {y_half + t + 0.004} 0.004" '
        f'pos="{-lid_len/2} 0 0.005" material="lid_mat" '
        f'friction="0.6 0.005 0.0001"/>'
        f'</body>',
    ]
    actuator = (
        '<actuator><position name="lid_act" joint="lid_hinge" kp="40" '
        f'ctrlrange="-0.1 {lid_open_angle + 0.15}" '
        'forcerange="-10 10"/></actuator>')
    return "\n".join(parts), actuator


# --------------------------------------------------------------------- tasks

def _press(rng, T) -> TaskSpec:
    xy = np.array([0.45, 0.0]) + rng.uniform([-0.015, -0.010], [0.015, 0.010])
    obj = _make_obj("sponge_block_default", DEFAULT_COUNT, DEFAULT_SPACING, xy)
    press_depth = float(0.026 + rng.uniform(-0.004, 0.005))
    top = obj.top_z
    press_z = top - press_depth + 0.001      # TCP ~ fingertip tip
    x, y = xy
    wps = [
        Waypoint(0.00 * T, np.array([x, y, 0.20]), 0.0),
        Waypoint(0.10 * T, np.array([x, y, 0.20]), 0.0),
        Waypoint(0.30 * T, np.array([x, y, top + 0.015]), 0.0),
        Waypoint(0.42 * T, np.array([x, y, press_z]), 0.0),
        Waypoint(0.58 * T, np.array([x, y, press_z]), 0.0),   # hold press
        Waypoint(0.72 * T, np.array([x, y, top + 0.06]), 0.0),
        Waypoint(0.85 * T, np.array([x, y, 0.20]), 0.0),
        Waypoint(1.00 * T, np.array([x, y, 0.20]), 0.0),
    ]
    return TaskSpec(
        task_id="press_release", subfamily="press_release",
        skill_label="press_hold_release",
        waypoints=wps, obj=obj, press_depth=press_depth,
        workspace_center=np.array([x, y, 0.0]),
        notes=("closed-gripper fingertips press the sponge top by "
               f"{press_depth*1000:.0f} mm, hold ~0.8 s, retract; recovery "
               "measured until episode end"),
    )


def _squeeze(rng, T) -> TaskSpec:
    xy = np.array([0.46, 0.0]) + rng.uniform([-0.015, -0.010], [0.015, 0.010])
    obj = _make_obj("sponge_block_squeeze", SQUEEZE_COUNT, SQUEEZE_SPACING, xy)
    width_target = float(0.023 + rng.uniform(-0.003, 0.003))
    g_t = width_target / 0.08                # normalized gripper command
    x, y = xy
    z_sq = 0.024                             # TCP height while squeezing
    wps = [
        Waypoint(0.00 * T, np.array([x, y, 0.20]), 1.0),
        Waypoint(0.10 * T, np.array([x, y, 0.20]), 1.0),
        Waypoint(0.26 * T, np.array([x, y, z_sq + 0.10]), 1.0),
        Waypoint(0.40 * T, np.array([x, y, z_sq]), 1.0),
        Waypoint(0.46 * T, np.array([x, y, z_sq]), 1.0),      # settle
        Waypoint(0.60 * T, np.array([x, y, z_sq]), g_t),      # close/squeeze
        Waypoint(0.72 * T, np.array([x, y, z_sq]), g_t),      # hold
        Waypoint(0.80 * T, np.array([x, y, z_sq]), 1.0),      # open
        Waypoint(0.88 * T, np.array([x, y, z_sq + 0.12]), 1.0),
        Waypoint(1.00 * T, np.array([x, y, 0.22]), 1.0),
    ]
    return TaskSpec(
        task_id="squeeze_with_gripper", subfamily="squeeze_with_gripper",
        skill_label="flank_squeeze_hold_open",
        waypoints=wps, obj=obj, squeeze_width_target=width_target,
        squeeze_band={"y_half": 0.012, "z_min": 0.012},
        workspace_center=np.array([x, y, 0.0]),
        notes=("real finger-pad contact squeeze along world x (finger travel "
               f"axis at home yaw); width target {width_target*1000:.0f} mm "
               f"vs object width {obj.outer_size[0]*1000:.0f} mm"),
    )


def _push(rng, T) -> TaskSpec:
    xy = np.array([0.40, 0.0]) + rng.uniform([-0.010, -0.008], [0.010, 0.008])
    obj = _make_obj("sponge_block_default", DEFAULT_COUNT, DEFAULT_SPACING, xy)
    x_lo, x_hi, y_half, wall_h, t = 0.535, 0.665, 0.055, 0.060, 0.008
    x, y = xy
    z_push = 0.035
    x_start, x_end = 0.325, 0.52
    wps = [
        Waypoint(0.00 * T, np.array([x_start, y, 0.18]), 0.0),
        Waypoint(0.14 * T, np.array([x_start, y, z_push]), 0.0),
        Waypoint(0.22 * T, np.array([x_start, y, z_push]), 0.0),  # settle
        Waypoint(0.66 * T, np.array([x_end, y, z_push]), 0.0),    # push
        Waypoint(0.74 * T, np.array([x_end, y, z_push]), 0.0),    # hold
        Waypoint(0.84 * T, np.array([0.44, y, 0.10]), 0.0),
        Waypoint(1.00 * T, np.array([0.40, y, 0.20]), 0.0),
    ]
    return TaskSpec(
        task_id="push_toward_box_opening", subfamily="push_toward_box_opening",
        skill_label="push_into_box_opening",
        waypoints=wps, obj=obj,
        push_distance=float(x_end - x_start),
        push_direction_world=[1.0, 0.0, 0.0],
        workspace_center=np.array([0.50, 0.0, 0.0]),
        box_pose_world={"pos": [(x_lo + x_hi) / 2, 0.0, 0.0],
                        "quat": [1, 0, 0, 0]},
        box_opening_pose_world={"pos": [x_lo, 0.0, wall_h / 2],
                                "normal": [-1.0, 0.0, 0.0],
                                "width": 2 * y_half, "height": wall_h},
        box_interior={"x": [x_lo, x_hi], "y": [-y_half, y_half],
                      "z": [0.0, wall_h]},
        extra_worldbody_xml=_open_front_box_xml(x_lo, x_hi, y_half, wall_h, t),
        notes=("closed gripper pushes the sponge ~0.2 m along +x toward the "
               "open front of a 3-walled box; entry/jam/slip/rotation "
               "recorded from vertex tracking"),
    )


def _compress_lid(rng, T) -> TaskSpec:
    x_lo, x_hi, y_half, wall_h, t = 0.485, 0.615, 0.060, 0.042, 0.008
    cx = (x_lo + x_hi) / 2
    xy = np.array([cx, 0.0]) + rng.uniform([-0.008, -0.008], [0.008, 0.008])
    obj = _make_obj("sponge_block_default", DEFAULT_COUNT, DEFAULT_SPACING, xy)
    open_angle = 1.85
    box_xml, act_xml = _lidded_box_xml(x_lo, x_hi, y_half, wall_h, t, open_angle)
    x, y = xy
    press_z = 0.040                          # TCP just below the rim plane
    lid_traj = [(0.0, open_angle), (0.60 * T, open_angle),
                (0.86 * T, 0.0), (1.00 * T, 0.0)]
    wps = [
        Waypoint(0.00 * T, np.array([x, y, 0.22]), 0.0),
        Waypoint(0.08 * T, np.array([x, y, 0.22]), 0.0),
        Waypoint(0.24 * T, np.array([x, y, 0.085]), 0.0),
        Waypoint(0.34 * T, np.array([x, y, press_z]), 0.0),   # compress
        Waypoint(0.48 * T, np.array([x, y, press_z]), 0.0),   # hold
        Waypoint(0.56 * T, np.array([x, y, 0.16]), 0.0),      # retract up
        Waypoint(0.68 * T, np.array([0.40, 0.0, 0.24]), 0.0), # clear lid sweep
        Waypoint(1.00 * T, np.array([0.40, 0.0, 0.24]), 0.0),
    ]
    return TaskSpec(
        task_id="compress_and_close_lid", subfamily="compress_and_close_lid",
        skill_label="compress_retract_close_lid",
        waypoints=wps, obj=obj,
        press_depth=float(obj.top_z - press_z),
        workspace_center=np.array([cx, 0.0, 0.0]),
        box_pose_world={"pos": [cx, 0.0, 0.0], "quat": [1, 0, 0, 0]},
        box_opening_pose_world={"pos": [cx, 0.0, wall_h],
                                "normal": [0.0, 0.0, 1.0],
                                "width": 2 * y_half, "depth": x_hi - x_lo},
        box_interior={"x": [x_lo, x_hi], "y": [-y_half, y_half],
                      "z": [0.0, wall_h]},
        lid={"open_angle": open_angle, "angle_trajectory": lid_traj,
             "hinge_pos_world": [x_hi + t, 0.0, wall_h],
             "hinge_axis_world": [0.0, 1.0, 0.0],
             "closed_plane_z": wall_h + 0.001,
             "kp": 40.0, "forcerange": [-10.0, 10.0]},
        extra_worldbody_xml=box_xml,
        extra_actuator_xml=act_xml,
        notes=("sponge sits in a 4-walled box protruding above the rim; "
               "closed gripper compresses it below the rim, retracts, then a "
               "real hinged lid (force-bounded position actuator) closes; a "
               "stiff sponge can hold the lid open"),
    )
