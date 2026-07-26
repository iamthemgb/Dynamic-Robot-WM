"""Rope task definitions: scene layout, scripted EE waypoints, grasp events.

All trajectories are deterministic scripted Cartesian waypoints (no grasp
planning). The rope endpoint is attached to the gripper by an equality
connect constraint toggled at scripted times (endpoint_attachment_method =
"equality_constraint"), synchronized with gripper open/close commands.

TODO(preview->dataset): replace the equality-connect attachment with real
friction-based pinch grasping of the rope.
"""

import zlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Waypoint:
    t: float
    pos: np.ndarray      # (3,) world TCP target
    grip: float          # 0 = closed, 1 = open


@dataclass
class GraspEvent:
    endpoint_body: str   # rope body name to weld ("RB_first")
    t_on: float
    t_off: Optional[float]


@dataclass
class RopeLayout:
    offset: np.ndarray   # (3,) world position of the grasped end (RB_first);
                         # rope extends along +x from here


@dataclass
class TaskSpec:
    task_id: str
    subfamily: str
    skill_label: str
    waypoints: List[Waypoint]
    rope: RopeLayout
    grasp: GraspEvent
    workspace_center: np.ndarray
    extra_worldbody_xml: str = ""
    release_time: Optional[float] = None
    post_pose_world: Optional[dict] = None       # wrap task
    ring_pose_world: Optional[dict] = None       # thread task
    ring_radius: Optional[float] = None          # inner radius
    ring_normal_world: Optional[list] = None
    threading_proxy: bool = False
    grasp_body_offset_m: float = 0.0     # arc length from rope end to grasp body
    bar_pose_world: Optional[dict] = None        # drape task horizontal bar
    coil_center_world: Optional[list] = None     # coil task spiral center
    twirl_center_world: Optional[list] = None    # twirl task circle center
    notes: str = ""


TASK_IDS = [
    # --- original four (spec) ---
    "tug_endpoint",
    "drag_endpoint",
    "wrap_around_post",
    "thread_through_ring",
    # --- creative dynamics-rich additions ---
    "shake_wave",          # oscillate endpoint -> traveling waves down the rope
    "lift_and_drape",      # carry endpoint up and over a bar so the rope drapes
    "coil_on_table",       # spiral the endpoint inward to coil the rope
    "twirl_overhead",      # lift endpoint and move it in a circle -> rope trails
    "sweep_aside",         # non-prehensile: push the rope sideways (no grasp)
]

TABLE_Z = 0.0
ROPE_Z = None            # set from material radius at scene build
GRASP_Z = 0.012          # TCP height when pinching the endpoint on the table

POST_RADIUS = 0.025
POST_HEIGHT = 0.22
RING_INNER_RADIUS = 0.05
RING_TUBE_RADIUS = 0.008
BAR_RADIUS = 0.018
BAR_HEIGHT = 0.24          # world z of the horizontal drape bar axis
BAR_HALF_SPAN = 0.20      # bar extends +/- this in y


def build_task_spec(task_id: str, seed: int, duration: float) -> TaskSpec:
    """Deterministic per-(task, seed); physics variants share it exactly."""
    rng = np.random.default_rng([zlib.crc32(task_id.encode()), seed])
    if task_id == "tug_endpoint":
        return _tug(rng, duration)
    if task_id == "drag_endpoint":
        return _drag(rng, duration)
    if task_id == "wrap_around_post":
        return _wrap(rng, duration)
    if task_id == "thread_through_ring":
        return _thread(rng, duration)
    if task_id == "shake_wave":
        return _shake_wave(rng, duration)
    if task_id == "lift_and_drape":
        return _lift_and_drape(rng, duration)
    if task_id == "coil_on_table":
        return _coil_on_table(rng, duration)
    if task_id == "twirl_overhead":
        return _twirl_overhead(rng, duration)
    if task_id == "sweep_aside":
        return _sweep_aside(rng, duration)
    raise ValueError(f"unknown task_id {task_id!r}")


def _hover(p, dz):
    return np.asarray(p) + np.array([0.0, 0.0, dz])


def _tug(rng, T) -> TaskSpec:
    end = np.array([0.42 + rng.uniform(-0.03, 0.03),
                    -0.08 + rng.uniform(-0.04, 0.04), TABLE_Z])
    lift = end + np.array([-0.06, 0.02, 0.30])       # up and slightly inward
    t_grasp, t_release = 0.30 * T, 0.72 * T
    wps = [
        Waypoint(0.00 * T, _hover(end, 0.20), 1.0),
        Waypoint(0.16 * T, _hover(end, 0.10), 1.0),
        Waypoint(0.28 * T, _hover(end, GRASP_Z), 1.0),
        Waypoint(0.32 * T, _hover(end, GRASP_Z), 0.0),   # close
        Waypoint(0.58 * T, lift, 0.0),                    # tug up
        Waypoint(0.70 * T, lift, 0.0),                    # hold
        Waypoint(0.74 * T, lift, 1.0),                    # release
        Waypoint(1.00 * T, _hover(lift, 0.06), 1.0),
    ]
    return TaskSpec(
        task_id="tug_endpoint", subfamily="tug_endpoint",
        skill_label="grasp_tug_up_release",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        grasp=GraspEvent("RB_first", t_on=t_grasp, t_off=t_release),
        workspace_center=end + np.array([0.12, 0.0, 0.0]),
        release_time=t_release,
        notes="tug one endpoint upward ~0.3 m, hold, release, rope settles",
    )


def _drag(rng, T) -> TaskSpec:
    end = np.array([0.36 + rng.uniform(-0.02, 0.02),
                    -0.20 + rng.uniform(-0.03, 0.03), TABLE_Z])
    # curved drag path: quarter-ish arc ending on the +y side
    via = np.array([0.30, 0.02, TABLE_Z])
    goal = np.array([0.46 + rng.uniform(-0.02, 0.02), 0.24, TABLE_Z])
    drag_z = 0.030
    t_grasp, t_release = 0.28 * T, 0.84 * T
    wps = [
        Waypoint(0.00 * T, _hover(end, 0.18), 1.0),
        Waypoint(0.14 * T, _hover(end, 0.09), 1.0),
        Waypoint(0.26 * T, _hover(end, GRASP_Z), 1.0),
        Waypoint(0.30 * T, _hover(end, GRASP_Z), 0.0),   # close
        Waypoint(0.38 * T, _hover(end, drag_z), 0.0),
        Waypoint(0.60 * T, _hover(via, drag_z), 0.0),    # curved drag
        Waypoint(0.82 * T, _hover(goal, drag_z), 0.0),
        Waypoint(0.86 * T, _hover(goal, drag_z), 1.0),   # open
        Waypoint(1.00 * T, _hover(goal, 0.16), 1.0),
    ]
    return TaskSpec(
        task_id="drag_endpoint", subfamily="drag_endpoint",
        skill_label="grasp_drag_curved_release",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        grasp=GraspEvent("RB_first", t_on=t_grasp, t_off=t_release),
        workspace_center=np.array([0.42, 0.0, 0.0]),
        release_time=t_release,
        notes="drag one endpoint along a curved low path; rope trails on table",
    )


def _wrap(rng, T) -> TaskSpec:
    # Rope passes just south of the post so the wrap consumes little rope;
    # the sweep is ~210 deg (spec: 180-360) to keep the trailing rope slack
    # (a 320-deg orbit demands more rope than exists -> taut -> arm snag).
    post = np.array([0.52, 0.10, TABLE_Z])
    end = np.array([0.36 + rng.uniform(-0.015, 0.015),
                    0.02 + rng.uniform(-0.015, 0.015), TABLE_Z])
    orbit_r = 0.12
    orbit_z = 0.05
    a0 = np.arctan2(end[1] - post[1], end[0] - post[0])   # start azimuth (~-153deg)
    sweep = np.deg2rad(210.0)                             # ~0.58 turns
    t_grasp, t_release = 0.24 * T, 0.92 * T

    wps = [
        Waypoint(0.00 * T, _hover(end, 0.16), 1.0),
        Waypoint(0.10 * T, _hover(end, 0.08), 1.0),
        Waypoint(0.20 * T, _hover(end, GRASP_Z), 1.0),
        Waypoint(0.25 * T, _hover(end, GRASP_Z), 0.0),   # close
    ]
    n_arc = 8
    for i in range(n_arc + 1):
        a = a0 + sweep * i / n_arc
        p = post + np.array([orbit_r * np.cos(a), orbit_r * np.sin(a), orbit_z])
        wps.append(Waypoint((0.32 + 0.58 * i / n_arc) * T, p, 0.0))
    last = wps[-1].pos
    wps += [
        Waypoint(0.93 * T, last, 1.0),                   # open / release
        Waypoint(1.00 * T, _hover(last, 0.14), 1.0),
    ]
    return TaskSpec(
        task_id="wrap_around_post", subfamily="wrap_around_post",
        skill_label="grasp_orbit_post_release",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        grasp=GraspEvent("RB_first", t_on=t_grasp, t_off=t_release),
        workspace_center=post + np.array([0.0, 0.0, 0.0]),
        release_time=t_release,
        post_pose_world={"pos": post.tolist(), "quat": [1, 0, 0, 0],
                         "radius": POST_RADIUS, "height": POST_HEIGHT},
        extra_worldbody_xml=(
            f'<geom name="post" type="cylinder" size="{POST_RADIUS} {POST_HEIGHT/2}" '
            f'pos="{post[0]} {post[1]} {POST_HEIGHT/2}" rgba="0.5 0.5 0.55 1" '
            f'friction="__POST_FRICTION__ 0.005 0.0001"/>'),
        notes=f"orbit grasped endpoint {np.rad2deg(sweep):.0f} deg around post at r={orbit_r}",
    )


def _thread(rng, T) -> TaskSpec:
    # Horizontal eyelet ring (normal +z): the rope is grasped ~9 cm from its
    # end so a free tail dangles below the gripper and is lowered through the
    # ring opening. (A vertical ring is infeasible with an endpoint weld: the
    # hand itself would have to pass through the ring.)
    ring_c = np.array([0.52 + rng.uniform(-0.015, 0.015),
                       0.10 + rng.uniform(-0.015, 0.015), 0.145])
    ring_normal = np.array([0.0, 0.0, 1.0])
    grasp_offset = 0.092                                  # m of free tail
    end = np.array([0.38 + rng.uniform(-0.015, 0.015), -0.14, TABLE_Z])
    grasp_pt = end + np.array([grasp_offset, 0.0, 0.0])   # rope extends +x
    above_ring = np.array([ring_c[0], ring_c[1], 0.32])
    lowered = np.array([ring_c[0], ring_c[1], 0.185])     # tail tip ~0.06 below ring
    t_grasp, t_release = 0.24 * T, 0.92 * T
    wps = [
        Waypoint(0.00 * T, _hover(grasp_pt, 0.16), 1.0),
        Waypoint(0.10 * T, _hover(grasp_pt, 0.08), 1.0),
        Waypoint(0.20 * T, _hover(grasp_pt, GRASP_Z), 1.0),
        Waypoint(0.25 * T, _hover(grasp_pt, GRASP_Z), 0.0),  # close
        Waypoint(0.42 * T, _hover(grasp_pt, 0.30), 0.0),     # lift, tail dangles
        Waypoint(0.58 * T, above_ring, 0.0),                 # over ring center
        Waypoint(0.68 * T, above_ring, 0.0),                 # hold: damp swing
        Waypoint(0.86 * T, lowered, 0.0),                    # lower tail thru ring
        Waypoint(0.93 * T, lowered, 1.0),                    # open / release
        Waypoint(1.00 * T, _hover(lowered, 0.10), 1.0),
    ]
    # ring: circle of capsules in the x-y plane (normal +z), plus a stand
    segs = []
    n_ring = 14
    R = RING_INNER_RADIUS + RING_TUBE_RADIUS
    for i in range(n_ring):
        a0, a1 = 2 * np.pi * i / n_ring, 2 * np.pi * (i + 1) / n_ring
        p0 = ring_c + np.array([R * np.cos(a0), R * np.sin(a0), 0.0])
        p1 = ring_c + np.array([R * np.cos(a1), R * np.sin(a1), 0.0])
        segs.append(
            f'<geom name="ringG{i}" type="capsule" size="{RING_TUBE_RADIUS}" '
            f'fromto="{p0[0]:.4f} {p0[1]:.4f} {p0[2]:.4f} '
            f'{p1[0]:.4f} {p1[1]:.4f} {p1[2]:.4f}" rgba="0.75 0.6 0.2 1" '
            f'friction="__RING_FRICTION__ 0.005 0.0001"/>')
    segs.append(
        f'<geom name="ring_stand" type="cylinder" size="0.008 {ring_c[2]/2:.4f}" '
        f'pos="{ring_c[0] + R:.4f} {ring_c[1]:.4f} {ring_c[2]/2:.4f}" '
        f'rgba="0.4 0.4 0.45 1"/>')
    return TaskSpec(
        task_id="thread_through_ring", subfamily="thread_through_ring",
        skill_label="grasp_thread_ring_release",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        grasp=GraspEvent("RB_6", t_on=t_grasp, t_off=t_release),
        grasp_body_offset_m=grasp_offset,
        workspace_center=ring_c * [1, 1, 0],
        release_time=t_release,
        ring_pose_world={"pos": ring_c.tolist(), "quat": [1, 0, 0, 0]},
        ring_radius=RING_INNER_RADIUS,
        ring_normal_world=ring_normal.tolist(),
        threading_proxy=False,   # real contact threading; the rope weld is the
                                 # only proxy (see endpoint_attachment_method)
        extra_worldbody_xml="\n".join(segs),
        notes=("lower a dangling ~9 cm rope tail down through a horizontal "
               "eyelet ring (inner r=0.05, normal +z); real ring contacts; "
               "rope grasped at body RB_6, not at the endpoint"),
    )


# ----------------------------------------------------------- creative tasks

def _shake_wave(rng, T) -> TaskSpec:
    """Grasp the endpoint, lift a little, then oscillate the hand laterally to
    send visible traveling waves down the rope. Rich transverse dynamics."""
    end = np.array([0.42 + rng.uniform(-0.03, 0.03),
                    -0.06 + rng.uniform(-0.04, 0.04), TABLE_Z])
    hold_z = 0.18 + rng.uniform(-0.02, 0.02)
    amp = 0.10 + rng.uniform(-0.02, 0.03)        # lateral shake amplitude (m)
    n_cycles = 3
    base = end + np.array([-0.04, 0.0, 0.0])     # shake about here (x,y)
    t_grasp, t_release = 0.22 * T, 0.90 * T
    wps = [
        Waypoint(0.00 * T, _hover(end, 0.18), 1.0),
        Waypoint(0.12 * T, _hover(end, 0.08), 1.0),
        Waypoint(0.18 * T, _hover(end, GRASP_Z), 1.0),
        Waypoint(0.22 * T, _hover(end, GRASP_Z), 0.0),      # close
        Waypoint(0.30 * T, np.array([base[0], base[1], hold_z]), 0.0),  # lift
    ]
    n_seg = 12
    for i in range(1, n_seg + 1):
        frac = i / n_seg
        phase = 2 * np.pi * n_cycles * frac
        y = base[1] + amp * np.sin(phase)
        x = base[0] + 0.02 * np.sin(0.5 * phase)
        wps.append(Waypoint((0.32 + 0.54 * frac) * T,
                            np.array([x, y, hold_z]), 0.0))
    wps += [
        Waypoint(0.90 * T, np.array([base[0], base[1], hold_z]), 1.0),  # release
        Waypoint(1.00 * T, np.array([base[0], base[1], hold_z + 0.05]), 1.0),
    ]
    return TaskSpec(
        task_id="shake_wave", subfamily="shake_wave",
        skill_label="grasp_oscillate_endpoint_release",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        grasp=GraspEvent("RB_first", t_on=t_grasp, t_off=t_release),
        workspace_center=end + np.array([0.12, 0.0, 0.0]),
        release_time=t_release,
        notes=(f"lift endpoint to ~{hold_z:.2f} m and shake laterally "
               f"+/-{amp:.2f} m for {n_cycles} cycles -> traveling waves"),
    )


def _lift_and_drape(rng, T) -> TaskSpec:
    """Carry the grasped endpoint up and over a fixed horizontal bar, then
    lower it on the far side so the rope drapes over the bar (gravity bend)."""
    bar_x = 0.48 + rng.uniform(-0.02, 0.02)
    end = np.array([bar_x - 0.02 + rng.uniform(-0.02, 0.02),
                    -0.16 + rng.uniform(-0.02, 0.02), TABLE_Z])
    apex = np.array([bar_x, 0.0, BAR_HEIGHT + 0.16])      # above the bar apex
    cross = np.array([bar_x, 0.15, BAR_HEIGHT + 0.06])    # past bar, still high
    far = np.array([bar_x, 0.14, 0.19])                   # gentle far-side descent
    t_grasp, t_release = 0.20 * T, 0.88 * T
    # Staged over-and-down path: lift high, cross the bar on the far side while
    # still above it, then descend gently. Avoids a harsh diagonal that the
    # fixed-orientation IK cannot track (which used to spike a false "snag").
    wps = [
        Waypoint(0.00 * T, _hover(end, 0.16), 1.0),
        Waypoint(0.11 * T, _hover(end, 0.08), 1.0),
        Waypoint(0.17 * T, _hover(end, GRASP_Z), 1.0),
        Waypoint(0.20 * T, _hover(end, GRASP_Z), 0.0),       # close
        Waypoint(0.38 * T, np.array([end[0], end[1], BAR_HEIGHT + 0.18]), 0.0),
        Waypoint(0.54 * T, apex, 0.0),                        # over the bar
        Waypoint(0.70 * T, cross, 0.0),                       # onto the far side
        Waypoint(0.84 * T, far, 0.0),                         # lower gently
        Waypoint(0.88 * T, far, 1.0),                         # release
        Waypoint(1.00 * T, _hover(far, 0.12), 1.0),
    ]
    bar_from = np.array([bar_x, -BAR_HALF_SPAN, BAR_HEIGHT])
    bar_to = np.array([bar_x, BAR_HALF_SPAN, BAR_HEIGHT])
    bar_xml = (
        f'<geom name="bar" type="cylinder" size="{BAR_RADIUS}" '
        f'fromto="{bar_from[0]:.4f} {bar_from[1]:.4f} {bar_from[2]:.4f} '
        f'{bar_to[0]:.4f} {bar_to[1]:.4f} {bar_to[2]:.4f}" '
        f'rgba="0.55 0.57 0.62 1" friction="__POST_FRICTION__ 0.005 0.0001"/>')
    # two support legs (visual/support, thin cylinders to the table)
    for sname, sy in (("bar_leg_l", -BAR_HALF_SPAN), ("bar_leg_r", BAR_HALF_SPAN)):
        bar_xml += (
            f'\n<geom name="{sname}" type="cylinder" size="0.012 {BAR_HEIGHT/2:.4f}" '
            f'pos="{bar_x:.4f} {sy:.4f} {BAR_HEIGHT/2:.4f}" rgba="0.45 0.46 0.5 1"/>')
    return TaskSpec(
        task_id="lift_and_drape", subfamily="lift_and_drape",
        skill_label="grasp_lift_over_bar_drape_release",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        grasp=GraspEvent("RB_first", t_on=t_grasp, t_off=t_release),
        workspace_center=np.array([bar_x, 0.0, 0.0]),
        release_time=t_release,
        bar_pose_world={"axis_from": bar_from.tolist(), "axis_to": bar_to.tolist(),
                        "radius": BAR_RADIUS, "height": BAR_HEIGHT},
        extra_worldbody_xml=bar_xml,
        notes=("carry endpoint up and over a horizontal bar at "
               f"z={BAR_HEIGHT}, lower on the far side so the rope drapes"),
    )


def _coil_on_table(rng, T) -> TaskSpec:
    """Grasp the endpoint and trace an inward spiral on the table, coiling the
    rope onto itself (self-collision / piling dynamics)."""
    center = np.array([0.44 + rng.uniform(-0.02, 0.02),
                       0.04 + rng.uniform(-0.02, 0.02), TABLE_Z])
    end = center + np.array([0.16, 0.0, 0.0])            # start on the rim
    coil_z = 0.028
    r0, r1 = 0.16, 0.03
    turns = 2.0
    t_grasp, t_release = 0.24 * T, 0.92 * T
    wps = [
        Waypoint(0.00 * T, _hover(end, 0.16), 1.0),
        Waypoint(0.12 * T, _hover(end, 0.08), 1.0),
        Waypoint(0.20 * T, _hover(end, GRASP_Z), 1.0),
        Waypoint(0.24 * T, _hover(end, GRASP_Z), 0.0),      # close
    ]
    n_arc = 16
    a0 = 0.0
    for i in range(n_arc + 1):
        frac = i / n_arc
        a = a0 + 2 * np.pi * turns * frac
        r = r0 + (r1 - r0) * frac
        p = center + np.array([r * np.cos(a), r * np.sin(a), coil_z])
        wps.append(Waypoint((0.30 + 0.60 * frac) * T, p, 0.0))
    last = wps[-1].pos
    wps += [
        Waypoint(0.92 * T, last, 1.0),                      # release
        Waypoint(1.00 * T, _hover(last, 0.12), 1.0),
    ]
    return TaskSpec(
        task_id="coil_on_table", subfamily="coil_on_table",
        skill_label="grasp_spiral_coil_release",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        grasp=GraspEvent("RB_first", t_on=t_grasp, t_off=t_release),
        workspace_center=center.copy(),
        release_time=t_release,
        coil_center_world=center.tolist(),
        notes=(f"spiral endpoint inward from r={r0} to r={r1} over {turns} "
               "turns to coil the rope on the table"),
    )


def _twirl_overhead(rng, T) -> TaskSpec:
    """Grasp one endpoint, lift it, and move the hand in a horizontal circle so
    the rope trails and lifts outward (rotational / centrifugal dynamics)."""
    center = np.array([0.44 + rng.uniform(-0.02, 0.02),
                       0.00 + rng.uniform(-0.02, 0.02), TABLE_Z])
    end = center + np.array([0.02, -0.06, 0.0])
    hold_z = 0.34 + rng.uniform(-0.02, 0.02)
    radius = 0.09 + rng.uniform(-0.01, 0.02)
    n_rev = 2.0
    t_grasp, t_release = 0.22 * T, 0.90 * T
    wps = [
        Waypoint(0.00 * T, _hover(end, 0.16), 1.0),
        Waypoint(0.11 * T, _hover(end, 0.08), 1.0),
        Waypoint(0.18 * T, _hover(end, GRASP_Z), 1.0),
        Waypoint(0.22 * T, _hover(end, GRASP_Z), 0.0),      # close
        Waypoint(0.32 * T, np.array([center[0] + radius, center[1], hold_z]), 0.0),
    ]
    n_arc = 16
    for i in range(1, n_arc + 1):
        frac = i / n_arc
        a = 2 * np.pi * n_rev * frac
        p = center + np.array([radius * np.cos(a), radius * np.sin(a),
                               hold_z - center[2]])
        wps.append(Waypoint((0.34 + 0.54 * frac) * T, p, 0.0))
    settle = np.array([center[0] + radius, center[1], hold_z])
    wps += [
        Waypoint(0.90 * T, settle, 1.0),                    # release
        Waypoint(1.00 * T, np.array([settle[0], settle[1], hold_z + 0.05]), 1.0),
    ]
    return TaskSpec(
        task_id="twirl_overhead", subfamily="twirl_overhead",
        skill_label="grasp_twirl_circle_release",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        grasp=GraspEvent("RB_first", t_on=t_grasp, t_off=t_release),
        workspace_center=center.copy(),
        release_time=t_release,
        twirl_center_world=(center + np.array([0.0, 0.0, hold_z])).tolist(),
        notes=(f"lift endpoint to ~{hold_z:.2f} m and move it in a circle of "
               f"r={radius:.2f} for {n_rev} revolutions -> the rope trails"),
    )


def _sweep_aside(rng, T) -> TaskSpec:
    """Non-prehensile: keep the gripper closed and push the rope sideways
    across the table (contact-only manipulation of a deformable). No grasp."""
    end = np.array([0.40 + rng.uniform(-0.02, 0.02),
                    0.00 + rng.uniform(-0.02, 0.02), TABLE_Z])
    mid = end + np.array([0.28, 0.0, 0.0])               # rope extends +x
    push_from = np.array([mid[0], -0.14, 0.0])           # start -y of the rope
    push_to = np.array([mid[0], 0.20, 0.0])              # push toward +y
    push_z = 0.018
    wps = [
        Waypoint(0.00 * T, _hover(push_from, 0.18), 0.0),   # closed gripper
        Waypoint(0.20 * T, _hover(push_from, 0.06), 0.0),
        Waypoint(0.30 * T, _hover(push_from, push_z), 0.0),
        Waypoint(0.62 * T, _hover(mid, push_z), 0.0),        # sweep across rope
        Waypoint(0.84 * T, _hover(push_to, push_z), 0.0),
        Waypoint(1.00 * T, _hover(push_to, 0.16), 0.0),
    ]
    return TaskSpec(
        task_id="sweep_aside", subfamily="sweep_aside",
        skill_label="nonprehensile_push_rope_aside",
        waypoints=wps, rope=RopeLayout(offset=end.copy()),
        # t_on far beyond the episode => the endpoint is never welded (no grasp)
        grasp=GraspEvent("RB_first", t_on=1.0e9, t_off=None),
        workspace_center=mid.copy(),
        release_time=None,
        notes="non-prehensile sideways push of the rope with a closed gripper",
    )
