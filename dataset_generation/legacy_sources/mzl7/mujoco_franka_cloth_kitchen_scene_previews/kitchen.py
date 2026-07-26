"""RoboCasa kitchen-island scene integration.

Reuses the ball-roll-interception scene machinery (native RoboCasa kitchen
fixtures, materials, lighting) to replace the bland table/floor scene: the
whole cloth rig (arms, cloth, box, markers) is rigid-transformed onto the
kitchen island top, and the kitchen XML is injected into the episode MJCF.

Layouts cycle per physics variant, matching the rolling-island-catch demos:
(38, 42), (48, 41), (51, 34).
"""

import sys
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np

BALL_ROLL_ROOT = "/home/mzl7/scratch"
KITCHEN_LAYOUTS = [(38, 42), (48, 41), (51, 34)]


def layout_for_variant(variant: int) -> tuple:
    return KITCHEN_LAYOUTS[variant % len(KITCHEN_LAYOUTS)]


def _ball_roll():
    """Import the ball-roll-interception package lazily."""
    if BALL_ROLL_ROOT not in sys.path:
        sys.path.insert(0, BALL_ROLL_ROOT)
    from ball_roll_interception_scripts import scene_builder, variants, yaml_scene
    return yaml_scene, scene_builder, variants


def resolve_island(layout_id: int, style_id: int):
    """Resolved scene + the island counter fixture the ball-roll demos used."""
    yaml_scene, _, _ = _ball_roll()
    scene = yaml_scene.resolve_scene(layout_id, style_id)
    counter = next(f for f in scene.fixtures
                   if f.name == scene.placement.counter_name)
    tabletop_z = float(counter.pos[2] + counter.half_size[2])
    return scene, counter, tabletop_z


def _rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def plan_rig_placement(task, counter, tabletop_z: float, rng=None):
    """(offset_xyz, yaw) mapping rig-local coords onto the island top.

    yaw = island yaw so the rig x-axis runs along the island long axis; the
    rig footprint bbox center lands on the island top center, optionally
    slid along the island long axis by a random amount (`rng`).
    """
    yaw = float(counter.yaw)

    pts = []  # rig-local xy points with padding
    for arm in task.arms:
        b = np.asarray(arm.base_pos)[:2]
        pts += [b + d for d in ([0.20, 0.20], [-0.20, -0.20],
                                [0.20, -0.20], [-0.20, 0.20])]
    w, h = task.cloth.size
    cc = np.asarray(task.cloth.center)[:2]
    pts += [cc + [w / 2 + 0.05, h / 2 + 0.05], cc - [w / 2 + 0.05, h / 2 + 0.05]]
    if task.box_pose_world:
        bp = np.asarray(task.box_pose_world["pos"])[:2]
        bx, by = (s / 2 + 0.03 for s in task.box_pose_world["inner_size_xy"])
        pts += [bp + [bx, by], bp - [bx, by]]
    if task.fold_line_world:
        pts += [np.asarray(p)[:2] for p in task.fold_line_world]
    pts = np.asarray(pts)
    rig_center_local = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    rig_half_long = 0.5 * float(pts[:, 0].max() - pts[:, 0].min())

    # island top center in world; rig bbox center lands there, optionally
    # slid along the island long axis (local x) by a random margin-safe amount
    island_center = np.asarray(counter.pos[:2], dtype=float)
    slide = 0.0
    if rng is not None:
        s = max(0.0, float(counter.half_size[0]) - rig_half_long - 0.10)
        slide = float(rng.uniform(-s, s))
        island_center = island_center + _rz(yaw)[:2, :2] @ np.array([slide, 0.0])
    offset_xy = island_center - _rz(yaw)[:2, :2] @ rig_center_local
    offset = np.array([offset_xy[0], offset_xy[1], tabletop_z])  # TABLE_Z == 0
    return offset, yaw, slide


def _tf(p, offset, yaw):
    """p' = Rz(yaw) @ p + offset for a 3-vector."""
    return _rz(yaw) @ np.asarray(p, dtype=float) + offset


def transform_task_spec(task, offset, yaw):
    """Rigid-transform a TaskSpec (arms, waypoints, grasps, cloth, markers)."""
    offset = np.asarray(offset, dtype=float)

    arms = []
    for arm in task.arms:
        wps = [replace(w, pos=_tf(w.pos, offset, yaw)) for w in arm.waypoints]
        arms.append(replace(arm, base_pos=_tf(arm.base_pos, offset, yaw),
                            base_yaw=arm.base_yaw + yaw, waypoints=wps))
    grasps = [replace(g, point=_tf(g.point, offset, yaw)) for g in task.grasps]
    cloth = replace(task.cloth, center=_tf(task.cloth.center, offset, yaw))

    fold_line = None
    if task.fold_line_world:
        fold_line = [_tf(p, offset, yaw).tolist() for p in task.fold_line_world]

    box_pose = None
    if task.box_pose_world:
        box_pose = dict(task.box_pose_world)
        box_pose["pos"] = _tf(box_pose["pos"], offset, yaw).tolist()
        box_pose["quat"] = [float(np.cos(yaw / 2)), 0.0, 0.0,
                            float(np.sin(yaw / 2))]

    extra = task.extra_worldbody_xml
    if extra.strip():
        # Wrap verbatim in a positioned frame; rig-local coords stay intact.
        extra = (f'<body name="task_extras" '
                 f'pos="{offset[0]:.6g} {offset[1]:.6g} {offset[2]:.6g}" '
                 f'euler="0 0 {yaw:.6g}">\n{extra}\n</body>')

    # NOTE: the cloth flexcomp has no orientation attribute in the scene XML,
    # so the grid stays world-axis-aligned. Its extent is symmetric (square
    # grids), so alignment with the rotated rig is preserved for the 90-degree
    # island yaws used here; grasp points are matched to nearest vertex anyway.
    return replace(task, arms=arms, grasps=grasps, cloth=cloth,
                   fold_line_world=fold_line, box_pose_world=box_pose,
                   extra_worldbody_xml=extra,
                   workspace_center=_tf(task.workspace_center, offset, yaw))


def inject_kitchen_xml(xml_str: str, layout_id: int, style_id: int,
                       seed: int, lighting_intensity: float = 1.0):
    """Inject the RoboCasa kitchen (fixtures/materials/lighting) into the
    episode MJCF string. Returns (xml_str, kitchen_meta)."""
    yaml_scene, brl_scene_builder, variants = _ball_roll()

    root = ET.fromstring(xml_str)
    world = root.find("worldbody")

    # Drop the bland-scene floor/table geoms and scripted lights if present.
    for geom in list(world.findall("geom")):
        if geom.get("name") in {"floor", "table"}:
            world.remove(geom)
    for light in list(world.findall("light")):
        world.remove(light)

    scene = yaml_scene.resolve_scene(layout_id, style_id)
    visual_settings = brl_scene_builder._seat_row_demo_visual_settings(
        scene, seed, "arm_relative_island_demo")
    variants.add_variant_xml(
        root, "robocasa_kitchen", lighting_intensity=lighting_intensity,
        floor_jitter=0.0, camera_jitter=(0.0, 0.0, 0.0),
        visual_settings=visual_settings)

    counter = next(f for f in scene.fixtures
                   if f.name == scene.placement.counter_name)
    meta = {
        "kitchen_layout_id": layout_id,
        "kitchen_style_id": style_id,
        "island_fixture": counter.name,
        "tabletop_height": float(counter.pos[2] + counter.half_size[2]),
        "scene_resolution_backend": scene.backend,
    }
    return ET.tostring(root, encoding="unicode"), meta
