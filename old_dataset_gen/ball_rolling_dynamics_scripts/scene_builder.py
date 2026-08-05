"""Robot-free RoboCasa island scenes for passive ball-rolling dynamics.

The scene contains only the kitchen and a free ball rolling across the island
counter.  Across episodes of one family everything is fixed — scene, style,
lighting, cameras, ball size/mass/color, and start position — except the
ball's initial velocity (speed and heading), which is the single randomized
quantity.  The historical ``robot_base_*`` visual-settings keys survive as
camera-placement anchors computed from the stool row; no robot is added to
the model.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from .utils import sphere_mass
from .variants import add_material, add_variant_xml
from .yaml_scene import TRAIN_ENVIRONMENTS, resolve_scene


DEMO_MODE = "ball_rolling_dynamics"
ROLLING_BALL_EDGE_MARGIN = 0.11
ROLLING_BALL_TANGENT_MARGIN = 0.18
ROLLING_BALL_START_DIST_RANGE = (0.38, 0.62)
ROLLING_BALL_CATCH_FORWARD_OFFSET = 0.12
ROLLING_BALL_ROBOT_EDGE_INSET = 0.18
BALL_SPAWN_CLEARANCE = 0.004

# The single randomized quantity: initial velocity.  The ball rolls along
# the island's LONG axis (started near one end), the heading swings +/-15
# degrees around that axis, and the speed is drawn from this range — but
# additionally capped per episode so the full 2.5 s roll stays inside the
# island's edge margin (rolling friction is so low that travel distance is
# essentially speed x duration; the ball must never leave the table).  The
# long-axis runway roughly doubles the containable speed compared to the
# old across-the-island path; the jitter is kept moderate so diagonal
# headings toward the narrow sides don't strangle the cap.
ROLLING_SPEED_RANGE_MPS = (0.12, 1.40)
ROLLING_HEADING_JITTER_RAD = math.radians(15.0)
ROLLING_EDGE_SAFETY_FRACTION = 0.92
ROLLING_BALL_START_INSET = 0.18

# Fixed (non-randomized) episode constants.
BALL_RADIUS_M = 0.04
BALL_DENSITY_KG_M3 = 42.0
BALL_COLOR = (0.88, 0.18, 0.16, 1.0)
BALL_FRICTION = (5.0, 0.01, 0.001)
BALL_SOLREF = (0.004, 1.0)
BALL_SOLIMP = (0.95, 0.99, 0.001)
LIGHTING_INTENSITY = 1.0


@dataclass(frozen=True)
class EpisodeSample:
    seed: int
    scene_variant: str
    timestep: float
    duration_s: float
    fps: int
    gravity: tuple[float, float, float]
    ball_radius: float
    ball_mass: float
    ball_color: tuple[float, float, float, float]
    ball_initial_position: tuple[float, float, float]
    ball_initial_velocity: tuple[float, float, float]
    ball_initial_angular_velocity: tuple[float, float, float]
    lighting_intensity: float
    camera_jitter: tuple[float, float, float]
    floor_material_jitter: float
    tabletop_height: float
    visual_settings: dict
    offscreen_width: int = 1920
    offscreen_height: int = 1080
    dataset_tags: dict | None = None


@dataclass
class EpisodeBundle:
    model: mujoco.MjModel
    data: mujoco.MjData
    xml: str
    camera_pose: dict


def _xy_axes_from_yaw(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    x_axis = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    y_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    return x_axis, y_axis


def _world_to_counter_local(counter, xy: np.ndarray) -> np.ndarray:
    delta = np.asarray(xy, dtype=np.float64) - np.asarray(counter.pos[:2], dtype=np.float64)
    c = math.cos(counter.yaw)
    s = math.sin(counter.yaw)
    return np.array([delta[0] * c + delta[1] * s, -delta[0] * s + delta[1] * c], dtype=np.float64)


def _counter_local_to_world(counter, local_xy: np.ndarray) -> np.ndarray:
    x_axis, y_axis = _xy_axes_from_yaw(counter.yaw)
    return np.asarray(counter.pos[:2], dtype=np.float64) + x_axis * float(local_xy[0]) + y_axis * float(local_xy[1])


def _rolling_spin_from_velocity(velocity_xy: np.ndarray, radius: float) -> tuple[float, float, float]:
    if radius <= 0.0:
        return (0.0, 0.0, 0.0)
    return (
        float(-velocity_xy[1] / radius),
        float(velocity_xy[0] / radius),
        0.0,
    )


def _island_visual_settings(scene, seed: int, camera_variant: str | None) -> dict:
    """Camera-placement anchors from the island's stool row (no robot exists)."""

    counter = next(
        fixture for fixture in scene.fixtures if fixture.name == scene.placement.counter_name
    )
    stools = [fixture for fixture in scene.fixtures if fixture.fixture_type == "stool"]
    local_stools = [
        (fixture, _world_to_counter_local(counter, np.asarray(fixture.pos[:2], dtype=np.float64)))
        for fixture in stools
    ]
    if not local_stools:
        raise ValueError(f"Scene {scene.layout_id}/{scene.style_id} has no stools for island placement.")

    counter_hx, counter_hy, _ = counter.half_size
    edge_scores = {
        "+x": sum(abs(local[0] - counter_hx) for _, local in local_stools) / len(local_stools),
        "-x": sum(abs(local[0] + counter_hx) for _, local in local_stools) / len(local_stools),
        "+y": sum(abs(local[1] - counter_hy) for _, local in local_stools) / len(local_stools),
        "-y": sum(abs(local[1] + counter_hy) for _, local in local_stools) / len(local_stools),
    }
    stool_edge = min(edge_scores, key=edge_scores.get)
    opposite_edge = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}[stool_edge]

    if stool_edge in {"+x", "-x"}:
        tangent_values = np.array([local[1] for _, local in local_stools], dtype=np.float64)
        tangent_limit = max(0.0, counter_hy - ROLLING_BALL_TANGENT_MARGIN)
        tangential_local = float(np.clip(np.mean(tangent_values), -tangent_limit, tangent_limit))
        anchor_local = np.array(
            [
                counter_hx - ROLLING_BALL_ROBOT_EDGE_INSET if opposite_edge == "+x" else -counter_hx + ROLLING_BALL_ROBOT_EDGE_INSET,
                tangential_local,
            ],
            dtype=np.float64,
        )
    else:
        tangent_values = np.array([local[0] for _, local in local_stools], dtype=np.float64)
        tangent_limit = max(0.0, counter_hx - ROLLING_BALL_TANGENT_MARGIN)
        tangential_local = float(np.clip(np.mean(tangent_values), -tangent_limit, tangent_limit))
        anchor_local = np.array(
            [
                tangential_local,
                counter_hy - ROLLING_BALL_ROBOT_EDGE_INSET if opposite_edge == "+y" else -counter_hy + ROLLING_BALL_ROBOT_EDGE_INSET,
            ],
            dtype=np.float64,
        )

    edge_to_yaw = {
        "+x": math.pi,
        "-x": 0.0,
        "+y": -math.pi * 0.5,
        "-y": math.pi * 0.5,
    }
    anchor_yaw = counter.yaw + edge_to_yaw[opposite_edge]
    forward_xy = np.array([math.cos(anchor_yaw), math.sin(anchor_yaw)], dtype=np.float64)
    target_xy = _counter_local_to_world(counter, anchor_local) + forward_xy * ROLLING_BALL_CATCH_FORWARD_OFFSET
    tabletop_height = float(counter.pos[2] + counter.half_size[2])

    return {
        "kitchen_env_index": int(seed % len(TRAIN_ENVIRONMENTS)),
        "kitchen_env_count": len(TRAIN_ENVIRONMENTS),
        "kitchen_layout_id": scene.layout_id,
        "kitchen_style_id": scene.style_id,
        "robot_counter_name": counter.name,
        "robot_counter_edge": opposite_edge,
        "robot_stool_edge": stool_edge,
        "robot_stool_names": [fixture.name for fixture, _ in local_stools],
        "robot_tabletop_height": tabletop_height,
        # Camera-placement anchor only; no robot is present in the scene.
        "robot_base_position": [float(v) for v in (*_counter_local_to_world(counter, anchor_local), tabletop_height)],
        "robot_base_yaw": float(anchor_yaw),
        "launch_position_xy": [float(v) for v in target_xy],
        "catch_position_xyz": [float(target_xy[0]), float(target_xy[1]), float(tabletop_height + ROLLING_BALL_EDGE_MARGIN + 0.02)],
        "room_bounds": list(scene.room_bounds),
        "scene_resolution_backend": scene.backend,
        "scene_resolution_warning": scene.native_error,
        # Unit vector from the island toward its stool side — the open side
        # the cameras stand on.
        "camera_side_dir_xy": [float(v) for v in forward_xy],
        "camera_variant": camera_variant,
        "demo_mode": DEMO_MODE,
    }


def sample_episode(
    variant: str,
    seed: int,
    *,
    fps: int,
    duration: float,
    layout_id_override: int | None = None,
    style_id_override: int | None = None,
    camera_variant: str | None = None,
) -> EpisodeSample:
    """Build the deterministic base episode for one family.

    Everything here is a pure function of the scene identifiers.  Episode-level
    variation is applied afterwards through :func:`randomize_initial_velocity`.
    """

    if variant != "robocasa_kitchen":
        raise ValueError(f"Ball-rolling dynamics supports only robocasa_kitchen scenes, not {variant!r}")
    if layout_id_override is None or style_id_override is None:
        raise ValueError("Ball-rolling dynamics requires explicit layout/style identifiers")
    scene = resolve_scene(int(layout_id_override), int(style_id_override))
    visual_settings = _island_visual_settings(scene, seed, camera_variant)

    radius = BALL_RADIUS_M
    mass = sphere_mass(radius, BALL_DENSITY_KG_M3)
    tabletop_height = float(visual_settings["robot_tabletop_height"])

    counter = next(fixture for fixture in scene.fixtures if fixture.name == visual_settings["robot_counter_name"])
    half = np.array([float(counter.half_size[0]), float(counter.half_size[1])], dtype=np.float64)
    long_axis = 0 if half[0] >= half[1] else 1
    runway_straight = 2.0 * half[long_axis] - ROLLING_BALL_START_INSET - ROLLING_BALL_EDGE_MARGIN
    if runway_straight <= 0.20:
        raise ValueError(f"Island for layout {scene.layout_id} is too short for a long-axis roll.")
    axis_world = _xy_axes_from_yaw(counter.yaw)[long_axis].copy()
    # The main camera stands on the stool side looking at the island; its
    # image-left direction in world coordinates is (side_y, -side_x).  Roll
    # that way so the ball always enters at the right frame edge and travels
    # toward the left one.
    side_dir = np.asarray(visual_settings["camera_side_dir_xy"], dtype=np.float64)
    screen_leftward = np.array([side_dir[1], -side_dir[0]], dtype=np.float64)
    axis_sign = 1.0 if float(np.dot(axis_world, screen_leftward)) >= 0.0 else -1.0
    base_direction = axis_sign * axis_world
    start_local = np.zeros(2, dtype=np.float64)
    start_local[long_axis] = -axis_sign * (half[long_axis] - ROLLING_BALL_START_INSET)
    end_local = np.zeros(2, dtype=np.float64)
    end_local[long_axis] = axis_sign * (half[long_axis] - ROLLING_BALL_EDGE_MARGIN)
    start_xy = _counter_local_to_world(counter, start_local)
    end_xy = _counter_local_to_world(counter, end_local)
    start_position = np.array(
        [float(start_xy[0]), float(start_xy[1]), tabletop_height + radius + BALL_SPAWN_CLEARANCE],
        dtype=np.float64,
    )
    base_speed = 0.5 * (ROLLING_SPEED_RANGE_MPS[0] + ROLLING_SPEED_RANGE_MPS[1])
    ball_initial_velocity = (base_direction * base_speed).tolist() + [0.0]
    ball_initial_angular_velocity = _rolling_spin_from_velocity(
        np.asarray(ball_initial_velocity[:2], dtype=np.float64), radius
    )

    frame_end_xy = start_xy + base_direction * (ROLLING_EDGE_SAFETY_FRACTION * runway_straight)
    visual_settings["launch_position_xy"] = [float(v) for v in start_xy]
    visual_settings["catch_position_xyz"] = [float(end_xy[0]), float(end_xy[1]), float(tabletop_height + radius)]
    visual_settings["rolling_path_start_xy"] = [float(v) for v in start_xy]
    visual_settings["rolling_path_end_xy"] = [float(v) for v in end_xy]
    # Farthest reachable stop under the speed cap — the segment the main
    # camera frames edge-to-edge.
    visual_settings["rolling_frame_end_xy"] = [float(v) for v in frame_end_xy]
    visual_settings["rolling_start_distance_m"] = float(runway_straight)
    visual_settings["rolling_speed_mps"] = float(base_speed)
    visual_settings["rolling_direction_xy"] = [float(v) for v in base_direction]
    # Counter geometry for the per-heading speed cap in randomize_initial_velocity.
    visual_settings["counter_center_xy"] = [float(v) for v in counter.pos[:2]]
    visual_settings["counter_yaw"] = float(counter.yaw)
    visual_settings["counter_half_size_xy"] = [float(half[0]), float(half[1])]
    visual_settings["rolling_start_local_xy"] = [float(v) for v in start_local]

    return EpisodeSample(
        seed=seed,
        scene_variant=variant,
        timestep=1.0 / 240.0,
        duration_s=duration,
        fps=fps,
        gravity=(0.0, 0.0, -9.81),
        ball_radius=radius,
        ball_mass=mass,
        ball_color=BALL_COLOR,
        ball_initial_position=tuple(float(v) for v in start_position),
        ball_initial_velocity=tuple(float(v) for v in ball_initial_velocity),
        ball_initial_angular_velocity=tuple(float(v) for v in ball_initial_angular_velocity),
        lighting_intensity=LIGHTING_INTENSITY,
        camera_jitter=(0.0, 0.0, 0.0),
        floor_material_jitter=0.0,
        tabletop_height=tabletop_height,
        visual_settings=visual_settings,
    )


def _runway_distance_m(visual_settings: dict, direction_world_xy: np.ndarray) -> float:
    """Distance from the fixed start to the edge-margin boundary along a heading."""

    yaw = float(visual_settings["counter_yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    d_local = np.array(
        [
            direction_world_xy[0] * c + direction_world_xy[1] * s,
            -direction_world_xy[0] * s + direction_world_xy[1] * c,
        ],
        dtype=np.float64,
    )
    start_local = np.asarray(visual_settings["rolling_start_local_xy"], dtype=np.float64)
    half = np.asarray(visual_settings["counter_half_size_xy"], dtype=np.float64)
    bounds = half - ROLLING_BALL_EDGE_MARGIN
    distance = math.inf
    for axis in range(2):
        if d_local[axis] > 1e-9:
            distance = min(distance, (bounds[axis] - start_local[axis]) / d_local[axis])
        elif d_local[axis] < -1e-9:
            distance = min(distance, (-bounds[axis] - start_local[axis]) / d_local[axis])
    return max(0.05, float(distance))


def rolling_speed_cap_mps(sample: EpisodeSample, direction_world_xy: np.ndarray) -> tuple[float, float]:
    """(speed cap, runway) so a full-duration roll along ``direction`` stays on the island."""

    runway = _runway_distance_m(sample.visual_settings, direction_world_xy)
    cap = min(
        ROLLING_SPEED_RANGE_MPS[1],
        ROLLING_EDGE_SAFETY_FRACTION * runway / sample.duration_s,
    )
    return cap, runway


def rolling_direction_for_offset(sample: EpisodeSample, heading_offset: float) -> np.ndarray:
    base_direction = np.asarray(sample.visual_settings["rolling_direction_xy"], dtype=np.float64)
    c, s = math.cos(heading_offset), math.sin(heading_offset)
    return np.array(
        [c * base_direction[0] - s * base_direction[1], s * base_direction[0] + c * base_direction[1]],
        dtype=np.float64,
    )


def randomize_initial_velocity(
    sample: EpisodeSample,
    rng: np.random.Generator | None,
    *,
    heading_offset: float | None = None,
    speed: float | None = None,
) -> EpisodeSample:
    """Resample the ball's initial velocity — the only per-episode variation.

    The heading rotates by up to ``ROLLING_HEADING_JITTER_RAD`` either way and
    the speed is drawn from ``ROLLING_SPEED_RANGE_MPS``, capped so the whole
    episode's travel stays inside the island's edge margin along that heading
    (the ball never rolls off the table).  Initial spin stays consistent with
    rolling without slipping.

    ``heading_offset`` and ``speed`` override the draws (grouped counterfactual
    generation plans both per sibling); the runway cap still applies to a
    planned speed, so a caller can never send the ball off the island.
    """

    if heading_offset is None or speed is None:
        if rng is None:
            raise ValueError("rng is required unless both heading_offset and speed are given")
    if heading_offset is None:
        heading_offset = float(rng.uniform(-ROLLING_HEADING_JITTER_RAD, ROLLING_HEADING_JITTER_RAD))
    heading_offset = float(heading_offset)
    direction = rolling_direction_for_offset(sample, heading_offset)
    speed_cap, runway = rolling_speed_cap_mps(sample, direction)
    if speed is None:
        speed_floor = min(ROLLING_SPEED_RANGE_MPS[0], speed_cap)
        speed = float(rng.uniform(speed_floor, speed_cap))
    speed = min(float(speed), speed_cap)
    velocity = (direction * speed).tolist() + [0.0]
    spin = _rolling_spin_from_velocity(np.asarray(velocity[:2], dtype=np.float64), sample.ball_radius)
    visual_settings = dict(sample.visual_settings)
    visual_settings["rolling_speed_mps"] = speed
    visual_settings["rolling_speed_cap_mps"] = float(speed_cap)
    visual_settings["rolling_runway_m"] = float(runway)
    visual_settings["rolling_heading_offset_rad"] = heading_offset
    visual_settings["rolling_direction_xy"] = [float(v) for v in direction]
    return replace(
        sample,
        ball_initial_velocity=tuple(float(v) for v in velocity),
        ball_initial_angular_velocity=spin,
        visual_settings=visual_settings,
    )


def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def _configure_options(root: ET.Element, sample: EpisodeSample) -> None:
    option = _ensure_child(root, "option")
    option.set("timestep", f"{sample.timestep:.10f}")
    option.set("gravity", " ".join(str(v) for v in sample.gravity))
    option.set("integrator", option.get("integrator", "implicitfast"))
    option.set("cone", "elliptic")
    option.set("impratio", "3")
    size = _ensure_child(root, "size")
    size.set("nconmax", "2048")
    size.set("njmax", "2048")
    visual = _ensure_child(root, "visual")
    quality = visual.find("quality") or ET.SubElement(visual, "quality")
    quality.set("shadowsize", "4096")
    quality.set("offsamples", "4")
    map_el = visual.find("map") or ET.SubElement(visual, "map")
    # The low tabletop closeup camera sits well inside the old 0.03*extent
    # clip distance for these kitchen-scale scenes; shrink znear so nothing
    # near the counter surface is silently discarded.
    map_el.set("znear", "0.005")
    map_el.set("zfar", "20")
    global_el = visual.find("global") or ET.SubElement(visual, "global")
    global_el.set("offwidth", str(sample.offscreen_width))
    global_el.set("offheight", str(sample.offscreen_height))


def _add_ball(root: ET.Element, sample: EpisodeSample) -> None:
    asset = _ensure_child(root, "asset")
    add_material(asset, "catch_ball_mat", sample.ball_color, roughness=0.45)
    world = _ensure_child(root, "worldbody")
    body = ET.SubElement(world, "body", name="catch_ball", pos=" ".join(f"{v:.6f}" for v in sample.ball_initial_position))
    ET.SubElement(body, "freejoint", name="ball_freejoint")
    ET.SubElement(
        body,
        "geom",
        name="catch_ball_geom",
        type="sphere",
        size=f"{sample.ball_radius:.6f}",
        mass=f"{sample.ball_mass:.8f}",
        material="catch_ball_mat",
        condim="4",
        friction=" ".join(str(v) for v in BALL_FRICTION),
        solref=" ".join(str(v) for v in BALL_SOLREF),
        solimp=" ".join(str(v) for v in BALL_SOLIMP),
    )


def build_episode(sample: EpisodeSample) -> EpisodeBundle:
    root = ET.Element("mujoco", model="ball_rolling_dynamics")
    ET.SubElement(root, "compiler", angle="radian")
    _configure_options(root, sample)
    camera_pose = add_variant_xml(
        root,
        sample.scene_variant,
        lighting_intensity=sample.lighting_intensity,
        floor_jitter=sample.floor_material_jitter,
        camera_jitter=sample.camera_jitter,
        visual_settings=sample.visual_settings,
    )
    _add_ball(root, sample)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return EpisodeBundle(model=model, data=data, xml=xml, camera_pose=camera_pose)
