from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from .assets import AssetInfo, locate_franka_asset
from .utils import camera_xyaxes, sphere_mass
from .variants import (
    VARIANTS,
    add_material,
    add_variant_xml,
)
from .yaml_scene import TRAIN_ENVIRONMENTS, resolve_scene, scene_ids_from_seed


TABLETOP_ROBOT_VARIANTS = {"robocasa_kitchen"}
WRIST_CAMERA_NAME = "wrist_rgb"
WRIST_CAMERA_PARENT_BODY = "hand"
WRIST_CAMERA_FOVY = 70.0
WRIST_CAMERA_LOCAL_POS = (-0.060, 0.0, 0.020)
WRIST_CAMERA_LOCAL_FORWARD = (0.62, 0.0, 0.78)
WRIST_CAMERA_LOCAL_UP = (0.0, 1.0, 0.0)
WRIST_CAMERA_LOCAL_XYAXES = camera_xyaxes(
    (0.0, 0.0, 0.0),
    WRIST_CAMERA_LOCAL_FORWARD,
    up=WRIST_CAMERA_LOCAL_UP,
)
PROJECTILE_LAUNCH_ANGLE_DEG = 65.0
BALL_SPAWN_CLEARANCE = 0.004
BALL_FRICTION = (5.0, 0.01, 0.001)
BALL_SOLREF = (0.004, 1.0)
BALL_SOLIMP = (0.95, 0.99, 0.001)


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
    lighting_intensity: float
    camera_jitter: tuple[float, float, float]
    floor_material_jitter: float
    robot_base_position: tuple[float, float, float]
    robot_base_euler: tuple[float, float, float]
    tabletop_height: float | None
    visual_settings: dict | None = None
    release_time_s: float = 0.60
    catch_center_z: float = 0.50
    offscreen_width: int = 1920
    offscreen_height: int = 1080
    controller_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controller_mode: str = "ballistic_intercept"
    enable_grasp_capture: bool = True
    close_lead_time_offset_s: float = 0.0
    reach_lead_time_offset_s: float = 0.0
    dataset_tags: dict | None = None


@dataclass
class EpisodeBundle:
    model: mujoco.MjModel
    data: mujoco.MjData
    xml: str
    asset_info: AssetInfo
    camera_pose: dict


def _main_camera_horizontal_right_direction(variant: str, camera_jitter) -> np.ndarray:
    spec = VARIANTS[variant]
    camera_pos = np.asarray(spec.camera_pos, dtype=np.float64) + np.asarray(camera_jitter, dtype=np.float64)
    lookat = np.asarray(spec.camera_lookat, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = lookat - camera_pos
    forward /= np.linalg.norm(forward)
    camera_right = np.cross(forward, up)
    horizontal_right = camera_right.copy()
    # Project camera-right onto the ground plane so the added launch is horizontal.
    horizontal_right[2] = 0.0
    norm = np.linalg.norm(horizontal_right)
    if norm < 1e-9:
        raise ValueError(f"Main camera right vector became degenerate for variant '{variant}'.")
    return horizontal_right / norm


def _projectile_launch_velocity(
    *,
    launch_position: np.ndarray,
    target_position: np.ndarray,
    launch_angle_degrees: float,
    gravity_z: float,
) -> np.ndarray:
    horizontal_delta = target_position[:2] - launch_position[:2]
    horizontal_distance = float(np.linalg.norm(horizontal_delta))
    if horizontal_distance < 1e-9:
        raise ValueError("Projectile launch geometry became degenerate: horizontal distance is zero.")

    horizontal_direction = horizontal_delta / horizontal_distance
    angle_radians = np.deg2rad(float(launch_angle_degrees))
    tan_theta = float(np.tan(angle_radians))
    cos_theta = float(np.cos(angle_radians))
    delta_z = float(target_position[2] - launch_position[2])
    denominator = 2.0 * (cos_theta**2) * (horizontal_distance * tan_theta - delta_z)
    if denominator <= 1e-9:
        raise ValueError("Projectile launch geometry is invalid for the chosen angle and catch target.")

    launch_speed_sq = abs(float(gravity_z)) * (horizontal_distance**2) / denominator
    launch_speed = float(np.sqrt(launch_speed_sq))
    horizontal_speed = launch_speed * cos_theta
    vertical_speed = launch_speed * np.sin(angle_radians)
    return np.array(
        [
            horizontal_direction[0] * horizontal_speed,
            horizontal_direction[1] * horizontal_speed,
            vertical_speed,
        ],
        dtype=np.float64,
    )
def _sample_visual_settings_for_seed(
    rng: np.random.Generator,
    variant: str,
    seed: int,
    *,
    layout_id_override: int | None = None,
    style_id_override: int | None = None,
    camera_variant: str | None = None,
) -> dict | None:
    if variant != "robocasa_kitchen":
        return None
    env_index, layout_id, style_id = scene_ids_from_seed(seed)
    if layout_id_override is not None:
        layout_id = int(layout_id_override)
    if style_id_override is not None:
        style_id = int(style_id_override)
    scene = resolve_scene(layout_id, style_id)
    visual_settings = {
        "kitchen_env_index": env_index,
        "kitchen_env_count": len(TRAIN_ENVIRONMENTS),
        "kitchen_layout_id": layout_id,
        "kitchen_style_id": style_id,
        "robot_counter_name": scene.placement.counter_name,
        "robot_counter_edge": scene.placement.edge_name,
        "robot_tabletop_height": scene.placement.tabletop_height,
        "robot_base_position": list(scene.placement.robot_base_position),
        "robot_base_yaw": scene.placement.robot_base_yaw,
        "launch_position_xy": list(scene.placement.launch_position[:2]),
        "catch_position_xyz": list(scene.placement.catch_position),
        "room_bounds": list(scene.room_bounds),
        "scene_resolution_backend": scene.backend,
    }
    if scene.native_error is not None:
        visual_settings["scene_resolution_warning"] = scene.native_error
    if camera_variant is not None:
        visual_settings["camera_variant"] = camera_variant
    if seed == 0 and layout_id_override is None:
        visual_settings["seed_override"] = "seed0_sink_left_of_stove"
        visual_settings["override_sink_position_xyz"] = [1.35, -0.30, 0.958]
        visual_settings["override_sink_yaw"] = 0.0
    if seed == 1000 and layout_id_override is None:
        visual_settings["seed_override"] = "seed1000_close_launch"
        visual_settings["launch_position_xy"] = [2.330350935975962, -5.605]
    if seed == 2350 and layout_id == 58:
        visual_settings["seed_override"] = "seed2350_left_reversed_chair_view"
        visual_settings["robot_base_position"] = [3.34028079207492, -2.314466071608008, 0.9200005]
        visual_settings["robot_base_yaw"] = 0.0
        visual_settings["launch_position_xy"] = [3.02028079207492, -2.234466071608008]
        visual_settings["catch_position_xyz"] = [3.60028079207492, -2.314466071608008, 1.3100005]
    return visual_settings


def sample_episode(
    rng: np.random.Generator,
    variant: str,
    seed: int,
    *,
    fps: int,
    duration: float,
    layout_id_override: int | None = None,
    style_id_override: int | None = None,
    camera_variant: str | None = None,
) -> EpisodeSample:
    radius = float(rng.uniform(0.0245, 0.0255))
    density = float(rng.uniform(36.0, 48.0))
    mass = sphere_mass(radius, density)
    tabletop_variant = variant in TABLETOP_ROBOT_VARIANTS
    color = tuple(float(v) for v in rng.uniform(0.15, 0.95, size=3)) + (1.0,)
    if seed == 0:
        color = (0.10, 0.85, 0.20, 1.0)
    visual_settings = _sample_visual_settings_for_seed(
        rng,
        variant,
        seed,
        layout_id_override=layout_id_override,
        style_id_override=style_id_override,
        camera_variant=camera_variant,
    )
    camera_jitter = tuple(float(v) for v in rng.normal(0.0, (0.035, 0.035, 0.025)))
    tabletop_height = float(visual_settings["robot_tabletop_height"]) if visual_settings is not None else 0.0
    robot_base_position = tuple(float(v) for v in visual_settings["robot_base_position"]) if visual_settings is not None else (0.0, 0.0, 0.0)
    robot_base_yaw = float(visual_settings["robot_base_yaw"]) if visual_settings is not None else 0.0
    catch_position = np.asarray(visual_settings["catch_position_xyz"], dtype=np.float64) if visual_settings is not None else np.array([0.34, 0.0, 0.81], dtype=np.float64)
    launch_xy = np.asarray(visual_settings["launch_position_xy"], dtype=np.float64) if visual_settings is not None else np.array([0.78, 0.0], dtype=np.float64)
    launch_position = np.array(
        [
            float(launch_xy[0]),
            float(launch_xy[1]),
            tabletop_height + radius + BALL_SPAWN_CLEARANCE,
        ],
        dtype=np.float64,
    )
    ball_initial_velocity = tuple(
        float(v)
        for v in _projectile_launch_velocity(
            launch_position=launch_position,
            target_position=catch_position,
            launch_angle_degrees=PROJECTILE_LAUNCH_ANGLE_DEG,
            gravity_z=-9.81,
        )
    )
    return EpisodeSample(
        seed=seed,
        scene_variant=variant,
        timestep=1.0 / 240.0,
        duration_s=duration,
        fps=fps,
        gravity=(0.0, 0.0, -9.81),
        ball_radius=radius,
        ball_mass=mass,
        ball_color=color,
        ball_initial_position=tuple(float(v) for v in launch_position),
        ball_initial_velocity=ball_initial_velocity,
        lighting_intensity=float(rng.uniform(0.85, 1.18)),
        camera_jitter=camera_jitter,
        floor_material_jitter=float(rng.uniform(-0.04, 0.04)),
        robot_base_position=robot_base_position if tabletop_variant else (0.0, 0.0, 0.0),
        robot_base_euler=(0.0, 0.0, robot_base_yaw) if tabletop_variant else (0.0, 0.0, 0.0),
        tabletop_height=tabletop_height if tabletop_variant else None,
        visual_settings=visual_settings,
        catch_center_z=float(catch_position[2]),
        release_time_s=0.25,
    )


def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def _set_robot_compiler_paths(root: ET.Element, asset_info: AssetInfo) -> None:
    compiler = _ensure_child(root, "compiler")
    compiler.set("angle", "radian")
    if asset_info.asset_dir.exists():
        compiler.set("meshdir", str(asset_info.asset_dir))


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
    map_el.set("znear", "0.03")
    map_el.set("zfar", "20")
    global_el = visual.find("global") or ET.SubElement(visual, "global")
    global_el.set("offwidth", str(sample.offscreen_width))
    global_el.set("offheight", str(sample.offscreen_height))


def _configure_robot_base(root: ET.Element, sample: EpisodeSample) -> None:
    world = _ensure_child(root, "worldbody")
    base = world.find("./body[@name='link0']")
    if base is None:
        return
    base.set("pos", " ".join(f"{v:.6f}" for v in sample.robot_base_position))
    if any(abs(v) > 1e-9 for v in sample.robot_base_euler):
        base.set("euler", " ".join(f"{v:.9f}" for v in sample.robot_base_euler))


def _remove_incompatible_keyframes(root: ET.Element) -> None:
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)


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
        friction=" ".join(f"{v:g}" for v in BALL_FRICTION),
        solref=" ".join(f"{v:g}" for v in BALL_SOLREF),
        solimp=" ".join(f"{v:g}" for v in BALL_SOLIMP),
    )


def _add_wrist_camera(root: ET.Element) -> dict:
    world = _ensure_child(root, "worldbody")
    hand = world.find(f".//body[@name='{WRIST_CAMERA_PARENT_BODY}']")
    if hand is None:
        raise ValueError(f"Cannot add wrist camera: body '{WRIST_CAMERA_PARENT_BODY}' was not found.")

    pos = " ".join(f"{v:.6f}" for v in WRIST_CAMERA_LOCAL_POS)
    ET.SubElement(
        hand,
        "camera",
        name=WRIST_CAMERA_NAME,
        pos=pos,
        xyaxes=WRIST_CAMERA_LOCAL_XYAXES,
        fovy=f"{WRIST_CAMERA_FOVY:.3f}",
    )
    return {
        "name": WRIST_CAMERA_NAME,
        "parent_body": WRIST_CAMERA_PARENT_BODY,
        "local_pos": list(WRIST_CAMERA_LOCAL_POS),
        "local_xyaxes": [float(v) for v in WRIST_CAMERA_LOCAL_XYAXES.split()],
        "local_forward": list(WRIST_CAMERA_LOCAL_FORWARD),
        "local_up": list(WRIST_CAMERA_LOCAL_UP),
        "fovy": WRIST_CAMERA_FOVY,
    }


def build_episode(sample: EpisodeSample) -> EpisodeBundle:
    asset_info = locate_franka_asset()
    root = ET.parse(asset_info.robot_xml).getroot()
    _set_robot_compiler_paths(root, asset_info)
    _configure_options(root, sample)
    _configure_robot_base(root, sample)
    _remove_incompatible_keyframes(root)
    camera_pose = add_variant_xml(
        root,
        sample.scene_variant,
        lighting_intensity=sample.lighting_intensity,
        floor_jitter=sample.floor_material_jitter,
        camera_jitter=sample.camera_jitter,
        visual_settings=sample.visual_settings,
    )
    camera_pose[WRIST_CAMERA_NAME] = _add_wrist_camera(root)
    _add_ball(root, sample)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return EpisodeBundle(model=model, data=data, xml=xml, asset_info=asset_info, camera_pose=camera_pose)
