from __future__ import annotations

import math
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
WRIST_CAMERA_VISUAL_BODY = "wrist_rgb_body"
WRIST_CAMERA_FOVY = 75.0
# At the hand body's own origin -- the fingertip pad center sits at local
# (~0, ~0, 0.103) regardless of the hand's world orientation (see
# MujocoCatchController._cache_hand_to_fingertip_center_local), a fixed fact
# about the Franka gripper's own kinematic chain, not specific to any one
# grasp scenario -- pulling the camera back along -Z to get more of the
# gripper into frame runs straight into the forearm's own solid mesh
# (verified: near-black renders a few cm back). Instead this stays at the
# origin (clear of self-geometry) and tilts 45deg off straight-down with a
# wide FOV, which keeps the gripper fingers in the lower part of the frame
# while pulling in the surrounding scene (counter, cabinets) above them --
# a typical eye-in-hand framing instead of a top-down close-up filling the
# whole frame with just the grasped object. The previous (-0.06, 0, 0.02)
# pos / (0.62, 0, 0.78) forward blend never showed the grasped object at
# all, in any tested position or orientation (see znear note below).
# wrist_rgb was previously never wired up to any renderer (see
# run_rollout.py/metadata.py), so nothing depends on the old pose.
WRIST_CAMERA_LOCAL_POS = (0.0, 0.0, 0.0)
WRIST_CAMERA_LOCAL_FORWARD = (0.342, 0.0, 0.940)
WRIST_CAMERA_LOCAL_UP = (1.0, 0.0, 0.0)
# The decorative camera-housing prop (box + lens + mount) is a separate body
# from the <camera> element itself. It must NOT sit at the same pos as the
# camera now that the camera is at the hand's own origin looking down +Z --
# a housing centered on the lens would enclose it and block the view (the
# housing previously shared WRIST_CAMERA_LOCAL_POS harmlessly because that
# pos was offset to the side, clear of whatever the camera looked at).
# Offset it behind the camera (-Z, back toward the wrist) so it stays out of
# frame.
WRIST_CAMERA_HOUSING_LOCAL_POS = (0.0, 0.0, -0.03)
WRIST_CAMERA_LOCAL_XYAXES = camera_xyaxes(
    (0.0, 0.0, 0.0),
    WRIST_CAMERA_LOCAL_FORWARD,
    up=WRIST_CAMERA_LOCAL_UP,
)
PROJECTILE_LAUNCH_ANGLE_DEG = 65.0
BALL_SPAWN_CLEARANCE = 0.004
CANNON_BARREL_LENGTH = 0.18
CANNON_BARREL_RADIUS = 0.045
CANNON_MOUNT_HEIGHT = 0.06
CANNON_MOUNT_RADIUS = 0.055
ROLLING_BALL_EDGE_MARGIN = 0.11
ROLLING_BALL_TANGENT_MARGIN = 0.18
ROLLING_BALL_START_DIST_RANGE = (0.38, 0.62)
ROLLING_BALL_SPEED_RANGE = (0.42, 0.70)
ROLLING_BALL_CATCH_FORWARD_OFFSET = 0.12
ROLLING_BALL_ROBOT_EDGE_INSET = 0.18
ROLLING_RELEASE_TIME_S = 0.30


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
    # rolling_surface_intercept only: absolute sim time (from episode start)
    # at which the arm begins reacting/tracking the ball; before this it
    # stays parked at home_q. Default 0.0 reproduces the original
    # react-from-t=0 behavior. Set above release_time_s to let the ball
    # roll on its own for a bit before the arm responds.
    arm_reaction_delay_s: float = 0.0
    dataset_tags: dict | None = None
    ball_initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class EpisodeBundle:
    model: mujoco.MjModel
    data: mujoco.MjData
    xml: str
    asset_info: AssetInfo
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


def _seat_row_demo_visual_settings(scene, seed: int, camera_variant: str | None) -> dict:
    counter = next(
        fixture for fixture in scene.fixtures if fixture.name == scene.placement.counter_name
    )
    stools = [fixture for fixture in scene.fixtures if fixture.fixture_type == "stool"]
    local_stools = [
        (fixture, _world_to_counter_local(counter, np.asarray(fixture.pos[:2], dtype=np.float64)))
        for fixture in stools
    ]
    if not local_stools:
        raise ValueError(f"Scene {scene.layout_id}/{scene.style_id} has no stools for island demo placement.")

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
        robot_local = np.array(
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
        robot_local = np.array(
            [
                tangential_local,
                counter_hy - ROLLING_BALL_ROBOT_EDGE_INSET if opposite_edge == "+y" else -counter_hy + ROLLING_BALL_ROBOT_EDGE_INSET,
            ],
            dtype=np.float64,
        )

    # Face the arm inward across the island, toward the stool side.
    edge_to_yaw = {
        "+x": math.pi,
        "-x": 0.0,
        "+y": -math.pi * 0.5,
        "-y": math.pi * 0.5,
    }
    robot_yaw = counter.yaw + edge_to_yaw[opposite_edge]
    forward_xy = np.array([math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float64)
    catch_xy = _counter_local_to_world(counter, robot_local) + forward_xy * ROLLING_BALL_CATCH_FORWARD_OFFSET
    tabletop_height = float(counter.pos[2] + counter.half_size[2])

    visual_settings = {
        "kitchen_env_index": int(seed % len(TRAIN_ENVIRONMENTS)),
        "kitchen_env_count": len(TRAIN_ENVIRONMENTS),
        "kitchen_layout_id": scene.layout_id,
        "kitchen_style_id": scene.style_id,
        "robot_counter_name": counter.name,
        "robot_counter_edge": opposite_edge,
        "robot_stool_edge": stool_edge,
        "robot_stool_names": [fixture.name for fixture, _ in local_stools],
        "robot_tabletop_height": tabletop_height,
        "robot_base_position": [float(v) for v in (*_counter_local_to_world(counter, robot_local), tabletop_height)],
        "robot_base_yaw": float(robot_yaw),
        "launch_position_xy": [float(v) for v in catch_xy],
        "catch_position_xyz": [float(catch_xy[0]), float(catch_xy[1]), float(tabletop_height + ROLLING_BALL_EDGE_MARGIN + 0.02)],
        "room_bounds": list(scene.room_bounds),
        "scene_resolution_backend": scene.backend,
        "scene_resolution_warning": scene.native_error,
        "camera_variant": camera_variant,
        "demo_mode": "rolling_island_catch",
    }
    return visual_settings


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
    demo_mode: str = "ballistic_intercept",
) -> dict | None:
    if variant != "robocasa_kitchen":
        return None
    env_index, layout_id, style_id = scene_ids_from_seed(seed)
    if layout_id_override is not None:
        layout_id = int(layout_id_override)
    if style_id_override is not None:
        style_id = int(style_id_override)
    scene = resolve_scene(layout_id, style_id)
    if demo_mode == "rolling_island_catch":
        return _seat_row_demo_visual_settings(scene, seed, camera_variant)
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
    demo_mode: str = "ballistic_intercept",
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
        demo_mode=demo_mode,
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
    ball_initial_angular_velocity = (0.0, 0.0, 0.0)
    controller_mode = "ballistic_intercept"
    release_time_s = 0.25
    catch_center_z = float(catch_position[2])
    if demo_mode == "rolling_island_catch":
        scene = resolve_scene(int(visual_settings["kitchen_layout_id"]), int(visual_settings["kitchen_style_id"]))
        counter = next(fixture for fixture in scene.fixtures if fixture.name == visual_settings["robot_counter_name"])
        robot_xy = np.asarray(visual_settings["robot_base_position"][:2], dtype=np.float64)
        forward_xy = np.asarray(catch_position[:2], dtype=np.float64) - robot_xy
        forward_norm = float(np.linalg.norm(forward_xy))
        if forward_norm <= 1e-6:
            raise ValueError("Rolling island demo placement produced a degenerate robot-to-catch direction.")
        forward_xy /= forward_norm
        tangent_xy = np.array([-forward_xy[1], forward_xy[0]], dtype=np.float64)
        catch_local = _world_to_counter_local(counter, catch_position[:2])
        if visual_settings["robot_counter_edge"] in {"+x", "-x"}:
            max_extra = (
                catch_local[0] - (-counter.half_size[0] + ROLLING_BALL_EDGE_MARGIN)
                if visual_settings["robot_counter_edge"] == "+x"
                else (counter.half_size[0] - ROLLING_BALL_EDGE_MARGIN) - catch_local[0]
            )
            tangent_limit = max(0.02, counter.half_size[1] - ROLLING_BALL_TANGENT_MARGIN)
            tangent_coord = float(np.clip(catch_local[1] + rng.uniform(-0.10, 0.10), -tangent_limit, tangent_limit))
            catch_local = np.array([catch_local[0], tangent_coord], dtype=np.float64)
        else:
            max_extra = (
                catch_local[1] - (-counter.half_size[1] + ROLLING_BALL_EDGE_MARGIN)
                if visual_settings["robot_counter_edge"] == "+y"
                else (counter.half_size[1] - ROLLING_BALL_EDGE_MARGIN) - catch_local[1]
            )
            tangent_limit = max(0.02, counter.half_size[0] - ROLLING_BALL_TANGENT_MARGIN)
            tangent_coord = float(np.clip(catch_local[0] + rng.uniform(-0.10, 0.10), -tangent_limit, tangent_limit))
            catch_local = np.array([tangent_coord, catch_local[1]], dtype=np.float64)
        far_side_margin = 0.06
        travel_upper = max(ROLLING_BALL_START_DIST_RANGE[0] + 0.03, max_extra - far_side_margin)
        travel_lower = max(
            ROLLING_BALL_START_DIST_RANGE[0],
            min(ROLLING_BALL_START_DIST_RANGE[1], travel_upper - 0.16),
        )
        if travel_upper <= travel_lower:
            raise ValueError(f"Island for layout {scene.layout_id} is too shallow for rolling demo start placement.")
        travel_dist = float(rng.uniform(travel_lower, travel_upper))
        catch_xy = _counter_local_to_world(counter, catch_local)
        launch_xy = catch_xy + forward_xy * travel_dist
        launch_position = np.array(
            [float(launch_xy[0]), float(launch_xy[1]), tabletop_height + radius + BALL_SPAWN_CLEARANCE],
            dtype=np.float64,
        )
        speed = float(rng.uniform(*ROLLING_BALL_SPEED_RANGE))
        ball_initial_velocity = (-forward_xy * speed).tolist() + [0.0]
        ball_initial_angular_velocity = _rolling_spin_from_velocity(np.asarray(ball_initial_velocity[:2], dtype=np.float64), radius)
        visual_settings["launch_position_xy"] = [float(v) for v in launch_xy]
        visual_settings["catch_position_xyz"] = [float(catch_xy[0]), float(catch_xy[1]), float(tabletop_height + radius)]
        visual_settings["rolling_start_distance_m"] = travel_dist
        visual_settings["rolling_speed_mps"] = speed
        visual_settings["rolling_direction_xy"] = [float(v) for v in (-forward_xy)]
        controller_mode = "rolling_surface_intercept"
        release_time_s = ROLLING_RELEASE_TIME_S
        catch_center_z = float(tabletop_height + radius)
    else:
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
        ball_initial_velocity=tuple(float(v) for v in ball_initial_velocity),
        lighting_intensity=float(rng.uniform(0.85, 1.18)),
        camera_jitter=camera_jitter,
        floor_material_jitter=float(rng.uniform(-0.04, 0.04)),
        robot_base_position=robot_base_position if tabletop_variant else (0.0, 0.0, 0.0),
        robot_base_euler=(0.0, 0.0, robot_base_yaw) if tabletop_variant else (0.0, 0.0, 0.0),
        tabletop_height=tabletop_height if tabletop_variant else None,
        visual_settings=visual_settings,
        catch_center_z=catch_center_z,
        release_time_s=release_time_s,
        controller_mode=controller_mode,
        ball_initial_angular_velocity=tuple(float(v) for v in ball_initial_angular_velocity),
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


def _add_launcher(root: ET.Element, sample: EpisodeSample) -> None:
    asset = _ensure_child(root, "asset")
    add_material(asset, "cannon_barrel_mat", (0.10, 0.11, 0.13, 1.0), roughness=0.30)
    add_material(asset, "cannon_trim_mat", (0.55, 0.47, 0.28, 1.0), roughness=0.25)
    add_material(asset, "cannon_mount_mat", (0.18, 0.16, 0.14, 1.0), roughness=0.55)

    world = _ensure_child(root, "worldbody")
    launch_position = np.asarray(sample.ball_initial_position, dtype=np.float64)
    launch_direction = np.asarray(sample.ball_initial_velocity, dtype=np.float64)
    launch_direction /= np.linalg.norm(launch_direction)
    barrel_pitch = float(np.arctan2(launch_direction[0], launch_direction[2]))

    barrel_center = launch_position - launch_direction * (0.5 * CANNON_BARREL_LENGTH - sample.ball_radius * 0.35)
    mount_center = barrel_center - launch_direction * 0.050
    mount_center[2] = sample.tabletop_height + CANNON_MOUNT_HEIGHT if sample.tabletop_height is not None else CANNON_MOUNT_HEIGHT
    support_height = max(0.03, float(barrel_center[2] - mount_center[2]))

    cannon = ET.SubElement(
        world,
        "body",
        name="projectile_launcher",
        pos=" ".join(f"{v:.6f}" for v in mount_center),
    )
    ET.SubElement(
        cannon,
        "geom",
        name="projectile_launcher_mount",
        type="cylinder",
        size=f"{CANNON_MOUNT_RADIUS:.6f} {CANNON_MOUNT_HEIGHT:.6f}",
        material="cannon_mount_mat",
        contype="0",
        conaffinity="0",
        group="1",
    )
    ET.SubElement(
        cannon,
        "geom",
        name="projectile_launcher_support",
        type="box",
        pos=f"0 0 {support_height * 0.5:.6f}",
        size=f"0.040000 0.060000 {support_height * 0.5:.6f}",
        material="cannon_mount_mat",
        contype="0",
        conaffinity="0",
        group="1",
    )

    barrel_body = ET.SubElement(
        cannon,
        "body",
        name="projectile_launcher_barrel",
        pos="0 0 0",
        euler=f"0 {barrel_pitch:.6f} 0",
    )
    barrel_local_z = float(barrel_center[2] - mount_center[2])
    ET.SubElement(
        barrel_body,
        "geom",
        name="projectile_launcher_barrel_geom",
        type="cylinder",
        pos=f"0 0 {barrel_local_z:.6f}",
        size=f"{CANNON_BARREL_RADIUS:.6f} {0.5 * CANNON_BARREL_LENGTH:.6f}",
        material="cannon_barrel_mat",
        contype="0",
        conaffinity="0",
        group="1",
    )
    ET.SubElement(
        barrel_body,
        "geom",
        name="projectile_launcher_muzzle_ring",
        type="cylinder",
        pos=f"0 0 {barrel_local_z + 0.5 * CANNON_BARREL_LENGTH - 0.012:.6f}",
        size=f"{CANNON_BARREL_RADIUS * 1.08:.6f} 0.012000",
        material="cannon_trim_mat",
        contype="0",
        conaffinity="0",
        group="1",
    )
    ET.SubElement(
        barrel_body,
        "geom",
        name="projectile_launcher_breech_ring",
        type="cylinder",
        pos=f"0 0 {barrel_local_z - 0.5 * CANNON_BARREL_LENGTH + 0.016:.6f}",
        size=f"{CANNON_BARREL_RADIUS * 1.15:.6f} 0.016000",
        material="cannon_trim_mat",
        contype="0",
        conaffinity="0",
        group="1",
    )


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
        friction="5.0 0.01 0.001",
        solref="0.004 1",
        solimp="0.95 0.99 0.001",
    )


def _add_wrist_camera(root: ET.Element) -> dict:
    asset = _ensure_child(root, "asset")
    add_material(asset, "wrist_camera_body_mat", (0.03, 0.035, 0.04, 1.0), roughness=0.38)
    add_material(asset, "wrist_camera_lens_mat", (0.02, 0.02, 0.025, 1.0), roughness=0.18)

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

    housing_pos = " ".join(f"{v:.6f}" for v in WRIST_CAMERA_HOUSING_LOCAL_POS)
    visual_body = ET.SubElement(
        hand,
        "body",
        name=WRIST_CAMERA_VISUAL_BODY,
        pos=housing_pos,
        xyaxes=WRIST_CAMERA_LOCAL_XYAXES,
    )
    visual_geom_common = {
        "contype": "0",
        "conaffinity": "0",
        "group": "2",
    }
    ET.SubElement(
        visual_body,
        "geom",
        name="wrist_rgb_body_geom",
        type="box",
        size="0.024 0.017 0.012",
        material="wrist_camera_body_mat",
        **visual_geom_common,
    )
    ET.SubElement(
        visual_body,
        "geom",
        name="wrist_rgb_lens_geom",
        type="cylinder",
        pos="0 0 -0.017",
        size="0.009 0.006",
        material="wrist_camera_lens_mat",
        **visual_geom_common,
    )
    ET.SubElement(
        visual_body,
        "geom",
        name="wrist_rgb_mount_geom",
        type="box",
        pos="0 0 0.018",
        size="0.018 0.012 0.005",
        rgba="0.18 0.18 0.18 1",
        **visual_geom_common,
    )
    return {
        "name": WRIST_CAMERA_NAME,
        "parent_body": WRIST_CAMERA_PARENT_BODY,
        "visual_body": WRIST_CAMERA_VISUAL_BODY,
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
