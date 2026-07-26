from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from .assets import AssetInfo, locate_franka_asset
from .utils import camera_xyaxes, sphere_mass
from .variants import add_material, add_variant_xml, rgba_str


TABLETOP_ROBOT_VARIANTS = {"robocasa_kitchen"}
TABLETOP_ROBOT_Z = 0.74
TABLETOP_ROBOT_YAW = 0.0
WRIST_CAMERA_NAME = "wrist_rgb"
WRIST_CAMERA_PARENT_BODY = "hand"
WRIST_CAMERA_VISUAL_BODY = "wrist_rgb_body"
WRIST_CAMERA_FOVY = 70.0
WRIST_CAMERA_LOCAL_POS = (-0.060, 0.0, 0.020)
WRIST_CAMERA_LOCAL_FORWARD = (0.62, 0.0, 0.78)
WRIST_CAMERA_LOCAL_UP = (0.0, 1.0, 0.0)
WRIST_CAMERA_LOCAL_XYAXES = camera_xyaxes(
    (0.0, 0.0, 0.0),
    WRIST_CAMERA_LOCAL_FORWARD,
    up=WRIST_CAMERA_LOCAL_UP,
)


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
    ball_color_name: str
    ball_initial_position: tuple[float, float, float]
    ball_initial_velocity: tuple[float, float, float]
    ball_initial_angular_velocity: tuple[float, float, float]
    ball_friction: float
    ball_restitution: float
    drop_height: float
    lighting_intensity: float
    camera_jitter: tuple[float, float, float]
    floor_material_jitter: float
    robot_base_position: tuple[float, float, float]
    robot_base_euler: tuple[float, float, float]
    tabletop_height: float | None
    catch_center_z: float
    clutter_seed: int
    release_time_s: float = 0.0
    offscreen_width: int = 1920
    offscreen_height: int = 1080


@dataclass
class EpisodeBundle:
    model: mujoco.MjModel
    data: mujoco.MjData
    xml: str
    asset_info: AssetInfo
    camera_pose: dict
    background_asset_ids: list


# Named ball colours (metadata-friendly) sampled per episode.
BALL_COLORS = {
    "orange": (0.95, 0.45, 0.10, 1.0),
    "green": (0.15, 0.65, 0.25, 1.0),
    "purple": (0.55, 0.20, 0.75, 1.0),
    "red": (0.85, 0.15, 0.12, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
    "yellow": (0.93, 0.82, 0.12, 1.0),
    "cyan": (0.10, 0.72, 0.78, 1.0),
    "magenta": (0.86, 0.16, 0.62, 1.0),
}


def sample_episode(rng: np.random.Generator, variant: str, seed: int, *, fps: int, duration: float) -> EpisodeSample:
    """Family F1 / subfamily F1_A_centered_vertical_drop.

    The robot is mounted on a table (base z=0.74) in every environment so the
    catch happens at a natural ~1.2 m height. The ball is released at t=0 from a
    high drop point and falls (near) vertically into the catch zone; the arm
    reacts. See ``controller.MujocoCatchController``.
    """
    # Ball must fit inside the Panda gripper (max pad gap ~0.08 m) for a real pinch.
    radius = float(rng.uniform(0.022, 0.030))
    density = float(rng.uniform(60.0, 260.0))  # ~foam..rubber -> mass ~0.005-0.03 kg
    mass = sphere_mass(radius, density)
    friction = float(rng.uniform(0.5, 0.9))
    restitution = float(rng.uniform(0.12, 0.32))

    height_offset = TABLETOP_ROBOT_Z  # all environments are tabletop-mounted
    catch_center_z = float(rng.uniform(0.44, 0.54)) + height_offset
    # Drop height (m above the catch point). Kept modest so the ball enters near
    # the TOP of the (higher, downward-looking) camera frame and is visible for its
    # whole ~0.4-0.5 s fall into the hand -- earlier 1.15-1.55 m drops spawned the
    # ball well above frame and it only flicked into view for the last instant.
    drop_height = float(rng.uniform(0.75, 1.05))

    # Centered vertical drop: XY sits in the arm's reachable catch zone, tight
    # spread; velocity ~0 (a little jitter/spin for variety).
    x = float(rng.uniform(0.42, 0.50))
    y = float(rng.uniform(-0.05, 0.05))
    z = catch_center_z + drop_height
    vx = float(rng.normal(0.0, 0.03))
    vy = float(rng.normal(0.0, 0.03))
    vz = float(rng.uniform(-0.06, 0.0))  # small downward toss allowed
    spin = tuple(float(v) for v in rng.uniform(-2.5, 2.5, size=3))

    color_name = str(rng.choice(list(BALL_COLORS.keys())))
    color = BALL_COLORS[color_name]
    camera_jitter = tuple(float(v) for v in rng.normal(0.0, (0.04, 0.04, 0.03)))
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
        ball_color_name=color_name,
        ball_initial_position=(x, y, z),
        ball_initial_velocity=(vx, vy, vz),
        ball_initial_angular_velocity=spin,
        ball_friction=friction,
        ball_restitution=restitution,
        drop_height=drop_height,
        lighting_intensity=float(rng.uniform(0.80, 1.22)),
        camera_jitter=camera_jitter,
        floor_material_jitter=float(rng.uniform(-0.04, 0.04)),
        robot_base_position=(0.0, 0.0, TABLETOP_ROBOT_Z),
        robot_base_euler=(0.0, 0.0, TABLETOP_ROBOT_YAW),
        tabletop_height=TABLETOP_ROBOT_Z,
        catch_center_z=catch_center_z,
        clutter_seed=int(rng.integers(0, 2**31 - 1)),
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


def _add_ball(root: ET.Element, sample: EpisodeSample) -> None:
    asset = _ensure_child(root, "asset")
    add_material(asset, "catch_ball_mat", sample.ball_color, roughness=0.45)
    world = _ensure_child(root, "worldbody")
    body = ET.SubElement(world, "body", name="catch_ball", pos=" ".join(f"{v:.6f}" for v in sample.ball_initial_position))
    ET.SubElement(body, "freejoint", name="ball_freejoint")
    # Restitution ~ bounciness: map to a slightly under-damped contact solref.
    solref_damp = float(np.clip(1.05 - sample.ball_restitution, 0.55, 1.0))
    ET.SubElement(
        body,
        "geom",
        name="catch_ball_geom",
        type="sphere",
        size=f"{sample.ball_radius:.6f}",
        mass=f"{sample.ball_mass:.8f}",
        material="catch_ball_mat",
        condim="4",
        friction=f"{sample.ball_friction:.4f} 0.01 0.001",
        solref=f"0.006 {solref_damp:.3f}",
        solimp="0.92 0.99 0.001",
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

    visual_body = ET.SubElement(
        hand,
        "body",
        name=WRIST_CAMERA_VISUAL_BODY,
        pos=pos,
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
    cams = add_variant_xml(
        root,
        sample.scene_variant,
        lighting_intensity=sample.lighting_intensity,
        floor_jitter=sample.floor_material_jitter,
        camera_jitter=sample.camera_jitter,
        catch_center_z=sample.catch_center_z,
        clutter_seed=sample.clutter_seed,
    )
    background_asset_ids = cams.pop("_background_asset_ids", [])
    cams[WRIST_CAMERA_NAME] = _add_wrist_camera(root)
    _add_ball(root, sample)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return EpisodeBundle(model=model, data=data, xml=xml, asset_info=asset_info,
                         camera_pose=cams, background_asset_ids=background_asset_ids)
