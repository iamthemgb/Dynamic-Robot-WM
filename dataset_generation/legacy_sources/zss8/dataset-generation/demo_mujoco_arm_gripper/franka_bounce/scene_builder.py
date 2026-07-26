"""Scene sampling + build for subfamily F1_B_ground_bounce_catch.

The ball is released at t=0 from a high drop point, falls, **bounces once** off a
tabletop surface in front of the Franka, and rebounds up into the catch zone; the
arm waits at home and then reactively snatches the ball near the post-bounce apex
(see ``controller.BounceCatchController``). Restitution is varied per episode and
the drop height is chosen so the rebound apex still lands in the arm's reachable
band, so the world model must infer the bounce dynamics to time the catch.

Reuses the franka_catch scene helpers (robot base, options, wrist camera,
backgrounds/cameras) verbatim; only the ball contact tuning and the bounce
surface are specific to this subfamily.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from franka_catch.assets import locate_franka_asset
from franka_catch.scene_builder import (
    BALL_COLORS,
    WRIST_CAMERA_NAME,
    EpisodeBundle,
    EpisodeSample,
    _add_wrist_camera,
    _configure_options,
    _configure_robot_base,
    _ensure_child,
    _set_robot_compiler_paths,
)
from franka_catch.utils import sphere_mass
from franka_catch.variants import add_material, add_variant_xml

# Height of the tabletop the robot is mounted on / the ball bounces off (m).
# Matches robot_table_top / robot_pedestal_top (top face at 0.74) so the bounce
# pad is coplanar with the existing furniture in every variant.
BOUNCE_SURFACE_Z = 0.74
G = 9.81


def sample_bounce_episode(rng: np.random.Generator, variant: str, seed: int, *,
                          fps: int, duration: float) -> EpisodeSample:
    """Family F1 / subfamily F1_B_ground_bounce_catch.

    ``ball_restitution`` is the *nominal* coefficient of restitution used to (a)
    tune the ball's contact ``solref`` and (b) pick a drop height that puts the
    rebound apex in the reachable catch band. The controller measures the *actual*
    post-bounce velocity and catches accordingly, so it is robust to the sim's
    true restitution (recorded in metadata as ``measured_restitution``).
    """
    radius = float(rng.uniform(0.022, 0.030))
    density = float(rng.uniform(60.0, 260.0))
    mass = sphere_mass(radius, density)
    friction = float(rng.uniform(0.5, 0.9))
    restitution = float(rng.uniform(0.55, 0.85))  # lively, clearly-visible bounce

    height_offset = BOUNCE_SURFACE_Z
    # Intended catch height = post-bounce apex, kept in the arm's comfortable band.
    catch_center_z = float(rng.uniform(0.42, 0.54)) + height_offset  # ~1.16-1.28

    # Choose the drop height so a bounce with the nominal restitution rebounds to
    # ~catch_center_z:  rise = e^2 * drop  ->  drop = rise / e^2.
    z_contact = BOUNCE_SURFACE_Z + radius
    rise = max(0.12, catch_center_z - z_contact)
    drop_height = float(np.clip(rise / (restitution ** 2), 0.45, 1.85))

    # Drop XY sits over the bounce pad, in the arm's reach; near-vertical descent.
    x = float(rng.uniform(0.42, 0.50))
    y = float(rng.uniform(-0.05, 0.05))
    z = z_contact + drop_height
    vx = float(rng.normal(0.0, 0.04))
    vy = float(rng.normal(0.0, 0.04))
    vz = float(rng.uniform(-0.05, 0.0))
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
        gravity=(0.0, 0.0, -G),
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
        robot_base_position=(0.0, 0.0, BOUNCE_SURFACE_Z),
        robot_base_euler=(0.0, 0.0, 0.0),
        tabletop_height=BOUNCE_SURFACE_Z,
        catch_center_z=catch_center_z,
        clutter_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _add_bounce_surface(root: ET.Element, sample: EpisodeSample) -> None:
    """A firm tabletop pad (top face at BOUNCE_SURFACE_Z) covering the drop zone,
    present in every variant so the ball always has a surface to bounce off."""
    asset = _ensure_child(root, "asset")
    add_material(asset, "bounce_pad_mat", (0.52, 0.48, 0.42, 1.0), roughness=0.55)
    world = _ensure_child(root, "worldbody")
    half_z = 0.02
    # Visual-only: the scripted analytic bounce owns the ball's trajectory (a real
    # stiff contact would cancel the injected rebound velocity), so the pad is a
    # non-colliding table surface for the ball to visibly bounce off of.
    ET.SubElement(
        world,
        "geom",
        name="bounce_surface",
        type="box",
        pos=f"0.46 0.0 {BOUNCE_SURFACE_Z - half_z:.5f}",
        size=f"0.30 0.30 {half_z:.5f}",
        material="bounce_pad_mat",
        contype="0",
        conaffinity="0",
        group="0",
    )


def _add_bounce_ball(root: ET.Element, sample: EpisodeSample) -> None:
    """Ball with a stiff, near-inelastic contact.

    MuJoCo's soft-contact restitution is weak and hard to tune, so the natural
    landing is kept inelastic (stiff, critically damped -> the ball would just
    stop) and the controller injects the analytic rebound ``v_post = e*v_impact``
    at the impact instant (see ``controller.BounceCatchController._maybe_bounce``).
    The stiff contact prevents deep penetration if detection is a step late."""
    asset = _ensure_child(root, "asset")
    add_material(asset, "catch_ball_mat", sample.ball_color, roughness=0.45)
    world = _ensure_child(root, "worldbody")
    body = ET.SubElement(world, "body", name="catch_ball",
                         pos=" ".join(f"{v:.6f}" for v in sample.ball_initial_position))
    ET.SubElement(body, "freejoint", name="ball_freejoint")
    # contype/conaffinity = bit 2 only: the ball ignores every world surface
    # (floor, table, furniture) so the scripted bounce fully owns its trajectory.
    # The gripper collision geoms are given bit 2 as well (see
    # ``_isolate_ball_collisions``) so the catch still registers.
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
        solref="0.006 1.0",
        solimp="0.95 0.99 0.001",
        contype="2",
        conaffinity="2",
    )


# Gripper bodies whose collision geoms must also collide with the isolated ball.
_GRIPPER_BODIES = ("hand", "left_finger", "right_finger")


def _isolate_ball_collisions(root: ET.Element) -> None:
    """Give the gripper's collision geoms bit 2 (-> contype/conaffinity 3) so they
    collide with both the world (bit 1) and the isolated catch ball (bit 2)."""
    world = _ensure_child(root, "worldbody")
    for body in world.iter("body"):
        if body.get("name") not in _GRIPPER_BODIES:
            continue
        for geom in body.findall("geom"):
            cls = geom.get("class", "")
            if cls == "collision" or cls.startswith("fingertip_pad"):
                geom.set("contype", "3")
                geom.set("conaffinity", "3")


def build_bounce_episode(sample: EpisodeSample) -> EpisodeBundle:
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
        cam_center_dz=0.12,  # frame a touch lower so the ground bounce stays visible
    )
    background_asset_ids = cams.pop("_background_asset_ids", [])
    cams[WRIST_CAMERA_NAME] = _add_wrist_camera(root)
    _add_bounce_surface(root, sample)
    _add_bounce_ball(root, sample)
    _isolate_ball_collisions(root)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return EpisodeBundle(model=model, data=data, xml=xml, asset_info=asset_info,
                         camera_pose=cams, background_asset_ids=background_asset_ids)
