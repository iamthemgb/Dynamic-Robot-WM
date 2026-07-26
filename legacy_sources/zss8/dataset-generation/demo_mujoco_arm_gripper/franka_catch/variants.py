from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .robocasa_assets import (
    append_robocasa_visual_model,
    discover_fixture_models,
    discover_object_models,
    robocasa_assets_available,
    robocasa_assets_root,
)
from .utils import camera_xyaxes


# Environments used for the catch dataset: two "clean lab/office" style rooms and
# the RoboCasa kitchen. (Other studio/warehouse variants remain available.)
VARIANT_NAMES = ("clean_lab", "office", "robocasa_kitchen")
ALL_VARIANT_NAMES = ("white_studio", "clean_lab", "office", "warehouse", "cluttered_lab", "robocasa_kitchen")


def _camera_from_angle(target, theta_deg: float, el_deg: float, radius: float):
    """Camera position at azimuth ``theta_deg`` (0 = straight front, i.e. -y) and
    elevation ``el_deg`` on a sphere of ``radius`` around ``target``. Positive
    theta swings to screen-right, negative to screen-left."""
    th = math.radians(theta_deg)
    el = math.radians(el_deg)
    ce = math.cos(el)
    x = target[0] + radius * math.sin(th) * ce
    y = target[1] - radius * math.cos(th) * ce
    z = target[2] + radius * math.sin(el)
    return (x, y, z)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    floor_rgba: tuple[float, float, float, float]
    wall_rgba: tuple[float, float, float, float]
    camera_pos: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]
    fovy: float
    light_pos: tuple[float, float, float]
    light_diffuse: tuple[float, float, float]


VARIANTS = {
    "white_studio": VariantSpec("white_studio", (0.88, 0.88, 0.86, 1), (0.96, 0.96, 0.94, 1), (1.25, -1.65, 1.02), (0.42, 0.0, 0.54), 40.0, (0.2, -0.4, 2.7), (0.8, 0.8, 0.8)),
    "clean_lab": VariantSpec("clean_lab", (0.62, 0.66, 0.66, 1), (0.82, 0.86, 0.86, 1), (1.38, -1.78, 1.10), (0.42, 0.0, 0.54), 42.0, (0.1, -0.5, 2.8), (0.95, 0.95, 0.9)),
    "office": VariantSpec("office", (0.48, 0.48, 0.44, 1), (0.75, 0.73, 0.68, 1), (1.48, -1.88, 1.12), (0.42, 0.0, 0.55), 44.0, (0.4, -0.6, 2.5), (0.78, 0.76, 0.70)),
    "warehouse": VariantSpec("warehouse", (0.38, 0.39, 0.38, 1), (0.47, 0.48, 0.47, 1), (1.55, -1.95, 1.18), (0.43, 0.0, 0.55), 46.0, (0.0, -0.8, 2.5), (0.58, 0.58, 0.55)),
    "cluttered_lab": VariantSpec("cluttered_lab", (0.60, 0.63, 0.63, 1), (0.80, 0.85, 0.85, 1), (1.42, -1.82, 1.10), (0.42, 0.0, 0.54), 43.0, (0.0, -0.5, 2.8), (0.9, 0.94, 0.9)),
    "robocasa_kitchen": VariantSpec("robocasa_kitchen", (0.60, 0.58, 0.54, 1), (0.88, 0.88, 0.83, 1), (1.86, -0.48, 1.58), (0.44, 0.00, 1.36), 48.0, (0.1, -0.45, 2.75), (0.95, 0.92, 0.84)),
}


def rgba_str(rgba) -> str:
    return " ".join(f"{float(v):.4f}" for v in rgba)


def add_material(asset, name: str, rgba, roughness: float = 0.85) -> None:
    ET.SubElement(asset, "material", name=name, rgba=rgba_str(rgba), specular="0.18", shininess=str(max(0.02, 1.0 - roughness)))


def add_textured_material(asset, name: str, rel_texture: str, *, rgba=(1, 1, 1, 1), texrepeat=(1, 1), roughness: float = 0.75) -> None:
    texture_path = robocasa_assets_root() / rel_texture
    if not texture_path.exists():
        add_material(asset, name, rgba, roughness=roughness)
        return
    texture_name = f"{name}_texture"
    ET.SubElement(asset, "texture", name=texture_name, type="2d", file=str(texture_path.resolve()))
    ET.SubElement(
        asset,
        "material",
        name=name,
        texture=texture_name,
        texrepeat=f"{float(texrepeat[0]):.3f} {float(texrepeat[1]):.3f}",
        rgba=rgba_str(rgba),
        specular="0.16",
        shininess=str(max(0.02, 1.0 - roughness)),
    )


def add_box(world, name: str, pos, size, material: str, group: int = 0) -> None:
    ET.SubElement(
        world,
        "geom",
        name=name,
        type="box",
        pos=" ".join(f"{v:.4f}" for v in pos),
        size=" ".join(f"{v:.4f}" for v in size),
        material=material,
        group=str(group),
        condim="3",
        friction="1.2 0.005 0.0001",
    )


def add_visual_box(world, name: str, pos, size, material: str, group: int = 1) -> None:
    ET.SubElement(
        world,
        "geom",
        name=name,
        type="box",
        pos=" ".join(f"{v:.4f}" for v in pos),
        size=" ".join(f"{v:.4f}" for v in size),
        material=material,
        group=str(group),
        contype="0",
        conaffinity="0",
    )


def _add_handle(world, name: str, pos, *, vertical: bool = False) -> None:
    size = (0.006, 0.010, 0.055) if vertical else (0.060, 0.010, 0.006)
    add_visual_box(world, name, pos, size, "handle_metal")


def _configure_robocasa_lighting(root: ET.Element, world: ET.Element, lighting_intensity: float) -> None:
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.30 0.30 0.27")
    headlight.set("diffuse", "0.42 0.40 0.35")
    headlight.set("specular", "0.08 0.08 0.07")

    for light in world.findall("light"):
        if light.get("name") == "top":
            light.attrib.pop("mode", None)
            light.set("pos", "0.10 -0.55 2.85")
            light.set("diffuse", "0.34 0.32 0.29")
            light.set("ambient", "0.14 0.14 0.12")
            light.set("specular", "0.04 0.04 0.04")
            light.set("castshadow", "false")

    def scaled(rgb) -> str:
        return " ".join(f"{min(1.0, float(v) * lighting_intensity):.4f}" for v in rgb)

    ET.SubElement(
        world,
        "light",
        name="kitchen_softbox_key",
        pos="-0.65 -0.95 2.45",
        diffuse=scaled((1.00, 0.94, 0.82)),
        ambient="0.16 0.16 0.14",
        specular="0.12 0.11 0.10",
        directional="false",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "light",
        name="kitchen_camera_fill",
        pos="0.90 -1.45 1.55",
        diffuse=scaled((0.54, 0.55, 0.58)),
        ambient="0.10 0.10 0.10",
        specular="0.04 0.04 0.04",
        directional="false",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "light",
        name="kitchen_wall_fill",
        pos="-0.80 0.90 1.95",
        diffuse=scaled((0.46, 0.42, 0.36)),
        ambient="0.08 0.08 0.07",
        specular="0.03 0.03 0.03",
        directional="false",
        castshadow="false",
    )


def _add_robocasa_kitchen(root: ET.Element, world: ET.Element, clutter_seed: int = 0) -> None:
    add_box(world, "robot_table_top", (0.05, -0.08, 0.70), (0.84, 0.72, 0.04), "countertop")
    add_box(world, "robot_table_front", (0.05, -0.78, 0.38), (0.84, 0.025, 0.31), "cabinet_front")
    add_box(world, "robot_table_left_side", (-0.79, -0.08, 0.38), (0.025, 0.70, 0.31), "cabinet_front")
    add_box(world, "robot_table_right_side", (0.89, -0.08, 0.38), (0.025, 0.70, 0.31), "cabinet_front")
    add_visual_box(world, "robot_table_toe_kick", (0.05, -0.815, 0.09), (0.78, 0.012, 0.035), "drawer_gap")
    for i, x in enumerate((-0.48, -0.16, 0.16, 0.48)):
        add_visual_box(world, f"robot_table_drawer_{i}", (x, -0.809, 0.53), (0.145, 0.010, 0.080), "cabinet_front")
        _add_handle(world, f"robot_table_drawer_handle_{i}", (x, -0.822, 0.555))
        add_visual_box(world, f"robot_table_cabinet_{i}", (x, -0.809, 0.30), (0.145, 0.010, 0.150), "cabinet_front")
        _add_handle(world, f"robot_table_cabinet_handle_{i}", (x + 0.055, -0.822, 0.305), vertical=True)

    add_box(world, "back_counter_top", (0.33, 0.64, 0.70), (1.08, 0.24, 0.04), "countertop")
    add_box(world, "back_counter_base", (0.33, 0.70, 0.36), (1.08, 0.18, 0.32), "cabinet_front")
    add_visual_box(world, "back_counter_toe_kick", (0.33, 0.505, 0.095), (1.02, 0.012, 0.035), "drawer_gap")
    for i, x in enumerate((-0.48, -0.12, 0.24, 0.60, 0.92)):
        add_visual_box(world, f"back_lower_panel_{i}", (x, 0.496, 0.36), (0.145, 0.010, 0.215), "cabinet_front")
        _add_handle(world, f"back_lower_handle_{i}", (x + 0.055, 0.483, 0.365), vertical=True)

    add_visual_box(world, "backsplash_main", (0.33, 1.063, 0.98), (1.10, 0.010, 0.255), "backsplash_tile")
    add_visual_box(world, "backsplash_left_return", (-0.926, 0.46, 0.98), (0.010, 0.56, 0.255), "backsplash_tile")
    add_visual_box(world, "under_cabinet_light_l", (-0.34, 1.025, 1.285), (0.30, 0.012, 0.010), "warm_light")
    add_visual_box(world, "under_cabinet_light_r", (0.38, 1.025, 1.285), (0.30, 0.012, 0.010), "warm_light")

    for i, x in enumerate((-0.52, -0.16, 0.20, 0.56)):
        add_box(world, f"upper_cabinet_{i}", (x, 0.99, 1.58), (0.165, 0.065, 0.255), "cabinet_front")
        add_visual_box(world, f"upper_cabinet_door_l_{i}", (x - 0.043, 0.918, 1.58), (0.070, 0.010, 0.225), "cabinet_front")
        add_visual_box(world, f"upper_cabinet_door_r_{i}", (x + 0.043, 0.918, 1.58), (0.070, 0.010, 0.225), "cabinet_front")
        _add_handle(world, f"upper_cabinet_handle_l_{i}", (x - 0.012, 0.905, 1.51), vertical=True)
        _add_handle(world, f"upper_cabinet_handle_r_{i}", (x + 0.012, 0.905, 1.51), vertical=True)

    add_visual_box(world, "right_open_shelf_back", (1.16, 0.92, 1.40), (0.19, 0.015, 0.28), "accent_wood")
    for i, z in enumerate((1.22, 1.40, 1.58)):
        add_visual_box(world, f"right_open_shelf_{i}", (1.16, 0.72, z), (0.20, 0.18, 0.012), "accent_wood")
    add_visual_box(world, "window_glass", (-0.66, 1.055, 1.55), (0.22, 0.008, 0.18), "window_glass")
    add_visual_box(world, "window_frame_top", (-0.66, 1.045, 1.74), (0.245, 0.014, 0.010), "handle_metal")
    add_visual_box(world, "window_frame_bottom", (-0.66, 1.045, 1.36), (0.245, 0.014, 0.010), "handle_metal")
    add_visual_box(world, "window_frame_left", (-0.905, 1.045, 1.55), (0.010, 0.014, 0.19), "handle_metal")
    add_visual_box(world, "window_frame_right", (-0.415, 1.045, 1.55), (0.010, 0.014, 0.19), "handle_metal")
    add_visual_box(world, "left_window_glass", (-0.928, -0.26, 1.52), (0.008, 0.27, 0.22), "window_glass")
    add_visual_box(world, "left_window_frame_top", (-0.918, -0.26, 1.745), (0.012, 0.295, 0.010), "handle_metal")
    add_visual_box(world, "left_window_frame_bottom", (-0.918, -0.26, 1.295), (0.012, 0.295, 0.010), "handle_metal")
    add_visual_box(world, "left_window_frame_front", (-0.918, -0.555, 1.52), (0.012, 0.010, 0.225), "handle_metal")
    add_visual_box(world, "left_window_frame_back", (-0.918, 0.035, 1.52), (0.012, 0.010, 0.225), "handle_metal")
    add_visual_box(world, "procedural_range_hood_body", (0.28, 0.69, 1.42), (0.34, 0.11, 0.055), "metal")
    add_visual_box(world, "procedural_range_hood_lip", (0.28, 0.53, 1.34), (0.40, 0.04, 0.025), "handle_metal")
    add_visual_box(world, "procedural_range_hood_stack", (0.28, 0.93, 1.70), (0.16, 0.06, 0.24), "metal")
    add_visual_box(world, "kitchen_ceiling", (0.45, 0.10, 2.55), (1.90, 1.12, 0.025), "scene_wall")

    if not robocasa_assets_available():
        add_box(world, "missing_robocasa_hint_appliance", (0.46, 0.54, 0.86), (0.10, 0.08, 0.12), "dark_mat")
        add_box(world, "missing_robocasa_hint_object", (0.78, 0.45, 0.80), (0.08, 0.05, 0.08), "prop_red")
        return

    # Real RoboCasa fixtures/objects, discovered at runtime and imported as fixed
    # visual-only geometry (the catch dynamics stay in our own scene). Instance
    # IDs are sampled so kitchens vary clip to clip. Slots pick from a category
    # pool and place at a fixed spot on the back counters / walls.
    rng = np.random.default_rng(clutter_seed ^ 0x0BB1E)
    imported_ids: list[str] = []

    def place(category, pos, euler, is_object=False):
        models = (discover_object_models(rng, 1) if is_object
                  else discover_fixture_models(category, rng, 1))
        if not models:
            return
        rel = models[0]
        inst = "rc_" + rel.split("/")[-2].lower()
        # unique-ify instance if the same folder gets picked twice
        base = inst
        k = 1
        while inst in imported_ids:
            inst = f"{base}_{k}"; k += 1
        if append_robocasa_visual_model(root, rel, instance=inst, pos=pos, euler=euler):
            imported_ids.append(inst)

    # Large fixtures against the back / side.
    place("coffee_machines", (0.83, 0.55, 0.92), (0.0, 0.0, float(rng.uniform(-0.2, 0.2))))
    place("microwaves", (-0.62, 0.78, 0.94), (0.0, 0.0, 0.0))
    place("fridges", (-0.88, 0.82, 0.96), (0.0, 0.0, 0.0))
    place("sinks", (-0.56, 0.56, 0.93), (0.0, 0.0, 0.0))
    place("ovens", (0.28, 0.70, 1.02), (0.0, 0.0, 0.0))
    place("dishwashers", (-0.12, 0.69, 0.59), (0.0, 0.0, 0.0))
    place("toasters", (0.62, 0.40, 0.80), (0.0, 0.0, float(rng.uniform(-0.4, 0.4))))
    for cat in ("blenders", "stand_mixers", "electric_kettles"):
        if rng.random() < 0.6:
            place(cat, (float(rng.uniform(0.3, 0.9)), 0.50, 1.00), (0.0, 0.0, float(rng.uniform(-0.3, 0.3))))
    # A few small countertop objects for extra variety.
    for i, x in enumerate(rng.uniform(0.55, 1.05, size=int(rng.integers(1, 4)))):
        place(None, (float(x), float(rng.uniform(0.25, 0.42)), 0.80),
              (0.0, 0.0, float(rng.uniform(-0.5, 0.5))), is_object=True)
    return imported_ids


def _add_robot_pedestal(world: ET.Element) -> None:
    """A simple table/podium under the tabletop-mounted Franka (non-kitchen)."""
    add_box(world, "robot_pedestal_top", (0.06, -0.02, 0.70), (0.42, 0.40, 0.04), "table_mat")
    for i, (x, y) in enumerate(((-0.30, -0.30), (0.42, -0.30), (-0.30, 0.34), (0.42, 0.34))):
        add_box(world, f"robot_pedestal_leg_{i}", (x, y, 0.35), (0.03, 0.03, 0.35), "metal")


def add_variant_xml(root, variant_name: str, *, lighting_intensity: float, floor_jitter: float, camera_jitter,
                    catch_center_z: float = 1.24, clutter_seed: int = 0, cam_center_dz: float = 0.20) -> dict:
    spec = VARIANTS[variant_name]
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")

    floor_rgba = tuple(min(1.0, max(0.0, c + floor_jitter)) if i < 3 else c for i, c in enumerate(spec.floor_rgba))
    if variant_name == "robocasa_kitchen":
        ET.SubElement(
            asset,
            "texture",
            name="kitchen_skybox",
            type="skybox",
            builtin="gradient",
            rgb1="0.84 0.85 0.82",
            rgb2="0.66 0.66 0.62",
            width="256",
            height="256",
        )
    if variant_name == "robocasa_kitchen" and robocasa_assets_available():
        add_textured_material(asset, "scene_floor", "textures/wood/warm_wood_planks.png", rgba=floor_rgba, texrepeat=(3.6, 3.6), roughness=0.70)
        add_textured_material(asset, "scene_wall", "textures/flat/warm_white_2.png", rgba=spec.wall_rgba, texrepeat=(2.2, 2.2), roughness=0.82)
    else:
        add_material(asset, "scene_floor", floor_rgba)
        add_material(asset, "scene_wall", spec.wall_rgba)
    add_material(asset, "table_mat", (0.52, 0.48, 0.42, 1))
    add_material(asset, "dark_mat", (0.08, 0.09, 0.10, 1))
    add_material(asset, "prop_blue", (0.12, 0.32, 0.76, 1))
    add_material(asset, "prop_red", (0.72, 0.18, 0.12, 1))
    add_material(asset, "prop_green", (0.10, 0.52, 0.28, 1))
    add_material(asset, "cardboard", (0.55, 0.39, 0.22, 1))
    add_material(asset, "metal", (0.25, 0.27, 0.28, 1), roughness=0.45)
    if variant_name == "robocasa_kitchen" and robocasa_assets_available():
        add_textured_material(asset, "countertop", "textures/marble/granite_2.png", rgba=(0.82, 0.80, 0.76, 1), texrepeat=(2.8, 1.4), roughness=0.50)
        add_textured_material(asset, "cabinet_front", "textures/wood/warm_wood_grain_2.png", rgba=(0.86, 0.78, 0.66, 1), texrepeat=(1.4, 1.4), roughness=0.64)
        add_textured_material(asset, "backsplash_tile", "textures/tiles/white_square_tiles.png", rgba=(0.98, 0.98, 0.94, 1), texrepeat=(5.5, 1.4), roughness=0.58)
        add_textured_material(asset, "accent_wood", "textures/wood/dark_wood_planks.png", rgba=(0.72, 0.58, 0.42, 1), texrepeat=(1.2, 1.2), roughness=0.68)
    else:
        add_material(asset, "countertop", (0.50, 0.48, 0.42, 1), roughness=0.55)
        add_material(asset, "cabinet_front", (0.76, 0.70, 0.60, 1), roughness=0.70)
        add_material(asset, "backsplash_tile", (0.92, 0.90, 0.84, 1), roughness=0.60)
        add_material(asset, "accent_wood", (0.50, 0.36, 0.22, 1), roughness=0.68)
    add_material(asset, "drawer_gap", (0.055, 0.048, 0.042, 1), roughness=0.80)
    add_material(asset, "handle_metal", (0.55, 0.52, 0.47, 1), roughness=0.35)
    add_material(asset, "warm_light", (1.0, 0.82, 0.48, 1), roughness=0.25)
    add_material(asset, "window_glass", (0.45, 0.62, 0.78, 0.55), roughness=0.15)

    ET.SubElement(world, "geom", name="floor", type="plane", size="4.0 4.0 0.05", material="scene_floor", condim="3", friction="1.2 0.005 0.0001")
    # Enclose the room on all four sides so the closer oblique cameras never see
    # past a backdrop edge into the void. The two front cameras sit inside the box
    # (x in ~[-0.8, 1.4], y ~ -1.0), looking toward +y; the front wall is behind
    # them (won't occlude) and the right wall fills what the front-left camera
    # used to see as black void. Walls are tall (to z=2.8) and overlap at the
    # corners.
    add_box(world, "back_wall", (0.30, 1.55, 1.40), (2.7, 0.03, 1.40), "scene_wall")
    add_box(world, "left_wall", (-1.85, 0.10, 1.40), (0.03, 1.70, 1.40), "scene_wall")
    add_box(world, "right_wall", (1.95, 0.00, 1.40), (0.03, 1.80, 1.40), "scene_wall")
    add_box(world, "front_wall", (0.30, -1.85, 1.40), (2.7, 0.03, 1.40), "scene_wall")

    background_asset_ids: list[str] = []
    if variant_name != "robocasa_kitchen":
        _add_robot_pedestal(world)

    if variant_name in {"clean_lab", "cluttered_lab"}:
        add_box(world, "lab_workbench", (0.90, 0.48, 0.36), (0.42, 0.24, 0.04), "table_mat")
        add_box(world, "lab_bench_leg_a", (0.58, 0.34, 0.18), (0.025, 0.025, 0.18), "metal")
        add_box(world, "lab_bench_leg_b", (1.22, 0.62, 0.18), (0.025, 0.025, 0.18), "metal")
        if variant_name == "cluttered_lab":
            add_box(world, "lab_prop_blue", (0.72, 0.45, 0.46), (0.035, 0.035, 0.07), "prop_blue")
            add_box(world, "lab_prop_red", (0.86, 0.43, 0.45), (0.045, 0.025, 0.06), "prop_red")
            add_box(world, "lab_prop_green", (1.03, 0.50, 0.48), (0.04, 0.04, 0.09), "prop_green")
    elif variant_name == "office":
        add_box(world, "office_desk", (0.92, 0.50, 0.35), (0.46, 0.25, 0.04), "table_mat")
        add_box(world, "office_monitor", (0.93, 0.68, 0.58), (0.20, 0.018, 0.13), "dark_mat")
        add_box(world, "office_cabinet", (-0.52, 0.72, 0.40), (0.16, 0.18, 0.40), "metal")
    elif variant_name == "warehouse":
        add_box(world, "warehouse_shelf_low", (0.95, 0.78, 0.45), (0.48, 0.04, 0.035), "metal")
        add_box(world, "warehouse_shelf_high", (0.95, 0.78, 0.76), (0.48, 0.04, 0.035), "metal")
        for i, x in enumerate((0.65, 0.90, 1.15)):
            add_box(world, f"warehouse_box_{i}", (x, 0.70, 0.16), (0.10, 0.09, 0.12), "cardboard")
    elif variant_name == "robocasa_kitchen":
        background_asset_ids = _add_robocasa_kitchen(root, world, clutter_seed) or []

    if variant_name == "robocasa_kitchen":
        _configure_robocasa_lighting(root, world, lighting_intensity)
    else:
        visual = root.find("visual") or ET.SubElement(root, "visual")
        headlight = visual.find("headlight") or ET.SubElement(visual, "headlight")
        headlight.set("ambient", "0.34 0.34 0.34")
        headlight.set("diffuse", "0.45 0.45 0.45")
        headlight.set("specular", "0.10 0.10 0.10")
        light = ET.SubElement(
            world,
            "light",
            name="scene_key_light",
            pos=" ".join(f"{v:.4f}" for v in spec.light_pos),
            diffuse=" ".join(f"{min(1.0, v * lighting_intensity):.4f}" for v in spec.light_diffuse),
            specular="0.2 0.2 0.2",
            directional="false",
        )
        light.set("castshadow", "true")
        # Front fill from the camera side (-y) so the two oblique views aren't dark.
        ET.SubElement(
            world, "light", name="scene_fill_light", pos="0.30 -1.80 2.10",
            diffuse=" ".join(f"{min(1.0, 0.5 * lighting_intensity):.4f}" for _ in range(3)),
            specular="0.05 0.05 0.05", directional="false", castshadow="false",
        )

    # --- two synchronized oblique views framing the full catch + lift arc -----
    # Higher and looking gently DOWN (~26 deg elevation) so the arm's reach-in-depth,
    # the ball's descent, and the post-catch LIFT all read clearly. The look-at is
    # raised ``cam_center_dz`` above the catch point so the raised present/hold pose
    # stays in frame. (The earlier near-horizontal 7-14 deg framing on a nearly
    # static arm was why "you can't see anything".)
    rng = np.random.default_rng(clutter_seed ^ 0xCA7C)
    target = (0.30, 0.0, catch_center_z + cam_center_dz)
    theta_l = float(rng.uniform(-52.0, -44.0))
    theta_r = float(rng.uniform(44.0, 52.0))
    el_l = float(rng.uniform(23.0, 30.0))
    el_r = float(rng.uniform(23.0, 30.0))
    # Orbit far enough that the whole reach+lift+ball arc stays in the square frame,
    # matched between the two views for a clean stereo-ish look.
    r_l = float(rng.uniform(1.58, 1.74))
    r_r = float(rng.uniform(1.58, 1.74))
    fovy_l = float(rng.uniform(44.0, 48.0))
    fovy_r = float(rng.uniform(44.0, 48.0))

    cams = {}
    for name, theta, el, radius, fovy in (
        ("main_camera", theta_l, el_l, r_l, fovy_l),
        ("side_camera", theta_r, el_r, r_r, fovy_r),
    ):
        pos = _camera_from_angle(target, theta, el, radius)
        pos = tuple(float(p + j) for p, j in zip(pos, camera_jitter))
        ET.SubElement(
            world,
            "camera",
            name=name,
            pos=" ".join(f"{v:.5f}" for v in pos),
            xyaxes=camera_xyaxes(pos, target),
            fovy=f"{fovy:.3f}",
        )
        cams[name] = {"pos": list(pos), "lookat": list(target), "fovy": fovy,
                      "azimuth_deg": theta, "elevation_deg": el}
    cams["_background_asset_ids"] = background_asset_ids
    return cams
