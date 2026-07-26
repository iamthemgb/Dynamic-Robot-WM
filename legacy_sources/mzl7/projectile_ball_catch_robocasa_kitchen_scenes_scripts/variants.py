from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance

from .robocasa_assets import append_robocasa_visual_model, robocasa_assets_available, style_model_rel_path
from .utils import camera_xyaxes
from .yaml_scene import ResolvedFixture, native_fixture_mjcf, resolve_scene, _style_color, _style_token


VARIANT_NAMES = ("white_studio", "clean_lab", "office", "warehouse", "cluttered_lab", "robocasa_kitchen")


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
    "robocasa_kitchen": VariantSpec("robocasa_kitchen", (0.60, 0.58, 0.54, 1), (0.88, 0.88, 0.83, 1), (2.22, -0.78, 1.52), (0.42, 0.02, 0.96), 60.0, (0.1, -0.45, 2.75), (0.95, 0.92, 0.84)),
}


def rgba_str(rgba) -> str:
    return " ".join(f"{float(v):.4f}" for v in rgba)


def add_material(asset: ET.Element, name: str, rgba, roughness: float = 0.85) -> None:
    ET.SubElement(asset, "material", name=name, rgba=rgba_str(rgba), specular="0.18", shininess=str(max(0.02, 1.0 - roughness)))


def add_box(world: ET.Element, name: str, pos, size, material: str, *, yaw: float = 0.0, group: int = 0, visual_only: bool = False) -> None:
    attrib = {
        "name": name,
        "type": "box",
        "pos": " ".join(f"{float(v):.4f}" for v in pos),
        "size": " ".join(f"{float(v):.4f}" for v in size),
        "material": material,
        "group": str(group),
    }
    if abs(float(yaw)) > 1e-9:
        attrib["euler"] = f"0 0 {float(yaw):.6f}"
    if visual_only:
        attrib["contype"] = "0"
        attrib["conaffinity"] = "0"
    else:
        attrib["condim"] = "3"
        attrib["friction"] = "1.2 0.005 0.0001"
    ET.SubElement(world, "geom", **attrib)


def _scaled_rgb(rgb, scale: float) -> str:
    return " ".join(f"{min(1.0, float(v) * scale):.4f}" for v in rgb)


def _scene_materials(style_config: dict) -> dict[str, tuple[float, float, float, float]]:
    return {
        "scene_floor": _style_color(_style_token(style_config, ("floor",), "floor_default"), min_value=0.20, max_value=0.78),
        "scene_wall": _style_color(_style_token(style_config, ("wall",), "wall_default"), min_value=0.55, max_value=0.95),
        "countertop": _style_color(_style_token(style_config, ("counter", "default"), "counter_default"), min_value=0.45, max_value=0.88),
        "countertop_island": _style_color(_style_token(style_config, ("counter", "island"), "counter_island_default"), min_value=0.45, max_value=0.88),
        "cabinet_front": _style_color(_style_token(style_config, ("cabinet", "default"), "cabinet_default"), min_value=0.35, max_value=0.80),
        "backsplash": _style_color(_style_token(style_config, ("wall",), "wall_default") + "_backsplash", min_value=0.65, max_value=0.95),
        "trim": _style_color(_style_token(style_config, ("cabinet", "default"), "cabinet_default") + "_trim", min_value=0.25, max_value=0.70),
        "glass": (0.55, 0.72, 0.84, 0.40),
        "metal": (0.28, 0.29, 0.31, 1.0),
        "dark": (0.12, 0.12, 0.13, 1.0),
    }


def _fixture_material_name(fixture: ResolvedFixture) -> str:
    if fixture.fixture_type in {"counter"}:
        return "countertop"
    if fixture.fixture_type in {"wall"}:
        return "scene_wall"
    if fixture.fixture_type in {"floor"}:
        return "scene_floor"
    if fixture.fixture_type in {"window", "window_proc"}:
        return "glass"
    if fixture.fixture_type in {"box"}:
        return "trim"
    return "cabinet_front"


def _draw_room_fixture(world: ET.Element, fixture: ResolvedFixture) -> None:
    if fixture.fixture_type == "wall":
        size = (fixture.half_size[0], 0.03, fixture.half_size[1])
        if abs(fixture.raw_config.get("wall_side") in {"left", "right"}) and False:
            pass
        wall_side = str(fixture.raw_config.get("wall_side", "back"))
        if wall_side in {"left", "right"}:
            size = (0.03, fixture.half_size[0], fixture.half_size[1])
        if not np.all(np.isfinite(np.asarray(size, dtype=np.float64))) or min(size) <= 1e-6:
            return
        add_box(world, fixture.name, fixture.pos, size, "scene_wall", visual_only=bool(fixture.raw_config.get("backing", False)))
    elif fixture.fixture_type == "floor":
        size = (fixture.half_size[0], fixture.half_size[1], max(0.02, fixture.half_size[2]))
        if not np.all(np.isfinite(np.asarray(size, dtype=np.float64))) or min(size) <= 1e-6:
            return
        add_box(world, fixture.name, (fixture.pos[0], fixture.pos[1], fixture.pos[2] - fixture.half_size[2]), size, "scene_floor", visual_only=bool(fixture.raw_config.get("backing", False)))


def _draw_fixture_box(world: ET.Element, fixture: ResolvedFixture) -> None:
    material_name = _fixture_material_name(fixture)
    pos = fixture.pos
    half_size = fixture.half_size
    visual_only = fixture.fixture_type in {"window", "window_proc", "wall_accessory", "stool", "accessory", "paper_towel", "candle", "turmeric", "cinnamon", "paprika", "flower_vase", "utensil_set", "plant", "knife_block", "soap_dispenser", "fruit_bowl", "oil_bottle", "vinegar_bottle", "salt_shaker", "pepper_shaker", "tiered_basket", "digital_scale", "glass_cup", "jar", "jar_lid", "utensil_holder", "utensil_rack"}
    min_extent = 5e-3 if visual_only else 1e-4
    if not np.all(np.isfinite(np.asarray(half_size, dtype=np.float64))) or min(half_size) <= min_extent:
        return
    if fixture.fixture_type in {"counter"}:
        top_half_height = 0.04
        base_half_height = max(0.10, half_size[2] - top_half_height)
        add_box(world, f"{fixture.name}_top", (pos[0], pos[1], pos[2] + half_size[2] - top_half_height), (half_size[0], half_size[1], top_half_height), "countertop", yaw=fixture.yaw)
        add_box(world, f"{fixture.name}_base", (pos[0], pos[1], pos[2] + base_half_height - half_size[2]), (max(0.05, half_size[0] - 0.04), max(0.05, half_size[1] - 0.04), base_half_height), "cabinet_front", yaw=fixture.yaw)
        return
    if fixture.fixture_type in {"window", "window_proc"}:
        add_box(world, fixture.name, pos, (half_size[0], 0.015, half_size[2]), material_name, yaw=fixture.yaw, visual_only=True)
        return
    add_box(world, fixture.name, pos, half_size, material_name, yaw=fixture.yaw, visual_only=visual_only)


def _append_style_models(root: ET.Element, fixtures: tuple[ResolvedFixture, ...]) -> None:
    if not robocasa_assets_available():
        return
    for fixture in fixtures:
        rel_path = style_model_rel_path(fixture.style_role, fixture.style_model)
        if rel_path is None:
            continue
        # Imported fixture models are visual-only and augment the YAML-driven boxes.
        append_robocasa_visual_model(
            root,
            rel_path,
            instance=fixture.name,
            pos=fixture.pos,
            euler=(0.0, 0.0, fixture.yaw),
        )


def _merge_asset_elements(asset_root: ET.Element, new_assets: list[ET.Element]) -> None:
    for asset in new_assets:
        name = asset.get("name")
        if name and any(existing.tag == asset.tag and existing.get("name") == name for existing in asset_root):
            continue
        asset_root.append(copy.deepcopy(asset))


def _desaturated_texture_copy(source_file: str, *, saturation: float) -> str:
    source_path = Path(source_file)
    if not source_path.exists():
        return source_file
    cache_dir = Path("/tmp/robocasa_texture_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_sat{int(round(saturation * 100)):03d}"
    output_path = cache_dir / f"{source_path.stem}{suffix}{source_path.suffix}"
    if output_path.exists():
        return str(output_path)
    with Image.open(source_path) as image:
        desaturated = ImageEnhance.Color(image).enhance(saturation)
        desaturated.save(output_path)
    return str(output_path)


def _retune_native_textures(asset_root: ET.Element) -> None:
    # Native RoboCasa textures can be too vivid in MuJoCo. Replace only the
    # dominant style textures with cached desaturated copies.
    for texture in asset_root.findall("texture"):
        file_value = texture.get("file")
        if not file_value:
            continue
        if file_value.endswith("/generative_textures/cabinet/tex027.png"):
            texture.set("file", _desaturated_texture_copy(file_value, saturation=0.42))
        elif file_value.endswith("/generative_textures/counter/tex059.png"):
            texture.set("file", _desaturated_texture_copy(file_value, saturation=0.55))
        elif file_value.endswith("/generative_textures/wall/tex058.png"):
            texture.set("file", _desaturated_texture_copy(file_value, saturation=0.72))


def _shift_body_position(body: ET.Element, offset: tuple[float, float, float]) -> None:
    pos = body.get("pos")
    if pos is None:
        return
    values = [float(value) for value in pos.split()]
    if len(values) != 3:
        return
    shifted = [values[0] + offset[0], values[1] + offset[1], values[2] + offset[2]]
    body.set("pos", " ".join(f"{value:.12g}" for value in shifted))


def _is_native_visual_geom(geom: ET.Element) -> bool:
    geom_class = geom.get("class")
    if geom_class == "visual":
        return True
    if geom_class in {"collision", "region", "spawn", "contact", "sensor"}:
        return False
    if geom.get("mesh"):
        return True
    if geom.get("material") and geom.get("contype", "0") == "0" and geom.get("conaffinity", "0") == "0":
        return True
    return False


def _strip_nonvisual_elements(elem: ET.Element) -> None:
    for child in list(elem):
        if child.tag == "site":
            elem.remove(child)
            continue
        if child.tag == "geom" and not _is_native_visual_geom(child):
            child.attrib.pop("material", None)
            child.set("rgba", "0 0 0 0")
            child.set("group", "3")
        _strip_nonvisual_elements(child)


def _apply_native_layout_overrides(native_bodies: list[ET.Element], layout_id: int, env_index: int | None = None) -> list[ET.Element]:
    adjusted: list[ET.Element] = []
    for body in native_bodies:
        copied = copy.deepcopy(body)
        _strip_nonvisual_elements(copied)
        name = copied.get("name", "")
        if env_index == 0 and name == "sink_left_group_main":
            copied.set("pos", "1.35 -0.30 0.958")
            copied.set("euler", "0 0 0")
        if layout_id != 58:
            adjusted.append(copied)
            continue
        if name in {"dishwasher_1_main_group_1_main", "stove_island_group_1_main", "stove_hood_island_group_1_main"}:
            continue
        island_offset = (0.0, 0.8, 0.0)
        if "_island_group_1_" in name:
            _shift_body_position(copied, island_offset)
        adjusted.append(copied)
    return adjusted


def _append_native_scene(root: ET.Element, world: ET.Element, asset: ET.Element, layout_id: int, style_id: int, env_index: int | None = None) -> None:
    native_assets, native_bodies = native_fixture_mjcf(layout_id, style_id)
    _merge_asset_elements(asset, native_assets)
    for body in _apply_native_layout_overrides(native_bodies, layout_id, env_index):
        world.append(body)
    if layout_id == 58:
        # Restore the missing island countertop span after removing the stove body.
        ET.SubElement(
            world,
            "geom",
            name="layout58_stove_counter_patch",
            type="box",
            pos="3.61025302083 -1.84069838252 0.89782904243",
            size="0.379004595316 0.316123825558 0.0186034471902",
            euler="0 0 3.14144134522",
            material="layout58_stove_counter_patch_mat",
            contype="0",
            conaffinity="0",
            group="1",
        )
        ET.SubElement(
            world,
            "geom",
            name="layout58_cabinet_wall_panel",
            type="box",
            pos="3.297 -0.010 1.500",
            size="1.669 0.005 0.750",
            material="wall_main_1_room_wall_mat",
            contype="0",
            conaffinity="0",
            group="1",
        )
    if env_index == 0 and layout_id == 11:
        ET.SubElement(
            world,
            "geom",
            name="seed0_old_sink_cover",
            type="box",
            pos="0.3000000621 -1.7962400201 0.905",
            euler="0 0 1.5707962926",
            size="0.44 0.325 0.015",
            material="counter_1_left_group_counter_top",
            contype="0",
            conaffinity="0",
            group="1",
        )


def _configure_lighting(root: ET.Element, world: ET.Element, lighting_intensity: float, room_bounds: tuple[float, float, float, float], lookat: tuple[float, float, float]) -> None:
    xmin, xmax, ymin, ymax = room_bounds
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", _scaled_rgb((0.26, 0.26, 0.25), lighting_intensity))
    headlight.set("diffuse", _scaled_rgb((0.44, 0.42, 0.39), lighting_intensity))
    headlight.set("specular", "0.08 0.08 0.08")
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)
    lights = [
        (center_x - 0.18 * span, center_y - 0.28 * span, 2.9, (0.95, 0.92, 0.86)),
        (center_x + 0.22 * span, center_y + 0.12 * span, 2.4, (0.62, 0.64, 0.68)),
        (lookat[0] - 0.15 * span, lookat[1] + 0.18 * span, 1.8, (0.48, 0.48, 0.46)),
    ]
    for index, (x, y, z, diffuse) in enumerate(lights):
        ET.SubElement(
            world,
            "light",
            name=f"yaml_scene_light_{index}",
            pos=f"{x:.4f} {y:.4f} {z:.4f}",
            diffuse=_scaled_rgb(diffuse, lighting_intensity),
            ambient="0.09 0.09 0.09",
            specular="0.03 0.03 0.03",
            directional="false",
            castshadow="false",
        )


def _camera_pose(scene, camera_jitter, env_index: int | None = None, camera_variant: str | None = None):
    xmin, xmax, ymin, ymax = scene.room_bounds
    placement = scene.placement
    center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
    span = max(xmax - xmin, ymax - ymin)
    launch_xy = np.asarray(placement.launch_position[:2], dtype=np.float64)
    catch_xyz = np.asarray(placement.catch_position, dtype=np.float64)
    lookat = (
        placement.catch_position[0],
        placement.catch_position[1],
        placement.tabletop_height + 0.24,
    )
    if env_index == 2350:
        stool_anchor = np.array(
            [
                placement.robot_base_position[0],
                placement.robot_base_position[1] - 1.08,
                placement.tabletop_height + 0.64,
            ],
            dtype=np.float64,
        )
        lookat = (
            float(placement.robot_base_position[0] + 0.02),
            float(placement.robot_base_position[1] + 0.10),
            float(placement.tabletop_height + 0.32),
        )
        cam_pos = (
            float(stool_anchor[0] + 0.10 * camera_jitter[0]),
            float(stool_anchor[1] + 0.10 * camera_jitter[1]),
            float(stool_anchor[2] + 0.08 * camera_jitter[2]),
        )
        closeup_pos = (
            placement.robot_base_position[0] - 0.12 * math.cos(placement.robot_base_yaw) - 0.42 * math.sin(placement.robot_base_yaw),
            placement.robot_base_position[1] - 0.12 * math.sin(placement.robot_base_yaw) + 0.42 * math.cos(placement.robot_base_yaw),
            placement.tabletop_height + 0.52,
        )
        closeup_lookat = (
            placement.catch_position[0],
            placement.catch_position[1],
            placement.catch_position[2] - 0.03,
        )
        return cam_pos, lookat, closeup_pos, closeup_lookat
    if scene.layout_id == 58:
        trajectory_mid = 0.5 * (launch_xy + catch_xyz[:2])
        lookat = (
            float(trajectory_mid[0] - 0.42),
            float(trajectory_mid[1] + 0.08),
            float(placement.tabletop_height + 0.30),
        )
        cam_pos = (
            float(trajectory_mid[0] + 0.92 + 0.12 * camera_jitter[0]),
            float(placement.robot_base_position[1] + 0.42 + 0.12 * camera_jitter[1]),
            float(placement.tabletop_height + 0.60 + 0.10 * camera_jitter[2]),
        )
        closeup_pos = (
            placement.robot_base_position[0] - 0.12 * math.cos(placement.robot_base_yaw) - 0.42 * math.sin(placement.robot_base_yaw),
            placement.robot_base_position[1] - 0.12 * math.sin(placement.robot_base_yaw) + 0.42 * math.cos(placement.robot_base_yaw),
            placement.tabletop_height + 0.52,
        )
        closeup_lookat = (
            placement.catch_position[0],
            placement.catch_position[1],
            placement.catch_position[2] - 0.03,
        )
        return cam_pos, lookat, closeup_pos, closeup_lookat
    if env_index == 0:
        trajectory_mid = 0.5 * (launch_xy + catch_xyz[:2])
        if camera_variant == "opposite_counter_side":
            lookat = (
                float(placement.robot_base_position[0] + 0.02),
                float(trajectory_mid[1] + 0.10),
                float(placement.tabletop_height + 0.22),
            )
            cam_pos = (
                float(trajectory_mid[0] - 1.10 + 0.08 * camera_jitter[0]),
                float(trajectory_mid[1] + 0.44 + 0.08 * camera_jitter[1]),
                float(placement.tabletop_height + 0.92 + 0.08 * camera_jitter[2]),
            )
            closeup_pos = (
                placement.robot_base_position[0] - 0.44,
                placement.robot_base_position[1] + 0.20,
                placement.tabletop_height + 0.48,
            )
            closeup_lookat = (
                placement.catch_position[0],
                placement.catch_position[1],
                placement.catch_position[2] - 0.04,
            )
            return cam_pos, lookat, closeup_pos, closeup_lookat
        lookat = (
            float(trajectory_mid[0] + 0.02),
            float(trajectory_mid[1] + 0.12),
            float(placement.tabletop_height + 0.28),
        )
        cam_pos = (
            float(trajectory_mid[0] + 1.18 + 0.08 * camera_jitter[0]),
            float(trajectory_mid[1] - 0.44 + 0.08 * camera_jitter[1]),
            float(placement.tabletop_height + 0.72 + 0.08 * camera_jitter[2]),
        )
        closeup_pos = (
            placement.robot_base_position[0] + 0.44,
            placement.robot_base_position[1] - 0.20,
            placement.tabletop_height + 0.48,
        )
        closeup_lookat = (
            placement.catch_position[0],
            placement.catch_position[1],
            placement.catch_position[2] - 0.04,
        )
        return cam_pos, lookat, closeup_pos, closeup_lookat
    if env_index == 1000:
        trajectory_mid = 0.5 * (launch_xy + catch_xyz[:2])
        if camera_variant == "opposite_counter_side":
            lookat = (
                float(placement.robot_base_position[0] + 0.02),
                float(trajectory_mid[1] + 0.10),
                float(placement.tabletop_height + 0.28),
            )
            cam_pos = (
                float(trajectory_mid[0] - 1.42 + 0.08 * camera_jitter[0]),
                float(trajectory_mid[1] + 0.92 + 0.08 * camera_jitter[1]),
                float(placement.tabletop_height + 0.90 + 0.08 * camera_jitter[2]),
            )
            closeup_pos = (
                placement.robot_base_position[0] - 0.42,
                placement.robot_base_position[1] + 0.28,
                placement.tabletop_height + 0.50,
            )
            closeup_lookat = (
                placement.catch_position[0],
                placement.catch_position[1],
                placement.catch_position[2] - 0.04,
            )
            return cam_pos, lookat, closeup_pos, closeup_lookat
        lookat = (
            float(trajectory_mid[0] + 0.02),
            float(trajectory_mid[1] + 0.10),
            float(placement.tabletop_height + 0.28),
        )
        cam_pos = (
            float(trajectory_mid[0] + 1.18 + 0.08 * camera_jitter[0]),
            float(trajectory_mid[1] - 0.68 + 0.08 * camera_jitter[1]),
            float(placement.tabletop_height + 0.80 + 0.08 * camera_jitter[2]),
        )
        closeup_pos = (
            placement.robot_base_position[0] + 0.42,
            placement.robot_base_position[1] - 0.28,
            placement.tabletop_height + 0.50,
        )
        closeup_lookat = (
            placement.catch_position[0],
            placement.catch_position[1],
            placement.catch_position[2] - 0.04,
        )
        return cam_pos, lookat, closeup_pos, closeup_lookat
    inset = max(0.35, 0.16 * span)
    def _clamp_inside(value: float, lower: float, upper: float) -> float:
        if lower > upper:
            return 0.5 * (lower + upper)
        return min(max(value, lower), upper)

    cam_x = _clamp_inside(center[0] + 0.18 * span + camera_jitter[0], xmin + inset, xmax - inset)
    cam_y = _clamp_inside(ymin + inset + camera_jitter[1], ymin + inset, ymax - inset)
    cam_pos = (
        cam_x,
        cam_y,
        placement.tabletop_height + 0.78 + 0.15 * span + camera_jitter[2],
    )
    closeup_pos = (
        placement.robot_base_position[0] - 0.10 * math.cos(placement.robot_base_yaw) - 0.45 * math.sin(placement.robot_base_yaw),
        placement.robot_base_position[1] - 0.10 * math.sin(placement.robot_base_yaw) + 0.45 * math.cos(placement.robot_base_yaw),
        placement.tabletop_height + 0.55,
    )
    closeup_lookat = (
        placement.catch_position[0],
        placement.catch_position[1],
        placement.catch_position[2] - 0.05,
    )
    return cam_pos, lookat, closeup_pos, closeup_lookat


def _camera_metadata(*, pos, lookat, fovy: float) -> dict:
    return {
        "pos": [float(v) for v in pos],
        "lookat": [float(v) for v in lookat],
        "fovy": float(fovy),
    }


def add_variant_xml(root, variant_name: str, *, lighting_intensity: float, floor_jitter: float, camera_jitter, visual_settings: dict | None = None) -> dict:
    spec = VARIANTS[variant_name]
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")

    if variant_name != "robocasa_kitchen":
        add_material(asset, "scene_floor", spec.floor_rgba)
        add_material(asset, "scene_wall", spec.wall_rgba)
        add_material(asset, "countertop", (0.55, 0.51, 0.47, 1.0))
        add_material(asset, "cabinet_front", (0.72, 0.68, 0.60, 1.0))
        ET.SubElement(world, "geom", name="floor", type="plane", size="3.0 3.0 0.05", material="scene_floor", condim="3", friction="1.2 0.005 0.0001")
        add_box(world, "back_wall", (0.45, 1.10, 1.10), (1.9, 0.03, 1.10), "scene_wall")
        add_box(world, "left_wall", (-0.95, 0.10, 1.10), (0.03, 1.10, 1.10), "scene_wall")
        cam_pos = tuple(float(a + b) for a, b in zip(spec.camera_pos, camera_jitter))
        side_pos = (-cam_pos[0] + 0.84, cam_pos[1] + 0.20, cam_pos[2] + 0.06)
        ET.SubElement(world, "camera", name="main_camera", pos=" ".join(f"{v:.5f}" for v in cam_pos), xyaxes=camera_xyaxes(cam_pos, spec.camera_lookat), fovy=f"{spec.fovy:.3f}")
        ET.SubElement(world, "camera", name="side_camera", pos=" ".join(f"{v:.5f}" for v in side_pos), xyaxes=camera_xyaxes(side_pos, spec.camera_lookat), fovy=f"{spec.fovy:.3f}")
        ET.SubElement(world, "camera", name="closeup_camera", pos="0.74 -0.66 0.76", xyaxes=camera_xyaxes((0.74, -0.66, 0.76), (0.46, 0.0, 0.52)), fovy="35")
        return {
            "main_camera": _camera_metadata(pos=cam_pos, lookat=spec.camera_lookat, fovy=spec.fovy),
            "side_camera": _camera_metadata(pos=side_pos, lookat=spec.camera_lookat, fovy=spec.fovy),
            "closeup_camera": _camera_metadata(pos=(0.74, -0.66, 0.76), lookat=(0.46, 0.0, 0.52), fovy=35.0),
        }

    visual_settings = visual_settings or {}
    env_index = visual_settings.get("kitchen_env_index")
    layout_id = int(visual_settings["kitchen_layout_id"])
    style_id = int(visual_settings["kitchen_style_id"])
    scene = resolve_scene(layout_id, style_id)
    materials = _scene_materials(scene.style_config)
    for name, rgba in materials.items():
        add_material(asset, name, rgba, roughness=0.6 if name in {"scene_floor", "countertop", "countertop_island"} else 0.75)
    if layout_id == 58:
        add_material(asset, "layout58_stove_counter_patch_mat", (0.66, 0.28, 0.25, 1.0), roughness=0.68)
    if scene.backend == "robocasa_native":
        _append_native_scene(root, world, asset, layout_id, style_id, int(env_index) if env_index is not None else None)
        _retune_native_textures(asset)
    else:
        for fixture in scene.fixtures:
            if fixture.group_name == "room":
                _draw_room_fixture(world, fixture)
            else:
                _draw_fixture_box(world, fixture)
        _append_style_models(root, scene.fixtures)
    main_variant = None
    side_variant = "opposite_counter_side"
    cam_pos, lookat, closeup_pos, closeup_lookat = _camera_pose(
        scene,
        camera_jitter,
        int(env_index) if env_index is not None else None,
        main_variant,
    )
    side_pos, side_lookat, _, _ = _camera_pose(
        scene,
        camera_jitter,
        int(env_index) if env_index is not None else None,
        side_variant,
    )
    _configure_lighting(root, world, lighting_intensity, scene.room_bounds, lookat)
    ET.SubElement(world, "camera", name="main_camera", pos=" ".join(f"{v:.5f}" for v in cam_pos), xyaxes=camera_xyaxes(cam_pos, lookat), fovy="55")
    ET.SubElement(world, "camera", name="side_camera", pos=" ".join(f"{v:.5f}" for v in side_pos), xyaxes=camera_xyaxes(side_pos, side_lookat), fovy="55")
    ET.SubElement(world, "camera", name="closeup_camera", pos=" ".join(f"{v:.5f}" for v in closeup_pos), xyaxes=camera_xyaxes(closeup_pos, closeup_lookat), fovy="35")
    return {
        "main_camera": {
            **_camera_metadata(pos=cam_pos, lookat=lookat, fovy=55.0),
            "camera_variant": main_variant,
            "room_bounds": list(scene.room_bounds),
            "scene_resolution_backend": scene.backend,
            "scene_resolution_warning": scene.native_error,
        },
        "side_camera": {
            **_camera_metadata(pos=side_pos, lookat=side_lookat, fovy=55.0),
            "camera_variant": side_variant,
            "room_bounds": list(scene.room_bounds),
            "scene_resolution_backend": scene.backend,
            "scene_resolution_warning": scene.native_error,
        },
        "closeup_camera": _camera_metadata(pos=closeup_pos, lookat=closeup_lookat, fovy=35.0),
    }
