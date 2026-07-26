from __future__ import annotations

import copy
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .robocasa_assets import append_robocasa_visual_model, robocasa_assets_available, robocasa_assets_root
from .utils import camera_xyaxes


VARIANT_NAMES = (
    "white_studio",
    "clean_lab",
    "office",
    "warehouse",
    "cluttered_lab",
    "robocasa_kitchen",
    "robocasa_tabletop",
    "robocasa_lab",
    "robocasa_workbench",
    "robocasa_storage",
    "robocasa_official_kitchen",
)
ROBOCASA_SCENE_VARIANTS = {
    "robocasa_kitchen",
    "robocasa_tabletop",
    "robocasa_lab",
    "robocasa_workbench",
    "robocasa_storage",
    "robocasa_official_kitchen",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_MODELS_ROOT = PROJECT_ROOT / "third_party" / "robotwin_1_0" / "models"
ROBOTWIN_2_OBJECTS_ROOT = PROJECT_ROOT / "third_party" / "robotwin_2_0" / "assets" / "objects"
_ROBOTWIN_2_OBJAVERSE_CACHE: tuple[tuple[str, Path, float], ...] | None = None
ROBOTWIN_2_AESTHETIC_TARGET_SPANS = {
    "bottle": 0.15,
    "bowl": 0.16,
    "can": 0.13,
    "chip_can": 0.14,
    "clock": 0.16,
    "drinkbox": 0.13,
    "hammer": 0.15,
    "marker": 0.13,
    "notebook": 0.18,
    "plate": 0.17,
    "pot": 0.16,
    "ramen_box": 0.14,
    "remote": 0.14,
    "snack_box": 0.14,
    "snack_package": 0.13,
    "spoon": 0.14,
    "tape": 0.12,
    "thermos": 0.15,
    "tissue": 0.14,
    "toy_car": 0.12,
    "wallet": 0.12,
}


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
    "robocasa_tabletop": VariantSpec("robocasa_tabletop", (0.57, 0.59, 0.58, 1), (0.86, 0.87, 0.84, 1), (1.70, -0.62, 1.45), (0.46, 0.00, 1.28), 47.0, (0.15, -0.55, 2.70), (0.92, 0.90, 0.84)),
    "robocasa_lab": VariantSpec("robocasa_lab", (0.52, 0.56, 0.57, 1), (0.82, 0.86, 0.86, 1), (1.72, -0.58, 1.48), (0.46, 0.00, 1.30), 47.0, (0.05, -0.62, 2.82), (0.86, 0.90, 0.92)),
    "robocasa_workbench": VariantSpec("robocasa_workbench", (0.48, 0.47, 0.43, 1), (0.76, 0.74, 0.68, 1), (1.78, -0.56, 1.50), (0.46, 0.00, 1.30), 48.0, (0.25, -0.68, 2.62), (0.94, 0.84, 0.70)),
    "robocasa_storage": VariantSpec("robocasa_storage", (0.43, 0.44, 0.43, 1), (0.68, 0.70, 0.68, 1), (1.82, -0.60, 1.52), (0.46, 0.00, 1.30), 49.0, (-0.10, -0.70, 2.72), (0.78, 0.80, 0.76)),
    "robocasa_official_kitchen": VariantSpec("robocasa_official_kitchen", (0.58, 0.56, 0.52, 1), (0.86, 0.86, 0.82, 1), (1.86, -0.48, 1.58), (0.44, 0.00, 1.36), 48.0, (0.1, -0.45, 2.75), (0.95, 0.92, 0.84)),
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


def add_visual_cylinder(world, name: str, pos, radius: float, halfheight: float, material: str, group: int = 1) -> None:
    ET.SubElement(
        world,
        "geom",
        name=name,
        type="cylinder",
        pos=" ".join(f"{v:.4f}" for v in pos),
        size=f"{float(radius):.4f} {float(halfheight):.4f}",
        material=material,
        group=str(group),
        contype="0",
        conaffinity="0",
    )


def _add_handle(world, name: str, pos, *, vertical: bool = False) -> None:
    size = (0.006, 0.010, 0.055) if vertical else (0.060, 0.010, 0.006)
    add_visual_box(world, name, pos, size, "handle_metal")


def _add_small_potted_plant(world: ET.Element, name: str, pos) -> list[dict]:
    x, y, z = (float(v) for v in pos)
    add_visual_cylinder(world, f"{name}_pot", (x, y, z + 0.030), 0.050, 0.030, "wall_art_warm")
    add_visual_box(world, f"{name}_stem", (x, y, z + 0.120), (0.010, 0.010, 0.075), "accent_wood")
    add_visual_box(world, f"{name}_leaf_l", (x - 0.030, y, z + 0.165), (0.055, 0.012, 0.026), "plant_leaf")
    add_visual_box(world, f"{name}_leaf_r", (x + 0.030, y, z + 0.150), (0.055, 0.012, 0.026), "plant_leaf")
    return [{"name": name, "role": "visual_plant", "pos": [x, y, z]}]


def _add_small_wall_lamp(world: ET.Element, name: str, pos) -> list[dict]:
    x, y, z = (float(v) for v in pos)
    add_visual_box(world, f"{name}_arm", (x, y, z), (0.105, 0.008, 0.008), "handle_metal")
    add_visual_box(world, f"{name}_shade", (x + 0.115, y - 0.010, z - 0.020), (0.042, 0.020, 0.038), "lamp_shade")
    add_visual_box(world, f"{name}_glow", (x + 0.115, y - 0.018, z - 0.060), (0.060, 0.004, 0.010), "warm_light")
    return [{"name": name, "role": "visual_wall_lamp", "pos": [x, y, z]}]


def _rng_choice(rng: random.Random, values: tuple):
    return values[rng.randrange(len(values))]


def _jitter_rgba(rgba, rng: random.Random, amount: float) -> tuple[float, float, float, float]:
    values = []
    for index, value in enumerate(rgba):
        if index == 3:
            values.append(float(value))
        else:
            values.append(min(1.0, max(0.0, float(value) + rng.uniform(-amount, amount))))
    return tuple(values)


def _configure_robocasa_lighting(
    root: ET.Element,
    world: ET.Element,
    lighting_intensity: float,
    *,
    profile: str = "robocasa_kitchen",
    rng: random.Random | None = None,
) -> dict:
    rng = rng or random.Random(0)
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    warmth = rng.uniform(0.94, 1.06)
    bright_official = profile == "robocasa_official_kitchen"
    ambient_base = 0.42 if bright_official else (0.34 if profile != "robocasa_storage" else 0.32)
    headlight.set("ambient", f"{ambient_base:.2f} {ambient_base:.2f} {ambient_base * 0.94:.2f}")
    if bright_official:
        headlight.set("diffuse", f"{0.56 * warmth:.2f} {0.54 * warmth:.2f} {0.49 * warmth:.2f}")
        headlight.set("specular", "0.11 0.10 0.09")
    else:
        headlight.set("diffuse", f"{0.48 * warmth:.2f} {0.46 * warmth:.2f} {0.41 * warmth:.2f}")
        headlight.set("specular", "0.08 0.08 0.07")

    for light in world.findall("light"):
        if light.get("name") == "top":
            light.attrib.pop("mode", None)
            light.set("pos", "0.10 -0.55 2.85")
            if bright_official:
                light.set("diffuse", f"{0.58 * warmth:.3f} {0.55 * warmth:.3f} {0.49 * warmth:.3f}")
                light.set("ambient", "0.24 0.24 0.21")
                light.set("specular", "0.08 0.08 0.07")
            else:
                light.set("diffuse", f"{0.42 * warmth:.3f} {0.40 * warmth:.3f} {0.35 * warmth:.3f}")
                light.set("ambient", "0.18 0.18 0.15")
                light.set("specular", "0.04 0.04 0.04")
            light.set("castshadow", "false")

    def scaled(rgb) -> str:
        return " ".join(f"{min(1.0, float(v) * lighting_intensity):.4f}" for v in rgb)

    if bright_official:
        key_rgb = (0.92, 0.90, 0.82)
        fill_rgb = (0.64, 0.64, 0.68)
        wall_rgb = (0.54, 0.50, 0.44)
    elif profile == "robocasa_lab":
        key_rgb = (0.80, 0.86, 0.92)
        fill_rgb = (0.42, 0.46, 0.52)
        wall_rgb = (0.38, 0.42, 0.46)
    elif profile == "robocasa_workbench":
        key_rgb = (1.00, 0.90, 0.72)
        fill_rgb = (0.58, 0.52, 0.44)
        wall_rgb = (0.50, 0.44, 0.36)
    elif profile == "robocasa_storage":
        key_rgb = (0.86, 0.87, 0.80)
        fill_rgb = (0.48, 0.50, 0.48)
        wall_rgb = (0.42, 0.44, 0.40)
    else:
        key_rgb = (1.00, 0.94, 0.82)
        fill_rgb = (0.54, 0.55, 0.58)
        wall_rgb = (0.46, 0.42, 0.36)

    key_pos = (
        -0.55 + rng.uniform(-0.10, 0.10),
        -0.92 + rng.uniform(-0.06, 0.08),
        2.58 + rng.uniform(-0.08, 0.08),
    )
    fill_pos = (
        0.78 + rng.uniform(-0.10, 0.10),
        -1.34 + rng.uniform(-0.06, 0.08),
        1.72 + rng.uniform(-0.06, 0.10),
    )
    wall_pos = (
        -0.80 + rng.uniform(-0.12, 0.12),
        0.90 + rng.uniform(-0.08, 0.08),
        1.95 + rng.uniform(-0.10, 0.10),
    )
    ET.SubElement(
        world,
        "light",
        name="robocasa_softbox_key",
        pos=" ".join(f"{v:.4f}" for v in key_pos),
        diffuse=scaled(key_rgb),
        ambient="0.18 0.18 0.16",
        specular="0.12 0.11 0.10",
        directional="false",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "light",
        name="robocasa_camera_fill",
        pos=" ".join(f"{v:.4f}" for v in fill_pos),
        diffuse=scaled(fill_rgb),
        ambient="0.13 0.13 0.12",
        specular="0.04 0.04 0.04",
        directional="false",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "light",
        name="robocasa_wall_fill",
        pos=" ".join(f"{v:.4f}" for v in wall_pos),
        diffuse=scaled(wall_rgb),
        ambient="0.08 0.08 0.07",
        specular="0.03 0.03 0.03",
        directional="false",
        castshadow="false",
    )
    if bright_official:
        ET.SubElement(
            world,
            "light",
            name="robocasa_demo_ceiling_fill",
            pos="0.35 -0.05 2.25",
            diffuse=scaled((0.58, 0.56, 0.52)),
            ambient="0.14 0.14 0.13",
            specular="0.05 0.05 0.05",
            directional="false",
            castshadow="false",
        )
    return {
        "profile": profile,
        "lighting_intensity": float(lighting_intensity),
        "warmth": float(warmth),
        "key_pos": [float(v) for v in key_pos],
        "fill_pos": [float(v) for v in fill_pos],
        "wall_fill_pos": [float(v) for v in wall_pos],
        "bright_official_kitchen": bool(bright_official),
    }


def _robocasa_model_choices(patterns: tuple[str, ...]) -> list[str]:
    root = robocasa_assets_root()
    choices: list[str] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.name == "model.xml":
                choices.append(path.relative_to(root).as_posix())
    return choices


def _jittered(values, rng: random.Random, scale) -> tuple[float, float, float]:
    return tuple(float(v) + rng.uniform(-float(s), float(s)) for v, s in zip(values, scale))


def _append_random_robocasa_model(
    root: ET.Element,
    rng: random.Random,
    *,
    slot: str,
    patterns: tuple[str, ...],
    fallback: str,
    pos: tuple[float, float, float],
    pos_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
    yaw_jitter: float = 0.0,
    imported_assets: list[dict],
) -> None:
    choices = _robocasa_model_choices(patterns)
    selected = rng.choice(choices) if choices else fallback
    selected_pos = _jittered(pos, rng, pos_jitter)
    selected_yaw = float(yaw) + rng.uniform(-float(yaw_jitter), float(yaw_jitter))
    imported = append_robocasa_visual_model(
        root,
        selected,
        instance=f"rc_{slot}",
        pos=selected_pos,
        euler=(0.0, 0.0, selected_yaw),
    )
    if not imported and selected != fallback:
        selected = fallback
        imported = append_robocasa_visual_model(
            root,
            selected,
            instance=f"rc_{slot}",
            pos=selected_pos,
            euler=(0.0, 0.0, selected_yaw),
        )
    imported_assets.append(
        {
            "slot": slot,
            "model_xml": selected,
            "position": [float(v) for v in selected_pos],
            "yaw": selected_yaw,
            "imported": bool(imported),
        }
    )


def _append_robotwin_visual_mesh(
    root: ET.Element,
    *,
    slot: str,
    rel_obj: str | Path,
    pos: tuple[float, float, float],
    yaw: float,
    scale: float,
    material: str,
    imported_assets: list[dict],
) -> None:
    rel_path = Path(rel_obj)
    obj_path = rel_path if rel_path.is_absolute() else ROBOTWIN_MODELS_ROOT / rel_path
    imported = False
    if obj_path.exists():
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        world = root.find("worldbody")
        if world is None:
            world = ET.SubElement(root, "worldbody")
        mesh_name = f"rw_{slot}_mesh"
        ET.SubElement(
            asset,
            "mesh",
            name=mesh_name,
            file=str(obj_path.resolve()),
            scale=f"{float(scale):.6f} {float(scale):.6f} {float(scale):.6f}",
        )
        body = ET.SubElement(
            world,
            "body",
            name=f"rw_{slot}_wrapper",
            pos=" ".join(f"{float(v):.5f}" for v in pos),
            euler=f"0 0 {float(yaw):.5f}",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"rw_{slot}_visual",
            type="mesh",
            mesh=mesh_name,
            material=material,
            contype="0",
            conaffinity="0",
            group="1",
        )
        imported = True
    imported_assets.append(
        {
            "slot": slot,
            "model_file": str(obj_path),
            "asset_source": "RobotWin",
            "position": [float(v) for v in pos],
            "yaw": float(yaw),
            "scale": float(scale),
            "imported": bool(imported),
        }
    )


def _obj_max_vertex_span(obj_path: Path) -> float | None:
    minimum: list[float] | None = None
    maximum: list[float] | None = None
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                if minimum is None or maximum is None:
                    minimum = list(xyz)
                    maximum = list(xyz)
                else:
                    for axis, value in enumerate(xyz):
                        minimum[axis] = min(minimum[axis], value)
                        maximum[axis] = max(maximum[axis], value)
    except OSError:
        return None
    if minimum is None or maximum is None:
        return None
    return max(maximum[axis] - minimum[axis] for axis in range(3))


def _robotwin_v2_objaverse_candidates() -> tuple[tuple[str, Path, float], ...]:
    global _ROBOTWIN_2_OBJAVERSE_CACHE
    if _ROBOTWIN_2_OBJAVERSE_CACHE is not None:
        return _ROBOTWIN_2_OBJAVERSE_CACHE
    if not ROBOTWIN_2_OBJECTS_ROOT.exists():
        _ROBOTWIN_2_OBJAVERSE_CACHE = ()
        return _ROBOTWIN_2_OBJAVERSE_CACHE

    candidates: list[tuple[str, Path, float]] = []
    for obj_path in sorted(ROBOTWIN_2_OBJECTS_ROOT.glob("objaverse/*/*/textured.obj")):
        category = obj_path.parents[1].name
        target_span = ROBOTWIN_2_AESTHETIC_TARGET_SPANS.get(category)
        if target_span is None:
            continue
        max_span = _obj_max_vertex_span(obj_path)
        if max_span is None or max_span <= 0.0:
            continue
        instance = obj_path.parent.name
        scale = float(target_span) / float(max_span)
        candidates.append((f"v2_{category}_{instance}", obj_path, scale))
    _ROBOTWIN_2_OBJAVERSE_CACHE = tuple(candidates)
    return _ROBOTWIN_2_OBJAVERSE_CACHE


def _kitchen_major_slots() -> list[dict]:
    return [
        {
            "slot": "counter_appliance_left",
            "patterns": (
                "fixtures/blenders/*/model.xml",
                "fixtures/stand_mixers/*/model.xml",
                "fixtures/electric_kettles/*/model.xml",
                "fixtures/toasters/*/model.xml",
            ),
            "fallback": "fixtures/blenders/Blender001/model.xml",
            "pos": (0.98, 0.82, 1.00),
            "pos_jitter": (0.030, 0.020, 0.015),
            "yaw": -0.20,
            "yaw_jitter": 0.22,
        },
        {
            "slot": "counter_appliance_right",
            "patterns": (
                "fixtures/coffee_machines/*/model.xml",
                "fixtures/electric_kettles/*/model.xml",
                "fixtures/toasters/*/model.xml",
            ),
            "fallback": "fixtures/coffee_machines/CoffeeMachine036/model.xml",
            "pos": (-0.66, 0.82, 0.92),
            "pos_jitter": (0.030, 0.020, 0.015),
            "yaw": 0.0,
            "yaw_jitter": 0.20,
        },
        {
            "slot": "microwave",
            "patterns": (
                "fixtures/microwaves/*/model.xml",
                "fixtures/toaster_ovens/*/model.xml",
            ),
            "fallback": "fixtures/microwaves/Microwave011/model.xml",
            "pos": (-0.62, 0.78, 0.94),
            "pos_jitter": (0.025, 0.020, 0.015),
            "yaw": 0.0,
            "yaw_jitter": 0.12,
        },
        {
            "slot": "refrigerator",
            "patterns": ("fixtures/fridges/*/model.xml",),
            "fallback": "fixtures/fridges/Refrigerator031/model.xml",
            "pos": (-0.88, 0.82, 0.96),
            "pos_jitter": (0.020, 0.020, 0.010),
            "yaw": 0.0,
            "yaw_jitter": 0.10,
        },
        {
            "slot": "sink",
            "patterns": ("fixtures/sinks/*/model.xml",),
            "fallback": "fixtures/sinks/Sink001/model.xml",
            "pos": (-0.56, 0.56, 0.93),
            "pos_jitter": (0.030, 0.025, 0.010),
            "yaw": 0.0,
            "yaw_jitter": 0.08,
        },
        {
            "slot": "range",
            "patterns": (
                "fixtures/ovens/*/model.xml",
                "fixtures/stoves/*/model.xml",
                "fixtures/stovetops/*/model.xml",
            ),
            "fallback": "fixtures/ovens/Oven031/model.xml",
            "pos": (0.28, 0.70, 1.02),
            "pos_jitter": (0.035, 0.020, 0.015),
            "yaw": 0.0,
            "yaw_jitter": 0.12,
        },
        {
            "slot": "dishwasher",
            "patterns": ("fixtures/dishwashers/*/model.xml",),
            "fallback": "fixtures/dishwashers/Dishwasher031/model.xml",
            "pos": (-0.12, 0.69, 0.59),
            "pos_jitter": (0.035, 0.020, 0.015),
            "yaw": 0.0,
            "yaw_jitter": 0.08,
        },
    ]


def _kitchen_clutter_slots() -> list[dict]:
    return [
        {
            "slot": "counter_basket",
            "patterns": ("objects/lightwheel/basket/*/model.xml",),
            "fallback": "objects/lightwheel/basket/Basket026/model.xml",
            "pos": (1.06, 0.82, 0.80),
            "pos_jitter": (0.025, 0.020, 0.010),
            "yaw": 0.15,
            "yaw_jitter": 0.45,
        },
        {
            "slot": "counter_scale",
            "patterns": ("objects/lightwheel/digital_scale/*/model.xml",),
            "fallback": "objects/lightwheel/digital_scale/DigitalScale001/model.xml",
            "pos": (0.78, 0.84, 0.79),
            "pos_jitter": (0.025, 0.020, 0.010),
            "yaw": 0.25,
            "yaw_jitter": 0.35,
        },
        {
            "slot": "shelf_glass_or_jar",
            "patterns": (
                "objects/lightwheel/glass_cup/*/model.xml",
                "objects/lightwheel/jar/*/model.xml",
                "objects/lightwheel/mug/*/model.xml",
            ),
            "fallback": "objects/lightwheel/jar/Jar003/model.xml",
            "pos": (1.03, 0.70, 1.43),
            "pos_jitter": (0.035, 0.030, 0.020),
            "yaw": -0.15,
            "yaw_jitter": 0.40,
        },
        {
            "slot": "shelf_food_or_plant",
            "patterns": (
                "objects/lightwheel/flour_bag/*/model.xml",
                "objects/lightwheel/plant/*/model.xml",
                "objects/lightwheel/soap_dispenser/*/model.xml",
            ),
            "fallback": "objects/lightwheel/plant/Plant006/model.xml",
            "pos": (1.14, 0.68, 1.255),
            "pos_jitter": (0.035, 0.030, 0.020),
            "yaw": 0.20,
            "yaw_jitter": 0.45,
        },
    ]


def _profile_major_slots(profile: str) -> list[dict]:
    if profile == "robocasa_kitchen":
        return _kitchen_major_slots()
    if profile == "robocasa_tabletop":
        return [
            {
                "slot": "rear_tray",
                "patterns": ("objects/lightwheel/tray/*/model.xml", "objects/lightwheel/tupperware/*/model.xml"),
                "fallback": "objects/lightwheel/tray/Tray006/model.xml",
                "pos": (0.95, 0.86, 0.80),
                "pos_jitter": (0.050, 0.030, 0.010),
                "yaw": 0.0,
                "yaw_jitter": 0.45,
            },
            {
                "slot": "side_plant",
                "patterns": ("objects/lightwheel/plant/*/model.xml", "objects/lightwheel/flower_vase/*/model.xml"),
                "fallback": "objects/lightwheel/plant/Plant006/model.xml",
                "pos": (-0.70, 0.82, 0.83),
                "pos_jitter": (0.030, 0.030, 0.015),
                "yaw": 0.0,
                "yaw_jitter": 0.55,
            },
        ]
    if profile == "robocasa_lab":
        return [
            {
                "slot": "lab_scale",
                "patterns": ("objects/lightwheel/digital_scale/*/model.xml",),
                "fallback": "objects/lightwheel/digital_scale/DigitalScale001/model.xml",
                "pos": (0.92, 0.86, 0.80),
                "pos_jitter": (0.035, 0.025, 0.010),
                "yaw": 0.15,
                "yaw_jitter": 0.35,
            },
            {
                "slot": "lab_rack",
                "patterns": (
                    "objects/lightwheel/dish_rack/*/model.xml",
                    "objects/lightwheel/utensil_rack/*/model.xml",
                    "objects/lightwheel/tiered_shelf/*/model.xml",
                ),
                "fallback": "objects/lightwheel/dish_rack/DishRack001/model.xml",
                "pos": (-0.62, 0.84, 0.84),
                "pos_jitter": (0.030, 0.025, 0.015),
                "yaw": 0.0,
                "yaw_jitter": 0.25,
            },
        ]
    if profile == "robocasa_workbench":
        return [
            {
                "slot": "workbench_tool_rack",
                "patterns": ("objects/lightwheel/utensil_rack/*/model.xml", "objects/lightwheel/knife_block/*/model.xml"),
                "fallback": "objects/lightwheel/utensil_rack/UtensilRack010/model.xml",
                "pos": (0.98, 0.86, 0.83),
                "pos_jitter": (0.035, 0.025, 0.015),
                "yaw": -0.10,
                "yaw_jitter": 0.35,
            },
            {
                "slot": "workbench_storage_bin",
                "patterns": ("objects/lightwheel/basket/*/model.xml", "objects/lightwheel/tupperware/*/model.xml"),
                "fallback": "objects/lightwheel/basket/Basket026/model.xml",
                "pos": (-0.64, 0.84, 0.80),
                "pos_jitter": (0.040, 0.025, 0.010),
                "yaw": 0.20,
                "yaw_jitter": 0.50,
            },
        ]
    if profile == "robocasa_storage":
        return [
            {
                "slot": "storage_tiered_shelf",
                "patterns": ("objects/lightwheel/tiered_shelf/*/model.xml", "objects/lightwheel/tiered_basket/*/model.xml"),
                "fallback": "objects/lightwheel/tiered_shelf/Shelf016/model.xml",
                "pos": (1.06, 0.84, 0.86),
                "pos_jitter": (0.035, 0.025, 0.015),
                "yaw": 0.0,
                "yaw_jitter": 0.30,
            },
            {
                "slot": "storage_container",
                "patterns": ("objects/lightwheel/tupperware/*/model.xml", "objects/lightwheel/basket/*/model.xml"),
                "fallback": "objects/lightwheel/tupperware/Tupperware028/model.xml",
                "pos": (-0.66, 0.84, 0.80),
                "pos_jitter": (0.040, 0.025, 0.010),
                "yaw": 0.12,
                "yaw_jitter": 0.45,
            },
        ]
    return []


def _profile_clutter_slots(profile: str) -> list[dict]:
    if profile == "robocasa_kitchen":
        return _kitchen_clutter_slots()
    common = [
        {
            "slot": "rear_cup_or_jar",
            "patterns": ("objects/lightwheel/glass_cup/*/model.xml", "objects/lightwheel/jar/*/model.xml"),
            "fallback": "objects/lightwheel/jar/Jar003/model.xml",
            "pos": (1.12, 0.78, 1.19),
            "pos_jitter": (0.035, 0.030, 0.020),
            "yaw": -0.15,
            "yaw_jitter": 0.45,
        },
        {
            "slot": "rear_dispenser_or_spray",
            "patterns": ("objects/lightwheel/soap_dispenser/*/model.xml", "objects/lightwheel/spray/*/model.xml"),
            "fallback": "objects/lightwheel/soap_dispenser/SoapDispenser001/model.xml",
            "pos": (-0.64, 0.78, 1.17),
            "pos_jitter": (0.035, 0.030, 0.020),
            "yaw": 0.10,
            "yaw_jitter": 0.45,
        },
        {
            "slot": "back_counter_pitcher",
            "patterns": ("objects/lightwheel/pitcher/*/model.xml", "objects/lightwheel/measuring_cup/*/model.xml"),
            "fallback": "objects/lightwheel/pitcher/Pitcher001/model.xml",
            "pos": (0.78, 0.84, 0.82),
            "pos_jitter": (0.035, 0.025, 0.015),
            "yaw": 0.20,
            "yaw_jitter": 0.45,
        },
        {
            "slot": "side_colander_or_pot",
            "patterns": ("objects/lightwheel/colander/*/model.xml", "objects/lightwheel/pot/*/model.xml"),
            "fallback": "objects/lightwheel/colander/Colander001/model.xml",
            "pos": (-0.82, 0.70, 0.82),
            "pos_jitter": (0.030, 0.030, 0.015),
            "yaw": -0.20,
            "yaw_jitter": 0.55,
        },
    ]
    if profile == "robocasa_workbench":
        common.extend(
            [
                {
                    "slot": "bench_brush_like_prop",
                    "patterns": ("objects/lightwheel/dish_brush/*/model.xml", "objects/lightwheel/wooden_spoon/*/model.xml"),
                    "fallback": "objects/lightwheel/dish_brush/DishBrush011/model.xml",
                    "pos": (0.58, 0.86, 0.80),
                    "pos_jitter": (0.030, 0.020, 0.010),
                    "yaw": 0.65,
                    "yaw_jitter": 0.50,
                }
            ]
        )
    return common


def _add_random_robocasa_background_assets(
    root: ET.Element,
    rng: random.Random,
    *,
    profile: str,
    randomization_level: str,
) -> list[dict]:
    imported_assets: list[dict] = []
    major_slots = _profile_major_slots(profile)
    for slot in major_slots:
        _append_random_robocasa_model(root, rng, imported_assets=imported_assets, **slot)

    clutter_slots = _profile_clutter_slots(profile)
    rng.shuffle(clutter_slots)
    if randomization_level == "clean":
        min_count, max_count = 0, min(1, len(clutter_slots))
    elif randomization_level == "cluttered":
        min_count, max_count = min(2, len(clutter_slots)), min(3, len(clutter_slots))
    else:
        min_count, max_count = min(1, len(clutter_slots)), min(2, len(clutter_slots))
    clutter_count = rng.randint(min_count, max_count) if clutter_slots else 0
    for slot in clutter_slots[:clutter_count]:
        _append_random_robocasa_model(root, rng, imported_assets=imported_assets, **slot)

    return imported_assets


def _texture_profile(profile: str, rng: random.Random) -> dict:
    if profile == "robocasa_lab":
        floor = _rng_choice(rng, ("textures/tiles/concrete_tiles.png", "textures/flat/light_gray.png", "textures/tiles/white_tiles.png"))
        wall = _rng_choice(rng, ("textures/flat/warm_white_2.png", "textures/flat/blue_gray.png", "textures/flat/lighter_gray.png"))
        counter = _rng_choice(rng, ("textures/metals/brighter_metal.png", "textures/marble/granite_2.png", "textures/flat/light_gray.png"))
        cabinet = _rng_choice(rng, ("textures/flat/white.png", "textures/flat/lighter_gray.png", "textures/metals/metal.png"))
        accent = _rng_choice(rng, ("textures/metals/bright_metal.png", "textures/flat/blue_gray.png"))
    elif profile == "robocasa_workbench":
        floor = _rng_choice(rng, ("textures/wood/gray_wood_planks.png", "textures/wood/warm_wood_planks.png", "textures/prev/plywood.png"))
        wall = _rng_choice(rng, ("textures/flat/warm_white.png", "textures/bricks/white_bricks.png", "textures/prev/gray-plaster.png"))
        counter = _rng_choice(rng, ("textures/wood/walnut_wood_grain.png", "textures/wood/warm_wood_grain.png", "textures/prev/plywood.png"))
        cabinet = _rng_choice(rng, ("textures/wood/dark_wood_planks.png", "textures/wood/wood_grain_2.png"))
        accent = _rng_choice(rng, ("textures/metals/steel-brushed.png", "textures/metals/metal.png"))
    elif profile == "robocasa_storage":
        floor = _rng_choice(rng, ("textures/others/concrete.png", "textures/tiles/concrete_tiles.png", "textures/flat/gray.png"))
        wall = _rng_choice(rng, ("textures/flat/light_gray.png", "textures/bricks/white_bricks_2.png", "textures/prev/stone_wall.png"))
        counter = _rng_choice(rng, ("textures/metals/steel-brushed.png", "textures/wood/gray_wood_grain.png"))
        cabinet = _rng_choice(rng, ("textures/metals/metal.png", "textures/flat/dark_gray.png"))
        accent = _rng_choice(rng, ("textures/metals/brighter_metal.png", "textures/flat/gray.png"))
    elif profile == "robocasa_tabletop":
        floor = _rng_choice(rng, ("textures/wood/light_wood_planks.png", "textures/wood/warm_wood_parquet.png", "textures/tiles/marble_tiles.png"))
        wall = _rng_choice(rng, ("textures/flat/warm_white_2.png", "textures/flat/cream.png", "textures/flat/lighter_gray.png"))
        counter = _rng_choice(rng, ("textures/marble/granite.png", "textures/wood/light_wood_planks_long.png"))
        cabinet = _rng_choice(rng, ("textures/wood/light_wood_planks.png", "textures/flat/warm_white.png"))
        accent = _rng_choice(rng, ("textures/wood/bamboo.png", "textures/metals/brass.png"))
    else:
        floor = _rng_choice(rng, ("textures/wood/warm_wood_planks.png", "textures/wood/light_wood_planks.png", "textures/tiles/marble_tiles.png"))
        wall = _rng_choice(rng, ("textures/flat/warm_white_2.png", "textures/tiles/white_square_tiles.png", "textures/flat/cream.png"))
        counter = _rng_choice(rng, ("textures/marble/granite_2.png", "textures/marble/granite.png", "textures/wood/warm_wood_grain_2.png"))
        cabinet = _rng_choice(rng, ("textures/wood/warm_wood_grain_2.png", "textures/wood/warm_wood_grain.png"))
        accent = _rng_choice(rng, ("textures/wood/dark_wood_planks.png", "textures/wood/bamboo.png"))
    return {
        "floor": floor,
        "wall": wall,
        "countertop": counter,
        "cabinet_front": cabinet,
        "backsplash_tile": _rng_choice(rng, ("textures/tiles/white_square_tiles.png", "textures/tiles/diamond_white_tiles.png", "textures/tiles/concrete_tiles.png")),
        "accent_wood": accent,
    }


def _add_robot_support_table(world: ET.Element, *, profile: str, rng: random.Random) -> dict:
    style_by_profile = {
        "robocasa_tabletop": ("clean_table", "countertop", "cabinet_front"),
        "robocasa_lab": ("stainless_lab_bench", "countertop", "metal"),
        "robocasa_workbench": ("wood_workbench", "countertop", "cabinet_front"),
        "robocasa_storage": ("utility_table", "countertop", "metal"),
    }
    style, top_mat, base_mat = style_by_profile.get(profile, ("clean_table", "countertop", "cabinet_front"))
    add_box(world, "robot_table_top", (0.05, -0.08, 0.70), (0.84, 0.72, 0.04), top_mat)
    if style in {"clean_table", "stainless_lab_bench", "utility_table"}:
        for i, (x, y) in enumerate(((-0.68, -0.68), (0.78, -0.68), (-0.68, 0.46), (0.78, 0.46))):
            add_box(world, f"robot_table_leg_{i}", (x, y, 0.35), (0.026, 0.026, 0.35), base_mat)
        add_visual_box(world, "robot_table_back_rail", (0.05, 0.50, 0.63), (0.77, 0.014, 0.032), base_mat)
        add_visual_box(world, "robot_table_front_rail", (0.05, -0.74, 0.63), (0.77, 0.014, 0.032), base_mat)
    else:
        add_box(world, "robot_table_front", (0.05, -0.78, 0.38), (0.84, 0.025, 0.31), base_mat)
        add_box(world, "robot_table_left_side", (-0.79, -0.08, 0.38), (0.025, 0.70, 0.31), base_mat)
        add_box(world, "robot_table_right_side", (0.89, -0.08, 0.38), (0.025, 0.70, 0.31), base_mat)
        add_visual_box(world, "robot_table_toe_kick", (0.05, -0.815, 0.09), (0.78, 0.012, 0.035), "drawer_gap")
        for i, x in enumerate((-0.42, 0.00, 0.42)):
            add_visual_box(world, f"robot_table_drawer_{i}", (x, -0.809, 0.50), (0.17, 0.010, 0.080), base_mat)
            _add_handle(world, f"robot_table_drawer_handle_{i}", (x, -0.822, 0.525))
    return {"support_furniture": style, "top_material": top_mat, "base_material": base_mat}


def _add_profile_room_furniture(world: ET.Element, *, profile: str, rng: random.Random) -> list[dict]:
    furniture: list[dict] = []
    if profile == "robocasa_lab":
        back_mat = "metal"
        top_mat = "countertop"
        shelf_mat = "metal"
    elif profile == "robocasa_workbench":
        back_mat = "cabinet_front"
        top_mat = "countertop"
        shelf_mat = "accent_wood"
    elif profile == "robocasa_storage":
        back_mat = "metal"
        top_mat = "countertop"
        shelf_mat = "metal"
    else:
        back_mat = "cabinet_front"
        top_mat = "countertop"
        shelf_mat = "accent_wood"

    add_box(world, "back_counter_top", (0.34, 0.68, 0.70), (1.05, 0.21, 0.04), top_mat)
    add_box(world, "back_counter_base", (0.34, 0.74, 0.36), (1.05, 0.15, 0.32), back_mat)
    furniture.extend(
        [
            {"name": "back_counter_top", "role": "background_support", "pos": [0.34, 0.68, 0.70]},
            {"name": "back_counter_base", "role": "background_support", "pos": [0.34, 0.74, 0.36]},
        ]
    )

    if profile == "robocasa_lab":
        add_visual_box(world, "rear_whiteboard", (-0.56, 1.063, 1.43), (0.27, 0.008, 0.18), "backsplash_tile")
        add_visual_box(world, "rear_whiteboard_note", (-0.46, 1.052, 1.46), (0.045, 0.006, 0.030), "wall_art_blue")
        add_visual_box(world, "left_window_panel", (-0.918, -0.30, 1.52), (0.008, 0.26, 0.20), "window_glass")
        furniture.extend(_add_small_potted_plant(world, "lab_corner_plant", (-0.82, 0.90, 0.72)))
        furniture.extend(
            [
                {"name": "rear_whiteboard", "role": "visual_wall_detail", "pos": [-0.56, 1.063, 1.43]},
                {"name": "left_window_panel", "role": "visual_wall_detail", "pos": [-0.918, -0.30, 1.52]},
            ]
        )
    elif profile == "robocasa_workbench":
        add_visual_box(world, "rear_pegboard", (-0.58, 1.063, 1.38), (0.30, 0.008, 0.22), "wall_art_warm")
        add_visual_box(world, "rear_pegboard_rail", (-0.58, 1.052, 1.43), (0.24, 0.006, 0.010), "handle_metal")
        add_visual_box(world, "rear_bench_lamp", (0.10, 0.965, 1.36), (0.34, 0.010, 0.010), "warm_light")
        add_visual_box(world, "left_wall_print", (-0.918, -0.34, 1.48), (0.008, 0.22, 0.16), "wall_art_blue")
        furniture.extend(_add_small_wall_lamp(world, "workbench_wall_lamp", (-0.24, 1.050, 1.57)))
        furniture.extend(
            [
                {"name": "rear_pegboard", "role": "visual_wall_detail", "pos": [-0.58, 1.063, 1.38]},
                {"name": "left_wall_print", "role": "visual_wall_detail", "pos": [-0.918, -0.34, 1.48]},
            ]
        )
    elif profile == "robocasa_storage":
        add_visual_box(world, "rear_label_panel", (-0.52, 1.063, 1.42), (0.28, 0.008, 0.16), "wall_art_blue")
        add_visual_box(world, "rear_label_strip", (-0.52, 1.052, 1.42), (0.22, 0.006, 0.016), "lamp_shade")
        add_visual_box(world, "storage_floor_marker", (0.80, -0.86, 0.014), (0.34, 0.010, 0.006), "wall_art_warm")
        furniture.extend(_add_small_wall_lamp(world, "storage_wall_lamp", (-0.18, 1.050, 1.54)))
        furniture.extend(
            [
                {"name": "rear_label_panel", "role": "visual_wall_detail", "pos": [-0.52, 1.063, 1.42]},
                {"name": "storage_floor_marker", "role": "visual_floor_detail", "pos": [0.80, -0.86, 0.014]},
            ]
        )
    elif profile == "robocasa_tabletop":
        add_visual_box(world, "rear_framed_print", (-0.56, 1.063, 1.43), (0.24, 0.008, 0.17), "wall_art_blue")
        add_visual_box(world, "left_small_window", (-0.918, -0.32, 1.50), (0.008, 0.22, 0.18), "window_glass")
        furniture.extend(_add_small_potted_plant(world, "tabletop_corner_plant", (-0.82, 0.88, 0.72)))
        furniture.extend(
            [
                {"name": "rear_framed_print", "role": "visual_wall_detail", "pos": [-0.56, 1.063, 1.43]},
                {"name": "left_small_window", "role": "visual_wall_detail", "pos": [-0.918, -0.32, 1.50]},
            ]
        )

    if profile == "robocasa_storage":
        for i, z in enumerate((0.48, 0.82, 1.16)):
            add_box(world, f"right_storage_shelf_{i}", (1.22, 0.84, z), (0.24, 0.09, 0.018), shelf_mat)
            furniture.append({"name": f"right_storage_shelf_{i}", "role": "background_shelf", "pos": [1.22, 0.84, z]})
        for i, x in enumerate((1.00, 1.44)):
            add_box(world, f"right_storage_post_{i}", (x, 0.84, 0.82), (0.018, 0.018, 0.52), shelf_mat)
    else:
        add_visual_box(world, "rear_shelf_back", (1.12, 0.96, 1.32), (0.23, 0.014, 0.26), shelf_mat)
        for i, z in enumerate((1.14, 1.32, 1.50)):
            add_visual_box(world, f"rear_open_shelf_{i}", (1.12, 0.78, z), (0.24, 0.16, 0.012), shelf_mat)
        furniture.append({"name": "rear_open_shelf", "role": "visual_background_shelf", "pos": [1.12, 0.78, 1.32]})

    if profile in {"robocasa_lab", "robocasa_workbench"}:
        add_visual_box(world, "rear_tool_panel", (-0.66, 0.985, 1.28), (0.24, 0.012, 0.23), "backsplash_tile")
        add_visual_box(world, "rear_task_light", (0.18, 0.965, 1.34), (0.42, 0.012, 0.010), "warm_light")
    elif profile == "robocasa_tabletop":
        add_visual_box(world, "rear_pinboard", (-0.66, 0.985, 1.34), (0.24, 0.012, 0.25), "accent_wood")

    return furniture


def _add_robotwin_background_assets(
    root: ET.Element,
    rng: random.Random,
    *,
    profile: str,
    randomization_level: str,
) -> list[dict]:
    if not ROBOTWIN_MODELS_ROOT.exists() and not ROBOTWIN_2_OBJECTS_ROOT.exists():
        return []
    candidates: list[tuple[str, str | Path, float]] = [
        ("cup", "022_cup/textured.obj", 0.030),
        ("brush", "024_brush/textured.obj", 0.050),
        ("red_bottle", "025_red_bottle/textured.obj", 0.036),
        ("green_bottle", "026_green_bottle/textured.obj", 0.036),
        ("coaster", "019_coaster/textured.obj", 0.040),
        ("rack", "040_rack/textured.obj", 0.045),
        ("wooden_box", "042_wooden_box/textured.obj", 0.036),
    ]
    candidates.extend(_robotwin_v2_objaverse_candidates())
    profile_keys = {
        "robocasa_tabletop": ("cup", "coaster", "wooden_box", "bowl", "plate", "can", "drinkbox", "snack_box"),
        "robocasa_lab": ("cup", "red_bottle", "green_bottle", "bottle", "can", "notebook", "tissue", "remote"),
        "robocasa_workbench": ("brush", "rack", "wooden_box", "hammer", "marker", "notebook", "tape", "wallet"),
        "robocasa_storage": ("rack", "wooden_box", "tape", "tissue", "remote", "wallet", "snack_box", "drinkbox"),
        "robocasa_kitchen": ("cup", "red_bottle", "green_bottle", "bottle", "bowl", "plate", "pot", "spoon", "snack_box", "ramen_box"),
    }

    def matches_profile(candidate_name: str) -> bool:
        keys = profile_keys.get(profile, ())
        if not keys:
            return True
        return any(candidate_name == key or candidate_name.startswith(f"v2_{key}_") for key in keys)

    profile_candidates = [candidate for candidate in candidates if matches_profile(candidate[0])]
    if profile_candidates:
        candidates = profile_candidates
    positions_by_profile = {
        "robocasa_tabletop": [(0.58, 0.86, 0.82), (1.08, 0.78, 1.15), (-0.70, 0.84, 0.82)],
        "robocasa_lab": [(0.62, 0.86, 0.82), (1.10, 0.78, 1.16), (-0.70, 0.84, 0.82)],
        "robocasa_workbench": [(0.72, 0.86, 0.82), (1.12, 0.78, 1.16), (-0.66, 0.84, 0.82)],
        "robocasa_storage": [(0.84, 0.86, 0.82), (1.20, 0.82, 1.15), (-0.70, 0.84, 0.82)],
        "robocasa_kitchen": [(0.64, 0.82, 0.82), (1.08, 0.78, 1.18), (-0.70, 0.82, 0.82)],
    }
    positions = positions_by_profile.get(profile, [(0.58, 0.86, 0.82), (1.12, 0.78, 1.16), (-0.70, 0.84, 0.82)])
    if randomization_level == "clean":
        count = 0
    elif randomization_level == "cluttered":
        count = 2
    else:
        count = 1
    imported_assets: list[dict] = []
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    for index, (name, rel_obj, scale) in enumerate(shuffled[:count]):
        base_pos = positions[index % len(positions)]
        pos = _jittered(base_pos, rng, (0.025, 0.020, 0.010))
        _append_robotwin_visual_mesh(
            root,
            slot=f"{name}_{index}",
            rel_obj=rel_obj,
            pos=pos,
            yaw=rng.uniform(-0.65, 0.65),
            scale=scale,
            material="robotwin_prop_mat",
            imported_assets=imported_assets,
        )
    return imported_assets


def _append_unique_assets(target_asset: ET.Element, source_asset: ET.Element | None) -> int:
    if source_asset is None:
        return 0
    existing = {
        (child.tag, child.get("name"))
        for child in target_asset
        if child.get("name") is not None
    }
    appended = 0
    for child in source_asset:
        key = (child.tag, child.get("name"))
        if child.get("name") is not None and key in existing:
            continue
        target_asset.append(copy.deepcopy(child))
        appended += 1
        if child.get("name") is not None:
            existing.add(key)
    return appended


def _make_geoms_visual_only(root: ET.Element) -> int:
    count = 0
    for geom in root.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        if geom.get("group") is None:
            geom.set("group", "1")
        count += 1
    return count


def _remove_movable_fixture_joints(root: ET.Element) -> int:
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"joint", "freejoint"}:
                parent.remove(child)
                removed += 1
    return removed


def _add_official_robocasa_kitchen(
    root: ET.Element,
    world: ET.Element,
    rng: random.Random,
    *,
    randomization_level: str,
) -> dict:
    try:
        import numpy as np
        from robocasa.models.scenes import KitchenArena
    except Exception as exc:
        add_visual_box(world, "official_robocasa_import_failed_marker", (0.52, 1.05, 0.86), (0.16, 0.03, 0.08), "prop_red")
        return {
            "asset_source": "RoboCasa KitchenArena",
            "imported": False,
            "error": repr(exc),
        }

    if randomization_level == "clean":
        layout_id = rng.randint(1, 10)
        style_id = layout_id
        kitchen_split = "target"
        clutter_mode = 0
    else:
        layout_id = rng.randint(11, 60)
        style_id = rng.randint(11, 60)
        kitchen_split = "pretrain"
        clutter_mode = 1 if randomization_level in {"balanced", "cluttered"} else 0

    arena_seed = rng.randrange(1, 2**31 - 1)
    arena = KitchenArena(
        layout_id=layout_id,
        style_id=style_id,
        rng=np.random.default_rng(arena_seed),
        clutter_mode=clutter_mode,
    )
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    wrapper_pos = (2.85, 1.05, 0.00)
    wrapper = ET.SubElement(
        world,
        "body",
        name="rc_official_kitchen_wrapper",
        pos=" ".join(f"{v:.5f}" for v in wrapper_pos),
        euler="0 0 3.14159",
    )

    appended_assets = 0
    appended_fixtures = 0
    skipped_shell_fixtures = 0
    visual_geom_count = 0
    removed_joint_count = 0
    for fixture_name, fixture in arena.fixtures.items():
        # Enclosing RoboCasa walls/floors often hide the styled cabinets from
        # our interception cameras. The project scene already provides a floor;
        # keep the real kitchen fixtures and styled assets as the background.
        if fixture_name.startswith(("wall_", "floor_")):
            skipped_shell_fixtures += 1
            continue
        appended_assets += _append_unique_assets(asset, getattr(fixture, "asset", None))
        if not hasattr(fixture, "get_obj"):
            continue
        body = copy.deepcopy(fixture.get_obj())
        visual_geom_count += _make_geoms_visual_only(body)
        removed_joint_count += _remove_movable_fixture_joints(body)
        wrapper.append(body)
        appended_fixtures += 1

    return {
        "asset_source": "RoboCasa KitchenArena",
        "imported": True,
        "kitchen_scene_split": kitchen_split,
        "layout_id": int(layout_id),
        "style_id": int(style_id),
        "arena_seed": int(arena_seed),
        "clutter_mode": int(clutter_mode),
        "wrapper_pos": [float(v) for v in wrapper_pos],
        "wrapper_euler": [0.0, 0.0, 3.14159],
        "fixture_count": int(appended_fixtures),
        "skipped_shell_fixture_count": int(skipped_shell_fixtures),
        "visual_geom_count": int(visual_geom_count),
        "removed_joint_count": int(removed_joint_count),
        "asset_count": int(appended_assets),
        "collision_policy": "visual_only_background; task contacts handled by generator surfaces and Robotiq pads",
    }


def _add_official_kitchen_visible_demo_layer(
    root: ET.Element,
    world: ET.Element,
    rng: random.Random,
    *,
    randomization_level: str,
) -> dict:
    decorative_geometry = []

    def add_named_visual(name: str, pos, size, material: str) -> None:
        add_visual_box(world, name, pos, size, material)
        decorative_geometry.append(
            {
                "name": name,
                "pos": [float(v) for v in pos],
                "size": [float(v) for v in size],
                "material": material,
                "collision_policy": "visual_only",
            }
        )

    add_named_visual("official_demo_back_wall_panel", (0.38, 1.082, 1.245), (1.36, 0.012, 0.880), "scene_wall")
    add_named_visual("official_demo_left_wall_panel", (-0.936, 0.16, 1.245), (0.012, 0.91, 0.880), "scene_wall")
    add_named_visual("official_demo_rear_counter_top", (0.38, 0.92, 0.700), (1.18, 0.105, 0.040), "countertop")
    add_named_visual("official_demo_rear_counter_base", (0.38, 0.985, 0.365), (1.18, 0.060, 0.315), "cabinet_front")
    add_named_visual("official_demo_backsplash", (0.38, 1.070, 1.030), (1.20, 0.010, 0.250), "backsplash_tile")
    add_named_visual("official_demo_under_cabinet_light_l", (-0.34, 1.045, 1.285), (0.28, 0.010, 0.009), "warm_light")
    add_named_visual("official_demo_under_cabinet_light_r", (0.42, 1.045, 1.285), (0.28, 0.010, 0.009), "warm_light")

    for i, x in enumerate((-0.48, -0.12, 0.24, 0.60, 0.92)):
        add_named_visual(f"official_demo_lower_panel_{i}", (x, 0.914, 0.385), (0.130, 0.010, 0.190), "cabinet_front")
        add_named_visual(f"official_demo_lower_handle_{i}", (x + 0.050, 0.900, 0.395), (0.005, 0.009, 0.045), "handle_metal")

    for i, x in enumerate((-0.44, -0.08, 0.28, 0.64)):
        add_named_visual(f"official_demo_upper_cabinet_{i}", (x, 1.015, 1.560), (0.155, 0.055, 0.235), "cabinet_front")
        add_named_visual(f"official_demo_upper_handle_{i}", (x + 0.052, 0.948, 1.505), (0.005, 0.009, 0.050), "handle_metal")

    add_named_visual("official_demo_window_glass", (-0.76, 1.056, 1.520), (0.180, 0.007, 0.170), "window_glass")
    add_named_visual("official_demo_window_frame_top", (-0.76, 1.047, 1.700), (0.205, 0.012, 0.009), "handle_metal")
    add_named_visual("official_demo_window_frame_bottom", (-0.76, 1.047, 1.340), (0.205, 0.012, 0.009), "handle_metal")
    add_named_visual("official_demo_range_hood_body", (0.25, 0.890, 1.395), (0.310, 0.070, 0.050), "metal")
    add_named_visual("official_demo_range_hood_stack", (0.25, 1.005, 1.675), (0.145, 0.045, 0.225), "metal")
    add_named_visual("official_demo_right_shelf_back", (1.16, 0.945, 1.335), (0.205, 0.012, 0.250), "accent_wood")
    for i, z in enumerate((1.155, 1.335, 1.515)):
        add_named_visual(f"official_demo_right_open_shelf_{i}", (1.16, 0.780, z), (0.215, 0.145, 0.011), "accent_wood")

    return {
        "enabled": True,
        "decorative_geometry": decorative_geometry,
        "foreground_assets": [],
        "foreground_asset_policy": "omitted_for_demo_memory; official KitchenArena fixtures remain imported",
    }


def _add_robocasa_kitchen(
    root: ET.Element,
    world: ET.Element,
    *,
    rng: random.Random,
    randomization_level: str,
) -> list[dict]:
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
    add_visual_box(world, "procedural_range_hood_body", (0.28, 0.88, 1.44), (0.34, 0.08, 0.055), "metal")
    add_visual_box(world, "procedural_range_hood_lip", (0.28, 0.78, 1.36), (0.36, 0.030, 0.025), "handle_metal")
    add_visual_box(world, "procedural_range_hood_stack", (0.28, 1.00, 1.72), (0.16, 0.045, 0.24), "metal")
    add_visual_box(world, "kitchen_ceiling", (0.45, 0.10, 2.22), (1.90, 1.12, 0.025), "scene_wall")

    if not robocasa_assets_available():
        add_box(world, "missing_robocasa_hint_appliance", (0.46, 0.54, 0.86), (0.10, 0.08, 0.12), "dark_mat")
        add_box(world, "missing_robocasa_hint_object", (0.78, 0.45, 0.80), (0.08, 0.05, 0.08), "prop_red")
        return []

    # These are static visual imports. They give the MP4 and GLB a real RoboCasa
    # kitchen look while leaving the catch dynamics controlled by our own scene.
    return _add_random_robocasa_background_assets(
        root,
        rng,
        profile="robocasa_kitchen",
        randomization_level=randomization_level,
    )


def _add_robocasa_profile_scene(
    root: ET.Element,
    world: ET.Element,
    *,
    profile: str,
    rng: random.Random,
    randomization_level: str,
) -> dict:
    scene_info = {
        "environment_type": profile.replace("robocasa_", ""),
        "scene_variant": profile,
        "randomization_level": randomization_level,
        "active_zone_policy": "keep_ball_path_and_robot_workspace_clear; use curated rear/peripheral visual props",
        "visual_design_policy": "coherent room vignette with normalized RobotWin props and soft multi-point lighting",
        "procedural_furniture": [],
        "robocasa_background_assets": [],
        "robotwin_background_assets": [],
    }
    if profile == "robocasa_official_kitchen":
        support_info = _add_robot_support_table(world, profile="robocasa_tabletop", rng=rng)
        scene_info.update(support_info)
        official_info = _add_official_robocasa_kitchen(
            root,
            world,
            rng,
            randomization_level=randomization_level,
        )
        demo_layer_info = _add_official_kitchen_visible_demo_layer(
            root,
            world,
            rng,
            randomization_level=randomization_level,
        )
        scene_info["environment_type"] = "official_kitchen"
        scene_info["official_robocasa_kitchen"] = official_info
        scene_info["official_kitchen_visible_demo_layer"] = demo_layer_info
        scene_info["robocasa_background_assets"] = [
            {
                "slot": "official_kitchen_scene",
                **official_info,
            }
        ] + list(demo_layer_info.get("foreground_assets", []))
    elif profile == "robocasa_kitchen":
        scene_info["support_furniture"] = "kitchen_counter_robot_table"
        scene_info["robocasa_background_assets"] = _add_robocasa_kitchen(
            root,
            world,
            rng=rng,
            randomization_level=randomization_level,
        )
    else:
        support_info = _add_robot_support_table(world, profile=profile, rng=rng)
        furniture = _add_profile_room_furniture(world, profile=profile, rng=rng)
        scene_info.update(support_info)
        scene_info["procedural_furniture"] = furniture
        if robocasa_assets_available():
            scene_info["robocasa_background_assets"] = _add_random_robocasa_background_assets(
                root,
                rng,
                profile=profile,
                randomization_level=randomization_level,
            )
        else:
            add_box(world, "missing_robocasa_hint_object", (0.78, 0.82, 0.82), (0.08, 0.05, 0.08), "prop_red")
        scene_info["robotwin_background_assets"] = _add_robotwin_background_assets(
            root,
            rng,
            profile=profile,
            randomization_level=randomization_level,
        )
    return scene_info


def add_variant_xml(
    root,
    variant_name: str,
    *,
    lighting_intensity: float,
    floor_jitter: float,
    camera_jitter,
    random_seed: int | None = None,
    scene_randomization_level: str = "balanced",
) -> dict:
    spec = VARIANTS[variant_name]
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")

    rng = random.Random(0 if random_seed is None else int(random_seed))
    is_robocasa_scene = variant_name in ROBOCASA_SCENE_VARIANTS
    texture_profile = _texture_profile(variant_name, rng) if is_robocasa_scene else {}
    floor_rgba = tuple(min(1.0, max(0.0, c + floor_jitter)) if i < 3 else c for i, c in enumerate(spec.floor_rgba))
    if is_robocasa_scene:
        ET.SubElement(
            asset,
            "texture",
            name="robocasa_scene_skybox",
            type="skybox",
            builtin="gradient",
            rgb1="0.84 0.85 0.82",
            rgb2="0.66 0.66 0.62",
            width="256",
            height="256",
        )
    if is_robocasa_scene and robocasa_assets_available():
        add_textured_material(asset, "scene_floor", texture_profile["floor"], rgba=floor_rgba, texrepeat=(3.6, 3.6), roughness=0.72)
        add_textured_material(asset, "scene_wall", texture_profile["wall"], rgba=spec.wall_rgba, texrepeat=(2.4, 2.0), roughness=0.84)
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
    if is_robocasa_scene and robocasa_assets_available():
        add_textured_material(asset, "countertop", texture_profile["countertop"], rgba=_jitter_rgba((0.82, 0.80, 0.76, 1), rng, 0.05), texrepeat=(2.8, 1.4), roughness=0.54)
        add_textured_material(asset, "cabinet_front", texture_profile["cabinet_front"], rgba=_jitter_rgba((0.78, 0.72, 0.64, 1), rng, 0.06), texrepeat=(1.4, 1.4), roughness=0.66)
        add_textured_material(asset, "backsplash_tile", texture_profile["backsplash_tile"], rgba=_jitter_rgba((0.95, 0.94, 0.90, 1), rng, 0.04), texrepeat=(5.5, 1.4), roughness=0.60)
        add_textured_material(asset, "accent_wood", texture_profile["accent_wood"], rgba=_jitter_rgba((0.58, 0.50, 0.40, 1), rng, 0.06), texrepeat=(1.2, 1.2), roughness=0.70)
    else:
        add_material(asset, "countertop", (0.50, 0.48, 0.42, 1), roughness=0.55)
        add_material(asset, "cabinet_front", (0.76, 0.70, 0.60, 1), roughness=0.70)
        add_material(asset, "backsplash_tile", (0.92, 0.90, 0.84, 1), roughness=0.60)
        add_material(asset, "accent_wood", (0.50, 0.36, 0.22, 1), roughness=0.68)
    add_material(asset, "drawer_gap", (0.055, 0.048, 0.042, 1), roughness=0.80)
    add_material(asset, "handle_metal", (0.55, 0.52, 0.47, 1), roughness=0.35)
    add_material(asset, "warm_light", (1.0, 0.82, 0.48, 1), roughness=0.25)
    add_material(asset, "window_glass", (0.45, 0.62, 0.78, 0.55), roughness=0.15)
    add_material(asset, "robotwin_prop_mat", (0.56, 0.52, 0.46, 1), roughness=0.62)
    add_material(asset, "wall_art_blue", (0.38, 0.53, 0.62, 1), roughness=0.74)
    add_material(asset, "wall_art_warm", (0.72, 0.58, 0.42, 1), roughness=0.78)
    add_material(asset, "plant_leaf", (0.20, 0.42, 0.28, 1), roughness=0.82)
    add_material(asset, "lamp_shade", (0.94, 0.86, 0.70, 1), roughness=0.58)

    ET.SubElement(world, "geom", name="floor", type="plane", size="3.0 3.0 0.05", material="scene_floor", condim="3", friction="1.2 0.005 0.0001")
    if variant_name != "robocasa_official_kitchen":
        add_box(world, "back_wall", (0.45, 1.10, 1.10), (1.9, 0.03, 1.10), "scene_wall")
        add_box(world, "left_wall", (-0.95, 0.10, 1.10), (0.03, 1.10, 1.10), "scene_wall")

    scene_randomization_info: dict = {
        "environment_type": variant_name,
        "scene_variant": variant_name,
        "randomization_level": scene_randomization_level,
        "texture_profile": texture_profile,
        "robocasa_background_assets": [],
        "robotwin_background_assets": [],
    }
    if is_robocasa_scene:
        scene_randomization_info = _add_robocasa_profile_scene(
            root,
            world,
            profile=variant_name,
            rng=rng,
            randomization_level=scene_randomization_level,
        )
        scene_randomization_info["texture_profile"] = texture_profile
    elif variant_name in {"clean_lab", "cluttered_lab"}:
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

    if is_robocasa_scene:
        lighting_info = _configure_robocasa_lighting(
            root,
            world,
            lighting_intensity,
            profile=variant_name,
            rng=rng,
        )
        scene_randomization_info["lighting"] = lighting_info
    else:
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

    cam_pos = tuple(float(a + b) for a, b in zip(spec.camera_pos, camera_jitter))
    ET.SubElement(
        world,
        "camera",
        name="main_camera",
        pos=" ".join(f"{v:.5f}" for v in cam_pos),
        xyaxes=camera_xyaxes(cam_pos, spec.camera_lookat),
        fovy=f"{spec.fovy:.3f}",
    )
    if is_robocasa_scene:
        closeup_pos = (0.92, -0.22, 1.30)
        closeup_lookat = (0.44, 0.0, 1.20)
    else:
        closeup_pos = (0.74, -0.66, 0.76)
        closeup_lookat = (0.46, 0.0, 0.52)
    ET.SubElement(
        world,
        "camera",
        name="closeup_camera",
        pos=" ".join(f"{v:.5f}" for v in closeup_pos),
        xyaxes=camera_xyaxes(closeup_pos, closeup_lookat),
        fovy="35",
    )
    camera_info = {"pos": list(cam_pos), "lookat": list(spec.camera_lookat), "fovy": spec.fovy}
    if is_robocasa_scene:
        camera_info["robocasa_background_random_seed"] = None if random_seed is None else int(random_seed)
        camera_info["robocasa_background_assets"] = scene_randomization_info.get("robocasa_background_assets", [])
        camera_info["robotwin_background_assets"] = scene_randomization_info.get("robotwin_background_assets", [])
        camera_info["scene_randomization"] = scene_randomization_info
    return camera_info
