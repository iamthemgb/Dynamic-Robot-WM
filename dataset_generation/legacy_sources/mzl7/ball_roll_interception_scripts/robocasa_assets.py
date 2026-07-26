from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from .yaml_scene import robocasa_assets_root


STYLE_MODEL_BUILDERS = {
    "sink": lambda model: f"fixtures/sinks/{model}/model.xml",
    "dishwasher": lambda model: f"fixtures/dishwashers/{model}/model.xml",
    "microwave": lambda model: f"fixtures/microwaves/{model}/model.xml",
    "oven": lambda model: f"fixtures/ovens/{model}/model.xml",
    "stove": lambda model: f"fixtures/stoves/{model}/model.xml",
    "stove_wide": lambda model: f"fixtures/stoves/{model}/model.xml",
    "stovetop": lambda model: f"fixtures/stovetops/{model}/model.xml",
    "fridge_bottom_freezer": lambda model: f"fixtures/fridges/{model}/model.xml",
    "fridge_french_door": lambda model: f"fixtures/fridges/{model}/model.xml",
    "fridge_side_by_side": lambda model: f"fixtures/fridges/{model}/model.xml",
    "dish_rack": lambda model: f"lightwheel/dish_rack/{model}/model.xml",
    "coffee_machine": lambda model: f"fixtures/coffee_machines/{model}/model.xml",
    "blender": lambda model: f"fixtures/blenders/{model}/model.xml",
    "blender_lid": lambda model: f"fixtures/blender_lids/{model}/model.xml",
    "toaster": lambda model: f"fixtures/toasters/{model}/model.xml",
    "toaster_oven": lambda model: f"fixtures/toaster_ovens/{model}/model.xml",
    "electric_kettle": lambda model: f"fixtures/electric_kettles/{model}/model.xml",
    "stand_mixer": lambda model: f"fixtures/stand_mixers/{model}/model.xml",
    "window": lambda model: f"fixtures/windows/{model}/model.xml",
}


def robocasa_assets_available() -> bool:
    root = robocasa_assets_root()
    return (root / "fixtures").exists() and ((root / "lightwheel").exists() or (root / "objects" / "lightwheel").exists())


def style_model_rel_path(role: str | None, model: str | None) -> str | None:
    if role is None or model is None:
        return None
    builder = STYLE_MODEL_BUILDERS.get(role)
    if builder is None:
        return None
    return builder(model)


def _resolve_model_xml(rel_model_xml: str) -> Path:
    direct = Path(rel_model_xml)
    if direct.is_absolute() and direct.exists():
        return direct
    root = robocasa_assets_root()
    candidate = root / rel_model_xml
    if candidate.exists():
        return candidate
    if rel_model_xml.startswith("lightwheel/"):
        candidate = root / "objects" / rel_model_xml
        if candidate.exists():
            return candidate
    return direct


def _prefix(name: str, instance: str) -> str:
    return f"{instance}_{name}"


def _absolute_asset_path(source_xml: Path, file_value: str) -> str:
    path = Path(file_value)
    if path.is_absolute():
        return str(path)
    return str((source_xml.parent / path).resolve())


def _copy_assets(target_asset: ET.Element, source_root: ET.Element, source_xml: Path, instance: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    source_asset = source_root.find("asset")
    mesh_names: dict[str, str] = {}
    texture_names: dict[str, str] = {}
    material_names: dict[str, str] = {}
    if source_asset is None:
        return mesh_names, texture_names, material_names
    for child in source_asset:
        if child.tag == "mesh" and child.get("name"):
            mesh_names[child.get("name", "")] = _prefix(child.get("name", ""), instance)
        elif child.tag == "texture" and child.get("name"):
            texture_names[child.get("name", "")] = _prefix(child.get("name", ""), instance)
        elif child.tag == "material" and child.get("name"):
            material_names[child.get("name", "")] = _prefix(child.get("name", ""), instance)
    for child in source_asset:
        if child.tag not in {"mesh", "texture", "material"}:
            continue
        copied = copy.deepcopy(child)
        name = copied.get("name")
        if child.tag == "mesh" and name:
            copied.set("name", mesh_names[name])
        elif child.tag == "texture" and name:
            copied.set("name", texture_names[name])
        elif child.tag == "material" and name:
            copied.set("name", material_names[name])
            texture = copied.get("texture")
            if texture in texture_names:
                copied.set("texture", texture_names[texture])
        file_value = copied.get("file")
        if file_value:
            copied.set("file", _absolute_asset_path(source_xml, file_value))
        target_asset.append(copied)
    return mesh_names, texture_names, material_names


def _is_visual_geom(geom: ET.Element) -> bool:
    geom_class = geom.get("class")
    if geom_class == "visual":
        return True
    if geom_class in {"collision", "region", "spawn"}:
        return False
    return bool(geom.get("mesh") and geom.get("material"))


def _copy_visual_body(source_body: ET.Element, instance: str, mesh_names: dict[str, str], material_names: dict[str, str]) -> ET.Element | None:
    copied_body = ET.Element("body")
    for key, value in source_body.attrib.items():
        copied_body.set(key, _prefix(value, instance) if key == "name" else value)
    for child in source_body:
        if child.tag == "body":
            nested = _copy_visual_body(child, instance, mesh_names, material_names)
            if nested is not None:
                copied_body.append(nested)
        elif child.tag == "geom" and _is_visual_geom(child):
            copied_geom = copy.deepcopy(child)
            name = copied_geom.get("name")
            if name:
                copied_geom.set("name", _prefix(name, instance))
            mesh = copied_geom.get("mesh")
            if mesh in mesh_names:
                copied_geom.set("mesh", mesh_names[mesh])
            material = copied_geom.get("material")
            if material in material_names:
                copied_geom.set("material", material_names[material])
            copied_geom.attrib.pop("class", None)
            copied_geom.attrib.pop("density", None)
            copied_geom.attrib.pop("mass", None)
            copied_geom.set("contype", "0")
            copied_geom.set("conaffinity", "0")
            copied_geom.set("group", "1")
            copied_body.append(copied_geom)
    return copied_body if len(copied_body) else None


def append_robocasa_visual_model(root: ET.Element, rel_model_xml: str, *, instance: str, pos: tuple[float, float, float], euler: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> bool:
    source_xml = _resolve_model_xml(rel_model_xml)
    if not source_xml.exists():
        return False
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    source_root = ET.parse(source_xml).getroot()
    mesh_names, _, material_names = _copy_assets(asset, source_root, source_xml, instance)
    source_world = source_root.find("worldbody")
    if source_world is None:
        return False
    wrapper = ET.SubElement(
        world,
        "body",
        name=f"{instance}_wrapper",
        pos=" ".join(f"{float(v):.5f}" for v in pos),
        euler=" ".join(f"{float(v):.5f}" for v in euler),
    )
    imported = 0
    for source_body in source_world.findall("body"):
        copied = _copy_visual_body(source_body, instance, mesh_names, material_names)
        if copied is not None:
            wrapper.append(copied)
            imported += 1
    if imported == 0:
        world.remove(wrapper)
        return False
    return True

