from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def robocasa_assets_root() -> Path:
    assets_root = os.environ.get("ROBOCASA_ASSETS_ROOT", "").strip()
    if assets_root:
        return Path(assets_root).expanduser().resolve()

    robocasa_root = os.environ.get("ROBOCASA_ROOT", "").strip()
    if robocasa_root:
        return Path(robocasa_root).expanduser().resolve() / "robocasa" / "models" / "assets"

    return Path(__file__).resolve().parents[1] / "third_party" / "robocasa" / "robocasa" / "models" / "assets"


def _resolve_model_xml(rel_model_xml: str) -> Path:
    root = robocasa_assets_root()
    source_xml = root / rel_model_xml
    if source_xml.exists():
        return source_xml
    if rel_model_xml.startswith("lightwheel/"):
        return root / "objects" / rel_model_xml
    return source_xml


def robocasa_assets_available() -> bool:
    root = robocasa_assets_root()
    return (root / "fixtures").exists() and (
        (root / "lightwheel").exists()
        or (root / "objects" / "lightwheel").exists()
    )


def _as_pos(values) -> str:
    return " ".join(f"{float(v):.5f}" for v in values)


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


def _copy_visual_body(
    source_body: ET.Element,
    instance: str,
    mesh_names: dict[str, str],
    material_names: dict[str, str],
) -> ET.Element | None:
    copied_body = ET.Element("body")
    for key, value in source_body.attrib.items():
        if key == "name":
            copied_body.set(key, _prefix(value, instance))
        else:
            copied_body.set(key, value)

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


def append_robocasa_visual_model(
    root: ET.Element,
    rel_model_xml: str,
    *,
    instance: str,
    pos: tuple[float, float, float],
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bool:
    """Append a RoboCasa MJCF model as fixed, visual-only background geometry.

    RoboCasa fixture/object XMLs often include joints, actuators, collision geoms,
    sites, and task regions. For this demo we only need textured visual meshes, so
    this importer copies assets and visual geoms under a fixed wrapper body.
    """

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
        pos=_as_pos(pos),
        euler=_as_pos(euler),
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
