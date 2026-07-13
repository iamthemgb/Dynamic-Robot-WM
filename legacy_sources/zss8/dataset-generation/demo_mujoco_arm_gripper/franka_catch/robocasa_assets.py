from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# RoboCasa kitchen assets are read-only shared assets downloaded by a collaborator
# (Michael). We reference them in place (MJCF file= paths) and NEVER copy/write
# there. Override with ROBOCASA_ASSETS_ROOT if the shared path ever moves.
DEFAULT_ROBOCASA_ROOT = Path("/gpfs/radev/project/sous/mzl7/robocasa/robocasa/models/assets")


def robocasa_assets_root() -> Path:
    env = os.environ.get("ROBOCASA_ASSETS_ROOT")
    if env:
        return Path(env)
    return DEFAULT_ROBOCASA_ROOT


def robocasa_assets_available() -> bool:
    root = robocasa_assets_root()
    return (root / "fixtures").exists() and (root / "textures").exists()


# --- dynamic asset discovery ------------------------------------------------
# Instance folders (e.g. Blender001) differ between RoboCasa snapshots, so we
# discover valid model.xml files at runtime instead of hard-coding IDs.

def _pick(rng, items, n):
    items = list(items)
    if not items or n <= 0:
        return []
    if n >= len(items):
        order = rng.permutation(len(items))
        return [items[i] for i in order]
    idx = rng.choice(len(items), size=n, replace=False)
    return [items[int(i)] for i in idx]


def discover_fixture_models(category: str, rng, n: int = 1) -> list[str]:
    """Return up to n relative model.xml paths for a fixture category.

    Paths are relative to ``robocasa_assets_root()`` so they can be fed straight
    to :func:`append_robocasa_visual_model`.
    """
    root = robocasa_assets_root()
    cat_dir = root / "fixtures" / category
    if not cat_dir.exists():
        return []
    models = []
    for inst in sorted(cat_dir.iterdir()):
        xml = inst / "model.xml"
        if xml.exists():
            models.append(f"fixtures/{category}/{inst.name}/model.xml")
    return _pick(rng, models, n)


def discover_object_models(rng, n: int = 3, groups=("lightwheel", "aigen_objs")) -> list[str]:
    """Return up to n relative model.xml paths from the RoboCasa object banks."""
    root = robocasa_assets_root()
    found: list[str] = []
    for group in groups:
        gdir = root / "objects" / group
        if not gdir.exists():
            continue
        for xml in gdir.rglob("model.xml"):
            found.append(str(xml.relative_to(root)))
    return _pick(rng, found, n)


def _as_pos(values) -> str:
    return " ".join(f"{float(v):.5f}" for v in values)


def _prefix(name: str, instance: str) -> str:
    return f"{instance}_{name}"


class _UnresolvedAsset(Exception):
    pass


def _absolute_asset_path(source_xml: Path, file_value: str) -> str:
    """Resolve a fixture asset ``file=`` to a real path on our shared root.

    Some RoboCasa fixture XMLs bake in absolute texture paths from the machine
    that authored them (e.g. /home/.../robosuite/models/assets/textures/...).
    We rebase any such path onto our read-only robocasa root using the
    ``models/assets`` / ``assets`` marker. Raises :class:`_UnresolvedAsset` if the
    file cannot be found anywhere so the caller can skip the fixture cleanly.
    """
    path = Path(file_value)
    root = robocasa_assets_root()
    if not path.is_absolute():
        cand = (source_xml.parent / path).resolve()
        if cand.exists():
            return str(cand)
    elif path.exists():
        return str(path)

    s = str(path).replace("\\", "/")
    for marker in ("/models/assets/", "/assets/"):
        if marker in s:
            tail = s.split(marker, 1)[1]
            cand = root / tail
            if cand.exists():
                return str(cand)
    # last resort: match by basename under the textures tree
    cand = root / "textures" / path.name
    if cand.exists():
        return str(cand)
    raise _UnresolvedAsset(file_value)


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

    source_xml = robocasa_assets_root() / rel_model_xml
    if not source_xml.exists():
        return False

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")

    source_root = ET.parse(source_xml).getroot()
    try:
        mesh_names, _, material_names = _copy_assets(asset, source_root, source_xml, instance)
    except _UnresolvedAsset:
        return False  # baked-in asset path we cannot find on the shared root -> skip fixture
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
