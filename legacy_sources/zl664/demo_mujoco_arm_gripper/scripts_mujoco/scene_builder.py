from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .assets import AssetInfo, locate_franka_asset, locate_franka_nohand_asset
from .utils import camera_xyaxes, sphere_mass
from .variants import add_material, add_variant_xml, rgba_str


ROOT_DIR = Path(__file__).resolve().parents[1]
ROBOTIQ_2F85_DIR = ROOT_DIR / "third_party" / "mujoco_menagerie" / "robotiq_2f85"
ROBOTIQ_2F85_XML = ROBOTIQ_2F85_DIR / "2f85.xml"
ROBOTIQ_PREFIX = "rq_"
ROBOTIQ_MOUNT_POS = (0.0, 0.0, 0.016)
# Roll the menagerie gripper about its approach axis so the Robotiq jaws close
# laterally in the scene instead of acting as a top-bottom clamp.
ROBOTIQ_MOUNT_QUAT = (0.5, 0.5, -0.5, -0.5)
ROBOTIQ_GRASP_SITE = "robotiq_grasp_center_site"
ROBOTIQ_LEFT_PAD_SITE = "robotiq_left_pad_site"
ROBOTIQ_RIGHT_PAD_SITE = "robotiq_right_pad_site"
TABLETOP_ROBOT_VARIANTS = {
    "robocasa_kitchen",
    "robocasa_tabletop",
    "robocasa_lab",
    "robocasa_workbench",
    "robocasa_storage",
    "robocasa_official_kitchen",
}
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
    ball_initial_position: tuple[float, float, float]
    ball_initial_velocity: tuple[float, float, float]
    lighting_intensity: float
    camera_jitter: tuple[float, float, float]
    floor_material_jitter: float
    robot_base_position: tuple[float, float, float]
    robot_base_euler: tuple[float, float, float]
    tabletop_height: float | None
    release_time_s: float = 0.60
    catch_center_z: float = 0.50
    offscreen_width: int = 1920
    offscreen_height: int = 1080
    planned_intercept_time_s: float | None = None
    planned_intercept_position: tuple[float, float, float] | None = None
    interception_subfamily: str = "direct_projectile_interception"
    expected_contact_sequence: tuple[str, ...] = ()
    interception_surface_specs: tuple[dict, ...] = ()
    surface_contact_groups: tuple[dict, ...] = ()
    scene_randomization_level: str = "balanced"
    projectile: dict | None = None


@dataclass
class EpisodeBundle:
    model: mujoco.MjModel
    data: mujoco.MjData
    xml: str
    asset_info: AssetInfo
    camera_pose: dict


def sample_episode(rng: np.random.Generator, variant: str, seed: int, *, fps: int, duration: float) -> EpisodeSample:
    radius = float(rng.uniform(0.0245, 0.0255))
    density = float(rng.uniform(36.0, 48.0))
    mass = sphere_mass(radius, density)
    tabletop_variant = variant in TABLETOP_ROBOT_VARIANTS
    height_offset = TABLETOP_ROBOT_Z if tabletop_variant else 0.0
    xy = rng.uniform((0.43, -0.025), (0.48, 0.025))
    z = float(rng.uniform(1.12 + height_offset, 1.17 + height_offset))
    color = tuple(float(v) for v in rng.uniform(0.15, 0.95, size=3)) + (1.0,)
    camera_jitter = tuple(float(v) for v in rng.normal(0.0, (0.035, 0.035, 0.025)))
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
        ball_initial_position=(float(xy[0]), float(xy[1]), z),
        ball_initial_velocity=(0.0, 0.0, 0.0),
        lighting_intensity=float(rng.uniform(0.85, 1.18)),
        camera_jitter=camera_jitter,
        floor_material_jitter=float(rng.uniform(-0.04, 0.04)),
        robot_base_position=(0.0, 0.0, TABLETOP_ROBOT_Z) if tabletop_variant else (0.0, 0.0, 0.0),
        robot_base_euler=(0.0, 0.0, TABLETOP_ROBOT_YAW) if tabletop_variant else (0.0, 0.0, 0.0),
        tabletop_height=TABLETOP_ROBOT_Z if tabletop_variant else None,
        catch_center_z=0.50 + height_offset,
    )


def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.SubElement(root, tag)
    return child


def _insert_before_worldbody(root: ET.Element, child: ET.Element) -> None:
    children = list(root)
    for index, existing in enumerate(children):
        if existing.tag == "worldbody":
            root.insert(index, child)
            return
    root.append(child)


def _walk_xml(element: ET.Element):
    yield element
    for child in element:
        yield from _walk_xml(child)


def _implicit_mesh_name(mesh_element: ET.Element) -> str:
    return mesh_element.get("name") or Path(mesh_element.get("file", "")).stem


def _collect_asset_names(xml_root: ET.Element) -> tuple[list[str], list[str]]:
    mesh_names: list[str] = []
    material_names: list[str] = []
    asset = xml_root.find("asset")
    if asset is None:
        return mesh_names, material_names
    for child in asset:
        if child.tag == "mesh":
            mesh_names.append(_implicit_mesh_name(child))
        elif child.tag == "material" and child.get("name"):
            material_names.append(child.get("name"))
    return mesh_names, material_names


def _prefix_menagerie_tree(
    element: ET.Element,
    prefix: str,
    mesh_names: list[str],
    material_names: list[str],
) -> ET.Element:
    prefixed = copy.deepcopy(element)
    for item in _walk_xml(prefixed):
        for attr_name in ("class", "childclass"):
            attr_value = item.get(attr_name)
            if attr_value:
                item.set(attr_name, prefix + attr_value)
        name = item.get("name")
        if name:
            item.set("name", prefix + name)
        for attr_name in ("joint", "joint1", "joint2", "body1", "body2", "tendon", "site"):
            attr_value = item.get(attr_name)
            if attr_value:
                item.set(attr_name, prefix + attr_value)
        mesh_name = item.get("mesh")
        if mesh_name and mesh_name in mesh_names:
            item.set("mesh", prefix + mesh_name)
        material_name = item.get("material")
        if material_name and material_name in material_names:
            item.set("material", prefix + material_name)
    return prefixed


def _prefixed_asset_child(
    child: ET.Element,
    asset_dir: Path,
    prefix: str,
    mesh_names: list[str],
    material_names: list[str],
) -> ET.Element:
    prefixed = _prefix_menagerie_tree(child, prefix, mesh_names, material_names)
    if child.tag == "mesh":
        prefixed.set("name", prefix + _implicit_mesh_name(child))
    mesh_file = prefixed.get("file")
    if mesh_file:
        prefixed.set("file", str((asset_dir / mesh_file).resolve()))
    return prefixed


def _body_by_name(root: ET.Element, name: str) -> ET.Element:
    body = _ensure_child(root, "worldbody").find(f".//body[@name='{name}']")
    if body is None:
        raise ValueError(f"Could not find body {name!r}.")
    return body


def _disable_franka_link_collisions(root: ET.Element) -> None:
    for geom in _ensure_child(root, "worldbody").findall(".//geom"):
        geom_class = geom.get("class", "")
        if "panda/collision" in geom_class:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def _set_robot_compiler_paths(root: ET.Element, asset_info: AssetInfo) -> None:
    compiler = _ensure_child(root, "compiler")
    compiler.set("angle", "radian")
    if asset_info.asset_dir.exists():
        compiler.set("meshdir", str(asset_info.asset_dir))


def _remove_source_keyframes(root: ET.Element) -> None:
    # The source Franka keyframes are sized for the unmodified robot. We add a
    # free ball and sometimes a Robotiq hand, so MuJoCo 3.3 rejects them.
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)


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
    projectile = getattr(sample, "projectile", None)
    if projectile:
        _add_robocasa_projectile(root, sample, projectile)
        return

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


def _parse_vector(raw: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return default
    return tuple(float(v) for v in raw.split())


def _set_scaled_vector_attr(element: ET.Element, attr: str, scale: float) -> None:
    raw = element.get(attr)
    if not raw:
        return
    values = [float(v) * float(scale) for v in raw.split()]
    element.set(attr, " ".join(f"{v:.9g}" for v in values))


def _absolute_xml_asset_path(source_xml: Path, file_value: str) -> str:
    path = Path(file_value)
    if path.is_absolute():
        return str(path)
    return str((source_xml.parent / path).resolve())


def _copy_projectile_assets(
    target_asset: ET.Element,
    source_root: ET.Element,
    source_xml: Path,
    *,
    prefix: str,
    scale: float,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    source_asset = source_root.find("asset")
    mesh_names: dict[str, str] = {}
    texture_names: dict[str, str] = {}
    material_names: dict[str, str] = {}
    if source_asset is None:
        return mesh_names, texture_names, material_names

    for child in source_asset:
        name = child.get("name")
        if child.tag == "mesh" and name:
            mesh_names[name] = f"{prefix}_{name}"
        elif child.tag == "texture" and name:
            texture_names[name] = f"{prefix}_{name}"
        elif child.tag == "material" and name:
            material_names[name] = f"{prefix}_{name}"

    for child in source_asset:
        if child.tag not in {"mesh", "texture", "material"}:
            continue
        copied = copy.deepcopy(child)
        name = copied.get("name")
        if child.tag == "mesh" and name:
            copied.set("name", mesh_names[name])
            existing_scale = _parse_vector(copied.get("scale"), (1.0, 1.0, 1.0))
            copied.set("scale", " ".join(f"{float(v) * float(scale):.9g}" for v in existing_scale[:3]))
        elif child.tag == "texture" and name:
            copied.set("name", texture_names[name])
        elif child.tag == "material" and name:
            copied.set("name", material_names[name])
            texture = copied.get("texture")
            if texture in texture_names:
                copied.set("texture", texture_names[texture])

        file_value = copied.get("file")
        if file_value:
            copied.set("file", _absolute_xml_asset_path(source_xml, file_value))
        target_asset.append(copied)

    return mesh_names, texture_names, material_names


def _is_projectile_visual_geom(geom: ET.Element) -> bool:
    geom_class = geom.get("class")
    if geom_class == "visual":
        return True
    if geom_class in {"collision", "region", "spawn"}:
        return False
    return bool(geom.get("mesh") and geom.get("material"))


def _copy_projectile_visual_body(
    source_body: ET.Element,
    *,
    prefix: str,
    scale: float,
    mesh_names: dict[str, str],
    material_names: dict[str, str],
) -> ET.Element | None:
    copied_body = ET.Element("body")
    for key, value in source_body.attrib.items():
        if key == "name":
            copied_body.set(key, f"{prefix}_{value}")
        else:
            copied_body.set(key, value)
    _set_scaled_vector_attr(copied_body, "pos", scale)

    for child in source_body:
        if child.tag == "body":
            nested = _copy_projectile_visual_body(
                child,
                prefix=prefix,
                scale=scale,
                mesh_names=mesh_names,
                material_names=material_names,
            )
            if nested is not None:
                copied_body.append(nested)
        elif child.tag == "geom" and _is_projectile_visual_geom(child):
            copied_geom = copy.deepcopy(child)
            name = copied_geom.get("name")
            if name:
                copied_geom.set("name", f"{prefix}_{name}")
            mesh = copied_geom.get("mesh")
            if mesh in mesh_names:
                copied_geom.set("mesh", mesh_names[mesh])
            material = copied_geom.get("material")
            if material in material_names:
                copied_geom.set("material", material_names[material])
            _set_scaled_vector_attr(copied_geom, "pos", scale)
            copied_geom.attrib.pop("class", None)
            copied_geom.attrib.pop("density", None)
            copied_geom.attrib.pop("mass", None)
            copied_geom.set("contype", "0")
            copied_geom.set("conaffinity", "0")
            copied_geom.set("group", "1")
            copied_body.append(copied_geom)

    return copied_body if len(copied_body) else None


def _add_robocasa_projectile(root: ET.Element, sample: EpisodeSample, projectile: dict) -> None:
    source_xml = Path(str(projectile["model_xml"])).expanduser().resolve()
    if not source_xml.exists():
        raise FileNotFoundError(f"Projectile model XML does not exist: {source_xml}")

    scale = float(projectile.get("scale", 1.0))
    prefix = str(projectile.get("instance", "catch_projectile"))
    bbox_pos = tuple(float(v) for v in projectile.get("bbox_pos", (0.0, 0.0, 0.0)))
    collision_shape = str(projectile.get("collision_shape", "box"))
    collision_size = tuple(float(v) for v in projectile.get("collision_size", (sample.ball_radius,)))

    asset = _ensure_child(root, "asset")
    add_material(asset, "catch_projectile_collision_mat", (0.0, 0.0, 0.0, 0.0), roughness=0.45)
    source_root = ET.parse(source_xml).getroot()
    mesh_names, _, material_names = _copy_projectile_assets(
        asset,
        source_root,
        source_xml,
        prefix=prefix,
        scale=scale,
    )

    world = _ensure_child(root, "worldbody")
    body = ET.SubElement(world, "body", name="catch_ball", pos=" ".join(f"{v:.6f}" for v in sample.ball_initial_position))
    ET.SubElement(body, "freejoint", name="ball_freejoint")
    collision_attrs = {
        "name": "catch_ball_geom",
        "type": collision_shape,
        "size": " ".join(f"{float(v):.6f}" for v in collision_size),
        "mass": f"{sample.ball_mass:.8f}",
        "material": "catch_projectile_collision_mat",
        "condim": "4",
        "friction": "5.0 0.01 0.001",
        "solref": "0.004 1",
        "solimp": "0.95 0.99 0.001",
        "group": "3",
    }
    ET.SubElement(body, "geom", **collision_attrs)

    visual_wrapper = ET.SubElement(
        body,
        "body",
        name=f"{prefix}_visual_wrapper",
        pos=" ".join(f"{-float(v) * scale:.9g}" for v in bbox_pos),
    )
    source_world = source_root.find("worldbody")
    if source_world is None:
        raise ValueError(f"Projectile model XML has no worldbody: {source_xml}")
    imported = 0
    for source_body in source_world.findall("body"):
        copied = _copy_projectile_visual_body(
            source_body,
            prefix=prefix,
            scale=scale,
            mesh_names=mesh_names,
            material_names=material_names,
        )
        if copied is not None:
            visual_wrapper.append(copied)
            imported += 1
    if imported == 0:
        raise ValueError(f"Projectile model XML had no visual geoms: {source_xml}")


def _add_interception_surfaces(root: ET.Element, sample: EpisodeSample) -> None:
    if not sample.interception_surface_specs:
        return
    asset = _ensure_child(root, "asset")
    material_names = set()
    for spec in sample.interception_surface_specs:
        material_name = str(spec.get("material", "interception_surface_mat"))
        if material_name not in material_names:
            rgba = spec.get("material_rgba", (0.18, 0.36, 0.74, 1.0))
            roughness = float(spec.get("roughness", 0.55))
            add_material(asset, material_name, rgba, roughness=roughness)
            material_names.add(material_name)

    world = _ensure_child(root, "worldbody")
    for spec in sample.interception_surface_specs:
        attrs = {
            "name": str(spec["name"]),
            "type": str(spec.get("type", "box")),
            "pos": " ".join(f"{float(v):.6f}" for v in spec.get("pos", (0.0, 0.0, 0.0))),
            "size": " ".join(f"{float(v):.6f}" for v in spec.get("size", (0.1, 0.1, 0.01))),
            "material": str(spec.get("material", "interception_surface_mat")),
            "contype": str(spec.get("contype", "1")),
            "conaffinity": str(spec.get("conaffinity", "1")),
            "condim": str(spec.get("condim", "4")),
            "friction": str(spec.get("friction", "1.2 0.02 0.001")),
            "solref": str(spec.get("solref", "0.006 0.35")),
            "solimp": str(spec.get("solimp", "0.94 0.995 0.001")),
            "priority": str(spec.get("priority", "1")),
        }
        if "rgba" in spec:
            attrs["rgba"] = " ".join(f"{float(v):.4f}" for v in spec["rgba"])
            attrs.pop("material", None)
        if "euler" in spec:
            attrs["euler"] = " ".join(f"{float(v):.9f}" for v in spec["euler"])
        if "quat" in spec:
            attrs["quat"] = " ".join(f"{float(v):.9f}" for v in spec["quat"])
        if "group" in spec:
            attrs["group"] = str(spec["group"])
        ET.SubElement(world, "geom", **attrs)


def _add_wrist_camera(
    root: ET.Element,
    *,
    parent_body_name: str = WRIST_CAMERA_PARENT_BODY,
    local_pos: tuple[float, float, float] = WRIST_CAMERA_LOCAL_POS,
    local_xyaxes: str = WRIST_CAMERA_LOCAL_XYAXES,
) -> dict:
    asset = _ensure_child(root, "asset")
    add_material(asset, "wrist_camera_body_mat", (0.03, 0.035, 0.04, 1.0), roughness=0.38)
    add_material(asset, "wrist_camera_lens_mat", (0.02, 0.02, 0.025, 1.0), roughness=0.18)

    world = _ensure_child(root, "worldbody")
    parent_body = world.find(f".//body[@name='{parent_body_name}']")
    if parent_body is None:
        raise ValueError(f"Cannot add wrist camera: body '{parent_body_name}' was not found.")

    pos = " ".join(f"{v:.6f}" for v in local_pos)
    ET.SubElement(
        parent_body,
        "camera",
        name=WRIST_CAMERA_NAME,
        pos=pos,
        xyaxes=local_xyaxes,
        fovy=f"{WRIST_CAMERA_FOVY:.3f}",
    )

    visual_body = ET.SubElement(
        parent_body,
        "body",
        name=WRIST_CAMERA_VISUAL_BODY,
        pos=pos,
        xyaxes=local_xyaxes,
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
        "parent_body": parent_body_name,
        "visual_body": WRIST_CAMERA_VISUAL_BODY,
        "local_pos": list(local_pos),
        "local_xyaxes": [float(v) for v in local_xyaxes.split()],
        "local_forward": list(WRIST_CAMERA_LOCAL_FORWARD),
        "local_up": list(WRIST_CAMERA_LOCAL_UP),
        "fovy": WRIST_CAMERA_FOVY,
    }


def _set_world_camera(root: ET.Element, name: str, *, pos, lookat, fovy: float) -> dict:
    world = _ensure_child(root, "worldbody")
    camera = world.find(f"./camera[@name='{name}']")
    if camera is None:
        camera = ET.SubElement(world, "camera", name=name)
    pos_values = tuple(float(v) for v in pos)
    lookat_values = tuple(float(v) for v in lookat)
    camera.set("pos", " ".join(f"{v:.5f}" for v in pos_values))
    camera.set("xyaxes", camera_xyaxes(pos_values, lookat_values))
    camera.set("fovy", f"{float(fovy):.3f}")
    return {"pos": list(pos_values), "lookat": list(lookat_values), "fovy": float(fovy)}


def _predicted_ballistic_envelope(
    sample: EpisodeSample,
    camera_ball_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if camera_ball_positions is not None:
        positions_array = np.asarray(camera_ball_positions, dtype=np.float64).reshape(-1, 3)
        if len(positions_array) >= 2:
            return positions_array.min(axis=0), positions_array.max(axis=0), positions_array

    times = np.linspace(0.0, float(sample.duration_s), num=96)
    initial = np.asarray(sample.ball_initial_position, dtype=np.float64)
    velocity = np.asarray(sample.ball_initial_velocity, dtype=np.float64)
    gravity = np.asarray(sample.gravity, dtype=np.float64)
    positions = []
    for t in times:
        dt = max(0.0, float(t) - float(sample.release_time_s))
        positions.append(initial + velocity * dt + 0.5 * gravity * dt * dt)
    positions.append(np.asarray([initial[0], initial[1], sample.catch_center_z], dtype=np.float64))
    positions_array = np.asarray(positions, dtype=np.float64)
    floor_z = (sample.tabletop_height or 0.0) + float(sample.ball_radius)
    positions_array[:, 2] = np.maximum(positions_array[:, 2], floor_z)
    return positions_array.min(axis=0), positions_array.max(axis=0), positions_array


def _trajectory_side_camera_pose(
    target: np.ndarray,
    half_extent: np.ndarray,
    *,
    azimuth_degrees: float,
    fovy: float,
    distance_scale: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    vertical_span = max(0.35, 2.0 * float(half_extent[2]))
    horizontal_span = max(0.55, 2.0 * float(np.linalg.norm(half_extent[:2])))
    vfov = np.radians(float(fovy))
    aspect = 16.0 / 9.0
    hfov = 2.0 * np.arctan(np.tan(0.5 * vfov) * aspect)
    distance = max(
        1.15,
        0.58 * vertical_span / np.tan(0.5 * vfov),
        0.58 * horizontal_span / np.tan(0.5 * hfov),
    ) * float(distance_scale)
    azimuth = np.radians(float(azimuth_degrees))
    xy = np.array([np.cos(azimuth), np.sin(azimuth)], dtype=np.float64) * distance
    pos = np.array(
        [
            target[0] + xy[0],
            target[1] + xy[1],
            target[2] + max(0.28, 0.32 * distance),
        ],
        dtype=np.float64,
    )
    pos[0] = float(np.clip(pos[0], -0.86, 1.60))
    pos[1] = float(np.clip(pos[1], -1.55, 1.10))
    pos[2] = float(np.clip(pos[2], 1.02, 2.08))
    lookat = target.copy()
    lookat[2] += 0.05
    return tuple(float(v) for v in pos), tuple(float(v) for v in lookat)


def _configure_robotiq_visibility_cameras(
    root: ET.Element,
    sample: EpisodeSample,
    camera_ball_positions: np.ndarray | None = None,
) -> dict:
    minimum, maximum, positions = _predicted_ballistic_envelope(sample, camera_ball_positions)
    center = 0.5 * (minimum + maximum)
    half_extent = np.maximum(0.5 * (maximum - minimum), np.array([0.18, 0.18, 0.32], dtype=np.float64))
    center[2] = float(np.clip(center[2], sample.catch_center_z + 0.08, sample.catch_center_z + 0.62))
    # Keep the robot and interception zone in frame even when the predicted
    # falling path has a taller vertical envelope than the actual catch.
    target = 0.74 * center + 0.26 * np.asarray([0.47, 0.0, sample.catch_center_z + 0.12], dtype=np.float64)
    rich_demo = sample.scene_randomization_level == "demo_rich"
    main_fovy = 58.0 if rich_demo else 52.0
    side_fovy = 64.0 if rich_demo else 60.0
    main_pos, main_lookat = _trajectory_side_camera_pose(
        target,
        half_extent,
        azimuth_degrees=-46.0,
        fovy=side_fovy,
        distance_scale=1.20 if rich_demo else 1.08,
    )
    # The RoboCasa kitchen/workbench variants place cabinetry along the rear
    # wall. A rear-side camera sees pretty rooms but often loses the arm behind
    # cupboards, so the dataset side view is a front-left task view.
    side_pos, side_lookat = _trajectory_side_camera_pose(
        target,
        half_extent,
        azimuth_degrees=-118.0,
        fovy=side_fovy,
        distance_scale=1.40 if rich_demo else 1.34,
    )
    closeup_pos, closeup_lookat = _trajectory_side_camera_pose(
        target,
        half_extent,
        azimuth_degrees=-132.0,
        fovy=side_fovy,
        distance_scale=1.32 if rich_demo else 1.24,
    )
    return {
        "main_camera": _set_world_camera(
            root,
            "main_camera",
            pos=main_pos,
            lookat=main_lookat,
            fovy=main_fovy,
        ),
        "side_camera": _set_world_camera(
            root,
            "side_camera",
            pos=side_pos,
            lookat=side_lookat,
            fovy=side_fovy,
        ),
        "closeup_camera": _set_world_camera(
            root,
            "closeup_camera",
            pos=closeup_pos,
            lookat=closeup_lookat,
            fovy=side_fovy,
        ),
        "framing_source": {
            "strategy": "pre_simulated_ball_envelope" if camera_ball_positions is not None else "predicted_ballistic_envelope",
            "ball_envelope_min": [float(v) for v in minimum],
            "ball_envelope_max": [float(v) for v in maximum],
            "ball_envelope_center": [float(v) for v in center],
            "ball_envelope_half_extent": [float(v) for v in half_extent],
            "sample_count": int(len(positions)),
            "rich_demo_context": bool(rich_demo),
            "view_policy": "main=close_front_right_oblique; side=front_left_oblique_to_avoid_rear_cupboard_occlusion",
        },
    }


def _add_robotiq_sites_and_thick_pads(root: ET.Element) -> None:
    asset = _ensure_child(root, "asset")
    add_material(asset, "robotiq_thick_pad_blue", (0.18, 0.43, 0.74, 1.0), roughness=0.52)

    base_body = _body_by_name(root, "rq_base")
    ET.SubElement(
        base_body,
        "site",
        name=ROBOTIQ_GRASP_SITE,
        pos="0 0 0.123",
        type="sphere",
        size="0.010",
        rgba="0.18 0.43 0.74 0.35",
    )

    for body_name, site_name in (
        ("rq_left_pad", ROBOTIQ_LEFT_PAD_SITE),
        ("rq_right_pad", ROBOTIQ_RIGHT_PAD_SITE),
    ):
        pad = _body_by_name(root, body_name)
        ET.SubElement(
            pad,
            "site",
            name=site_name,
            pos="0 -0.0035 0.0190",
            type="sphere",
            size="0.008",
            rgba="0.18 0.43 0.74 0.30",
        )
        ET.SubElement(
            pad,
            "geom",
            name=f"{body_name}_thick_visual_pad",
            type="box",
            pos="0 -0.0035 0.0190",
            size="0.0220 0.0100 0.0310",
            material="robotiq_thick_pad_blue",
            contype="0",
            conaffinity="0",
            group="2",
        )
        ET.SubElement(
            pad,
            "geom",
            name=f"{body_name}_thick_collision_pad",
            type="box",
            pos="0 -0.0035 0.0190",
            size="0.0220 0.0100 0.0310",
            rgba="0.18 0.43 0.74 0.18",
            mass="0",
            condim="6",
            friction="1.6 0.10 0.10",
            solimp="0.92 0.995 0.001",
            solref="0.012 1.2",
            priority="2",
            group="3",
        )


def _disable_robotiq_non_pad_collisions(root: ET.Element) -> None:
    allowed = {
        "rq_left_pad1",
        "rq_left_pad2",
        "rq_left_pad_thick_collision_pad",
        "rq_right_pad1",
        "rq_right_pad2",
        "rq_right_pad_thick_collision_pad",
    }
    for geom in _ensure_child(root, "worldbody").findall(".//geom"):
        body_name = ""
        parent = geom
        # ElementTree has no parent pointer; use the geom name/body path heuristic
        # below after checking explicit pad geometry names.
        geom_name = geom.get("name", "")
        if geom_name in allowed:
            continue
        if geom_name.startswith("rq_") or geom_name == "":
            # Menagerie collision geoms are often unnamed under rq_* bodies. They
            # can clip the ball before the clasp closes, so only the pad contact
            # geoms remain active for interception.
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def _add_robotiq_2f85(root: ET.Element, *, thick_pads: bool = True) -> None:
    robotiq_root = ET.parse(ROBOTIQ_2F85_XML).getroot()
    mesh_names, material_names = _collect_asset_names(robotiq_root)

    for default in robotiq_root.findall("default"):
        _insert_before_worldbody(
            root,
            _prefix_menagerie_tree(default, ROBOTIQ_PREFIX, mesh_names, material_names),
        )

    asset = _ensure_child(root, "asset")
    source_asset = robotiq_root.find("asset")
    if source_asset is not None:
        for child in source_asset:
            asset.append(
                _prefixed_asset_child(
                    child,
                    ROBOTIQ_2F85_DIR / "assets",
                    ROBOTIQ_PREFIX,
                    mesh_names,
                    material_names,
                )
            )

    attachment = _body_by_name(root, "attachment")
    source_world = robotiq_root.find("worldbody")
    if source_world is None:
        raise ValueError(f"{ROBOTIQ_2F85_XML} has no worldbody.")
    for source_body in source_world.findall("body"):
        body = _prefix_menagerie_tree(source_body, ROBOTIQ_PREFIX, mesh_names, material_names)
        body.set("pos", " ".join(f"{value:g}" for value in ROBOTIQ_MOUNT_POS))
        body.set("quat", " ".join(f"{value:g}" for value in ROBOTIQ_MOUNT_QUAT))
        attachment.append(body)

    for section_name in ("contact", "tendon", "equality", "actuator"):
        source_section = robotiq_root.find(section_name)
        if source_section is None:
            continue
        target_section = _ensure_child(root, section_name)
        for child in source_section:
            target_section.append(
                _prefix_menagerie_tree(child, ROBOTIQ_PREFIX, mesh_names, material_names)
            )

    if thick_pads:
        _add_robotiq_sites_and_thick_pads(root)
    _disable_robotiq_non_pad_collisions(root)


def build_episode(sample: EpisodeSample) -> EpisodeBundle:
    asset_info = locate_franka_asset()
    root = ET.parse(asset_info.robot_xml).getroot()
    _remove_source_keyframes(root)
    _set_robot_compiler_paths(root, asset_info)
    _configure_options(root, sample)
    _configure_robot_base(root, sample)
    camera_pose = add_variant_xml(
        root,
        sample.scene_variant,
        lighting_intensity=sample.lighting_intensity,
        floor_jitter=sample.floor_material_jitter,
        camera_jitter=sample.camera_jitter,
        random_seed=sample.seed,
        scene_randomization_level=sample.scene_randomization_level,
    )
    camera_pose = {"main_camera": camera_pose, WRIST_CAMERA_NAME: _add_wrist_camera(root)}
    _add_interception_surfaces(root, sample)
    _add_ball(root, sample)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return EpisodeBundle(model=model, data=data, xml=xml, asset_info=asset_info, camera_pose=camera_pose)


def build_robotiq_thick_pad_episode(
    sample: EpisodeSample,
    *,
    camera_ball_positions: np.ndarray | None = None,
) -> EpisodeBundle:
    asset_info = locate_franka_nohand_asset()
    root = ET.parse(asset_info.robot_xml).getroot()
    _remove_source_keyframes(root)
    root.set("model", "franka_robotiq_2f85_thick_pad_interception")
    _set_robot_compiler_paths(root, asset_info)
    _configure_options(root, sample)
    _configure_robot_base(root, sample)
    _disable_franka_link_collisions(root)
    camera_pose = add_variant_xml(
        root,
        sample.scene_variant,
        lighting_intensity=sample.lighting_intensity,
        floor_jitter=sample.floor_material_jitter,
        camera_jitter=sample.camera_jitter,
        random_seed=sample.seed,
        scene_randomization_level=sample.scene_randomization_level,
    )
    robocasa_background_assets = camera_pose.get("robocasa_background_assets", [])
    robotwin_background_assets = camera_pose.get("robotwin_background_assets", [])
    robocasa_background_random_seed = camera_pose.get("robocasa_background_random_seed")
    scene_randomization = camera_pose.get("scene_randomization", {})
    _add_robotiq_2f85(root, thick_pads=True)
    visible_camera_pose = _configure_robotiq_visibility_cameras(root, sample, camera_ball_positions)
    visible_camera_pose["main_camera"]["robocasa_background_random_seed"] = robocasa_background_random_seed
    visible_camera_pose["main_camera"]["robocasa_background_assets"] = robocasa_background_assets
    visible_camera_pose["main_camera"]["robotwin_background_assets"] = robotwin_background_assets
    visible_camera_pose["main_camera"]["scene_randomization"] = scene_randomization
    camera_pose = {
        **visible_camera_pose,
        WRIST_CAMERA_NAME: _add_wrist_camera(
            root,
            parent_body_name="rq_base",
            local_pos=(-0.072, 0.0, 0.038),
        ),
    }
    _add_interception_surfaces(root, sample)
    _add_ball(root, sample)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mounted_asset_info = AssetInfo(
        robot_xml=asset_info.robot_xml,
        source=f"{asset_info.source}+robotiq_2f85_thick_pad",
        asset_dir=asset_info.asset_dir,
        note=f"{asset_info.note}; Robotiq 2F-85 from {ROBOTIQ_2F85_XML}",
    )
    return EpisodeBundle(model=model, data=data, xml=xml, asset_info=mounted_asset_info, camera_pose=camera_pose)
