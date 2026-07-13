"""Procedural task geometry around the official MuJoCo Menagerie Panda.

Only robot joint controls and MuJoCo contact dynamics advance the rollout.  The
builder removes the Panda finger-coupling equality because the mounted task
tool, rather than an assisted gripper latch, performs object interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Iterable, Sequence

from ..base import (
    CameraSpec,
    NativeBackendError,
    NativeFamily,
    RigidScenario,
    RigidShape,
    ScenarioSpec,
    ToolKind,
)


OBJECT_BODY = "task_object"
OBJECT_GEOM = "task_object_geom"
OBJECT_JOINT = "task_object_freejoint"
TOOL_BODY = "native_task_tool"
TOOL_SITE = "native_task_tool_site"
TOOL_GEOM_PREFIX = "native_tool_"
OBJECT_CONTACT_PRIORITY = 1
RESTITUTION_SOLVER_PROFILE_VERSION = "native-restitution-solver-map/v1"
_RESTITUTION_DAMPING_KNOTS = (
    (0.05, 1.000),
    (0.25, 0.740),
    (0.50, 0.545),
    (0.75, 0.450),
    (0.95, 0.400),
)


@dataclass(frozen=True)
class FrankaAsset:
    xml_path: Path
    root: Path
    source: str
    xml_sha256: str
    license_sha256: str | None
    verified_menagerie_layout: bool


@dataclass(frozen=True)
class CompiledModelDescription:
    xml: str
    asset: FrankaAsset
    surface_geoms: tuple[str, ...]
    task_tool_geoms: tuple[str, ...]
    expected_camera_names: tuple[str, str]
    model_hash: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_franka_asset(explicit_path: str | Path | None = None) -> FrankaAsset:
    """Resolve a Panda MJCF without importing or copying external assets."""

    candidates: list[tuple[str, Path]] = []
    if explicit_path is not None:
        candidates.append(("explicit", Path(explicit_path).expanduser()))
    if os.environ.get("FRANKA_MJCF_PATH"):
        candidates.append(("FRANKA_MJCF_PATH", Path(os.environ["FRANKA_MJCF_PATH"])))
    if os.environ.get("MUJOCO_MENAGERIE_ROOT"):
        candidates.append(
            (
                "MUJOCO_MENAGERIE_ROOT",
                Path(os.environ["MUJOCO_MENAGERIE_ROOT"])
                / "franka_emika_panda"
                / "panda.xml",
            )
        )
    # Repository sibling used by the migration sources.  This is a read-only
    # discovery fallback; provenance records the resolved absolute path/hash.
    project_owner_root = Path(__file__).resolve().parents[5]
    candidates.append(
        (
            "project_mujoco_menagerie",
            project_owner_root
            / "demo_mujoco_arm_gripper"
            / "third_party"
            / "mujoco_menagerie"
            / "franka_emika_panda"
            / "panda.xml",
        )
    )
    checked: list[str] = []
    for source, candidate in candidates:
        path = candidate.expanduser().resolve()
        checked.append(str(path))
        if not path.is_file():
            continue
        root = path.parent
        license_path = root / "LICENSE"
        assets = root / "assets"
        verified = assets.is_dir() and license_path.is_file() and root.name == "franka_emika_panda"
        return FrankaAsset(
            xml_path=path,
            root=root,
            source=source,
            xml_sha256=_sha256(path),
            license_sha256=_sha256(license_path) if license_path.is_file() else None,
            verified_menagerie_layout=verified,
        )
    raise NativeBackendError(
        "Could not resolve the Franka Panda MJCF. Set FRANKA_MJCF_PATH or "
        "MUJOCO_MENAGERIE_ROOT. Checked: " + ", ".join(checked)
    )


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _ensure(root: ET.Element, tag: str) -> ET.Element:
    value = root.find(tag)
    return value if value is not None else ET.SubElement(root, tag)


def _normalize(values: Sequence[float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return tuple(float(value) / norm for value in values)  # type: ignore[return-value]


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def camera_axes(camera: CameraSpec) -> tuple[tuple[float, float, float], ...]:
    """Return right/up/forward world axes for a MuJoCo camera."""

    forward = _normalize(
        tuple(target - position for target, position in zip(camera.look_at_m, camera.position_m))
    )
    right = _normalize(_cross(forward, camera.up_world))
    up = _normalize(_cross(right, forward))
    return right, up, forward


def _configure_root(root: ET.Element, spec: ScenarioSpec, asset: FrankaAsset) -> None:
    compiler = _ensure(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", str(asset.root / "assets"))
    compiler.set("autolimits", "true")
    option = _ensure(root, "option")
    option.set("timestep", f"{1.0 / spec.sim_hz:.12g}")
    option.set("gravity", _fmt(spec.gravity_m_s2))
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("impratio", "3")
    option.set("solver", "Newton")
    size = _ensure(root, "size")
    size.set("nconmax", "4096")
    size.set("njmax", "8192")
    visual = _ensure(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "832")
    global_visual.set("offheight", "480")
    map_element = visual.find("map")
    if map_element is None:
        map_element = ET.SubElement(visual, "map")
    map_element.set("znear", "0.02")
    map_element.set("zfar", "12")
    # The Panda equality only couples the two fingers.  The native task tool is
    # mounted to the hand and does not need that equality; removing it makes the
    # absence of assistance mechanisms mechanically auditable.
    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)
    world = _ensure(root, "worldbody")
    link0 = world.find("./body[@name='link0']")
    if link0 is None:
        raise NativeBackendError("Resolved Franka MJCF does not contain body link0")
    base = spec.extras.get("robot_base_position_m", (-0.35, 0.0, -0.10))
    link0.set("pos", _fmt(base))


def _add_materials(root: ET.Element, spec: ScenarioSpec) -> None:
    asset = _ensure(root, "asset")
    style = {
        "clean_franka_lab": ((0.78, 0.80, 0.82, 1.0), (0.92, 0.94, 0.96, 1.0)),
        "kitchen_tabletop": ((0.42, 0.27, 0.15, 1.0), (0.83, 0.78, 0.68, 1.0)),
        "cluttered_tabletop": ((0.34, 0.38, 0.42, 1.0), (0.78, 0.80, 0.82, 1.0)),
    }
    style_aliases = {
        "robocasa_kitchen_tabletop": "kitchen_tabletop",
        "robotwin_cluttered_tabletop": "cluttered_tabletop",
    }
    resolved_style = style_aliases.get(spec.scene_style, spec.scene_style)
    table_rgba, background_rgba = style.get(resolved_style, style["clean_franka_lab"])
    sky = {
        "clean_franka_lab": ((0.30, 0.36, 0.44), (0.58, 0.64, 0.72)),
        "kitchen_tabletop": ((0.27, 0.21, 0.16), (0.58, 0.49, 0.39)),
        "cluttered_tabletop": ((0.20, 0.24, 0.30), (0.46, 0.52, 0.59)),
    }[resolved_style if resolved_style in style else "clean_franka_lab"]
    ET.SubElement(
        asset,
        "texture",
        name="native_skybox",
        type="skybox",
        builtin="gradient",
        rgb1=_fmt(sky[0]),
        rgb2=_fmt(sky[1]),
        width="512",
        height="3072",
    )
    ET.SubElement(asset, "material", name="native_table_mat", rgba=_fmt(table_rgba), roughness="0.7")
    ET.SubElement(asset, "material", name="native_background_mat", rgba=_fmt(background_rgba), roughness="0.9")
    ET.SubElement(asset, "material", name="native_tool_mat", rgba=_fmt(spec.tool.rgba), roughness="0.42")
    ET.SubElement(asset, "material", name="native_object_mat", rgba=_fmt(spec.object.rgba), roughness="0.38")


def _surface_geom(
    world: ET.Element,
    *,
    name: str,
    position: Sequence[float],
    size: Sequence[float],
    euler: Sequence[float] = (0.0, 0.0, 0.0),
    collision: bool = True,
    rgba: Sequence[float] | None = None,
    friction: Sequence[float] = (0.55, 0.01, 0.001),
) -> str:
    attributes = {
        "name": name,
        "type": "box",
        "pos": _fmt(position),
        "size": _fmt(size),
        "euler": _fmt(euler),
        "material": "native_table_mat",
        "contype": "8" if collision else "0",
        "conaffinity": "4" if collision else "0",
        "condim": "4",
        "friction": _fmt(friction),
        "solref": "0.006 0.85",
        "solimp": "0.92 0.99 0.001",
    }
    if rgba is not None:
        attributes.pop("material")
        attributes["rgba"] = _fmt(rgba)
    ET.SubElement(world, "geom", **attributes)
    return name


def _add_world_geometry(root: ET.Element, spec: ScenarioSpec) -> tuple[str, ...]:
    world = _ensure(root, "worldbody")
    surfaces: list[str] = []
    if spec.family == NativeFamily.ROLLING_INTERCEPTION:
        half_x = 0.55 if spec.scenario == RigidScenario.ROLL_OFF_EDGE else 0.90
        table_pitch = 0.035 if spec.scenario == RigidScenario.SMALL_SLOPE else 0.0
        surfaces.append(
            _surface_geom(
                world,
                name="table_surface",
                position=(0.0, 0.0, -0.025),
                size=(half_x, 0.55, 0.025),
                euler=(0.0, table_pitch, 0.0),
                friction=spec.surface_friction,
            )
        )
        if spec.scenario == RigidScenario.RAMP_TO_TABLE:
            surfaces.append(
                _surface_geom(
                    world,
                    name="ramp_surface",
                    position=(-0.62, 0.0, 0.075),
                    size=(0.42, 0.34, 0.025),
                    euler=(0.0, 0.18, 0.0),
                    friction=spec.surface_friction,
                )
            )
    else:
        projectile_half_x = (
            0.55 if spec.scenario == RigidScenario.PROJECTILE_ROLL_OFF_EDGE else 1.45
        )
        surfaces.append(
            _surface_geom(
                world,
                name="table_surface",
                position=(0.35, 0.0, -0.025),
                size=(projectile_half_x, 0.80, 0.025),
                friction=spec.surface_friction,
            )
        )
        if spec.scenario == RigidScenario.RAMP_LAUNCH:
            surfaces.append(
                _surface_geom(
                    world,
                    name="ramp_surface",
                    position=(-0.62, 0.0, 0.075),
                    size=(0.42, 0.34, 0.025),
                    euler=(0.0, 0.18, 0.0),
                    friction=spec.surface_friction,
                )
            )
    if spec.scenario in {
        RigidScenario.WALL_REBOUND,
        RigidScenario.FLOOR_TO_WALL,
    }:
        wall_x = 0.15 if spec.scenario == RigidScenario.WALL_REBOUND else 0.92
        surfaces.append(
            _surface_geom(
                world,
                name="wall_surface",
                position=(wall_x, 0.0, 0.75),
                size=(0.025, 0.60, 0.75),
                friction=spec.surface_friction,
            )
        )
    if spec.scenario == RigidScenario.ANGLED_BARRIER_REBOUND:
        surfaces.append(
            _surface_geom(
                world,
                name="angled_barrier_surface",
                position=(0.15, 0.0, 0.75),
                size=(0.025, 0.48, 0.75),
                euler=(0.0, 0.0, 0.38),
                friction=spec.surface_friction,
            )
        )
    # CONTAINER_RECEIVE uses the bin mounted on the Franka hand.  A previous
    # version also created a second static container at the goal, making it
    # impossible to tell whether the robot or an uncommanded fixture received
    # the object.  The moving native tool is now the sole container geometry.
    if spec.scenario == RigidScenario.OCCLUDED_INTERCEPTION:
        _surface_geom(
            world,
            name="visual_occluder",
            position=(-0.08, -0.12, 0.20),
            size=(0.12, 0.035, 0.20),
            collision=False,
            rgba=(0.22, 0.25, 0.29, 1.0),
        )
    # Background clutter is collision-free and depends only on the scene seed,
    # never on intended or measured outcome.
    if spec.scene_style in {"cluttered_tabletop", "robotwin_cluttered_tabletop"}:
        for index, (x, y, scale, color) in enumerate(
            (
                (-0.28, 0.38, 0.06, (0.70, 0.28, 0.20, 1.0)),
                (0.52, -0.40, 0.075, (0.18, 0.48, 0.70, 1.0)),
                (0.80, 0.34, 0.05, (0.62, 0.58, 0.20, 1.0)),
            )
        ):
            _surface_geom(
                world,
                name=f"visual_clutter_{index}",
                position=(x, y, scale),
                size=(scale, scale, scale),
                collision=False,
                rgba=color,
            )
    return tuple(surfaces)


def contact_damping_for_target(effective_restitution_target: float) -> float:
    # MuJoCo soft contacts do not expose a literal coefficient of restitution.
    # This versioned, monotone interpolation is only a solver setting; the
    # rollout still measures the effective response from separated samples.
    target = float(effective_restitution_target)
    if target <= _RESTITUTION_DAMPING_KNOTS[0][0]:
        return _RESTITUTION_DAMPING_KNOTS[0][1]
    if target >= _RESTITUTION_DAMPING_KNOTS[-1][0]:
        return _RESTITUTION_DAMPING_KNOTS[-1][1]
    for (left_target, left_damping), (right_target, right_damping) in zip(
        _RESTITUTION_DAMPING_KNOTS,
        _RESTITUTION_DAMPING_KNOTS[1:],
    ):
        if left_target <= target <= right_target:
            fraction = (target - left_target) / (right_target - left_target)
            return left_damping + fraction * (right_damping - left_damping)
    raise AssertionError("restitution damping interpolation did not bracket target")


def _add_object(root: ET.Element, spec: ScenarioSpec) -> None:
    world = _ensure(root, "worldbody")
    body = ET.SubElement(world, "body", name=OBJECT_BODY)
    ET.SubElement(body, "freejoint", name=OBJECT_JOINT)
    common = {
        "name": OBJECT_GEOM,
        "mass": f"{spec.object.mass_kg:.10g}",
        "material": "native_object_mat",
        "contype": "4",
        "conaffinity": "10",
        "condim": "6",
        # Give the object's explicitly versioned contact profile precedence
        # over fixture defaults.  Without this, MuJoCo averages the object and
        # surface solref values and a requested restitution sweep collapses to
        # a narrow, non-calibratable response range.
        "priority": str(OBJECT_CONTACT_PRIORITY),
        "friction": _fmt(spec.object.friction),
        "solref": f"0.006 {contact_damping_for_target(spec.object.effective_restitution_target):.6g}",
        "solimp": "0.92 0.99 0.001",
    }
    if spec.object.shape == RigidShape.SPHERE:
        ET.SubElement(body, "geom", type="sphere", size=f"{spec.object.radius_m:.10g}", **common)
    elif spec.object.shape == RigidShape.PUCK:
        radius = spec.object.half_extents_m[0]
        half_height = spec.object.half_extents_m[2]
        ET.SubElement(body, "geom", type="cylinder", size=_fmt((radius, half_height)), **common)
    else:
        ET.SubElement(body, "geom", type="box", size=_fmt(spec.object.half_extents_m), **common)


def _add_tool(root: ET.Element, spec: ScenarioSpec) -> tuple[str, ...]:
    hand = _ensure(root, "worldbody").find(".//body[@name='hand']")
    if hand is None:
        raise NativeBackendError("Resolved Franka MJCF does not contain body hand")
    # Panda home pose points local +Z downward.  A +0.10 m local mount therefore
    # places the tool beneath the wrist, matching an ordinary flange attachment.
    tool = ET.SubElement(hand, "body", name=TOOL_BODY, pos="0 0 0.10")
    ET.SubElement(tool, "site", name=TOOL_SITE, pos="0 0 0", size="0.006", rgba="0 1 0 0")
    common = {
        "material": "native_tool_mat",
        "contype": "2",
        "conaffinity": "4",
        "condim": "6",
        "friction": "0.65 0.01 0.001",
        # A slightly compliant, still-fast contact avoids unrealistically
        # impulsive joint acceleration when a moving rigid object strikes the
        # Franka-mounted tool.  This remains native contact; no pose/velocity
        # rewrite or retention constraint is introduced.
        "solref": "0.006 0.9",
        "solimp": "0.95 0.995 0.0005",
    }
    names: list[str] = []

    def add(name: str, *, size: Sequence[float], pos: Sequence[float] = (0, 0, 0)) -> None:
        full_name = f"{TOOL_GEOM_PREFIX}{name}"
        ET.SubElement(tool, "geom", name=full_name, type="box", size=_fmt(size), pos=_fmt(pos), **common)
        names.append(full_name)

    if spec.tool.kind in {ToolKind.FLAT_PADDLE, ToolKind.ANGLED_PADDLE}:
        # Thin local Y axis is approximately world X in the Panda home pose.
        # Local Z remains approximately world vertical, so it must carry the
        # requested paddle height.  The previous width/height/thickness order
        # put the thin dimension on local Z and silently built a horizontal
        # plate that a rolling object passed underneath.
        add(
            "paddle",
            size=(
                spec.tool.half_extents_m[1],
                spec.tool.half_extents_m[0],
                spec.tool.half_extents_m[2],
            ),
        )
    else:
        hx, hy, hz = spec.tool.half_extents_m
        add("tray_base", size=(hy, hx, hz))
        wall = 0.012
        height = spec.tool.wall_height_m
        add("tray_left", size=(wall, hx, height / 2), pos=(hy - wall, 0, -height / 2))
        add("tray_right", size=(wall, hx, height / 2), pos=(-hy + wall, 0, -height / 2))
        add("tray_front", size=(hy, wall, height / 2), pos=(0, hx - wall, -height / 2))
        add("tray_back", size=(hy, wall, height / 2), pos=(0, -hx + wall, -height / 2))
    return tuple(names)


def _add_cameras_lights(root: ET.Element, spec: ScenarioSpec) -> None:
    world = _ensure(root, "worldbody")
    ET.SubElement(
        world,
        "light",
        name="native_key_light",
        pos="0.4 -0.8 2.4",
        dir="0 0 -1",
        directional="true",
        diffuse="0.9 0.9 0.9",
    )
    ET.SubElement(
        world,
        "light",
        name="native_fill_light",
        pos="-1.0 1.0 1.4",
        diffuse="0.45 0.45 0.48",
    )
    for camera in spec.cameras:
        right, up, _forward = camera_axes(camera)
        ET.SubElement(
            world,
            "camera",
            name=camera.name,
            pos=_fmt(camera.position_m),
            xyaxes=_fmt((*right, *up)),
            fovy=f"{camera.fovy_deg:.10g}",
        )


def build_model_description(spec: ScenarioSpec, asset: FrankaAsset) -> CompiledModelDescription:
    spec.validate()
    root = ET.parse(asset.xml_path).getroot()
    _configure_root(root, spec, asset)
    _add_materials(root, spec)
    surfaces = _add_world_geometry(root, spec)
    _add_object(root, spec)
    tools = _add_tool(root, spec)
    _add_cameras_lights(root, spec)
    xml = ET.tostring(root, encoding="unicode")
    model_hash = hashlib.sha256(xml.encode("utf-8")).hexdigest()
    return CompiledModelDescription(
        xml=xml,
        asset=asset,
        surface_geoms=surfaces,
        task_tool_geoms=tools,
        expected_camera_names=("main", "secondary"),
        model_hash=model_hash,
    )
