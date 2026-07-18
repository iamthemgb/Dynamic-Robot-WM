"""Audited RoboCasa island scenes for F3b without importing assisted control.

The collaborator snapshot is used only to resolve RoboCasa YAML fixtures and
to construct appearance geometry.  Robot and object state are never exposed to
this adapter; rollout control remains owned by :mod:`controller` and passes
through the actuator-only mutation wall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import importlib
import math
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from ...common.hashing import sha256_file
from .provenance import (
    PINNED_ROLLING_ISLAND_MANIFEST_SHA256,
    RoboCasaDependency,
    RollingIslandDependencyManifest,
)


ROLLING_ISLAND_SCENE_SCHEMA = "source-mujoco-rolling-island-scene/v1"
ROLLING_ISLAND_SURFACE_NAME = "supported_rolling_pickup_island_top"
ROLLING_ISLAND_WRAPPER_NAME = "rolling_island_scene"

# The three layouts are the exact families demonstrated in Michael's reference
# rollouts.  Reusing their geometry with different audited RoboCasa styles gives
# all five fixed R1 review appearances a genuine full-sized island surface.
_PROFILE_SCENES: Mapping[str, tuple[int, int]] = {
    "robocasa_lab": (38, 42),
    "robocasa_kitchen": (48, 41),
    "robocasa_workbench": (51, 34),
    "robocasa_storage": (38, 41),
    "robocasa_tabletop": (48, 34),
}


@dataclass(frozen=True, slots=True)
class RollingIslandScenePlan:
    scene_profile: str
    layout_id: int
    style_id: int
    counter_name: str
    source_counter_top_geom_name: str
    task_frame_origin_world_xy_m: tuple[float, float]
    task_frame_yaw_world_rad: float
    counter_position_task_m: tuple[float, float, float]
    counter_yaw_task_rad: float
    counter_half_size_m: tuple[float, float, float]
    surface_position_task_m: tuple[float, float, float]
    surface_half_size_m: tuple[float, float, float]
    table_top_z_m: float
    layout_yaml_sha256: str
    style_yaml_sha256: str
    collaborator_manifest_sha256: str
    schema_version: str = ROLLING_ISLAND_SCENE_SCHEMA

    def validate(self) -> None:
        if self.schema_version != ROLLING_ISLAND_SCENE_SCHEMA:
            raise ValueError("unsupported rolling-island scene schema")
        expected = _PROFILE_SCENES.get(self.scene_profile)
        if expected != (self.layout_id, self.style_id):
            raise ValueError("rolling-island profile does not match its fixed layout/style")
        if not self.counter_name or not self.source_counter_top_geom_name:
            raise ValueError("rolling-island scene lacks a selected counter")
        if self.collaborator_manifest_sha256 != PINNED_ROLLING_ISLAND_MANIFEST_SHA256:
            raise ValueError("rolling-island plan differs from the pinned collaborator source")
        values = (
            *self.task_frame_origin_world_xy_m,
            self.task_frame_yaw_world_rad,
            *self.counter_position_task_m,
            self.counter_yaw_task_rad,
            *self.counter_half_size_m,
            *self.surface_position_task_m,
            *self.surface_half_size_m,
            self.table_top_z_m,
        )
        if not np.isfinite(values).all():
            raise ValueError("rolling-island plan contains non-finite geometry")
        if any(value <= 0.0 for value in self.counter_half_size_m):
            raise ValueError("rolling-island counter size must be positive")
        if any(value <= 0.0 for value in self.surface_half_size_m):
            raise ValueError("rolling-island surface size must be positive")
        if abs(
            self.surface_position_task_m[2]
            + self.surface_half_size_m[2]
            - self.table_top_z_m
        ) > 1e-12:
            raise ValueError("rolling-island task surface does not end at table-top height")
        if len(self.layout_yaml_sha256) != 64 or len(self.style_yaml_sha256) != 64:
            raise ValueError("rolling-island YAML provenance is incomplete")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _package_modules(
    dependency: RollingIslandDependencyManifest,
    robocasa: RoboCasaDependency,
) -> tuple[Any, Any, Any]:
    """Load the pinned files under a private namespace and patch only asset lookup."""

    root = Path(dependency.source_root).resolve(strict=True)
    package_name = "_dynamic_robot_mzl7_rolling_island"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package
    elif tuple(getattr(package, "__path__", ())) != (str(root),):
        raise RuntimeError("rolling-island module cache points at another source root")

    yaml_scene = importlib.import_module(f"{package_name}.yaml_scene")
    asset_root = Path(robocasa.asset_root).resolve(strict=True)

    def audited_asset_root() -> Path:
        return asset_root

    # The collaborator checkout moved from its original ``robocasa_full``
    # sibling to the canonical shared RoboCasa tree.  The bytes are pinned by
    # the RoboCasa license/catalog contract; only the root resolver is rebound.
    yaml_scene.robocasa_assets_root = audited_asset_root
    robocasa_assets = importlib.import_module(f"{package_name}.robocasa_assets")
    robocasa_assets.robocasa_assets_root = audited_asset_root
    variants = importlib.import_module(f"{package_name}.variants")
    variants.robocasa_assets_root = audited_asset_root
    scene_builder = importlib.import_module(f"{package_name}.scene_builder")
    return yaml_scene, variants, scene_builder


def _resolved_scene(
    dependency: RollingIslandDependencyManifest,
    robocasa: RoboCasaDependency,
    *,
    layout_id: int,
    style_id: int,
) -> tuple[Any, Any, Any, Any]:
    yaml_scene, variants, scene_builder = _package_modules(dependency, robocasa)
    # Force the deterministic YAML path.  It is the portion of Michael's scene
    # stack audited here and avoids importing a second unpinned robosuite tree.
    fixtures, style_config = yaml_scene._resolve_scene_fixtures(layout_id, style_id)
    fixtures = yaml_scene._apply_layout_fixture_overrides(layout_id, fixtures)
    placement = (
        yaml_scene._override_layout_placement(layout_id, fixtures)
        or yaml_scene._placement_for_scene(fixtures)
    )
    scene = yaml_scene.ResolvedScene(
        layout_id=layout_id,
        style_id=style_id,
        fixtures=tuple(fixtures),
        style_config=style_config,
        room_bounds=yaml_scene._scene_bounds(fixtures),
        placement=placement,
        backend="audited_yaml_fallback",
        native_error=None,
    )
    return scene, yaml_scene, variants, scene_builder


def _task_xy(
    world_xy: Sequence[float],
    *,
    origin_xy: Sequence[float],
    yaw_world_rad: float,
) -> np.ndarray:
    delta = np.asarray(world_xy, dtype=np.float64) - np.asarray(
        origin_xy, dtype=np.float64
    )
    cosine = math.cos(-float(yaw_world_rad))
    sine = math.sin(-float(yaw_world_rad))
    return np.array(
        (
            cosine * delta[0] - sine * delta[1],
            sine * delta[0] + cosine * delta[1],
        ),
        dtype=np.float64,
    )


def _normalized_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _yaml_path(asset_root: Path, *, kind: str, identifier: int) -> Path:
    stem = "layout" if kind == "layouts" else "style"
    return (
        asset_root
        / "scenes"
        / f"kitchen_{kind}"
        / "train"
        / f"{stem}{identifier:03d}.yaml"
    )


def resolve_rolling_island_plan(
    scene_profile: str,
    *,
    dependency: RollingIslandDependencyManifest,
    robocasa: RoboCasaDependency,
    seed: int,
) -> RollingIslandScenePlan:
    """Resolve Michael's selected island into a robot-local task frame."""

    try:
        layout_id, style_id = _PROFILE_SCENES[scene_profile]
    except KeyError as error:
        raise ValueError(f"no rolling-island scene for profile {scene_profile!r}") from error
    scene, _, _, scene_builder = _resolved_scene(
        dependency,
        robocasa,
        layout_id=layout_id,
        style_id=style_id,
    )
    visual_settings = scene_builder._seat_row_demo_visual_settings(
        scene, int(seed), "arm_relative_island_demo"
    )
    counter_name = str(visual_settings["robot_counter_name"])
    counter = next(
        fixture for fixture in scene.fixtures if fixture.name == counter_name
    )
    origin_xy = tuple(float(value) for value in visual_settings["robot_base_position"][:2])
    task_yaw = float(visual_settings["robot_base_yaw"])
    counter_xy = _task_xy(
        counter.pos[:2], origin_xy=origin_xy, yaw_world_rad=task_yaw
    )
    counter_position = (
        float(counter_xy[0]),
        float(counter_xy[1]),
        float(counter.pos[2]),
    )
    counter_half_size = tuple(float(value) for value in counter.half_size)
    surface_half_z = 0.04
    surface_position = (
        counter_position[0],
        counter_position[1],
        counter_position[2] + counter_half_size[2] - surface_half_z,
    )
    table_top = surface_position[2] + surface_half_z
    asset_root = Path(robocasa.asset_root).resolve(strict=True)
    layout_yaml = _yaml_path(asset_root, kind="layouts", identifier=layout_id)
    style_yaml = _yaml_path(asset_root, kind="styles", identifier=style_id)
    plan = RollingIslandScenePlan(
        scene_profile=scene_profile,
        layout_id=layout_id,
        style_id=style_id,
        counter_name=counter_name,
        source_counter_top_geom_name=f"{counter_name}_top",
        task_frame_origin_world_xy_m=origin_xy,
        task_frame_yaw_world_rad=task_yaw,
        counter_position_task_m=counter_position,
        counter_yaw_task_rad=_normalized_angle(float(counter.yaw) - task_yaw),
        counter_half_size_m=counter_half_size,
        surface_position_task_m=surface_position,
        surface_half_size_m=(
            counter_half_size[0],
            counter_half_size[1],
            surface_half_z,
        ),
        table_top_z_m=table_top,
        layout_yaml_sha256=sha256_file(layout_yaml),
        style_yaml_sha256=sha256_file(style_yaml),
        collaborator_manifest_sha256=dependency.manifest_sha256,
    )
    plan.validate()
    return plan


def _prefix_assets(temp_root: ET.Element, target_root: ET.Element) -> None:
    source = temp_root.find("asset")
    if source is None:
        return
    target = target_root.find("asset")
    if target is None:
        target = ET.SubElement(target_root, "asset")
    maps: dict[str, dict[str, str]] = {
        "material": {},
        "texture": {},
        "mesh": {},
    }
    for element in source:
        name = str(element.get("name") or "")
        if name and element.tag in maps:
            maps[element.tag][name] = f"rolling_island_{name}"
    for element in source:
        copied = copy.deepcopy(element)
        name = str(copied.get("name") or "")
        if name and element.tag in maps:
            copied.set("name", maps[element.tag][name])
        for attribute, kind in (("material", "material"), ("texture", "texture"), ("mesh", "mesh")):
            value = copied.get(attribute)
            if value in maps[kind]:
                copied.set(attribute, maps[kind][value])
        target.append(copied)

    for geom in temp_root.findall(".//geom"):
        material = geom.get("material")
        mesh = geom.get("mesh")
        if material in maps["material"]:
            geom.set("material", maps["material"][material])
        if mesh in maps["mesh"]:
            geom.set("mesh", maps["mesh"][mesh])


def _prefix_scene_names(element: ET.Element, selected_geom_name: str) -> None:
    for item in element.iter():
        name = str(item.get("name") or "")
        if item.tag == "geom" and name == selected_geom_name:
            item.set("name", ROLLING_ISLAND_SURFACE_NAME)
            continue
        if item.tag == "geom" and name == "floor_1":
            # Replace the old coplanar demo floor with the layout's own room
            # floor.  Keeping both produced severe z-fighting in the rendered
            # kitchen; this one remains the real anchored room support.
            item.set("name", "floor")
            continue
        if name:
            item.set("name", f"rolling_island_{name}")
        if item.tag == "geom":
            item.set("contype", "0")
            item.set("conaffinity", "0")


def _camera_xyaxes(position: Sequence[float], target: Sequence[float]) -> str:
    position_array = np.asarray(position, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    forward = target_array - position_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return " ".join(f"{value:.9g}" for value in (*right, *up))


def _task_xyz(value: Sequence[float], plan: RollingIslandScenePlan) -> tuple[float, float, float]:
    xy = _task_xy(
        value[:2],
        origin_xy=plan.task_frame_origin_world_xy_m,
        yaw_world_rad=plan.task_frame_yaw_world_rad,
    )
    return float(xy[0]), float(xy[1]), float(value[2])


def _world_xyz(value: Sequence[float], plan: RollingIslandScenePlan) -> tuple[float, float, float]:
    cosine = math.cos(plan.task_frame_yaw_world_rad)
    sine = math.sin(plan.task_frame_yaw_world_rad)
    x = cosine * float(value[0]) - sine * float(value[1])
    y = sine * float(value[0]) + cosine * float(value[1])
    return (
        x + plan.task_frame_origin_world_xy_m[0],
        y + plan.task_frame_origin_world_xy_m[1],
        float(value[2]),
    )


def install_rolling_island_scene(
    root: ET.Element,
    plan: RollingIslandScenePlan,
    *,
    dependency: RollingIslandDependencyManifest,
    robocasa: RoboCasaDependency,
    seed: int,
    lighting_intensity: float,
    camera_jitter: Sequence[float],
    object_initial_position_m: Sequence[float],
    physical_target_position_m: Sequence[float],
) -> Mapping[str, Any]:
    """Replace the old miniature room with a normalized full RoboCasa island."""

    plan.validate()
    scene, _, variants, scene_builder = _resolved_scene(
        dependency,
        robocasa,
        layout_id=plan.layout_id,
        style_id=plan.style_id,
    )
    visual_settings = scene_builder._seat_row_demo_visual_settings(
        scene, int(seed), "arm_relative_island_demo"
    )
    if str(visual_settings["robot_counter_name"]) != plan.counter_name:
        raise RuntimeError("rolling-island counter selection changed after planning")
    launch_world = _world_xyz(object_initial_position_m, plan)
    target_world = _world_xyz(physical_target_position_m, plan)
    visual_settings["launch_position_xy"] = [launch_world[0], launch_world[1]]
    visual_settings["catch_position_xyz"] = list(target_world)

    temp_root = ET.Element("mujoco")
    ET.SubElement(temp_root, "asset")
    ET.SubElement(temp_root, "worldbody")
    original_resolver = variants.resolve_scene
    original_style_models = variants._append_style_models
    variants.resolve_scene = lambda layout_id, style_id: scene
    # The audited task geometry comes from RoboCasa YAML.  Imported appliance
    # meshes are not needed for support and would require a second catalog of
    # descendant mesh identities; retain the full layout boxes and materials.
    variants._append_style_models = lambda root, fixtures: None
    try:
        camera_pose = variants.add_variant_xml(
            temp_root,
            "robocasa_kitchen",
            lighting_intensity=float(lighting_intensity),
            floor_jitter=0.0,
            camera_jitter=tuple(float(value) for value in camera_jitter),
            visual_settings=visual_settings,
        )
    finally:
        variants.resolve_scene = original_resolver
        variants._append_style_models = original_style_models

    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("source model lacks worldbody")
    # Keep only the owned robot, task object, and canonical camera.  The
    # layout's own floor replaces the old coplanar demo floor below.
    for child in list(world):
        name = str(child.get("name") or "")
        if child.tag == "body" and name in {"link0", "catch_ball"}:
            continue
        if child.tag == "camera" and name == "main_camera":
            continue
        world.remove(child)

    _prefix_assets(temp_root, root)
    temp_world = temp_root.find("worldbody")
    if temp_world is None:
        raise RuntimeError("rolling-island appearance builder returned no worldbody")
    selected = temp_world.find(
        f".//geom[@name='{plan.source_counter_top_geom_name}']"
    )
    if selected is None:
        raise RuntimeError("rolling-island appearance omitted its selected counter top")

    angle = -plan.task_frame_yaw_world_rad
    cosine = math.cos(angle)
    sine = math.sin(angle)
    origin_x, origin_y = plan.task_frame_origin_world_xy_m
    wrapper_position = (
        -(cosine * origin_x - sine * origin_y),
        -(sine * origin_x + cosine * origin_y),
        0.0,
    )
    wrapper = ET.SubElement(
        world,
        "body",
        name=ROLLING_ISLAND_WRAPPER_NAME,
        pos=" ".join(f"{value:.12g}" for value in wrapper_position),
        euler=f"0 0 {angle:.12g}",
    )
    for child in list(temp_world):
        if child.tag == "camera":
            continue
        _prefix_scene_names(child, plan.source_counter_top_geom_name)
        wrapper.append(child)

    main = world.find("./camera[@name='main_camera']")
    if main is None:
        main = ET.SubElement(world, "camera", name="main_camera")
    main_info = camera_pose["main_camera"]
    main_position = _task_xyz(main_info["pos"], plan)
    main_target = _task_xyz(main_info["lookat"], plan)
    main.set("pos", " ".join(f"{value:.9g}" for value in main_position))
    main.set("xyaxes", _camera_xyaxes(main_position, main_target))
    main.set("fovy", f"{float(main_info['fovy']):.9g}")

    return {
        "main_camera": {
            "pos": list(main_position),
            "lookat": list(main_target),
            "fovy": float(main_info["fovy"]),
            "camera_variant": "arm_relative_island_demo",
            "layout_id": plan.layout_id,
            "style_id": plan.style_id,
            "counter_name": plan.counter_name,
            "scene_resolution_backend": "audited_yaml_fallback",
        }
    }


__all__ = [
    "ROLLING_ISLAND_SCENE_SCHEMA",
    "ROLLING_ISLAND_SURFACE_NAME",
    "ROLLING_ISLAND_WRAPPER_NAME",
    "RollingIslandScenePlan",
    "install_rolling_island_scene",
    "resolve_rolling_island_plan",
]
