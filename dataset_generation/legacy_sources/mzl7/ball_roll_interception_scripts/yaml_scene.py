from __future__ import annotations

import copy
import functools
import hashlib
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .utils import repo_root


TRAIN_LAYOUT_IDS = tuple(range(11, 61))
TRAIN_STYLE_IDS = tuple(range(11, 61))
TRAIN_ENVIRONMENTS = tuple(
    (layout_id, style_id)
    for layout_id in TRAIN_LAYOUT_IDS
    for style_id in TRAIN_STYLE_IDS
)

REF_KEYS = ("align_to", "attach_to", "interior_obj", "stack_on")
SIZE_KEYS = ("size",)
FULL_SIZE_TYPES = {
    "counter",
    "box",
    "stack",
    "stove",
    "stove_wide",
    "stovetop",
    "oven",
    "microwave",
    "dishwasher",
    "sink",
    "fridge_bottom_freezer",
    "fridge_french_door",
    "fridge_side_by_side",
    "hinge_cabinet",
    "single_cabinet",
    "open_cabinet",
    "panel_cabinet",
    "housing_cabinet",
    "drawer",
    "dish_rack",
    "coffee_machine",
    "blender",
    "blender_lid",
    "toaster",
    "toaster_oven",
    "electric_kettle",
    "stand_mixer",
    "window",
    "window_proc",
    "stool",
    "wall_accessory",
    "accessory",
}
ATTACHABLE_TYPES = {
    "wall_accessory",
    "window",
    "window_proc",
    "stool",
    "accessory",
    "paper_towel",
    "candle",
    "turmeric",
    "cinnamon",
    "paprika",
    "flower_vase",
    "utensil_set",
    "plant",
    "knife_block",
    "soap_dispenser",
    "fruit_bowl",
    "oil_bottle",
    "vinegar_bottle",
    "salt_shaker",
    "pepper_shaker",
    "tiered_basket",
    "digital_scale",
    "glass_cup",
    "jar",
    "jar_lid",
    "utensil_holder",
    "dish_rack",
    "coffee_machine",
    "toaster",
    "toaster_oven",
    "blender",
    "blender_lid",
    "stand_mixer",
    "utensil_rack",
    "electric_kettle",
}
ROOM_SURFACE_DEPTH = 0.04
ROOM_SURFACE_HEIGHT = 3.0
COUNTER_THICKNESS = 0.08
COUNTER_BASE_INSET = 0.02
COUNTER_BASE_HEIGHT_MARGIN = 0.06
ROBOT_FORWARD_REACH = 0.42
LAUNCH_EDGE_INSET = 0.10

LAYOUT_POSITION_OFFSETS = {
    # Bring the stove island a bit closer to the sink counter in layout 58.
    58: {
        "group_offsets": {
            "island_group_1": (0.0, 0.8, 0.0),
        },
        "remove_fixtures": {
            "dishwasher_1_main_group_1",
            "stove_island_group_1",
            "stove_hood_island_group_1",
        },
    },
}


@dataclass(frozen=True)
class ResolvedFixture:
    name: str
    fixture_type: str
    pos: tuple[float, float, float]
    half_size: tuple[float, float, float]
    yaw: float
    raw_config: dict
    group_name: str
    style_model: str | None = None
    style_role: str | None = None
    interior_parent: str | None = None


@dataclass(frozen=True)
class PlacementPlan:
    counter_name: str
    edge_name: str
    robot_base_position: tuple[float, float, float]
    robot_base_yaw: float
    tabletop_height: float
    launch_position: tuple[float, float, float]
    catch_position: tuple[float, float, float]


@dataclass(frozen=True)
class ResolvedScene:
    layout_id: int
    style_id: int
    fixtures: tuple[ResolvedFixture, ...]
    style_config: dict
    room_bounds: tuple[float, float, float, float]
    placement: PlacementPlan
    backend: str
    native_error: str | None = None


def robocasa_assets_root() -> Path:
    workspace_root = repo_root() / "robocasa_full" / "robocasa" / "models" / "assets"
    if workspace_root.exists():
        return workspace_root
    home_root = Path("/home/mzl7/scratch/robocasa_full/robocasa/models/assets")
    if home_root.exists():
        return home_root
    raise FileNotFoundError("Could not locate RoboCasa assets root.")


def _yaml_path(kind: str, scene_id: int) -> Path:
    split = "test" if 1 <= int(scene_id) <= 10 else "train"
    return robocasa_assets_root() / "scenes" / f"kitchen_{kind}" / split / f"{kind[:-1]}{int(scene_id):03d}.yaml"


def layout_yaml_path(layout_id: int) -> Path:
    return _yaml_path("layouts", layout_id)


def style_yaml_path(style_id: int) -> Path:
    return _yaml_path("styles", style_id)


def scene_ids_from_seed(seed: int) -> tuple[int, int, int]:
    env_index = int(seed % len(TRAIN_ENVIRONMENTS))
    layout_id, style_id = TRAIN_ENVIRONMENTS[env_index]
    return env_index, layout_id, style_id


def _as_np(values, *, dims: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (dims,):
        raise ValueError(f"Expected shape {(dims,)}, got {arr.shape}")
    return arr


def _deepcopy_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return copy.deepcopy(yaml.safe_load(handle))


def _native_python_roots() -> tuple[Path, ...]:
    root = repo_root()
    return (
        root / "robosuite",
        root / "robocasa_full",
    )


def _prepare_native_import_paths() -> None:
    workspace_root = str(repo_root())
    while workspace_root in sys.path:
        sys.path.remove(workspace_root)
    for path in reversed(_native_python_roots()):
        path_str = str(path)
        if path.exists() and path_str in sys.path:
            sys.path.remove(path_str)
        if path.exists():
            sys.path.insert(0, path_str)
    robosuite_mod = sys.modules.get("robosuite")
    if robosuite_mod is not None and not hasattr(robosuite_mod, "__version__"):
        setattr(robosuite_mod, "__version__", "1.5.2")
    if robosuite_mod is not None and getattr(robosuite_mod, "__file__", None) is None:
        for name in [module_name for module_name in sys.modules if module_name == "robosuite" or module_name.startswith("robosuite.")]:
            del sys.modules[name]
    robocasa_mod = sys.modules.get("robocasa")
    if robocasa_mod is not None and getattr(robocasa_mod, "__file__", None) is None:
        for name in [module_name for module_name in sys.modules if module_name == "robocasa" or module_name.startswith("robocasa.")]:
            del sys.modules[name]


def _with_group_suffix(value, suffix: str):
    if isinstance(value, str):
        return f"{value}_{suffix}"
    if isinstance(value, list):
        return [_with_group_suffix(elem, suffix) if isinstance(elem, str) else elem for elem in value]
    return value


def _normalize_group_fixtures(layout_config: dict) -> list[dict]:
    fixtures: list[dict] = []
    for group_name, group_config in layout_config.items():
        group_origin = group_config.get("group_origin")
        group_pos = group_config.get("group_pos")
        group_z_rot = float(group_config.get("group_z_rot", 0.0))
        for collection_name, fixture_list in group_config.items():
            if collection_name in {"group_origin", "group_pos", "group_z_rot"}:
                continue
            if not isinstance(fixture_list, list):
                continue
            for raw_fixture in fixture_list:
                fixture = copy.deepcopy(raw_fixture)
                fixture["group_name"] = group_name
                if group_name != "room":
                    fixture["name"] = f"{fixture['name']}_{group_name}"
                    for key in REF_KEYS:
                        if key in fixture:
                            fixture[key] = _with_group_suffix(fixture[key], group_name)
                    if "size" in fixture and isinstance(fixture["size"], list):
                        fixture["size"] = [
                            f"{value}_{group_name}" if isinstance(value, str) else value
                            for value in fixture["size"]
                        ]
                fixture["group_origin"] = group_origin
                fixture["group_pos"] = group_pos
                fixture["group_z_rot"] = group_z_rot
                fixtures.append(fixture)
    return fixtures


def _resolve_size(config: dict, resolved: dict[str, ResolvedFixture]) -> np.ndarray:
    raw_size = config.get("size")
    if raw_size is None:
        return np.zeros(3, dtype=np.float64)
    size = []
    for axis, value in enumerate(raw_size):
        if isinstance(value, str):
            size.append(float(resolved[value].half_size[axis] * 2.0))
        elif value is None:
            size.append(0.0)
        else:
            size.append(float(value))
    return np.asarray(size, dtype=np.float64)


def _fixture_half_size(config: dict, full_size: np.ndarray) -> np.ndarray:
    fixture_type = str(config["type"])
    if fixture_type in {"wall", "floor"}:
        return full_size.astype(np.float64)
    return full_size.astype(np.float64) * 0.5


def _relative_pos(config: dict, full_size: np.ndarray, resolved: dict[str, ResolvedFixture]) -> np.ndarray:
    prev = resolved[str(config["align_to"])]
    prev_pos = np.asarray(prev.pos, dtype=np.float64)
    prev_full_size = np.asarray(prev.half_size, dtype=np.float64) * 2.0
    side = str(config["side"]).lower()
    alignment = str(config.get("alignment", "center")).lower()
    pos = prev_pos.copy()
    axis_keywords = {
        0: ("left", "right"),
        1: ("front", "back"),
        2: ("bottom", "top"),
    }
    for axis, keywords in axis_keywords.items():
        if side == keywords[1]:
            pos[axis] = prev_pos[axis] + prev_full_size[axis] * 0.5 + full_size[axis] * 0.5
        elif side == keywords[0]:
            pos[axis] = prev_pos[axis] - prev_full_size[axis] * 0.5 - full_size[axis] * 0.5
    for axis, keywords in axis_keywords.items():
        if keywords[0] in alignment:
            pos[axis] = prev_pos[axis] - prev_full_size[axis] * 0.5 + full_size[axis] * 0.5
        elif keywords[1] in alignment:
            pos[axis] = prev_pos[axis] + prev_full_size[axis] * 0.5 - full_size[axis] * 0.5
    if "offset" in config:
        offset = np.asarray(config["offset"], dtype=np.float64)
        if offset.shape == (2,):
            offset = np.array([offset[0], offset[1], 0.0], dtype=np.float64)
        pos += offset
    return pos


def _apply_group_transform(pos: np.ndarray, yaw: float, config: dict) -> tuple[np.ndarray, float]:
    origin = config.get("group_origin")
    group_pos = config.get("group_pos")
    group_yaw = float(config.get("group_z_rot", 0.0))
    if origin is None or group_pos is None:
        return pos, yaw
    origin_xy = np.asarray(origin[:2], dtype=np.float64)
    group_xy = np.asarray(group_pos[:2], dtype=np.float64)
    delta = pos[:2] - origin_xy
    c = math.cos(group_yaw)
    s = math.sin(group_yaw)
    rot_xy = np.array([delta[0] * c - delta[1] * s, delta[0] * s + delta[1] * c], dtype=np.float64)
    transformed = pos.copy()
    transformed[:2] = origin_xy + rot_xy + (group_xy - origin_xy)
    return transformed, yaw + group_yaw


def _style_token(style_config: dict, path: tuple[str, ...], default: str) -> str:
    current = style_config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if isinstance(current, list):
        return str(current[0]) if current else default
    return str(current)


def _style_color(token: str, *, min_value: float = 0.25, max_value: float = 0.88) -> tuple[float, float, float, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    values = []
    span = max_value - min_value
    for index in range(3):
        values.append(min_value + span * (digest[index] / 255.0))
    return (float(values[0]), float(values[1]), float(values[2]), 1.0)


STYLE_ROLE_KEYS = {
    "sink": ("sink",),
    "dishwasher": ("dishwasher",),
    "microwave": ("microwave",),
    "oven": ("oven",),
    "stove": ("stove",),
    "stove_wide": ("stove_wide",),
    "stovetop": ("stovetop",),
    "fridge_bottom_freezer": ("fridge_bottom_freezer",),
    "fridge_french_door": ("fridge_french_door",),
    "fridge_side_by_side": ("fridge_side_by_side",),
    "dish_rack": ("dish_rack",),
    "coffee_machine": ("coffee_machine",),
    "blender": ("blender",),
    "blender_lid": ("blender_lid",),
    "toaster": ("toaster",),
    "toaster_oven": ("toaster_oven",),
    "electric_kettle": ("electric_kettle",),
    "stand_mixer": ("stand_mixer",),
    "window": ("window",),
    "window_proc": ("window",),
    "wall_accessory": ("wall_accessory",),
}


def _fixture_style_model(fixture_type: str, style_config: dict) -> tuple[str | None, str | None]:
    key_path = STYLE_ROLE_KEYS.get(fixture_type)
    if key_path is None:
        return None, None
    model = _style_token(style_config, key_path, "")
    return (model or None), key_path[0]


def _resolve_interior_position(
    parent: ResolvedFixture,
    child_type: str,
    child_name: str,
    style_config: dict,
    parent_config: dict,
) -> ResolvedFixture:
    counter_half = np.asarray(parent.half_size, dtype=np.float64)
    obj_x_percent = float(parent_config.get("obj_x_percent", 0.5))
    obj_y_percent = float(parent_config.get("obj_y_percent", 0.5))
    local_xy = np.array(
        [
            (obj_x_percent - 0.5) * (counter_half[0] * 1.6),
            (obj_y_percent - 0.5) * (counter_half[1] * 1.6),
        ],
        dtype=np.float64,
    )
    c = math.cos(parent.yaw)
    s = math.sin(parent.yaw)
    world_offset = np.array([local_xy[0] * c - local_xy[1] * s, local_xy[0] * s + local_xy[1] * c, 0.0], dtype=np.float64)
    top_z = parent.pos[2] + parent.half_size[2]
    full_size = np.array([0.50, 0.50, 0.30], dtype=np.float64)
    if child_type == "sink":
        full_size = np.array([0.55, 0.45, 0.24], dtype=np.float64)
    elif child_type == "stovetop":
        full_size = np.array([0.60, 0.45, 0.10], dtype=np.float64)
    pos = np.asarray(parent.pos, dtype=np.float64) + world_offset
    pos[2] = top_z
    style_model, style_role = _fixture_style_model(child_type, style_config)
    return ResolvedFixture(
        name=child_name,
        fixture_type=child_type,
        pos=(float(pos[0]), float(pos[1]), float(pos[2])),
        half_size=(float(full_size[0] * 0.5), float(full_size[1] * 0.5), float(full_size[2] * 0.5)),
        yaw=parent.yaw,
        raw_config=copy.deepcopy(parent_config),
        group_name=parent.group_name,
        style_model=style_model,
        style_role=style_role,
        interior_parent=parent.name,
    )


def _attachable_fixture_position(config: dict, resolved: dict[str, ResolvedFixture]) -> tuple[np.ndarray, float]:
    attach_name = str(config["attach_to"])
    parent = resolved[attach_name]
    raw_pos = config.get("pos", [0.0, 0.0, 0.0])
    local = np.zeros(3, dtype=np.float64)
    for axis, value in enumerate(raw_pos):
        if value is not None:
            local[axis] = float(value)
    if parent.fixture_type == "wall":
        wall_side = str(parent.raw_config.get("wall_side", "back"))
        if wall_side == "left":
            local = np.array([parent.pos[0] + parent.half_size[0], parent.pos[1] + local[1], local[2]], dtype=np.float64)
        elif wall_side == "right":
            local = np.array([parent.pos[0] - parent.half_size[0], parent.pos[1] + local[1], local[2]], dtype=np.float64)
        elif wall_side == "front":
            local = np.array([parent.pos[0] + local[0], parent.pos[1] + parent.half_size[1], local[2]], dtype=np.float64)
        else:
            local = np.array([parent.pos[0] + local[0], parent.pos[1] - parent.half_size[1], local[2]], dtype=np.float64)
        yaw = parent.yaw
    elif parent.fixture_type == "floor":
        local = np.array([local[0], local[1], 0.0], dtype=np.float64)
        yaw = float(config.get("z_rot", 0.0))
    else:
        local = np.asarray(parent.pos, dtype=np.float64) + local
        yaw = parent.yaw + float(config.get("z_rot", 0.0))
    if raw_pos and raw_pos[-1] is None:
        local[2] = max(local[2], 0.0)
    return local, yaw


def _resolve_scene_fixtures(layout_id: int, style_id: int) -> tuple[list[ResolvedFixture], dict]:
    layout_config = _deepcopy_yaml(layout_yaml_path(layout_id))
    style_config = _deepcopy_yaml(style_yaml_path(style_id))
    normalized = _normalize_group_fixtures(layout_config)
    resolved: dict[str, ResolvedFixture] = {}
    deferred_interior: list[tuple[str, str, dict, str]] = []

    for config in normalized:
        fixture_type = str(config["type"])
        if fixture_type in {"room"}:
            continue
        full_size = _resolve_size(config, resolved)
        yaw = float(config.get("z_rot", 0.0))
        if fixture_type in {"sink", "stovetop"} and "pos" not in config and "align_to" not in config and "attach_to" not in config:
            deferred_interior.append((config["name"], fixture_type, config, str(config["group_name"])))
            continue
        if "align_to" in config:
            pos = _relative_pos(config, full_size, resolved)
        elif "attach_to" in config:
            pos, yaw = _attachable_fixture_position(config, resolved)
        elif "pos" in config:
            pos = np.asarray(config["pos"], dtype=np.float64)
        else:
            pos = np.zeros(3, dtype=np.float64)
        pos, yaw = _apply_group_transform(pos, yaw, config)
        style_model, style_role = _fixture_style_model(fixture_type, style_config)
        fixture = ResolvedFixture(
            name=str(config["name"]),
            fixture_type=fixture_type,
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            half_size=tuple(float(v) for v in _fixture_half_size(config, full_size)),
            yaw=float(yaw),
            raw_config=copy.deepcopy(config),
            group_name=str(config["group_name"]),
            style_model=style_model,
            style_role=style_role,
        )
        resolved[fixture.name] = fixture
        if fixture.fixture_type == "counter":
            child_name = fixture.raw_config.get("interior_obj")
            if child_name is not None and child_name not in resolved:
                child_type = "sink" if "sink" in child_name else "stovetop"
                resolved_child = _resolve_interior_position(fixture, child_type, str(child_name), style_config, fixture.raw_config)
                resolved[resolved_child.name] = resolved_child

    for parent in list(resolved.values()):
        if parent.fixture_type != "counter":
            continue
        child_name = parent.raw_config.get("interior_obj")
        if child_name is None or child_name in resolved:
            continue
        child_type = "sink" if "sink" in child_name else "stovetop"
        resolved_child = _resolve_interior_position(parent, child_type, str(child_name), style_config, parent.raw_config)
        resolved[resolved_child.name] = resolved_child

    return list(resolved.values()), style_config


def _resolve_scene_fixtures_native(layout_id: int, style_id: int) -> tuple[list[ResolvedFixture], dict]:
    _prepare_native_import_paths()
    arena, layout_config, style_config = build_native_arena(layout_id, style_id)
    normalized = _normalize_group_fixtures(layout_config)
    config_by_name = {str(config["name"]): copy.deepcopy(config) for config in normalized if "name" in config}
    for config in normalized:
        fixture_name = str(config.get("name", ""))
        if fixture_name and fixture_name not in config_by_name:
            config_by_name[fixture_name] = copy.deepcopy(config)
    resolved: list[ResolvedFixture] = []
    for name, fixture in arena.fixtures.items():
        config = config_by_name.get(name, {"name": name, "type": type(fixture).__name__.lower(), "group_name": "native"})
        fixture_type = str(config.get("type", type(fixture).__name__.lower()))
        pos = np.asarray(getattr(fixture, "pos"), dtype=np.float64)
        size = np.asarray(getattr(fixture, "size", np.zeros(3, dtype=np.float64)), dtype=np.float64)
        if pos.shape != (3,) or size.shape != (3,) or np.any(~np.isfinite(pos)) or np.any(~np.isfinite(size)):
            continue
        yaw = 0.0
        euler = getattr(fixture, "euler", None)
        if euler is not None:
            euler_arr = np.asarray(euler, dtype=np.float64)
            if euler_arr.shape == (3,):
                yaw = float(euler_arr[2])
        style_model, style_role = _fixture_style_model(fixture_type, style_config)
        resolved.append(
            ResolvedFixture(
                name=str(name),
                fixture_type=fixture_type,
                pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                half_size=(float(size[0] * 0.5), float(size[1] * 0.5), float(size[2] * 0.5)),
                yaw=yaw,
                raw_config=config,
                group_name=str(config.get("group_name", "native")),
                style_model=style_model,
                style_role=style_role,
                interior_parent=str(config.get("align_to")) if fixture_type in {"sink", "stovetop"} else None,
            )
        )
    if not resolved:
        raise ValueError(f"Native RoboCasa scene builder returned no fixtures for layout {layout_id}, style {style_id}.")
    return resolved, style_config


def build_native_arena(layout_id: int, style_id: int):
    _prepare_native_import_paths()
    from robocasa.models.scenes.kitchen_arena import KitchenArena

    layout_config = _deepcopy_yaml(layout_yaml_path(layout_id))
    style_config = _deepcopy_yaml(style_yaml_path(style_id))
    arena = KitchenArena(layout_id=copy.deepcopy(layout_config), style_id=copy.deepcopy(style_config), rng=None, clutter_mode=0)
    return arena, layout_config, style_config


def native_fixture_mjcf(layout_id: int, style_id: int) -> tuple[list[ET.Element], list[ET.Element]]:
    arena, _, _ = build_native_arena(layout_id, style_id)
    asset_elems: list[ET.Element] = []
    body_elems: list[ET.Element] = []
    for fixture in arena.fixtures.values():
        for asset in list(fixture.asset):
            asset_elems.append(copy.deepcopy(asset))
        body_elems.append(copy.deepcopy(fixture.get_obj()))
    return asset_elems, body_elems


def _scene_bounds(fixtures: list[ResolvedFixture]) -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for fixture in fixtures:
        if fixture.fixture_type not in {"wall", "floor", "counter", "stack", "box"}:
            continue
        xs.extend([fixture.pos[0] - fixture.half_size[0], fixture.pos[0] + fixture.half_size[0]])
        ys.extend([fixture.pos[1] - fixture.half_size[1], fixture.pos[1] + fixture.half_size[1]])
    if not xs or not ys:
        return (-2.0, 2.0, -2.0, 2.0)
    return (min(xs), max(xs), min(ys), max(ys))


def _fixture_local_overlap(a_center: float, a_half: float, b_center: float, b_half: float) -> float:
    return min(a_center + a_half, b_center + b_half) - max(a_center - a_half, b_center - b_half)


def _transform_into_local(reference: ResolvedFixture, other: ResolvedFixture) -> tuple[float, float]:
    dx = other.pos[0] - reference.pos[0]
    dy = other.pos[1] - reference.pos[1]
    c = math.cos(reference.yaw)
    s = math.sin(reference.yaw)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return float(local_x), float(local_y)


def _edge_exposure(counter: ResolvedFixture, fixtures: list[ResolvedFixture]) -> dict[str, float]:
    counter_hx, counter_hy, _ = counter.half_size
    exposures = {"+x": 2.0, "-x": 2.0, "+y": 2.0, "-y": 2.0}
    neighbor_types = {"counter", "stack", "box", "dishwasher", "microwave", "oven", "stove", "stove_wide", "stovetop", "fridge_bottom_freezer", "fridge_french_door", "fridge_side_by_side"}
    for other in fixtures:
        if other.name == counter.name or other.fixture_type not in neighbor_types:
            continue
        local_x, local_y = _transform_into_local(counter, other)
        other_hx, other_hy, _ = other.half_size
        x_overlap = _fixture_local_overlap(0.0, counter_hx, local_x, other_hx)
        y_overlap = _fixture_local_overlap(0.0, counter_hy, local_y, other_hy)
        if y_overlap > 0.20:
            gap_pos = local_x - other_hx - counter_hx
            gap_neg = -local_x - other_hx - counter_hx
            if gap_pos >= -0.18:
                exposures["+x"] = min(exposures["+x"], max(0.0, gap_pos))
            if gap_neg >= -0.18:
                exposures["-x"] = min(exposures["-x"], max(0.0, gap_neg))
        if x_overlap > 0.20:
            gap_pos = local_y - other_hy - counter_hy
            gap_neg = -local_y - other_hy - counter_hy
            if gap_pos >= -0.18:
                exposures["+y"] = min(exposures["+y"], max(0.0, gap_pos))
            if gap_neg >= -0.18:
                exposures["-y"] = min(exposures["-y"], max(0.0, gap_neg))
    return exposures


def _placement_for_scene(fixtures: list[ResolvedFixture]) -> PlacementPlan:
    counters = [fixture for fixture in fixtures if fixture.fixture_type == "counter" and fixture.half_size[0] >= 0.45 and fixture.half_size[1] >= 0.30]
    if not counters:
        raise ValueError("No suitable counters found for robot placement.")
    counters.sort(key=lambda fixture: (-(fixture.half_size[0] * fixture.half_size[1]), abs(fixture.pos[0]) + abs(fixture.pos[1]), fixture.name))
    chosen_counter = counters[0]
    exposures = _edge_exposure(chosen_counter, fixtures)
    edge_half_extent = {
        "+x": chosen_counter.half_size[0],
        "-x": chosen_counter.half_size[0],
        "+y": chosen_counter.half_size[1],
        "-y": chosen_counter.half_size[1],
    }
    max_extent = max(edge_half_extent.values())
    candidate_edges = [key for key, extent in edge_half_extent.items() if extent >= max_extent - 0.05]
    edge_name = max(candidate_edges, key=lambda key: (exposures[key], edge_half_extent[key], key in {"+x", "-x"}))
    counter_hx, counter_hy, _ = chosen_counter.half_size
    local_robot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    edge_to_yaw = {
        "+x": 0.0,
        "-x": math.pi,
        "+y": math.pi * 0.5,
        "-y": -math.pi * 0.5,
    }
    c = math.cos(chosen_counter.yaw)
    s = math.sin(chosen_counter.yaw)
    robot_xy = np.array(
        [
            chosen_counter.pos[0] + local_robot[0] * c - local_robot[1] * s,
            chosen_counter.pos[1] + local_robot[0] * s + local_robot[1] * c,
        ],
        dtype=np.float64,
    )
    tabletop_height = float(chosen_counter.pos[2] + chosen_counter.half_size[2])
    yaw = chosen_counter.yaw + edge_to_yaw[edge_name]
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    if edge_name == "+x":
        edge_distance = counter_hx - LAUNCH_EDGE_INSET
    elif edge_name == "-x":
        edge_distance = counter_hx - LAUNCH_EDGE_INSET
    elif edge_name == "+y":
        edge_distance = counter_hy - LAUNCH_EDGE_INSET
    else:
        edge_distance = counter_hy - LAUNCH_EDGE_INSET
    launch_distance = max(edge_distance, ROBOT_FORWARD_REACH + 0.05)
    launch_xy = robot_xy + forward[:2] * launch_distance
    catch = np.array([robot_xy[0], robot_xy[1], tabletop_height], dtype=np.float64) + forward * ROBOT_FORWARD_REACH
    catch[2] += 0.31
    return PlacementPlan(
        counter_name=chosen_counter.name,
        edge_name=edge_name,
        robot_base_position=(float(robot_xy[0]), float(robot_xy[1]), tabletop_height),
        robot_base_yaw=float(yaw),
        tabletop_height=tabletop_height,
        launch_position=(float(launch_xy[0]), float(launch_xy[1]), tabletop_height),
        catch_position=(float(catch[0]), float(catch[1]), float(catch[2])),
    )


def _apply_layout_fixture_overrides(layout_id: int, fixtures: list[ResolvedFixture]) -> list[ResolvedFixture]:
    overrides = LAYOUT_POSITION_OFFSETS.get(layout_id)
    if not overrides:
        return fixtures
    group_offsets = overrides.get("group_offsets", {})
    remove_fixtures = overrides.get("remove_fixtures", set())
    if not group_offsets:
        return [fixture for fixture in fixtures if fixture.name not in remove_fixtures]
    adjusted: list[ResolvedFixture] = []
    for fixture in fixtures:
        if fixture.name in remove_fixtures:
            continue
        offset = group_offsets.get(fixture.group_name)
        if offset is None:
            adjusted.append(fixture)
            continue
        adjusted.append(
            ResolvedFixture(
                name=fixture.name,
                fixture_type=fixture.fixture_type,
                pos=(
                    float(fixture.pos[0] + offset[0]),
                    float(fixture.pos[1] + offset[1]),
                    float(fixture.pos[2] + offset[2]),
                ),
                half_size=fixture.half_size,
                yaw=fixture.yaw,
                raw_config=copy.deepcopy(fixture.raw_config),
                group_name=fixture.group_name,
                style_model=fixture.style_model,
                style_role=fixture.style_role,
                interior_parent=fixture.interior_parent,
            )
        )
    return adjusted


def _override_layout_placement(layout_id: int, fixtures: list[ResolvedFixture]) -> PlacementPlan | None:
    if layout_id != 58:
        return None
    island_strip = next((fixture for fixture in fixtures if fixture.name == "island_3_island_group_1"), None)
    if island_strip is None:
        return None
    tabletop_height = float(island_strip.pos[2] + island_strip.half_size[2])
    robot_xy = np.array([island_strip.pos[0], island_strip.pos[1]], dtype=np.float64)
    yaw = math.pi * 0.5
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    catch = np.array([robot_xy[0], robot_xy[1], tabletop_height], dtype=np.float64) + forward * ROBOT_FORWARD_REACH
    catch[2] += 0.31
    # Keep the cannon on the same rear island strip as the robot while still
    # giving the ball enough horizontal travel to arc into the gripper.
    launch_xy = np.array(
        [
            robot_xy[0] + min(0.30, island_strip.half_size[0] - 0.05),
            robot_xy[1] + min(0.08, island_strip.half_size[1] - 0.03),
        ],
        dtype=np.float64,
    )
    return PlacementPlan(
        counter_name=island_strip.name,
        edge_name="+y",
        robot_base_position=(float(robot_xy[0]), float(robot_xy[1]), tabletop_height),
        robot_base_yaw=float(yaw),
        tabletop_height=tabletop_height,
        launch_position=(float(launch_xy[0]), float(launch_xy[1]), tabletop_height),
        catch_position=(float(catch[0]), float(catch[1]), float(catch[2])),
    )


@functools.lru_cache(maxsize=256)
def resolve_scene(layout_id: int, style_id: int) -> ResolvedScene:
    backend = "yaml_fallback"
    native_error = None
    try:
        fixtures, style_config = _resolve_scene_fixtures_native(layout_id, style_id)
        backend = "robocasa_native"
    except Exception as exc:
        fixtures, style_config = _resolve_scene_fixtures(layout_id, style_id)
        native_error = f"{type(exc).__name__}: {exc}"
    fixtures = _apply_layout_fixture_overrides(layout_id, fixtures)
    placement = _override_layout_placement(layout_id, fixtures) or _placement_for_scene(fixtures)
    return ResolvedScene(
        layout_id=layout_id,
        style_id=style_id,
        fixtures=tuple(fixtures),
        style_config=style_config,
        room_bounds=_scene_bounds(fixtures),
        placement=placement,
        backend=backend,
        native_error=native_error,
    )
