from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .robocasa_assets import robocasa_assets_root
from .utils import sphere_mass


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "projectile"


def _parse_vector(raw: str | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw:
        return default
    return tuple(float(v) for v in raw.split())


def _robocasa_lightwheel_root() -> Path:
    root = robocasa_assets_root()
    candidates = (
        root / "objects" / "lightwheel",
        root / "lightwheel",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def robocasa_projectile_model_xmls(category: str) -> list[Path]:
    category_root = _robocasa_lightwheel_root() / category
    return sorted(category_root.glob("*/model.xml"))


def robocasa_projectile_bbox(model_xml: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    root = ET.parse(model_xml).getroot()
    region = root.find(".//geom[@name='reg_bbox']")
    if region is None:
        raise ValueError(f"No reg_bbox geom found in {model_xml}")
    pos = _parse_vector(region.get("pos"), default=(0.0, 0.0, 0.0))
    size = _parse_vector(region.get("size"), default=(0.025, 0.025, 0.025))
    if len(pos) != 3 or len(size) != 3:
        raise ValueError(f"Expected 3D reg_bbox pos/size in {model_xml}")
    return (float(pos[0]), float(pos[1]), float(pos[2])), (float(size[0]), float(size[1]), float(size[2]))


def make_robocasa_projectile_spec(
    category: str,
    *,
    variant_index: int = 0,
    collision_shape: str = "box",
    target_max_half_extent: float | None = 0.0245,
    collision_padding: float = 0.0015,
    density: float = 42.0,
) -> dict:
    model_xmls = robocasa_projectile_model_xmls(category)
    if not model_xmls:
        raise FileNotFoundError(f"No RoboCasa lightwheel model.xml files found for category {category!r}")
    model_xml = model_xmls[int(variant_index) % len(model_xmls)]
    bbox_pos, bbox_half_size = robocasa_projectile_bbox(model_xml)
    max_half = max(float(v) for v in bbox_half_size)
    scale = 1.0
    if target_max_half_extent is not None and target_max_half_extent > 0.0 and max_half > target_max_half_extent:
        scale = float(target_max_half_extent) / max_half

    scaled_half_size = tuple(float(v) * scale for v in bbox_half_size)
    effective_radius = max(scaled_half_size)
    shape = str(collision_shape).strip().lower()
    if shape == "sphere":
        collision_size = (effective_radius + float(collision_padding),)
    elif shape == "box":
        collision_size = tuple(max(0.006, float(v) + float(collision_padding)) for v in scaled_half_size)
    else:
        raise ValueError("--collision-shape must be 'box' or 'sphere'")

    return {
        "type": "robocasa_model",
        "source": "robocasa",
        "category": category,
        "variant": model_xml.parent.name,
        "label": f"{category}/{model_xml.parent.name}",
        "model_xml": str(model_xml),
        "bbox_pos": [float(v) for v in bbox_pos],
        "bbox_half_size": [float(v) for v in bbox_half_size],
        "scaled_bbox_half_size": [float(v) for v in scaled_half_size],
        "scale": float(scale),
        "effective_radius": float(effective_radius),
        "collision_shape": shape,
        "collision_size": [float(v) for v in collision_size],
        "collision_padding": float(collision_padding),
        "mass": sphere_mass(effective_radius, density),
        "density": float(density),
        "instance": f"catch_{_safe_name(category)}_{_safe_name(model_xml.parent.name)}",
    }
