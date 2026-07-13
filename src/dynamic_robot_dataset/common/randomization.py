"""Deterministic, outcome-independent scene randomization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence, TypeVar

from .hashing import stable_uint64

T = TypeVar("T")

BACKGROUND_STYLE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("clean_franka_lab", 0.60),
    ("robocasa_kitchen_tabletop", 0.25),
    ("robotwin_cluttered_tabletop", 0.15),
)


def deterministic_uniform(key: Any, *, seed: int = 0, namespace: str = "") -> float:
    """Return a stable value in [0, 1), independent of process RNG state."""

    integer = stable_uint64({"seed": seed, "key": key}, namespace=namespace)
    return integer / float(1 << 64)


def weighted_choice(
    options: Sequence[tuple[T, float]],
    key: Any,
    *,
    seed: int = 0,
    namespace: str = "",
) -> T:
    """Choose deterministically from non-negative weighted options."""

    if not options or any(weight < 0 for _, weight in options):
        raise ValueError("Weighted options must be non-empty and non-negative")
    total = sum(weight for _, weight in options)
    if total <= 0:
        raise ValueError("At least one option must have positive weight")
    target = deterministic_uniform(key, seed=seed, namespace=namespace) * total
    cumulative = 0.0
    for option, weight in options:
        cumulative += weight
        if target < cumulative:
            return option
    return options[-1][0]


def deterministic_choice(values: Sequence[T], key: Any, *, seed: int = 0, namespace: str = "") -> T:
    """Choose a stable item from a sequence."""

    if not values:
        raise ValueError("Cannot choose from an empty sequence")
    index = stable_uint64({"seed": seed, "key": key}, namespace=namespace) % len(values)
    return values[index]


@dataclass(slots=True, frozen=True)
class RandomizationCatalog:
    """Allowlisted asset and visual identifiers available to randomization."""

    scene_asset_ids: tuple[str, ...] = ()
    lighting_ids: tuple[str, ...] = ("neutral",)
    object_asset_ids: tuple[str, ...] = ()
    object_color_ids: tuple[str, ...] = ("default",)
    tool_asset_ids: tuple[str, ...] = ()
    camera_preset_ids: tuple[str, ...] = ("main_secondary_v1",)


@dataclass(slots=True, frozen=True)
class BundleRandomization:
    """Visual configuration shared by every branch in one bundle."""

    counterfactual_bundle_id: str
    scene_asset_id: str | None
    background_style: str
    lighting_id: str
    object_asset_id: str | None
    object_color_id: str
    tool_asset_id: str | None
    camera_preset_id: str
    randomization_level: str
    randomization_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RandomizationPlanner:
    """Plan bundle-level randomization without reading outcome or branch labels."""

    catalog: RandomizationCatalog = field(default_factory=RandomizationCatalog)
    seed: int = 0
    background_weights: tuple[tuple[str, float], ...] = BACKGROUND_STYLE_WEIGHTS

    def plan(self, bundle_id: str, *, randomization_level: str = "R1") -> BundleRandomization:
        """Create a deterministic plan based only on bundle identity and seed."""

        if not bundle_id:
            raise ValueError("bundle_id is required")
        key = {"bundle_id": bundle_id, "level": randomization_level}

        def optional(values: Sequence[str], namespace: str) -> str | None:
            return deterministic_choice(values, key, seed=self.seed, namespace=namespace) if values else None

        return BundleRandomization(
            counterfactual_bundle_id=bundle_id,
            scene_asset_id=optional(self.catalog.scene_asset_ids, "scene_asset"),
            background_style=weighted_choice(
                self.background_weights, key, seed=self.seed, namespace="background_style"
            ),
            lighting_id=deterministic_choice(
                self.catalog.lighting_ids, key, seed=self.seed, namespace="lighting"
            ),
            object_asset_id=optional(self.catalog.object_asset_ids, "object_asset"),
            object_color_id=deterministic_choice(
                self.catalog.object_color_ids, key, seed=self.seed, namespace="object_color"
            ),
            tool_asset_id=optional(self.catalog.tool_asset_ids, "tool_asset"),
            camera_preset_id=deterministic_choice(
                self.catalog.camera_preset_ids, key, seed=self.seed, namespace="camera_preset"
            ),
            randomization_level=randomization_level,
            randomization_seed=stable_uint64(key, namespace=f"bundle-randomization:{self.seed}"),
        )


def assert_bundle_randomization_invariant(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject bundles whose visual fields vary across action/physics branches."""

    visual_fields = (
        "scene_asset_id",
        "background_style",
        "lighting_id",
        "object_asset_id",
        "object_color_id",
        "tool_asset_id",
        "camera_preset_id",
    )
    signatures: dict[tuple[str, str], tuple[Any, ...]] = {}
    for record in records:
        source = record.get("randomization", record)
        signature = tuple(source.get(field) for field in visual_fields)
        for field in ("counterfactual_bundle_id", "physics_counterfactual_family_id"):
            identifier = record.get(field)
            if identifier is None:
                continue
            key = (field, str(identifier))
            if key in signatures and signatures[key] != signature:
                raise ValueError(f"Visual randomization changed within {field}={identifier}")
            signatures[key] = signature
