"""Deterministic, outcome-independent scene randomization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

import yaml

from .hashing import sha256_json, stable_uint64

T = TypeVar("T")

BACKGROUND_STYLE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("clean_franka_lab", 0.20),
    ("robocasa_lab", 0.15),
    ("robocasa_kitchen", 0.20),
    ("robocasa_workbench", 0.15),
    ("robocasa_storage", 0.15),
    ("robocasa_tabletop", 0.15),
)

RANDOMIZATION_SCHEMA_VERSION = "dynamic-robot-randomization/v3"
RANDOMIZATION_LEVELS = ("R0", "R1", "R2", "OOD")
_ALLOWED_VARY_FIELDS = frozenset(
    {
        "scene_asset",
        "background_style",
        "lighting",
        "texture",
        "object_color",
        "object_asset",
        "camera_pose",
        "robot_start",
        "control_latency",
        "camera_latency",
        "controller_profile",
    }
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
    scene_asset_manifest_sha256: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        fields = (
            self.scene_asset_ids,
            self.lighting_ids,
            self.object_asset_ids,
            self.object_color_ids,
            self.tool_asset_ids,
            self.camera_preset_ids,
        )
        if any(len(values) != len(set(values)) for values in fields):
            raise ValueError("randomization catalog IDs must be unique within each field")
        unknown_hash_ids = sorted(
            set(self.scene_asset_manifest_sha256) - set(self.scene_asset_ids)
        )
        if unknown_hash_ids:
            raise ValueError(
                f"scene-asset hashes reference unknown IDs: {unknown_hash_ids}"
            )
        for asset_id in self.scene_asset_ids:
            digest = self.scene_asset_manifest_sha256.get(asset_id)
            if digest is None:
                raise ValueError(
                    f"scene asset {asset_id!r} is not bound to an admission-manifest hash"
                )
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"scene asset {asset_id!r} has a malformed manifest hash")


@dataclass(slots=True, frozen=True)
class RandomizationPolicy:
    """Versioned R0/R1/R2 field-admission policy loaded from YAML."""

    schema_version: str
    vary_by_level: Mapping[str, tuple[str, ...]]
    background_weights: tuple[tuple[str, float], ...]
    r2_requires_r1_acceptance: bool = True
    outcome_conditioned_resampling_forbidden: bool = True

    def validate(self) -> None:
        if self.schema_version != RANDOMIZATION_SCHEMA_VERSION:
            raise ValueError(
                f"randomization policy must use {RANDOMIZATION_SCHEMA_VERSION}"
            )
        if set(self.vary_by_level) != set(RANDOMIZATION_LEVELS):
            raise ValueError("randomization policy must define R0, R1, R2, and OOD")
        for level, values in self.vary_by_level.items():
            unknown = sorted(set(values) - _ALLOWED_VARY_FIELDS)
            if unknown:
                raise ValueError(f"randomization level {level} has unknown fields: {unknown}")
        if self.vary_by_level["R0"]:
            raise ValueError("R0 cannot vary any field")
        if not self.outcome_conditioned_resampling_forbidden:
            raise ValueError("outcome-conditioned resampling must remain forbidden")
        if (
            not self.background_weights
            or any(weight < 0 for _, weight in self.background_weights)
            or sum(weight for _, weight in self.background_weights) <= 0
        ):
            raise ValueError("background weights must have positive mass")

    def varies(self, level: str, field_name: str) -> bool:
        self.validate()
        if level not in self.vary_by_level:
            raise ValueError(f"unsupported randomization level {level!r}")
        return field_name in self.vary_by_level[level]

    @property
    def policy_hash(self) -> str:
        self.validate()
        return sha256_json(asdict(self))


def default_randomization_policy() -> RandomizationPolicy:
    return RandomizationPolicy(
        schema_version=RANDOMIZATION_SCHEMA_VERSION,
        vary_by_level={
            "R0": (),
            "R1": (
                "scene_asset",
                "background_style",
                "lighting",
                "texture",
                "object_color",
            ),
            "R2": (
                "scene_asset",
                "background_style",
                "lighting",
                "texture",
                "object_color",
                "object_asset",
                "camera_pose",
                "robot_start",
                "control_latency",
                "camera_latency",
                "controller_profile",
            ),
            "OOD": (
                "scene_asset",
                "background_style",
                "lighting",
                "texture",
                "object_color",
                "object_asset",
                "camera_pose",
                "robot_start",
                "control_latency",
                "camera_latency",
                "controller_profile",
            ),
        },
        background_weights=BACKGROUND_STYLE_WEIGHTS,
    )


def load_randomization_policy(path: str | Path) -> RandomizationPolicy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("randomization policy must contain a YAML mapping")
    levels = raw.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError("randomization policy lacks levels")
    vary_by_level: dict[str, tuple[str, ...]] = {}
    for level in RANDOMIZATION_LEVELS:
        value = levels.get(level)
        if not isinstance(value, Mapping):
            raise ValueError(f"randomization policy lacks level {level}")
        vary_by_level[level] = tuple(str(item) for item in value.get("vary", ()))
    weights = raw.get("background_mix")
    if not isinstance(weights, Mapping):
        raise ValueError("randomization policy lacks background_mix")
    validation = raw.get("validation") if isinstance(raw.get("validation"), Mapping) else {}
    rules = raw.get("rules") if isinstance(raw.get("rules"), Mapping) else {}
    policy = RandomizationPolicy(
        schema_version=str(raw.get("schema_version") or ""),
        vary_by_level=vary_by_level,
        background_weights=tuple((str(key), float(value)) for key, value in weights.items()),
        r2_requires_r1_acceptance=bool(validation.get("r2_requires_r1_acceptance", True)),
        outcome_conditioned_resampling_forbidden=bool(
            rules.get("outcome_conditioned_resampling_forbidden", False)
        ),
    )
    policy.validate()
    return policy


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
    rng_subseeds: Mapping[str, int]
    varied_fields: tuple[str, ...]
    randomization_policy_sha256: str
    scene_asset_manifest_sha256: str | None
    training_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RandomizationPlanner:
    """Plan bundle-level randomization without reading outcome or branch labels."""

    catalog: RandomizationCatalog = field(default_factory=RandomizationCatalog)
    seed: int = 0
    background_weights: tuple[tuple[str, float], ...] = BACKGROUND_STYLE_WEIGHTS
    policy: RandomizationPolicy = field(default_factory=default_randomization_policy)

    def plan(self, bundle_id: str, *, randomization_level: str = "R1") -> BundleRandomization:
        """Create a deterministic plan based only on bundle identity and seed."""

        if not bundle_id:
            raise ValueError("bundle_id is required")
        self.policy.validate()
        self.catalog.validate()
        if randomization_level not in RANDOMIZATION_LEVELS:
            raise ValueError(f"unsupported randomization level {randomization_level!r}")
        key = {"bundle_id": bundle_id, "level": randomization_level}

        def optional(values: Sequence[str], namespace: str) -> str | None:
            return deterministic_choice(values, key, seed=self.seed, namespace=namespace) if values else None

        def selected(
            values: Sequence[str],
            field_name: str,
            namespace: str,
        ) -> str | None:
            if not values:
                return None
            if self.policy.varies(randomization_level, field_name):
                return optional(values, namespace)
            return str(values[0])

        weights = (
            self.background_weights
            if self.background_weights != BACKGROUND_STYLE_WEIGHTS
            else self.policy.background_weights
        )
        background_style = (
            weighted_choice(weights, key, seed=self.seed, namespace="background_style")
            if self.policy.varies(randomization_level, "background_style")
            else "clean_franka_lab"
        )
        scene_asset_id = selected(
            self.catalog.scene_asset_ids, "scene_asset", "scene_asset"
        )
        rng_subseeds = {
            namespace: stable_uint64(
                {"bundle_id": bundle_id, "root_seed": self.seed},
                namespace=f"randomization-stream:{namespace}",
            )
            for namespace in (
                "physics",
                "initial_state",
                "camera",
                "assets",
                "controller",
                "scene_construction",
            )
        }
        return BundleRandomization(
            counterfactual_bundle_id=bundle_id,
            scene_asset_id=scene_asset_id,
            background_style=background_style,
            lighting_id=selected(
                self.catalog.lighting_ids, "lighting", "lighting"
            ) or "neutral",
            object_asset_id=selected(
                self.catalog.object_asset_ids, "object_asset", "object_asset"
            ),
            object_color_id=selected(
                self.catalog.object_color_ids, "object_color", "object_color"
            ) or "default",
            tool_asset_id=(
                str(self.catalog.tool_asset_ids[0])
                if self.catalog.tool_asset_ids
                else None
            ),
            camera_preset_id=selected(
                self.catalog.camera_preset_ids, "camera_pose", "camera_preset"
            ) or "main_secondary_v1",
            randomization_level=randomization_level,
            randomization_seed=stable_uint64(key, namespace=f"bundle-randomization:{self.seed}"),
            rng_subseeds=rng_subseeds,
            varied_fields=tuple(self.policy.vary_by_level[randomization_level]),
            randomization_policy_sha256=self.policy.policy_hash,
            scene_asset_manifest_sha256=(
                self.catalog.scene_asset_manifest_sha256.get(scene_asset_id)
                if scene_asset_id is not None
                else None
            ),
            training_eligible=randomization_level != "OOD",
        )


def validate_randomization_admission(
    randomization: Mapping[str, Any],
    *,
    r1_accepted: bool,
) -> None:
    """Reject unbound RoboCasa selections and premature R2 production use."""

    level = str(randomization.get("randomization_level") or "")
    if level not in RANDOMIZATION_LEVELS:
        raise ValueError(f"unknown randomization level {level!r}")
    if level == "R2" and not r1_accepted:
        raise ValueError("R2 cannot run before the same leaf passes R1 acceptance")
    background = str(randomization.get("background_style") or "")
    asset_id = randomization.get("scene_asset_id")
    asset_hash = randomization.get("scene_asset_manifest_sha256")
    if background.startswith("robocasa_"):
        if not asset_id:
            raise ValueError("RoboCasa background selection lacks an admitted asset ID")
        if (
            not isinstance(asset_hash, str)
            or len(asset_hash) != 64
            or any(character not in "0123456789abcdef" for character in asset_hash)
        ):
            raise ValueError("RoboCasa background selection lacks an admission-manifest hash")
    seeds = randomization.get("rng_subseeds")
    expected = {
        "physics",
        "initial_state",
        "camera",
        "assets",
        "controller",
        "scene_construction",
    }
    if not isinstance(seeds, Mapping) or set(seeds) != expected:
        raise ValueError("randomization lacks the complete independent RNG stream map")


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
        "rng_subseeds",
        "randomization_policy_sha256",
        "scene_asset_manifest_sha256",
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


def scene_attempt_seed(bundle_id: str, attempt_index: int, *, seed: int = 0) -> int:
    """Derive a recorded construction-attempt seed independently of outcomes."""

    if not bundle_id or attempt_index < 0:
        raise ValueError("scene attempts require a bundle ID and non-negative index")
    return stable_uint64(
        {"bundle_id": bundle_id, "attempt_index": attempt_index, "seed": seed},
        namespace="scene-construction-attempt/v1",
    )


def validate_scene_attempt_history(attempts: Sequence[Mapping[str, Any]]) -> None:
    """Allow resampling only for deterministic scene-construction failures."""

    permitted_failures = {
        "asset_unavailable",
        "asset_admission_failed",
        "swept_volume_intersection",
        "fixture_intersection",
        "camera_occlusion",
        "scene_compile_failed",
    }
    for expected_index, attempt in enumerate(attempts):
        if int(attempt.get("attempt_index", -1)) != expected_index:
            raise ValueError("scene construction attempts must be contiguous and ordered")
        if not isinstance(attempt.get("attempt_seed"), int):
            raise ValueError("scene construction attempt lacks its integer seed")
        status = str(attempt.get("status") or "")
        if status == "accepted":
            if expected_index != len(attempts) - 1:
                raise ValueError("no scene attempts may follow an accepted construction")
            continue
        failure = str(attempt.get("failure_code") or "")
        if status != "construction_failed" or failure not in permitted_failures:
            raise ValueError(
                "runtime physics/outcome failures can never trigger scene resampling"
            )


__all__ = [
    "BACKGROUND_STYLE_WEIGHTS",
    "BundleRandomization",
    "RANDOMIZATION_LEVELS",
    "RANDOMIZATION_SCHEMA_VERSION",
    "RandomizationCatalog",
    "RandomizationPlanner",
    "RandomizationPolicy",
    "assert_bundle_randomization_invariant",
    "default_randomization_policy",
    "deterministic_choice",
    "deterministic_uniform",
    "load_randomization_policy",
    "scene_attempt_seed",
    "validate_randomization_admission",
    "validate_scene_attempt_history",
    "weighted_choice",
]
