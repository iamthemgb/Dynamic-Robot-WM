"""Backend-neutral, serializable source scenario specification.

``SourceScenarioSpec`` is the immutable hand-off between deterministic planning
and a simulator-specific compiler.  It intentionally describes *what* must be
simulated without pretending that rigid, deformable, and fluid solvers share an
internal model.  Validation consults the authoritative corpus and backend
registries and rejects unsupported combinations before any output is created.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
import re
from typing import Any, Mapping, Sequence

from ..backends.actuator_only import ACTION_SEMANTICS, action_spec
from .corpus_registry import (
    BackendCapabilityRegistry,
    CorpusRegistry,
    KNOWN_EMBODIMENTS,
    load_backend_capability_registry,
    load_corpus_registry,
)
from .hashing import canonical_json_bytes, sha256_json
from .grasp_retention import DEFAULT_GRASP_RETENTION, GraspRetentionThresholds


LEGACY_SOURCE_SCENARIO_SCHEMA_VERSION = "dynamic-robot-source-scenario/v1"
SOURCE_SCENARIO_SCHEMA_VERSION = "dynamic-robot-source-scenario/v2"
_CONTENT_HASH_PATTERN = re.compile(
    r"^(?:[0-9a-f]{64}|sha256:[0-9a-f]{64}|git:[0-9a-f]{40})$"
)


class SourceScenarioValidationError(ValueError):
    """A source scenario is unsafe, incomplete, or non-serializable."""


def _finite_vector(value: Sequence[Any], length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise SourceScenarioValidationError(f"{label} must contain {length} values")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise SourceScenarioValidationError(f"{label} contains a non-finite value")
    return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceScenarioValidationError(f"{label} must be a mapping")
    return value


def _content_hash(value: str, label: str) -> str:
    normalized = str(value).lower()
    if not _CONTENT_HASH_PATTERN.fullmatch(normalized):
        raise SourceScenarioValidationError(
            f"{label} must be a SHA-256 content hash or an explicit git commit hash"
        )
    return normalized


def _validate_json_value(value: Any, label: str) -> None:
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise SourceScenarioValidationError(
            f"{label} is not finite canonical-JSON data: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class PoseSpec:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def validate(self) -> None:
        _finite_vector(self.position_m, 3, "pose position")
        quaternion = _finite_vector(self.quaternion_wxyz, 4, "pose quaternion")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(norm - 1.0) > 1e-5:
            raise SourceScenarioValidationError("pose quaternion must be normalized")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoseSpec":
        result = cls(
            position_m=_finite_vector(value.get("position_m", ()), 3, "pose position"),
            quaternion_wxyz=_finite_vector(
                value.get("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0)),
                4,
                "pose quaternion",
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class EmbodimentSpec:
    end_effector: str
    robot_model: str
    action_names: tuple[str, ...]
    action_semantics: str = ACTION_SEMANTICS

    def validate(self) -> None:
        if self.end_effector not in KNOWN_EMBODIMENTS:
            raise SourceScenarioValidationError(
                f"unknown embodiment {self.end_effector!r}"
            )
        if not self.robot_model:
            raise SourceScenarioValidationError("embodiment robot_model is required")
        if len(set(self.action_names)) != len(self.action_names) or any(
            not name for name in self.action_names
        ):
            raise SourceScenarioValidationError("action_names must be unique and non-empty")
        if self.end_effector == "no_robot":
            if (
                self.robot_model != "none"
                or self.action_names
                or self.action_semantics != "no_actuators/v1"
            ):
                raise SourceScenarioValidationError(
                    "no_robot requires robot_model='none', no actuator actions, and "
                    "action_semantics='no_actuators/v1'"
                )
        else:
            expected = action_spec(self.end_effector)
            if self.action_semantics != expected.semantics:
                raise SourceScenarioValidationError(
                    f"real-gripper actions must use {expected.semantics}"
                )
            if self.action_names != expected.actuator_names:
                raise SourceScenarioValidationError(
                    "real-gripper action_names must be the exact seven Panda arm "
                    "controls followed by the embodiment's physical gripper actuator"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EmbodimentSpec":
        result = cls(
            end_effector=str(value.get("end_effector", "")),
            robot_model=str(value.get("robot_model", "")),
            action_names=tuple(str(item) for item in value.get("action_names", ())),
            action_semantics=str(
                value.get(
                    "action_semantics",
                    "no_actuators/v1"
                    if str(value.get("end_effector", "")) == "no_robot"
                    else ACTION_SEMANTICS,
                )
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class FixtureSpec:
    fixture_id: str
    fixture_type: str
    pose: PoseSpec
    physical: bool = True
    anchored: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.fixture_id or not self.fixture_type:
            raise SourceScenarioValidationError("fixture ID and type are required")
        if not self.physical or not self.anchored:
            raise SourceScenarioValidationError(
                f"task fixture {self.fixture_id} must be physical and anchored"
            )
        self.pose.validate()
        _validate_json_value(self.parameters, f"fixture {self.fixture_id} parameters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FixtureSpec":
        result = cls(
            fixture_id=str(value.get("fixture_id", "")),
            fixture_type=str(value.get("fixture_type", "")),
            pose=PoseSpec.from_dict(_mapping(value.get("pose"), "fixture pose")),
            physical=bool(value.get("physical", True)),
            anchored=bool(value.get("anchored", True)),
            parameters=dict(_mapping(value.get("parameters", {}), "fixture parameters")),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class ActuatorPhaseSpec:
    name: str
    start_s: float
    end_s: float
    commands: Mapping[str, float]
    interpolation: str = "jerk_limited"

    def validate(self) -> None:
        if not self.name or not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise SourceScenarioValidationError("actuator phase name/times are required")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise SourceScenarioValidationError("actuator phase interval must be positive")
        if self.interpolation not in {"hold", "linear", "smoothstep", "jerk_limited"}:
            raise SourceScenarioValidationError(
                f"unsupported actuator interpolation {self.interpolation!r}"
            )
        if not self.commands:
            raise SourceScenarioValidationError("actuator phases require commands")
        for name, value in self.commands.items():
            if not name or not math.isfinite(float(value)):
                raise SourceScenarioValidationError(
                    f"actuator phase {self.name} has an invalid command"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActuatorPhaseSpec":
        commands = _mapping(value.get("commands"), "actuator commands")
        result = cls(
            name=str(value.get("name", "")),
            start_s=float(value.get("start_s", math.nan)),
            end_s=float(value.get("end_s", math.nan)),
            commands={str(key): float(item) for key, item in commands.items()},
            interpolation=str(value.get("interpolation", "jerk_limited")),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class SourceCameraSpec:
    name: str
    role: str
    pose: PoseSpec
    look_at_m: tuple[float, float, float]
    width: int = 832
    height: int = 480
    fps: int = 30
    fovy_deg: float = 48.0

    def validate(self) -> None:
        if not self.name or not self.role:
            raise SourceScenarioValidationError("camera name and role are required")
        self.pose.validate()
        look_at = _finite_vector(self.look_at_m, 3, "camera look-at")
        if math.dist(self.pose.position_m, look_at) < 1e-6:
            raise SourceScenarioValidationError("camera position and look-at must differ")
        if (self.width, self.height, self.fps) != (832, 480, 30):
            raise SourceScenarioValidationError("canonical cameras must be 832x480 at 30 Hz")
        if not math.isfinite(self.fovy_deg) or not 10.0 <= self.fovy_deg <= 120.0:
            raise SourceScenarioValidationError("camera fovy_deg is implausible")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceCameraSpec":
        result = cls(
            name=str(value.get("name", "")),
            role=str(value.get("role", "")),
            pose=PoseSpec.from_dict(_mapping(value.get("pose"), "camera pose")),
            look_at_m=_finite_vector(value.get("look_at_m", ()), 3, "camera look-at"),
            width=int(value.get("width", 832)),
            height=int(value.get("height", 480)),
            fps=int(value.get("fps", 30)),
            fovy_deg=float(value.get("fovy_deg", 48.0)),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class RNGSubseeds:
    physics: int
    initial_state: int
    camera: int
    assets: int
    controller: int
    scene_construction: int

    def validate(self) -> None:
        values = (
            self.physics,
            self.initial_state,
            self.camera,
            self.assets,
            self.controller,
            self.scene_construction,
        )
        if any(isinstance(value, bool) or value < 0 or value >= 2**64 for value in values):
            raise SourceScenarioValidationError("RNG subseeds must be unsigned 64-bit integers")
        if len(set(values)) != len(values):
            raise SourceScenarioValidationError(
                "physics, initial-state, camera, asset, controller, and scene RNG "
                "streams must be independent"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RNGSubseeds":
        required = (
            "physics",
            "initial_state",
            "camera",
            "assets",
            "controller",
            "scene_construction",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise SourceScenarioValidationError(
                f"missing RNG sub-seeds: {', '.join(missing)}"
            )
        result = cls(**{name: int(value[name]) for name in required})
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class RoboCasaAssetSpec:
    asset_id: str
    asset_type: str
    source_xml: str
    xml_sha256: str
    mesh_sha256: Mapping[str, str]
    texture_sha256: Mapping[str, str]
    scale_xyz: tuple[float, float, float]
    aabb_min_m: tuple[float, float, float]
    aabb_max_m: tuple[float, float, float]
    pose: PoseSpec
    collision_enabled: bool = False

    def validate(self) -> None:
        if not self.asset_id or not self.asset_type or not self.source_xml:
            raise SourceScenarioValidationError("RoboCasa asset identity/XML are required")
        _content_hash(self.xml_sha256, f"RoboCasa XML {self.asset_id}")
        for label, manifest in (
            ("mesh", self.mesh_sha256),
            ("texture", self.texture_sha256),
        ):
            for path, digest in manifest.items():
                if not path:
                    raise SourceScenarioValidationError(
                        f"RoboCasa {label} manifest contains an empty path"
                    )
                _content_hash(digest, f"RoboCasa {label} {path}")
        scale = _finite_vector(self.scale_xyz, 3, "RoboCasa asset scale")
        if any(value <= 0 for value in scale):
            raise SourceScenarioValidationError("RoboCasa asset scales must be positive")
        lower = _finite_vector(self.aabb_min_m, 3, "RoboCasa AABB minimum")
        upper = _finite_vector(self.aabb_max_m, 3, "RoboCasa AABB maximum")
        if any(left >= right for left, right in zip(lower, upper)):
            raise SourceScenarioValidationError("RoboCasa asset AABB must have positive extent")
        self.pose.validate()
        if self.collision_enabled:
            raise SourceScenarioValidationError(
                "RoboCasa background assets must be visual-only with collisions disabled"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoboCasaAssetSpec":
        meshes = _mapping(value.get("mesh_sha256", {}), "RoboCasa mesh hashes")
        textures = _mapping(value.get("texture_sha256", {}), "RoboCasa texture hashes")
        result = cls(
            asset_id=str(value.get("asset_id", "")),
            asset_type=str(value.get("asset_type", "")),
            source_xml=str(value.get("source_xml", "")),
            xml_sha256=str(value.get("xml_sha256", "")),
            mesh_sha256={str(key): str(item) for key, item in meshes.items()},
            texture_sha256={str(key): str(item) for key, item in textures.items()},
            scale_xyz=_finite_vector(value.get("scale_xyz", ()), 3, "RoboCasa scale"),
            aabb_min_m=_finite_vector(value.get("aabb_min_m", ()), 3, "RoboCasa AABB min"),
            aabb_max_m=_finite_vector(value.get("aabb_max_m", ()), 3, "RoboCasa AABB max"),
            pose=PoseSpec.from_dict(_mapping(value.get("pose"), "RoboCasa pose")),
            collision_enabled=bool(value.get("collision_enabled", False)),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class RoboCasaAssetManifest:
    catalog_id: str
    catalog_version: str
    asset_root_id: str
    catalog_sha256: str
    license_manifest_sha256: str
    assets: tuple[RoboCasaAssetSpec, ...] = ()

    def validate(self) -> None:
        if not self.catalog_id or not self.catalog_version or not self.asset_root_id:
            raise SourceScenarioValidationError(
                "RoboCasa catalog identity, version, and root ID are required"
            )
        _content_hash(self.catalog_sha256, "RoboCasa catalog")
        _content_hash(self.license_manifest_sha256, "RoboCasa license manifest")
        ids = [asset.asset_id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise SourceScenarioValidationError("RoboCasa asset IDs must be unique")
        for asset in self.assets:
            asset.validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoboCasaAssetManifest":
        result = cls(
            catalog_id=str(value.get("catalog_id", "")),
            catalog_version=str(value.get("catalog_version", "")),
            asset_root_id=str(value.get("asset_root_id", "")),
            catalog_sha256=str(value.get("catalog_sha256", "")),
            license_manifest_sha256=str(value.get("license_manifest_sha256", "")),
            assets=tuple(
                RoboCasaAssetSpec.from_dict(_mapping(item, "RoboCasa asset"))
                for item in value.get("assets", ())
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class CounterfactualIdentity:
    bundle_id: str
    split_group_id: str
    branch_id: str
    sibling_index: int
    physics_family_id: str | None = None

    def validate(self) -> None:
        if not self.bundle_id or not self.split_group_id or not self.branch_id:
            raise SourceScenarioValidationError(
                "counterfactual bundle, split group, and branch IDs are required"
            )
        if self.sibling_index < 0:
            raise SourceScenarioValidationError("counterfactual sibling_index cannot be negative")
        if self.physics_family_id is not None and not self.physics_family_id:
            raise SourceScenarioValidationError("physics_family_id cannot be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CounterfactualIdentity":
        result = cls(
            bundle_id=str(value.get("bundle_id", "")),
            split_group_id=str(value.get("split_group_id", "")),
            branch_id=str(value.get("branch_id", "")),
            sibling_index=int(value.get("sibling_index", -1)),
            physics_family_id=(
                None
                if value.get("physics_family_id") is None
                else str(value.get("physics_family_id"))
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class SourceScenarioSpec:
    """Complete deterministic input to one admitted source physics backend."""

    scenario_id: str
    corpus_leaf_id: str
    task_variant: str
    backend: str
    duration_s: float
    physics: Mapping[str, Any]
    initial_state: Mapping[str, Any]
    embodiment: EmbodimentSpec
    fixtures: tuple[FixtureSpec, ...]
    actuator_phases: tuple[ActuatorPhaseSpec, ...]
    cameras: tuple[SourceCameraSpec, SourceCameraSpec]
    rng_subseeds: RNGSubseeds
    robocasa_manifest: RoboCasaAssetManifest
    counterfactual: CounterfactualIdentity
    source_hashes: Mapping[str, str]
    schema_version: str = SOURCE_SCENARIO_SCHEMA_VERSION

    def validate(
        self,
        *,
        corpus_registry: CorpusRegistry | None = None,
        backend_registry: BackendCapabilityRegistry | None = None,
        require_released: bool = False,
    ) -> None:
        if self.schema_version not in {
            LEGACY_SOURCE_SCENARIO_SCHEMA_VERSION,
            SOURCE_SCENARIO_SCHEMA_VERSION,
        }:
            raise SourceScenarioValidationError(
                f"unsupported source scenario schema {self.schema_version!r}"
            )
        if not self.scenario_id:
            raise SourceScenarioValidationError("scenario_id is required")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise SourceScenarioValidationError("duration_s must be finite and positive")
        corpus = corpus_registry or load_corpus_registry()
        if backend_registry is not None:
            backends = backend_registry
        elif corpus_registry is None:
            backends = load_backend_capability_registry()
        else:
            backends = load_backend_capability_registry(corpus=corpus)
        leaf = corpus.resolve(
            self.corpus_leaf_id,
            embodiment=self.embodiment.end_effector,
            task_variant=self.task_variant,
            require_released=require_released,
        )
        if self.backend != leaf.backend:
            raise SourceScenarioValidationError(
                f"{self.corpus_leaf_id} must use backend {leaf.backend}, not {self.backend}"
            )
        backend_capability = backends.resolve(
            self.backend,
            self.corpus_leaf_id,
            self.embodiment.end_effector,
            require_released=require_released,
        )
        self.embodiment.validate()
        if not self.physics or not self.initial_state:
            raise SourceScenarioValidationError("physics and initial_state cannot be empty")
        _validate_json_value(self.physics, "physics")
        _validate_json_value(self.initial_state, "initial_state")
        if (
            self.schema_version == SOURCE_SCENARIO_SCHEMA_VERSION
            and self.backend == "source_mujoco"
        ):
            if self.physics.get("evaluator") != leaf.evaluator:
                raise SourceScenarioValidationError(
                    "source_mujoco physics evaluator differs from the corpus registry"
                )
            if self.embodiment.end_effector != "no_robot":
                raw_retention = self.physics.get("grasp_retention")
                if not isinstance(raw_retention, Mapping):
                    raise SourceScenarioValidationError(
                        "actuated source_mujoco physics lacks grasp retention"
                    )
                try:
                    retention = GraspRetentionThresholds.from_dict(raw_retention)
                except ValueError as error:
                    raise SourceScenarioValidationError(str(error)) from error
                if retention != DEFAULT_GRASP_RETENTION:
                    raise SourceScenarioValidationError(
                        "source_mujoco grasp retention differs from evaluator v1.5.0"
                    )
        fixture_ids = [fixture.fixture_id for fixture in self.fixtures]
        if len(fixture_ids) != len(set(fixture_ids)):
            raise SourceScenarioValidationError("fixture IDs must be unique")
        for fixture in self.fixtures:
            fixture.validate()
        allowed_actions = set(self.embodiment.action_names)
        previous_end = -math.inf
        for phase in self.actuator_phases:
            phase.validate()
            if phase.start_s < previous_end - 1e-12:
                raise SourceScenarioValidationError("actuator phases cannot overlap")
            if phase.end_s > self.duration_s + 1e-12:
                raise SourceScenarioValidationError("actuator phase exceeds scenario duration")
            unexpected = sorted(set(phase.commands) - allowed_actions)
            if unexpected:
                raise SourceScenarioValidationError(
                    f"phase {phase.name} commands undeclared actuators: {unexpected}"
                )
            missing = sorted(allowed_actions - set(phase.commands))
            if missing:
                raise SourceScenarioValidationError(
                    f"phase {phase.name} omits actuator commands without explicit hold "
                    f"semantics: {missing}"
                )
            previous_end = phase.end_s
        if self.embodiment.end_effector == "no_robot":
            if self.actuator_phases:
                raise SourceScenarioValidationError(
                    "passive no_robot scenes cannot have actuator phases"
                )
        elif not self.actuator_phases:
            raise SourceScenarioValidationError(
                "actuated scenes require at least one actuator phase"
            )
        if len(self.cameras) != 2 or {camera.name for camera in self.cameras} != {
            "main",
            "secondary",
        }:
            raise SourceScenarioValidationError(
                "source scenarios require exactly main and secondary cameras"
            )
        for camera in self.cameras:
            camera.validate()
        self.rng_subseeds.validate()
        self.robocasa_manifest.validate()
        self.counterfactual.validate()
        if not self.source_hashes:
            raise SourceScenarioValidationError("source_hashes cannot be empty")
        normalized_hashes = {
            str(key): _content_hash(str(value), f"source hash {key}")
            for key, value in self.source_hashes.items()
        }
        if any(not key for key in normalized_hashes):
            raise SourceScenarioValidationError("source hash IDs cannot be empty")
        missing_backend_hashes = sorted(
            set(backend_capability.source_hashes) - set(normalized_hashes)
        )
        mismatched_backend_hashes = sorted(
            name
            for name, expected in backend_capability.source_hashes.items()
            if normalized_hashes.get(name) not in {None, expected.lower()}
        )
        if missing_backend_hashes or mismatched_backend_hashes:
            raise SourceScenarioValidationError(
                "source scenario does not match the backend capability pins: "
                f"missing={missing_backend_hashes}, mismatched={mismatched_backend_hashes}"
            )
        catalog_hash = self.robocasa_manifest.catalog_sha256.lower()
        if normalized_hashes.get("robocasa_catalog") != catalog_hash:
            raise SourceScenarioValidationError(
                "source_hashes.robocasa_catalog must bind the selected RoboCasa manifest"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()

        def convert(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    @property
    def spec_hash(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> "SourceScenarioSpec":
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as error:
            raise SourceScenarioValidationError("invalid source scenario JSON") from error
        return cls.from_dict(_mapping(raw, "source scenario"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceScenarioSpec":
        physics = dict(_mapping(value.get("physics"), "physics"))
        initial_state = dict(_mapping(value.get("initial_state"), "initial_state"))
        hashes = _mapping(value.get("source_hashes"), "source_hashes")
        cameras = tuple(
            SourceCameraSpec.from_dict(_mapping(item, "camera"))
            for item in value.get("cameras", ())
        )
        if len(cameras) != 2:
            raise SourceScenarioValidationError("source scenario requires two cameras")
        result = cls(
            scenario_id=str(value.get("scenario_id", "")),
            corpus_leaf_id=str(value.get("corpus_leaf_id", "")),
            task_variant=str(value.get("task_variant", "")),
            backend=str(value.get("backend", "")),
            duration_s=float(value.get("duration_s", math.nan)),
            physics=physics,
            initial_state=initial_state,
            embodiment=EmbodimentSpec.from_dict(
                _mapping(value.get("embodiment"), "embodiment")
            ),
            fixtures=tuple(
                FixtureSpec.from_dict(_mapping(item, "fixture"))
                for item in value.get("fixtures", ())
            ),
            actuator_phases=tuple(
                ActuatorPhaseSpec.from_dict(_mapping(item, "actuator phase"))
                for item in value.get("actuator_phases", ())
            ),
            cameras=(cameras[0], cameras[1]),
            rng_subseeds=RNGSubseeds.from_dict(
                _mapping(value.get("rng_subseeds"), "RNG sub-seeds")
            ),
            robocasa_manifest=RoboCasaAssetManifest.from_dict(
                _mapping(value.get("robocasa_manifest"), "RoboCasa manifest")
            ),
            counterfactual=CounterfactualIdentity.from_dict(
                _mapping(value.get("counterfactual"), "counterfactual identity")
            ),
            source_hashes={str(key): str(item) for key, item in hashes.items()},
            schema_version=str(
                value.get("schema_version", SOURCE_SCENARIO_SCHEMA_VERSION)
            ),
        )
        result.validate()
        return result


__all__ = [
    "ActuatorPhaseSpec",
    "CounterfactualIdentity",
    "EmbodimentSpec",
    "FixtureSpec",
    "LEGACY_SOURCE_SCENARIO_SCHEMA_VERSION",
    "PoseSpec",
    "RNGSubseeds",
    "RoboCasaAssetManifest",
    "RoboCasaAssetSpec",
    "SOURCE_SCENARIO_SCHEMA_VERSION",
    "SourceCameraSpec",
    "SourceScenarioSpec",
    "SourceScenarioValidationError",
]
