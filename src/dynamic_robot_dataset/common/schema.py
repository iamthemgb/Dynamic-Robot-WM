"""Versioned canonical metadata schema for dynamic-robot episodes.

The schema deliberately keeps persistent physics, transient state, and robot
actions in separate named structures.  Dataclasses are the in-process contract;
their ``to_dict`` output is the portable Arrow/JSON representation.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .hashing import sha256_json
from .paths import portable_relative_path

if TYPE_CHECKING:
    from .outcomes import OutcomeResult

SCHEMA_VERSION = "dynamic-robot-dataset/v1"
WAN_MANIFEST_VERSION = "wan-physics-manifest/v1"


class SchemaValidationError(ValueError):
    """A record violates a canonical dataset invariant."""


class LabelStatus(str, Enum):
    """Confidence tier for an objective task label."""

    VERIFIED = "verified_objective"
    UNVERIFIED = "unverified"
    PROXY = "proxy"


class DynamicsMode(str, Enum):
    """How object dynamics were produced during an episode."""

    FREE_CONTACT = "free_contact"
    ASSISTED_CONTACT = "assisted_contact"
    SCRIPTED_MOTION = "scripted_motion"


class ReleaseTier(str, Enum):
    """Dataset publication tier."""

    FREE_CONTACT = "free_contact"
    ASSISTED_CONTACT = "assisted_contact"
    SCRIPTED_MOTION = "scripted_motion"
    UNVERIFIED = "unverified"
    QUARANTINE = "quarantine"


class Split(str, Enum):
    """Dataset-local split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNASSIGNED = "unassigned"


class PhysicsValueKind(str, Enum):
    """Interpretation of a saved physics parameter."""

    PHYSICAL = "physical"
    CALIBRATED_EFFECTIVE = "calibrated_effective"
    SIMULATOR_PROXY = "simulator_proxy"
    UNKNOWN = "unknown"
    NOT_IMPLEMENTED = "not_implemented"


# Versioned in configs/schema/failure_codes_v1.yaml. Keeping the runtime set in
# the package makes validation work after installation; a contract test requires
# exact equality with the human-readable YAML catalog.
FAILURE_CODES = frozenset(
    {
        "unexpected_failure",
        "spatial_near_miss",
        "timing_near_miss",
        "no_contact",
        "contact_without_completion",
        "slip",
        "unstable_retention",
        "wrong_action",
        "no_op",
        "bad_rebound_prediction",
        "floor_contact",
        "excessive_penetration",
        "unstable_physics",
        "cropped_event",
        "truncated_outcome",
        "cloth_metric_failed",
        "rope_metric_failed",
        "unintended_snag",
        "ring_not_crossed",
        "soft_body_metric_failed",
        "label_unverified",
        "controller_no_op",
        "receptacle_near_miss",
        "receptacle_missed_object",
        "object_not_retained",
        "tool_near_miss",
        "tool_missed_rolling_object",
        "object_not_retained_in_goal",
        "paddle_missed_projectile",
        "paddle_near_miss",
        "incorrect_rebound_direction",
        "target_region_not_contacted",
        "insufficient_cloth_displacement",
        "corner_not_grasped",
        "insufficient_corner_lift",
        "fold_edge_not_grasped",
        "insufficient_fold_overlap",
        "cloth_not_grasped",
        "cloth_spill_outside_box",
        "coil_geometry_not_formed",
        "endpoint_not_dragged_to_target",
        "rope_not_draped_over_support",
        "wave_amplitude_or_frequency_too_low",
        "insufficient_rope_sweep",
        "rope_did_not_cross_and_remain_beyond_ring",
        "insufficient_tug_displacement",
        "rope_not_twirl_elevated",
        "insufficient_wrap_around_post",
        "insufficient_compression",
        "insufficient_indentation_depth",
        "insufficient_tensile_strain",
        "insufficient_rebound_height",
        "legacy_proxy_quarantined",
    }
)


def validate_failure_mode(task_success: bool, failure_mode: str) -> None:
    """Enforce the versioned success/failure taxonomy."""

    normalized = failure_mode.strip().lower()
    if task_success:
        if normalized != "none":
            raise SchemaValidationError("Successful episodes must use failure_mode=none")
        return
    if normalized in {"", "none", "null"}:
        raise SchemaValidationError("Failed episodes require a concrete failure_mode")
    if normalized not in FAILURE_CODES:
        raise SchemaValidationError(f"Unknown failure_mode: {failure_mode}")


@dataclass(slots=True, frozen=True)
class CoordinateConvention:
    """Dataset-wide coordinate and pose declaration."""

    units: str = "SI"
    handedness: str = "right"
    up_axis: str = "+Z"
    quaternion_order: str = "WXYZ"
    pose_convention: str = "position_then_quaternion"
    default_velocity_frame: str = "world"

    def validate(self) -> None:
        if self.units != "SI":
            raise SchemaValidationError("Canonical datasets must use SI units")
        if self.handedness != "right" or self.up_axis != "+Z":
            raise SchemaValidationError("Canonical world frame must be right-handed with +Z up")
        if self.quaternion_order != "WXYZ":
            raise SchemaValidationError("Canonical quaternion ordering must be WXYZ")


@dataclass(slots=True, frozen=True)
class TimeBase:
    """Exact episode timing declaration."""

    sim_hz: float
    control_hz: float
    video_hz: float = 30.0
    timestamp_dtype: str = "float64_seconds"
    video_time_base_num: int = 1
    video_time_base_den: int = 30

    def validate(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (self.sim_hz, self.control_hz, self.video_hz)):
            raise SchemaValidationError("Simulation, control, and video frequencies must be positive")
        if self.video_time_base_num <= 0 or self.video_time_base_den <= 0:
            raise SchemaValidationError("Video time base must be a positive rational")


@dataclass(slots=True, frozen=True)
class NamedFeature:
    """One named scalar or fixed-size vector feature."""

    name: str
    dtype: str
    unit: str
    shape: tuple[int, ...] = ()
    frame: str | None = None
    description: str = ""

    def validate(self) -> None:
        if not self.name or not self.dtype or not self.unit:
            raise SchemaValidationError("Named features require name, dtype, and unit")
        if any(size <= 0 for size in self.shape):
            raise SchemaValidationError(f"Invalid feature shape for {self.name}: {self.shape}")


@dataclass(slots=True, frozen=True)
class PhysicsValue:
    """One named, masked persistent physics parameter."""

    name: str
    value: Any
    unit: str
    valid: bool
    implemented: bool
    kind: PhysicsValueKind = PhysicsValueKind.PHYSICAL
    observable_in_prefix: bool | None = None
    source: str = "simulator_config"

    def validate(self) -> None:
        if not self.name or not self.unit:
            raise SchemaValidationError("Physics values require a name and unit")
        if self.valid and self.value is None:
            raise SchemaValidationError(f"Valid physics value {self.name!r} cannot be null")
        if self.kind == PhysicsValueKind.NOT_IMPLEMENTED and self.implemented:
            raise SchemaValidationError("not_implemented physics values must have implemented=false")
        if self.kind == PhysicsValueKind.UNKNOWN and self.valid:
            raise SchemaValidationError("unknown physics values must have valid=false")


@dataclass(slots=True)
class PhysicsMetadata:
    """Named physics fields plus simulator integration details."""

    parameters: dict[str, PhysicsValue] = field(default_factory=dict)
    gravity_world_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    gravity_valid: bool = False
    simulation_timestep_s: float | None = None
    substeps: int | None = None
    solver_settings: dict[str, Any] = field(default_factory=dict)
    external_impulses: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.gravity_world_m_s2) != 3 or not all(math.isfinite(v) for v in self.gravity_world_m_s2):
            raise SchemaValidationError("gravity_world_m_s2 must contain three finite values")
        if self.simulation_timestep_s is not None and self.simulation_timestep_s <= 0:
            raise SchemaValidationError("simulation_timestep_s must be positive")
        if self.substeps is not None and self.substeps < 1:
            raise SchemaValidationError("substeps must be positive")
        for key, value in self.parameters.items():
            value.validate()
            if key != value.name:
                raise SchemaValidationError(f"Physics parameter key/name mismatch: {key} != {value.name}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["parameters"] = {
            key: {**asdict(value), "kind": value.kind.value} for key, value in self.parameters.items()
        }
        return result


@dataclass(slots=True)
class DatasetInfo:
    """Dataset-level schema, timing, and feature declarations."""

    name: str
    dataset_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION
    description: str = ""
    coordinate_convention: CoordinateConvention = field(default_factory=CoordinateConvention)
    time_base: TimeBase | None = None
    state_features: list[NamedFeature] = field(default_factory=list)
    action_features: list[NamedFeature] = field(default_factory=list)
    camera_names: list[str] = field(
        default_factory=lambda: ["observation.images.main", "observation.images.secondary"]
    )
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    generator_version: str = "unknown"
    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(f"Unsupported schema version: {self.schema_version}")
        if not self.name:
            raise SchemaValidationError("Dataset name is required")
        try:
            uuid.UUID(self.dataset_uuid)
        except ValueError as exc:
            raise SchemaValidationError("dataset_uuid is not a UUID") from exc
        self.coordinate_convention.validate()
        if self.time_base is not None:
            self.time_base.validate()
        for feature in [*self.state_features, *self.action_features]:
            feature.validate()
        state_names = [value.name for value in self.state_features]
        action_names = [value.name for value in self.action_features]
        if len(state_names) != len(set(state_names)) or len(action_names) != len(set(action_names)):
            raise SchemaValidationError("State and action feature names must be unique")
        if not self.camera_names or len(self.camera_names) != len(set(self.camera_names)):
            raise SchemaValidationError("Camera names must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class EpisodeRecord:
    """Canonical episode-branch metadata.

    ``intended_branch`` and measured ``actual_outcome`` are separate required
    fields.  Physics fields do not contain initial velocity, robot state, or
    commands; those belong in frame/high-rate tables.
    """

    episode_uuid: str
    episode_index: int
    counterfactual_bundle_id: str
    scene_seed: int
    branch_seed: int
    family: str
    subfamily: str
    intended_branch: str
    actual_outcome: str
    task_success: bool
    failure_mode: str
    variant: str = "default"
    robot_model: str = ""
    tool_type: str = ""
    action_mode: str = "family_specific_named_command"
    partial_success_score: float | None = None
    label_confidence: float | None = None
    label_status: LabelStatus = LabelStatus.VERIFIED
    dynamics_mode: DynamicsMode = DynamicsMode.FREE_CONTACT
    release_tier: ReleaseTier = ReleaseTier.FREE_CONTACT
    physics_qc_pass: bool = True
    split: Split = Split.UNASSIGNED
    physics_counterfactual_family_id: str | None = None
    split_group_id: str | None = None
    parent_episode_uuid: str | None = None
    source_generator: str = ""
    source_generator_version: str = ""
    generator_git_commit: str = "unknown"
    config_hash: str = ""
    asset_ids: list[str] = field(default_factory=list)
    asset_hashes: dict[str, str] = field(default_factory=dict)
    simulator_name: str = ""
    simulator_version: str = ""
    renderer: str = ""
    creation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    video_paths: dict[str, str] = field(default_factory=dict)
    frame_data_path: str | None = None
    high_rate_path: str | None = None
    events_path: str | None = None
    object_states_path: str | None = None
    camera_ids: list[str] = field(default_factory=list)
    task_index: int | None = None
    frame_count: int | None = None
    duration_s: float | None = None
    event_time_s: float | None = None
    physics: PhysicsMetadata = field(default_factory=PhysicsMetadata)
    assistance: dict[str, Any] = field(
        default_factory=lambda: {
            "assisted_grasp": False,
            "assisted_retention": False,
            "equality_constraint_active": False,
            "latch_active": False,
            "constraint_activation_time": None,
            "constraint_deactivation_time": None,
        }
    )
    objective_metrics: dict[str, Any] = field(default_factory=dict)
    randomization: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    content_hashes: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def release_eligible(self) -> bool:
        """Whether this record may enter the default training manifest."""

        assistance_active = any(
            bool(self.assistance.get(name, False))
            for name in (
                "assisted_grasp",
                "assisted_retention",
                "equality_constraint_active",
                "latch_active",
            )
        )
        return (
            self.label_status == LabelStatus.VERIFIED
            and self.physics_qc_pass
            and self.release_tier == ReleaseTier.FREE_CONTACT
            and self.dynamics_mode == DynamicsMode.FREE_CONTACT
            and not assistance_active
            and not self.quality_flags
        )

    def validate(self) -> None:
        required = {
            "episode_uuid": self.episode_uuid,
            "counterfactual_bundle_id": self.counterfactual_bundle_id,
            "family": self.family,
            "subfamily": self.subfamily,
            "intended_branch": self.intended_branch,
            "actual_outcome": self.actual_outcome,
            "physics_counterfactual_family_id": self.physics_counterfactual_family_id,
            "split_group_id": self.split_group_id,
            "source_generator": self.source_generator,
            "source_generator_version": self.source_generator_version,
            "generator_git_commit": self.generator_git_commit,
            "config_hash": self.config_hash,
            "simulator_name": self.simulator_name,
            "simulator_version": self.simulator_version,
            "renderer": self.renderer,
            "action_mode": self.action_mode,
        }
        missing = [name for name, value in required.items() if value is None or str(value).strip() == ""]
        if missing:
            raise SchemaValidationError(f"Missing required episode fields: {', '.join(missing)}")
        try:
            parsed_episode_uuid = uuid.UUID(self.episode_uuid)
        except (AttributeError, TypeError, ValueError) as error:
            raise SchemaValidationError("episode_uuid must be a canonical UUID") from error
        if str(parsed_episode_uuid) != self.episode_uuid:
            raise SchemaValidationError("episode_uuid must use canonical lowercase UUID text")
        if self.episode_index < 0:
            raise SchemaValidationError("episode_index must be non-negative")
        if not isinstance(self.task_success, bool):
            raise SchemaValidationError("task_success must be a measured boolean")
        validate_failure_mode(self.task_success, self.failure_mode)
        if self.partial_success_score is not None and not 0.0 <= self.partial_success_score <= 1.0:
            raise SchemaValidationError("partial_success_score must be in [0, 1]")
        if self.label_confidence is not None and not 0.0 <= self.label_confidence <= 1.0:
            raise SchemaValidationError("label_confidence must be in [0, 1]")
        if self.label_status != LabelStatus.VERIFIED and self.release_tier == ReleaseTier.FREE_CONTACT:
            raise SchemaValidationError("Unverified/proxy labels cannot use the free-contact release tier")
        if self.dynamics_mode == DynamicsMode.ASSISTED_CONTACT and self.release_tier == ReleaseTier.FREE_CONTACT:
            raise SchemaValidationError("Assisted dynamics cannot use the free-contact release tier")
        if self.dynamics_mode == DynamicsMode.SCRIPTED_MOTION and self.release_tier != ReleaseTier.SCRIPTED_MOTION:
            raise SchemaValidationError("Scripted motion must use the scripted_motion release tier")
        assistance_fields = (
            "assisted_grasp",
            "assisted_retention",
            "equality_constraint_active",
            "latch_active",
        )
        missing_assistance = [name for name in assistance_fields if name not in self.assistance]
        missing_assistance += [
            name
            for name in ("constraint_activation_time", "constraint_deactivation_time")
            if name not in self.assistance
        ]
        if missing_assistance:
            raise SchemaValidationError(
                f"Missing assistance fields: {', '.join(missing_assistance)}"
            )
        if any(not isinstance(self.assistance[name], bool) for name in assistance_fields):
            raise SchemaValidationError("Assistance flags must be boolean")
        assistance_active = any(self.assistance[name] for name in assistance_fields)
        if self.dynamics_mode == DynamicsMode.FREE_CONTACT and assistance_active:
            raise SchemaValidationError("Free-contact episodes cannot declare active assistance")
        if self.dynamics_mode == DynamicsMode.ASSISTED_CONTACT and not assistance_active:
            raise SchemaValidationError("Assisted-contact episodes require a named assistance mechanism")
        for name in ("constraint_activation_time", "constraint_deactivation_time"):
            value = self.assistance[name]
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise SchemaValidationError(f"{name} must be null or a finite non-negative time")
        activation = self.assistance["constraint_activation_time"]
        deactivation = self.assistance["constraint_deactivation_time"]
        if assistance_active and activation is None:
            raise SchemaValidationError("Active assistance requires constraint_activation_time")
        if activation is not None and deactivation is not None and deactivation < activation:
            raise SchemaValidationError("constraint_deactivation_time precedes activation")
        for relative in [
            *self.video_paths.values(),
            self.frame_data_path,
            self.high_rate_path,
            self.events_path,
            self.object_states_path,
        ]:
            if relative is not None:
                portable_relative_path(relative)
        if self.frame_count is not None and self.frame_count <= 0:
            raise SchemaValidationError("frame_count must be positive")
        if self.duration_s is not None and (not math.isfinite(self.duration_s) or self.duration_s <= 0):
            raise SchemaValidationError("duration_s must be finite and positive")
        if self.event_time_s is not None:
            if not math.isfinite(self.event_time_s) or self.event_time_s < 0:
                raise SchemaValidationError("event_time_s must be finite and non-negative")
            if self.duration_s is not None and self.event_time_s > self.duration_s:
                raise SchemaValidationError("event_time_s lies outside the episode duration")
        self.physics.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value.update(
            label_status=self.label_status.value,
            dynamics_mode=self.dynamics_mode.value,
            release_tier=self.release_tier.value,
            split=self.split.value,
            physics=self.physics.to_dict(),
            release_eligible=self.release_eligible,
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeRecord":
        """Construct a record from its Arrow/JSON representation."""

        data = dict(value)
        data.pop("release_eligible", None)
        raw_label_status = data.get("label_status", LabelStatus.VERIFIED)
        if raw_label_status == "verified":  # import compatibility; never emitted
            raw_label_status = LabelStatus.VERIFIED
        data["label_status"] = LabelStatus(raw_label_status)
        data["dynamics_mode"] = DynamicsMode(data.get("dynamics_mode", DynamicsMode.FREE_CONTACT))
        data["release_tier"] = ReleaseTier(data.get("release_tier", ReleaseTier.FREE_CONTACT))
        data["split"] = Split(data.get("split", Split.UNASSIGNED))
        raw_physics = data.get("physics") or {}
        parameters = {
            key: PhysicsValue(
                **{
                    **item,
                    "kind": PhysicsValueKind(item.get("kind", PhysicsValueKind.PHYSICAL)),
                }
            )
            for key, item in (raw_physics.get("parameters") or {}).items()
        }
        data["physics"] = PhysicsMetadata(
            parameters=parameters,
            gravity_world_m_s2=tuple(raw_physics.get("gravity_world_m_s2", (0.0, 0.0, -9.81))),
            gravity_valid=bool(raw_physics.get("gravity_valid", False)),
            simulation_timestep_s=raw_physics.get("simulation_timestep_s"),
            substeps=raw_physics.get("substeps"),
            solver_settings=dict(raw_physics.get("solver_settings") or {}),
            external_impulses=list(raw_physics.get("external_impulses") or []),
        )
        record = cls(**data)
        record.validate()
        return record


def validate_episode_records(records: Iterable[EpisodeRecord]) -> None:
    """Validate records jointly for global identity invariants."""

    uuids: set[str] = set()
    indices: set[int] = set()
    for record in records:
        record.validate()
        if record.episode_uuid in uuids:
            raise SchemaValidationError(f"Duplicate episode_uuid: {record.episode_uuid}")
        if record.episode_index in indices:
            raise SchemaValidationError(f"Duplicate episode_index: {record.episode_index}")
        uuids.add(record.episode_uuid)
        indices.add(record.episode_index)


def resolved_config_hash(config: Mapping[str, Any]) -> str:
    """Return the canonical hash stored by generators and resume guards."""

    return sha256_json(config)


def __getattr__(name: str) -> Any:
    """Lazily expose OutcomeResult without introducing a schema/outcomes cycle."""

    if name == "OutcomeResult":
        from .outcomes import OutcomeResult

        return OutcomeResult
    raise AttributeError(name)
