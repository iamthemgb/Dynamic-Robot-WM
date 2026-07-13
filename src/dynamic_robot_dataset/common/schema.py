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

LEGACY_SCHEMA_VERSION = "dynamic-robot-dataset/v1"
SCHEMA_VERSION = "dynamic-robot-dataset/v2"
SUPPORTED_READ_SCHEMA_VERSIONS = frozenset({LEGACY_SCHEMA_VERSION, SCHEMA_VERSION})
FAILURE_TAXONOMY_VERSION = "dynamic-robot-failure-codes/v2"
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


class ActualOutcomeClass(str, Enum):
    """Closed, task-independent measured outcome classes for v2 records."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NEAR_MISS = "near_miss"
    CONTACT_FAILURE = "contact_failure"
    MISS = "miss"
    NO_OP = "no_op"
    WRONG_ACTION = "wrong_action"
    INVALID = "invalid"
    UNVERIFIED = "unverified"


class ParameterRangePartition(str, Enum):
    """Versioned parameter-range partition used to prevent OOD leakage."""

    NOMINAL = "nominal"
    TRAIN_ID = "train_id"
    VALIDATION_ID = "validation_id"
    TEST_OOD = "test_ood"
    UNCALIBRATED = "uncalibrated"


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
FAILURE_TAGS = frozenset(
    {
        "outcome_failure",
        "near_miss",
        "contact",
        "no_op",
        "wrong_action",
        "invalid_rollout",
        "unverified_label",
    }
)


_CONTACT_FAILURE_CODES = frozenset(
    {
        "contact_without_completion",
        "slip",
        "unstable_retention",
        "object_not_retained",
        "object_not_retained_in_goal",
        "incorrect_rebound_direction",
        "unintended_snag",
    }
)
_INVALID_FAILURE_CODES = frozenset(
    {
        "excessive_penetration",
        "unstable_physics",
        "cropped_event",
        "truncated_outcome",
    }
)


def infer_actual_outcome_class(
    *,
    task_success: bool,
    actual_outcome: str,
    failure_code: str,
    label_status: LabelStatus = LabelStatus.VERIFIED,
    partial_success_score: float | None = None,
) -> ActualOutcomeClass:
    """Map legacy task-specific outcome strings into the closed v2 classes.

    This function exists for v1 import and for old family adapters. New native
    generators should provide ``actual_outcome_class`` explicitly.
    """

    outcome = actual_outcome.strip().lower().replace("-", "_").replace(" ", "_")
    code = failure_code.strip().lower()
    if task_success:
        return ActualOutcomeClass.SUCCESS
    if label_status != LabelStatus.VERIFIED or outcome in {"unverified", "quarantined"}:
        return ActualOutcomeClass.UNVERIFIED
    if code in {"no_op", "controller_no_op"} or outcome in {"no_op", "noop"}:
        return ActualOutcomeClass.NO_OP
    if code == "wrong_action" or outcome in {"wrong_action", "bad_action"}:
        return ActualOutcomeClass.WRONG_ACTION
    if "near_miss" in outcome or "near_miss" in code:
        return ActualOutcomeClass.NEAR_MISS
    if outcome == "contact_failure" or code in _CONTACT_FAILURE_CODES:
        return ActualOutcomeClass.CONTACT_FAILURE
    if code in _INVALID_FAILURE_CODES or outcome in {"invalid", "unstable_physics"}:
        return ActualOutcomeClass.INVALID
    if outcome == "partial_success":
        return ActualOutcomeClass.PARTIAL_SUCCESS
    return ActualOutcomeClass.MISS


def default_failure_tags(failure_code: str) -> list[str]:
    """Return stable coarse tags while retaining the concrete primary code."""

    code = failure_code.strip().lower()
    if code == "none":
        return []
    tags: set[str] = {"outcome_failure"}
    if "near_miss" in code:
        tags.add("near_miss")
    if code in _CONTACT_FAILURE_CODES or any(
        token in code for token in ("contact", "retained", "rebound", "grasp", "snag")
    ):
        tags.add("contact")
    if code in {"no_op", "controller_no_op"}:
        tags.add("no_op")
    if code == "wrong_action":
        tags.add("wrong_action")
    if code in _INVALID_FAILURE_CODES:
        tags.add("invalid_rollout")
    if code == "label_unverified" or code == "legacy_proxy_quarantined":
        tags.add("unverified_label")
    return sorted(tags)


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


def validate_outcome_contract(
    *,
    task_success: bool,
    outcome_class: ActualOutcomeClass,
    primary_failure_code: str,
    compatibility_failure_mode: str,
    partial_success_score: float | None,
) -> None:
    """Validate the v2 outcome class and failure-taxonomy cross-product."""

    validate_failure_mode(task_success, compatibility_failure_mode)
    code = primary_failure_code.strip().lower()
    if code != compatibility_failure_mode.strip().lower():
        raise SchemaValidationError(
            "primary_failure_code must equal compatibility failure_mode"
        )
    validate_failure_mode(task_success, code)
    if task_success and outcome_class != ActualOutcomeClass.SUCCESS:
        raise SchemaValidationError("task_success=true requires actual_outcome_class=success")
    if not task_success and outcome_class == ActualOutcomeClass.SUCCESS:
        raise SchemaValidationError("actual_outcome_class=success requires task_success=true")
    if outcome_class == ActualOutcomeClass.PARTIAL_SUCCESS:
        if partial_success_score is None or not 0.0 < partial_success_score < 1.0:
            raise SchemaValidationError(
                "partial_success requires partial_success_score strictly between 0 and 1"
            )
    if outcome_class == ActualOutcomeClass.NO_OP and code not in {"no_op", "controller_no_op"}:
        raise SchemaValidationError("no_op outcome requires a no-op failure code")
    if outcome_class == ActualOutcomeClass.WRONG_ACTION and code != "wrong_action":
        raise SchemaValidationError("wrong_action outcome requires failure code wrong_action")
    if outcome_class == ActualOutcomeClass.UNVERIFIED and code not in {
        "label_unverified",
        "legacy_proxy_quarantined",
    }:
        raise SchemaValidationError("unverified outcome requires an unverified-label failure code")


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
    parameter_range_provenance: dict[str, Any] = field(
        default_factory=lambda: {
            "profile_id": "legacy_unversioned",
            "profile_version": "unversioned",
            "partition": ParameterRangePartition.UNCALIBRATED.value,
            "source": "legacy_or_unspecified",
            "calibrated": False,
            "calibration_verified": False,
            "calibration_artifact_hash": None,
        }
    )

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
        provenance = self.parameter_range_provenance
        required = ("profile_id", "profile_version", "partition", "source", "calibrated")
        missing = [name for name in required if name not in provenance]
        if missing:
            raise SchemaValidationError(
                f"parameter_range_provenance missing fields: {', '.join(missing)}"
            )
        if any(not str(provenance[name]).strip() for name in required[:4]):
            raise SchemaValidationError("Parameter-range identifiers and source must be non-empty")
        try:
            ParameterRangePartition(str(provenance["partition"]))
        except ValueError as error:
            raise SchemaValidationError(
                f"Unknown parameter-range partition: {provenance['partition']}"
            ) from error
        if not isinstance(provenance["calibrated"], bool):
            raise SchemaValidationError("parameter-range calibrated must be boolean")
        if not isinstance(provenance.get("calibration_verified", False), bool):
            raise SchemaValidationError("parameter-range calibration_verified must be boolean")
        artifact_hash = provenance.get("calibration_artifact_hash")
        if provenance["calibrated"] and (
            provenance.get("calibration_verified") is not True
            or not isinstance(artifact_hash, str)
            or len(artifact_hash) != 64
            or any(character not in "0123456789abcdef" for character in artifact_hash)
        ):
            raise SchemaValidationError(
                "Calibrated parameter ranges require a lowercase SHA-256 artifact hash"
            )

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
    camera_roles: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.main": "main_three_quarter_external",
            "observation.images.secondary": "task_specific_secondary",
        }
    )
    frame_semantic_fields: list[str] = field(
        default_factory=lambda: [
            "task_phase",
            "motion_mode",
            "active_surface",
            "contact_role",
        ]
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
        if set(self.camera_roles) != set(self.camera_names):
            raise SchemaValidationError("camera_roles must map every declared camera stream exactly once")
        if any(not str(role).strip() for role in self.camera_roles.values()):
            raise SchemaValidationError("Camera roles must be non-empty")
        if len(self.frame_semantic_fields) != len(set(self.frame_semantic_fields)):
            raise SchemaValidationError("Frame semantic field names must be unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetInfo":
        """Read v2 or upgrade a v1 dataset-info record in memory."""

        data = dict(value)
        source_version = str(data.get("schema_version", LEGACY_SCHEMA_VERSION))
        if source_version not in SUPPORTED_READ_SCHEMA_VERSIONS:
            raise SchemaValidationError(f"Unsupported schema version: {source_version}")
        convention = data.get("coordinate_convention")
        if isinstance(convention, Mapping):
            data["coordinate_convention"] = CoordinateConvention(**dict(convention))
        time_base = data.get("time_base")
        if isinstance(time_base, Mapping):
            data["time_base"] = TimeBase(**dict(time_base))
        for field_name in ("state_features", "action_features"):
            data[field_name] = [
                item if isinstance(item, NamedFeature) else NamedFeature(**dict(item))
                for item in data.get(field_name, ())
            ]
        if source_version == LEGACY_SCHEMA_VERSION:
            extras = dict(data.get("extras") or {})
            extras.setdefault("source_schema_version", LEGACY_SCHEMA_VERSION)
            data["extras"] = extras
            data["schema_version"] = SCHEMA_VERSION
        camera_names = list(
            data.get("camera_names")
            or ("observation.images.main", "observation.images.secondary")
        )
        data.setdefault("camera_names", camera_names)
        data.setdefault(
            "camera_roles",
            {
                name: (
                    "main_three_quarter_external"
                    if name.endswith(".main")
                    else "task_specific_secondary"
                )
                for name in camera_names
            },
        )
        record = cls(**data)
        record.validate()
        return record


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
    schema_version: str = SCHEMA_VERSION
    actual_outcome_class: ActualOutcomeClass | None = None
    primary_failure_code: str | None = None
    failure_tags: list[str] = field(default_factory=list)
    failure_taxonomy_version: str = FAILURE_TAXONOMY_VERSION
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
    transition_events_path: str | None = None
    object_states_path: str | None = None
    camera_ids: list[str] = field(default_factory=list)
    task_index: int | None = None
    frame_count: int | None = None
    duration_s: float | None = None
    event_time_s: float | None = None
    key_event_name: str | None = None
    key_event_time_s: float | None = None
    objective_evaluator_id: str = "legacy_embedded"
    objective_evaluator_version: str = "unversioned"
    objective_threshold_set_hash: str = field(default_factory=lambda: sha256_json({}))
    objective_evidence: dict[str, Any] = field(default_factory=dict)
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
    camera_stream_calibration_ids: dict[str, str] = field(default_factory=dict)
    controller_profile: dict[str, Any] = field(
        default_factory=lambda: {
            "profile_id": "legacy_unspecified",
            "profile_version": "unversioned",
            "control_latency_s": None,
            "camera_latency_s": None,
        }
    )
    robot_start_provenance: dict[str, Any] = field(default_factory=dict)
    tool_calibration_provenance: dict[str, Any] = field(default_factory=dict)
    randomization: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    content_hashes: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate v2 compatibility fields for legacy family adapters.

        Native production writers should provide these values explicitly. The
        compatibility defaults are intentionally marked legacy/uncalibrated so
        they cannot silently enter the default release manifest.
        """

        if not isinstance(self.label_status, LabelStatus):
            self.label_status = LabelStatus(self.label_status)
        if not isinstance(self.dynamics_mode, DynamicsMode):
            self.dynamics_mode = DynamicsMode(self.dynamics_mode)
        if not isinstance(self.release_tier, ReleaseTier):
            self.release_tier = ReleaseTier(self.release_tier)
        if not isinstance(self.split, Split):
            self.split = Split(self.split)
        if self.actual_outcome_class is None:
            self.actual_outcome_class = infer_actual_outcome_class(
                task_success=self.task_success,
                actual_outcome=self.actual_outcome,
                failure_code=self.primary_failure_code or self.failure_mode,
                label_status=self.label_status,
                partial_success_score=self.partial_success_score,
            )
        elif not isinstance(self.actual_outcome_class, ActualOutcomeClass):
            self.actual_outcome_class = ActualOutcomeClass(self.actual_outcome_class)
        if self.primary_failure_code is None:
            self.primary_failure_code = self.failure_mode.strip().lower()
        if not self.failure_tags and self.primary_failure_code != "none":
            self.failure_tags = default_failure_tags(self.primary_failure_code)
        if self.key_event_time_s is None and self.event_time_s is not None:
            self.key_event_time_s = self.event_time_s
        if self.event_time_s is None and self.key_event_time_s is not None:
            self.event_time_s = self.key_event_time_s
        if self.key_event_name is None and self.key_event_time_s is not None:
            self.key_event_name = "task_interaction"
        if not self.objective_evidence:
            measured = next(
                (
                    self.objective_metrics[key]
                    for key in ("objective_success", "task_success", "success")
                    if isinstance(self.objective_metrics.get(key), bool)
                ),
                None,
            )
            self.objective_evidence = {
                "stored_objective_success": measured,
                "independently_recomputed": False,
                "source": "legacy_embedded_metrics",
            }

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
            self.schema_version == SCHEMA_VERSION
            and self.label_status == LabelStatus.VERIFIED
            and self.physics_qc_pass
            and self.release_tier == ReleaseTier.FREE_CONTACT
            and self.dynamics_mode == DynamicsMode.FREE_CONTACT
            and not assistance_active
            and self.objective_evaluator_id != "legacy_embedded"
            and self.objective_evaluator_version != "unversioned"
            and self.objective_evidence.get("independently_recomputed") is True
            and self.physics.parameter_range_provenance.get("calibrated") is True
            and self.physics.parameter_range_provenance.get("calibration_verified") is True
            and not self.quality_flags
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(f"Episode writes require {SCHEMA_VERSION}")
        required = {
            "episode_uuid": self.episode_uuid,
            "counterfactual_bundle_id": self.counterfactual_bundle_id,
            "family": self.family,
            "subfamily": self.subfamily,
            "intended_branch": self.intended_branch,
            "actual_outcome": self.actual_outcome,
            "split_group_id": self.split_group_id,
            "source_generator": self.source_generator,
            "source_generator_version": self.source_generator_version,
            "generator_git_commit": self.generator_git_commit,
            "config_hash": self.config_hash,
            "simulator_name": self.simulator_name,
            "simulator_version": self.simulator_version,
            "renderer": self.renderer,
            "action_mode": self.action_mode,
            "objective_evaluator_id": self.objective_evaluator_id,
            "objective_evaluator_version": self.objective_evaluator_version,
            "objective_threshold_set_hash": self.objective_threshold_set_hash,
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
        assert self.actual_outcome_class is not None
        assert self.primary_failure_code is not None
        validate_outcome_contract(
            task_success=self.task_success,
            outcome_class=self.actual_outcome_class,
            primary_failure_code=self.primary_failure_code,
            compatibility_failure_mode=self.failure_mode,
            partial_success_score=self.partial_success_score,
        )
        inferred_outcome_class = infer_actual_outcome_class(
            task_success=self.task_success,
            actual_outcome=self.actual_outcome,
            failure_code=self.primary_failure_code,
            label_status=self.label_status,
            partial_success_score=self.partial_success_score,
        )
        if inferred_outcome_class != self.actual_outcome_class:
            raise SchemaValidationError(
                "compatibility actual_outcome disagrees with actual_outcome_class"
            )
        if self.failure_taxonomy_version != FAILURE_TAXONOMY_VERSION:
            raise SchemaValidationError(
                f"failure_taxonomy_version must be {FAILURE_TAXONOMY_VERSION}"
            )
        if len(self.failure_tags) != len(set(self.failure_tags)) or any(
            not isinstance(tag, str) or not tag.strip() for tag in self.failure_tags
        ):
            raise SchemaValidationError("failure_tags must contain unique non-empty strings")
        unknown_failure_tags = sorted(set(self.failure_tags) - FAILURE_TAGS)
        if unknown_failure_tags:
            raise SchemaValidationError(f"Unknown failure_tags: {unknown_failure_tags}")
        if self.task_success and self.failure_tags:
            raise SchemaValidationError("Successful episodes cannot declare failure_tags")
        if self.partial_success_score is not None and not 0.0 <= self.partial_success_score <= 1.0:
            raise SchemaValidationError("partial_success_score must be in [0, 1]")
        if self.label_confidence is not None and not 0.0 <= self.label_confidence <= 1.0:
            raise SchemaValidationError("label_confidence must be in [0, 1]")
        if (
            len(self.objective_threshold_set_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.objective_threshold_set_hash)
        ):
            raise SchemaValidationError("objective_threshold_set_hash must be a lowercase SHA-256")
        if not isinstance(self.objective_evidence, dict):
            raise SchemaValidationError("objective_evidence must be a mapping")
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
        self._validate_assistance_mechanisms()
        if self.video_paths and not self.camera_stream_calibration_ids:
            self.camera_stream_calibration_ids = {
                stream: stream for stream in self.video_paths
            }
        if self.camera_stream_calibration_ids:
            if set(self.camera_stream_calibration_ids) != set(self.video_paths):
                raise SchemaValidationError(
                    "camera_stream_calibration_ids must map every video stream exactly once"
                )
            if any(
                not str(identifier).strip()
                for identifier in self.camera_stream_calibration_ids.values()
            ):
                raise SchemaValidationError("Camera calibration IDs must be non-empty")
            if self.camera_ids and set(self.camera_ids) != set(
                self.camera_stream_calibration_ids.values()
            ):
                raise SchemaValidationError(
                    "camera_ids must contain the referenced calibration IDs"
                )
        self._validate_execution_provenance()
        for relative in [
            *self.video_paths.values(),
            self.frame_data_path,
            self.high_rate_path,
            self.events_path,
            self.transition_events_path,
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
        if self.key_event_time_s != self.event_time_s:
            raise SchemaValidationError(
                "key_event_time_s and compatibility event_time_s must agree"
            )
        if self.key_event_time_s is not None and not str(self.key_event_name or "").strip():
            raise SchemaValidationError("A timed key event requires key_event_name")
        if (
            self.duration_s is not None
            and self.objective_evaluator_id != "legacy_embedded"
            and (
                self.key_event_time_s is None
                or not str(self.key_event_name or "").strip()
            )
        ):
            raise SchemaValidationError(
                "Objective-labelled v2 episodes require a named, timed key event"
            )
        self.physics.validate()

    def _validate_execution_provenance(self) -> None:
        profile = self.controller_profile
        for name in ("profile_id", "profile_version", "control_latency_s", "camera_latency_s"):
            if name not in profile:
                raise SchemaValidationError(f"controller_profile missing {name}")
        if not str(profile["profile_id"]).strip() or not str(profile["profile_version"]).strip():
            raise SchemaValidationError("Controller profile ID/version must be non-empty")
        for name in ("control_latency_s", "camera_latency_s"):
            value = profile[name]
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise SchemaValidationError(f"{name} must be null or finite and non-negative")
        for name, provenance in (
            ("robot_start_provenance", self.robot_start_provenance),
            ("tool_calibration_provenance", self.tool_calibration_provenance),
        ):
            if not isinstance(provenance, dict):
                raise SchemaValidationError(f"{name} must be a mapping")
        tool = self.tool_calibration_provenance
        if tool.get("calibrated") is True:
            declared = tool.get("artifact_sha256")
            actual = tool.get("actual_artifact_sha256")
            if (
                not isinstance(tool.get("artifact_path"), str)
                or not str(tool.get("artifact_path")).strip()
                or not isinstance(declared, str)
                or len(declared) != 64
                or any(character not in "0123456789abcdef" for character in declared)
                or actual != declared
            ):
                raise SchemaValidationError(
                    "Calibrated tools require a resolved, content-verified artifact"
                )

    def _validate_assistance_mechanisms(self) -> None:
        mechanisms = self.assistance.get("mechanisms", [])
        if not isinstance(mechanisms, list):
            raise SchemaValidationError("assistance.mechanisms must be a list")
        assistance_active = any(
            self.assistance[name]
            for name in (
                "assisted_grasp",
                "assisted_retention",
                "equality_constraint_active",
                "latch_active",
            )
        )
        if assistance_active and not mechanisms:
            raise SchemaValidationError(
                "Active assistance requires simulator-observed mechanism records"
            )
        seen: set[str] = set()
        all_intervals: list[tuple[float, float | None]] = []
        for mechanism in mechanisms:
            if not isinstance(mechanism, Mapping):
                raise SchemaValidationError("Each assistance mechanism must be a mapping")
            identifier = str(mechanism.get("mechanism_id", "")).strip()
            mechanism_type = str(mechanism.get("mechanism_type", "")).strip()
            source = str(mechanism.get("source", "")).strip()
            if not identifier or identifier in seen:
                raise SchemaValidationError("Assistance mechanism IDs must be non-empty and unique")
            seen.add(identifier)
            if mechanism_type not in {
                "assisted_grasp",
                "assisted_retention",
                "equality_constraint",
                "latch",
                "scripted_state",
            }:
                raise SchemaValidationError(f"Unknown assistance mechanism type: {mechanism_type}")
            if source != "simulator_observed":
                raise SchemaValidationError(
                    "Assistance mechanisms must use source=simulator_observed"
                )
            intervals = mechanism.get("activation_intervals")
            if not isinstance(intervals, list) or not intervals:
                raise SchemaValidationError(
                    "Assistance mechanisms require non-empty activation_intervals"
                )
            previous_end = -math.inf
            for interval in intervals:
                if not isinstance(interval, Mapping):
                    raise SchemaValidationError("Assistance intervals must be mappings")
                start = interval.get("start_time_s")
                end = interval.get("end_time_s")
                if not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0:
                    raise SchemaValidationError("Assistance interval start must be finite and non-negative")
                if end is not None and (
                    not isinstance(end, (int, float)) or not math.isfinite(end) or end < start
                ):
                    raise SchemaValidationError("Assistance interval end precedes its start")
                if start < previous_end:
                    raise SchemaValidationError("Assistance intervals overlap or are unsorted")
                previous_end = math.inf if end is None else float(end)
                all_intervals.append((float(start), None if end is None else float(end)))
            if mechanism_type in {"equality_constraint", "latch"} and not mechanism.get(
                "constraint_ids"
            ):
                raise SchemaValidationError(
                    f"{mechanism_type} assistance requires constraint_ids"
                )
            if not mechanism.get("target_body_ids") and not mechanism.get("target_element_ids"):
                raise SchemaValidationError(
                    "Assistance mechanisms require target body or element IDs"
                )
        if mechanisms and not any(
            self.assistance[name]
            for name in (
                "assisted_grasp",
                "assisted_retention",
                "equality_constraint_active",
                "latch_active",
            )
        ) and self.dynamics_mode != DynamicsMode.SCRIPTED_MOTION:
            raise SchemaValidationError("Mechanism records require an active assistance summary")
        if all_intervals:
            first = min(start for start, _ in all_intervals)
            finite_ends = [end for _, end in all_intervals if end is not None]
            last = max(finite_ends) if len(finite_ends) == len(all_intervals) else None
            if self.assistance["constraint_activation_time"] is None or abs(
                float(self.assistance["constraint_activation_time"]) - first
            ) > 1e-9:
                raise SchemaValidationError(
                    "Assistance summary activation time disagrees with mechanism intervals"
                )
            if last is not None and (
                self.assistance["constraint_deactivation_time"] is None
                or abs(float(self.assistance["constraint_deactivation_time"]) - last) > 1e-9
            ):
                raise SchemaValidationError(
                    "Assistance summary deactivation time disagrees with mechanism intervals"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value.update(
            schema_version=SCHEMA_VERSION,
            actual_outcome_class=self.actual_outcome_class.value,
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
        """Construct a v2 record, upgrading v1 metadata without guessing physics."""

        data = dict(value)
        data.pop("release_eligible", None)
        source_schema_version = str(data.get("schema_version", LEGACY_SCHEMA_VERSION))
        if source_schema_version not in SUPPORTED_READ_SCHEMA_VERSIONS:
            raise SchemaValidationError(f"Unsupported schema version: {source_schema_version}")
        if source_schema_version == LEGACY_SCHEMA_VERSION:
            extras = dict(data.get("extras") or {})
            extras.setdefault("source_schema_version", LEGACY_SCHEMA_VERSION)
            data["extras"] = extras
        data["schema_version"] = SCHEMA_VERSION
        raw_label_status = data.get("label_status", LabelStatus.VERIFIED)
        if raw_label_status == "verified":  # import compatibility; never emitted
            raw_label_status = LabelStatus.VERIFIED
        data["label_status"] = LabelStatus(raw_label_status)
        data["dynamics_mode"] = DynamicsMode(data.get("dynamics_mode", DynamicsMode.FREE_CONTACT))
        data["release_tier"] = ReleaseTier(data.get("release_tier", ReleaseTier.FREE_CONTACT))
        data["split"] = Split(data.get("split", Split.UNASSIGNED))
        raw_outcome_class = data.get("actual_outcome_class")
        data["actual_outcome_class"] = (
            None if raw_outcome_class is None else ActualOutcomeClass(raw_outcome_class)
        )
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
            parameter_range_provenance=dict(
                raw_physics.get("parameter_range_provenance")
                or {
                    "profile_id": "legacy_unversioned",
                    "profile_version": "unversioned",
                    "partition": ParameterRangePartition.UNCALIBRATED.value,
                    "source": "legacy_or_unspecified",
                    "calibrated": False,
                    "calibration_verified": False,
                    "calibration_artifact_hash": None,
                }
            ),
        )
        data.setdefault("primary_failure_code", data.get("failure_mode"))
        data.setdefault("failure_taxonomy_version", FAILURE_TAXONOMY_VERSION)
        data.setdefault("failure_tags", [])
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
