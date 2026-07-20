"""Dataset, video, label, split, and rigid-body physics quality checks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..backends.actuator_only import validate_action_rows
from .assets import validate_robocasa_asset_manifest
from .contacts import ContactEvent, normalize_contact_row, select_task_event_time
from .contract_v2 import (
    DEFAULT_OBJECTIVE_EVALUATORS,
    CounterfactualFamilyRecord,
    ObjectiveEvaluatorRegistry,
    ObjectiveRecomputeInput,
    compare_recomputed_objective,
    derived_action_hash_from_rows,
    derived_initial_state_hash_from_rows,
    validate_assistance_observations,
    validate_counterfactual_family_records,
    validate_v2_frame_semantics,
)
from .cameras import CameraCalibration
from .episode_writer import load_episode_records, read_parquet_rows, write_parquet_atomic
from .hashing import hamming_distance_hex, sha256_file, sha256_json
from .embodiments import PRODUCTION_END_EFFECTORS
from .paths import atomic_write_bytes, atomic_write_json, ensure_not_source_path, resolve_dataset_path
from .physics_contract import (
    STRICT_RIGID_THRESHOLDS,
    rigid_task_evidence_failures,
    strict_contact_penetration_check,
    strict_persisted_physics_failures,
    strict_runtime_audit_failures,
)
from .randomization import validate_randomization_admission
from .schema import DynamicsMode, EpisodeRecord, ReleaseTier, SchemaValidationError
from .splits import validate_no_split_leakage
from .synchronization import (
    SynchronizationError,
    validate_monotonic_timestamps,
    validate_persisted_render_schedule,
    validate_synchronized_streams,
)
from .video_writer import VideoProbe, VideoSpec, iter_rgb_frames, probe_frame_timestamps, probe_video, validate_video_probe
from .visual_qc import (
    NATIVE_VISUAL_QC_SCHEMA,
    NATIVE_VISUAL_THRESHOLDS,
    SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA,
    SOURCE_MUJOCO_VISIBILITY_MEDIA_BINDING_SCHEMA,
    SOURCE_MUJOCO_TASK_VISIBILITY_CONTRACT_SCHEMA,
    SOURCE_MUJOCO_VISUAL_THRESHOLDS,
    source_mujoco_task_visibility_requirement,
    source_mujoco_visibility_media_binding,
)


@dataclass(slots=True, frozen=True)
class PhysicsCheck:
    """Result of one numerical physics-consistency check."""

    name: str
    passed: bool
    metrics: dict[str, float]
    message: str = ""


def _vectors(values: Sequence[Sequence[float]], width: int, name: str) -> list[tuple[float, ...]]:
    result = [tuple(float(component) for component in row) for row in values]
    if any(len(row) != width or any(not math.isfinite(value) for value in row) for row in result):
        raise ValueError(f"{name} must contain finite vectors of length {width}")
    return result


def finite_difference_velocity_check(
    timestamps: Sequence[float],
    positions: Sequence[Sequence[float]],
    velocities: Sequence[Sequence[float]],
    *,
    absolute_tolerance_m_s: float = 0.08,
    relative_tolerance: float = 0.15,
    valid_interval_mask: Sequence[bool] | None = None,
) -> PhysicsCheck:
    """Compare saved velocity against position finite differences."""

    validate_monotonic_timestamps(timestamps)
    position = _vectors(positions, 3, "positions")
    velocity = _vectors(velocities, 3, "velocities")
    if len(timestamps) != len(position) or len(position) != len(velocity) or len(position) < 2:
        raise ValueError("Position, velocity, and timestamps need equal lengths of at least two")
    interval_mask = list(valid_interval_mask) if valid_interval_mask is not None else [True] * (len(position) - 1)
    if len(interval_mask) != len(position) - 1:
        raise ValueError("valid_interval_mask must have one value per adjacent sample interval")
    errors: list[float] = []
    references: list[float] = []
    for index in range(len(position) - 1):
        if not interval_mask[index]:
            continue
        dt = timestamps[index + 1] - timestamps[index]
        difference = tuple((position[index + 1][axis] - position[index][axis]) / dt for axis in range(3))
        reference = tuple((velocity[index + 1][axis] + velocity[index][axis]) / 2 for axis in range(3))
        errors.append(math.sqrt(sum((left - right) ** 2 for left, right in zip(difference, reference))))
        references.append(math.sqrt(sum(value * value for value in reference)))
    if not errors:
        return PhysicsCheck(
            "finite_difference_velocity",
            False,
            {"sample_count": 0.0},
            "No contact-free finite-difference intervals",
        )
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    reference_rms = math.sqrt(sum(value * value for value in references) / len(references))
    tolerance = absolute_tolerance_m_s + relative_tolerance * reference_rms
    return PhysicsCheck(
        "finite_difference_velocity",
        rmse <= tolerance,
        {
            "velocity_rmse_m_s": rmse,
            "reference_rms_m_s": reference_rms,
            "tolerance_m_s": tolerance,
            "sample_count": float(len(errors)),
        },
        "" if rmse <= tolerance else "Saved velocity disagrees with position finite differences",
    )


def gravity_consistency_check(
    timestamps: Sequence[float],
    velocities: Sequence[Sequence[float]],
    gravity_world_m_s2: Sequence[float],
    *,
    free_fall_mask: Sequence[bool] | None = None,
    tolerance_m_s2: float = 1.0,
) -> PhysicsCheck:
    """Compare acceleration on contact-free intervals with configured gravity."""

    validate_monotonic_timestamps(timestamps)
    velocity = _vectors(velocities, 3, "velocities")
    gravity = _vectors([gravity_world_m_s2], 3, "gravity")[0]
    if len(timestamps) != len(velocity) or len(velocity) < 2:
        raise ValueError("Velocity and timestamps need equal lengths of at least two")
    mask = list(free_fall_mask) if free_fall_mask is not None else [True] * len(velocity)
    if len(mask) != len(velocity):
        raise ValueError("free_fall_mask length differs from velocity")
    errors: list[float] = []
    for index in range(len(velocity) - 1):
        if not (mask[index] and mask[index + 1]):
            continue
        dt = timestamps[index + 1] - timestamps[index]
        acceleration = tuple((velocity[index + 1][axis] - velocity[index][axis]) / dt for axis in range(3))
        errors.append(math.sqrt(sum((left - right) ** 2 for left, right in zip(acceleration, gravity))))
    if not errors:
        return PhysicsCheck("gravity_consistency", False, {"sample_count": 0.0}, "No free-fall intervals")
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    return PhysicsCheck(
        "gravity_consistency",
        rmse <= tolerance_m_s2,
        {"acceleration_rmse_m_s2": rmse, "tolerance_m_s2": tolerance_m_s2, "sample_count": float(len(errors))},
        "" if rmse <= tolerance_m_s2 else "Free-flight acceleration is inconsistent with gravity",
    )


def _ballistic_evidence_mode(record: EpisodeRecord) -> str:
    """Return the native scenario contract's required free-flight evidence."""

    scenario_spec = record.extras.get("native_scenario_spec")
    if not isinstance(scenario_spec, Mapping):
        return "not_required"
    scenario = str(scenario_spec.get("scenario") or "")
    if scenario in {"ramp_launch", "projectile_roll_off_edge", "roll_off_edge"}:
        return "post_release"
    if record.family in {"falling_catch", "projectile_rebound"}:
        return "precontact"
    return "not_required"


def _requires_ballistic_evidence(record: EpisodeRecord) -> bool:
    return _ballistic_evidence_mode(record) != "not_required"


def _objective_metric_success_reference(
    record: EpisodeRecord,
) -> tuple[bool | None, str]:
    """Return the label that persisted objective metrics are meant to support."""

    if record.label_status.value == "verified_objective":
        return record.task_success, "task_success"
    candidate = record.objective_evidence.get("diagnostic_candidate_outcome")
    if isinstance(candidate, Mapping) and isinstance(
        candidate.get("task_success"), bool
    ):
        return bool(candidate["task_success"]), "diagnostic candidate task_success"
    return None, "unverified candidate"


def _ordered_contact_sequence(
    expected: Sequence[str],
    observed: Sequence[str],
) -> tuple[bool, bool]:
    """Return ``(complete, malformed)`` for an ordered contact sequence.

    Repeated contacts with a stage already reached are physically ordinary and
    do not undo progress. Once the declared sequence is complete, later
    recontacts are also harmless. A future stage observed before its next
    required predecessor is the only malformed ordering.
    """

    progress = 0
    for value in observed:
        if progress == len(expected):
            break
        if value == expected[progress]:
            progress += 1
        elif value in expected[:progress]:
            continue
        elif value in expected[progress + 1 :]:
            return False, True
        # Values outside the contract and repetitions of completed stages do
        # not affect sequence progress.
    return progress == len(expected), False


def _event_aware_free_fall_mask(
    rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    free_fall_field: str,
    *,
    precontact_only: bool = True,
    release_x_m: float | None = None,
    release_direction: int = 1,
    stop_at_next_contact: bool = False,
) -> list[bool]:
    """Exclude contact-adjacent samples from encoded-rate gravity checks.

    A short impact can occur entirely between adjacent 30 Hz samples, leaving
    both endpoint rows classified as free flight.  Mask the frame nearest each
    persisted contact plus one-half neighboring frame so its impulse is not
    divided by the encoded frame interval and mistaken for acceleration.
    """

    timestamps = [float(row["timestamp"]) for row in rows]
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    positive_intervals = sorted(value for value in intervals if value > 0.0)
    median_interval = (
        positive_intervals[len(positive_intervals) // 2]
        if positive_intervals
        else 1.0 / 30.0
    )
    exclusion_radius_s = 1.5 * median_interval + 1e-9
    contact_times = [
        float(event["timestamp"])
        for event in event_rows
        if event.get("timestamp") is not None
    ]
    first_contact_time = min(contact_times, default=math.inf)
    next_contact_time = min(
        (
            timestamp
            for timestamp in contact_times
            if timestamp > first_contact_time + 1e-9
        ),
        default=math.inf,
    )
    mask: list[bool] = []
    for row, timestamp in zip(rows, timestamps):
        motion_mode = str(
            row.get("object.motion_mode", row.get("motion_mode", ""))
        )
        contact_role = str(
            row.get("contact.role", row.get("contact_role", "none"))
        )
        mask.append(
            bool(row[free_fall_field])
            and motion_mode in {"", "free_flight"}
            and contact_role in {"", "none"}
            and (not precontact_only or timestamp < first_contact_time)
            and (not stop_at_next_contact or timestamp < next_contact_time)
            and (
                release_x_m is None
                or release_direction
                * (
                    float(
                        row.get("object.position", row.get("object.position_world_m"))[0]
                    )
                    - release_x_m
                )
                >= 0.0
            )
            and all(
                abs(timestamp - event_time) > exclusion_radius_s
                for event_time in contact_times
            )
        )
    return mask


def bounce_restitution_check(
    preimpact_normal_velocity_m_s: float,
    postimpact_normal_velocity_m_s: float,
    configured_restitution: float,
    *,
    absolute_tolerance: float = 0.15,
) -> PhysicsCheck:
    """Compare measured normal velocity ratio with configured/effective restitution."""

    if preimpact_normal_velocity_m_s >= 0 or postimpact_normal_velocity_m_s < 0:
        return PhysicsCheck(
            "bounce_restitution",
            False,
            {
                "preimpact_normal_velocity_m_s": preimpact_normal_velocity_m_s,
                "postimpact_normal_velocity_m_s": postimpact_normal_velocity_m_s,
            },
            "Impact velocities have implausible signs",
        )
    measured = -postimpact_normal_velocity_m_s / preimpact_normal_velocity_m_s
    error = abs(measured - configured_restitution)
    energy_ratio = measured * measured
    passed = error <= absolute_tolerance and energy_ratio <= 1.05
    return PhysicsCheck(
        "bounce_restitution",
        passed,
        {
            "configured_restitution": configured_restitution,
            "measured_effective_restitution": measured,
            "absolute_error": error,
            "normal_energy_ratio": energy_ratio,
        },
        "" if passed else "Bounce response is inconsistent with restitution or gains energy",
    )


@dataclass(slots=True)
class EpisodeQC:
    """Aggregated hard failures, warnings, and measurements for one episode."""

    episode_uuid: str
    episode_index: int
    release_eligible: bool
    passed: bool = True
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.hard_failures.append(message)
        self.passed = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SOURCE_MUJOCO_TOOL_VISIBILITY_TOPOLOGY_SCHEMA = (
    "source-mujoco-tool-visibility-topology/v1"
)


def _validate_source_mujoco_visibility_topology(
    result: EpisodeQC,
    visibility: Mapping[str, Any],
    *,
    source_scenario: Mapping[str, Any] | None,
    tool_applicable: bool,
) -> None:
    """Bind proxy and bilateral identities to the immutable compiled topology."""

    physics = (
        source_scenario.get("physics")
        if isinstance(source_scenario, Mapping)
        else None
    )
    topology = (
        physics.get("tool_visibility_topology")
        if isinstance(physics, Mapping)
        else None
    )
    if not isinstance(topology, Mapping):
        result.fail(
            "source_mujoco SourceScenarioSpec lacks compiled tool visibility topology"
        )
        return
    expected_fields = {
        "schema_version",
        "tool_geom_body_ids",
        "left_tool_geom_ids",
        "right_tool_geom_ids",
        "left_tool_body_id",
        "right_tool_body_id",
    }
    if set(topology) != expected_fields:
        result.fail("source_mujoco compiled tool visibility topology fields changed")
    if (
        topology.get("schema_version")
        != _SOURCE_MUJOCO_TOOL_VISIBILITY_TOPOLOGY_SCHEMA
    ):
        result.fail("source_mujoco compiled tool visibility topology schema changed")
    try:
        topology_digest = sha256_json(topology)
    except (TypeError, ValueError):
        result.fail("source_mujoco compiled tool visibility topology is not canonical JSON")
        return
    if not isinstance(physics, Mapping) or physics.get(
        "tool_visibility_topology_sha256"
    ) != topology_digest:
        result.fail("source_mujoco compiled tool visibility topology hash changed")
    if visibility.get("tool_visibility_topology_sha256") != topology_digest:
        result.fail("source_mujoco visibility tool topology hash differs from SourceScenarioSpec")

    geom_body_ids = topology.get("tool_geom_body_ids")
    left_geom_ids = topology.get("left_tool_geom_ids")
    right_geom_ids = topology.get("right_tool_geom_ids")
    left_body_id = topology.get("left_tool_body_id")
    right_body_id = topology.get("right_tool_body_id")
    canonical_map = isinstance(geom_body_ids, Mapping)
    if canonical_map:
        for raw_geom_id, raw_body_id in geom_body_ids.items():
            if (
                not isinstance(raw_geom_id, str)
                or not raw_geom_id.isdigit()
                or str(int(raw_geom_id)) != raw_geom_id
                or not isinstance(raw_body_id, int)
                or isinstance(raw_body_id, bool)
                or raw_body_id < 0
            ):
                canonical_map = False
                break
    canonical_sides = all(
        isinstance(values, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        )
        and values == sorted(set(values))
        for values in (left_geom_ids, right_geom_ids)
    )
    canonical_bodies = all(
        value is None
        or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)
        for value in (left_body_id, right_body_id)
    )
    if not canonical_map or not canonical_sides or not canonical_bodies:
        result.fail("source_mujoco compiled tool visibility topology is malformed")
    elif tool_applicable:
        if (
            not geom_body_ids
            or not left_geom_ids
            or not right_geom_ids
            or left_body_id is None
            or right_body_id is None
            or left_body_id == right_body_id
            or any(geom_body_ids.get(str(geom_id)) != left_body_id for geom_id in left_geom_ids)
            or any(geom_body_ids.get(str(geom_id)) != right_body_id for geom_id in right_geom_ids)
        ):
            result.fail(
                "source_mujoco compiled bilateral tool identities are inconsistent"
            )
    elif (
        geom_body_ids != {}
        or left_geom_ids != []
        or right_geom_ids != []
        or left_body_id is not None
        or right_body_id is not None
    ):
        result.fail("source_mujoco passive scenario declares tool topology")

    for field_name in expected_fields - {"schema_version"}:
        if visibility.get(field_name) != topology.get(field_name):
            result.fail(
                f"source_mujoco visibility {field_name} differs from compiled SourceScenarioSpec topology"
            )


def _visibility_number(
    result: EpisodeQC,
    mapping: Mapping[str, Any],
    name: str,
    *,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | int | None:
    """Parse an untrusted visibility scalar without ever aborting QC."""

    raw = mapping.get(name)
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        result.fail(f"source_mujoco visibility QC has invalid numeric {name}")
        return None
    value = float(raw)
    if not math.isfinite(value):
        result.fail(f"source_mujoco visibility QC has non-finite {name}")
        return None
    if integer and not value.is_integer():
        result.fail(f"source_mujoco visibility QC has non-integer {name}")
        return None
    if minimum is not None and value < minimum:
        result.fail(f"source_mujoco visibility QC {name} is below {minimum}")
    if maximum is not None and value > maximum:
        result.fail(f"source_mujoco visibility QC {name} exceeds {maximum}")
    return int(value) if integer else value


def _source_visibility_vector(
    value: Any,
    *,
    width: int,
    label: str,
) -> tuple[float, ...]:
    """Parse a finite vector from an untrusted saved SourceScenarioSpec."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != width
    ):
        raise ValueError(f"{label} must contain {width} values")
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric values") from error
    if any(not math.isfinite(item) for item in parsed):
        raise ValueError(f"{label} contains a non-finite value")
    return parsed


def _source_visibility_unit_vector(
    value: Sequence[float],
    *,
    label: str,
) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{label} has zero or invalid length")
    normalized = tuple(float(item) / norm for item in value)
    if len(normalized) != 3:
        raise ValueError(f"{label} must contain three values")
    return normalized[0], normalized[1], normalized[2]


def _rotate_source_visibility_axis(
    quaternion_wxyz: Sequence[float],
    local_axis: int,
) -> tuple[float, float, float]:
    """Rotate one local box axis by a normalized WXYZ quaternion."""

    if local_axis not in {0, 1, 2}:
        raise ValueError("wall local normal axis is invalid")
    w, x, y, z = _source_visibility_vector(
        quaternion_wxyz,
        width=4,
        label="physical wall quaternion",
    )
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if abs(norm - 1.0) > 1e-5:
        raise ValueError("physical wall quaternion is not normalized")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    rotation = (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        (
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ),
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )
    return _source_visibility_unit_vector(
        tuple(rotation[row][local_axis] for row in range(3)),
        label="physical wall normal",
    )


def _p0c_wall_side_on_evidence(
    source_scenario: Mapping[str, Any],
    *,
    required_view: str,
    maximum_alignment: float,
) -> dict[str, Any]:
    """Recompute the saved review camera's alignment to the physical wall."""

    cameras = source_scenario.get("cameras")
    if not isinstance(cameras, Sequence) or isinstance(
        cameras, (str, bytes, bytearray)
    ):
        raise ValueError("P0c wall SourceScenarioSpec lacks canonical cameras")
    selected_cameras = [
        camera
        for camera in cameras
        if isinstance(camera, Mapping) and camera.get("name") == required_view
    ]
    if len(selected_cameras) != 1:
        raise ValueError(
            f"P0c wall SourceScenarioSpec must contain one {required_view} camera"
        )
    camera = selected_cameras[0]
    if camera.get("role") != "task_specific_secondary":
        raise ValueError("P0c wall secondary camera role is not canonical")
    camera_pose = camera.get("pose")
    if not isinstance(camera_pose, Mapping):
        raise ValueError("P0c wall secondary camera lacks a pose")
    camera_position = _source_visibility_vector(
        camera_pose.get("position_m"),
        width=3,
        label="P0c wall secondary camera position",
    )
    camera_target = _source_visibility_vector(
        camera.get("look_at_m"),
        width=3,
        label="P0c wall secondary camera look-at",
    )
    view_direction = _source_visibility_unit_vector(
        tuple(
            target - position
            for target, position in zip(camera_target, camera_position)
        ),
        label="P0c wall secondary camera view direction",
    )

    fixtures = source_scenario.get("fixtures")
    if not isinstance(fixtures, Sequence) or isinstance(
        fixtures, (str, bytes, bytearray)
    ):
        raise ValueError("P0c wall SourceScenarioSpec lacks physical fixtures")
    wall_fixtures = []
    for fixture in fixtures:
        if not isinstance(fixture, Mapping) or fixture.get("fixture_type") != "wall":
            continue
        parameters = fixture.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("P0c wall fixture parameters are malformed")
        if (
            fixture.get("physical") is not True
            or fixture.get("anchored") is not True
            or parameters.get("expected_task_contact") is not True
            or parameters.get("collision_enabled") is not True
        ):
            raise ValueError("P0c wall fixture is not an anchored physical contact surface")
        wall_fixtures.append((fixture, parameters))
    if len(wall_fixtures) != 1:
        raise ValueError("P0c wall SourceScenarioSpec must contain one physical task wall")
    wall, wall_parameters = wall_fixtures[0]
    half_size = _source_visibility_vector(
        wall_parameters.get("half_size_m"),
        width=3,
        label="physical wall half-size",
    )
    if any(value <= 0.0 for value in half_size):
        raise ValueError("physical wall half-size must be positive")
    ordered_axes = sorted(range(3), key=lambda axis: half_size[axis])
    normal_axis = ordered_axes[0]
    if half_size[ordered_axes[1]] - half_size[normal_axis] <= 1e-9:
        raise ValueError("physical wall has no unique thin normal axis")
    wall_pose = wall.get("pose")
    if not isinstance(wall_pose, Mapping):
        raise ValueError("physical wall lacks a pose")
    wall_normal = _rotate_source_visibility_axis(
        wall_pose.get("quaternion_wxyz"),
        normal_axis,
    )
    alignment = abs(
        sum(
            view_component * normal_component
            for view_component, normal_component in zip(
                view_direction, wall_normal
            )
        )
    )
    side_on = alignment <= maximum_alignment + 1e-12
    return {
        "wall_fixture_id": str(wall.get("fixture_id") or ""),
        "wall_local_normal_axis": normal_axis,
        "wall_normal_world": list(wall_normal),
        "camera_view_direction_world": list(view_direction),
        "absolute_view_wall_normal_dot": alignment,
        "maximum_absolute_view_wall_normal_dot": maximum_alignment,
        "side_on": side_on,
    }


def _validate_source_mujoco_task_visibility_contract(
    result: EpisodeQC,
    *,
    source_scenario: Mapping[str, Any] | None,
    recomputed_checkpoints: Mapping[str, Mapping[str, bool]],
    thresholds: Mapping[str, Any],
) -> None:
    """Apply task-specific required-view checks to independently replayed data."""

    metrics: dict[str, Any] = {
        "schema_version": SOURCE_MUJOCO_TASK_VISIBILITY_CONTRACT_SCHEMA,
        "applicable": False,
        "source_scenario_valid": False,
    }
    try:
        requirement = source_mujoco_task_visibility_requirement(source_scenario)
    except (TypeError, ValueError) as error:
        metrics["source_scenario_error"] = str(error)
        result.metrics["source_mujoco_task_visibility"] = metrics
        result.fail(
            "source_mujoco task-specific visibility SourceScenarioSpec is malformed: "
            f"{error}"
        )
        return
    metrics["source_scenario_valid"] = True
    if requirement is None:
        result.metrics["source_mujoco_task_visibility"] = metrics
        return

    required_view = str(requirement.get("required_view") or "")
    raw_checkpoints = requirement.get("required_checkpoints")
    if (
        required_view not in {"main", "secondary"}
        or not isinstance(raw_checkpoints, Sequence)
        or isinstance(raw_checkpoints, (str, bytes, bytearray))
    ):
        metrics["contract_error"] = "internal required-view contract is malformed"
        result.metrics["source_mujoco_task_visibility"] = metrics
        result.fail("source_mujoco task-specific visibility contract is malformed")
        return
    required_checkpoints = tuple(str(value) for value in raw_checkpoints)
    checkpoint_visibility = {
        checkpoint: bool(
            recomputed_checkpoints.get(required_view, {}).get(checkpoint, False)
        )
        for checkpoint in required_checkpoints
    }
    metrics.update(
        {
            "applicable": True,
            "contract_id": str(requirement.get("contract_id") or ""),
            "required_view": required_view,
            "required_checkpoints": list(required_checkpoints),
            "required_checkpoint_visibility": checkpoint_visibility,
            "all_required_checkpoints_visible": all(
                checkpoint_visibility.values()
            ),
        }
    )
    for checkpoint, visible in checkpoint_visibility.items():
        if not visible:
            result.fail(
                "source_mujoco task-specific visibility requires "
                f"{required_view}/{checkpoint} to be visible"
            )

    if requirement.get("require_side_on_wall_normal") is True:
        maximum_alignment_raw = thresholds.get("maximum_side_on_wall_normal_dot")
        if (
            not isinstance(maximum_alignment_raw, (int, float))
            or isinstance(maximum_alignment_raw, bool)
            or not math.isfinite(float(maximum_alignment_raw))
            or not 0.0 <= float(maximum_alignment_raw) < 1.0
        ):
            metrics["side_on_wall_error"] = "side-on threshold is malformed"
            result.fail("source_mujoco P0c wall side-on threshold is malformed")
        elif not isinstance(source_scenario, Mapping):
            metrics["side_on_wall_error"] = "SourceScenarioSpec is unavailable"
            result.fail("source_mujoco P0c wall SourceScenarioSpec is unavailable")
        else:
            try:
                side_on_evidence = _p0c_wall_side_on_evidence(
                    source_scenario,
                    required_view=required_view,
                    maximum_alignment=float(maximum_alignment_raw),
                )
            except (TypeError, ValueError) as error:
                metrics["side_on_wall_error"] = str(error)
                result.fail(
                    "source_mujoco P0c wall camera/fixture specification is malformed: "
                    f"{error}"
                )
            else:
                metrics["side_on_wall"] = side_on_evidence
                if side_on_evidence["side_on"] is not True:
                    result.fail(
                        "source_mujoco P0c wall secondary view is not side-on "
                        "to the physical wall normal"
                    )
    result.metrics["source_mujoco_task_visibility"] = metrics


def _validate_source_mujoco_visibility_qc(
    result: EpisodeQC,
    visibility: Mapping[str, Any],
    *,
    end_effector: str,
    expected_frame_count: int | None,
    frame_rows: Sequence[Mapping[str, Any]] = (),
    event_rows: Sequence[Mapping[str, Any]] = (),
    record_key_event_name: str | None = None,
    record_key_event_time_s: float | None = None,
    objective_key_event_source: str | None = None,
    event_time_tolerance_s: float = 1e-9,
    source_scenario: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed on recomputable persisted rendered visibility evidence."""

    thresholds = SOURCE_MUJOCO_VISUAL_THRESHOLDS
    required_views = ("main", "secondary")
    required_checkpoints = ("initial", "apex", "key_event", "final")
    tool_applicable = end_effector != "no_robot"
    source_physics = (
        source_scenario.get("physics")
        if isinstance(source_scenario, Mapping)
        else None
    )
    support_identity_declared = bool(
        isinstance(source_physics, Mapping)
        and "structural_support_geom_ids" in source_physics
    )
    structural_support_geom_ids: list[int] = []
    structural_support_station_by_geom: dict[int, str] = {}
    if support_identity_declared:
        raw_support_geom_ids = source_physics.get("structural_support_geom_ids")
        if not isinstance(raw_support_geom_ids, Sequence) or isinstance(
            raw_support_geom_ids, (str, bytes, bytearray)
        ):
            result.fail(
                "source_mujoco visibility QC lacks bound structural support geom IDs"
            )
        elif any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in raw_support_geom_ids
        ):
            result.fail(
                "source_mujoco visibility QC structural support geom IDs are invalid"
            )
        else:
            structural_support_geom_ids = sorted(set(raw_support_geom_ids))
            if structural_support_geom_ids != list(raw_support_geom_ids):
                result.fail(
                    "source_mujoco visibility QC structural support geom IDs are not canonical"
                )
            raw_support_fixture_ids = source_physics.get(
                "structural_support_fixture_ids"
            )
            if (
                not isinstance(raw_support_fixture_ids, Sequence)
                or isinstance(
                    raw_support_fixture_ids, (str, bytes, bytearray)
                )
                or len(raw_support_fixture_ids)
                != len(structural_support_geom_ids)
            ):
                result.fail(
                    "source_mujoco visibility QC support geom/fixture identities disagree"
                )
            raw_station_by_geom = source_physics.get(
                "structural_support_station_by_geom"
            )
            if not isinstance(raw_station_by_geom, Mapping):
                result.fail(
                    "source_mujoco visibility QC lacks bound support-station identities"
                )
            else:
                try:
                    structural_support_station_by_geom = {
                        int(geom_id): str(station_id)
                        for geom_id, station_id in raw_station_by_geom.items()
                    }
                except (TypeError, ValueError):
                    structural_support_station_by_geom = {}
                if set(structural_support_station_by_geom) != set(
                    structural_support_geom_ids
                ) or any(
                    not value
                    for value in structural_support_station_by_geom.values()
                ):
                    result.fail(
                        "source_mujoco visibility QC support-station identities are invalid"
                    )
    _validate_source_mujoco_visibility_topology(
        result,
        visibility,
        source_scenario=source_scenario,
        tool_applicable=tool_applicable,
    )
    # Reconstruct the renderer's left/right masks from the immutable compiled
    # geom/body topology.  Contact geoms alone are not enough for Robotiq: its
    # visible pad geoms share the same rigid finger bodies.  Keeping these IDs
    # here lets persisted QC independently replay each claimed side-pixel count
    # instead of trusting a self-authored aggregate.
    render_geom_ids_by_side: dict[str, set[int]] = {
        "left": set(),
        "right": set(),
    }
    physics = (
        source_scenario.get("physics")
        if isinstance(source_scenario, Mapping)
        else None
    )
    topology = (
        physics.get("tool_visibility_topology")
        if isinstance(physics, Mapping)
        else None
    )
    if isinstance(topology, Mapping):
        raw_body_map = topology.get("tool_geom_body_ids")
        left_body_id = topology.get("left_tool_body_id")
        right_body_id = topology.get("right_tool_body_id")
        if isinstance(raw_body_map, Mapping):
            for raw_geom_id, raw_body_id in raw_body_map.items():
                try:
                    geom_id = int(raw_geom_id)
                    body_id = int(raw_body_id)
                except (TypeError, ValueError):
                    continue
                if body_id == left_body_id:
                    render_geom_ids_by_side["left"].add(geom_id)
                if body_id == right_body_id:
                    render_geom_ids_by_side["right"].add(geom_id)
    if visibility.get("schema_version") != SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA:
        result.fail(
            "source_mujoco visibility QC does not use "
            f"{SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA}"
        )
    if visibility.get("thresholds") != thresholds:
        result.fail("source_mujoco visibility QC thresholds changed or are incomplete")
    if visibility.get("evaluated") is not True:
        result.fail("source_mujoco visibility QC was not evaluated from rendered streams")
    if visibility.get("rendered_streams_complete") is not True:
        result.fail("source_mujoco visibility QC lacks both complete rendered streams")
    if visibility.get("required_views") != list(required_views):
        result.fail("source_mujoco visibility QC does not require both canonical views")
    if visibility.get("required_checkpoint_names") != list(required_checkpoints):
        result.fail("source_mujoco visibility QC checkpoint contract changed")

    rendered_count = _visibility_number(
        result, visibility, "rendered_frame_count", integer=True, minimum=1
    )
    declared_count = _visibility_number(
        result, visibility, "expected_frame_count", integer=True, minimum=1
    )
    if expected_frame_count is None:
        result.fail("source_mujoco visibility QC cannot bind a missing saved frame count")
    else:
        if rendered_count is not None and rendered_count != expected_frame_count:
            result.fail("source_mujoco visibility QC frame count differs from saved media")
        if declared_count is not None and declared_count != expected_frame_count:
            result.fail("source_mujoco visibility QC expected frame count changed")
    if frame_rows and expected_frame_count is not None and len(frame_rows) != expected_frame_count:
        result.fail("source_mujoco visibility QC frame evidence differs from frame Parquet")
    elif not frame_rows:
        result.fail("source_mujoco visibility QC lacks persisted frame rows for replay")

    persisted_timestamps: list[float] = []
    persisted_z: list[float] = []
    for index, row in enumerate(frame_rows):
        timestamp = _visibility_number(result, row, "timestamp", minimum=0.0)
        position = row.get("object.position")
        if timestamp is None:
            persisted_timestamps.append(math.nan)
        else:
            persisted_timestamps.append(float(timestamp))
        if (
            not isinstance(position, Sequence)
            or isinstance(position, (str, bytes, bytearray))
            or len(position) < 3
            or not isinstance(position[2], (int, float))
            or isinstance(position[2], bool)
            or not math.isfinite(float(position[2]))
        ):
            result.fail(
                f"source_mujoco visibility QC cannot derive apex from frame {index}"
            )
            persisted_z.append(-math.inf)
        else:
            persisted_z.append(float(position[2]))

    event_values: dict[str, float | int | None] = {}
    for prefix in ("planned", "actual"):
        event_values[f"{prefix}_time"] = _visibility_number(
            result, visibility, f"{prefix}_key_event_time_s", minimum=0.0
        )
        event_values[f"{prefix}_index"] = _visibility_number(
            result,
            visibility,
            f"{prefix}_key_event_frame_index",
            integer=True,
            minimum=0,
            maximum=(None if expected_frame_count is None else expected_frame_count - 1),
        )
        event_values[f"{prefix}_frame_time"] = _visibility_number(
            result,
            visibility,
            f"{prefix}_key_event_frame_timestamp_s",
            minimum=0.0,
        )
        event_time = event_values[f"{prefix}_time"]
        frame_index = event_values[f"{prefix}_index"]
        frame_time = event_values[f"{prefix}_frame_time"]
        if event_time is not None and frame_time is not None and abs(
            float(frame_time) - float(event_time)
        ) > 1.0 / 60.0 + 1e-9:
            result.fail(
                f"source_mujoco visibility QC {prefix} event/frame time binding changed"
            )
        if (
            isinstance(frame_index, int)
            and frame_index < len(persisted_timestamps)
            and math.isfinite(persisted_timestamps[frame_index])
        ):
            if frame_time is None or abs(
                float(frame_time) - persisted_timestamps[frame_index]
            ) > 1e-9:
                result.fail(
                    f"source_mujoco visibility QC {prefix} frame timestamp differs from frame Parquet"
                )
            if event_time is not None:
                nearest = min(
                    range(len(persisted_timestamps)),
                    key=lambda item: abs(persisted_timestamps[item] - float(event_time)),
                )
                if frame_index != nearest:
                    result.fail(
                        f"source_mujoco visibility QC {prefix} event frame index is not nearest persisted frame"
                    )

    actual_name = visibility.get("actual_key_event_name")
    actual_source = visibility.get("actual_key_event_source")
    if not isinstance(actual_name, str) or not actual_name.strip():
        result.fail("source_mujoco visibility QC lacks actual key-event name")
    if not isinstance(actual_source, str) or not actual_source.strip():
        result.fail("source_mujoco visibility QC lacks actual key-event source")
    if record_key_event_name is not None and actual_name != record_key_event_name:
        result.fail("source_mujoco visibility key-event name differs from objective replay")
    actual_time = event_values.get("actual_time")
    if record_key_event_time_s is not None and (
        actual_time is None
        or not math.isfinite(float(record_key_event_time_s))
        or abs(float(actual_time) - float(record_key_event_time_s)) > 1e-9
    ):
        result.fail("source_mujoco visibility key-event time differs from objective replay")
    if objective_key_event_source is not None and actual_source != objective_key_event_source:
        result.fail("source_mujoco visibility key-event source differs from objective evidence")

    target_fraction = _visibility_number(
        result,
        visibility,
        "target_visible_frame_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    minimum_margin = _visibility_number(result, visibility, "minimum_bbox_margin_px")
    key_area = _visibility_number(
        result, visibility, "key_event_object_area_px", integer=True, minimum=0
    )
    maximum_under = _visibility_number(
        result,
        visibility,
        "maximum_underexposed_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    maximum_over = _visibility_number(
        result,
        visibility,
        "maximum_overexposed_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if target_fraction is None or target_fraction < float(
        thresholds["minimum_target_visible_frame_fraction"]
    ):
        result.fail("source_mujoco target is not visible in at least 90% of frames")
    for field_name, label in (
        ("initial_state_visible_in_any_view", "initial target state"),
        ("apex_visible_in_any_view", "target apex"),
        ("key_event_visible_in_any_view", "key event"),
        ("final_state_visible_in_any_view", "final target state"),
    ):
        if visibility.get(field_name) is not True:
            result.fail(f"source_mujoco {label} is not visible with bbox margin")
    if visibility.get("critically_cropped") is not False:
        result.fail("source_mujoco target is critically cropped at the key event")
    if minimum_margin is None or minimum_margin < float(thresholds["minimum_bbox_margin_px"]):
        result.fail("source_mujoco key-event bbox margin is below 8 pixels")
    if key_area is None or key_area < int(thresholds["minimum_key_event_object_area_px"]):
        result.fail("source_mujoco key-event target footprint is below 64 pixels")
    if maximum_under is None or maximum_under > float(thresholds["maximum_underexposed_fraction"]):
        result.fail("source_mujoco underexposed image fraction exceeds 0.35")
    if maximum_over is None or maximum_over > float(thresholds["maximum_overexposed_fraction"]):
        result.fail("source_mujoco overexposed image fraction exceeds 0.30")
    if visibility.get("camera_roles_correct") is not True:
        result.fail("source_mujoco visibility QC camera roles are incomplete")

    fixture_applicable = not tool_applicable
    if visibility.get("tool_visibility_applicable") is not tool_applicable:
        result.fail("source_mujoco tool visibility applicability is incorrect")
    if visibility.get("fixture_visibility_applicable") is not fixture_applicable:
        result.fail("source_mujoco fixture visibility applicability is incorrect")
    if tool_applicable:
        if visibility.get("tool_visible_at_key_event") is not True:
            result.fail("source_mujoco gripper is not visible at the key event")
        if visibility.get("fixture_visible_at_key_event") is not None:
            result.fail("source_mujoco F1 fixture visibility must be inapplicable")
    else:
        if visibility.get("fixture_visible_at_key_event") is not True:
            result.fail("source_mujoco P0 task fixture is not visible at the key event")
        if visibility.get("tool_visible_at_key_event") is not None:
            result.fail("source_mujoco P0 tool visibility must be inapplicable")
    if visibility.get("counterpart_visible_at_key_event") is not True:
        result.fail("source_mujoco applicable counterpart is not visible at the key event")

    views = visibility.get("views")
    recomputed_presence: dict[str, list[bool]] = {}
    recomputed_checkpoints: dict[str, dict[str, bool]] = {}
    recomputed_key_areas: list[int] = []
    recomputed_key_margins: list[float] = []
    recomputed_under: list[float] = []
    recomputed_over: list[float] = []
    frame_metrics_by_view: dict[str, list[Mapping[str, Any]]] = {}
    if not isinstance(views, Mapping) or set(views) != set(required_views):
        result.fail("source_mujoco visibility QC views must be exactly main and secondary")
        views = {}
    metric_fields = (
        "frame_index",
        "timestamp_s",
        "object_pixel_count",
        "segmentation_bbox_margin_px",
        "projected_sphere_margin_px",
        "projected_center_visible",
        "target_present",
        "tool_pixel_count",
        "left_tool_pixel_count",
        "right_tool_pixel_count",
        "fixture_pixel_count",
        "geom_pixel_counts",
        "underexposed_fraction",
        "overexposed_fraction",
    )
    for view_name in required_views:
        view = views.get(view_name)
        if not isinstance(view, Mapping):
            result.fail(f"source_mujoco visibility QC lacks {view_name} view evidence")
            continue
        metrics = view.get("frames")
        if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes, bytearray)):
            result.fail(f"source_mujoco visibility QC lacks {view_name} per-frame metrics")
            continue
        if expected_frame_count is None or len(metrics) != expected_frame_count:
            result.fail(f"source_mujoco visibility QC {view_name} per-frame count changed")
        parsed_metrics: list[Mapping[str, Any]] = []
        view_presence: list[bool] = []
        for index, metric in enumerate(metrics):
            if not isinstance(metric, Mapping):
                result.fail(f"source_mujoco visibility QC {view_name} frame {index} is malformed")
                continue
            parsed_metrics.append(metric)
            metric_index = _visibility_number(result, metric, "frame_index", integer=True, minimum=0)
            timestamp = _visibility_number(result, metric, "timestamp_s", minimum=0.0)
            object_pixels = _visibility_number(result, metric, "object_pixel_count", integer=True, minimum=0)
            _visibility_number(result, metric, "segmentation_bbox_margin_px")
            _visibility_number(result, metric, "projected_sphere_margin_px")
            tool_pixels = _visibility_number(result, metric, "tool_pixel_count", integer=True, minimum=0)
            left_pixels = _visibility_number(result, metric, "left_tool_pixel_count", integer=True, minimum=0)
            right_pixels = _visibility_number(result, metric, "right_tool_pixel_count", integer=True, minimum=0)
            _visibility_number(result, metric, "fixture_pixel_count", integer=True, minimum=0)
            under = _visibility_number(result, metric, "underexposed_fraction", minimum=0.0, maximum=1.0)
            over = _visibility_number(result, metric, "overexposed_fraction", minimum=0.0, maximum=1.0)
            if metric_index != index:
                result.fail(f"source_mujoco visibility QC {view_name} frame indices are not contiguous")
            if index < len(persisted_timestamps) and timestamp is not None and abs(
                float(timestamp) - persisted_timestamps[index]
            ) > 1e-9:
                result.fail(f"source_mujoco visibility QC {view_name} timestamp differs from frame Parquet")
            if metric.get("projected_center_visible") not in {True, False}:
                result.fail(f"source_mujoco visibility QC {view_name} projected visibility is malformed")
            expected_presence = bool(
                object_pixels is not None
                and object_pixels >= int(thresholds["minimum_trajectory_object_area_px"])
                and metric.get("projected_center_visible") is True
            )
            if metric.get("target_present") is not expected_presence:
                result.fail(f"source_mujoco visibility QC {view_name} target presence is not recomputable")
            view_presence.append(expected_presence)
            if tool_pixels is not None and left_pixels is not None and right_pixels is not None and tool_pixels < max(left_pixels, right_pixels):
                result.fail(f"source_mujoco visibility QC {view_name} tool union is smaller than a side mask")
            geom_counts = metric.get("geom_pixel_counts")
            if not isinstance(geom_counts, Mapping):
                result.fail(f"source_mujoco visibility QC {view_name} lacks per-geom segmentation counts")
            else:
                normalized_geom_counts: dict[int, int] = {}
                for geom_id, count in geom_counts.items():
                    canonical_geom_id = (
                        isinstance(geom_id, str)
                        and geom_id.isdigit()
                        and str(int(geom_id)) == geom_id
                    )
                    if not canonical_geom_id:
                        result.fail(f"source_mujoco visibility QC {view_name} has invalid geom ID")
                    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                        result.fail(f"source_mujoco visibility QC {view_name} has invalid geom pixel count")
                    if canonical_geom_id and isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                        normalized_geom_counts[int(geom_id)] = count
                expected_left_pixels = sum(
                    normalized_geom_counts.get(geom_id, 0)
                    for geom_id in render_geom_ids_by_side["left"]
                )
                expected_right_pixels = sum(
                    normalized_geom_counts.get(geom_id, 0)
                    for geom_id in render_geom_ids_by_side["right"]
                )
                expected_tool_pixels = sum(
                    normalized_geom_counts.get(geom_id, 0)
                    for geom_id in (
                        render_geom_ids_by_side["left"]
                        | render_geom_ids_by_side["right"]
                    )
                )
                if left_pixels != expected_left_pixels:
                    result.fail(
                        f"source_mujoco visibility QC {view_name} left-tool pixels "
                        "differ from compiled same-body render geoms"
                    )
                if right_pixels != expected_right_pixels:
                    result.fail(
                        f"source_mujoco visibility QC {view_name} right-tool pixels "
                        "differ from compiled same-body render geoms"
                    )
                if tool_pixels != expected_tool_pixels:
                    result.fail(
                        f"source_mujoco visibility QC {view_name} tool pixels differ "
                        "from the compiled side-mask union"
                    )
            if under is not None:
                recomputed_under.append(float(under))
            if over is not None:
                recomputed_over.append(float(over))
        frame_metrics_by_view[view_name] = parsed_metrics
        recomputed_presence[view_name] = view_presence
        view_fraction = _visibility_number(result, view, "target_visible_frame_fraction", minimum=0.0, maximum=1.0)
        if len(view_presence) == expected_frame_count and view_fraction is not None and abs(
            float(view_fraction) - sum(view_presence) / len(view_presence)
        ) > 1e-12:
            result.fail(f"source_mujoco visibility QC {view_name} target fraction changed")
        view_checkpoints = view.get("checkpoints")
        if not isinstance(view_checkpoints, Mapping) or set(view_checkpoints) != set(required_checkpoints):
            result.fail(f"source_mujoco visibility QC {view_name} checkpoints are incomplete")
            continue
        recomputed_checkpoints[view_name] = {}
        for checkpoint_name in required_checkpoints:
            checkpoint = view_checkpoints.get(checkpoint_name)
            if not isinstance(checkpoint, Mapping):
                result.fail(f"source_mujoco visibility QC lacks {view_name}/{checkpoint_name} metrics")
                continue
            checkpoint_index = _visibility_number(result, checkpoint, "frame_index", integer=True, minimum=0)
            if not isinstance(checkpoint_index, int) or checkpoint_index >= len(parsed_metrics):
                continue
            metric = parsed_metrics[checkpoint_index]
            for field_name in metric_fields:
                if checkpoint.get(field_name) != metric.get(field_name):
                    result.fail(f"source_mujoco visibility QC {view_name}/{checkpoint_name} differs from per-frame metrics")
                    break
            minimum_area = int(
                thresholds["minimum_key_event_object_area_px"]
                if checkpoint_name == "key_event"
                else thresholds["minimum_trajectory_object_area_px"]
            )
            object_pixels = metric.get("object_pixel_count")
            segmentation_margin = metric.get("segmentation_bbox_margin_px")
            projected_margin = metric.get("projected_sphere_margin_px")
            expected_visible = bool(
                isinstance(object_pixels, int)
                and not isinstance(object_pixels, bool)
                and object_pixels >= minimum_area
                and isinstance(segmentation_margin, (int, float))
                and not isinstance(segmentation_margin, bool)
                and math.isfinite(float(segmentation_margin))
                and isinstance(projected_margin, (int, float))
                and not isinstance(projected_margin, bool)
                and math.isfinite(float(projected_margin))
                and min(float(segmentation_margin), float(projected_margin))
                >= float(thresholds["minimum_bbox_margin_px"])
            )
            if checkpoint.get("visible") is not expected_visible:
                result.fail(f"source_mujoco visibility QC {view_name}/{checkpoint_name} visibility changed")
            recomputed_checkpoints[view_name][checkpoint_name] = expected_visible

    if support_identity_declared:
        support_visibility = visibility.get("structural_support_visibility")
        if not isinstance(support_visibility, Mapping):
            result.fail(
                "source_mujoco visibility QC lacks structural support render evidence"
            )
        else:
            minimum_support_area = int(
                thresholds["minimum_structural_support_area_px"]
            )
            maximum_pixels = {
                geom_id: max(
                    (
                        int(
                            metric.get("geom_pixel_counts", {}).get(
                                str(geom_id), 0
                            )
                        )
                        for metrics in frame_metrics_by_view.values()
                        for metric in metrics
                        if isinstance(metric.get("geom_pixel_counts"), Mapping)
                    ),
                    default=0,
                )
                for geom_id in structural_support_geom_ids
            }
            visible_by_view = {
                view_name: [
                    geom_id
                    for geom_id in structural_support_geom_ids
                    if max(
                        (
                            int(
                                metric.get("geom_pixel_counts", {}).get(
                                    str(geom_id), 0
                                )
                            )
                            for metric in frame_metrics_by_view.get(
                                view_name, ()
                            )
                            if isinstance(
                                metric.get("geom_pixel_counts"), Mapping
                            )
                        ),
                        default=0,
                    )
                    >= minimum_support_area
                ]
                for view_name in required_views
            }
            visible_any_view = sorted(
                {
                    geom_id
                    for values in visible_by_view.values()
                    for geom_id in values
                }
            )
            support_station_ids = sorted(
                set(structural_support_station_by_geom.values())
            )
            visible_support_station_ids = sorted(
                {
                    structural_support_station_by_geom[geom_id]
                    for geom_id in visible_any_view
                    if geom_id in structural_support_station_by_geom
                }
            )
            expected_support_visibility = {
                "applicable": bool(structural_support_geom_ids),
                "evaluated": True,
                "support_geom_ids": structural_support_geom_ids,
                "minimum_visible_area_px": minimum_support_area,
                "maximum_pixel_count_by_geom": {
                    str(geom_id): maximum_pixels[geom_id]
                    for geom_id in structural_support_geom_ids
                },
                "visible_geom_ids_by_view": visible_by_view,
                "visible_geom_ids_any_view": visible_any_view,
                "all_supports_visible_in_any_view": (
                    visible_any_view == structural_support_geom_ids
                ),
                "support_station_by_geom": {
                    str(geom_id): structural_support_station_by_geom[geom_id]
                    for geom_id in structural_support_geom_ids
                    if geom_id in structural_support_station_by_geom
                },
                "support_station_ids": support_station_ids,
                "visible_support_station_ids": visible_support_station_ids,
                "all_support_stations_visible": (
                    visible_support_station_ids == support_station_ids
                ),
            }
            if support_visibility != expected_support_visibility:
                result.fail(
                    "source_mujoco structural support render evidence cannot be replayed"
                )
            if structural_support_geom_ids and (
                expected_support_visibility[
                    "all_support_stations_visible"
                ]
                is not True
            ):
                result.fail(
                    "source_mujoco structural support frame is not visually grounded in either view"
                )

    if expected_frame_count and all(
        len(recomputed_presence.get(name, ())) == expected_frame_count
        for name in required_views
    ):
        aggregate_fraction = sum(
            any(recomputed_presence[name][index] for name in required_views)
            for index in range(expected_frame_count)
        ) / expected_frame_count
        if target_fraction is None or abs(float(target_fraction) - aggregate_fraction) > 1e-12:
            result.fail("source_mujoco visibility QC aggregate target fraction changed")

    expected_exposure_sample_count = (
        None
        if expected_frame_count is None
        else expected_frame_count * len(required_views)
    )
    if (
        expected_exposure_sample_count is None
        or len(recomputed_under) != expected_exposure_sample_count
        or len(recomputed_over) != expected_exposure_sample_count
    ):
        result.fail(
            "source_mujoco visibility QC cannot recompute complete exposure aggregates"
        )
    else:
        expected_maximum_under = max(recomputed_under)
        expected_maximum_over = max(recomputed_over)
        if maximum_under is None or abs(
            float(maximum_under) - expected_maximum_under
        ) > 1e-12:
            result.fail(
                "source_mujoco visibility QC maximum underexposed fraction changed"
            )
        if maximum_over is None or abs(
            float(maximum_over) - expected_maximum_over
        ) > 1e-12:
            result.fail(
                "source_mujoco visibility QC maximum overexposed fraction changed"
            )
        if expected_maximum_under > float(
            thresholds["maximum_underexposed_fraction"]
        ):
            result.fail(
                "source_mujoco recomputed underexposed image fraction exceeds 0.35"
            )
        if expected_maximum_over > float(
            thresholds["maximum_overexposed_fraction"]
        ):
            result.fail(
                "source_mujoco recomputed overexposed image fraction exceeds 0.30"
            )

    planned_index = event_values.get("planned_index")
    actual_index = event_values.get("actual_index")
    expected_checkpoint_indices: dict[str, int] = {}
    if expected_frame_count:
        expected_checkpoint_indices = {
            "initial": 0,
            "apex": max(range(len(persisted_z)), key=lambda index: persisted_z[index]) if persisted_z else -1,
            "key_event": int(actual_index) if isinstance(actual_index, int) else -1,
            "final": expected_frame_count - 1,
        }
    checkpoints = visibility.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(required_checkpoints):
        result.fail("source_mujoco visibility QC checkpoint evidence must be exact")
        checkpoints = {}
    for checkpoint_name in required_checkpoints:
        checkpoint = checkpoints.get(checkpoint_name)
        if not isinstance(checkpoint, Mapping):
            result.fail(f"source_mujoco visibility QC lacks {checkpoint_name} checkpoint")
            continue
        checkpoint_index = _visibility_number(result, checkpoint, "frame_index", integer=True, minimum=0)
        checkpoint_time = _visibility_number(result, checkpoint, "timestamp_s", minimum=0.0)
        expected_index = expected_checkpoint_indices.get(checkpoint_name)
        if expected_index is not None and checkpoint_index != expected_index:
            result.fail(f"source_mujoco {checkpoint_name} checkpoint index differs from persisted state")
        if isinstance(checkpoint_index, int) and checkpoint_index < len(persisted_timestamps) and checkpoint_time is not None and abs(
            float(checkpoint_time) - persisted_timestamps[checkpoint_index]
        ) > 1e-9:
            result.fail(f"source_mujoco {checkpoint_name} checkpoint timestamp differs from frame Parquet")
        expected_visible = any(
            recomputed_checkpoints.get(view_name, {}).get(checkpoint_name, False)
            for view_name in required_views
        )
        if checkpoint.get("visible_in_any_view") is not expected_visible:
            result.fail(f"source_mujoco {checkpoint_name} checkpoint aggregate changed")
        if checkpoint.get("visible_in_any_view") is not True:
            result.fail(f"source_mujoco {checkpoint_name} checkpoint is not visible in either view")

    checkpoint_aggregate_fields = {
        "initial": "initial_state_visible_in_any_view",
        "apex": "apex_visible_in_any_view",
        "key_event": "key_event_visible_in_any_view",
        "final": "final_state_visible_in_any_view",
    }
    recomputed_checkpoint_aggregates: dict[str, bool] = {}
    for checkpoint_name, field_name in checkpoint_aggregate_fields.items():
        expected_visible = any(
            recomputed_checkpoints.get(view_name, {}).get(
                checkpoint_name, False
            )
            for view_name in required_views
        )
        recomputed_checkpoint_aggregates[checkpoint_name] = expected_visible
        if visibility.get(field_name) is not expected_visible:
            result.fail(
                f"source_mujoco visibility QC aggregate {checkpoint_name} visibility changed"
            )
    _validate_source_mujoco_task_visibility_contract(
        result,
        source_scenario=source_scenario,
        recomputed_checkpoints=recomputed_checkpoints,
        thresholds=thresholds,
    )
    expected_critically_cropped = not recomputed_checkpoint_aggregates.get(
        "key_event", False
    )
    if visibility.get("critically_cropped") is not expected_critically_cropped:
        result.fail("source_mujoco visibility QC critical-crop aggregate changed")

    def replay_nonnegative_integer(
        mapping: Mapping[str, Any], field_name: str
    ) -> int | None:
        raw = mapping.get(field_name)
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or not float(raw).is_integer()
            or float(raw) < 0.0
        ):
            return None
        return int(raw)

    def replay_finite_number(
        mapping: Mapping[str, Any], field_name: str
    ) -> float | None:
        raw = mapping.get(field_name)
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        ):
            return None
        return float(raw)

    def replay_key_target_visible(mapping: Mapping[str, Any]) -> bool:
        object_pixels = replay_nonnegative_integer(mapping, "object_pixel_count")
        segmentation_margin = replay_finite_number(
            mapping, "segmentation_bbox_margin_px"
        )
        projected_margin = replay_finite_number(
            mapping, "projected_sphere_margin_px"
        )
        return bool(
            object_pixels is not None
            and object_pixels
            >= int(thresholds["minimum_key_event_object_area_px"])
            and segmentation_margin is not None
            and projected_margin is not None
            and min(segmentation_margin, projected_margin)
            >= float(thresholds["minimum_bbox_margin_px"])
        )

    planned_index = event_values.get("planned_index")
    actual_index = event_values.get("actual_index")
    actual_metrics: dict[str, Mapping[str, Any]] = {}
    planned_metrics: dict[str, Mapping[str, Any]] = {}
    for view_name in required_views:
        metrics = frame_metrics_by_view.get(view_name, ())
        if not isinstance(actual_index, int) or not (0 <= actual_index < len(metrics)):
            result.fail(
                f"source_mujoco visibility QC cannot replay {view_name} actual key frame"
            )
        else:
            actual_metrics[view_name] = metrics[actual_index]
        if not isinstance(planned_index, int) or not (
            0 <= planned_index < len(metrics)
        ):
            result.fail(
                f"source_mujoco visibility QC cannot replay {view_name} planned key frame"
            )
        else:
            planned_metrics[view_name] = metrics[planned_index]

    key_target_visible = {
        view_name: replay_key_target_visible(metric)
        for view_name, metric in actual_metrics.items()
    }
    key_tool_pixels = {
        view_name: replay_nonnegative_integer(metric, "tool_pixel_count")
        for view_name, metric in actual_metrics.items()
    }
    key_left_pixels = {
        view_name: replay_nonnegative_integer(metric, "left_tool_pixel_count")
        for view_name, metric in actual_metrics.items()
    }
    key_right_pixels = {
        view_name: replay_nonnegative_integer(metric, "right_tool_pixel_count")
        for view_name, metric in actual_metrics.items()
    }
    key_fixture_pixels = {
        view_name: replay_nonnegative_integer(metric, "fixture_pixel_count")
        for view_name, metric in actual_metrics.items()
    }
    key_geom_pixels = {
        view_name: (
            metric.get("geom_pixel_counts")
            if isinstance(metric.get("geom_pixel_counts"), Mapping)
            else {}
        )
        for view_name, metric in actual_metrics.items()
    }

    per_view_summary_fields = {
        "key_event_object_pixel_count": "object_pixel_count",
        "key_event_tool_pixel_count": "tool_pixel_count",
        "key_event_left_tool_pixel_count": "left_tool_pixel_count",
        "key_event_right_tool_pixel_count": "right_tool_pixel_count",
        "key_event_fixture_pixel_count": "fixture_pixel_count",
    }
    for view_name, metric in actual_metrics.items():
        view = views.get(view_name)
        if not isinstance(view, Mapping):
            continue
        for summary_name, metric_name in per_view_summary_fields.items():
            if view.get(summary_name) != metric.get(metric_name):
                result.fail(
                    f"source_mujoco visibility QC {view_name} {summary_name} changed"
                )
        segmentation_margin = replay_finite_number(
            metric, "segmentation_bbox_margin_px"
        )
        projected_margin = replay_finite_number(
            metric, "projected_sphere_margin_px"
        )
        if segmentation_margin is not None and projected_margin is not None:
            expected_key_margin = min(segmentation_margin, projected_margin)
            if view.get("key_event_bbox_margin_px") != expected_key_margin:
                result.fail(
                    f"source_mujoco visibility QC {view_name} key-event bbox summary changed"
                )
            recomputed_key_margins.append(expected_key_margin)
        object_pixels = replay_nonnegative_integer(metric, "object_pixel_count")
        if object_pixels is not None:
            recomputed_key_areas.append(object_pixels)

    if len(recomputed_key_margins) != len(required_views):
        result.fail("source_mujoco visibility QC cannot recompute key-event margins")
    elif minimum_margin is None or abs(
        float(minimum_margin) - max(recomputed_key_margins)
    ) > 1e-12:
        result.fail("source_mujoco visibility QC key-event margin aggregate changed")
    if len(recomputed_key_areas) != len(required_views):
        result.fail("source_mujoco visibility QC cannot recompute key-event target area")
    elif key_area is None or int(key_area) != max(recomputed_key_areas):
        result.fail("source_mujoco visibility QC key-event target area aggregate changed")

    physical_contact = visibility.get("physical_contact_applicable")
    if not isinstance(physical_contact, bool):
        result.fail("source_mujoco visibility QC lacks physical-contact applicability")

    def event_timestamp(row: Mapping[str, Any]) -> float | None:
        raw = row.get("timestamp")
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        ):
            return None
        return float(raw)

    valid_event_rows = [row for row in event_rows if isinstance(row, Mapping)]
    tool_contact_rows = [
        row
        for row in valid_event_rows
        if row.get("contact_category") in {"gripper", "robot_arm"}
        and event_timestamp(row) is not None
    ]
    expected_physical_contact: bool | None
    expected_contact_rows: list[Mapping[str, Any]] = []
    if actual_source == "persisted_bilateral_contact":
        expected_physical_contact = True
        if not tool_applicable:
            result.fail(
                "source_mujoco passive episode claims a persisted bilateral contact"
            )
        if actual_time is not None:
            expected_contact_rows = [
                row
                for row in tool_contact_rows
                if row.get("contact_category") == "gripper"
                and abs(float(event_timestamp(row)) - float(actual_time))
                <= float(event_time_tolerance_s) + 1e-12
            ]
        if not expected_contact_rows:
            result.fail(
                "source_mujoco bilateral event source lacks persisted contact rows"
            )
    elif actual_source == "persisted_contact_event":
        expected_physical_contact = True
        if not tool_applicable:
            result.fail(
                "source_mujoco passive episode claims a persisted tool contact"
            )
        if actual_time is not None:
            expected_contact_rows = [
                row
                for row in tool_contact_rows
                if abs(float(event_timestamp(row)) - float(actual_time)) <= 1e-12
            ]
        if not expected_contact_rows:
            result.fail(
                "source_mujoco contact event source lacks persisted contact rows"
            )
    elif actual_source == "persisted_task_surface_contact":
        expected_physical_contact = True
        if tool_applicable:
            result.fail(
                "source_mujoco actuated episode claims a passive task-surface contact"
            )
        if actual_time is not None:
            expected_contact_rows = [
                row
                for row in valid_event_rows
                if row.get("contact_category") == "task_surface"
                and event_timestamp(row) is not None
                and abs(float(event_timestamp(row)) - float(actual_time))
                <= float(event_time_tolerance_s) + 1e-12
            ]
        if not expected_contact_rows:
            result.fail(
                "source_mujoco task-surface event source lacks persisted contact rows"
            )
    elif actual_source == "persisted_free_flight_apex":
        expected_physical_contact = False
        if tool_applicable:
            result.fail(
                "source_mujoco actuated episode claims a passive free-flight apex"
            )
        if actual_name != "projectile_apex":
            result.fail(
                "source_mujoco free-flight apex source has the wrong event name"
            )
    elif actual_source == "planned_interception_for_measured_miss":
        expected_physical_contact = False
        if not tool_applicable:
            result.fail(
                "source_mujoco passive episode claims an actuated interception miss"
            )
        if tool_contact_rows:
            result.fail(
                "source_mujoco measured-miss event source conflicts with persisted tool contact"
            )
    elif actual_source == "planned_source_scenario_event":
        if tool_applicable:
            result.fail(
                "source_mujoco actuated episode claims a passive planned event source"
            )
        # The online selector deliberately treats a passive task-surface contact
        # within one 30 Hz video interval of the planned event as the physical
        # counterpart. Reproduce that rule from saved contacts instead of
        # trusting the summary boolean.
        if actual_time is not None:
            expected_contact_rows = [
                row
                for row in valid_event_rows
                if row.get("contact_category") == "task_surface"
                and event_timestamp(row) is not None
                and abs(float(event_timestamp(row)) - float(actual_time))
                <= 1.0 / 30.0 + 1e-12
            ]
        expected_physical_contact = bool(expected_contact_rows)
    else:
        expected_physical_contact = None
        result.fail("source_mujoco visibility QC has unsupported key-event source")

    if (
        expected_physical_contact is not None
        and physical_contact is not expected_physical_contact
    ):
        result.fail(
            "source_mujoco physical-contact applicability differs from persisted event evidence"
        )

    expected_contact_ids = sorted(
        {
            int(row["counterpart_geom_id"])
            for row in expected_contact_rows
            if isinstance(row.get("counterpart_geom_id"), int)
            and not isinstance(row.get("counterpart_geom_id"), bool)
            and int(row["counterpart_geom_id"]) >= 0
        }
    )
    counterpart_ids = visibility.get("contact_counterpart_geom_ids")
    if not isinstance(counterpart_ids, Sequence) or isinstance(counterpart_ids, (str, bytes, bytearray)) or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counterpart_ids if isinstance(counterpart_ids, Sequence) and not isinstance(counterpart_ids, (str, bytes, bytearray))
    ):
        result.fail("source_mujoco visibility QC has invalid contact counterpart geom IDs")
        counterpart_ids = []
    else:
        counterpart_ids = list(counterpart_ids)
    if len(counterpart_ids) != len(set(counterpart_ids)):
        result.fail("source_mujoco visibility QC contact counterpart geom IDs are duplicated")
    if counterpart_ids != expected_contact_ids:
        result.fail(
            "source_mujoco visibility contact geom IDs differ from saved contact evidence"
        )

    counterpart_threshold = int(thresholds["minimum_counterpart_area_px"])
    planned_counterpart_threshold = int(
        thresholds["minimum_planned_counterpart_area_px"]
    )
    expected_tool_visible: bool | None = None
    expected_fixture_visible: bool | None = None
    if tool_applicable:
        applicable_threshold = (
            counterpart_threshold
            if expected_physical_contact is True
            else planned_counterpart_threshold
        )
        expected_tool_visible = any(
            pixel_count is not None and pixel_count >= applicable_threshold
            for pixel_count in key_tool_pixels.values()
        )
        if visibility.get("tool_visible_at_key_event") is not expected_tool_visible:
            result.fail(
                "source_mujoco key-event tool visibility differs from per-frame segmentation"
            )
        expected_counterpart_visible = expected_tool_visible
    else:
        expected_fixture_visible = any(
            pixel_count is not None and pixel_count >= counterpart_threshold
            for pixel_count in key_fixture_pixels.values()
        )
        if (
            visibility.get("fixture_visible_at_key_event")
            is not expected_fixture_visible
        ):
            result.fail(
                "source_mujoco key-event fixture visibility differs from per-frame segmentation"
            )
        expected_counterpart_visible = expected_fixture_visible
    if (
        visibility.get("counterpart_visible_at_key_event")
        is not expected_counterpart_visible
    ):
        result.fail(
            "source_mujoco applicable counterpart visibility differs from per-frame segmentation"
        )

    expected_planned_covisible = False
    if len(planned_metrics) == len(required_views):
        for view_name, metric in planned_metrics.items():
            counterpart_pixels = replay_nonnegative_integer(
                metric,
                "tool_pixel_count" if tool_applicable else "fixture_pixel_count",
            )
            if (
                replay_key_target_visible(metric)
                and counterpart_pixels is not None
                and counterpart_pixels >= planned_counterpart_threshold
            ):
                expected_planned_covisible = True
                break
    if (
        visibility.get("planned_checkpoint_covisible_in_any_view")
        is not expected_planned_covisible
    ):
        result.fail(
            "source_mujoco planned checkpoint co-visibility differs from per-frame segmentation"
        )

    expected_bilateral_visible: bool | None = None
    expected_actual_contact_visible: bool | None = None
    expected_proxy_resolution: bool | None = None
    expected_contact_body_ids: list[int] = []
    expected_proxy_geom_ids: list[int] = []
    expected_visible_proxy_ids_by_view: dict[str, list[int]] = {}
    expected_proxy_pixels_by_view: dict[str, int] = {}
    expected_unresolved_contact_ids: list[int] = []
    expected_exact_fixture_ids: list[int] = []
    expected_visible_fixture_ids_by_view: dict[str, list[int]] = {}
    expected_fixture_pixels_by_view: dict[str, int] = {}
    left_set: set[int] = set()
    right_set: set[int] = set()
    if actual_source == "persisted_bilateral_contact":
        left_ids = visibility.get("left_tool_geom_ids")
        right_ids = visibility.get("right_tool_geom_ids")
        if not isinstance(left_ids, Sequence) or isinstance(
            left_ids, (str, bytes, bytearray)
        ):
            left_ids = []
        if not isinstance(right_ids, Sequence) or isinstance(
            right_ids, (str, bytes, bytearray)
        ):
            right_ids = []
        left_set = {
            value
            for value in left_ids
            if isinstance(value, int) and not isinstance(value, bool)
        }
        right_set = {
            value
            for value in right_ids
            if isinstance(value, int) and not isinstance(value, bool)
        }
        contacted_both_sides = bool(
            set(expected_contact_ids).intersection(left_set)
            and set(expected_contact_ids).intersection(right_set)
        )
        expected_bilateral_visible = bool(
            contacted_both_sides
            and any(
                key_target_visible.get(view_name, False)
                and key_left_pixels.get(view_name) is not None
                and int(key_left_pixels[view_name]) >= counterpart_threshold
                and key_right_pixels.get(view_name) is not None
                and int(key_right_pixels[view_name]) >= counterpart_threshold
                for view_name in required_views
            )
        )
        expected_actual_contact_visible = expected_bilateral_visible
    elif (
        expected_physical_contact is True
        and tool_applicable
        and actual_source == "persisted_contact_event"
    ):
        raw_body_map = visibility.get("tool_geom_body_ids")
        normalized_body_map: dict[int, int] = {}
        invalid_body_map = not isinstance(raw_body_map, Mapping)
        if isinstance(raw_body_map, Mapping):
            for raw_geom_id, raw_body_id in raw_body_map.items():
                try:
                    if isinstance(raw_geom_id, bool) or isinstance(
                        raw_body_id, bool
                    ):
                        raise ValueError
                    geom_id = int(raw_geom_id)
                    body_id = int(raw_body_id)
                    if str(geom_id) != str(raw_geom_id) or geom_id < 0 or body_id < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    invalid_body_map = True
                    continue
                normalized_body_map[geom_id] = body_id
        if invalid_body_map:
            result.fail("source_mujoco contact proxy has an invalid geom/body map")
        expected_unresolved_contact_ids = sorted(
            geom_id
            for geom_id in expected_contact_ids
            if geom_id not in normalized_body_map
        )
        expected_proxy_resolution = bool(expected_contact_ids) and not (
            invalid_body_map or expected_unresolved_contact_ids
        )
        if expected_proxy_resolution:
            expected_contact_body_ids = sorted(
                {
                    normalized_body_map[geom_id]
                    for geom_id in expected_contact_ids
                }
            )
            expected_proxy_geom_ids = sorted(
                geom_id
                for geom_id, body_id in normalized_body_map.items()
                if body_id in expected_contact_body_ids
            )
        for view_name in required_views:
            pixels = key_geom_pixels.get(view_name, {})
            visible_proxy_ids: list[int] = []
            proxy_pixel_count = 0
            for geom_id in expected_proxy_geom_ids:
                raw_count = pixels.get(str(geom_id), 0)
                if (
                    not isinstance(raw_count, int)
                    or isinstance(raw_count, bool)
                    or raw_count < 0
                ):
                    result.fail(
                        "source_mujoco contact proxy has invalid per-geom pixels"
                    )
                    continue
                proxy_pixel_count += raw_count
                if raw_count > 0:
                    visible_proxy_ids.append(geom_id)
            expected_visible_proxy_ids_by_view[view_name] = visible_proxy_ids
            expected_proxy_pixels_by_view[view_name] = proxy_pixel_count
        expected_actual_contact_visible = bool(
            expected_proxy_resolution
            and any(
                key_target_visible.get(view_name, False)
                and expected_proxy_pixels_by_view.get(view_name, 0)
                >= counterpart_threshold
                for view_name in required_views
            )
        )
    elif expected_physical_contact is True:
        # Passive task fixtures are rendered physical geoms, so their exact
        # persisted contact IDs—not the robot-body proxy map—are authoritative.
        expected_exact_fixture_ids = list(expected_contact_ids)
        for view_name in required_views:
            pixels = key_geom_pixels.get(view_name, {})
            visible_fixture_ids: list[int] = []
            fixture_pixel_count = 0
            for geom_id in expected_exact_fixture_ids:
                raw_count = pixels.get(str(geom_id), 0)
                if (
                    not isinstance(raw_count, int)
                    or isinstance(raw_count, bool)
                    or raw_count < 0
                ):
                    result.fail(
                        "source_mujoco fixture contact has invalid per-geom pixels"
                    )
                    continue
                fixture_pixel_count += raw_count
                if raw_count > 0:
                    visible_fixture_ids.append(geom_id)
            expected_visible_fixture_ids_by_view[view_name] = (
                visible_fixture_ids
            )
            expected_fixture_pixels_by_view[view_name] = fixture_pixel_count
        expected_actual_contact_visible = bool(
            expected_exact_fixture_ids
            and any(
                key_target_visible.get(view_name, False)
                and expected_fixture_pixels_by_view.get(view_name, 0)
                >= counterpart_threshold
                for view_name in required_views
            )
        )

    proxy_claims = {
        "contact_body_proxy_resolution_complete": expected_proxy_resolution,
        "contact_counterpart_body_ids": expected_contact_body_ids,
        "contact_proxy_geom_ids": expected_proxy_geom_ids,
        "contact_visible_proxy_geom_ids_by_view": (
            expected_visible_proxy_ids_by_view
        ),
        "contact_proxy_pixel_counts_by_view": expected_proxy_pixels_by_view,
        "unresolved_contact_counterpart_geom_ids": (
            expected_unresolved_contact_ids
        ),
    }
    for field_name, expected_value in proxy_claims.items():
        if visibility.get(field_name) != expected_value:
            result.fail(
                f"source_mujoco {field_name} differs from contact-body proxy replay"
            )
    fixture_contact_claims = {
        "contact_exact_fixture_geom_ids": expected_exact_fixture_ids,
        "contact_visible_fixture_geom_ids_by_view": (
            expected_visible_fixture_ids_by_view
        ),
        "contact_fixture_pixel_counts_by_view": (
            expected_fixture_pixels_by_view
        ),
    }
    for field_name, expected_value in fixture_contact_claims.items():
        if visibility.get(field_name) != expected_value:
            result.fail(
                f"source_mujoco {field_name} differs from exact fixture-contact replay"
            )

    if expected_physical_contact is True:
        if not expected_contact_ids:
            result.fail("source_mujoco physical contact lacks persisted counterpart geom IDs")
        if (
            visibility.get("actual_contact_counterpart_visible_at_key_event")
            is not expected_actual_contact_visible
        ):
            result.fail(
                "source_mujoco actual-contact visibility differs from per-frame segmentation"
            )
        if visibility.get("contact_occluded_both_views") is not (
            not bool(expected_actual_contact_visible)
        ):
            result.fail(
                "source_mujoco contact-occlusion claim differs from per-frame segmentation"
            )
        if expected_actual_contact_visible is not True:
            result.fail(
                "source_mujoco actual contact counterpart is not visible at the key event"
            )
    elif expected_physical_contact is False:
        if (
            visibility.get("actual_contact_counterpart_visible_at_key_event")
            is not None
        ):
            result.fail("source_mujoco no-contact event claims a visible physical contact")
        if visibility.get("contact_occluded_both_views") is not None:
            result.fail("source_mujoco no-contact event claims physical-contact occlusion")
        if expected_planned_covisible is not True:
            result.fail("source_mujoco no-contact event lacks planned object/counterpart co-visibility")

    if actual_source == "persisted_bilateral_contact":
        geom_body_ids = visibility.get("tool_geom_body_ids")
        left_body_id = visibility.get("left_tool_body_id")
        right_body_id = visibility.get("right_tool_body_id")
        if not left_set or not right_set or left_set.intersection(right_set):
            result.fail("source_mujoco bilateral visibility lacks distinct left/right geom identities")
        if (
            not isinstance(geom_body_ids, Mapping)
            or not isinstance(left_body_id, int)
            or isinstance(left_body_id, bool)
            or not isinstance(right_body_id, int)
            or isinstance(right_body_id, bool)
            or left_body_id == right_body_id
            or any(geom_body_ids.get(str(geom_id)) != left_body_id for geom_id in left_set)
            or any(geom_body_ids.get(str(geom_id)) != right_body_id for geom_id in right_set)
        ):
            result.fail("source_mujoco bilateral contact geoms are not bound to distinct visible tool bodies")
        if not set(expected_contact_ids).intersection(left_set) or not set(
            expected_contact_ids
        ).intersection(right_set):
            result.fail("source_mujoco bilateral contact evidence does not bind both tool sides")
        if (
            visibility.get("bilateral_tool_sides_visible_at_key_event")
            is not expected_bilateral_visible
        ):
            result.fail(
                "source_mujoco bilateral visibility claim differs from per-frame side masks"
            )
        if expected_bilateral_visible is not True:
            result.fail("source_mujoco opposing bilateral contacts are not visible")
    elif visibility.get("bilateral_tool_sides_visible_at_key_event") is not None:
        result.fail("source_mujoco non-bilateral event claims bilateral visibility")


def _validate_source_mujoco_background_clearance(
    result: EpisodeQC,
    clearance: Mapping[str, Any],
    *,
    high_rate_rows: Sequence[Mapping[str, Any]],
    source_scenario: Mapping[str, Any] | None,
    backend_provenance: Mapping[str, Any] | None,
    runtime_audit: Mapping[str, Any] | None,
    stored_sha256: Any,
) -> None:
    """Bind runtime background clearance to the complete persisted trajectory."""

    schema = "source-mujoco-background-clearance/v4"
    if clearance.get("schema_version") != schema:
        result.fail("source_mujoco background clearance schema changed")
    for name in (
        "evaluated",
        "clearance_pass",
        "all_background_collision_disabled",
        "all_background_anchored",
        "object_swept_volume_clear",
        "fixture_intersection_clear",
        "all_physical_fixtures_anchored",
        "all_physical_fixtures_collision_enabled",
        "structural_support_chain_pass",
        "structural_support_swept_volume_clear",
    ):
        if clearance.get(name) is not True:
            result.fail(f"source_mujoco background clearance failed {name}")
    background_rows = clearance.get("background_rows")
    fixture_rows = clearance.get("fixture_rows")
    if not isinstance(background_rows, Sequence) or isinstance(
        background_rows, (str, bytes, bytearray)
    ):
        result.fail("source_mujoco background clearance rows are malformed")
        background_row_values: list[Mapping[str, Any]] | None = None
    elif not all(isinstance(row, Mapping) for row in background_rows):
        result.fail("source_mujoco background clearance rows contain malformed entries")
        background_row_values = None
    elif clearance.get("background_rows_sha256") != sha256_json(background_rows):
        result.fail("source_mujoco background clearance row hash changed")
        background_row_values = list(background_rows)  # type: ignore[list-item]
    else:
        background_row_values = list(background_rows)  # type: ignore[list-item]
    if not isinstance(fixture_rows, Sequence) or isinstance(
        fixture_rows, (str, bytes, bytearray)
    ):
        result.fail("source_mujoco fixture clearance rows are malformed")
        fixture_row_values: list[Mapping[str, Any]] | None = None
    elif not all(isinstance(row, Mapping) for row in fixture_rows):
        result.fail("source_mujoco fixture clearance rows contain malformed entries")
        fixture_row_values = None
    elif clearance.get("fixture_rows_sha256") != sha256_json(fixture_rows):
        result.fail("source_mujoco fixture clearance row hash changed")
        fixture_row_values = list(fixture_rows)  # type: ignore[list-item]
    else:
        fixture_row_values = list(fixture_rows)  # type: ignore[list-item]
    object_sweep = clearance.get("object_sweep")
    if not isinstance(object_sweep, Mapping):
        result.fail("source_mujoco background clearance lacks object sweep")
    else:
        sample_count = object_sweep.get("sample_count")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count != len(high_rate_rows)
        ):
            result.fail(
                "source_mujoco background sweep sample count differs from high-rate Parquet"
            )
        try:
            exact_rows = [
                {
                    "sample_index": index,
                    "timestamp_s": float(row["timestamp"]),
                    "object_position_m": [
                        float(value) for value in row["object.position"]
                    ],
                }
                for index, row in enumerate(high_rate_rows)
            ]
            if any(
                len(row["object_position_m"]) != 3
                or not math.isfinite(row["timestamp_s"])
                or not all(
                    math.isfinite(value) for value in row["object_position_m"]
                )
                for row in exact_rows
            ):
                raise ValueError("non-finite sweep row")
            if object_sweep.get("exact_rows_sha256") != sha256_json(exact_rows):
                result.fail(
                    "source_mujoco background sweep differs from persisted high-rate trajectory"
                )
        except (KeyError, TypeError, ValueError):
            result.fail(
                "source_mujoco background sweep cannot be replayed from high-rate Parquet"
            )

    physics = (
        source_scenario.get("physics")
        if isinstance(source_scenario, Mapping)
        else None
    )
    try:
        if not isinstance(physics, Mapping):
            raise TypeError("missing scenario physics")
        object_radius_m = float(physics["object_radius_m"])
        if not math.isfinite(object_radius_m) or object_radius_m <= 0.0:
            raise ValueError("invalid object radius")
    except (KeyError, TypeError, ValueError):
        object_radius_m = math.nan
        result.fail(
            "source_mujoco background clearance lacks its SourceScenarioSpec object radius"
        )
    if isinstance(object_sweep, Mapping):
        try:
            stored_radius = float(object_sweep["object_radius_m"])
        except (KeyError, TypeError, ValueError):
            stored_radius = math.nan
        if not math.isfinite(stored_radius) or stored_radius != object_radius_m:
            result.fail(
                "source_mujoco background sweep object radius differs from SourceScenarioSpec"
            )

    classification = clearance.get("classification")
    if not isinstance(classification, Mapping):
        result.fail("source_mujoco background clearance lacks its static classification")
    if background_row_values is not None:
        stable_ids = [str(row.get("stable_id") or "") for row in background_row_values]
        if any(not value for value in stable_ids) or len(set(stable_ids)) != len(stable_ids):
            result.fail("source_mujoco background clearance stable IDs are invalid")
        background_static_fields = (
            "stable_id",
            "geom_id",
            "classification",
            "source_name",
            "catalog_slot",
            "fixture_support_id",
            "body_id",
            "body_name",
            "body_weld_id",
            "world_aabb",
            "contype",
            "conaffinity",
        )
        background_static_rows = sorted(
            (
                {name: row.get(name) for name in background_static_fields}
                for row in background_row_values
            ),
            key=lambda row: str(row["stable_id"]),
        )
        descriptor_rows = [
            {
                name: row[name]
                for name in (
                    "stable_id",
                    "geom_id",
                    "classification",
                    "source_name",
                    "catalog_slot",
                    "fixture_support_id",
                )
            }
            for row in background_static_rows
        ]
        if any(
            not isinstance(row.get("world_aabb"), Mapping)
            or row["world_aabb"].get("method")
            != "mujoco_compiled_local_aabb_transformed/v1"
            for row in background_static_rows
        ):
            result.fail("source_mujoco background clearance AABB method changed")
        background_static_sha256 = sha256_json(background_static_rows)
        if isinstance(classification, Mapping) and (
            classification.get("descriptor_count") != len(background_static_rows)
            or classification.get("descriptors_sha256") != sha256_json(descriptor_rows)
            or classification.get("background_static_rows_sha256")
            != background_static_sha256
        ):
            result.fail(
                "source_mujoco background static classification cannot be replayed"
            )
        if not isinstance(physics, Mapping) or (
            physics.get("background_clearance_static_row_count")
            != len(background_static_rows)
            or physics.get("background_clearance_static_rows_sha256")
            != background_static_sha256
        ):
            result.fail(
                "source_mujoco background geometry differs from SourceScenarioSpec"
            )

    if fixture_row_values is not None:
        fixture_ids = [str(row.get("fixture_id") or "") for row in fixture_row_values]
        if any(not value for value in fixture_ids) or len(set(fixture_ids)) != len(fixture_ids):
            result.fail("source_mujoco fixture clearance stable IDs are invalid")
        if any(
            not isinstance(row.get("world_aabb"), Mapping)
            or row["world_aabb"].get("method")
            != "mujoco_compiled_local_aabb_transformed/v1"
            for row in fixture_row_values
        ):
            result.fail("source_mujoco fixture clearance AABB method changed")
        fixture_static_fields = (
            "fixture_id",
            "role",
            "geom_id",
            "fixture_class",
            "expected_task_contact",
            "supports_fixture_id",
            "grounded_fixture_id",
            "body_id",
            "body_weld_id",
            "contype",
            "conaffinity",
            "ground_contact_distance_m",
            "supported_contact_distance_m",
            "support_interface_maximum_mismatch_m",
            "support_interface_tolerance_m",
            "world_aabb",
        )
        normalized_fixture_rows = sorted(
            (
                {name: row.get(name) for name in fixture_static_fields}
                for row in fixture_row_values
            ),
            key=lambda row: str(row["fixture_id"]),
        )
        fixture_static_sha256 = sha256_json(normalized_fixture_rows)
        if isinstance(classification, Mapping) and classification.get(
            "fixture_static_rows_sha256"
        ) != fixture_static_sha256:
            result.fail(
                "source_mujoco fixture static classification cannot be replayed"
            )
        if not isinstance(physics, Mapping) or (
            physics.get("fixture_clearance_static_row_count")
            != len(normalized_fixture_rows)
            or physics.get("fixture_clearance_static_rows_sha256")
            != fixture_static_sha256
        ):
            result.fail(
                "source_mujoco fixture geometry differs from SourceScenarioSpec"
            )

        source_fixtures = (
            source_scenario.get("fixtures")
            if isinstance(source_scenario, Mapping)
            else None
        )
        if not isinstance(source_fixtures, Sequence) or isinstance(
            source_fixtures, (str, bytes, bytearray)
        ):
            result.fail(
                "source_mujoco structural support contract lacks SourceScenarioSpec fixtures"
            )
        else:
            expected_fixtures: dict[str, Mapping[str, Any]] = {}
            malformed_source_fixture = False
            for raw_fixture in source_fixtures:
                if not isinstance(raw_fixture, Mapping):
                    malformed_source_fixture = True
                    continue
                fixture_id = str(raw_fixture.get("fixture_id") or "")
                if not fixture_id or fixture_id in expected_fixtures:
                    malformed_source_fixture = True
                    continue
                expected_fixtures[fixture_id] = raw_fixture
            if malformed_source_fixture or set(expected_fixtures) != set(
                fixture_ids
            ):
                result.fail(
                    "source_mujoco fixture identities differ from SourceScenarioSpec"
                )
            for row in fixture_row_values:
                fixture_id = str(row.get("fixture_id") or "")
                expected = expected_fixtures.get(fixture_id)
                if expected is None:
                    continue
                parameters = expected.get("parameters")
                if not isinstance(parameters, Mapping):
                    result.fail(
                        f"source_mujoco fixture {fixture_id} lacks bound parameters"
                    )
                    continue
                expected_values = {
                    "role": expected.get("fixture_type"),
                    "fixture_class": parameters.get("fixture_class"),
                    "expected_task_contact": parameters.get(
                        "expected_task_contact"
                    ),
                    "supports_fixture_id": parameters.get(
                        "supports_fixture_id"
                    ),
                    "grounded_fixture_id": (
                        "floor"
                        if parameters.get("fixture_class")
                        == "structural_support"
                        else None
                    ),
                    "support_interface_maximum_mismatch_m": parameters.get(
                        "support_interface_maximum_mismatch_m"
                    ),
                    "support_interface_tolerance_m": parameters.get(
                        "support_interface_tolerance_m"
                    ),
                }
                if any(
                    row.get(name) != value
                    for name, value in expected_values.items()
                ):
                    result.fail(
                        f"source_mujoco fixture {fixture_id} semantics differ "
                        "from SourceScenarioSpec"
                    )
                expected_anchored = expected.get("anchored") is True
                expected_physical = expected.get("physical") is True
                expected_collision = parameters.get("collision_enabled") is True
                derived_anchored = row.get("body_weld_id") == 0
                derived_collision = bool(
                    isinstance(row.get("contype"), int)
                    and not isinstance(row.get("contype"), bool)
                    and int(row["contype"]) > 0
                    and isinstance(row.get("conaffinity"), int)
                    and not isinstance(row.get("conaffinity"), bool)
                    and int(row["conaffinity"]) > 0
                )
                if not expected_anchored or derived_anchored is not True:
                    result.fail(
                        f"source_mujoco fixture {fixture_id} is not physically anchored"
                    )
                if (
                    not expected_physical
                    or not expected_collision
                    or derived_collision is not True
                ):
                    result.fail(
                        f"source_mujoco fixture {fixture_id} is not a physical collision fixture"
                    )

            structural_ids = sorted(
                fixture_id
                for fixture_id, expected in expected_fixtures.items()
                if isinstance(expected.get("parameters"), Mapping)
                and expected["parameters"].get("fixture_class")
                == "structural_support"
            )
            task_contact_ids = sorted(
                fixture_id
                for fixture_id, expected in expected_fixtures.items()
                if isinstance(expected.get("parameters"), Mapping)
                and expected["parameters"].get("expected_task_contact") is True
            )
            if set(structural_ids).intersection(task_contact_ids):
                result.fail(
                    "source_mujoco structural supports are expected task contacts"
                )
            if (
                clearance.get("structural_support_fixture_ids")
                != structural_ids
                or clearance.get("structural_support_count")
                != len(structural_ids)
                or clearance.get("task_contact_fixture_ids")
                != task_contact_ids
            ):
                result.fail(
                    "source_mujoco structural support identity aggregates changed"
                )
            if isinstance(physics, Mapping) and (
                physics.get("structural_support_fixture_ids")
                != structural_ids
                or physics.get("task_contact_fixture_ids")
                != task_contact_ids
            ):
                result.fail(
                    "source_mujoco structural support identities differ from "
                    "SourceScenarioSpec physics"
                )

    if isinstance(classification, Mapping):
        exclusions = classification.get("exclusions")
        if not isinstance(exclusions, Mapping) or classification.get(
            "exclusions_sha256"
        ) != sha256_json(exclusions):
            result.fail("source_mujoco background exclusion binding changed")

    if (
        background_row_values is not None
        and fixture_row_values is not None
        and math.isfinite(object_radius_m)
    ):
        try:
            # This pure replay recomputes AABBs over the exact persisted
            # trajectory, collision/anchoring claims from primitive model
            # fields, per-row intersections, aggregates, and all failure IDs.
            from ..backends.source_mujoco.backend import (
                _evaluate_background_clearance_rows,
            )

            recomputed = _evaluate_background_clearance_rows(
                background_rows=background_row_values,
                fixture_rows=fixture_row_values,
                high_rate_rows=high_rate_rows,
                object_radius_m=object_radius_m,
            )
            replay_mismatches = sorted(
                key
                for key, value in recomputed.items()
                if clearance.get(key) != value
            )
            if replay_mismatches:
                result.fail(
                    "source_mujoco background clearance replay changed: "
                    + ", ".join(replay_mismatches)
                )
            result.metrics["background_clearance_replay"] = {
                "replay_match": not replay_mismatches,
                "object_radius_m": object_radius_m,
                "background_geom_count": len(background_row_values),
                "fixture_count": len(fixture_row_values),
            }
        except Exception as error:
            result.fail(f"source_mujoco background clearance replay: {error}")
    digest = sha256_json(clearance)
    if stored_sha256 != digest:
        result.fail("source_mujoco background clearance hash binding changed")
    if not isinstance(backend_provenance, Mapping) or (
        backend_provenance.get("background_clearance_sha256") != digest
        or backend_provenance.get("background_clearance") != clearance
    ):
        result.fail("source_mujoco provenance background clearance binding changed")
    if not isinstance(runtime_audit, Mapping) or runtime_audit.get(
        "background_clearance_sha256"
    ) != digest:
        result.fail("source_mujoco runtime-audit background clearance binding changed")


@dataclass(slots=True)
class DatasetQCReport:
    """Full-dataset QC result with release-aware pass semantics."""

    dataset_root: str
    episodes: list[EpisodeQC]
    global_failures: list[str] = field(default_factory=list)
    global_warnings: list[str] = field(default_factory=list)
    exact_duplicate_groups: list[list[str]] = field(default_factory=list)
    perceptual_duplicate_pairs: list[tuple[str, str, int]] = field(default_factory=list)
    strict_all: bool = False

    @property
    def passed(self) -> bool:
        return not self.global_failures and all(
            result.passed or (not self.strict_all and not result.release_eligible)
            for result in self.episodes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "passed": self.passed,
            "episode_count": len(self.episodes),
            "release_eligible_count": sum(result.release_eligible for result in self.episodes),
            "strict_all": self.strict_all,
            "failed_episode_count": sum(not result.passed for result in self.episodes),
            "failed_release_eligible_count": sum(
                result.release_eligible and not result.passed for result in self.episodes
            ),
            "global_failures": self.global_failures,
            "global_warnings": self.global_warnings,
            "exact_duplicate_groups": self.exact_duplicate_groups,
            "perceptual_duplicate_pairs": self.perceptual_duplicate_pairs,
            "episodes": [result.to_dict() for result in self.episodes],
        }


def exact_duplicate_split_leakage(
    duplicate_groups: Sequence[Sequence[str]], split_by_uuid: Mapping[str, str]
) -> list[str]:
    """Describe byte-identical video groups assigned to multiple splits."""

    problems: list[str] = []
    for group in duplicate_groups:
        episode_uuids = {item.split(":", 1)[0] for item in group}
        splits = {
            split_by_uuid[episode_uuid]
            for episode_uuid in episode_uuids
            if episode_uuid in split_by_uuid
        }
        if len(splits) > 1:
            problems.append(
                f"exact duplicate video group crosses splits {sorted(splits)}: {list(group)}"
            )
    return sorted(problems)


def _frame_signature(frame: bytes, width: int, height: int) -> int:
    luminance: list[int] = []
    for row in range(8):
        y = min(height - 1, int((row + 0.5) * height / 8))
        for column in range(8):
            x = min(width - 1, int((column + 0.5) * width / 8))
            offset = (y * width + x) * 3
            r, g, b = frame[offset], frame[offset + 1], frame[offset + 2]
            luminance.append((77 * r + 150 * g + 29 * b) >> 8)
    mean = sum(luminance) / len(luminance)
    bits = 0
    for value in luminance:
        bits = (bits << 1) | int(value >= mean)
    return bits


def _luma_grid(frame: bytes, width: int, height: int, columns: int = 64, rows: int = 36) -> list[int]:
    """Downsample a decoded RGB frame on an aligned spatial grid."""

    try:
        from PIL import Image

        image = Image.frombytes("RGB", (width, height), frame)
        return list(image.resize((columns, rows), Image.Resampling.BOX).convert("L").tobytes())
    except ImportError:
        pass
    result: list[int] = []
    for row in range(rows):
        y = min(height - 1, int((row + 0.5) * height / rows))
        for column in range(columns):
            x = min(width - 1, int((column + 0.5) * width / columns))
            offset = (y * width + x) * 3
            r, g, b = frame[offset], frame[offset + 1], frame[offset + 2]
            result.append((77 * r + 150 * g + 29 * b) >> 8)
    return result


def video_perceptual_hash(
    path: str | Path, probe: VideoProbe | None = None
) -> tuple[str, float, float, dict[str, float]]:
    """Hash temporal samples and estimate motion, black, and exposure metrics."""

    metadata = probe or probe_video(path)
    targets = {round((metadata.frame_count - 1) * fraction) for fraction in (0, 0.25, 0.5, 0.75, 1.0)}
    hashes: list[int] = []
    previous: list[int] | None = None
    differences: list[tuple[float, float]] = []
    tail_frames: list[list[int]] = []
    black_samples = 0
    white_samples = 0
    pixel_samples = 0
    decoded_frame_count = 0
    for index, frame in enumerate(iter_rgb_frames(path, metadata.width, metadata.height)):
        decoded_frame_count = index + 1
        if index in targets:
            hashes.append(_frame_signature(frame, metadata.width, metadata.height))
        grid = _luma_grid(frame, metadata.width, metadata.height)
        if previous is not None:
            deltas = [abs(left - right) for left, right in zip(grid, previous)]
            differences.append((sum(deltas) / len(deltas), sum(value >= 3 for value in deltas) / len(deltas)))
        previous = grid
        for luminance in grid:
            black_samples += int(luminance < 12)
            white_samples += int(luminance > 245)
            pixel_samples += 1
        tail_frames.append(grid)
        if len(tail_frames) > 12:
            tail_frames.pop(0)
    digest = "".join(f"{value:016x}" for value in hashes)
    frozen_fraction = (
        sum(mad < 0.08 and changed < 0.0002 for mad, changed in differences) / len(differences)
        if differences
        else 1.0
    )
    tail_differences: list[tuple[float, float]] = []
    for left, right in zip(tail_frames, tail_frames[1:]):
        deltas = [abs(left_value - right_value) for left_value, right_value in zip(left, right)]
        tail_differences.append((sum(deltas) / len(deltas), sum(value >= 3 for value in deltas) / len(deltas)))
    static_tail_fraction = (
        sum(mad < 0.08 and changed < 0.0002 for mad, changed in tail_differences) / len(tail_differences)
        if tail_differences
        else 1.0
    )
    visual = {
        "black_pixel_fraction": black_samples / pixel_samples if pixel_samples else 1.0,
        "overexposed_pixel_fraction": white_samples / pixel_samples if pixel_samples else 1.0,
        "mean_changed_grid_fraction": (
            sum(changed for _, changed in differences) / len(differences) if differences else 0.0
        ),
        "decoded_frame_count": float(decoded_frame_count),
    }
    return digest, frozen_fraction, static_tail_fraction, visual


class QCValidator:
    """Validate canonical metadata, files, encoded clocks, hashes, labels, and splits."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        deep_video_checks: bool = True,
        objective_evaluators: ObjectiveEvaluatorRegistry | None = None,
        strict_all: bool = False,
    ):
        self.root = Path(dataset_root).resolve(strict=True)
        self.deep_video_checks = deep_video_checks
        self.objective_evaluators = objective_evaluators or DEFAULT_OBJECTIVE_EVALUATORS
        self.strict_all = strict_all

    def _validate_episode(self, record: EpisodeRecord) -> tuple[EpisodeQC, dict[str, str]]:
        result = EpisodeQC(record.episode_uuid, record.episode_index, record.release_eligible)
        video_hashes: dict[str, str] = {}
        frame_rows: list[dict[str, Any]] = []
        high_rate_rows: list[dict[str, Any]] = []
        transition_rows: list[dict[str, Any]] = []
        object_state_rows: list[dict[str, Any]] = []
        source_scenario = record.extras.get("source_scenario_spec")
        source_embodiment = (
            source_scenario.get("embodiment")
            if isinstance(source_scenario, Mapping)
            else None
        )
        end_effector = str(
            record.extras.get("end_effector")
            or (
                source_embodiment.get("end_effector")
                if isinstance(source_embodiment, Mapping)
                else ""
            )
            or record.tool_type
            or ""
        )
        backend_provenance = record.extras.get("backend_provenance")
        backend_name = str(
            (
                backend_provenance.get("backend")
                if isinstance(backend_provenance, Mapping)
                else None
            )
            or record.extras.get("backend")
            or (
                source_scenario.get("backend")
                if isinstance(source_scenario, Mapping)
                else None
            )
            or ""
        ).strip().lower()
        source_mujoco_backend = backend_name == "source_mujoco"
        source_rigid_backend = source_mujoco_backend and (
            end_effector in PRODUCTION_END_EFFECTORS or end_effector == "no_robot"
        )
        release_claimed = (
            record.label_status.value == "verified_objective"
            and record.release_tier == ReleaseTier.FREE_CONTACT
            and record.dynamics_mode == DynamicsMode.FREE_CONTACT
            and record.physics_qc_pass
            and not record.quality_flags
        )
        try:
            record.validate()
        except (SchemaValidationError, ValueError) as error:
            result.fail(f"schema: {error}")
        required_cameras = {
            "observation.images.main",
            "observation.images.secondary",
        }
        missing_cameras = sorted(required_cameras - set(record.video_paths))
        if missing_cameras:
            result.fail(f"missing required synchronized camera streams: {missing_cameras}")
        expected_calibration_ids = set(record.camera_stream_calibration_ids.values())
        if set(record.camera_ids) != expected_calibration_ids:
            result.fail("camera_ids disagree with camera_stream_calibration_ids")
        pts: dict[str, Sequence[float]] = {}
        perceptual: dict[str, str] = {}
        for camera, relative in record.video_paths.items():
            try:
                path = resolve_dataset_path(self.root, relative)
                if not path.is_file():
                    raise FileNotFoundError(path)
                digest = sha256_file(path)
                video_hashes[camera] = digest
                expected = record.content_hashes.get(relative)
                if not expected:
                    result.fail(f"missing content hash: {relative}")
                elif expected != digest:
                    result.fail(f"content hash mismatch: {relative}")
                probe = probe_video(path)
                validate_video_probe(probe, VideoSpec(), expected_frames=record.frame_count)
                if record.frame_count is None:
                    result.fail("frame_count is required")
                if record.duration_s is None:
                    result.fail("duration_s is required")
                elif abs(record.duration_s - probe.duration_s) > 1.0 / 30.0 + 1e-6:
                    result.fail(
                        f"record duration {record.duration_s}s differs from video {probe.duration_s}s: {camera}"
                    )
                pts[camera] = probe_frame_timestamps(path)
                result.metrics[f"{camera}.duration_s"] = probe.duration_s
                if self.deep_video_checks:
                    signature, frozen, static_tail, visual = video_perceptual_hash(path, probe)
                    perceptual[camera] = signature
                    result.metrics[f"{camera}.perceptual_hash"] = signature
                    result.metrics[f"{camera}.frozen_transition_fraction"] = frozen
                    result.metrics[f"{camera}.static_tail_fraction"] = static_tail
                    result.metrics.update({f"{camera}.{key}": value for key, value in visual.items()})
                    if int(visual["decoded_frame_count"]) != probe.frame_count:
                        result.fail(
                            f"decoded frame count {int(visual['decoded_frame_count'])} != container metadata {probe.frame_count}: {camera}"
                        )
                    if frozen > 0.95:
                        result.fail(f"frozen video: {camera}")
                    if static_tail > 0.90:
                        result.warnings.append(f"long static tail: {camera}")
                    if visual["black_pixel_fraction"] > 0.50:
                        result.fail(f"large black void: {camera}")
                    elif visual["black_pixel_fraction"] > 0.30:
                        result.warnings.append(f"substantial black region: {camera}")
                    if visual["overexposed_pixel_fraction"] > 0.50:
                        result.fail(f"severe overexposure: {camera}")
                    elif visual["overexposed_pixel_fraction"] > 0.30:
                        result.warnings.append(f"substantial overexposure: {camera}")
            except Exception as error:  # preserve every episode result rather than aborting dataset QC
                result.fail(f"video {camera}: {error}")
        if len(pts) >= 2:
            try:
                validate_synchronized_streams(pts, tolerance_s=1e-6)
            except SynchronizationError as error:
                result.fail(f"camera synchronization: {error}")

        if record.frame_data_path:
            try:
                frame_path = resolve_dataset_path(self.root, record.frame_data_path)
                expected_frame_hash = record.content_hashes.get(record.frame_data_path)
                if not expected_frame_hash:
                    result.fail(f"missing content hash: {record.frame_data_path}")
                elif sha256_file(frame_path) != expected_frame_hash:
                    result.fail(f"content hash mismatch: {record.frame_data_path}")
                rows = read_parquet_rows(frame_path)
                frame_rows = rows
                timestamps = [float(row["timestamp"]) for row in rows]
                validate_monotonic_timestamps(timestamps)
                source_scenario = record.extras.get("source_scenario_spec")
                if isinstance(source_scenario, Mapping):
                    source_physics = source_scenario.get("physics")
                    if not isinstance(source_physics, Mapping):
                        raise ValueError(
                            "persisted source scenario lacks its physics mapping"
                        )
                    simulation_hz = float(
                        source_physics.get(
                            "simulation_hz", source_physics.get("sim_hz")
                        )
                    )
                    validate_persisted_render_schedule(
                        rows,
                        duration_s=float(source_scenario["duration_s"]),
                        maximum_sample_error_s=1.0 / simulation_hz + 1e-12,
                    )
                    result.metrics["maximum_simulation_sample_error_s"] = max(
                        float(row["synchronization_error_s"]) for row in rows
                    )
                if record.frame_count is None:
                    result.fail("frame_count is required")
                elif len(rows) != record.frame_count:
                    result.fail(f"frame table count {len(rows)} != metadata {record.frame_count}")
                if pts:
                    reference = next(iter(pts.values()))
                    if len(reference) != len(timestamps):
                        result.fail(
                            f"frame table count {len(timestamps)} != decoded PTS count {len(reference)}"
                        )
                    else:
                        worst = max(abs(left - right) for left, right in zip(reference, timestamps))
                        result.metrics["maximum_pts_error_s"] = worst
                        if worst > 1e-6:
                            result.fail(f"Parquet timestamps differ from PTS by {worst}s")
                if record.task_index is None:
                    result.fail("task_index is required")
                elif any(int(row.get("task_index", -1)) != record.task_index for row in rows):
                    result.fail("frame-table task_index disagrees with episode metadata")
                result.metrics["derived_action_hash"] = derived_action_hash_from_rows(rows)
                result.metrics["derived_initial_state_hash"] = (
                    derived_initial_state_hash_from_rows(rows)
                )
                position_field = next(
                    (name for name in ("object.position", "object.position_world_m", "primary_target.position_world_m") if rows and name in rows[0]),
                    None,
                )
                velocity_field = next(
                    (name for name in ("object.linear_velocity", "object.linear_velocity_world_m_s", "primary_target.linear_velocity_world_m_s") if rows and name in rows[0]),
                    None,
                )
                if position_field and velocity_field:
                    contact_samples = [
                        bool(row.get("contact.active", False) or row.get("event.contact", False))
                        for row in rows
                    ]
                    assistance_samples = [bool(row.get("assistance.active", False)) for row in rows]
                    valid_intervals = [
                        not any(contact_samples[max(0, index - 1) : min(len(rows), index + 3)])
                        and not any(assistance_samples[max(0, index - 1) : min(len(rows), index + 3)])
                        for index in range(len(rows) - 1)
                    ]
                    check = finite_difference_velocity_check(
                        timestamps,
                        [row[position_field] for row in rows],
                        [row[velocity_field] for row in rows],
                        valid_interval_mask=valid_intervals,
                    )
                    result.metrics[f"physics.{check.name}"] = check.metrics
                    if not check.passed:
                        if float(check.metrics.get("sample_count", 0.0)) == 0.0:
                            if _requires_ballistic_evidence(record):
                                result.fail(f"physics {check.name}: {check.message}")
                            else:
                                result.warnings.append(
                                    f"physics {check.name} not applicable: {check.message}"
                                )
                        else:
                            result.fail(f"physics {check.name}: {check.message}")
                else:
                    message = (
                        "position/velocity finite-difference QC not evaluated: "
                        "named fields unavailable"
                    )
                    if _requires_ballistic_evidence(record):
                        result.fail(message)
                    else:
                        result.warnings.append(message)
                assistance_fields = (
                    "assisted_grasp",
                    "assisted_retention",
                    "equality_constraint_active",
                    "latch_active",
                )
                for field_name in assistance_fields:
                    measured = any(
                        bool(row.get(f"assistance.{field_name}", False)) for row in rows
                    )
                    declared = bool(record.assistance.get(field_name, False))
                    if measured != declared:
                        result.fail(
                            f"episode assistance summary disagrees with frame mask: {field_name}"
                        )
                any_frame_assistance = any(
                    bool(row.get("assistance.active", False))
                    or any(bool(row.get(f"assistance.{name}", False)) for name in assistance_fields)
                    for row in rows
                )
                if any(
                    bool(row.get("assistance.active", False))
                    != any(bool(row.get(f"assistance.{name}", False)) for name in assistance_fields)
                    for row in rows
                ):
                    result.fail("assistance.active must equal the OR of named assistance flags per frame")
                if record.dynamics_mode == DynamicsMode.FREE_CONTACT and any_frame_assistance:
                    result.fail("free_contact episode has active per-frame assistance")
                if record.dynamics_mode == DynamicsMode.ASSISTED_CONTACT and not any_frame_assistance:
                    result.fail("assisted_contact episode has no active per-frame assistance")
                if any_frame_assistance and not any(
                    any(bool(row.get(f"assistance.{name}", False)) for name in assistance_fields)
                    for row in rows
                ):
                    result.fail("assistance.active is set without a named assistance mechanism")
                for problem in validate_v2_frame_semantics(rows, strict=release_claimed):
                    if release_claimed:
                        result.fail(f"frame semantics: {problem}")
                    else:
                        result.warnings.append(f"frame semantics: {problem}")
                end_effector = str(
                    record.extras.get("end_effector") or record.tool_type or ""
                )
                if end_effector in PRODUCTION_END_EFFECTORS:
                    for problem in validate_action_rows(
                        rows,
                        embodiment=end_effector,
                        action_semantics=record.action_mode,
                    ):
                        result.fail(f"canonical actuator action: {problem}")
                control_hz = float(record.extras.get("control_hz", 0.0) or 0.0)
                assistance_tolerance = 1.0 / control_hz if control_hz > 0 else 1.0 / 60.0
                for problem in validate_assistance_observations(
                    record, rows, tolerance_s=assistance_tolerance
                ):
                    result.fail(f"assistance observations: {problem}")
            except Exception as error:
                result.fail(f"frame parquet: {error}")
        else:
            result.fail("missing frame_data_path")

        for label, relative in (
            ("high-rate", record.high_rate_path),
            ("events", record.events_path),
            ("transitions", record.transition_events_path),
            ("object states", record.object_states_path),
        ):
            if relative is None:
                if label == "transitions" and not release_claimed:
                    result.warnings.append("missing transition-events path")
                else:
                    result.fail(f"missing {label} path")
                continue
            path = resolve_dataset_path(self.root, relative)
            if not path.is_file():
                result.fail(f"missing {label} file: {relative}")
            else:
                expected = record.content_hashes.get(relative)
                if not expected:
                    result.fail(f"missing content hash: {relative}")
                elif sha256_file(path) != expected:
                    result.fail(f"content hash mismatch: {relative}")
                try:
                    rows = read_parquet_rows(path)
                    if label == "high-rate" and rows:
                        high_rate_rows = rows
                        validate_monotonic_timestamps(
                            [float(row["timestamp"]) for row in rows],
                            name="high-rate timestamps",
                        )
                        if any(
                            any(name.startswith("action.") for name in row)
                            for row in rows
                        ):
                            result.metrics["derived_action_hash"] = (
                                derived_action_hash_from_rows(rows)
                            )
                        if end_effector in PRODUCTION_END_EFFECTORS:
                            for problem in validate_action_rows(
                                rows,
                                embodiment=end_effector,
                                action_semantics=record.action_mode,
                            ):
                                result.fail(
                                    f"control-rate canonical actuator action: {problem}"
                                )
                            for row_index, row in enumerate(rows):
                                applied = row.get("simulator.applied_actuator_ctrl")
                                action = row.get("action.actuator_command")
                                if applied is None:
                                    result.fail(
                                        "control-rate action lacks simulator applied-control echo"
                                    )
                                    break
                                try:
                                    applied_values = tuple(float(value) for value in applied)
                                    action_values = tuple(float(value) for value in action)
                                except (TypeError, ValueError):
                                    result.fail(
                                        f"control-rate applied-control echo is invalid at row {row_index}"
                                    )
                                    break
                                if applied_values != action_values:
                                    result.fail(
                                        f"persisted action differs from applied data.ctrl at row {row_index}"
                                    )
                                    break
                    elif label == "transitions" and rows:
                        transition_rows = rows
                        validate_monotonic_timestamps(
                            [float(row["timestamp"]) for row in rows],
                            strictly=False,
                            name="transition timestamps",
                        )
                        if any(not str(row.get("event_type", "")).strip() for row in rows):
                            result.fail("transition rows require event_type")
                        chains: dict[str, str] = {}
                        for row in rows:
                            event_type = str(row["event_type"])
                            source = str(row.get("from", ""))
                            destination = str(row.get("to", ""))
                            if not source or not destination or source == destination:
                                result.fail(
                                    "transition rows require distinct non-empty from/to states"
                                )
                                continue
                            if event_type in chains and source != chains[event_type]:
                                result.fail(
                                    f"transition chain is discontinuous for {event_type}"
                                )
                            chains[event_type] = destination
                    elif label == "object states" and rows:
                        object_state_rows = rows
                        validate_monotonic_timestamps(
                            [float(row["timestamp"]) for row in rows],
                            strictly=False,
                            name="object-state timestamps",
                        )
                        if any(not str(row.get("object_id", "")).strip() for row in rows):
                            result.fail("object-state rows require object_id")
                except Exception as error:
                    result.fail(f"{label} parquet: {error}")
        event_rows: list[dict[str, Any]] = []
        if record.events_path:
            try:
                event_rows = read_parquet_rows(resolve_dataset_path(self.root, record.events_path))
                for row in event_rows:
                    normalized = normalize_contact_row(row)
                    ContactEvent(
                        timestamp=float(normalized["timestamp"]),
                        object_a=str(normalized["object_a"]),
                        object_b=str(normalized["object_b"]),
                        point_world_m=tuple(float(value) for value in normalized["point_world_m"]),
                        normal_world=tuple(float(value) for value in normalized["normal_world"]),
                        penetration_depth_m=float(normalized["penetration_depth_m"]),
                        normal_force_n=(
                            None
                            if normalized.get("normal_force_n") is None
                            else float(normalized["normal_force_n"])
                        ),
                        normal_impulse_n_s=(
                            None
                            if normalized.get("normal_impulse_n_s") is None
                            else float(normalized["normal_impulse_n_s"])
                        ),
                        relative_velocity_world_m_s=(
                            None
                            if normalized.get("relative_velocity_world_m_s") is None
                            else tuple(
                                float(value)
                                for value in normalized["relative_velocity_world_m_s"]
                            )
                        ),
                    ).validate()
                event_timestamps = [float(row["timestamp"]) for row in event_rows if row.get("timestamp") is not None]
                if event_timestamps:
                    validate_monotonic_timestamps(event_timestamps, strictly=False, name="contact events")
                    if record.duration_s is not None and max(event_timestamps) > record.duration_s + 1e-6:
                        result.fail("contact event lies outside clip")
                penetrations = [
                    float(row.get("penetration_depth_m", 0.0) or 0.0) for row in event_rows
                ]
                if penetrations:
                    maximum_penetration = max(penetrations)
                    result.metrics["maximum_penetration_depth_m"] = maximum_penetration
                    if maximum_penetration > STRICT_RIGID_THRESHOLDS.maximum_task_surface_penetration_m:
                        result.fail(f"explosive/excessive penetration: {maximum_penetration} m")
                end_effector = str(
                    record.extras.get("end_effector") or record.tool_type or ""
                )
                source_scenario = record.extras.get("source_scenario_spec")
                strict_source_mujoco = bool(
                    isinstance(source_scenario, Mapping)
                    and source_scenario.get("backend") == "source_mujoco"
                )
                if end_effector in PRODUCTION_END_EFFECTORS or strict_source_mujoco:
                    strict_penetration = strict_contact_penetration_check(
                        event_rows,
                        require_classification=True,
                    )
                    result.metrics["physics.strict_contact_penetration"] = dict(
                        strict_penetration.metrics
                    )
                    for failure in strict_penetration.failures:
                        result.fail(f"strict contact penetration: {failure}")
                if record.extras.get("event_time_semantics") == "first_non_fixture_task_contact":
                    measured_task_time = select_task_event_time(event_rows)
                    if measured_task_time is None and record.event_time_s is not None:
                        result.fail("event_time_s declares a task contact absent from the event table")
                    elif measured_task_time is not None and (
                        record.event_time_s is None
                        or abs(record.event_time_s - measured_task_time) > 1e-9
                    ):
                        result.fail("event_time_s disagrees with the first non-fixture task contact")
            except Exception as error:
                result.fail(f"contact events: {error}")
        if frame_rows:
            ballistic_mode = _ballistic_evidence_mode(record)
            velocity_field = next(
                (
                    name
                    for name in (
                        "object.linear_velocity",
                        "object.linear_velocity_world_m_s",
                        "primary_target.linear_velocity_world_m_s",
                    )
                    if name in frame_rows[0]
                ),
                None,
            )
            free_fall_field = next(
                (
                    name
                    for name in ("free_fall", "event.free_fall", "object.free_fall")
                    if name in frame_rows[0]
                ),
                None,
            )
            if (
                ballistic_mode != "not_required"
                and velocity_field
                and free_fall_field
            ):
                check = gravity_consistency_check(
                    [float(row["timestamp"]) for row in frame_rows],
                    [row[velocity_field] for row in frame_rows],
                    record.physics.gravity_world_m_s2,
                    free_fall_mask=_event_aware_free_fall_mask(
                        frame_rows,
                        event_rows,
                        free_fall_field,
                        precontact_only=(ballistic_mode == "precontact"),
                        release_x_m=(
                            float(record.extras["native_scenario_spec"]["extras"]["transition_release_x_m"])
                            if ballistic_mode == "post_release"
                            and isinstance(record.extras.get("native_scenario_spec"), Mapping)
                            and isinstance(
                                record.extras["native_scenario_spec"].get("extras"),
                                Mapping,
                            )
                            and record.extras["native_scenario_spec"]["extras"].get(
                                "transition_release_x_m"
                            )
                            is not None
                            else None
                        ),
                        release_direction=(
                            int(
                                record.extras["native_scenario_spec"]["extras"].get(
                                    "transition_direction", 1
                                )
                            )
                            if isinstance(record.extras.get("native_scenario_spec"), Mapping)
                            and isinstance(
                                record.extras["native_scenario_spec"].get("extras"),
                                Mapping,
                            )
                            else 1
                        ),
                        stop_at_next_contact=(ballistic_mode == "post_release"),
                    ),
                )
                result.metrics[f"physics.{check.name}"] = check.metrics
                if not check.passed:
                    if float(check.metrics.get("sample_count", 0.0)) == 0.0:
                        result.fail(f"physics {check.name}: {check.message}")
                    else:
                        result.fail(f"physics {check.name}: {check.message}")
            elif ballistic_mode != "not_required":
                result.fail(
                    "gravity consistency not independently evaluated: no free-fall mask"
                )
        measured_contact = record.objective_metrics.get("object_contacted_tool")
        if measured_contact is True and not event_rows:
            result.fail("objective metrics report contact but contact-event table is empty")
        if measured_contact is False and event_rows:
            result.warnings.append("contact events exist while object_contacted_tool=false; inspect object pairs")
        if record.event_time_s is not None and record.duration_s is not None:
            post_event_s = record.duration_s - record.event_time_s
            result.metrics["post_event_evidence_s"] = post_event_s
            if post_event_s < 0.20:
                result.fail("insufficient post-event outcome evidence")
        else:
            result.warnings.append("post-event evidence not evaluated: no event_time_s")
        metric_success = next(
            (
                record.objective_metrics[key]
                for key in ("objective_success", "task_success", "success")
                if key in record.objective_metrics and isinstance(record.objective_metrics[key], bool)
            ),
            None,
        )
        expected_metric_success, metric_subject = _objective_metric_success_reference(
            record
        )
        if (
            metric_success is not None
            and expected_metric_success is not None
            and metric_success != expected_metric_success
        ):
            result.fail(f"{metric_subject} disagrees with objective evaluator metric")
        elif metric_success is None:
            result.warnings.append("label/metric agreement not independently evaluated: no objective_success metric")
        elif expected_metric_success is None:
            result.warnings.append(
                "label/metric agreement not evaluated: unverified candidate evidence unavailable"
            )
        evaluator = self.objective_evaluators.get(
            record.objective_evaluator_id, record.objective_evaluator_version
        )
        if evaluator is None:
            message = (
                "no persisted-artifact objective evaluator registered for "
                f"{record.objective_evaluator_id}/{record.objective_evaluator_version}"
            )
            if release_claimed:
                result.fail(message)
            else:
                result.warnings.append(message)
        else:
            try:
                recomputed = evaluator(
                    ObjectiveRecomputeInput(
                        record=record,
                        frame_rows=frame_rows,
                        event_rows=event_rows,
                        object_state_rows=object_state_rows,
                        transition_rows=transition_rows,
                    )
                )
                recompute_problems = compare_recomputed_objective(record, recomputed)
                if recomputed.key_event_name != record.key_event_name:
                    recompute_problems.append(
                        "key_event_name disagrees with independent objective recomputation"
                    )
                recompute_problems = sorted(set(recompute_problems))
                for problem in recompute_problems:
                    result.fail(problem)
                recomputed_evidence_hash = sha256_json(recomputed.evidence)
                result.metrics["objective_recompute"] = {
                    "evaluator_id": record.objective_evaluator_id,
                    "evaluator_version": record.objective_evaluator_version,
                    "evidence_version": recomputed.evidence_version,
                    "evidence_hash": recomputed_evidence_hash,
                    "task_success": recomputed.task_success,
                    "actual_outcome_class": recomputed.actual_outcome_class.value,
                    "primary_failure_code": recomputed.primary_failure_code,
                    "key_event_name": recomputed.key_event_name,
                    "key_event_time_s": recomputed.key_event_time_s,
                    "replay_match": not recompute_problems,
                }
                if record.objective_evidence.get("independently_recomputed") is not True:
                    result.fail(
                        "objective evaluator ran but metadata does not attest independent recomputation"
                    )
                stored_evidence_hash = record.objective_evidence.get("evidence_hash")
                if stored_evidence_hash and stored_evidence_hash != recomputed_evidence_hash:
                    result.fail("stored objective evidence hash disagrees with recomputation")
            except Exception as error:
                result.fail(f"objective recomputation: {error}")
        visibility = record.extras.get("visibility_qc")
        if isinstance(visibility, Mapping):
            if source_mujoco_backend:
                independent_evidence = record.extras.get(
                    "independent_objective_evidence"
                )
                source_physics = (
                    source_scenario.get("physics")
                    if isinstance(source_scenario, Mapping)
                    else None
                )
                try:
                    visibility_event_tolerance = (
                        1.0 / float(source_physics["simulation_hz"])
                        if isinstance(source_physics, Mapping)
                        else 1e-9
                    )
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    visibility_event_tolerance = 1e-9
                _validate_source_mujoco_visibility_qc(
                    result,
                    visibility,
                    end_effector=end_effector,
                    expected_frame_count=record.frame_count,
                    frame_rows=frame_rows,
                    event_rows=event_rows,
                    record_key_event_name=record.key_event_name,
                    record_key_event_time_s=record.key_event_time_s,
                    objective_key_event_source=(
                        str(independent_evidence.get("measured_key_event_source"))
                        if isinstance(independent_evidence, Mapping)
                        and independent_evidence.get("measured_key_event_source")
                        is not None
                        else None
                    ),
                    event_time_tolerance_s=visibility_event_tolerance,
                    source_scenario=(
                        source_scenario
                        if isinstance(source_scenario, Mapping)
                        else None
                    ),
                )
                expected_visibility_hash = sha256_json(visibility)
                if record.extras.get("visibility_qc_sha256") != expected_visibility_hash:
                    result.fail("source_mujoco visibility QC hash binding changed")
                if isinstance(backend_provenance, Mapping) and (
                    backend_provenance.get("visibility_qc_sha256")
                    != expected_visibility_hash
                ):
                    result.fail(
                        "source_mujoco provenance visibility QC hash binding changed"
                    )
                binding = record.extras.get("visibility_media_binding")
                camera_rows = record.extras.get("camera_calibrations")
                if not isinstance(binding, Mapping):
                    result.fail(
                        "source_mujoco visibility QC lacks encoded-media binding"
                    )
                elif not isinstance(camera_rows, Sequence) or isinstance(
                    camera_rows, (str, bytes, bytearray)
                ):
                    result.fail(
                        "source_mujoco visibility QC lacks bound camera rows"
                    )
                else:
                    try:
                        expected_binding = source_mujoco_visibility_media_binding(
                            visibility_qc_sha256=expected_visibility_hash,
                            camera_rows=camera_rows,
                            camera_stream_calibration_ids=(
                                record.camera_stream_calibration_ids
                            ),
                            video_paths=record.video_paths,
                            content_hashes=record.content_hashes,
                        )
                        if binding != expected_binding:
                            result.fail(
                                "source_mujoco visibility encoded-media binding changed"
                            )
                        if binding.get("schema_version") != (
                            SOURCE_MUJOCO_VISIBILITY_MEDIA_BINDING_SCHEMA
                        ):
                            result.fail(
                                "source_mujoco visibility media-binding schema changed"
                            )
                        if record.extras.get(
                            "visibility_media_binding_sha256"
                        ) != sha256_json(binding):
                            result.fail(
                                "source_mujoco visibility media-binding hash changed"
                            )
                    except Exception as error:
                        result.fail(
                            f"source_mujoco visibility encoded-media binding: {error}"
                        )
            else:
                if visibility.get("schema_version") != NATIVE_VISUAL_QC_SCHEMA:
                    result.fail(f"visibility QC does not use {NATIVE_VISUAL_QC_SCHEMA}")
                if visibility.get("evaluated") is not True:
                    result.fail("visibility QC was not evaluated from rendered streams")
                if visibility.get("key_event_visible_in_any_view") is not True:
                    result.fail("key event is not visible in either view")
                if visibility.get("critically_cropped") is True:
                    result.fail("target critically cropped during key event")
                if visibility.get("contact_occluded_both_views") is True:
                    result.fail("critical contact occluded in both views")
                if float(visibility.get("target_visible_frame_fraction", 0.0)) < float(
                    NATIVE_VISUAL_THRESHOLDS["minimum_target_visible_frame_fraction"]
                ):
                    result.fail("target is not visible in at least 90% of episode frames")
                if float(visibility.get("minimum_bbox_margin_px", 0.0)) < float(
                    NATIVE_VISUAL_THRESHOLDS["minimum_bbox_margin_px"]
                ):
                    result.fail("target crop margin is below 8 pixels at the key event")
                if int(visibility.get("key_event_object_area_px", 0)) < int(
                    NATIVE_VISUAL_THRESHOLDS["minimum_key_event_object_area_px"]
                ):
                    result.fail("target key-event footprint is below 64 pixels")
                if visibility.get("tool_visible_at_key_event") is not True:
                    result.fail("tool is not visible at the key event")
                if visibility.get("fixture_visible_at_key_event") is not True:
                    result.fail("fixture is not visible at the key event")
                if visibility.get("camera_roles_correct") is not True:
                    result.fail("camera roles do not match the task family")
                if float(visibility.get("maximum_underexposed_fraction", 1.0)) > float(
                    NATIVE_VISUAL_THRESHOLDS["maximum_underexposed_fraction"]
                ):
                    result.fail("underexposed/black image fraction exceeds 0.35")
                if float(visibility.get("maximum_overexposed_fraction", 1.0)) > float(
                    NATIVE_VISUAL_THRESHOLDS["maximum_overexposed_fraction"]
                ):
                    result.fail("overexposed image fraction exceeds 0.30")
        elif source_mujoco_backend:
            result.fail("source_mujoco episode lacks rendered visibility_qc metadata")
        else:
            result.warnings.extend(
                [
                    "target visibility/cropping not evaluated: no projected bbox or segmentation metric",
                    "critical-contact occlusion not evaluated: no visibility_qc metadata",
                ]
            )
        if source_mujoco_backend:
            background_clearance = record.extras.get("background_clearance")
            runtime_audit_evidence = record.extras.get("runtime_audit")
            if not isinstance(background_clearance, Mapping):
                result.fail(
                    "source_mujoco episode lacks runtime background clearance"
                )
            else:
                _validate_source_mujoco_background_clearance(
                    result,
                    background_clearance,
                    high_rate_rows=high_rate_rows,
                    source_scenario=(
                        source_scenario
                        if isinstance(source_scenario, Mapping)
                        else None
                    ),
                    backend_provenance=(
                        backend_provenance
                        if isinstance(backend_provenance, Mapping)
                        else None
                    ),
                    runtime_audit=(
                        runtime_audit_evidence
                        if isinstance(runtime_audit_evidence, Mapping)
                        else None
                    ),
                    stored_sha256=record.extras.get(
                        "background_clearance_sha256"
                    ),
                )
        if source_mujoco_backend:
            objective_recompute = result.metrics.get("objective_recompute")
            if not isinstance(objective_recompute, Mapping):
                result.fail(
                    "source_mujoco episode lacks persisted-artifact objective recomputation"
                )
            else:
                if objective_recompute.get("evaluator_id") != record.objective_evaluator_id:
                    result.fail(
                        "source_mujoco objective recomputation evaluator_id disagrees with the record"
                    )
                if not str(objective_recompute.get("evaluator_id") or "").strip() or (
                    objective_recompute.get("evaluator_id") == "legacy_embedded"
                ):
                    result.fail(
                        "source_mujoco objective recomputation lacks a versioned evaluator_id"
                    )
                if not str(objective_recompute.get("key_event_name") or "").strip():
                    result.fail(
                        "source_mujoco objective recomputation lacks key_event_name"
                    )
                key_event_time_s = objective_recompute.get("key_event_time_s")
                if (
                    not isinstance(key_event_time_s, (int, float))
                    or isinstance(key_event_time_s, bool)
                    or not math.isfinite(float(key_event_time_s))
                    or float(key_event_time_s) < 0
                ):
                    result.fail(
                        "source_mujoco objective recomputation lacks finite key_event_time_s"
                    )
                if objective_recompute.get("replay_match") is not True:
                    result.fail(
                        "source_mujoco persisted-artifact objective replay does not match the recorded outcome"
                    )
                recomputed_evidence_hash = str(
                    objective_recompute.get("evidence_hash") or ""
                )
                if len(recomputed_evidence_hash) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in recomputed_evidence_hash
                ):
                    result.fail(
                        "source_mujoco objective recomputation evidence is not content-bound"
                    )
                if not str(
                    objective_recompute.get("evidence_version") or ""
                ).strip():
                    result.fail(
                        "source_mujoco objective recomputation lacks evidence_version"
                    )
            stored_evidence_hash = str(
                record.objective_evidence.get("evidence_hash") or ""
            )
            if record.objective_evidence.get("independently_recomputed") is not True:
                result.fail(
                    "source_mujoco record lacks independent persisted-artifact objective evidence"
                )
            if len(stored_evidence_hash) != 64 or any(
                character not in "0123456789abcdef"
                for character in stored_evidence_hash
            ):
                result.fail(
                    "source_mujoco record lacks a content-bound objective evidence hash"
                )
            if not str(record.key_event_name or "").strip():
                result.fail("source_mujoco record lacks key_event_name")
            if (
                record.key_event_time_s is None
                or not math.isfinite(float(record.key_event_time_s))
                or record.key_event_time_s < 0
            ):
                result.fail("source_mujoco record lacks finite key_event_time_s")
        if source_rigid_backend:
            audit = (
                backend_provenance.get("runtime_audit")
                if isinstance(backend_provenance, Mapping)
                else None
            )
            if not isinstance(audit, Mapping):
                result.fail("source_mujoco provenance lacks a strict runtime audit")
            else:
                for failure in strict_runtime_audit_failures(
                    audit,
                    require_control_updates=end_effector != "no_robot",
                ):
                    result.fail(f"strict runtime audit: {failure}")
            persisted_qc = record.extras.get("physics_qc")
            if not isinstance(persisted_qc, Mapping):
                result.fail("source_mujoco episode lacks persisted strict physics QC")
            else:
                for failure in strict_persisted_physics_failures(persisted_qc):
                    result.fail(f"strict physics QC: {failure}")
                task_evidence = persisted_qc.get("task_evidence")
                if not isinstance(task_evidence, Mapping):
                    result.fail("source_mujoco strict QC lacks family-specific task evidence")
                else:
                    source_spec = record.extras.get("source_scenario_spec")
                    task_variant = str(
                        source_spec.get("task_variant")
                        if isinstance(source_spec, Mapping)
                        else record.variant
                    )
                    for failure in rigid_task_evidence_failures(
                        family=record.family,
                        subfamily=record.subfamily,
                        task_variant=task_variant,
                        evidence=task_evidence,
                        task_success=record.task_success,
                        require_final_retention=bool(
                            isinstance(source_spec, Mapping)
                            and source_spec.get("schema_version")
                            == "dynamic-robot-source-scenario/v2"
                            and record.objective_evaluator_version in {"1.5.0", "1.6.0"}
                        ),
                    ):
                        result.fail(f"strict task physics: {failure}")
            try:
                validate_randomization_admission(
                    record.randomization,
                    r1_accepted=bool(record.extras.get("r1_accepted", False)),
                )
            except ValueError as error:
                result.fail(f"randomization admission: {error}")
            background = str(record.randomization.get("background_style") or "")
            if background.startswith("robocasa_"):
                manifest = record.extras.get("robocasa_asset_manifest")
                if not isinstance(manifest, Sequence) or isinstance(
                    manifest, (str, bytes, bytearray)
                ):
                    result.fail("RoboCasa scene lacks a runtime asset-admission manifest")
                else:
                    allow_pending_render_review = bool(
                        isinstance(backend_provenance, Mapping)
                        and backend_provenance.get("review_only") is True
                        and not record.release_eligible
                    )
                    try:
                        validate_robocasa_asset_manifest(
                            manifest,
                            required_asset_ids=(
                                str(record.randomization.get("scene_asset_id")),
                            ),
                            allow_pending_render_review=allow_pending_render_review,
                        )
                    except ValueError as error:
                        result.fail(f"RoboCasa asset admission: {error}")
                    else:
                        if allow_pending_render_review and any(
                            value.get("blockers")
                            == ["rendered_occlusion_review_pending"]
                            for value in manifest
                            if isinstance(value, Mapping)
                        ):
                            result.warnings.append(
                                "RoboCasa rendered occlusion human review pending"
                            )
        native_backend = (
            isinstance(backend_provenance, Mapping)
            and backend_provenance.get("backend") == "native_mujoco"
            and record.simulator_name.lower() == "mujoco"
        )
        if native_backend:
            def verify_external_artifact(
                *, label: str, path_value: Any, expected_value: Any
            ) -> None:
                expected = str(expected_value or "")
                try:
                    path = Path(str(path_value)).resolve(strict=True)
                except (FileNotFoundError, OSError):
                    result.fail(f"{label} artifact path is unavailable")
                    return
                if (
                    not path.is_file()
                    or len(expected) != 64
                    or any(character not in "0123456789abcdef" for character in expected)
                    or sha256_file(path) != expected
                ):
                    result.fail(f"{label} artifact is not content-bound")

            audit = backend_provenance.get("runtime_audit")
            if not isinstance(audit, Mapping):
                result.fail("native backend provenance lacks runtime_audit")
            else:
                for field_name in (
                    "object_state_writes_after_initialization",
                    "direct_robot_state_writes_after_initialization",
                    "equality_constraint_count",
                ):
                    if int(audit.get(field_name, -1)) != 0:
                        result.fail(f"native runtime audit has nonzero {field_name}")
                if int(audit.get("initial_object_state_writes", 0)) != 1:
                    result.fail("native runtime audit must record exactly one initial object-state write")
                if int(audit.get("initial_robot_state_writes", 0)) != 1:
                    result.fail("native runtime audit must record exactly one initial robot-state write")
                if int(audit.get("simulation_steps", 0)) <= 0:
                    result.fail("native runtime audit has no simulation steps")
                if int(audit.get("control_updates", 0)) <= 0:
                    result.fail("native runtime audit has no actuator control updates")
            if backend_provenance.get("visual_style_validated") is True:
                verify_external_artifact(
                    label="visual-style validation",
                    path_value=backend_provenance.get(
                        "visual_style_validation_artifact_path"
                    ),
                    expected_value=backend_provenance.get(
                        "visual_style_validation_artifact_hash"
                    ),
                )
            if backend_provenance.get("tool_calibrated") is True:
                verify_external_artifact(
                    label="tool calibration",
                    path_value=backend_provenance.get(
                        "tool_calibration_artifact_path"
                    ),
                    expected_value=backend_provenance.get(
                        "tool_calibration_artifact_sha256"
                    ),
                )
            physics_range = backend_provenance.get("physics_range_provenance")
            if isinstance(physics_range, Mapping) and physics_range.get("calibrated") is True:
                verify_external_artifact(
                    label="physics-range calibration",
                    path_value=physics_range.get("resolved_artifact_path"),
                    expected_value=physics_range.get("calibration_artifact"),
                )
            persisted_qc = record.extras.get("physics_qc")
            persisted_checks = (
                persisted_qc.get("checks")
                if isinstance(persisted_qc, Mapping)
                else None
            )
            required_native_checks = (
                "finite_state",
                "no_post_initialization_object_state_writes",
                "no_direct_robot_state_writes_after_initialization",
                "no_equality_or_latch_assistance",
                "contact_penetration_bounded",
                "control_within_declared_ranges",
                "free_flight_acceleration_consistent",
                "gravity_sweep_measurement_available",
                "position_velocity_consistent",
                "free_flight_energy_consistent",
                "joint_velocity_bounded",
                "joint_acceleration_bounded",
                "joint_positions_within_limits",
                "actuator_forces_within_limits",
                "contact_forces_finite",
                "momentum_impulse_accounting_consistent",
                "task_contact_count_bounded",
                "distinct_contact_count_bounded",
                "no_measured_contact_energy_gain",
                "measured_restitution_matches_target",
                "restitution_sweep_measurement_available",
                "friction_sweep_measurement_available",
                "rolling_slip_bounded",
            )
            if not isinstance(persisted_checks, Mapping):
                result.fail("native episode lacks persisted physics-QC checks")
            else:
                missing_checks = sorted(set(required_native_checks) - set(persisted_checks))
                if missing_checks:
                    result.fail(f"native physics-QC checks are missing: {missing_checks}")
                failed_checks = sorted(
                    name for name in required_native_checks if persisted_checks.get(name) is not True
                )
                if record.physics_qc_pass and failed_checks:
                    result.fail(
                        f"physics_qc_pass=true despite failed native checks: {failed_checks}"
                    )

            if high_rate_rows:
                joint_rows = [
                    row
                    for row in high_rate_rows
                    if isinstance(row.get("robot.joint_velocity"), Sequence)
                ]
                if joint_rows:
                    maximum_velocity = max(
                        abs(float(value))
                        for row in joint_rows
                        for value in row["robot.joint_velocity"]
                    )
                    maximum_acceleration = 0.0
                    for left, right in zip(joint_rows, joint_rows[1:]):
                        dt = float(right["timestamp"]) - float(left["timestamp"])
                        if dt <= 0:
                            continue
                        maximum_acceleration = max(
                            maximum_acceleration,
                            max(
                                abs(float(b) - float(a)) / dt
                                for a, b in zip(
                                    left["robot.joint_velocity"],
                                    right["robot.joint_velocity"],
                                )
                            ),
                        )
                    result.metrics["native_recomputed_maximum_joint_velocity_rad_s"] = maximum_velocity
                    result.metrics["native_recomputed_maximum_joint_acceleration_rad_s2"] = maximum_acceleration
                    if maximum_velocity > 3.5 + 1e-9:
                        result.fail("recomputed native joint velocity exceeds limit")
                    if maximum_acceleration > 80.0 + 1e-9:
                        result.fail("recomputed native joint acceleration exceeds limit")

                first_contact = min(
                    (float(row["timestamp"]) for row in event_rows), default=math.inf
                )
                ballistic_mode = _ballistic_evidence_mode(record)
                scenario_extras = (
                    record.extras.get("native_scenario_spec", {}).get("extras", {})
                    if isinstance(record.extras.get("native_scenario_spec"), Mapping)
                    else {}
                )
                release_x = scenario_extras.get("transition_release_x_m")
                release_direction = int(
                    scenario_extras.get("transition_direction", 1)
                )
                next_contact = min(
                    (
                        float(row["timestamp"])
                        for row in event_rows
                        if float(row["timestamp"]) > first_contact + 1e-9
                    ),
                    default=math.inf,
                )
                free_flight = [
                    row
                    for row in high_rate_rows
                    if str(row.get("object.motion_mode")) == "free_flight"
                    and (
                        ballistic_mode != "precontact"
                        or float(row["timestamp"]) < first_contact
                    )
                    and (
                        ballistic_mode != "post_release"
                        or (
                            release_x is not None
                            and release_direction
                            * (
                                float(row["object.position"][0])
                                - float(release_x)
                            )
                            >= 0.0
                        )
                    )
                    and (
                        ballistic_mode != "post_release"
                        or float(row["timestamp"]) < next_contact
                    )
                    and isinstance(row.get("object.linear_velocity"), Sequence)
                ]
                if ballistic_mode != "not_required" and len(free_flight) >= 2:
                    check = gravity_consistency_check(
                        [float(row["timestamp"]) for row in free_flight],
                        [row["object.linear_velocity"] for row in free_flight],
                        record.physics.gravity_world_m_s2,
                        tolerance_m_s2=1.2,
                    )
                    result.metrics["physics.native_recomputed_gravity"] = check.metrics
                    if not check.passed:
                        result.fail(f"physics {check.name}: {check.message}")

                radius = record.physics.parameters.get("radius")
                rolling_rows = [
                    row
                    for row in high_rate_rows
                    if str(row.get("object.motion_mode")) == "rolling"
                    and isinstance(row.get("object.linear_velocity"), Sequence)
                    and isinstance(row.get("object.angular_velocity"), Sequence)
                ]
                if rolling_rows and radius is not None and radius.valid and radius.implemented:
                    radius_m = float(radius.value)
                    slip_speeds = []
                    for row in rolling_rows:
                        vx, vy, _ = (float(value) for value in row["object.linear_velocity"])
                        wx, wy, _ = (float(value) for value in row["object.angular_velocity"])
                        slip_speeds.append(
                            math.hypot(vx - wy * radius_m, vy + wx * radius_m)
                        )
                    maximum_slip = max(slip_speeds)
                    result.metrics["physics.native_recomputed_maximum_rolling_slip_m_s"] = maximum_slip
                    if maximum_slip > 0.12 + 1e-9:
                        result.fail("recomputed native rolling slip exceeds limit")

            raw_spec = record.extras.get("native_scenario_spec")
            if isinstance(raw_spec, Mapping):
                expected = [str(value) for value in raw_spec.get("expected_contact_sequence", ())]
                if expected:
                    chronological = [
                        *[
                            (float(row["timestamp"]), str(row.get("object_b")))
                            for row in event_rows
                        ],
                        *[
                            (float(row["timestamp"]), str(row.get("motion_mode")))
                            for row in frame_rows
                        ],
                    ]
                    observed = [value for _, value in sorted(chronological)]
                    ordered, malformed_prefix = _ordered_contact_sequence(
                        expected,
                        observed,
                    )
                    if malformed_prefix:
                        result.fail(
                            "native episode violates declared contact/transition order"
                        )
                    elif record.task_success and not ordered:
                        result.fail(
                            "successful native episode violates declared contact/transition order"
                        )
        if not record.physics_qc_pass:
            if record.dynamics_mode == DynamicsMode.FREE_CONTACT:
                result.fail("physics_qc_pass=false")
            else:
                result.warnings.append(
                    f"physics_qc_pass=false for quarantined {record.dynamics_mode.value} episode"
                )
        range_provenance = record.physics.parameter_range_provenance
        result.metrics["parameter_range_partition"] = range_provenance.get("partition")
        result.metrics["parameter_range_calibrated"] = range_provenance.get("calibrated")
        if release_claimed and range_provenance.get("calibrated") is not True:
            result.fail("release candidate uses an uncalibrated parameter-range profile")
        if release_claimed and (
            record.controller_profile.get("profile_id") == "legacy_unspecified"
            or not record.robot_start_provenance
            or not record.tool_calibration_provenance
        ):
            result.fail(
                "release candidate lacks controller, robot-start, or tool-calibration provenance"
            )
        bounce_keys = (
            "preimpact_normal_velocity_m_s",
            "postimpact_normal_velocity_m_s",
            "configured_restitution",
        )
        if all(key in record.objective_metrics for key in bounce_keys):
            check = bounce_restitution_check(
                float(record.objective_metrics[bounce_keys[0]]),
                float(record.objective_metrics[bounce_keys[1]]),
                float(record.objective_metrics[bounce_keys[2]]),
            )
            result.metrics[f"physics.{check.name}"] = check.metrics
            if not check.passed:
                result.fail(f"physics {check.name}: {check.message}")
        elif record.family == "projectile_rebound":
            result.warnings.append("bounce restitution not independently evaluated: objective velocities unavailable")
        result.metrics["video_hashes"] = video_hashes
        result.metrics["perceptual_hashes"] = perceptual
        return result, video_hashes

    def validate(self) -> DatasetQCReport:
        records = load_episode_records(self.root)
        split_table = self.root / "meta" / "splits.parquet"
        split_rows = read_parquet_rows(split_table) if split_table.is_file() else []
        results: list[EpisodeQC] = []
        exact: dict[str, list[str]] = defaultdict(list)
        perceptual: list[tuple[str, str]] = []
        for record in records:
            result, hashes = self._validate_episode(record)
            results.append(result)
            for camera, digest in hashes.items():
                exact[digest].append(f"{record.episode_uuid}:{camera}")
            for camera, signature in result.metrics.get("perceptual_hashes", {}).items():
                perceptual.append((f"{record.episode_uuid}:{camera}", signature))
        duplicate_groups = sorted([sorted(values) for values in exact.values() if len(values) > 1])
        perceptual_pairs: list[tuple[str, str, int]] = []
        for index, (left_name, left_hash) in enumerate(perceptual):
            for right_name, right_hash in perceptual[index + 1 :]:
                if len(left_hash) == len(right_hash):
                    distance = hamming_distance_hex(left_hash, right_hash)
                    if distance <= 8:
                        perceptual_pairs.append((left_name, right_name, distance))
        leakage = validate_no_split_leakage(
            records,
            split_rows if split_rows else None,
        )
        global_failures = [f"split leakage: {value}" for value in leakage]
        release_uuids = {record.episode_uuid for record in records if record.release_eligible}
        split_by_uuid = (
            {
                str(row["episode_uuid"]): str(row["split"])
                for row in split_rows
            }
            if split_rows
            else {record.episode_uuid: record.split.value for record in records}
        )
        global_failures.extend(exact_duplicate_split_leakage(duplicate_groups, split_by_uuid))
        exact_duplicate_warnings: list[str] = []
        for group in duplicate_groups:
            episode_uuids = {item.split(":", 1)[0] for item in group}
            duplicate_splits = {
                split_by_uuid[episode_uuid]
                for episode_uuid in episode_uuids
                if episode_uuid in split_by_uuid
            }
            if len(duplicate_splits) > 1:
                continue
            elif episode_uuids & release_uuids:
                global_failures.append(f"exact duplicate release video: {group}")
            else:
                exact_duplicate_warnings.append(f"exact duplicate nonrelease video: {group}")
        if any(
            record.release_eligible
            and split_by_uuid.get(record.episode_uuid, "unassigned") == "unassigned"
            for record in records
        ):
            global_failures.append("release-eligible episodes remain split=unassigned")
        global_warnings = exact_duplicate_warnings + [
            f"approximate duplicate candidate (Hamming distance {distance}): {left}, {right}"
            for left, right, distance in perceptual_pairs
        ]
        physics_families: dict[str, list[EpisodeRecord]] = defaultdict(list)
        for record in records:
            if record.physics_counterfactual_family_id:
                physics_families[record.physics_counterfactual_family_id].append(record)
        for family_id, siblings in physics_families.items():
            if len(siblings) < 2:
                continue
            action_hashes = {record.extras.get("action_hash") for record in siblings}
            invariant_hashes = {
                record.extras.get("counterfactual_invariant_hash") for record in siblings
            }
            visual_signatures = {
                json.dumps(record.randomization, sort_keys=True, separators=(",", ":"), allow_nan=False)
                for record in siblings
            }
            camera_signatures = {tuple(record.camera_ids) for record in siblings}
            result_by_uuid = {result.episode_uuid: result for result in results}
            derived_action_hashes = {
                result_by_uuid[record.episode_uuid].metrics.get("derived_action_hash")
                for record in siblings
            }
            derived_initial_state_hashes = {
                result_by_uuid[record.episode_uuid].metrics.get("derived_initial_state_hash")
                for record in siblings
            }
            appearance_signatures = {
                sha256_json(
                    {
                        "asset_ids": record.asset_ids,
                        "asset_hashes": record.asset_hashes,
                        "randomization": record.randomization,
                    }
                )
                for record in siblings
            }
            if None in action_hashes or None in invariant_hashes:
                global_warnings.append(
                    f"physics family {family_id} lacks action/invariant hashes; equality not evaluated"
                )
            if len(action_hashes - {None}) > 1:
                global_failures.append(f"physics family {family_id} changes the action hash")
            if None in derived_action_hashes or len(derived_action_hashes) > 1:
                global_failures.append(f"physics family {family_id} changes saved action columns")
            if None in derived_initial_state_hashes or len(derived_initial_state_hashes) > 1:
                global_failures.append(f"physics family {family_id} changes saved initial state")
            if len(invariant_hashes - {None}) > 1:
                global_failures.append(f"physics family {family_id} changes non-physics invariants")
            if len(visual_signatures) > 1:
                global_failures.append(f"physics family {family_id} changes visual randomization")
            if len(camera_signatures) > 1:
                global_failures.append(f"physics family {family_id} changes camera streams")
            if len(appearance_signatures) > 1:
                global_failures.append(f"physics family {family_id} changes assets or appearance")
        claimed_release_uuids = {
            record.episode_uuid
            for record in records
            if record.label_status.value == "verified_objective"
            and record.release_tier == ReleaseTier.FREE_CONTACT
            and record.dynamics_mode == DynamicsMode.FREE_CONTACT
            and record.physics_qc_pass
            and not record.quality_flags
        }
        counterfactual_table = self.root / "meta" / "counterfactual_families.parquet"
        declarations: list[CounterfactualFamilyRecord] = []
        if counterfactual_table.is_file():
            try:
                declarations = [
                    CounterfactualFamilyRecord.from_dict(row)
                    for row in read_parquet_rows(counterfactual_table)
                ]
                derived_by_uuid = {
                    result.episode_uuid: {
                        "derived_action_hash": result.metrics.get("derived_action_hash"),
                        "derived_initial_state_hash": result.metrics.get(
                            "derived_initial_state_hash"
                        ),
                    }
                    for result in results
                }
                global_failures.extend(
                    f"counterfactual: {problem}"
                    for problem in validate_counterfactual_family_records(
                        declarations, records, derived_by_uuid=derived_by_uuid
                    )
                )
                declared_action_ids = {
                    declaration.family_id
                    for declaration in declarations
                    if declaration.relation.value == "action"
                }
                declared_physics_ids = {
                    declaration.family_id
                    for declaration in declarations
                    if declaration.relation.value == "physics"
                }
                for record in records:
                    if record.episode_uuid not in claimed_release_uuids:
                        continue
                    if record.counterfactual_bundle_id not in declared_action_ids:
                        global_failures.append(
                            f"release episode {record.episode_uuid} lacks action-family declaration"
                        )
                    if (
                        record.physics_counterfactual_family_id is not None
                        and record.physics_counterfactual_family_id
                        not in declared_physics_ids
                    ):
                        global_failures.append(
                            f"release episode {record.episode_uuid} lacks physics-family declaration"
                        )
            except Exception as error:
                global_failures.append(f"invalid counterfactual family table: {error}")
        elif claimed_release_uuids:
            global_failures.append(
                "release candidates have no meta/counterfactual_families.parquet declaration table"
            )
        camera_table = self.root / "meta" / "cameras.parquet"
        camera_rows = read_parquet_rows(camera_table) if camera_table.is_file() else []
        if not camera_rows:
            if release_uuids:
                global_failures.append("release-eligible episodes have no camera-calibration table")
            else:
                global_warnings.append("camera-calibration table is empty")
        else:
            camera_ids: set[str] = set()
            camera_stream_by_id: dict[str, str] = {}
            for row in camera_rows:
                identifier = str(row.get("camera_id") or row.get("camera_name") or "")
                try:
                    if not identifier or identifier in camera_ids:
                        raise SchemaValidationError(
                            "camera_id must be non-empty and unique"
                        )
                    calibration_value = dict(row)
                    calibration_value.pop("camera_id", None)
                    calibration = CameraCalibration.from_dict(calibration_value)
                    if (calibration.width, calibration.height, calibration.fps) != (832, 480, 30.0):
                        raise SchemaValidationError(
                            f"camera {identifier} is not canonical 832x480 at 30 FPS"
                        )
                    camera_ids.add(identifier)
                    camera_stream_by_id[identifier] = calibration.camera_name
                except Exception as error:
                    global_failures.append(f"invalid camera calibration {identifier or '<unnamed>'}: {error}")
            for record in records:
                calibration_mapping = record.camera_stream_calibration_ids or {
                    stream: stream for stream in record.video_paths
                }
                missing = sorted(set(calibration_mapping.values()) - camera_ids)
                if missing:
                    message = f"episode {record.episode_uuid} lacks calibration rows for {missing}"
                    if record.release_eligible:
                        global_failures.append(message)
                    else:
                        global_warnings.append(message)
                for stream, calibration_id in calibration_mapping.items():
                    calibrated_stream = camera_stream_by_id.get(calibration_id)
                    if calibrated_stream is not None and calibrated_stream != stream:
                        message = (
                            f"episode {record.episode_uuid} maps stream {stream} to calibration "
                            f"{calibration_id} for {calibrated_stream}"
                        )
                        if record.episode_uuid in claimed_release_uuids:
                            global_failures.append(message)
                        else:
                            global_warnings.append(message)
        return DatasetQCReport(
            str(self.root),
            results,
            sorted(set(global_failures)),
            sorted(set(global_warnings)),
            duplicate_groups,
            perceptual_pairs,
            self.strict_all,
        )


def validate_dataset(
    dataset_root: str | Path,
    *,
    deep_video_checks: bool = True,
    write_reports: bool = False,
    report_dir: str | Path | None = None,
    objective_evaluators: ObjectiveEvaluatorRegistry | None = None,
    strict_all: bool = False,
) -> DatasetQCReport:
    """Stable public validator API used by tests, CLI, and smoke orchestration."""

    report = QCValidator(
        dataset_root,
        deep_video_checks=deep_video_checks,
        objective_evaluators=objective_evaluators,
        strict_all=strict_all,
    ).validate()
    if write_reports:
        write_qc_reports(report, report_dir or (Path(dataset_root) / "qc"))
    return report


def _csv_bytes(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_qc_reports(report: DatasetQCReport, directory: str | Path) -> None:
    """Write all required machine-readable QC summaries without replacement."""

    root = ensure_not_source_path(directory)
    root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(report.dataset_root)
    completion_path = dataset_root / "meta" / ".complete.json"
    if not completion_path.is_file():
        raise RuntimeError(
            "QC reports may only be published for an atomically finalized dataset"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    metadata_content_hashes = dict(completion.get("content_hashes") or {})
    report_payload = report.to_dict()
    report_payload.update(
        schema_version="dynamic-robot-qc-report/v2",
        dataset_episodes_sha256=sha256_file(dataset_root / "meta" / "episodes.parquet"),
        metadata_complete_manifest_sha256=sha256_file(completion_path),
        metadata_content_hashes=metadata_content_hashes,
    )
    atomic_write_json(root / "dataset_report.json", report_payload)
    write_parquet_atomic(root / "episode_qc.parquet", [episode.to_dict() for episode in report.episodes])
    failure_counter: Counter[str] = Counter(
        failure for episode in report.episodes for failure in episode.hard_failures
    )
    atomic_write_bytes(
        root / "failure_summary.csv",
        _csv_bytes(("failure", "count"), ({"failure": key, "count": value} for key, value in sorted(failure_counter.items()))),
    )
    records = load_episode_records(report.dataset_root)
    qc_by_uuid = {episode.episode_uuid: episode for episode in report.episodes}
    outcome_counter: Counter[tuple[str, str, str]] = Counter(
        (record.family, record.subfamily, record.actual_outcome_class.value)
        for record in records
    )
    atomic_write_bytes(
        root / "outcome_distribution.csv",
        _csv_bytes(
            ("family", "subfamily", "actual_outcome_class", "count"),
            (
                {
                    "family": key[0],
                    "subfamily": key[1],
                    "actual_outcome_class": key[2],
                    "count": value,
                }
                for key, value in sorted(outcome_counter.items())
            ),
        ),
    )
    confusion_counter: Counter[tuple[str, str]] = Counter(
        (record.intended_branch, record.actual_outcome_class.value)
        for record in records
    )
    atomic_write_bytes(
        root / "intended_actual_confusion.csv",
        _csv_bytes(
            ("intended_branch", "actual_outcome_class", "count"),
            (
                {
                    "intended_branch": key[0],
                    "actual_outcome_class": key[1],
                    "count": value,
                }
                for key, value in sorted(confusion_counter.items())
            ),
        ),
    )
    atomic_write_bytes(
        root / "physics_validation.csv",
        _csv_bytes(
            ("episode_uuid", "physics_qc_pass", "release_eligible"),
            (
                {
                    "episode_uuid": record.episode_uuid,
                    "physics_qc_pass": record.physics_qc_pass,
                    "release_eligible": record.release_eligible,
                }
                for record in records
            ),
        ),
    )
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    tier_names = (
        "free_contact",
        "assisted_contact",
        "scripted_motion",
        "unverified",
        "quarantine",
    )

    def manifest_bytes(values: Iterable[EpisodeRecord]) -> bytes:
        return b"".join(
            (
                json.dumps(
                    {
                        "episode_uuid": record.episode_uuid,
                        "episode_index": record.episode_index,
                        "family": record.family,
                        "subfamily": record.subfamily,
                        "split": record.split.value,
                        "release_tier": record.release_tier.value,
                        "dynamics_mode": record.dynamics_mode.value,
                        "task_success": record.task_success,
                        "quality_flags": record.quality_flags,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            for record in values
        )

    default_records = [
        record
        for record in records
        if record.release_eligible and qc_by_uuid.get(record.episode_uuid) is not None
        and qc_by_uuid[record.episode_uuid].passed
    ]
    atomic_write_bytes(manifest_root / "default_training.jsonl", manifest_bytes(default_records))
    for tier in tier_names:
        tier_records = [
            record
            for record in records
            if record.release_tier.value == tier
            or (tier == "unverified" and record.label_status.value == "unverified")
            or (tier == "quarantine" and bool(record.quality_flags))
        ]
        atomic_write_bytes(manifest_root / f"{tier}.jsonl", manifest_bytes(tier_records))
