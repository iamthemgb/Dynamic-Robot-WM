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
from .visual_qc import NATIVE_VISUAL_QC_SCHEMA, NATIVE_VISUAL_THRESHOLDS


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
        else:
            result.warnings.extend(
                [
                    "target visibility/cropping not evaluated: no projected bbox or segmentation metric",
                    "critical-contact occlusion not evaluated: no visibility_qc metadata",
                ]
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
