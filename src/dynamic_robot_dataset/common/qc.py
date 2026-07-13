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

from .contacts import ContactEvent, normalize_contact_row, select_task_event_time
from .cameras import CameraCalibration
from .episode_writer import load_episode_records, read_parquet_rows, write_parquet_atomic
from .hashing import hamming_distance_hex, sha256_file, sha256_json
from .paths import atomic_write_bytes, atomic_write_json, ensure_not_source_path, resolve_dataset_path
from .schema import DynamicsMode, EpisodeRecord, ReleaseTier, SchemaValidationError
from .splits import validate_no_split_leakage
from .synchronization import SynchronizationError, validate_monotonic_timestamps, validate_synchronized_streams
from .video_writer import VideoProbe, VideoSpec, iter_rgb_frames, probe_frame_timestamps, probe_video, validate_video_probe


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

    @property
    def passed(self) -> bool:
        return not self.global_failures and all(
            result.passed or not result.release_eligible for result in self.episodes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "passed": self.passed,
            "episode_count": len(self.episodes),
            "release_eligible_count": sum(result.release_eligible for result in self.episodes),
            "failed_release_eligible_count": sum(
                result.release_eligible and not result.passed for result in self.episodes
            ),
            "global_failures": self.global_failures,
            "global_warnings": self.global_warnings,
            "exact_duplicate_groups": self.exact_duplicate_groups,
            "perceptual_duplicate_pairs": self.perceptual_duplicate_pairs,
            "episodes": [result.to_dict() for result in self.episodes],
        }


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

    def __init__(self, dataset_root: str | Path, *, deep_video_checks: bool = True):
        self.root = Path(dataset_root).resolve(strict=True)
        self.deep_video_checks = deep_video_checks

    def _validate_episode(self, record: EpisodeRecord) -> tuple[EpisodeQC, dict[str, str]]:
        result = EpisodeQC(record.episode_uuid, record.episode_index, record.release_eligible)
        video_hashes: dict[str, str] = {}
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
        if set(record.camera_ids) != set(record.video_paths):
            result.fail("camera_ids disagree with video_paths")
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
                timestamps = [float(row["timestamp"]) for row in rows]
                validate_monotonic_timestamps(timestamps)
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
                action_fields = sorted(
                    name for name in (rows[0] if rows else {}) if name.startswith("action.")
                )
                result.metrics["derived_action_hash"] = sha256_json(
                    [[row.get(name) for name in action_fields] for row in rows]
                )
                excluded_initial_prefixes = ("action.", "contact.", "event.", "assistance.")
                initial_fields = sorted(
                    name
                    for name in (rows[0] if rows else {})
                    if name not in {
                        "episode_index",
                        "frame_index",
                        "video_frame_index",
                        "task_index",
                        "timestamp",
                    }
                    and not name.startswith(excluded_initial_prefixes)
                )
                result.metrics["derived_initial_state_hash"] = sha256_json(
                    {name: rows[0].get(name) for name in initial_fields} if rows else {}
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
                        result.fail(f"physics {check.name}: {check.message}")
                else:
                    result.warnings.append("position/velocity finite-difference QC not evaluated: named fields unavailable")
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
                free_fall_field = next(
                    (name for name in ("free_fall", "event.free_fall", "object.free_fall") if rows and name in rows[0]),
                    None,
                )
                if velocity_field and free_fall_field:
                    check = gravity_consistency_check(
                        timestamps,
                        [row[velocity_field] for row in rows],
                        record.physics.gravity_world_m_s2,
                        free_fall_mask=[bool(row[free_fall_field]) for row in rows],
                    )
                    result.metrics[f"physics.{check.name}"] = check.metrics
                    if not check.passed:
                        result.fail(f"physics {check.name}: {check.message}")
                elif record.family in {"falling_catch", "projectile_rebound"}:
                    result.warnings.append("gravity consistency not independently evaluated: no free-fall mask")
            except Exception as error:
                result.fail(f"frame parquet: {error}")
        else:
            result.fail("missing frame_data_path")

        for label, relative in (
            ("high-rate", record.high_rate_path),
            ("events", record.events_path),
            ("object states", record.object_states_path),
        ):
            if relative is None:
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
                        validate_monotonic_timestamps(
                            [float(row["timestamp"]) for row in rows],
                            name="high-rate timestamps",
                        )
                    elif label == "object states" and rows:
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
                    if maximum_penetration > 0.02:
                        result.fail(f"explosive/excessive penetration: {maximum_penetration} m")
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
        if metric_success is not None and metric_success != record.task_success:
            result.fail("task_success disagrees with objective evaluator metric")
        elif metric_success is None:
            result.warnings.append("label/metric agreement not independently evaluated: no objective_success metric")
        visibility = record.extras.get("visibility_qc")
        if isinstance(visibility, Mapping):
            if visibility.get("critically_cropped") is True:
                result.fail("target critically cropped during key event")
            if visibility.get("contact_occluded_both_views") is True:
                result.fail("critical contact occluded in both views")
            if float(visibility.get("minimum_visible_fraction", 1.0)) < 0.25:
                result.fail("target visibility is insufficient")
        else:
            result.warnings.extend(
                [
                    "target visibility/cropping not evaluated: no projected bbox or segmentation metric",
                    "critical-contact occlusion not evaluated: no visibility_qc metadata",
                ]
            )
        if not record.physics_qc_pass:
            if record.dynamics_mode == DynamicsMode.FREE_CONTACT:
                result.fail("physics_qc_pass=false")
            else:
                result.warnings.append(
                    f"physics_qc_pass=false for quarantined {record.dynamics_mode.value} episode"
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
        leakage = validate_no_split_leakage(records)
        global_failures = [f"split leakage: {value}" for value in leakage]
        release_uuids = {record.episode_uuid for record in records if record.release_eligible}
        for group in duplicate_groups:
            if any(item.split(":", 1)[0] in release_uuids for item in group):
                global_failures.append(f"exact duplicate release video: {group}")
        if any(record.release_eligible and record.split.value == "unassigned" for record in records):
            global_failures.append("release-eligible episodes remain split=unassigned")
        global_warnings = [
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
        camera_table = self.root / "meta" / "cameras.parquet"
        camera_rows = read_parquet_rows(camera_table) if camera_table.is_file() else []
        if not camera_rows:
            if release_uuids:
                global_failures.append("release-eligible episodes have no camera-calibration table")
            else:
                global_warnings.append("camera-calibration table is empty")
        else:
            camera_ids: set[str] = set()
            for row in camera_rows:
                identifier = str(row.get("camera_id") or row.get("camera_name") or "")
                try:
                    calibration_value = dict(row)
                    calibration_value.pop("camera_id", None)
                    calibration = CameraCalibration.from_dict(calibration_value)
                    if identifier != calibration.camera_name:
                        raise SchemaValidationError(
                            f"camera_id {identifier!r} differs from camera_name {calibration.camera_name!r}"
                        )
                    if (calibration.width, calibration.height, calibration.fps) != (832, 480, 30.0):
                        raise SchemaValidationError(
                            f"camera {identifier} is not canonical 832x480 at 30 FPS"
                        )
                    camera_ids.add(identifier)
                except Exception as error:
                    global_failures.append(f"invalid camera calibration {identifier or '<unnamed>'}: {error}")
            for record in records:
                missing = sorted(set(record.camera_ids) - camera_ids)
                if missing:
                    message = f"episode {record.episode_uuid} lacks calibration rows for {missing}"
                    if record.release_eligible:
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
        )


def validate_dataset(
    dataset_root: str | Path,
    *,
    deep_video_checks: bool = True,
    write_reports: bool = False,
    report_dir: str | Path | None = None,
) -> DatasetQCReport:
    """Stable public validator API used by tests, CLI, and smoke orchestration."""

    report = QCValidator(dataset_root, deep_video_checks=deep_video_checks).validate()
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
    atomic_write_json(root / "dataset_report.json", report.to_dict())
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
        (record.family, record.subfamily, record.actual_outcome) for record in records
    )
    atomic_write_bytes(
        root / "outcome_distribution.csv",
        _csv_bytes(
            ("family", "subfamily", "actual_outcome", "count"),
            (
                {"family": key[0], "subfamily": key[1], "actual_outcome": key[2], "count": value}
                for key, value in sorted(outcome_counter.items())
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
