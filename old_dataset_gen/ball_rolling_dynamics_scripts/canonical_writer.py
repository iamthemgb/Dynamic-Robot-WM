"""Canonical rollout artifacts in the ``dynamic-robot-writer-layout/v2`` format.

Self-contained port of the episode-artifact format defined by the unified
generator (``dataset_generation/src/dynamic_robot_dataset/common``:
``episode_writer.py``, ``video_writer.py``, ``arrow_schema.py``,
``synchronization.py``, ``schema.py``).  It reproduces the on-disk rollout
contract — chunked layout, H.264/yuv420p canonical videos, the exact
``round(duration x fps)`` half-open frame clock, typed Arrow sidecar tables,
and the v2 episode-metadata Parquet schema — without the unified repository's
fail-closed orchestration (locks, seals, resume guards), which is out of scope
for this legacy pipeline.  It only needs pyarrow and imageio-ffmpeg's bundled
ffmpeg, both present in the pbc-native environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "dynamic-robot-dataset/v2"
WRITER_LAYOUT_VERSION = "dynamic-robot-writer-layout/v2"
ARROW_SIDECAR_SCHEMA_VERSION = "dynamic-robot-arrow-sidecars/v2"
FAILURE_TAXONOMY_VERSION = "dynamic-robot-failure-codes/v2"
EPISODE_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "ball-rolling-dynamics-scripts")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def episode_uuid(family: str, episode_index: int) -> str:
    return str(uuid.uuid5(EPISODE_UUID_NAMESPACE, f"{family}/{episode_index:06d}"))


# ---------------------------------------------------------------------------
# Frame clock (port of synchronization.fixed_duration_frame_timestamps)
# ---------------------------------------------------------------------------


def exact_frame_timestamps(frame_count: int, fps_num: int, fps_den: int = 1) -> list[float]:
    period = Fraction(fps_den, fps_num)
    return [float(index * period) for index in range(frame_count)]


def fixed_duration_frame_timestamps(duration_s: float, fps_num: int = 30, fps_den: int = 1) -> list[float]:
    """Canonical half-open render clock: round(duration x fps) frames at k/fps."""

    if not math.isfinite(duration_s) or duration_s <= 0 or fps_num <= 0 or fps_den <= 0:
        raise ValueError("Duration must be finite and positive and FPS must be positive")
    duration = Fraction(str(duration_s))
    frames_exact = duration * fps_num / fps_den
    frame_count = (2 * frames_exact.numerator + frames_exact.denominator) // (2 * frames_exact.denominator)
    if frame_count <= 0:
        raise ValueError("Duration produces no canonical video frames")
    return exact_frame_timestamps(frame_count, fps_num, fps_den)


# ---------------------------------------------------------------------------
# Canonical video encoding (port of video_writer.FrameVideoWriter)
# ---------------------------------------------------------------------------


class VideoEncodingError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoSpec:
    width: int = 832
    height: int = 480
    fps_num: int = 30
    fps_den: int = 1
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    input_pixel_format: str = "rgb24"
    crf: int = 18
    preset: str = "medium"

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("H.264/yuv420p dimensions must be positive and even")
        if self.fps_num <= 0 or self.fps_den <= 0:
            raise ValueError("FPS must be a positive rational")
        if self.pixel_format != "yuv420p":
            raise ValueError("Canonical output pixel format is yuv420p")


def _ffmpeg_executable() -> str:
    import shutil

    executable = shutil.which("ffmpeg")
    if executable is None:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    return executable


def encode_canonical_video(frames: Sequence[Any], destination: Path, spec: VideoSpec) -> int:
    """Encode RGB frames with the canonical H.264 contract; return frame count."""

    spec.validate()
    if not frames:
        raise VideoEncodingError("Cannot encode a video with no frames")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".mp4", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    command = [
        _ffmpeg_executable(), "-v", "error", "-y",
        "-f", "rawvideo", "-pixel_format", spec.input_pixel_format,
        "-video_size", f"{spec.width}x{spec.height}",
        "-framerate", f"{spec.fps_num}/{spec.fps_den}", "-i", "pipe:0",
        "-an", "-c:v", spec.codec, "-preset", spec.preset,
        "-crf", str(spec.crf), "-pix_fmt", spec.pixel_format,
        "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert process.stdin is not None
    expected = spec.width * spec.height * 3
    try:
        for frame in frames:
            shape = getattr(frame, "shape", None)
            if shape is not None and tuple(shape) != (spec.height, spec.width, 3):
                raise VideoEncodingError(f"Frame shape {shape} != {(spec.height, spec.width, 3)}")
            data = frame.tobytes(order="C") if hasattr(frame, "tobytes") else bytes(frame)
            if len(data) != expected:
                raise VideoEncodingError(f"RGB frame has {len(data)} bytes, expected {expected}")
            process.stdin.write(data)
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    if return_code != 0:
        temporary.unlink(missing_ok=True)
        raise VideoEncodingError(stderr.strip() or "FFmpeg failed")
    decoded = count_video_frames(temporary, spec.width, spec.height)
    if decoded != len(frames):
        temporary.unlink(missing_ok=True)
        raise VideoEncodingError(f"Encoded {decoded} frames, expected {len(frames)}: {destination}")
    destination.unlink(missing_ok=True)
    os.replace(temporary, destination)
    return len(frames)


def count_video_frames(path: Path, width: int, height: int) -> int:
    """Count decodable frames without ffprobe (not installed on this cluster)."""

    command = [
        _ffmpeg_executable(), "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    size = width * height * 3
    count = 0
    remainder = 0
    while True:
        data = process.stdout.read(size)
        if not data:
            break
        remainder += len(data)
        count += remainder // size
        remainder %= size
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise VideoEncodingError(f"FFmpeg decode failed for {path}: {stderr.strip()}")
    if remainder:
        raise VideoEncodingError(f"Truncated decoded frame in {path}")
    return count


# ---------------------------------------------------------------------------
# Sharded layout (port of episode_writer.DatasetLayout)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    chunk_size: int = 1000

    def chunk_name(self, episode_index: int) -> str:
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        return f"chunk-{episode_index // self.chunk_size:03d}"

    def file_name(self, episode_index: int, suffix: str) -> str:
        return f"file-{episode_index:06d}.{suffix}"

    def frame_data(self, episode_index: int) -> str:
        return f"data/{self.chunk_name(episode_index)}/{self.file_name(episode_index, 'parquet')}"

    def high_rate(self, episode_index: int) -> str:
        return f"high_rate/{self.chunk_name(episode_index)}/{self.file_name(episode_index, 'parquet')}"

    def events(self, episode_index: int) -> str:
        return f"events/{self.chunk_name(episode_index)}/{self.file_name(episode_index, 'parquet')}"

    def transitions(self, episode_index: int) -> str:
        return f"transitions/{self.chunk_name(episode_index)}/{self.file_name(episode_index, 'parquet')}"

    def object_states(self, episode_index: int) -> str:
        return f"object_states/{self.chunk_name(episode_index)}/{self.file_name(episode_index, 'parquet')}"

    def video(self, camera_name: str, episode_index: int) -> str:
        camera = canonical_camera_name(camera_name)
        return f"videos/{camera}/{self.chunk_name(episode_index)}/{self.file_name(episode_index, 'mp4')}"

    def marker(self, episode_uuid_text: str) -> str:
        return f".records/{episode_uuid_text}.json"


def canonical_camera_name(name: str) -> str:
    value = name.strip()
    if value in {"main", "secondary", "top", "wrist"}:
        value = f"observation.images.{value}"
    if not value.startswith("observation."):
        raise ValueError(f"Camera stream must be an observation path: {name}")
    return value


# ---------------------------------------------------------------------------
# Typed Arrow sidecar tables (port of arrow_schema.py)
# ---------------------------------------------------------------------------

TABLE_KINDS = ("frame", "high_rate", "events", "transitions", "objects")

_CORE_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "frame": (
        ("episode_index", "int64"),
        ("task_index", "int64"),
        ("frame_index", "int64"),
        ("video_frame_index", "int64"),
        ("timestamp", "float64"),
        ("simulation_timestamp", "float64"),
        ("synchronization_error_s", "float64"),
        ("action.actuator_command", "list<float64>"),
        ("simulator.applied_actuator_ctrl", "list<float64>"),
        ("action.mode", "string"),
        ("contact.active", "bool"),
        ("event.contact", "bool"),
        ("assistance.active", "bool"),
        ("assistance.assisted_grasp", "bool"),
        ("assistance.assisted_retention", "bool"),
        ("assistance.equality_constraint_active", "bool"),
        ("assistance.latch_active", "bool"),
        ("assistance.mechanism_ids", "list<string>"),
    ),
    "high_rate": (
        ("episode_index", "int64"),
        ("timestamp", "float64"),
        ("action.actuator_command", "list<float64>"),
        ("simulator.applied_actuator_ctrl", "list<float64>"),
        ("action.mode", "string"),
    ),
    "events": (
        ("episode_index", "int64"),
        ("timestamp", "float64"),
        ("object_a", "string"),
        ("object_b", "string"),
        ("point_world_m", "list<float64>"),
        ("normal_world", "list<float64>"),
        ("penetration_depth_m", "float64"),
        ("contact_category", "string"),
        ("counterpart_geom_id", "int64"),
        ("normal_force_n", "float64"),
        ("normal_impulse_n_s", "float64"),
        ("relative_velocity_world_m_s", "list<float64>"),
        ("expected_fixture_contact", "bool"),
        ("snag", "bool"),
    ),
    "transitions": (
        ("episode_index", "int64"),
        ("timestamp", "float64"),
        ("event_type", "string"),
        ("from", "string"),
        ("to", "string"),
        ("active_surface", "string"),
    ),
    "objects": (
        ("episode_index", "int64"),
        ("timestamp", "float64"),
        ("object_id", "string"),
    ),
}

_CONTRACTS = {
    "frame": "dynamic-robot-frames/v1",
    "high_rate": "dynamic-robot-high-rate/v1",
    "events": "dynamic-robot-contact-events/v2",
    "transitions": "dynamic-robot-transitions/v1",
    "objects": "dynamic-robot-object-states/v1",
}


def _arrow_type(type_name: str) -> Any:
    scalar = {
        "bool": pa.bool_(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "string": pa.string(),
        "json": pa.string(),
    }
    if type_name in scalar:
        return scalar[type_name]
    element_name = type_name.removeprefix("list<").removesuffix(">")
    return pa.list_(_arrow_type(element_name))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _semantic_extension_type(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith("_json"):
        return "string"
    if lower.endswith(("_index", ".index", "_count", ".count")):
        return "int64"
    if lower.startswith(("is_", "has_")) or lower.endswith(
        (".active", ".enabled", "_flag", ".bilateral", ".retained")
    ):
        return "bool"
    if lower.endswith(
        (
            ".position",
            ".joint_position",
            ".linear_velocity",
            ".angular_velocity",
            ".joint_velocity",
            ".actuator_force",
            ".quaternion_wxyz",
            "_world_m",
            "_world_m_s",
            "_world_n_s",
            "_position_m",
            "_velocity_m_s",
        )
    ):
        return "list<float64>"
    if lower.endswith(("_mode", ".mode", "_phase", ".phase", "_role", ".role")):
        return "string"
    return None


def _inferred_extension_type(name: str, values: Sequence[Any]) -> str:
    semantic = _semantic_extension_type(name)
    if semantic is not None:
        return semantic
    present = [value for value in values if value is not None]
    if not present:
        raise ValueError(f"Arrow extension field {name!r} is all-null; declare its type explicitly")
    if all(isinstance(value, bool) for value in present):
        return "bool"
    if all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in present):
        return "float64"
    if all(isinstance(value, str) for value in present):
        return "string"
    if all(isinstance(value, Mapping) for value in present):
        return "json"
    if all(_is_sequence(value) for value in present):
        elements = [element for value in present for element in value]
        if not elements:
            raise ValueError(f"Arrow extension field {name!r} contains only empty lists")
        if all(isinstance(value, bool) for value in elements):
            return "list<bool>"
        if all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in elements):
            return "list<float64>"
        if all(isinstance(value, str) for value in elements):
            return "list<string>"
    raise ValueError(f"Arrow extension field {name!r} has mixed or unsupported values")


def _validate_finite(value: Any, label: str) -> None:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite numeric value")
    elif _is_sequence(value):
        for item in value:
            _validate_finite(item, label)


def canonical_sidecar_table(kind: str, rows: Sequence[Mapping[str, Any]]) -> Any:
    """Build a typed table with stable core column order and typed extensions."""

    if kind not in TABLE_KINDS:
        raise ValueError(f"Unknown Arrow sidecar table kind: {kind}")
    values = [dict(row) for row in rows]
    core = dict(_CORE_FIELDS[kind])
    extra_names = sorted({name for row in values for name in row} - set(core))
    extra_types = {
        name: _inferred_extension_type(name, [row.get(name) for row in values])
        for name in extra_names
    }
    declarations = [*_CORE_FIELDS[kind], *sorted(extra_types.items())]
    normalized_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(values):
        normalized: dict[str, Any] = {}
        for name, type_name in declarations:
            value = row.get(name)
            _validate_finite(value, f"{kind} row {row_index} field {name!r}")
            if type_name == "json" and value is not None:
                value = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            normalized[name] = value
        normalized_rows.append(normalized)
    identity = {
        "version": ARROW_SIDECAR_SCHEMA_VERSION,
        "contract": _CONTRACTS[kind],
        "fields": declarations,
    }
    schema = pa.schema(
        [pa.field(name, _arrow_type(type_name)) for name, type_name in declarations],
        metadata={
            b"contract": _CONTRACTS[kind].encode("utf-8"),
            b"schema_strategy": ARROW_SIDECAR_SCHEMA_VERSION.encode("utf-8"),
            b"schema_identity_sha256": sha256_json(identity).encode("ascii"),
            b"field_declarations_json": json.dumps(declarations, separators=(",", ":"), ensure_ascii=True).encode("ascii"),
        },
    )
    return pa.Table.from_pylist(normalized_rows, schema=schema)


def write_parquet(destination: Path, table: Any, *, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Parquet output already exists: {destination}")
        destination.unlink()
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary, compression="zstd", version="2.6")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def flexible_mapping_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Store nested maps as canonical JSON scalars with a ``_json`` suffix."""

    result: list[dict[str, Any]] = []
    for source in rows:
        value = dict(source)
        for key in list(value):
            if isinstance(value[key], Mapping):
                value[f"{key}_json"] = json.dumps(
                    value.pop(key), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                )
        result.append(value)
    return result


# ---------------------------------------------------------------------------
# Episode metadata schema (port of episode_writer._episode_metadata_schema)
# ---------------------------------------------------------------------------

FLEXIBLE_EPISODE_FIELDS = (
    "physics",
    "assistance",
    "objective_metrics",
    "objective_evidence",
    "randomization",
    "content_hashes",
    "video_paths",
    "asset_hashes",
    "camera_stream_calibration_ids",
    "controller_profile",
    "robot_start_provenance",
    "tool_calibration_provenance",
    "extras",
)

_EPISODE_FIELD_ORDER = (
    "episode_uuid", "episode_index", "counterfactual_bundle_id", "scene_seed",
    "branch_seed", "family", "subfamily", "intended_branch", "actual_outcome",
    "task_success", "failure_mode", "schema_version", "actual_outcome_class",
    "primary_failure_code", "failure_tags", "failure_taxonomy_version", "variant",
    "robot_model", "tool_type", "action_mode", "partial_success_score",
    "label_confidence", "label_status", "dynamics_mode", "release_tier",
    "physics_qc_pass", "split", "physics_counterfactual_family_id",
    "split_group_id", "parent_episode_uuid", "source_generator",
    "source_generator_version", "generator_git_commit", "config_hash", "asset_ids",
    "simulator_name", "simulator_version", "renderer", "creation_timestamp",
    "frame_data_path", "high_rate_path", "events_path", "transition_events_path",
    "object_states_path", "camera_ids", "task_index", "frame_count", "duration_s",
    "event_time_s", "key_event_name", "key_event_time_s", "objective_evaluator_id",
    "objective_evaluator_version", "objective_threshold_set_hash", "quality_flags",
    "release_eligible",
    *(f"{name}_json" for name in FLEXIBLE_EPISODE_FIELDS),
)

_EPISODE_STRINGS = {
    "episode_uuid", "counterfactual_bundle_id", "family", "subfamily",
    "intended_branch", "actual_outcome", "failure_mode", "schema_version",
    "actual_outcome_class", "primary_failure_code", "failure_taxonomy_version",
    "variant", "robot_model", "tool_type", "action_mode", "label_status",
    "dynamics_mode", "release_tier", "split", "physics_counterfactual_family_id",
    "split_group_id", "parent_episode_uuid", "source_generator",
    "source_generator_version", "generator_git_commit", "config_hash",
    "simulator_name", "simulator_version", "renderer", "creation_timestamp",
    "frame_data_path", "high_rate_path", "events_path", "transition_events_path",
    "object_states_path", "key_event_name", "objective_evaluator_id",
    "objective_evaluator_version", "objective_threshold_set_hash",
}
_EPISODE_INTEGERS = {"episode_index", "scene_seed", "branch_seed", "task_index", "frame_count"}
_EPISODE_FLOATS = {
    "partial_success_score", "label_confidence", "duration_s", "event_time_s", "key_event_time_s",
}
_EPISODE_BOOLEANS = {"task_success", "physics_qc_pass", "release_eligible"}
_EPISODE_STRING_LISTS = {"failure_tags", "asset_ids", "camera_ids", "quality_flags"}


def episode_metadata_schema() -> Any:
    json_fields = {f"{name}_json" for name in FLEXIBLE_EPISODE_FIELDS}
    fields = []
    for name in _EPISODE_FIELD_ORDER:
        if name in _EPISODE_STRINGS or name in json_fields:
            data_type = pa.string()
        elif name in _EPISODE_INTEGERS:
            data_type = pa.int64()
        elif name in _EPISODE_FLOATS:
            data_type = pa.float64()
        elif name in _EPISODE_BOOLEANS:
            data_type = pa.bool_()
        elif name in _EPISODE_STRING_LISTS:
            data_type = pa.list_(pa.string())
        else:
            raise RuntimeError(f"Episode Arrow field has no declared type: {name}")
        fields.append(pa.field(name, data_type))
    return pa.schema(fields, metadata={b"contract": b"dynamic-robot-episodes/v2"})


def episode_record_to_table_row(record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    for field_name in FLEXIBLE_EPISODE_FIELDS:
        value[f"{field_name}_json"] = json.dumps(
            value.pop(field_name), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    missing = [name for name in _EPISODE_FIELD_ORDER if name not in value]
    extra = sorted(set(value) - set(_EPISODE_FIELD_ORDER))
    if missing or extra:
        raise ValueError(f"Episode row differs from the v2 contract: missing={missing}, extra={extra}")
    return {name: value[name] for name in _EPISODE_FIELD_ORDER}


def episode_metadata_table(records: Sequence[Mapping[str, Any]]) -> Any:
    rows = [episode_record_to_table_row(record) for record in records]
    return pa.Table.from_pylist(rows, schema=episode_metadata_schema())


def counterfactual_families_schema() -> Any:
    return pa.schema(
        [
            ("family_id", pa.string()),
            ("relation", pa.string()),
            ("split_group_id", pa.string()),
            ("expected_member_count", pa.int64()),
            ("expected_episode_uuids_json", pa.string()),
            ("intervention_fields_json", pa.string()),
            ("fixed_field_hashes_json", pa.string()),
            ("table_version", pa.string()),
        ]
    )


# ---------------------------------------------------------------------------
# Episode artifact writing
# ---------------------------------------------------------------------------


def write_episode_artifacts(
    root: Path,
    episode_index: int,
    *,
    frame_rows: Sequence[Mapping[str, Any]],
    high_rate_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    object_state_rows: Sequence[Mapping[str, Any]],
    videos: Mapping[str, Sequence[Any]],
    video_spec: VideoSpec,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one rollout's canonical artifact set; return paths + hashes."""

    layout = DatasetLayout(root)
    if not frame_rows:
        raise ValueError("frame_rows cannot be empty")
    video_paths = {canonical_camera_name(name): layout.video(name, episode_index) for name in videos}
    required = {"observation.images.main", "observation.images.secondary"}
    missing = sorted(required - set(video_paths))
    if missing:
        raise ValueError(f"Canonical episodes require two synchronized views; missing {missing}")
    for name, frames in videos.items():
        if len(frames) != len(frame_rows):
            raise ValueError(
                f"Camera {name} has {len(frames)} frames but the frame table has {len(frame_rows)} rows"
            )
    paths = {
        "frame": layout.frame_data(episode_index),
        "high_rate": layout.high_rate(episode_index),
        "events": layout.events(episode_index),
        "transitions": layout.transitions(episode_index),
        "objects": layout.object_states(episode_index),
    }
    write_parquet(root / paths["frame"], canonical_sidecar_table("frame", frame_rows), overwrite=overwrite)
    write_parquet(root / paths["high_rate"], canonical_sidecar_table("high_rate", high_rate_rows), overwrite=overwrite)
    write_parquet(root / paths["events"], canonical_sidecar_table("events", event_rows), overwrite=overwrite)
    write_parquet(root / paths["transitions"], canonical_sidecar_table("transitions", transition_rows), overwrite=overwrite)
    write_parquet(root / paths["objects"], canonical_sidecar_table("objects", object_state_rows), overwrite=overwrite)
    for source_name, frames in videos.items():
        camera = canonical_camera_name(source_name)
        target = root / video_paths[camera]
        if target.exists() and not overwrite:
            raise FileExistsError(f"Video already exists: {target}")
        encode_canonical_video(frames, target, video_spec)
    every_output = [*paths.values(), *video_paths.values()]
    content_hashes = {relative: sha256_file(root / relative) for relative in every_output}
    return {
        "frame_data_path": paths["frame"],
        "high_rate_path": paths["high_rate"],
        "events_path": paths["events"],
        "transition_events_path": paths["transitions"],
        "object_states_path": paths["objects"],
        "video_paths": video_paths,
        "content_hashes": content_hashes,
        "frame_count": len(frame_rows),
    }


def write_episode_marker(root: Path, record: Mapping[str, Any], config_hash: str) -> Path:
    layout = DatasetLayout(root)
    marker = root / layout.marker(str(record["episode_uuid"]))
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config_hash": config_hash, "episode": dict(record)}
    fd, name = tempfile.mkstemp(prefix=f".{marker.name}.", suffix=".tmp", dir=marker.parent)
    os.close(fd)
    temporary = Path(name)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, marker)
    return marker


def load_episode_markers(root: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))["episode"]
        for path in sorted((root / ".records").glob("*.json"))
    ]
    records.sort(key=lambda record: int(record["episode_index"]))
    return records


def writer_settings(video_spec: VideoSpec, chunk_size: int = 1000) -> dict[str, Any]:
    return {
        "layout_version": WRITER_LAYOUT_VERSION,
        "chunk_size": chunk_size,
        "video": asdict(video_spec),
        "required_camera_streams": [
            "observation.images.main",
            "observation.images.secondary",
        ],
    }
