"""Atomic episode artifacts and final MP4/Parquet dataset layout."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import fcntl

from .hashing import sha256_file, sha256_json
from .arrow_schema import (
    ARROW_SIDECAR_SCHEMA_VERSION,
    arrow_schema_identity,
    canonical_sidecar_table,
    normalize_extra_field_declarations,
)
from .contacts import ContactEvent, normalize_contact_row
from .paths import (
    ExistingOutputError,
    ResumeGuard,
    ResumeMismatchError,
    as_dataset_relative,
    atomic_copy,
    atomic_write_json,
    ensure_not_source_path,
    portable_relative_path,
    resolve_dataset_path,
)
from .schema import DatasetInfo, EpisodeRecord, Split, validate_episode_records
from .synchronization import (
    validate_monotonic_timestamps,
    validate_persisted_render_schedule,
    validate_synchronized_streams,
)
from .video_writer import VideoProbe, VideoSpec, encode_video, probe_frame_timestamps, probe_video, validate_video_probe
from .visual_qc import source_mujoco_visibility_media_binding


class MissingParquetDependency(RuntimeError):
    """Canonical Parquet I/O requires the optional PyArrow dependency."""


class DatasetSealedError(ExistingOutputError):
    """The generation root has entered its irreversible finalization phase."""


WRITER_LAYOUT_VERSION = "dynamic-robot-writer-layout/v2"
DEFAULT_SPLIT_SETTINGS: dict[str, Any] = {
    "strategy": "leakage_aware_stratified",
    "seed": 0,
    "train_fraction": 0.80,
    "validation_fraction": 0.10,
    "test_fraction": 0.10,
}
_WORKER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
FINALIZED_METADATA_ARTIFACTS = frozenset(
    {
        "info.json",
        "episodes.parquet",
        "tasks.parquet",
        "cameras.parquet",
        "provenance.parquet",
        "splits.parquet",
        "counterfactual_families.parquet",
    }
)


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise MissingParquetDependency(
            "Parquet support requires pyarrow; install the repository base requirements"
        ) from exc
    return pa, pq


def write_parquet_atomic(
    destination: str | Path,
    rows: Iterable[Mapping[str, Any]] | Any,
    *,
    schema: Any | None = None,
) -> Path:
    """Write one Parquet file atomically without replacing an existing file."""

    pa, pq = _pyarrow()
    target = ensure_not_source_path(destination)
    if target.exists():
        raise ExistingOutputError(f"Parquet output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pa.Table):
        table = rows
    else:
        values = list(rows)
        if values and schema is None:
            # ``Table.from_pylist`` otherwise infers its struct fields from
            # the first mapping and silently drops keys that appear only in
            # later heterogeneous event/transition rows. Materialize the
            # union so solver-rate boundary and restitution evidence survives
            # the immutable Parquet round trip.
            field_names = sorted(
                {name for value in values for name in value}
            )
            values = [
                {name: value.get(name) for name in field_names}
                for value in values
            ]
        table = pa.Table.from_pylist(values, schema=schema) if values else (
            pa.Table.from_pylist([], schema=schema) if schema is not None else pa.table({})
        )
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        pq.write_table(table, temporary, compression="zstd", version="2.6")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise ExistingOutputError(f"Parquet output already exists: {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return target


_ASSISTANCE_MECHANISM_IDS_FIELD = "assistance.mechanism_ids"


def _canonical_frame_table(rows: Iterable[Mapping[str, Any]]) -> Any:
    """Build a frame table with stable types for semantically typed columns.

    Arrow infers ``list<null>`` when every mechanism list in one episode is
    empty. That makes an otherwise valid unassisted episode incompatible with
    an assisted episode whose corresponding column is ``list<string>``. The
    canonical frame contract always stores mechanism IDs as strings, including
    when the list is empty or the producer omitted the optional field.
    """

    pa, _ = _pyarrow()
    values = [dict(row) for row in rows]
    mechanism_ids_by_frame: list[list[str]] = []
    for frame_index, row in enumerate(values):
        raw_ids = row.get(_ASSISTANCE_MECHANISM_IDS_FIELD, [])
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            raise ValueError(
                f"Frame {frame_index} {_ASSISTANCE_MECHANISM_IDS_FIELD!r} "
                "must be a sequence of strings"
            )
        mechanism_ids = list(raw_ids)
        if any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in mechanism_ids
        ):
            raise ValueError(
                f"Frame {frame_index} {_ASSISTANCE_MECHANISM_IDS_FIELD!r} contains an invalid ID"
            )
        row[_ASSISTANCE_MECHANISM_IDS_FIELD] = mechanism_ids
        mechanism_ids_by_frame.append(mechanism_ids)

    table = pa.Table.from_pylist(values)
    typed_column = pa.array(
        mechanism_ids_by_frame,
        type=pa.list_(pa.string()),
    )
    column_index = table.schema.get_field_index(_ASSISTANCE_MECHANISM_IDS_FIELD)
    if column_index < 0:
        return table.append_column(_ASSISTANCE_MECHANISM_IDS_FIELD, typed_column)
    return table.set_column(
        column_index,
        _ASSISTANCE_MECHANISM_IDS_FIELD,
        typed_column,
    )


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a Parquet table into Python records."""

    _, pq = _pyarrow()
    return pq.read_table(Path(path)).to_pylist()


_FLEXIBLE_EPISODE_FIELDS = (
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


def _episode_metadata_schema() -> Any:
    """Return the explicit, versioned Arrow schema for episode metadata."""

    pa, _ = _pyarrow()
    strings = {
        "episode_uuid",
        "counterfactual_bundle_id",
        "family",
        "subfamily",
        "intended_branch",
        "actual_outcome",
        "failure_mode",
        "schema_version",
        "actual_outcome_class",
        "primary_failure_code",
        "failure_taxonomy_version",
        "variant",
        "robot_model",
        "tool_type",
        "action_mode",
        "label_status",
        "dynamics_mode",
        "release_tier",
        "split",
        "physics_counterfactual_family_id",
        "split_group_id",
        "parent_episode_uuid",
        "source_generator",
        "source_generator_version",
        "generator_git_commit",
        "config_hash",
        "simulator_name",
        "simulator_version",
        "renderer",
        "creation_timestamp",
        "frame_data_path",
        "high_rate_path",
        "events_path",
        "transition_events_path",
        "object_states_path",
        "key_event_name",
        "objective_evaluator_id",
        "objective_evaluator_version",
        "objective_threshold_set_hash",
    }
    integers = {"episode_index", "scene_seed", "branch_seed", "task_index", "frame_count"}
    floats = {
        "partial_success_score",
        "label_confidence",
        "duration_s",
        "event_time_s",
        "key_event_time_s",
    }
    booleans = {"task_success", "physics_qc_pass", "release_eligible"}
    string_lists = {"failure_tags", "asset_ids", "camera_ids", "quality_flags"}
    json_fields = {f"{name}_json" for name in _FLEXIBLE_EPISODE_FIELDS}
    fields = []
    # The order is deliberately fixed rather than inherited from dataclass
    # construction or the first episode in a shard.
    for name in (
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
        *(f"{name}_json" for name in _FLEXIBLE_EPISODE_FIELDS),
    ):
        if name in strings or name in json_fields:
            data_type = pa.string()
        elif name in integers:
            data_type = pa.int64()
        elif name in floats:
            data_type = pa.float64()
        elif name in booleans:
            data_type = pa.bool_()
        elif name in string_lists:
            data_type = pa.list_(pa.string())
        else:  # pragma: no cover - protects edits to the explicit field order
            raise RuntimeError(f"Episode Arrow field has no declared type: {name}")
        fields.append(pa.field(name, data_type))
    return pa.schema(fields, metadata={b"contract": b"dynamic-robot-episodes/v2"})


def _episode_metadata_table(records: Iterable[EpisodeRecord]) -> Any:
    pa, _ = _pyarrow()
    rows = [episode_record_to_table_row(record) for record in records]
    schema = _episode_metadata_schema()
    expected = set(schema.names)
    for row in rows:
        if set(row) != expected:
            raise ValueError(
                "Episode metadata row differs from the explicit Arrow contract: "
                f"missing={sorted(expected - set(row))}, extra={sorted(set(row) - expected)}"
            )
    return pa.Table.from_pylist(rows, schema=schema)


def episode_record_to_table_row(record: EpisodeRecord) -> dict[str, Any]:
    """Encode flexible episode metadata as canonical JSON scalar columns.

    Arrow structs require each child to have one physical type across every
    row. Physics values intentionally mix scalars and vectors, while evaluator
    metrics vary by task. Encoding only those flexible maps as sorted JSON keeps
    the Parquet schema stable and lossless; fixed identity/label columns remain
    typed and directly queryable.
    """

    value = record.to_dict()
    for field_name in _FLEXIBLE_EPISODE_FIELDS:
        value[f"{field_name}_json"] = json.dumps(
            value.pop(field_name), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    return value


def episode_record_from_table_row(row: Mapping[str, Any]) -> EpisodeRecord:
    """Reverse :func:`episode_record_to_table_row`."""

    value = dict(row)
    for field_name in _FLEXIBLE_EPISODE_FIELDS:
        encoded = value.pop(f"{field_name}_json", None)
        if encoded is not None:
            value[field_name] = json.loads(encoded)
    return EpisodeRecord.from_dict(value)


def flexible_mapping_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Make arbitrary metadata maps Arrow-safe without losing information.

    Empty dictionaries become unsupported zero-child Arrow structs, and values
    from different provenance providers can have incompatible child types. Map
    columns are therefore stored as canonical JSON scalars with a ``_json``
    suffix; ordinary scalar and list columns remain directly queryable.
    """

    result: list[dict[str, Any]] = []
    for source in rows:
        value = dict(source)
        for key in list(value):
            if isinstance(value[key], Mapping):
                value[f"{key}_json"] = json.dumps(
                    value.pop(key),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
        result.append(value)
    return result


def _empty_sidecar_schema(kind: str) -> Any:
    """Return a typed empty schema so absence of events is still annotated."""

    pa, _ = _pyarrow()
    if kind == "events":
        return pa.schema(
            [
                ("episode_index", pa.int64()),
                ("timestamp", pa.float64()),
                ("object_a", pa.string()),
                ("object_b", pa.string()),
                ("point_world_m", pa.list_(pa.float64(), 3)),
                ("normal_world", pa.list_(pa.float64(), 3)),
                ("penetration_depth_m", pa.float64()),
                ("normal_force_n", pa.float64()),
                ("normal_impulse_n_s", pa.float64()),
                ("relative_velocity_world_m_s", pa.list_(pa.float64(), 3)),
                ("expected_fixture_contact", pa.bool_()),
                ("snag", pa.bool_()),
            ]
        )
    if kind == "high_rate":
        return pa.schema([("episode_index", pa.int64()), ("timestamp", pa.float64())])
    if kind == "objects":
        return pa.schema(
            [("episode_index", pa.int64()), ("timestamp", pa.float64()), ("object_id", pa.string())]
        )
    if kind == "transitions":
        return pa.schema(
            [
                ("episode_index", pa.int64()),
                ("timestamp", pa.float64()),
                ("event_type", pa.string()),
                ("from", pa.string()),
                ("to", pa.string()),
                ("active_surface", pa.string()),
            ]
        )
    raise ValueError(f"Unknown sidecar schema kind: {kind}")


def _validated_contact_row(value: Mapping[str, Any]) -> dict[str, Any]:
    result = normalize_contact_row(value)
    event = ContactEvent(
        timestamp=float(result["timestamp"]),
        object_a=str(result["object_a"]),
        object_b=str(result["object_b"]),
        point_world_m=tuple(float(component) for component in result["point_world_m"]),
        normal_world=tuple(float(component) for component in result["normal_world"]),
        penetration_depth_m=float(result["penetration_depth_m"]),
        normal_force_n=(
            None if result.get("normal_force_n") is None else float(result["normal_force_n"])
        ),
        normal_impulse_n_s=(
            None
            if result.get("normal_impulse_n_s") is None
            else float(result["normal_impulse_n_s"])
        ),
        relative_velocity_world_m_s=(
            None
            if result.get("relative_velocity_world_m_s") is None
            else tuple(float(component) for component in result["relative_velocity_world_m_s"])
        ),
        expected_fixture_contact=bool(result.get("expected_fixture_contact", False)),
        snag=bool(result.get("snag", False)),
    )
    event.validate()
    result.update(event.to_dict())
    return result


@dataclass(slots=True, frozen=True)
class DatasetLayout:
    """Canonical sharded relative paths for one dataset root."""

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


def canonical_camera_name(name: str) -> str:
    """Normalize short camera roles to canonical observation stream names."""

    value = name.strip()
    if value in {"main", "secondary", "top", "wrist"}:
        value = f"observation.images.{value}"
    if not value.startswith("observation."):
        raise ValueError(f"Camera stream must be an observation path: {name}")
    portable_relative_path(value)
    return value


class EpisodeWriter:
    """Write episode branches with config-locked resume and a marker-last commit."""

    def __init__(
        self,
        dataset_root: str | Path,
        resolved_config: Mapping[str, Any],
        *,
        resume: bool = False,
        chunk_size: int | None = None,
        video_spec: VideoSpec | None = None,
        split_settings: Mapping[str, Any] | None = None,
        arrow_extra_fields: Mapping[str, Mapping[str, str]] | None = None,
        layout_version: str | None = None,
        worker_id: str = "single",
    ):
        self.root = ensure_not_source_path(dataset_root)
        self.config = dict(resolved_config)
        if not _WORKER_ID.fullmatch(worker_id) or worker_id in {".", ".."}:
            raise ValueError(
                "worker_id must be 1-128 portable alphanumeric/dot/underscore/hyphen characters"
            )
        self.worker_id = worker_id

        marker_path = self.root / ResumeGuard.FILE_NAME
        marker_value: Mapping[str, Any] = {}
        if resume and marker_path.is_file():
            marker_value = json.loads(marker_path.read_text(encoding="utf-8"))
        stored_settings = marker_value.get("identity_settings")
        if stored_settings is not None and not isinstance(stored_settings, Mapping):
            raise ResumeMismatchError("Generation marker identity_settings is not a mapping")

        stored_video = (
            stored_settings.get("video") if isinstance(stored_settings, Mapping) else None
        )
        if video_spec is None:
            video_spec = VideoSpec(**dict(stored_video)) if isinstance(stored_video, Mapping) else VideoSpec()
        video_spec.validate()
        if chunk_size is None:
            chunk_size = int(stored_settings.get("chunk_size", 1000)) if stored_settings else 1000
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if split_settings is None:
            split_settings = (
                dict(stored_settings.get("split_settings") or {})
                if stored_settings
                else dict(DEFAULT_SPLIT_SETTINGS)
            )
        normalized_split_settings = {**DEFAULT_SPLIT_SETTINGS, **dict(split_settings)}
        if normalized_split_settings["strategy"] != "leakage_aware_stratified":
            raise ValueError("Unsupported split strategy")
        fractions = sum(
            float(normalized_split_settings[name])
            for name in ("train_fraction", "validation_fraction", "test_fraction")
        )
        if abs(fractions - 1.0) > 1e-9:
            raise ValueError("Split fractions must sum to one")
        normalized_split_settings["seed"] = int(normalized_split_settings["seed"])
        for name in ("train_fraction", "validation_fraction", "test_fraction"):
            normalized_split_settings[name] = float(normalized_split_settings[name])
        if layout_version is None:
            layout_version = (
                str(stored_settings.get("layout_version"))
                if stored_settings and stored_settings.get("layout_version")
                else WRITER_LAYOUT_VERSION
            )
        if layout_version != WRITER_LAYOUT_VERSION:
            raise ValueError(f"Unsupported writer layout version: {layout_version}")

        stored_arrow = (
            stored_settings.get("arrow_schema")
            if isinstance(stored_settings, Mapping)
            else None
        )
        configured_arrow_extras = self.config.get("arrow_extra_fields")
        if arrow_extra_fields is None:
            if configured_arrow_extras is not None:
                arrow_extra_fields = configured_arrow_extras
            elif isinstance(stored_arrow, Mapping):
                arrow_extra_fields = stored_arrow.get("extra_fields")
        self.arrow_extra_fields = normalize_extra_field_declarations(arrow_extra_fields)
        arrow_identity = arrow_schema_identity(self.arrow_extra_fields)

        self.writer_settings = {
            "layout_version": layout_version,
            "chunk_size": chunk_size,
            "video": asdict(video_spec),
            "split_settings": normalized_split_settings,
            "required_camera_streams": [
                "observation.images.main",
                "observation.images.secondary",
            ],
            "arrow_schema": {
                "version": ARROW_SIDECAR_SCHEMA_VERSION,
                "identity_sha256": arrow_identity,
                "extra_fields": self.arrow_extra_fields,
            },
        }
        # A legacy run has no recorded storage identity.  It remains readable
        # and finalizable under the historical config hash, but all newly
        # created runs bind the full writer settings into resume identity.
        guard_identity = None if resume and marker_path.is_file() and stored_settings is None else self.writer_settings
        self.config_hash = ResumeGuard(
            self.root,
            self.config,
            resume=resume,
            identity_settings=guard_identity,
        ).initialize()
        self.resume = resume
        self.layout = DatasetLayout(self.root, chunk_size)
        self.video_spec = video_spec
        (self.root / ".records").mkdir(parents=True, exist_ok=True)
        self._worker_staging_root = self.root / ".staging" / "workers" / worker_id
        self._worker_staging_root.mkdir(parents=True, exist_ok=True)

    @property
    def seal_path(self) -> Path:
        return self.root / ".seal.json"

    @contextmanager
    def _dataset_lock(self, *, exclusive: bool) -> Iterator[None]:
        """Coordinate episode publication with the irreversible seal."""

        lock_path = self.root / ".dataset.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _assert_not_sealed(self) -> None:
        if self.seal_path.is_file() or (self.root / "meta" / ".complete.json").is_file():
            raise DatasetSealedError(f"Dataset is sealed; episode commits are forbidden: {self.root}")

    def _marker(self, episode_uuid: str) -> Path:
        safe = self._episode_storage_key(episode_uuid)
        return self.root / ".records" / f"{safe}.json"

    def _transaction_dir(self, episode_uuid: str) -> Path:
        safe = self._episode_storage_key(episode_uuid)
        staging_root = self._worker_staging_root.resolve()
        candidate = (staging_root / safe).resolve()
        if candidate.parent != staging_root:
            raise ValueError("Episode staging path escaped its private root")
        return candidate

    @staticmethod
    def _episode_storage_key(episode_uuid: str) -> str:
        try:
            parsed = uuid.UUID(episode_uuid)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("episode_uuid must be a canonical UUID before filesystem use") from error
        canonical = str(parsed)
        if canonical != episode_uuid:
            raise ValueError("episode_uuid must use canonical lowercase UUID text")
        return canonical

    def _existing_record(self, record: EpisodeRecord) -> EpisodeRecord | None:
        marker = self._marker(record.episode_uuid)
        if not marker.exists():
            return None
        if not self.resume:
            raise ExistingOutputError(f"Episode already committed: {record.episode_uuid}")
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value.get("config_hash") != self.config_hash:
            raise ResumeMismatchError(f"Episode config hash differs: {record.episode_uuid}")
        shutil.rmtree(self._transaction_dir(record.episode_uuid), ignore_errors=True)
        return EpisodeRecord.from_dict(value["episode"])

    def _commit_transaction(self, transaction_dir: Path, transaction: Mapping[str, Any]) -> EpisodeRecord:
        """Publish staged files idempotently and write the episode marker last."""

        if transaction.get("config_hash") != self.config_hash:
            raise ResumeMismatchError("Staged transaction configuration differs from this run")
        record = EpisodeRecord.from_dict(transaction["episode"])
        expected_hashes = dict(transaction["content_hashes"])
        for relative, expected_hash in sorted(expected_hashes.items()):
            staged = resolve_dataset_path(transaction_dir, relative)
            if not staged.is_file() or sha256_file(staged) != expected_hash:
                raise RuntimeError(f"Staged artifact is missing or corrupt: {relative}")
            destination = resolve_dataset_path(self.root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(destination) != expected_hash:
                    raise ExistingOutputError(f"Conflicting artifact will not be overwritten: {destination}")
                continue
            try:
                os.link(staged, destination)
            except FileExistsError:
                if sha256_file(destination) != expected_hash:
                    raise ExistingOutputError(f"Concurrent conflicting artifact: {destination}")
        marker = self._marker(record.episode_uuid)
        if marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("config_hash") != self.config_hash or existing.get("episode") != record.to_dict():
                raise ExistingOutputError(f"Conflicting episode marker: {marker}")
        else:
            atomic_write_json(marker, {"config_hash": self.config_hash, "episode": record.to_dict()})
        shutil.rmtree(transaction_dir, ignore_errors=True)
        return record

    def _resume_transaction(self, record: EpisodeRecord) -> EpisodeRecord | None:
        transaction_dir = self._transaction_dir(record.episode_uuid)
        if not transaction_dir.exists():
            return None
        manifest = transaction_dir / ".transaction.json"
        if not self.resume:
            raise ExistingOutputError(f"Incomplete staged episode exists: {transaction_dir}")
        if not manifest.is_file():
            # No files were published before the transaction manifest, so this
            # private staging directory can be rebuilt safely after a crash.
            shutil.rmtree(transaction_dir)
            return None
        transaction = json.loads(manifest.read_text(encoding="utf-8"))
        staged_episode = transaction.get("episode", {})
        if (
            staged_episode.get("episode_uuid") != record.episode_uuid
            or int(staged_episode.get("episode_index", -1)) != record.episode_index
        ):
            raise ResumeMismatchError(f"Staged transaction identity differs: {record.episode_uuid}")
        return self._commit_transaction(transaction_dir, transaction)

    def write_episode(
        self,
        record: EpisodeRecord,
        *,
        frame_rows: Iterable[Mapping[str, Any]],
        videos: Mapping[str, str | Path | Iterable[Any]],
        high_rate_rows: Iterable[Mapping[str, Any]] = (),
        event_rows: Iterable[Mapping[str, Any]] = (),
        transition_rows: Iterable[Mapping[str, Any]] = (),
        object_state_rows: Iterable[Mapping[str, Any]] = (),
        camera_calibration_ids: Mapping[str, str] | None = None,
    ) -> EpisodeRecord:
        """Write one episode while holding the shared publication lock."""

        with self._dataset_lock(exclusive=False):
            self._assert_not_sealed()
            return self._write_episode_unlocked(
                record,
                frame_rows=frame_rows,
                videos=videos,
                high_rate_rows=high_rate_rows,
                event_rows=event_rows,
                transition_rows=transition_rows,
                object_state_rows=object_state_rows,
                camera_calibration_ids=camera_calibration_ids,
            )

    def _write_episode_unlocked(
        self,
        record: EpisodeRecord,
        *,
        frame_rows: Iterable[Mapping[str, Any]],
        videos: Mapping[str, str | Path | Iterable[Any]],
        high_rate_rows: Iterable[Mapping[str, Any]] = (),
        event_rows: Iterable[Mapping[str, Any]] = (),
        transition_rows: Iterable[Mapping[str, Any]] = (),
        object_state_rows: Iterable[Mapping[str, Any]] = (),
        camera_calibration_ids: Mapping[str, str] | None = None,
    ) -> EpisodeRecord:
        """Write one episode. Existing committed output is only reused under resume."""

        # Bind the record to this writer configuration and validate identity
        # before deriving marker/staging paths from it.
        record.config_hash = self.config_hash
        record.validate()
        existing = self._existing_record(record)
        if existing is not None:
            return existing
        staged = self._resume_transaction(record)
        if staged is not None:
            return staged
        if not videos:
            raise ValueError("At least one observation video is required")
        index = record.episode_index
        paths = {
            "frame": self.layout.frame_data(index),
            "high_rate": self.layout.high_rate(index),
            "events": self.layout.events(index),
            "transitions": self.layout.transitions(index),
            "objects": self.layout.object_states(index),
        }
        video_paths = {canonical_camera_name(name): self.layout.video(name, index) for name in videos}
        required_cameras = {
            "observation.images.main",
            "observation.images.secondary",
        }
        missing_cameras = sorted(required_cameras - set(video_paths))
        if missing_cameras:
            raise ValueError(f"Canonical episodes require two synchronized views; missing {missing_cameras}")
        every_output = [*paths.values(), *video_paths.values()]
        transaction_dir = self._transaction_dir(record.episode_uuid)
        transaction_dir.mkdir(parents=False, exist_ok=False)

        frames = [dict(row) for row in frame_rows]
        high_rate = [dict(row) for row in high_rate_rows]
        events = [_validated_contact_row(row) for row in event_rows]
        transitions = [dict(row) for row in transition_rows]
        objects = [dict(row) for row in object_state_rows]
        if not frames:
            raise ValueError("frame_rows cannot be empty")
        timestamps = [float(row["timestamp"]) for row in frames]
        validate_monotonic_timestamps(timestamps, name="frame_rows.timestamp")
        source_scenario = record.extras.get("source_scenario_spec")
        if isinstance(source_scenario, Mapping):
            physics = source_scenario.get("physics")
            if not isinstance(physics, Mapping):
                raise ValueError("Source scenario physics must be persisted with the rollout")
            simulation_hz_raw = physics.get("simulation_hz", physics.get("sim_hz"))
            try:
                simulation_hz = float(simulation_hz_raw)
                planned_duration_s = float(source_scenario["duration_s"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Source scenario must declare duration_s and simulation_hz"
                ) from error
            if not math.isfinite(simulation_hz) or simulation_hz <= 0:
                raise ValueError("Source scenario simulation_hz must be finite and positive")
            validate_persisted_render_schedule(
                frames,
                duration_s=planned_duration_s,
                fps_num=self.video_spec.fps_num,
                fps_den=self.video_spec.fps_den,
                maximum_sample_error_s=1.0 / simulation_hz + 1e-12,
            )
        for expected_index, row in enumerate(frames):
            if int(row.get("frame_index", expected_index)) != expected_index:
                raise ValueError("Frame rows must have contiguous frame_index values")
            row.setdefault("frame_index", expected_index)
            row.setdefault("video_frame_index", expected_index)
            row.setdefault("episode_index", index)
            if int(row["episode_index"]) != index:
                raise ValueError("Frame row episode_index differs from its episode")
            if record.task_index is None:
                raise ValueError("Episode task_index must be assigned before writing frame rows")
            row.setdefault("task_index", record.task_index)
            if int(row["task_index"]) != record.task_index:
                raise ValueError("Frame row task_index differs from episode metadata")
        for row in high_rate:
            row.setdefault("episode_index", index)
            if int(row["episode_index"]) != index:
                raise ValueError("High-rate row episode_index differs from its episode")
        if high_rate:
            validate_monotonic_timestamps(
                [float(row["timestamp"]) for row in high_rate],
                name="high_rate.timestamp",
            )
        for row in events:
            row.setdefault("episode_index", index)
            if int(row["episode_index"]) != index:
                raise ValueError("Contact row episode_index differs from its episode")
        if events:
            validate_monotonic_timestamps(
                [float(row["timestamp"]) for row in events],
                strictly=False,
                name="events.timestamp",
            )
        for row in transitions:
            row.setdefault("episode_index", index)
            if int(row["episode_index"]) != index:
                raise ValueError("Transition row episode_index differs from its episode")
            if "timestamp" not in row or not str(row.get("event_type", "")).strip():
                raise ValueError("Transition rows require timestamp and event_type")
        if transitions:
            validate_monotonic_timestamps(
                [float(row["timestamp"]) for row in transitions],
                strictly=False,
                name="transitions.timestamp",
            )
        for row in objects:
            row.setdefault("episode_index", index)
            if int(row["episode_index"]) != index:
                raise ValueError("Object-state row episode_index differs from its episode")
            if "timestamp" not in row or not str(row.get("object_id", "")).strip():
                raise ValueError("Object-state rows require timestamp and object_id")
        if objects:
            validate_monotonic_timestamps(
                [float(row["timestamp"]) for row in objects],
                strictly=False,
                name="object_states.timestamp",
            )

        probes: dict[str, VideoProbe] = {}
        pts_by_camera: dict[str, Sequence[float]] = {}
        for source_name, source in videos.items():
            camera = canonical_camera_name(source_name)
            target = resolve_dataset_path(transaction_dir, video_paths[camera])
            if isinstance(source, (str, Path)):
                source_probe = probe_video(source)
                validate_video_probe(source_probe, self.video_spec, expected_frames=len(frames))
                atomic_copy(source, target)
            else:
                encode_video(source, target, self.video_spec)
            probes[camera] = probe_video(target)
            pts_by_camera[camera] = probe_frame_timestamps(target)
        validate_synchronized_streams(dict(pts_by_camera), tolerance_s=1e-6)
        reference_pts = next(iter(pts_by_camera.values()))
        if len(reference_pts) != len(timestamps):
            raise ValueError("Frame table and encoded video have different frame counts")
        # Frame timestamps must represent actual encoded PTS, not an assumed index/FPS.
        worst_timestamp_error = max(abs(left - right) for left, right in zip(reference_pts, timestamps))
        if worst_timestamp_error > 1e-6:
            raise ValueError(
                f"Frame timestamps differ from encoded PTS by up to {worst_timestamp_error}s"
            )

        pa, _ = _pyarrow()
        write_parquet_atomic(
            resolve_dataset_path(transaction_dir, paths["frame"]),
            canonical_sidecar_table(
                pa,
                "frame",
                frames,
                declared_extra_fields=self.arrow_extra_fields["frame"],
                source_scenario=(source_scenario if isinstance(source_scenario, Mapping) else None),
            ),
        )
        write_parquet_atomic(
            resolve_dataset_path(transaction_dir, paths["high_rate"]),
            canonical_sidecar_table(
                pa,
                "high_rate",
                high_rate,
                declared_extra_fields=self.arrow_extra_fields["high_rate"],
                source_scenario=(source_scenario if isinstance(source_scenario, Mapping) else None),
            ),
        )
        write_parquet_atomic(
            resolve_dataset_path(transaction_dir, paths["events"]),
            canonical_sidecar_table(
                pa,
                "events",
                events,
                declared_extra_fields=self.arrow_extra_fields["events"],
                source_scenario=(source_scenario if isinstance(source_scenario, Mapping) else None),
            ),
        )
        write_parquet_atomic(
            resolve_dataset_path(transaction_dir, paths["transitions"]),
            canonical_sidecar_table(
                pa,
                "transitions",
                transitions,
                declared_extra_fields=self.arrow_extra_fields["transitions"],
                source_scenario=(source_scenario if isinstance(source_scenario, Mapping) else None),
            ),
        )
        write_parquet_atomic(
            resolve_dataset_path(transaction_dir, paths["objects"]),
            canonical_sidecar_table(
                pa,
                "objects",
                objects,
                declared_extra_fields=self.arrow_extra_fields["objects"],
                source_scenario=(source_scenario if isinstance(source_scenario, Mapping) else None),
            ),
        )

        record.video_paths = video_paths
        record.frame_data_path = paths["frame"]
        record.high_rate_path = paths["high_rate"]
        record.events_path = paths["events"]
        record.transition_events_path = paths["transitions"]
        record.object_states_path = paths["objects"]
        if camera_calibration_ids is None:
            calibration_mapping = {stream: stream for stream in video_paths}
        else:
            calibration_mapping = {
                canonical_camera_name(stream): str(identifier)
                for stream, identifier in camera_calibration_ids.items()
            }
            if set(calibration_mapping) != set(video_paths):
                raise ValueError(
                    "camera_calibration_ids must map every encoded video stream exactly once"
                )
            if any(not identifier.strip() for identifier in calibration_mapping.values()):
                raise ValueError("camera calibration IDs must be non-empty")
        record.camera_stream_calibration_ids = calibration_mapping
        record.camera_ids = sorted(set(calibration_mapping.values()))
        record.frame_count = len(frames)
        encoded_duration_s = float(
            reference_pts[-1]
            - reference_pts[0]
            + self.video_spec.fps_den / self.video_spec.fps_num
        )
        if isinstance(source_scenario, Mapping):
            # The scenario's logical duration is the immutable physics-plan
            # duration.  A half-open round(duration*fps) video clock can differ
            # from it by up to half a frame; replacing it with encoded duration
            # would silently change the planned rollout identity.
            planned_duration_s = float(source_scenario["duration_s"])
            if abs(encoded_duration_s - planned_duration_s) > (
                0.5 * self.video_spec.fps_den / self.video_spec.fps_num + 1e-9
            ):
                raise ValueError(
                    "Encoded frame duration is inconsistent with source scenario duration"
                )
            if record.duration_s is not None and abs(
                float(record.duration_s) - planned_duration_s
            ) > 1e-9:
                raise ValueError("Episode duration differs from its source scenario")
            record.duration_s = planned_duration_s
        else:
            record.duration_s = encoded_duration_s
        record.content_hashes = {
            relative: sha256_file(resolve_dataset_path(transaction_dir, relative)) for relative in every_output
        }
        backend_provenance = record.extras.get("backend_provenance")
        if (
            isinstance(backend_provenance, Mapping)
            and backend_provenance.get("backend") == "source_mujoco"
        ):
            visibility_hash = str(
                record.extras.get("visibility_qc_sha256") or ""
            )
            camera_rows = record.extras.get("camera_calibrations")
            if not isinstance(camera_rows, Sequence) or isinstance(
                camera_rows, (str, bytes, bytearray)
            ):
                raise ValueError(
                    "source_mujoco visibility binding requires camera calibration rows"
                )
            binding = source_mujoco_visibility_media_binding(
                visibility_qc_sha256=visibility_hash,
                camera_rows=camera_rows,
                camera_stream_calibration_ids=record.camera_stream_calibration_ids,
                video_paths=record.video_paths,
                content_hashes=record.content_hashes,
            )
            record.extras["visibility_media_binding"] = binding
            record.extras["visibility_media_binding_sha256"] = sha256_json(
                binding
            )
        record.validate()
        atomic_write_json(
            transaction_dir / ".transaction.json",
            {
                "config_hash": self.config_hash,
                "episode": record.to_dict(),
                "content_hashes": record.content_hashes,
            },
        )
        transaction = json.loads((transaction_dir / ".transaction.json").read_text(encoding="utf-8"))
        return self._commit_transaction(transaction_dir, transaction)

    def records(self) -> list[EpisodeRecord]:
        """Read every committed marker in episode-index order."""

        records = [
            EpisodeRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))["episode"])
            for path in sorted((self.root / ".records").glob("*.json"))
        ]
        records.sort(key=lambda record: record.episode_index)
        validate_episode_records(records)
        return records

    def _validate_meta_transaction(self, staging: Path, meta: Path) -> dict[str, str]:
        """Preflight every staged byte and destination before sealing."""

        manifest_path = staging / ".transaction.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Metadata transaction manifest is missing: {manifest_path}")
        transaction = json.loads(manifest_path.read_text(encoding="utf-8"))
        if transaction.get("config_hash") != self.config_hash:
            raise ResumeMismatchError("Staged metadata configuration differs from this run")
        content_hashes = dict(transaction.get("content_hashes", {}))
        if set(content_hashes) != FINALIZED_METADATA_ARTIFACTS:
            raise RuntimeError(
                "Metadata transaction must hash every finalized artifact exactly: "
                f"missing={sorted(FINALIZED_METADATA_ARTIFACTS - set(content_hashes))}, "
                f"extra={sorted(set(content_hashes) - FINALIZED_METADATA_ARTIFACTS)}"
            )
        for name, expected_hash in sorted(content_hashes.items()):
            source = staging / name
            if not source.is_file() or sha256_file(source) != expected_hash:
                raise RuntimeError(f"Staged metadata is missing or corrupt: {source}")
            destination = meta / name
            if destination.exists():
                if sha256_file(destination) != expected_hash:
                    raise ExistingOutputError(
                        f"Conflicting finalized metadata will not be overwritten: {destination}"
                    )
        return {str(name): str(value) for name, value in content_hashes.items()}

    def _commit_meta_transaction(self, staging: Path, meta: Path) -> None:
        """Idempotently publish a fully staged metadata set and mark it last."""

        content_hashes = self._validate_meta_transaction(staging, meta)
        meta.mkdir(parents=True, exist_ok=True)
        for name, expected_hash in sorted(content_hashes.items()):
            source = staging / name
            destination = meta / name
            if destination.exists():
                continue
            try:
                os.link(source, destination)
            except FileExistsError:
                if sha256_file(destination) != expected_hash:
                    raise ExistingOutputError(f"Concurrent metadata conflict: {destination}")
        completion = meta / ".complete.json"
        marker_value = {
            "config_hash": self.config_hash,
            "content_hashes": content_hashes,
            "seal_sha256": sha256_file(self.seal_path),
        }
        if completion.exists():
            if json.loads(completion.read_text(encoding="utf-8")) != marker_value:
                raise ExistingOutputError(f"Conflicting metadata completion marker: {completion}")
        else:
            atomic_write_json(completion, marker_value)
        shutil.rmtree(staging)

    def finalize(
        self,
        info: DatasetInfo,
        *,
        tasks: Iterable[Mapping[str, Any]] | None = None,
        cameras: Iterable[Mapping[str, Any]] = (),
        provenance: Iterable[Mapping[str, Any]] = (),
        splits: Iterable[Mapping[str, Any]] = (),
        counterfactual_families: Iterable[Mapping[str, Any]] | None = None,
        expected_episode_membership: Mapping[str, int] | None = None,
        run_plan_sha256: str | None = None,
    ) -> list[EpisodeRecord]:
        """Seal and finalize exactly one immutable set of episode markers.

        The exclusive dataset lock closes the race in which a worker could
        publish a marker after finalization counted episodes.  Once the seal is
        written, episode publication is permanently rejected; an interrupted
        metadata transaction may only be resumed with the same semantic
        finalization context.
        """

        task_rows = None if tasks is None else [dict(row) for row in tasks]
        camera_rows = [dict(row) for row in cameras]
        provenance_rows = [dict(row) for row in provenance]
        split_rows = [dict(row) for row in splits]
        counterfactual_rows = (
            None
            if counterfactual_families is None
            else [dict(row) for row in counterfactual_families]
        )
        with self._dataset_lock(exclusive=True):
            records = self.records()
            actual_membership = {
                record.episode_uuid: record.episode_index for record in records
            }
            expected_membership = (
                actual_membership
                if expected_episode_membership is None
                else {str(key): int(value) for key, value in expected_episode_membership.items()}
            )
            for episode_uuid in expected_membership:
                self._episode_storage_key(episode_uuid)
            if actual_membership != expected_membership:
                missing = sorted(set(expected_membership) - set(actual_membership))
                unexpected = sorted(set(actual_membership) - set(expected_membership))
                changed = sorted(
                    episode_uuid
                    for episode_uuid in set(actual_membership) & set(expected_membership)
                    if actual_membership[episode_uuid] != expected_membership[episode_uuid]
                )
                raise RuntimeError(
                    "Cannot finalize partial or changed plan membership: "
                    f"missing={missing}, unexpected={unexpected}, changed_indices={changed}"
                )
            if run_plan_sha256 is not None and (
                len(run_plan_sha256) != 64
                or any(character not in "0123456789abcdef" for character in run_plan_sha256)
            ):
                raise ValueError("run_plan_sha256 must be a lowercase SHA-256 digest")

            completion = self.root / "meta" / ".complete.json"
            if (self.seal_path.exists() or completion.exists()) and not self.resume:
                raise ExistingOutputError(f"Dataset is already sealed: {self.root}")
            info_for_identity = info.to_dict()
            info_for_identity.pop("dataset_uuid", None)
            info_for_identity.pop("created_at", None)
            seal_value = {
                "schema_version": "dynamic-robot-dataset-seal/v1",
                "config_hash": self.config_hash,
                "writer_settings_sha256": sha256_json(self.writer_settings),
                "run_plan_sha256": run_plan_sha256,
                "expected_episode_membership": [
                    {"episode_uuid": episode_uuid, "episode_index": expected_membership[episode_uuid]}
                    for episode_uuid in sorted(expected_membership)
                ],
                "finalization_context_sha256": sha256_json(
                    {
                        "info": info_for_identity,
                        "tasks": task_rows,
                        "cameras": camera_rows,
                        "provenance": provenance_rows,
                        "splits": split_rows,
                        "counterfactual_families": counterfactual_rows,
                    }
                ),
            }
            seal_preexisted = self.seal_path.is_file()
            if seal_preexisted:
                existing = json.loads(self.seal_path.read_text(encoding="utf-8"))
                if existing != seal_value:
                    raise ResumeMismatchError("Dataset seal differs from finalization request")
                return self._finalize_unlocked(
                    info,
                    tasks=task_rows,
                    cameras=camera_rows,
                    provenance=provenance_rows,
                    splits=split_rows,
                    counterfactual_families=counterfactual_rows,
                )

            # The irreversible seal is marker-last with respect to validation:
            # construct all Arrow artifacts, hash them, and verify that their
            # destinations are conflict-free before publishing the seal.
            records = self._finalize_unlocked(
                info,
                tasks=task_rows,
                cameras=camera_rows,
                provenance=provenance_rows,
                splits=split_rows,
                counterfactual_families=counterfactual_rows,
                defer_commit=True,
            )
            staging = self.root / ".staging" / "_meta"
            meta = self.root / "meta"
            self._validate_meta_transaction(staging, meta)
            atomic_write_json(self.seal_path, seal_value)
            self._commit_meta_transaction(staging, meta)
            return records

    def _finalize_unlocked(
        self,
        info: DatasetInfo,
        *,
        tasks: Iterable[Mapping[str, Any]] | None = None,
        cameras: Iterable[Mapping[str, Any]] = (),
        provenance: Iterable[Mapping[str, Any]] = (),
        splits: Iterable[Mapping[str, Any]] = (),
        counterfactual_families: Iterable[Mapping[str, Any]] | None = None,
        defer_commit: bool = False,
    ) -> list[EpisodeRecord]:
        """Aggregate markers into a resumable, marker-last metadata transaction."""

        records = self.records()
        if not records:
            raise ValueError("Cannot finalize a dataset with no committed episodes")
        meta = self.root / "meta"
        completion = meta / ".complete.json"
        if tasks is None:
            unique = sorted({(record.family, record.subfamily) for record in records})
            tasks = [
                {"task_index": index, "family": family, "subfamily": subfamily}
                for index, (family, subfamily) in enumerate(unique)
            ]
        split_rows = list(splits)
        existing_splits = meta / "splits.parquet"
        if not split_rows and existing_splits.is_file():
            split_rows = read_parquet_rows(existing_splits)
        if not split_rows:
            from .splits import SplitAssigner

            settings = self.writer_settings["split_settings"]
            split_rows = [
                {
                    "episode_uuid": assignment.episode_uuid,
                    "episode_index": assignment.episode_index,
                    "split_group_id": assignment.split_group_id,
                    "split": assignment.split,
                }
                for assignment in SplitAssigner(
                    seed=settings["seed"],
                    train_fraction=settings["train_fraction"],
                    validation_fraction=settings["validation_fraction"],
                    test_fraction=settings["test_fraction"],
                ).assign(records)
            ]
        split_by_uuid = {str(row["episode_uuid"]): row for row in split_rows}
        if len(split_by_uuid) != len(split_rows):
            raise ValueError("Split rows contain duplicate episode UUIDs")
        record_by_uuid = {record.episode_uuid: record for record in records}
        if set(split_by_uuid) != set(record_by_uuid):
            raise ValueError("Split rows must match finalized episode membership exactly")
        for record in records:
            assignment = split_by_uuid.get(record.episode_uuid)
            assert assignment is not None
            if int(assignment["episode_index"]) != record.episode_index:
                raise ValueError(f"Split episode index mismatch: {record.episode_uuid}")
            record.split = Split(str(assignment["split"]))
            record.split_group_id = str(assignment["split_group_id"])
        task_rows = list(tasks)
        task_index_by_key: dict[tuple[str, str], int] = {}
        seen_task_indices: set[int] = set()
        for row in task_rows:
            key = (str(row["family"]), str(row["subfamily"]))
            task_index = int(row["task_index"])
            if key in task_index_by_key or task_index in seen_task_indices:
                raise ValueError("Task rows require unique family/subfamily keys and task_index values")
            task_index_by_key[key] = task_index
            seen_task_indices.add(task_index)
        for record in records:
            expected_task_index = task_index_by_key.get((record.family, record.subfamily))
            if expected_task_index is None:
                raise ValueError(
                    f"No task row for episode family/subfamily: {record.family}/{record.subfamily}"
                )
            if record.task_index != expected_task_index:
                raise ValueError(
                    f"Episode task_index mismatch for {record.episode_uuid}: "
                    f"{record.task_index} != {expected_task_index}"
                )
        camera_rows = list(cameras)
        provenance_rows = list(provenance)
        counterfactual_rows = (
            list(counterfactual_families)
            if counterfactual_families is not None
            else None
        )
        if counterfactual_rows is None:
            from .contract_v2 import build_counterfactual_family_records

            counterfactual_rows = [
                declaration.to_table_row()
                for declaration in build_counterfactual_family_records(records)
            ]
        expected_names = tuple(sorted(FINALIZED_METADATA_ARTIFACTS))
        if completion.is_file():
            marker = json.loads(completion.read_text(encoding="utf-8"))
            if marker.get("config_hash") != self.config_hash:
                raise ResumeMismatchError("Finalized metadata configuration differs from this run")
            marker_hashes = dict(marker.get("content_hashes", {}))
            if set(marker_hashes) != FINALIZED_METADATA_ARTIFACTS:
                raise RuntimeError("Finalized completion marker has incomplete artifact membership")
            if marker.get("seal_sha256") not in {None, sha256_file(self.seal_path)}:
                raise RuntimeError("Finalized completion marker is bound to a different dataset seal")
            for name, expected_hash in marker_hashes.items():
                path = meta / name
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise RuntimeError(f"Finalized metadata is incomplete or corrupt: {path}")
            if not self.resume:
                raise ExistingOutputError(f"Dataset metadata already finalized: {meta}")
            return records

        staging = self.root / ".staging" / "_meta"
        if staging.exists():
            if not self.resume:
                raise ExistingOutputError(f"Incomplete metadata transaction exists: {staging}")
            if (staging / ".transaction.json").is_file():
                if defer_commit:
                    self._validate_meta_transaction(staging, meta)
                    return records
                self._commit_meta_transaction(staging, meta)
                return records
            shutil.rmtree(staging)
        staging.mkdir(parents=False)
        info_value = info.to_dict()
        existing_info_path = meta / "info.json"
        if existing_info_path.is_file():
            existing_info = json.loads(existing_info_path.read_text(encoding="utf-8"))
            comparable = dict(info_value)
            for volatile_field in ("dataset_uuid", "created_at"):
                comparable[volatile_field] = existing_info.get(volatile_field)
            if comparable != existing_info:
                raise ExistingOutputError(
                    f"Existing partial dataset info differs and will not be overwritten: {existing_info_path}"
                )
            info_value = existing_info
        atomic_write_json(staging / "info.json", info_value)
        write_parquet_atomic(
            staging / "episodes.parquet",
            _episode_metadata_table(records),
        )
        write_parquet_atomic(staging / "tasks.parquet", flexible_mapping_rows(task_rows))
        write_parquet_atomic(staging / "cameras.parquet", flexible_mapping_rows(camera_rows))
        write_parquet_atomic(
            staging / "provenance.parquet", flexible_mapping_rows(provenance_rows)
        )
        write_parquet_atomic(staging / "splits.parquet", flexible_mapping_rows(split_rows))
        if counterfactual_rows:
            write_parquet_atomic(
                staging / "counterfactual_families.parquet",
                counterfactual_rows,
            )
        else:
            pa, _ = _pyarrow()
            write_parquet_atomic(
                staging / "counterfactual_families.parquet",
                [],
                schema=pa.schema(
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
                ),
            )
        content_hashes = {name: sha256_file(staging / name) for name in expected_names}
        atomic_write_json(
            staging / ".transaction.json",
            {"config_hash": self.config_hash, "content_hashes": content_hashes},
        )
        if defer_commit:
            return records
        self._commit_meta_transaction(staging, meta)
        return records


def load_episode_records(dataset_root: str | Path) -> list[EpisodeRecord]:
    """Load finalized metadata, falling back to marker files before finalization."""

    root = Path(dataset_root).resolve(strict=True)
    finalized = root / "meta" / "episodes.parquet"
    completion_path = root / "meta" / ".complete.json"
    finalized_complete = completion_path.is_file()
    if finalized_complete:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        required = FINALIZED_METADATA_ARTIFACTS
        hashes = dict(completion.get("content_hashes") or {})
        if not required.issubset(hashes):
            raise RuntimeError("Finalized metadata completion marker is missing required hashes")
        for name, expected_hash in hashes.items():
            if Path(name).name != name:
                raise RuntimeError(f"Invalid metadata filename in completion marker: {name}")
            path = root / "meta" / name
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"Finalized metadata hash mismatch: {path}")
        records = [episode_record_from_table_row(row) for row in read_parquet_rows(finalized)]
    else:
        markers = sorted((root / ".records").glob("*.json"))
        records = [EpisodeRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))["episode"]) for path in markers]
    split_path = root / "meta" / "splits.parquet"
    # Before finalization, build-splits may intentionally create only this
    # table. If episodes.parquet also exists without its marker, the meta tree
    # is a crashed transaction and none of it is authoritative.
    if split_path.is_file() and (finalized_complete or not finalized.exists()):
        assignments = {str(row["episode_uuid"]): row for row in read_parquet_rows(split_path)}
        for record in records:
            assignment = assignments.get(record.episode_uuid)
            if assignment is not None:
                record.split = Split(str(assignment["split"]))
                record.split_group_id = str(assignment["split_group_id"])
    return records
