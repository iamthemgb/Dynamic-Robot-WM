"""Atomic episode artifacts and final MP4/Parquet dataset layout."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import sha256_file
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
from .synchronization import validate_monotonic_timestamps, validate_synchronized_streams
from .video_writer import VideoProbe, VideoSpec, encode_video, probe_frame_timestamps, probe_video, validate_video_probe


class MissingParquetDependency(RuntimeError):
    """Canonical Parquet I/O requires the optional PyArrow dependency."""


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


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a Parquet table into Python records."""

    _, pq = _pyarrow()
    return pq.read_table(Path(path)).to_pylist()


_FLEXIBLE_EPISODE_FIELDS = (
    "physics",
    "assistance",
    "objective_metrics",
    "randomization",
    "content_hashes",
    "video_paths",
    "asset_hashes",
    "extras",
)


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
        chunk_size: int = 1000,
        video_spec: VideoSpec = VideoSpec(),
    ):
        self.root = ensure_not_source_path(dataset_root)
        self.config = dict(resolved_config)
        self.config_hash = ResumeGuard(self.root, self.config, resume=resume).initialize()
        self.resume = resume
        self.layout = DatasetLayout(self.root, chunk_size)
        self.video_spec = video_spec
        (self.root / ".records").mkdir(parents=True, exist_ok=True)
        (self.root / ".staging").mkdir(parents=True, exist_ok=True)

    def _marker(self, episode_uuid: str) -> Path:
        safe = self._episode_storage_key(episode_uuid)
        return self.root / ".records" / f"{safe}.json"

    def _transaction_dir(self, episode_uuid: str) -> Path:
        safe = self._episode_storage_key(episode_uuid)
        staging_root = (self.root / ".staging").resolve()
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
        object_state_rows: Iterable[Mapping[str, Any]] = (),
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
        objects = [dict(row) for row in object_state_rows]
        if not frames:
            raise ValueError("frame_rows cannot be empty")
        timestamps = [float(row["timestamp"]) for row in frames]
        validate_monotonic_timestamps(timestamps, name="frame_rows.timestamp")
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

        write_parquet_atomic(resolve_dataset_path(transaction_dir, paths["frame"]), frames)
        write_parquet_atomic(
            resolve_dataset_path(transaction_dir, paths["high_rate"]),
            high_rate,
            schema=None if high_rate else _empty_sidecar_schema("high_rate"),
        )
        write_parquet_atomic(
            resolve_dataset_path(transaction_dir, paths["events"]),
            events,
            schema=None if events else _empty_sidecar_schema("events"),
        )
        write_parquet_atomic(
            resolve_dataset_path(transaction_dir, paths["objects"]),
            objects,
            schema=None if objects else _empty_sidecar_schema("objects"),
        )

        record.video_paths = video_paths
        record.frame_data_path = paths["frame"]
        record.high_rate_path = paths["high_rate"]
        record.events_path = paths["events"]
        record.object_states_path = paths["objects"]
        record.camera_ids = sorted(video_paths)
        record.frame_count = len(frames)
        record.duration_s = float(reference_pts[-1] - reference_pts[0] + self.video_spec.fps_den / self.video_spec.fps_num)
        record.content_hashes = {
            relative: sha256_file(resolve_dataset_path(transaction_dir, relative)) for relative in every_output
        }
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

    def _commit_meta_transaction(self, staging: Path, meta: Path) -> None:
        """Idempotently publish a fully staged metadata set and mark it last."""

        manifest_path = staging / ".transaction.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Metadata transaction manifest is missing: {manifest_path}")
        transaction = json.loads(manifest_path.read_text(encoding="utf-8"))
        if transaction.get("config_hash") != self.config_hash:
            raise ResumeMismatchError("Staged metadata configuration differs from this run")
        content_hashes = dict(transaction.get("content_hashes", {}))
        meta.mkdir(parents=True, exist_ok=True)
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
                continue
            try:
                os.link(source, destination)
            except FileExistsError:
                if sha256_file(destination) != expected_hash:
                    raise ExistingOutputError(f"Concurrent metadata conflict: {destination}")
        completion = meta / ".complete.json"
        marker_value = {"config_hash": self.config_hash, "content_hashes": content_hashes}
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

            split_rows = [
                {
                    "episode_uuid": assignment.episode_uuid,
                    "episode_index": assignment.episode_index,
                    "split_group_id": assignment.split_group_id,
                    "split": assignment.split,
                }
                for assignment in SplitAssigner(seed=0).assign(records)
            ]
        split_by_uuid = {str(row["episode_uuid"]): row for row in split_rows}
        for record in records:
            assignment = split_by_uuid.get(record.episode_uuid)
            if assignment is not None:
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
        expected_names = (
            "info.json",
            "episodes.parquet",
            "tasks.parquet",
            "cameras.parquet",
            "provenance.parquet",
            "splits.parquet",
        )
        if completion.is_file():
            marker = json.loads(completion.read_text(encoding="utf-8"))
            if marker.get("config_hash") != self.config_hash:
                raise ResumeMismatchError("Finalized metadata configuration differs from this run")
            for name, expected_hash in marker.get("content_hashes", {}).items():
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
            [episode_record_to_table_row(record) for record in records],
        )
        write_parquet_atomic(staging / "tasks.parquet", flexible_mapping_rows(task_rows))
        write_parquet_atomic(staging / "cameras.parquet", flexible_mapping_rows(camera_rows))
        write_parquet_atomic(
            staging / "provenance.parquet", flexible_mapping_rows(provenance_rows)
        )
        write_parquet_atomic(staging / "splits.parquet", flexible_mapping_rows(split_rows))
        content_hashes = {name: sha256_file(staging / name) for name in expected_names}
        atomic_write_json(
            staging / ".transaction.json",
            {"config_hash": self.config_hash, "content_hashes": content_hashes},
        )
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
        required = {
            "info.json",
            "episodes.parquet",
            "tasks.parquet",
            "cameras.parquet",
            "provenance.parquet",
            "splits.parquet",
        }
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
