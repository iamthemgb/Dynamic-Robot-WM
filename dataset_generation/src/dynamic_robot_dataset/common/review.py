"""Hash-bound qualitative review contract for fixed acceptance rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .episode_writer import (
    FINALIZED_METADATA_ARTIFACTS,
    load_episode_records,
    read_parquet_rows,
)
from .hashing import sha256_file, sha256_json
from .paths import AtomicDirectory, atomic_write_bytes, atomic_write_json, portable_relative_path
from .synchronization import validate_synchronized_streams
from .video_writer import (
    VideoSpec,
    iter_rgb_frames,
    probe_frame_timestamps,
    probe_video,
    validate_video_probe,
)


REVIEW_LEDGER_SCHEMA = "dynamic-robot-human-review-ledger/v2"
REVIEW_ARTIFACT_SCHEMA = "dynamic-robot-review-artifacts/v2"
REVIEW_MEDIA_PACK_SCHEMA = "dynamic-robot-review-media-pack/v1"
STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA = "dynamic-robot-review-source-manifest/v1"
REVIEW_MASTER_SEED = 20260717
REVIEW_CHECKS = (
    "continuity",
    "penetration",
    "grasp_support",
    "rebound_deflection_plausibility",
    "sticking",
    "tunneling",
    "clipping",
    "fixture_support",
    "camera_occlusion",
    "robocasa_intersections",
)
REVIEW_SCENE_SEQUENCE = (
    "clean_R0",
    "robocasa_lab",
    "robocasa_kitchen",
    "robocasa_workbench",
    "robocasa_storage",
    "robocasa_tabletop",
)
EVENT_STRIP_LABELS = (
    "pre_event",
    "event",
    "post_0p1_s",
    "post_0p3_s",
    "final",
)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class EventStripSpec:
    """Event-relative frames required for both synchronized views."""

    pre_event_offset_s: float = -0.1
    post_event_offsets_s: tuple[float, float] = (0.1, 0.3)
    include_event: bool = True
    include_final: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.pre_event_offset_s) or self.pre_event_offset_s >= 0:
            raise ValueError("pre-event strip offset must be finite and negative")
        if len(self.post_event_offsets_s) != 2 or any(
            not math.isfinite(value) or value <= 0
            for value in self.post_event_offsets_s
        ):
            raise ValueError("event strips require two positive post-event offsets")
        if tuple(sorted(self.post_event_offsets_s)) != self.post_event_offsets_s:
            raise ValueError("post-event offsets must be increasing")
        if not self.include_event or not self.include_final:
            raise ValueError("acceptance strips must include event and final frames")


def event_strip_frame_indices(
    timestamps_s: Sequence[float],
    key_event_time_s: float,
    *,
    spec: EventStripSpec = EventStripSpec(),
) -> dict[str, int]:
    """Select nearest persisted frames for pre/event/+0.1/+0.3/final."""

    spec.validate()
    timestamps = tuple(float(value) for value in timestamps_s)
    if not timestamps or any(not math.isfinite(value) for value in timestamps):
        raise ValueError("event strips require finite persisted timestamps")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("event-strip timestamps must be strictly increasing")
    if not math.isfinite(key_event_time_s) or key_event_time_s < 0:
        raise ValueError("key-event time must be finite and non-negative")

    required_pre_context_s = abs(spec.pre_event_offset_s)
    required_post_context_s = max(spec.post_event_offsets_s)
    if key_event_time_s - timestamps[0] < required_pre_context_s - 1e-12:
        raise ValueError(
            "event strip lacks at least 0.1 s of persisted pre-event context"
        )
    if timestamps[-1] - key_event_time_s < required_post_context_s - 1e-12:
        raise ValueError(
            "event strip lacks at least 0.3 s of persisted post-event context"
        )

    def closest(target: float) -> int:
        return min(
            range(len(timestamps)),
            key=lambda index: (abs(timestamps[index] - target), index),
        )

    result = {
        "pre_event": closest(key_event_time_s + spec.pre_event_offset_s),
        "event": closest(key_event_time_s),
        "post_0p1_s": closest(key_event_time_s + spec.post_event_offsets_s[0]),
        "post_0p3_s": closest(key_event_time_s + spec.post_event_offsets_s[1]),
        "final": len(timestamps) - 1,
    }
    ordered = tuple(
        result[name]
        for name in ("pre_event", "event", "post_0p1_s", "post_0p3_s", "final")
    )
    if len(set(ordered)) != len(ordered) or any(
        right <= left for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError(
            "event-strip targets resolve to duplicate, clamped, or non-ordered frames"
        )
    return result


@dataclass(frozen=True, slots=True)
class ReviewMediaPackManifest:
    """Immutable derivation record for the two finalized review strips."""

    case_id: str
    leaf_id: str
    episode_uuid: str
    review_matrix_episode_index: int
    dataset_episode_index: int
    review_plan_sha256: str
    review_case_sha256: str
    review_request_ledger_sha256: str
    review_request_sha256: str
    dataset_seal_sha256: str
    finalized_completion_sha256: str
    finalized_episode_record_sha256: str
    frame_data_path: str
    frame_data_sha256: str
    frame_timestamps_s: tuple[float, ...]
    frame_timestamps_sha256: str
    qc_report_path: str
    qc_report_sha256: str
    qc_episode_result_sha256: str
    key_event_time_s: float
    video_paths: Mapping[str, str]
    video_sha256: Mapping[str, str]
    video_pts_sha256: Mapping[str, str]
    video_frame_count: Mapping[str, int]
    event_strip_paths: Mapping[str, str]
    event_strip_sha256: Mapping[str, str]
    event_strip_indices: Mapping[str, int]
    selected_frame_timestamps_s: Mapping[str, float]
    strip_width_px: int
    strip_height_px: int
    schema_version: str = REVIEW_MEDIA_PACK_SCHEMA

    def validate(self) -> None:
        if self.schema_version != REVIEW_MEDIA_PACK_SCHEMA:
            raise ValueError(f"review media packs must use {REVIEW_MEDIA_PACK_SCHEMA}")
        if not self.case_id or not self.leaf_id or not self.episode_uuid:
            raise ValueError("review media pack identity is incomplete")
        if self.review_matrix_episode_index < 0 or self.dataset_episode_index < 0:
            raise ValueError("review media pack episode indices must be non-negative")
        for label, digest in (
            ("review plan", self.review_plan_sha256),
            ("review case", self.review_case_sha256),
            ("review request ledger", self.review_request_ledger_sha256),
            ("review request", self.review_request_sha256),
            ("dataset seal", self.dataset_seal_sha256),
            ("finalized completion", self.finalized_completion_sha256),
            ("finalized episode", self.finalized_episode_record_sha256),
            ("frame data", self.frame_data_sha256),
            ("frame timestamps", self.frame_timestamps_sha256),
            ("QC report", self.qc_report_sha256),
            ("QC episode", self.qc_episode_result_sha256),
        ):
            if not _valid_sha256(digest):
                raise ValueError(f"review media pack {label} hash is malformed")
        portable_relative_path(self.frame_data_path)
        portable_relative_path(self.qc_report_path)
        timestamps = tuple(float(value) for value in self.frame_timestamps_s)
        if not timestamps or any(not math.isfinite(value) for value in timestamps):
            raise ValueError("review media pack requires finite frame timestamps")
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("review media pack frame timestamps are not increasing")
        if self.frame_timestamps_sha256 != sha256_json(timestamps):
            raise ValueError("review media pack frame timestamp hash disagrees with its values")
        if not math.isfinite(self.key_event_time_s) or self.key_event_time_s < 0:
            raise ValueError("review media pack key-event time is invalid")
        cameras = {"main", "secondary"}
        for label, values in (
            ("video paths", self.video_paths),
            ("video hashes", self.video_sha256),
            ("video PTS hashes", self.video_pts_sha256),
            ("video frame counts", self.video_frame_count),
            ("event-strip paths", self.event_strip_paths),
            ("event-strip hashes", self.event_strip_sha256),
        ):
            if set(values) != cameras:
                raise ValueError(f"review media pack {label} must cover both fixed views")
        for path in (*self.video_paths.values(), *self.event_strip_paths.values()):
            portable_relative_path(path)
        for digest in (
            *self.video_sha256.values(),
            *self.video_pts_sha256.values(),
            *self.event_strip_sha256.values(),
        ):
            if not _valid_sha256(digest):
                raise ValueError("review media pack contains a malformed media hash")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value != len(timestamps)
            for value in self.video_frame_count.values()
        ):
            raise ValueError("review media pack video frame counts differ from frame data")
        if set(self.event_strip_indices) != set(EVENT_STRIP_LABELS):
            raise ValueError("review media pack event-strip index map is incomplete")
        expected_indices = event_strip_frame_indices(timestamps, self.key_event_time_s)
        if dict(self.event_strip_indices) != expected_indices:
            raise ValueError("review media pack event-strip indices are not reproducible")
        if set(self.selected_frame_timestamps_s) != set(EVENT_STRIP_LABELS):
            raise ValueError("review media pack selected timestamps are incomplete")
        for label in EVENT_STRIP_LABELS:
            selected = float(self.selected_frame_timestamps_s[label])
            if (
                not math.isfinite(selected)
                or selected != timestamps[self.event_strip_indices[label]]
            ):
                raise ValueError(
                    "review media pack selected timestamps disagree with frame indices"
                )
        if self.strip_width_px <= 0 or self.strip_height_px <= 0:
            raise ValueError("review media pack strip dimensions must be positive")
        if self.strip_width_px % len(EVENT_STRIP_LABELS):
            raise ValueError("review media pack strip must contain five equal-width panels")

    @property
    def binding_sha256(self) -> str:
        self.validate()
        return sha256_json(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**asdict(self), "binding_sha256": self.binding_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewMediaPackManifest":
        result = cls(
            case_id=str(value.get("case_id", "")),
            leaf_id=str(value.get("leaf_id", "")),
            episode_uuid=str(value.get("episode_uuid", "")),
            review_matrix_episode_index=int(
                value.get("review_matrix_episode_index", -1)
            ),
            dataset_episode_index=int(value.get("dataset_episode_index", -1)),
            review_plan_sha256=str(value.get("review_plan_sha256", "")),
            review_case_sha256=str(value.get("review_case_sha256", "")),
            review_request_ledger_sha256=str(
                value.get("review_request_ledger_sha256", "")
            ),
            review_request_sha256=str(value.get("review_request_sha256", "")),
            dataset_seal_sha256=str(value.get("dataset_seal_sha256", "")),
            finalized_completion_sha256=str(
                value.get("finalized_completion_sha256", "")
            ),
            finalized_episode_record_sha256=str(
                value.get("finalized_episode_record_sha256", "")
            ),
            frame_data_path=str(value.get("frame_data_path", "")),
            frame_data_sha256=str(value.get("frame_data_sha256", "")),
            frame_timestamps_s=tuple(
                float(item) for item in value.get("frame_timestamps_s", ())
            ),
            frame_timestamps_sha256=str(value.get("frame_timestamps_sha256", "")),
            qc_report_path=str(value.get("qc_report_path", "")),
            qc_report_sha256=str(value.get("qc_report_sha256", "")),
            qc_episode_result_sha256=str(value.get("qc_episode_result_sha256", "")),
            key_event_time_s=float(value.get("key_event_time_s", math.nan)),
            video_paths=dict(value.get("video_paths") or {}),
            video_sha256=dict(value.get("video_sha256") or {}),
            video_pts_sha256=dict(value.get("video_pts_sha256") or {}),
            video_frame_count={
                str(key): int(item)
                for key, item in dict(value.get("video_frame_count") or {}).items()
            },
            event_strip_paths=dict(value.get("event_strip_paths") or {}),
            event_strip_sha256=dict(value.get("event_strip_sha256") or {}),
            event_strip_indices={
                str(key): int(item)
                for key, item in dict(value.get("event_strip_indices") or {}).items()
            },
            selected_frame_timestamps_s={
                str(key): float(item)
                for key, item in dict(
                    value.get("selected_frame_timestamps_s") or {}
                ).items()
            },
            strip_width_px=int(value.get("strip_width_px", 0)),
            strip_height_px=int(value.get("strip_height_px", 0)),
            schema_version=str(value.get("schema_version", "")),
        )
        result.validate()
        if value.get("binding_sha256") != result.binding_sha256:
            raise ValueError("review media pack binding hash mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ReviewMediaPackPublication:
    """Published pack manifest plus its physical file hash."""

    manifest: ReviewMediaPackManifest
    manifest_path: str
    manifest_sha256: str

    def validate(self) -> None:
        self.manifest.validate()
        portable_relative_path(self.manifest_path)
        if not _valid_sha256(self.manifest_sha256):
            raise ValueError("review media pack publication hash is malformed")


@dataclass(frozen=True, slots=True)
class ReviewArtifactManifest:
    """Every byte shown to a reviewer, bound to its source rollout."""

    leaf_id: str
    episode_uuid: str
    rollout_index: int
    fixed_master_seed: int
    review_plan_sha256: str
    review_case_sha256: str
    review_request_ledger_sha256: str
    review_request_sha256: str
    scenario_spec_sha256: str
    qc_report_sha256: str
    qc_report_schema: str
    qc_episode_result_sha256: str
    qc_strict_all: bool
    automated_qc_passed: bool
    source_manifest_sha256: str
    frame_timestamps_sha256: str
    key_event_time_s: float
    video_paths: Mapping[str, str]
    video_sha256: Mapping[str, str]
    event_strip_paths: Mapping[str, str]
    event_strip_sha256: Mapping[str, str]
    event_strip_indices: Mapping[str, int]
    schema_version: str = REVIEW_ARTIFACT_SCHEMA

    def validate(self) -> None:
        if self.schema_version != REVIEW_ARTIFACT_SCHEMA:
            raise ValueError(f"review artifacts must use {REVIEW_ARTIFACT_SCHEMA}")
        if not self.leaf_id or not self.episode_uuid:
            raise ValueError("review artifacts require leaf and episode identities")
        if self.rollout_index not in range(6):
            raise ValueError("review artifacts require a fixed rollout index in [0, 6)")
        if self.fixed_master_seed != REVIEW_MASTER_SEED:
            raise ValueError("review artifacts use a replacement acceptance seed")
        for label, value in (
            ("review plan", self.review_plan_sha256),
            ("review case", self.review_case_sha256),
            ("review request ledger", self.review_request_ledger_sha256),
            ("review request", self.review_request_sha256),
            ("scenario spec", self.scenario_spec_sha256),
            ("QC report", self.qc_report_sha256),
            ("QC episode result", self.qc_episode_result_sha256),
            ("source manifest", self.source_manifest_sha256),
            ("frame timestamps", self.frame_timestamps_sha256),
        ):
            if not _valid_sha256(value):
                raise ValueError(f"review {label} hash is malformed")
        if self.qc_report_schema != "dynamic-robot-qc-report/v2":
            raise ValueError("review artifacts require canonical v2 QC")
        if self.qc_strict_all is not True:
            raise ValueError("review artifacts require strict-all QC, even for a failed case")
        if not isinstance(self.automated_qc_passed, bool):
            raise ValueError("review automated QC result must be boolean")
        if not math.isfinite(self.key_event_time_s) or self.key_event_time_s < 0:
            raise ValueError("review key-event time must be finite and non-negative")
        if set(self.video_paths) != {"main", "secondary"}:
            raise ValueError("review artifacts require main and secondary videos")
        if set(self.video_sha256) != set(self.video_paths):
            raise ValueError("review video hashes do not match video paths")
        if set(self.event_strip_paths) != {"main", "secondary"}:
            raise ValueError("review artifacts require event strips for both views")
        if set(self.event_strip_sha256) != set(self.event_strip_paths):
            raise ValueError("review strip hashes do not match strip paths")
        expected_indices = {
            "pre_event",
            "event",
            "post_0p1_s",
            "post_0p3_s",
            "final",
        }
        if set(self.event_strip_indices) != expected_indices or any(
            not isinstance(value, int) or value < 0
            for value in self.event_strip_indices.values()
        ):
            raise ValueError("review artifacts have an invalid event-strip index map")
        ordered_indices = tuple(
            self.event_strip_indices[name]
            for name in ("pre_event", "event", "post_0p1_s", "post_0p3_s", "final")
        )
        if any(
            right <= left
            for left, right in zip(ordered_indices, ordered_indices[1:])
        ):
            raise ValueError(
                "review event-strip indices must be distinct and strictly ordered"
            )
        for path in (*self.video_paths.values(), *self.event_strip_paths.values()):
            portable_relative_path(path)
        for digest in (*self.video_sha256.values(), *self.event_strip_sha256.values()):
            if not _valid_sha256(digest):
                raise ValueError("review media hash is malformed")

    @property
    def binding_sha256(self) -> str:
        self.validate()
        return sha256_json(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReviewArtifactManifest":
        """Load a persisted artifact while rejecting derived-field spoofing."""

        result = cls(
            leaf_id=str(value.get("leaf_id", "")),
            episode_uuid=str(value.get("episode_uuid", "")),
            rollout_index=int(value.get("rollout_index", -1)),
            fixed_master_seed=int(value.get("fixed_master_seed", -1)),
            review_plan_sha256=str(value.get("review_plan_sha256", "")),
            review_case_sha256=str(value.get("review_case_sha256", "")),
            review_request_ledger_sha256=str(
                value.get("review_request_ledger_sha256", "")
            ),
            review_request_sha256=str(value.get("review_request_sha256", "")),
            scenario_spec_sha256=str(value.get("scenario_spec_sha256", "")),
            qc_report_sha256=str(value.get("qc_report_sha256", "")),
            qc_report_schema=str(value.get("qc_report_schema", "")),
            qc_episode_result_sha256=str(
                value.get("qc_episode_result_sha256", "")
            ),
            qc_strict_all=value.get("qc_strict_all") is True,
            automated_qc_passed=value.get("automated_qc_passed") is True,
            source_manifest_sha256=str(value.get("source_manifest_sha256", "")),
            frame_timestamps_sha256=str(
                value.get("frame_timestamps_sha256", "")
            ),
            key_event_time_s=float(value.get("key_event_time_s", math.nan)),
            video_paths={
                str(key): str(item)
                for key, item in dict(value.get("video_paths") or {}).items()
            },
            video_sha256={
                str(key): str(item)
                for key, item in dict(value.get("video_sha256") or {}).items()
            },
            event_strip_paths={
                str(key): str(item)
                for key, item in dict(value.get("event_strip_paths") or {}).items()
            },
            event_strip_sha256={
                str(key): str(item)
                for key, item in dict(value.get("event_strip_sha256") or {}).items()
            },
            event_strip_indices={
                str(key): int(item)
                for key, item in dict(value.get("event_strip_indices") or {}).items()
            },
            schema_version=str(value.get("schema_version", "")),
        )
        result.validate()
        declared = value.get("binding_sha256")
        if declared is not None and declared != result.binding_sha256:
            raise ValueError("review artifact binding hash mismatch")
        return result


def bind_review_artifacts(
    dataset_root: str | Path,
    *,
    leaf_id: str,
    episode_uuid: str,
    rollout_index: int,
    review_plan_sha256: str,
    review_case_sha256: str,
    review_request_ledger_sha256: str,
    review_request_sha256: str,
    scenario_spec_sha256: str,
    qc_report_path: str | Path,
    source_manifest_sha256: str,
    video_paths: Mapping[str, str],
    event_strip_paths: Mapping[str, str],
    frame_timestamps_s: Sequence[float],
    key_event_time_s: float,
    fixed_master_seed: int = REVIEW_MASTER_SEED,
    unsafe_compatibility_acknowledged: bool = False,
) -> ReviewArtifactManifest:
    """Deprecated caller-identity binder retained only for unsafe compatibility.

    New executable review paths must call :func:`bind_review_artifacts_strict`.
    This compatibility function accepts identity hashes and an event time from
    its caller, so use now requires an explicit acknowledgement at the call
    site and cannot accidentally masquerade as the strict acceptance path.
    """

    if unsafe_compatibility_acknowledged is not True:
        raise ValueError(
            "unsafe compatibility review binding requires an explicit acknowledgement; "
            "executable review code must use bind_review_artifacts_strict"
        )

    root = Path(dataset_root).resolve(strict=True)

    def hash_paths(paths: Mapping[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, relative in paths.items():
            normalized = portable_relative_path(relative)
            path = (root / normalized).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"review artifact escapes dataset root: {relative}") from error
            result[key] = sha256_file(path)
        return result

    qc_path = Path(qc_report_path)
    if not qc_path.is_absolute():
        qc_path = root / qc_path
    qc_path = qc_path.resolve(strict=True)
    try:
        qc_path.relative_to(root)
    except ValueError as error:
        raise ValueError("review QC report escapes dataset root") from error
    try:
        qc_report = json.loads(qc_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("review QC report is not canonical JSON") from error
    if not isinstance(qc_report, Mapping):
        raise ValueError("review QC report must be a mapping")
    if qc_report.get("schema_version") != "dynamic-robot-qc-report/v2":
        raise ValueError("review QC report must use dynamic-robot-qc-report/v2")
    if qc_report.get("strict_all") is not True:
        raise ValueError("review QC report must be generated with strict_all=True")
    global_failures = qc_report.get("global_failures")
    if (
        not isinstance(global_failures, Sequence)
        or isinstance(global_failures, (str, bytes, bytearray))
        or any(not isinstance(value, str) for value in global_failures)
    ):
        raise ValueError("review QC report lacks canonical global failure evidence")
    episode_results = qc_report.get("episodes")
    if not isinstance(episode_results, Sequence) or isinstance(
        episode_results, (str, bytes, bytearray)
    ):
        raise ValueError("review QC report lacks episode results")
    selected = [
        value
        for value in episode_results
        if isinstance(value, Mapping) and value.get("episode_uuid") == episode_uuid
    ]
    if len(selected) != 1:
        raise ValueError("review QC report must contain the target episode exactly once")
    episode_qc = dict(selected[0])
    # ``qc_report.passed`` is the dataset-wide strict-all aggregate.  Binding it
    # here would let an unrelated leaf poison this episode's review artifact.
    # Leaf activation remains fail-closed in ``HumanReviewLedger`` by requiring
    # all six episode-level results for that leaf to pass. Dataset-global hard
    # failures still block every artifact because they are not leaf-local.
    automated_qc_passed = bool(
        episode_qc.get("passed") is True and not global_failures
    )
    timestamps = tuple(float(value) for value in frame_timestamps_s)
    event_indices = event_strip_frame_indices(timestamps, key_event_time_s)
    manifest = ReviewArtifactManifest(
        leaf_id=leaf_id,
        episode_uuid=episode_uuid,
        rollout_index=rollout_index,
        fixed_master_seed=fixed_master_seed,
        review_plan_sha256=review_plan_sha256,
        review_case_sha256=review_case_sha256,
        review_request_ledger_sha256=review_request_ledger_sha256,
        review_request_sha256=review_request_sha256,
        scenario_spec_sha256=scenario_spec_sha256,
        qc_report_sha256=sha256_file(qc_path),
        qc_report_schema=str(qc_report["schema_version"]),
        qc_episode_result_sha256=sha256_json(episode_qc),
        qc_strict_all=True,
        automated_qc_passed=automated_qc_passed,
        source_manifest_sha256=source_manifest_sha256,
        frame_timestamps_sha256=sha256_json(timestamps),
        key_event_time_s=float(key_event_time_s),
        video_paths=dict(video_paths),
        video_sha256=hash_paths(video_paths),
        event_strip_paths=dict(event_strip_paths),
        event_strip_sha256=hash_paths(event_strip_paths),
        event_strip_indices=event_indices,
    )
    manifest.validate()
    return manifest


def _review_dataset_file(
    root: Path,
    value: str | Path,
    *,
    label: str,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes dataset root") from error
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    return path


def _review_media_hashes(root: Path, paths: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, relative in paths.items():
        normalized = portable_relative_path(relative)
        result[str(key)] = sha256_file(
            _review_dataset_file(root, normalized, label=f"review media {key}")
        )
    return result


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON mapping")
    return value


def _strict_qc_episode(
    qc_report: Mapping[str, Any],
    *,
    episode_uuid: str,
    dataset_episode_index: int | None,
    evaluator_id: str,
) -> tuple[dict[str, Any], float]:
    if qc_report.get("schema_version") != "dynamic-robot-qc-report/v2":
        raise ValueError("review QC report must use dynamic-robot-qc-report/v2")
    if qc_report.get("strict_all") is not True:
        raise ValueError("review QC report must be generated with strict_all=True")
    global_failures = qc_report.get("global_failures")
    if (
        not isinstance(global_failures, Sequence)
        or isinstance(global_failures, (str, bytes, bytearray))
        or any(not isinstance(value, str) for value in global_failures)
    ):
        raise ValueError("review QC report lacks canonical global failure evidence")
    episode_results = qc_report.get("episodes")
    if not isinstance(episode_results, Sequence) or isinstance(
        episode_results, (str, bytes, bytearray)
    ):
        raise ValueError("review QC report lacks episode results")
    selected = [
        value
        for value in episode_results
        if isinstance(value, Mapping) and value.get("episode_uuid") == episode_uuid
    ]
    if len(selected) != 1:
        raise ValueError("review QC report must contain the fixed episode exactly once")
    episode_qc = dict(selected[0])
    try:
        persisted_index = int(episode_qc["episode_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("review QC episode lacks its fixed episode_index") from error
    if persisted_index < 0:
        raise ValueError("review QC episode index must be non-negative")
    if dataset_episode_index is not None and persisted_index != dataset_episode_index:
        raise ValueError("review QC episode index differs from finalized dataset metadata")
    metrics = episode_qc.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("review QC episode lacks evaluator metrics")
    evaluator_evidence = metrics.get("objective_recompute")
    if not isinstance(evaluator_evidence, Mapping):
        raise ValueError("review QC episode lacks independent objective evaluator evidence")
    if evaluator_evidence.get("evaluator_id") != evaluator_id:
        raise ValueError("review QC objective evidence uses a different evaluator")
    if (
        not str(evaluator_evidence.get("evidence_version") or "").strip()
        or not _valid_sha256(evaluator_evidence.get("evidence_hash"))
    ):
        raise ValueError("review QC objective evidence is not content-bound")
    if not str(evaluator_evidence.get("key_event_name") or "").strip():
        raise ValueError("review QC objective evidence lacks a named key event")
    if not isinstance(evaluator_evidence.get("replay_match"), bool):
        raise ValueError("review QC objective evidence lacks a measured replay match")
    try:
        key_event_time_s = float(evaluator_evidence["key_event_time_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "review QC objective evidence lacks a finite key-event time"
        ) from error
    if not math.isfinite(key_event_time_s) or key_event_time_s < 0:
        raise ValueError(
            "review QC objective key-event time must be finite and non-negative"
        )
    return episode_qc, key_event_time_s


def _fixed_review_context(
    root: Path,
    *,
    review_suite_root: str | Path,
    source_scenario_spec_path: str | Path,
) -> tuple[Any, Any, Any, Any]:
    """Resolve one immutable review case without accepting caller identity."""

    from .review_suite import load_review_suite_bundle
    from .source_scenario import SourceScenarioSpec

    bundle = load_review_suite_bundle(review_suite_root)
    scenario_path = _review_dataset_file(
        root, source_scenario_spec_path, label="source scenario spec"
    )
    scenario = SourceScenarioSpec.from_json(
        scenario_path.read_text(encoding="utf-8")
    )
    cases = [case for case in bundle.plan.cases if case.case_id == scenario.scenario_id]
    requests = [
        request
        for request in bundle.requests.requests
        if request.case_id == scenario.scenario_id
    ]
    if len(cases) != 1 or len(requests) != 1:
        raise ValueError("source scenario does not identify one fixed review request")
    case = cases[0]
    request = requests[0]
    if not case.execution_eligible or not request.execution_eligible:
        raise ValueError("blocked review cases cannot produce fixed review media")
    scenario_value = scenario.to_dict()
    expected_identity = request.source_scenario_identity
    representable_identity = {
        "schema_version": scenario.schema_version,
        "scenario_id": scenario.scenario_id,
        "corpus_leaf_id": scenario.corpus_leaf_id,
        "backend": scenario.backend,
        "embodiment": scenario_value["embodiment"],
        "task_variant": scenario.task_variant,
        "rng_subseeds": scenario_value["rng_subseeds"],
        "counterfactual_bundle_id": scenario.counterfactual.bundle_id,
        "counterfactual_branch_id": scenario.counterfactual.branch_id,
        "counterfactual_sibling_index": scenario.counterfactual.sibling_index,
        "robocasa_catalog_sha256": scenario.robocasa_manifest.catalog_sha256,
        "robocasa_license_sha256": (
            scenario.robocasa_manifest.license_manifest_sha256
        ),
    }
    mismatched = sorted(
        name
        for name, actual in representable_identity.items()
        if expected_identity.get(name) != actual
    )
    if mismatched:
        raise ValueError(
            "source scenario differs from the immutable review request: "
            + ", ".join(mismatched)
        )
    if (
        case.episode_uuid != request.episode_uuid
        or case.corpus_leaf_id != scenario.corpus_leaf_id
        or case.backend != scenario.backend
        or case.task_variant != scenario.task_variant
        or case.embodiment != scenario.embodiment.end_effector
        or case.rng_subseeds != scenario.rng_subseeds
        or bundle.plan.fixed_master_seed != REVIEW_MASTER_SEED
    ):
        raise ValueError("source scenario differs from the fixed review case")
    if case.requires_real_robocasa != bool(scenario.robocasa_manifest.assets):
        raise ValueError(
            "source scenario RoboCasa assets differ from the fixed scene requirement"
        )
    return bundle, case, request, scenario


def _finalized_review_record(
    root: Path,
    *,
    episode_uuid: str,
) -> tuple[Any, Path, Path]:
    """Load exactly one record after validating the irreversible finalization."""

    seal_path = _review_dataset_file(root, ".seal.json", label="dataset seal")
    completion_path = _review_dataset_file(
        root, "meta/.complete.json", label="finalized metadata marker"
    )
    seal = _load_json_mapping(seal_path, label="dataset seal")
    completion = _load_json_mapping(
        completion_path, label="finalized metadata marker"
    )
    if seal.get("schema_version") != "dynamic-robot-dataset-seal/v1":
        raise ValueError("review media requires a canonical finalized dataset seal")
    if completion.get("seal_sha256") != sha256_file(seal_path):
        raise ValueError("finalized metadata marker is not bound to the dataset seal")
    if completion.get("config_hash") != seal.get("config_hash"):
        raise ValueError("finalized metadata and dataset seal use different configurations")
    metadata_hashes = completion.get("content_hashes")
    if not isinstance(metadata_hashes, Mapping) or set(metadata_hashes) != set(
        FINALIZED_METADATA_ARTIFACTS
    ):
        raise ValueError("finalized metadata marker has incomplete artifact membership")
    records = load_episode_records(root)
    selected = [record for record in records if record.episode_uuid == episode_uuid]
    if len(selected) != 1:
        raise ValueError("finalized dataset must contain the fixed review episode once")
    record = selected[0]
    return record, seal_path, completion_path


def _validate_finalized_qc_report(
    root: Path,
    qc_report: Mapping[str, Any],
    *,
    completion_path: Path,
    records: Sequence[Any],
) -> None:
    """Require the canonical QC writer's finalized-dataset hash bindings."""

    episodes_path = _review_dataset_file(
        root, "meta/episodes.parquet", label="finalized episode metadata"
    )
    completion = _load_json_mapping(
        completion_path, label="finalized metadata marker"
    )
    if Path(str(qc_report.get("dataset_root") or "")).resolve() != root:
        raise ValueError("review QC report is bound to a different dataset root")
    if qc_report.get("dataset_episodes_sha256") != sha256_file(episodes_path):
        raise ValueError("review QC report is not bound to finalized episode metadata")
    if qc_report.get("metadata_complete_manifest_sha256") != sha256_file(
        completion_path
    ):
        raise ValueError("review QC report is not bound to finalization completion")
    if dict(qc_report.get("metadata_content_hashes") or {}) != dict(
        completion.get("content_hashes") or {}
    ):
        raise ValueError("review QC report metadata hashes differ from finalization")
    episode_results = qc_report.get("episodes")
    if not isinstance(episode_results, Sequence) or isinstance(
        episode_results, (str, bytes, bytearray)
    ):
        raise ValueError("review QC report lacks exact episode membership")
    expected = {(record.episode_uuid, record.episode_index) for record in records}
    try:
        actual = {
            (str(item["episode_uuid"]), int(item["episode_index"]))
            for item in episode_results
            if isinstance(item, Mapping)
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("review QC report has invalid episode membership") from error
    if actual != expected or len(episode_results) != len(expected):
        raise ValueError("review QC report does not exactly match finalized episodes")


def _content_bound_episode_file(
    root: Path,
    record: Any,
    relative: str,
    *,
    label: str,
) -> Path:
    normalized = portable_relative_path(relative)
    path = _review_dataset_file(root, normalized, label=label)
    expected = record.content_hashes.get(normalized)
    if not _valid_sha256(expected) or sha256_file(path) != expected:
        raise ValueError(f"{label} is not bound by finalized episode metadata")
    return path


def _validate_finalized_review_identity(
    record: Any,
    *,
    case: Any,
    scenario: Any,
) -> None:
    expected_background = (
        "clean_franka_lab"
        if case.randomization_level == "R0"
        else case.scene_profile
    )
    if (
        record.extras.get("source_scenario_spec") != scenario.to_dict()
        or record.extras.get("source_scenario_spec_sha256") != scenario.spec_hash
        or record.extras.get("review_case_sha256") != case.case_sha256
        or record.extras.get("review_case") != case.to_dict()
        or record.extras.get("review_suite_episode_index") != case.episode_index
        or record.randomization.get("randomization_level")
        != case.randomization_level
        or record.randomization.get("background_style") != expected_background
    ):
        raise ValueError(
            "finalized episode identity differs from the fixed review request"
        )


def _review_video_paths(record: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for view in ("main", "secondary"):
        canonical = f"observation.images.{view}"
        relative = record.video_paths.get(canonical)
        if not isinstance(relative, str) or not relative:
            raise ValueError(
                "finalized review episode is missing synchronized main and secondary views"
            )
        result[view] = portable_relative_path(relative)
    return result


def _decode_selected_frames(
    path: Path,
    *,
    width: int,
    height: int,
    frame_count: int,
    indices: Mapping[str, int],
) -> dict[str, bytes]:
    by_index = {int(index): label for label, index in indices.items()}
    selected: dict[str, bytes] = {}
    decoded_count = 0
    for decoded_index, frame in enumerate(iter_rgb_frames(path, width, height)):
        if decoded_index >= frame_count:
            raise ValueError(f"decoded video has more than {frame_count} finalized frames")
        decoded_count = decoded_index + 1
        label = by_index.get(decoded_index)
        if label is not None:
            selected[label] = bytes(frame)
    if decoded_count != frame_count:
        raise ValueError(
            f"decoded video has {decoded_count} frames, expected {frame_count}"
        )
    missing = sorted(set(EVENT_STRIP_LABELS) - set(selected))
    if missing:
        raise ValueError(f"decoded video lacks selected event-strip frames: {missing}")
    return selected


def _event_strip_png(
    frames: Mapping[str, bytes],
    *,
    width: int,
    height: int,
) -> bytes:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - Pillow is a base dependency
        raise RuntimeError("review event strips require Pillow") from error

    strip = Image.new("RGB", (width * len(EVENT_STRIP_LABELS), height))
    try:
        for panel_index, label in enumerate(EVENT_STRIP_LABELS):
            frame = Image.frombytes("RGB", (width, height), frames[label])
            try:
                strip.paste(frame, (panel_index * width, 0))
            finally:
                frame.close()
        output = BytesIO()
        strip.save(output, format="PNG", optimize=False, compress_level=6)
        return output.getvalue()
    finally:
        strip.close()


def _validate_event_strip_png(path: Path, *, width: int, height: int) -> None:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - Pillow is a base dependency
        raise RuntimeError("review event strips require Pillow") from error
    with Image.open(path) as image:
        image.load()
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size != (width * len(EVENT_STRIP_LABELS), height)
        ):
            raise ValueError("published review event strip has an invalid PNG layout")


def write_fixed_review_media_pack(
    dataset_root: str | Path,
    *,
    review_suite_root: str | Path,
    source_scenario_spec_path: str | Path,
) -> ReviewMediaPackPublication:
    """Create both event strips only from a sealed episode's persisted bytes.

    Case identity, output paths, evaluator identity, and key-event time are
    derived from the immutable review bundle, persisted scenario, and strict
    QC report. The caller cannot replace a difficult event or output location.
    """

    root = Path(dataset_root).resolve(strict=True)
    bundle, case, request, scenario = _fixed_review_context(
        root,
        review_suite_root=review_suite_root,
        source_scenario_spec_path=source_scenario_spec_path,
    )
    record, seal_path, completion_path = _finalized_review_record(
        root,
        episode_uuid=case.episode_uuid,
    )
    _validate_finalized_review_identity(record, case=case, scenario=scenario)
    if record.frame_data_path is None:
        raise ValueError("finalized review episode lacks frame data")
    frame_path = _content_bound_episode_file(
        root,
        record,
        record.frame_data_path,
        label="finalized frame data",
    )
    frame_rows = read_parquet_rows(frame_path)
    timestamps = tuple(float(row["timestamp"]) for row in frame_rows)
    if record.frame_count != len(timestamps):
        raise ValueError("finalized frame count differs from persisted frame data")
    expected_frame_count = round(scenario.duration_s * 30)
    if len(timestamps) != expected_frame_count:
        raise ValueError(
            "finalized review episode does not contain round(duration * 30) frames"
        )
    if any(
        abs(timestamp - index / 30.0) > 1e-9
        for index, timestamp in enumerate(timestamps)
    ):
        raise ValueError("persisted review timestamps do not use the exact k/30 clock")
    for expected_index, row in enumerate(frame_rows):
        if (
            int(row.get("frame_index", -1)) != expected_index
            or int(row.get("video_frame_index", -1)) != expected_index
        ):
            raise ValueError("persisted frame/video indices are not contiguous and aligned")

    qc_path = _review_dataset_file(
        root, "qc/dataset_report.json", label="canonical review QC report"
    )
    qc_report = _load_json_mapping(qc_path, label="review QC report")
    _validate_finalized_qc_report(
        root,
        qc_report,
        completion_path=completion_path,
        records=load_episode_records(root),
    )
    episode_qc, key_event_time_s = _strict_qc_episode(
        qc_report,
        episode_uuid=case.episode_uuid,
        dataset_episode_index=record.episode_index,
        evaluator_id=case.evaluator,
    )
    event_indices = event_strip_frame_indices(timestamps, key_event_time_s)

    video_paths = _review_video_paths(record)
    video_hashes: dict[str, str] = {}
    video_pts_hashes: dict[str, str] = {}
    video_frame_counts: dict[str, int] = {}
    decoded: dict[str, dict[str, bytes]] = {}
    pts_by_view: dict[str, tuple[float, ...]] = {}
    video_spec = VideoSpec()
    width = video_spec.width
    height = video_spec.height
    for view, relative in video_paths.items():
        path = _content_bound_episode_file(
            root, record, relative, label=f"finalized {view} video"
        )
        probe = probe_video(path)
        validate_video_probe(probe, video_spec, expected_frames=len(timestamps))
        pts = tuple(float(value) for value in probe_frame_timestamps(path))
        if len(pts) != len(timestamps):
            raise ValueError(f"{view} video PTS count differs from persisted frame data")
        maximum_error_s = max(
            abs(video_time - frame_time)
            for video_time, frame_time in zip(pts, timestamps)
        )
        if maximum_error_s > 1e-6:
            raise ValueError(
                f"{view} video PTS differ from persisted timestamps by {maximum_error_s}s"
            )
        pts_by_view[view] = pts
        video_hashes[view] = sha256_file(path)
        video_pts_hashes[view] = sha256_json(pts)
        video_frame_counts[view] = probe.frame_count
        decoded[view] = _decode_selected_frames(
            path,
            width=probe.width,
            height=probe.height,
            frame_count=probe.frame_count,
            indices=event_indices,
        )
        if sha256_file(path) != video_hashes[view]:
            raise ValueError(f"{view} video changed while deriving its review strip")
        width, height = probe.width, probe.height
    validate_synchronized_streams(pts_by_view, tolerance_s=1e-6)

    output_paths_raw = request.event_strip_request.get("output_paths")
    if not isinstance(output_paths_raw, Mapping) or set(output_paths_raw) != {
        "main",
        "secondary",
    }:
        raise ValueError("fixed review request lacks both event-strip output paths")
    event_strip_paths = {
        view: portable_relative_path(str(output_paths_raw[view]))
        for view in ("main", "secondary")
    }
    output_parents = {str(Path(path).parent) for path in event_strip_paths.values()}
    if len(output_parents) != 1 or {
        Path(event_strip_paths[view]).name for view in ("main", "secondary")
    } != {"main.png", "secondary.png"}:
        raise ValueError("fixed review strips must share their immutable case directory")
    output_parent_relative = portable_relative_path(output_parents.pop())
    output_directory = root / output_parent_relative
    manifest_relative = f"{output_parent_relative}/media_pack.json"
    qc_relative = portable_relative_path(qc_path.relative_to(root).as_posix())

    with AtomicDirectory(output_directory) as staging:
        strip_hashes: dict[str, str] = {}
        for view in ("main", "secondary"):
            data = _event_strip_png(decoded[view], width=width, height=height)
            staged_path = atomic_write_bytes(staging / f"{view}.png", data)
            _validate_event_strip_png(staged_path, width=width, height=height)
            strip_hashes[view] = sha256_file(staged_path)
        manifest = ReviewMediaPackManifest(
            case_id=case.case_id,
            leaf_id=case.corpus_leaf_id,
            episode_uuid=case.episode_uuid,
            review_matrix_episode_index=case.episode_index,
            dataset_episode_index=record.episode_index,
            review_plan_sha256=bundle.plan.plan_sha256,
            review_case_sha256=case.case_sha256,
            review_request_ledger_sha256=bundle.requests.ledger_sha256,
            review_request_sha256=request.request_sha256,
            dataset_seal_sha256=sha256_file(seal_path),
            finalized_completion_sha256=sha256_file(completion_path),
            finalized_episode_record_sha256=sha256_json(record.to_dict()),
            frame_data_path=record.frame_data_path,
            frame_data_sha256=sha256_file(frame_path),
            frame_timestamps_s=timestamps,
            frame_timestamps_sha256=sha256_json(timestamps),
            qc_report_path=qc_relative,
            qc_report_sha256=sha256_file(qc_path),
            qc_episode_result_sha256=sha256_json(episode_qc),
            key_event_time_s=key_event_time_s,
            video_paths=video_paths,
            video_sha256=video_hashes,
            video_pts_sha256=video_pts_hashes,
            video_frame_count=video_frame_counts,
            event_strip_paths=event_strip_paths,
            event_strip_sha256=strip_hashes,
            event_strip_indices=event_indices,
            selected_frame_timestamps_s={
                label: timestamps[event_indices[label]] for label in EVENT_STRIP_LABELS
            },
            strip_width_px=width * len(EVENT_STRIP_LABELS),
            strip_height_px=height,
        )
        manifest.validate()
        staged_manifest = atomic_write_json(staging / "media_pack.json", manifest.to_dict())
        manifest_sha256 = sha256_file(staged_manifest)
    publication = ReviewMediaPackPublication(
        manifest=manifest,
        manifest_path=manifest_relative,
        manifest_sha256=manifest_sha256,
    )
    publication.validate()
    if sha256_file(root / publication.manifest_path) != publication.manifest_sha256:
        raise RuntimeError("published review media manifest hash changed during commit")
    return publication


def _verify_review_media_pack(
    root: Path,
    *,
    media_pack_manifest_path: str | Path,
    review_suite_root: str | Path,
    source_scenario_spec_path: str | Path,
) -> ReviewMediaPackManifest:
    pack_path = _review_dataset_file(
        root, media_pack_manifest_path, label="review media pack manifest"
    )
    pack = ReviewMediaPackManifest.from_dict(
        _load_json_mapping(pack_path, label="review media pack manifest")
    )
    bundle, case, request, scenario = _fixed_review_context(
        root,
        review_suite_root=review_suite_root,
        source_scenario_spec_path=source_scenario_spec_path,
    )
    if (
        pack.case_id != case.case_id
        or pack.leaf_id != case.corpus_leaf_id
        or pack.episode_uuid != case.episode_uuid
        or pack.review_matrix_episode_index != case.episode_index
        or pack.review_plan_sha256 != bundle.plan.plan_sha256
        or pack.review_case_sha256 != case.case_sha256
        or pack.review_request_ledger_sha256 != bundle.requests.ledger_sha256
        or pack.review_request_sha256 != request.request_sha256
        or dict(pack.event_strip_paths)
        != dict(request.event_strip_request["output_paths"])
    ):
        raise ValueError("review media pack differs from the immutable review request")
    record, seal_path, completion_path = _finalized_review_record(
        root,
        episode_uuid=case.episode_uuid,
    )
    _validate_finalized_review_identity(record, case=case, scenario=scenario)
    if (
        pack.dataset_seal_sha256 != sha256_file(seal_path)
        or pack.finalized_completion_sha256 != sha256_file(completion_path)
        or pack.finalized_episode_record_sha256 != sha256_json(record.to_dict())
        or pack.dataset_episode_index != record.episode_index
        or record.frame_data_path != pack.frame_data_path
    ):
        raise ValueError("review media pack differs from finalized episode metadata")
    frame_path = _content_bound_episode_file(
        root, record, pack.frame_data_path, label="finalized frame data"
    )
    frame_rows = read_parquet_rows(frame_path)
    timestamps = tuple(float(row["timestamp"]) for row in frame_rows)
    if (
        pack.frame_data_sha256 != sha256_file(frame_path)
        or pack.frame_timestamps_s != timestamps
        or pack.frame_timestamps_sha256 != sha256_json(timestamps)
    ):
        raise ValueError("review media pack differs from persisted frame data")
    expected_video_paths = _review_video_paths(record)
    if dict(pack.video_paths) != expected_video_paths:
        raise ValueError("review media pack uses different finalized videos")
    for view, relative in expected_video_paths.items():
        video_path = _content_bound_episode_file(
            root, record, relative, label=f"finalized {view} video"
        )
        if pack.video_sha256[view] != sha256_file(video_path):
            raise ValueError("review media pack finalized video hash changed")
    for view, relative in pack.event_strip_paths.items():
        strip_path = _review_dataset_file(
            root, relative, label=f"review {view} event strip"
        )
        if pack.event_strip_sha256[view] != sha256_file(strip_path):
            raise ValueError("review event-strip hash changed after publication")
    qc_path = _review_dataset_file(
        root, "qc/dataset_report.json", label="canonical review QC report"
    )
    qc_relative = portable_relative_path(qc_path.relative_to(root).as_posix())
    qc_report = _load_json_mapping(qc_path, label="review QC report")
    _validate_finalized_qc_report(
        root,
        qc_report,
        completion_path=completion_path,
        records=load_episode_records(root),
    )
    episode_qc, key_event_time_s = _strict_qc_episode(
        qc_report,
        episode_uuid=case.episode_uuid,
        dataset_episode_index=record.episode_index,
        evaluator_id=case.evaluator,
    )
    if (
        pack.qc_report_path != qc_relative
        or pack.qc_report_sha256 != sha256_file(qc_path)
        or pack.qc_episode_result_sha256 != sha256_json(episode_qc)
        or pack.key_event_time_s != key_event_time_s
    ):
        raise ValueError("review media pack differs from strict persisted QC evidence")
    return pack


def bind_fixed_review_media_pack_strict(
    dataset_root: str | Path,
    *,
    review_suite_root: str | Path,
    source_scenario_spec_path: str | Path,
    source_manifest_path: str | Path,
    media_pack_manifest_path: str | Path,
) -> ReviewArtifactManifest:
    """Bind a published, revalidated media pack into the strict review ledger."""

    root = Path(dataset_root).resolve(strict=True)
    pack = _verify_review_media_pack(
        root,
        media_pack_manifest_path=media_pack_manifest_path,
        review_suite_root=review_suite_root,
        source_scenario_spec_path=source_scenario_spec_path,
    )
    artifact = bind_review_artifacts_strict(
        root,
        review_suite_root=review_suite_root,
        source_scenario_spec_path=source_scenario_spec_path,
        source_manifest_path=source_manifest_path,
        qc_report_path="qc/dataset_report.json",
        video_paths=pack.video_paths,
        event_strip_paths=pack.event_strip_paths,
        frame_timestamps_s=pack.frame_timestamps_s,
    )
    if (
        dict(artifact.video_sha256) != dict(pack.video_sha256)
        or dict(artifact.event_strip_sha256) != dict(pack.event_strip_sha256)
        or dict(artifact.event_strip_indices) != dict(pack.event_strip_indices)
        or artifact.frame_timestamps_sha256 != pack.frame_timestamps_sha256
        or artifact.qc_episode_result_sha256 != pack.qc_episode_result_sha256
    ):
        raise ValueError("strict review binding differs from the published media pack")
    return artifact


def bind_review_artifacts_strict(
    dataset_root: str | Path,
    *,
    review_suite_root: str | Path,
    source_scenario_spec_path: str | Path,
    source_manifest_path: str | Path,
    qc_report_path: str | Path,
    video_paths: Mapping[str, str],
    event_strip_paths: Mapping[str, str],
    frame_timestamps_s: Sequence[float],
) -> ReviewArtifactManifest:
    """Bind executable review artifacts to immutable requests and measured evidence.

    No case identity, digest, seed, or event time is accepted from the caller.
    Those values are loaded from the validated review-suite bundle, a complete
    :class:`SourceScenarioSpec`, and the strict persisted-artifact QC result.
    """

    # Lazy imports avoid the review <-> review-suite constant import cycle.
    from .review_suite import load_review_suite_bundle
    from .source_scenario import SourceScenarioSpec

    root = Path(dataset_root).resolve(strict=True)
    bundle = load_review_suite_bundle(review_suite_root)
    scenario_path = _review_dataset_file(
        root, source_scenario_spec_path, label="source scenario spec"
    )
    scenario = SourceScenarioSpec.from_json(
        scenario_path.read_text(encoding="utf-8")
    )
    selected_cases = [
        case for case in bundle.plan.cases if case.case_id == scenario.scenario_id
    ]
    selected_requests = [
        request
        for request in bundle.requests.requests
        if request.case_id == scenario.scenario_id
    ]
    if len(selected_cases) != 1 or len(selected_requests) != 1:
        raise ValueError("source scenario does not identify exactly one fixed review request")
    case = selected_cases[0]
    request = selected_requests[0]
    if not case.execution_eligible or not request.execution_eligible:
        raise ValueError("blocked review cases cannot bind executable review artifacts")

    scenario_value = scenario.to_dict()
    expected_identity = request.source_scenario_identity
    identity_checks = {
        "schema_version": scenario.schema_version,
        "scenario_id": scenario.scenario_id,
        "corpus_leaf_id": scenario.corpus_leaf_id,
        "backend": scenario.backend,
        "embodiment": scenario_value["embodiment"],
        "task_variant": scenario.task_variant,
        "rng_subseeds": scenario_value["rng_subseeds"],
        "counterfactual_bundle_id": scenario.counterfactual.bundle_id,
        "counterfactual_branch_id": scenario.counterfactual.branch_id,
        "counterfactual_sibling_index": scenario.counterfactual.sibling_index,
        "robocasa_catalog_sha256": scenario.robocasa_manifest.catalog_sha256,
        "robocasa_license_sha256": scenario.robocasa_manifest.license_manifest_sha256,
    }
    mismatched = sorted(
        name
        for name, actual in identity_checks.items()
        if expected_identity.get(name) != actual
    )
    if mismatched:
        raise ValueError(
            "source scenario differs from the immutable review request: "
            + ", ".join(mismatched)
        )
    if (
        case.corpus_leaf_id != scenario.corpus_leaf_id
        or case.backend != scenario.backend
        or case.task_variant != scenario.task_variant
        or case.embodiment != scenario.embodiment.end_effector
        or case.rng_subseeds != scenario.rng_subseeds
        or bundle.plan.fixed_master_seed != REVIEW_MASTER_SEED
    ):
        raise ValueError("source scenario differs from the fixed review case or seed")
    if case.requires_real_robocasa != bool(scenario.robocasa_manifest.assets):
        raise ValueError(
            "source scenario RoboCasa assets differ from the fixed scene requirement"
        )

    source_path = _review_dataset_file(
        root, source_manifest_path, label="source manifest"
    )
    source_manifest = _load_json_mapping(source_path, label="source manifest")
    if source_manifest.get("schema_version") != STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA:
        raise ValueError("review source manifest uses an unsupported schema")
    if source_manifest.get("backend") != scenario.backend:
        raise ValueError("review source manifest is bound to another backend")
    source_hashes = source_manifest.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or dict(source_hashes) != dict(
        scenario.source_hashes
    ):
        raise ValueError("review source manifest differs from the scenario source pins")

    qc_path = _review_dataset_file(root, qc_report_path, label="review QC report")
    qc_report = _load_json_mapping(qc_path, label="review QC report")
    episode_qc, key_event_time_s = _strict_qc_episode(
        qc_report,
        episode_uuid=case.episode_uuid,
        dataset_episode_index=None,
        evaluator_id=case.evaluator,
    )
    timestamps = tuple(float(value) for value in frame_timestamps_s)
    event_indices = event_strip_frame_indices(timestamps, key_event_time_s)
    manifest = ReviewArtifactManifest(
        leaf_id=case.corpus_leaf_id,
        episode_uuid=case.episode_uuid,
        rollout_index=case.rollout_index,
        fixed_master_seed=bundle.plan.fixed_master_seed,
        review_plan_sha256=bundle.plan.plan_sha256,
        review_case_sha256=case.case_sha256,
        review_request_ledger_sha256=bundle.requests.ledger_sha256,
        review_request_sha256=request.request_sha256,
        scenario_spec_sha256=scenario.spec_hash,
        qc_report_sha256=sha256_file(qc_path),
        qc_report_schema=str(qc_report["schema_version"]),
        qc_episode_result_sha256=sha256_json(episode_qc),
        qc_strict_all=True,
        # The report is already required to be strict-all by
        # ``_strict_qc_episode``.  Use the selected episode result here so a
        # failure in another leaf cannot poison this artifact. Dataset-global
        # hard failures remain applicable to every episode.
        automated_qc_passed=bool(
            episode_qc.get("passed") is True
            and not qc_report["global_failures"]
        ),
        source_manifest_sha256=sha256_file(source_path),
        frame_timestamps_sha256=sha256_json(timestamps),
        key_event_time_s=key_event_time_s,
        video_paths=dict(video_paths),
        video_sha256=_review_media_hashes(root, video_paths),
        event_strip_paths=dict(event_strip_paths),
        event_strip_sha256=_review_media_hashes(root, event_strip_paths),
        event_strip_indices=event_indices,
    )
    manifest.validate()
    return manifest


@dataclass(frozen=True, slots=True)
class HumanReview:
    artifact_binding_sha256: str
    reviewer: str
    reviewed_at: str
    checks: Mapping[str, bool]
    notes: str = ""

    def validate(self) -> None:
        if not _valid_sha256(self.artifact_binding_sha256):
            raise ValueError("human review is not bound to an artifact manifest")
        if not self.reviewer.strip():
            raise ValueError("human review requires a reviewer identity")
        try:
            timestamp = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
        except (TypeError, ValueError) as error:
            raise ValueError("human review timestamp must be ISO-8601") from error
        if timestamp.tzinfo is None:
            raise ValueError("human review timestamp must include a timezone")
        if set(self.checks) != set(REVIEW_CHECKS) or any(
            not isinstance(value, bool) for value in self.checks.values()
        ):
            raise ValueError("human review must answer every canonical boolean check")

    @property
    def accepted(self) -> bool:
        self.validate()
        return all(self.checks.values())


@dataclass(frozen=True, slots=True)
class ReviewItem:
    rollout_index: int
    scene_profile: str
    artifacts: ReviewArtifactManifest
    automated_qc_passed: bool
    review: HumanReview | None = None

    def validate(self) -> None:
        if self.rollout_index < 0:
            raise ValueError("review rollout index must be non-negative")
        if not self.scene_profile:
            raise ValueError("review item requires a scene profile")
        self.artifacts.validate()
        if self.rollout_index != self.artifacts.rollout_index:
            raise ValueError("review item rollout index differs from its fixed case")
        if self.automated_qc_passed != self.artifacts.automated_qc_passed:
            raise ValueError("review item automated QC claim differs from its QC artifact")
        if self.review is not None:
            self.review.validate()
            if self.review.artifact_binding_sha256 != self.artifacts.binding_sha256:
                raise ValueError("human review hash does not match the reviewed artifacts")

    @property
    def accepted(self) -> bool:
        self.validate()
        return bool(
            self.automated_qc_passed
            and self.review is not None
            and self.review.accepted
        )


@dataclass(frozen=True, slots=True)
class HumanReviewLedger:
    items: tuple[ReviewItem, ...]
    review_plan_sha256: str
    review_request_ledger_sha256: str
    expected_rollouts_per_leaf: int = 6
    schema_version: str = REVIEW_LEDGER_SCHEMA
    extras: Mapping[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        required_leaf_ids: Sequence[str] = (),
        require_complete_reviews: bool = False,
    ) -> None:
        if self.schema_version != REVIEW_LEDGER_SCHEMA:
            raise ValueError(f"review ledger must use {REVIEW_LEDGER_SCHEMA}")
        if not _valid_sha256(self.review_plan_sha256) or not _valid_sha256(
            self.review_request_ledger_sha256
        ):
            raise ValueError("review ledger is not bound to its immutable plan and requests")
        if self.expected_rollouts_per_leaf != 6:
            raise ValueError("leaf activation requires exactly six fixed review rollouts")
        identities: set[tuple[str, int]] = set()
        scenes_by_leaf: dict[str, list[str]] = {}
        for item in self.items:
            item.validate()
            if item.artifacts.review_plan_sha256 != self.review_plan_sha256:
                raise ValueError("review item is bound to a different review plan")
            if (
                item.artifacts.review_request_ledger_sha256
                != self.review_request_ledger_sha256
            ):
                raise ValueError("review item is bound to a different request ledger")
            identity = (item.artifacts.leaf_id, item.rollout_index)
            if identity in identities:
                raise ValueError(f"duplicate review rollout identity: {identity}")
            identities.add(identity)
            scenes_by_leaf.setdefault(item.artifacts.leaf_id, []).append(item.scene_profile)
            if require_complete_reviews and not item.accepted:
                raise ValueError(
                    f"review rollout is not accepted: {item.artifacts.leaf_id}/{item.rollout_index}"
                )
        for leaf_id, scenes in scenes_by_leaf.items():
            if len(scenes) != self.expected_rollouts_per_leaf:
                raise ValueError(f"leaf {leaf_id} does not have exactly six review rollouts")
            ordered = [
                item.scene_profile
                for item in sorted(
                    (value for value in self.items if value.artifacts.leaf_id == leaf_id),
                    key=lambda value: value.rollout_index,
                )
            ]
            if tuple(ordered) != REVIEW_SCENE_SEQUENCE:
                raise ValueError(f"leaf {leaf_id} does not use the fixed review scene sequence")
        missing = sorted(set(required_leaf_ids) - set(scenes_by_leaf))
        if missing:
            raise ValueError(f"review ledger lacks required leaves: {missing}")

    @property
    def ledger_sha256(self) -> str:
        self.validate()
        return sha256_json(
            {
                "schema_version": self.schema_version,
                "review_plan_sha256": self.review_plan_sha256,
                "review_request_ledger_sha256": self.review_request_ledger_sha256,
                "expected_rollouts_per_leaf": self.expected_rollouts_per_leaf,
                "items": [asdict(value) for value in self.items],
                "extras": self.extras,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "review_plan_sha256": self.review_plan_sha256,
            "review_request_ledger_sha256": self.review_request_ledger_sha256,
            "expected_rollouts_per_leaf": self.expected_rollouts_per_leaf,
            "items": [asdict(value) for value in self.items],
            "extras": dict(self.extras),
            "ledger_sha256": self.ledger_sha256,
        }

    def activation_failures(self, leaf_id: str) -> list[str]:
        selected = sorted(
            (value for value in self.items if value.artifacts.leaf_id == leaf_id),
            key=lambda value: value.rollout_index,
        )
        failures: list[str] = []
        if len(selected) != self.expected_rollouts_per_leaf:
            failures.append("leaf lacks exactly six fixed review rollouts")
        for item in selected:
            if not item.automated_qc_passed:
                failures.append(f"rollout {item.rollout_index} failed automated strict-all QC")
            if item.review is None:
                failures.append(f"rollout {item.rollout_index} lacks human review")
            elif not item.review.accepted:
                failures.append(f"rollout {item.rollout_index} failed human review")
        return failures

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HumanReviewLedger":
        """Load a persisted pending or completed ledger and verify its digest."""

        raw_items = value.get("items")
        if not isinstance(raw_items, Sequence) or isinstance(
            raw_items, (str, bytes, bytearray)
        ):
            raise ValueError("review ledger items must be a sequence")
        items: list[ReviewItem] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise ValueError("review ledger item must be a mapping")
            raw_artifact = raw_item.get("artifacts")
            if not isinstance(raw_artifact, Mapping):
                raise ValueError("review ledger item lacks its artifact manifest")
            artifact = ReviewArtifactManifest.from_dict(raw_artifact)
            raw_review = raw_item.get("review")
            review = None
            if raw_review is not None:
                if not isinstance(raw_review, Mapping):
                    raise ValueError("human review must be a mapping")
                checks = raw_review.get("checks")
                if not isinstance(checks, Mapping):
                    raise ValueError("human review lacks canonical checks")
                review = HumanReview(
                    artifact_binding_sha256=str(
                        raw_review.get("artifact_binding_sha256", "")
                    ),
                    reviewer=str(raw_review.get("reviewer", "")),
                    reviewed_at=str(raw_review.get("reviewed_at", "")),
                    checks={str(key): item for key, item in checks.items()},
                    notes=str(raw_review.get("notes", "")),
                )
            automated = raw_item.get("automated_qc_passed")
            if not isinstance(automated, bool):
                raise ValueError("review item automated QC value must be boolean")
            items.append(
                ReviewItem(
                    rollout_index=int(raw_item.get("rollout_index", -1)),
                    scene_profile=str(raw_item.get("scene_profile", "")),
                    artifacts=artifact,
                    automated_qc_passed=automated,
                    review=review,
                )
            )
        extras = value.get("extras") or {}
        if not isinstance(extras, Mapping):
            raise ValueError("review ledger extras must be a mapping")
        result = cls(
            items=tuple(items),
            review_plan_sha256=str(value.get("review_plan_sha256", "")),
            review_request_ledger_sha256=str(
                value.get("review_request_ledger_sha256", "")
            ),
            expected_rollouts_per_leaf=int(
                value.get("expected_rollouts_per_leaf", -1)
            ),
            schema_version=str(value.get("schema_version", "")),
            extras=dict(extras),
        )
        result.validate()
        declared = value.get("ledger_sha256")
        if declared is not None and declared != result.ledger_sha256:
            raise ValueError("human review ledger hash mismatch")
        return result


def write_human_review_ledger(
    ledger: HumanReviewLedger,
    destination: str | Path,
) -> Path:
    """Publish a ledger immutably; a changed review requires a new artifact."""

    ledger.validate()
    return atomic_write_json(destination, ledger.to_dict())


__all__ = [
    "EVENT_STRIP_LABELS",
    "EventStripSpec",
    "HumanReview",
    "HumanReviewLedger",
    "REVIEW_ARTIFACT_SCHEMA",
    "REVIEW_CHECKS",
    "REVIEW_LEDGER_SCHEMA",
    "REVIEW_MEDIA_PACK_SCHEMA",
    "REVIEW_SCENE_SEQUENCE",
    "STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA",
    "ReviewArtifactManifest",
    "ReviewItem",
    "ReviewMediaPackManifest",
    "ReviewMediaPackPublication",
    "bind_review_artifacts",
    "bind_review_artifacts_strict",
    "bind_fixed_review_media_pack_strict",
    "event_strip_frame_indices",
    "write_fixed_review_media_pack",
    "write_human_review_ledger",
]
