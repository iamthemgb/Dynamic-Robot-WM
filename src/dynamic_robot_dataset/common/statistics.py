"""Unambiguous logical-episode, stream, and derived-export statistics.

Camera streams are observations of one rollout, not additional physical
experiences.  This module therefore keeps logical episode time separate from
encoded stream time and from padded, fixed-window model exports.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .episode_writer import load_episode_records
from .hashing import sha256_file
from .legacy_quarantine import LEGACY_ASSISTED_NAMESPACE
from .paths import resolve_dataset_path
from .schema import EpisodeRecord
from .video_writer import probe_video


def _hours(seconds: float) -> float:
    return float(seconds) / 3600.0


@dataclass(slots=True, frozen=True)
class DurationAccounting:
    """Duration totals whose denominators are explicit."""

    unique_source_episode_seconds: float = 0.0
    encoded_source_stream_seconds: float = 0.0
    unique_derived_clip_seconds: float = 0.0
    encoded_derived_stream_seconds: float = 0.0
    endpoint_padding_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        values = asdict(self)
        values.update(
            unique_source_episode_hours=_hours(self.unique_source_episode_seconds),
            encoded_source_stream_hours=_hours(self.encoded_source_stream_seconds),
            unique_derived_clip_hours=_hours(self.unique_derived_clip_seconds),
            encoded_derived_stream_hours=_hours(self.encoded_derived_stream_seconds),
            endpoint_padding_hours=_hours(self.endpoint_padding_seconds),
        )
        return values


@dataclass(slots=True)
class DatasetStatistics:
    """Auditable counts and duration totals for one canonical dataset."""

    dataset_root: str
    # Headline generated/training-source inventory excludes quarantined legacy,
    # assisted-contact, and scripted-motion episodes.
    logical_episode_count: int
    encoded_source_view_count: int
    inventory_logical_episode_count: int
    inventory_encoded_source_view_count: int
    quarantine_logical_episode_count: int
    quarantine_encoded_source_view_count: int
    release_eligible_episode_count: int
    qc_passed_release_episode_count: int
    unique_scene_count: int
    action_bundle_count: int
    physics_family_count: int
    durations: DurationAccounting
    inventory_durations: DurationAccounting
    quarantine_durations: DurationAccounting
    release_durations: DurationAccounting
    counts_by_family: dict[str, int] = field(default_factory=dict)
    counts_by_tier: dict[str, int] = field(default_factory=dict)
    counts_by_split: dict[str, int] = field(default_factory=dict)
    counts_by_outcome: dict[str, int] = field(default_factory=dict)
    release_hours_by_family: dict[str, float] = field(default_factory=dict)
    release_hours_by_tier: dict[str, float] = field(default_factory=dict)
    release_hours_by_split: dict[str, float] = field(default_factory=dict)
    release_hours_by_outcome: dict[str, float] = field(default_factory=dict)
    intended_actual_confusion: dict[str, int] = field(default_factory=dict)
    categorical_associations: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["durations"] = self.durations.to_dict()
        result["inventory_durations"] = self.inventory_durations.to_dict()
        result["quarantine_durations"] = self.quarantine_durations.to_dict()
        result["release_durations"] = self.release_durations.to_dict()
        result["accounting_semantics"] = {
            "durations": (
                "headline generated/training source; excludes legacy_assisted, "
                "assisted_contact, and scripted_motion quarantine"
            ),
            "inventory_durations": "all canonical episode records",
            "quarantine_durations": (
                "legacy_assisted namespace or assisted/scripted dynamics/release tiers"
            ),
            "unique_duration_rule": "each logical rollout counts once regardless of camera views",
            "encoded_stream_duration_rule": "each encoded camera stream counts separately",
        }
        return result


def cramers_v(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    """Return bias-corrected Cramer's V for two categorical fields.

    The implementation is dependency-free and returns zero for empty or
    degenerate tables.  It is a review signal, not evidence of causality.
    """

    pairs = [(str(row.get(left, "<missing>")), str(row.get(right, "<missing>"))) for row in rows]
    if not pairs:
        return 0.0
    left_values = sorted({value[0] for value in pairs})
    right_values = sorted({value[1] for value in pairs})
    if len(left_values) < 2 or len(right_values) < 2:
        return 0.0
    li = {value: index for index, value in enumerate(left_values)}
    ri = {value: index for index, value in enumerate(right_values)}
    table = [[0 for _ in right_values] for _ in left_values]
    for left_value, right_value in pairs:
        table[li[left_value]][ri[right_value]] += 1
    n = float(len(pairs))
    row_totals = [sum(row) for row in table]
    column_totals = [sum(table[row][column] for row in range(len(table))) for column in range(len(right_values))]
    chi2 = 0.0
    for row in range(len(left_values)):
        for column in range(len(right_values)):
            expected = row_totals[row] * column_totals[column] / n
            if expected > 0:
                chi2 += (table[row][column] - expected) ** 2 / expected
    phi2 = chi2 / n
    rows_count, columns_count = len(left_values), len(right_values)
    correction = ((columns_count - 1) * (rows_count - 1)) / max(n - 1.0, 1.0)
    phi2_corrected = max(0.0, phi2 - correction)
    rows_corrected = rows_count - ((rows_count - 1) ** 2) / max(n - 1.0, 1.0)
    columns_corrected = columns_count - ((columns_count - 1) ** 2) / max(n - 1.0, 1.0)
    denominator = min(rows_corrected - 1.0, columns_corrected - 1.0)
    return math.sqrt(phi2_corrected / denominator) if denominator > 0 else 0.0


def _qc_pass_map(
    dataset_root: Path, records: Sequence[EpisodeRecord]
) -> tuple[dict[str, bool], list[str]]:
    report_path = dataset_root / "qc" / "dataset_report.json"
    if not report_path.is_file():
        return {}, []
    report = json.loads(report_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    episodes_path = dataset_root / "meta" / "episodes.parquet"
    completion_path = dataset_root / "meta" / ".complete.json"
    if report.get("schema_version") != "dynamic-robot-qc-report/v2":
        problems.append("QC report schema is not v2")
    try:
        reported_root = Path(str(report.get("dataset_root"))).resolve(strict=True)
    except (FileNotFoundError, OSError):
        reported_root = None
    if reported_root != dataset_root:
        problems.append("QC report dataset root is stale")
    if not episodes_path.is_file() or report.get("dataset_episodes_sha256") != sha256_file(
        episodes_path
    ):
        problems.append("QC report episodes hash is stale")
    if not completion_path.is_file() or report.get(
        "metadata_complete_manifest_sha256"
    ) != sha256_file(completion_path):
        problems.append("QC report metadata completion hash is stale")
    else:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        content_hashes = dict(completion.get("content_hashes") or {})
        if dict(report.get("metadata_content_hashes") or {}) != content_hashes:
            problems.append("QC report metadata content manifest is stale")
        for name, expected in content_hashes.items():
            path = dataset_root / "meta" / str(name)
            if not path.is_file() or sha256_file(path) != expected:
                problems.append(f"finalized metadata artifact is stale: {name}")
    episode_rows = [
        item for item in report.get("episodes", ()) if isinstance(item, Mapping)
    ]
    if {str(item.get("episode_uuid")) for item in episode_rows} != {
        record.episode_uuid for record in records
    }:
        problems.append("QC report episode membership is stale")
    for record in records:
        for relative, expected in record.content_hashes.items():
            try:
                path = resolve_dataset_path(dataset_root, relative)
            except (TypeError, ValueError):
                problems.append(f"unsafe episode artifact path: {relative}")
                continue
            if not path.is_file() or sha256_file(path) != expected:
                problems.append(f"episode artifact is stale: {relative}")
    if problems:
        return {}, [
            "QC report content binding failed; QC-passed release hours are zero: "
            + "; ".join(sorted(set(problems)))
        ]
    return {
        str(item["episode_uuid"]): bool(item.get("passed", False))
        for item in episode_rows
        if item.get("episode_uuid")
    }, []


def _wan_accounting(
    wan_root: Path | None,
    *,
    include_episode_uuids: set[str] | None = None,
) -> tuple[float, float, float, list[str]]:
    if wan_root is None:
        return 0.0, 0.0, 0.0, []
    manifest_path = wan_root / "manifest.jsonl"
    if not manifest_path.is_file():
        return 0.0, 0.0, 0.0, [f"Wan manifest not found: {manifest_path}"]
    unique = 0.0
    streams = 0.0
    padding = 0.0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            include_episode_uuids is not None
            and str(row.get("episode_uuid") or "") not in include_episode_uuids
        ):
            continue
        duration = float(row.get("frame_count", 0)) / float(row.get("fps", 1))
        unique += duration
        videos = row.get("videos") or {"primary": row.get("video")}
        streams += duration * len([value for value in videos.values() if value])
        for sample in (row.get("sampling") or {}).values():
            padding += (
                int(sample.get("endpoint_padding_start_frames", 0))
                + int(sample.get("endpoint_padding_end_frames", 0))
            ) / float(row.get("fps", 24))
    return unique, streams, padding, []


def _namespace_values(record: EpisodeRecord) -> set[str]:
    """Return explicitly declared dataset/corpus quarantine namespaces."""

    extras = record.extras if isinstance(record.extras, Mapping) else {}
    values = {
        str(value).strip().lower()
        for value in (
            record.family,
            extras.get("namespace"),
            extras.get("dataset_namespace"),
            extras.get("corpus_namespace"),
            extras.get("quarantine_namespace"),
        )
        if value is not None and str(value).strip()
    }
    legacy = extras.get("legacy_quarantine")
    if isinstance(legacy, Mapping) and legacy.get("namespace") is not None:
        values.add(str(legacy["namespace"]).strip().lower())
    return values


def _excluded_from_generated_hours(record: EpisodeRecord) -> bool:
    """Apply the fail-closed legacy/assisted/scripted hour policy."""

    legacy_prefix = LEGACY_ASSISTED_NAMESPACE.lower()
    in_legacy_namespace = any(
        value == legacy_prefix
        or value.startswith(legacy_prefix + "/")
        or value.startswith(legacy_prefix + ":")
        for value in _namespace_values(record)
    )
    release_tier = record.release_tier.value
    dynamics_mode = record.dynamics_mode.value
    return in_legacy_namespace or release_tier in {
        "assisted_contact",
        "scripted_motion",
    } or dynamics_mode in {
        "assisted_contact",
        "scripted_motion",
    }


def _dataset_is_legacy_assisted(root: Path) -> bool:
    """Recognize an explicitly namespaced all-quarantine dataset root."""

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    extras = info.get("extras") if isinstance(info, Mapping) else None
    extras = extras if isinstance(extras, Mapping) else {}
    candidates = (
        info.get("namespace") if isinstance(info, Mapping) else None,
        extras.get("namespace"),
        extras.get("dataset_namespace"),
        extras.get("corpus_namespace"),
        extras.get("quarantine_namespace"),
    )
    prefix = LEGACY_ASSISTED_NAMESPACE.lower()
    return any(
        str(value).strip().lower() == prefix
        or str(value).strip().lower().startswith(prefix + "/")
        or str(value).strip().lower().startswith(prefix + ":")
        for value in candidates
        if value is not None
    )


def collect_dataset_statistics(
    dataset_root: str | Path,
    *,
    wan_root: str | Path | None = None,
    probe_streams: bool = True,
) -> DatasetStatistics:
    """Collect exact logical and encoded durations without conflating views."""

    root = Path(dataset_root).resolve(strict=True)
    records = load_episode_records(root)
    qc_pass, qc_warnings = _qc_pass_map(root, records)
    warnings: list[str] = list(qc_warnings)
    root_is_legacy_assisted = _dataset_is_legacy_assisted(root)
    quarantine = [
        record
        for record in records
        if root_is_legacy_assisted or _excluded_from_generated_hours(record)
    ]
    quarantined_episode_uuids = {record.episode_uuid for record in quarantine}
    generated = [
        record
        for record in records
        if record.episode_uuid not in quarantined_episode_uuids
    ]
    if quarantine:
        warnings.append(
            f"excluded {len(quarantine)} legacy/assisted/scripted episodes and "
            f"{sum(len(record.video_paths) for record in quarantine)} encoded views "
            "from headline generated/training source hours; see quarantine_durations "
            "and inventory_durations"
        )
    generated_ids = {record.episode_uuid for record in generated}
    quarantine_ids = {record.episode_uuid for record in quarantine}
    inventory_source_unique = sum(float(record.duration_s or 0.0) for record in records)
    generated_source_unique = sum(float(record.duration_s or 0.0) for record in generated)
    quarantine_source_unique = sum(float(record.duration_s or 0.0) for record in quarantine)
    inventory_source_stream = 0.0
    generated_source_stream = 0.0
    quarantine_source_stream = 0.0
    inventory_encoded_views = 0
    generated_encoded_views = 0
    quarantine_encoded_views = 0
    for record in records:
        is_quarantine = record.episode_uuid in quarantine_ids
        inventory_encoded_views += len(record.video_paths)
        if is_quarantine:
            quarantine_encoded_views += len(record.video_paths)
        else:
            generated_encoded_views += len(record.video_paths)
        record_stream_duration = 0.0
        if not probe_streams:
            record_stream_duration = float(record.duration_s or 0.0) * len(
                record.video_paths
            )
        else:
            for relative in record.video_paths.values():
                try:
                    record_stream_duration += probe_video(root / relative).duration_s
                except Exception as error:
                    warnings.append(f"could not probe {relative}: {error}")
        inventory_source_stream += record_stream_duration
        if is_quarantine:
            quarantine_source_stream += record_stream_duration
        else:
            generated_source_stream += record_stream_duration
    inventory_derived_unique, inventory_derived_stream, inventory_padding, wan_warnings = _wan_accounting(
        None if wan_root is None else Path(wan_root).resolve(strict=True)
    )
    warnings.extend(wan_warnings)
    generated_derived_unique, generated_derived_stream, generated_padding, _ = _wan_accounting(
        None if wan_root is None else Path(wan_root).resolve(strict=True),
        include_episode_uuids=generated_ids,
    )
    quarantine_derived_unique, quarantine_derived_stream, quarantine_padding, _ = _wan_accounting(
        None if wan_root is None else Path(wan_root).resolve(strict=True),
        include_episode_uuids=quarantine_ids,
    )

    # Namespace/tier quarantine is authoritative even if malformed imported
    # metadata were to claim release eligibility.
    release = [record for record in generated if record.release_eligible]
    qc_release = [record for record in release if qc_pass.get(record.episode_uuid, False)]
    if not qc_pass and release:
        warnings.append(
            "QC report is absent; release hours are reported as zero until hard QC is persisted"
        )
    release_unique = sum(float(record.duration_s or 0.0) for record in qc_release)
    release_stream = sum(
        float(record.duration_s or 0.0) * len(record.video_paths) for record in qc_release
    )
    rows = [record.to_dict() for record in records]
    confusion = Counter(
        f"{record.intended_branch} -> {record.actual_outcome_class.value}"
        for record in records
    )

    associations: dict[str, float] = {}
    association_rows = [
        {
            **row,
            "background_style": (row.get("randomization") or {}).get(
                "background_style",
                (row.get("randomization") or {}).get("scene_style", "<missing>"),
            ),
            "object_asset_id": (row.get("randomization") or {}).get("object_asset_id", "<missing>"),
            "camera_preset_id": (row.get("randomization") or {}).get("camera_preset_id", "<missing>"),
            "object_rgba": (row.get("randomization") or {}).get("object_rgba", "<missing>"),
            "lighting_style": (row.get("randomization") or {}).get("lighting_style", "<missing>"),
            "object_material_id": (row.get("randomization") or {}).get("object_material_id", "<missing>"),
            "tool_geometry_id": (row.get("randomization") or {}).get("tool_geometry_id", "<missing>"),
            "tool_type": (row.get("randomization") or {}).get("tool_type", "<missing>"),
        }
        for row in rows
    ]
    for field_name in (
        "background_style",
        "object_asset_id",
        "camera_preset_id",
        "object_rgba",
        "lighting_style",
        "object_material_id",
        "tool_geometry_id",
        "tool_type",
    ):
        value = cramers_v(association_rows, field_name, "actual_outcome_class")
        associations[f"{field_name}_vs_actual_outcome_class_cramers_v"] = value
        if value > 0.1:
            warnings.append(
                f"review association: Cramer's V({field_name}, actual_outcome_class)="
                f"{value:.4f} > 0.1"
            )

    def count(field_name: str) -> dict[str, int]:
        values = Counter(str(getattr(record, field_name).value if hasattr(getattr(record, field_name), "value") else getattr(record, field_name)) for record in records)
        return dict(sorted(values.items()))

    def release_hours(field_name: str) -> dict[str, float]:
        totals: defaultdict[str, float] = defaultdict(float)
        for record in qc_release:
            raw = getattr(record, field_name)
            key = str(raw.value if hasattr(raw, "value") else raw)
            totals[key] += float(record.duration_s or 0.0) / 3600.0
        return dict(sorted(totals.items()))

    return DatasetStatistics(
        dataset_root=str(root),
        logical_episode_count=len(generated),
        encoded_source_view_count=generated_encoded_views,
        inventory_logical_episode_count=len(records),
        inventory_encoded_source_view_count=inventory_encoded_views,
        quarantine_logical_episode_count=len(quarantine),
        quarantine_encoded_source_view_count=quarantine_encoded_views,
        release_eligible_episode_count=len(release),
        qc_passed_release_episode_count=len(qc_release),
        unique_scene_count=len({(record.family, record.subfamily, record.scene_seed) for record in records}),
        action_bundle_count=len({record.counterfactual_bundle_id for record in records}),
        physics_family_count=len({record.physics_counterfactual_family_id for record in records if record.physics_counterfactual_family_id}),
        durations=DurationAccounting(
            generated_source_unique,
            generated_source_stream,
            generated_derived_unique,
            generated_derived_stream,
            generated_padding,
        ),
        inventory_durations=DurationAccounting(
            inventory_source_unique,
            inventory_source_stream,
            inventory_derived_unique,
            inventory_derived_stream,
            inventory_padding,
        ),
        quarantine_durations=DurationAccounting(
            quarantine_source_unique,
            quarantine_source_stream,
            quarantine_derived_unique,
            quarantine_derived_stream,
            quarantine_padding,
        ),
        release_durations=DurationAccounting(release_unique, release_stream),
        counts_by_family=count("family"),
        counts_by_tier=count("release_tier"),
        counts_by_split=count("split"),
        counts_by_outcome=count("actual_outcome_class"),
        release_hours_by_family=release_hours("family"),
        release_hours_by_tier=release_hours("release_tier"),
        release_hours_by_split=release_hours("split"),
        release_hours_by_outcome=release_hours("actual_outcome_class"),
        intended_actual_confusion=dict(sorted(confusion.items())),
        categorical_associations=associations,
        warnings=sorted(set(warnings)),
    )
