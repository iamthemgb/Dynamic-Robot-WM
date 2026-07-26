"""Event-centered Wan video export and ``wan-physics-manifest/v1`` records."""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .episode_writer import DatasetLayout, canonical_camera_name, load_episode_records
from .hashing import sha256_file
from .paths import AtomicDirectory, atomic_write_bytes, atomic_write_json, resolve_dataset_path
from .schema import EpisodeRecord, WAN_MANIFEST_VERSION
from .synchronization import exact_frame_timestamps
from .video_writer import FrameVideoWriter, VideoSpec, iter_rgb_frames, probe_frame_timestamps, probe_video

WAN_VIDEO_SPEC = VideoSpec(width=832, height=480, fps_num=24, fps_den=1)
WAN_FRAME_COUNT = 121
WAN_PHYSICS_FIELDS_VERSION = "wan-physics-fields/v1"
WAN_PHYSICS_FIELDS = (
    "gravity_x_m_s2", "gravity_y_m_s2", "gravity_z_m_s2", "mass_kg", "radius_m",
    "static_friction", "dynamic_friction", "torsional_friction", "rolling_friction",
    "restitution", "contact_stiffness_n_m", "density_kg_m3", "thickness_m",
    "linear_damping", "angular_damping", "young_modulus_pa", "poisson_ratio",
    "stretch_stiffness", "bend_stiffness", "shear_stiffness", "twist_stiffness",
    "rope_length_m", "rope_radius_m", "rope_linear_density_kg_m",
    "cloth_areal_density_kg_m2", "table_friction", "gripper_friction", "air_drag",
    "fluid_density_kg_m3", "sim_timestep_s", "control_timestep_s", "video_timestep_s",
    "deformable_edge_damping_source_units",
)


@dataclass(slots=True, frozen=True)
class WanExportSummary:
    """Counts and paths returned by the stable export API."""

    source_root: str
    output_root: str
    manifest_path: str
    episode_count: int
    video_count: int
    excluded_count: int
    unique_source_episode_seconds: float
    encoded_source_stream_seconds: float
    unique_derived_clip_seconds: float
    encoded_derived_stream_seconds: float
    endpoint_padding_seconds: float
    total_duration_s: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            unique_source_episode_hours=self.unique_source_episode_seconds / 3600.0,
            encoded_source_stream_hours=self.encoded_source_stream_seconds / 3600.0,
            unique_derived_clip_hours=self.unique_derived_clip_seconds / 3600.0,
            encoded_derived_stream_hours=self.encoded_derived_stream_seconds / 3600.0,
            endpoint_padding_hours=self.endpoint_padding_seconds / 3600.0,
            total_duration_semantics="deprecated_alias_of_unique_derived_clip_seconds",
        )
        return value


def _nearest_indices(source_timestamps: Sequence[float], target_timestamps: Sequence[float]) -> list[int]:
    if not source_timestamps:
        raise ValueError("Source video has no presentation timestamps")
    result: list[int] = []
    for target in target_timestamps:
        right = bisect.bisect_left(source_timestamps, target)
        candidates = [index for index in (right - 1, right) if 0 <= index < len(source_timestamps)]
        result.append(min(candidates, key=lambda index: (abs(source_timestamps[index] - target), index)))
    return result


def _selected_frames(
    source: Path,
    source_indices: Sequence[int],
    width: int,
    height: int,
) -> Iterator[bytes]:
    """Decode once and yield the monotonic nearest-neighbor frame map."""

    if any(right < left for left, right in zip(source_indices, source_indices[1:])):
        raise ValueError("Source indices must be monotonic")
    desired_position = 0
    last_frame: bytes | None = None
    for source_index, frame in enumerate(iter_rgb_frames(source, width, height)):
        last_frame = frame
        while desired_position < len(source_indices) and source_indices[desired_position] == source_index:
            yield frame
            desired_position += 1
        if desired_position >= len(source_indices):
            break
    if last_frame is None:
        raise ValueError(f"Could not decode source video: {source}")
    while desired_position < len(source_indices):
        # FFprobe and decoder counts can differ for malformed files. Endpoint
        # padding remains explicit in the manifest rather than stretching time.
        yield last_frame
        desired_position += 1


def _export_one_video(
    source: Path,
    destination: Path,
    event_time_s: float,
) -> tuple[list[int], int, int]:
    probe = probe_video(source)
    if (probe.width, probe.height) != (WAN_VIDEO_SPEC.width, WAN_VIDEO_SPEC.height):
        raise ValueError(
            f"Wan export refuses to stretch {probe.width}x{probe.height}; expected 832x480: {source}"
        )
    source_pts = probe_frame_timestamps(source)
    target_relative = exact_frame_timestamps(WAN_FRAME_COUNT, 24)
    window_start = event_time_s - target_relative[-1] / 2
    absolute_targets = [window_start + value for value in target_relative]
    indices = _nearest_indices(source_pts, absolute_targets)
    start_padding = sum(target < source_pts[0] for target in absolute_targets)
    end_padding = sum(target > source_pts[-1] for target in absolute_targets)
    with FrameVideoWriter(destination, WAN_VIDEO_SPEC) as writer:
        for frame in _selected_frames(source, indices, probe.width, probe.height):
            writer.write(frame)
    return indices, start_padding, end_padding


def _physics_and_masks(
    record: EpisodeRecord,
    time_base: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any], dict[str, str], dict[str, Any]]:
    """Map source physics into the fixed Wan field order without guessing."""

    values = {name: None for name in WAN_PHYSICS_FIELDS}
    masks = {name: False for name in WAN_PHYSICS_FIELDS}
    metadata: dict[str, Any] = {name: None for name in WAN_PHYSICS_FIELDS}
    source_mapping: dict[str, str] = {}
    raw_source: dict[str, Any] = {}
    for name, parameter in sorted(record.physics.parameters.items()):
        raw_source[name] = {
            "value": parameter.value,
            "unit": parameter.unit,
            "valid": parameter.valid,
            "implemented": parameter.implemented,
            "kind": parameter.kind.value,
            "source": parameter.source,
            "observable_in_prefix": parameter.observable_in_prefix,
        }

    def assign(canonical: str, source: str, value: Any | None = None) -> bool:
        parameter = record.physics.parameters.get(source)
        if parameter is None or not (parameter.valid and parameter.implemented):
            return False
        values[canonical] = parameter.value if value is None else value
        masks[canonical] = True
        metadata[canonical] = {
            "unit": parameter.unit,
            "kind": parameter.kind.value,
            "source": parameter.source,
            "observable_in_prefix": parameter.observable_in_prefix,
        }
        source_mapping[canonical] = source
        return True

    if record.physics.gravity_valid:
        for axis, component in zip("xyz", record.physics.gravity_world_m_s2):
            field = f"gravity_{axis}_m_s2"
            values[field] = component
            masks[field] = True
            metadata[field] = {"unit": "m/s^2", "kind": "physical", "source": "gravity_world_m_s2"}
            source_mapping[field] = "gravity_world_m_s2"
    assign("mass_kg", "mass")
    if record.family == "rope":
        assign("rope_radius_m", "radius")
        assign("rope_length_m", "length")
        assign("rope_linear_density_kg_m", "density")
    else:
        assign("radius_m", "radius")
    for source in ("surface_static_friction", "floor_static_friction", "tool_static_friction"):
        if assign("static_friction", source):
            break
    for source in ("surface_dynamic_friction", "floor_dynamic_friction", "dynamic_friction"):
        if assign("dynamic_friction", source):
            break
    for source in ("restitution", "floor_restitution", "surface_restitution", "tool_restitution"):
        if assign("restitution", source):
            break
    direct = {
        "torsional_friction": "torsional_friction",
        "rolling_friction": "rolling_friction",
        "contact_stiffness_n_m": "contact_stiffness",
        "thickness_m": "thickness",
        "linear_damping": "linear_damping",
        "angular_damping": "angular_damping",
        "young_modulus_pa": "youngs_modulus",
        "poisson_ratio": "poisson_ratio",
        "stretch_stiffness": "stretch_stiffness",
        "bend_stiffness": "bend_stiffness",
        "shear_stiffness": "shear_stiffness",
        "twist_stiffness": "twist_stiffness",
        "air_drag": "air_drag",
        "sim_timestep_s": "simulation_timestep",
    }
    for canonical, source in direct.items():
        assign(canonical, source)
    if record.family == "cloth":
        assign("cloth_areal_density_kg_m2", "density")
        assign("table_friction", "cloth_table_friction")
        assign("gripper_friction", "cloth_tool_friction")
        assign("deformable_edge_damping_source_units", "damping")
    elif record.family == "rope":
        assign("table_friction", "rope_table_friction")
        assign("gripper_friction", "rope_tool_friction")
        assign("deformable_edge_damping_source_units", "damping")
    elif record.family == "soft_body":
        assign("density_kg_m3", "density")
        assign("dynamic_friction", "friction")
        assign("deformable_edge_damping_source_units", "damping")
    else:
        assign("density_kg_m3", "density")
    if time_base.get("control_hz"):
        values["control_timestep_s"] = 1.0 / float(time_base["control_hz"])
        masks["control_timestep_s"] = True
        metadata["control_timestep_s"] = {"unit": "s", "kind": "physical", "source": "dataset_info.time_base.control_hz"}
        source_mapping["control_timestep_s"] = "dataset_info.time_base.control_hz"
    if time_base.get("video_hz"):
        values["video_timestep_s"] = 1.0 / float(time_base["video_hz"])
        masks["video_timestep_s"] = True
        metadata["video_timestep_s"] = {"unit": "s", "kind": "physical", "source": "dataset_info.time_base.video_hz"}
        source_mapping["video_timestep_s"] = "dataset_info.time_base.video_hz"
    # Reserved only for loader shape compatibility; fluids are never generated.
    values["fluid_density_kg_m3"] = None
    masks["fluid_density_kg_m3"] = False
    return values, masks, metadata, source_mapping, raw_source


def _parquet_summary(dataset_root: Path, relative: str | None) -> tuple[int, list[str]]:
    if relative is None:
        return 0, []
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Wan availability masks require pyarrow") from exc
    parquet = pq.ParquetFile(resolve_dataset_path(dataset_root, relative))
    return int(parquet.metadata.num_rows), list(parquet.schema_arrow.names)


def _modality_manifest(dataset_root: Path, record: EpisodeRecord) -> dict[str, Any]:
    frame_rows, frame_fields = _parquet_summary(dataset_root, record.frame_data_path)
    high_rows, high_fields = _parquet_summary(dataset_root, record.high_rate_path)
    object_rows, object_fields = _parquet_summary(dataset_root, record.object_states_path)
    event_rows, event_fields = _parquet_summary(dataset_root, record.events_path)
    action_fields = sorted(field for field in frame_fields if field.startswith("action."))
    state_fields = sorted(
        field
        for field in frame_fields
        if field not in {"episode_index", "frame_index", "video_frame_index", "task_index", "timestamp"}
        and not field.startswith("action.")
        and not field.startswith("contact.")
        and not field.startswith("event.")
        and not field.startswith("assistance.")
    )
    trajectory_fields = sorted(
        field
        for field in frame_fields
        if any(token in field.lower() for token in ("position", "vertices", "points", "centroid"))
        and not field.startswith("action.")
    )
    if object_rows and object_fields:
        trajectory_path, trajectory_rows, trajectory_fields = record.object_states_path, object_rows, object_fields
    else:
        trajectory_path, trajectory_rows = record.frame_data_path, frame_rows

    def source_reference(path: str | None) -> dict[str, Any]:
        return {
            "path": path,
            "path_root": "canonical_source_root",
            "source_sha256": (
                sha256_file(resolve_dataset_path(dataset_root, path)) if path is not None else None
            ),
        }

    contact_required = {
        "timestamp",
        "object_a",
        "object_b",
        "point_world_m",
        "normal_world",
        "penetration_depth_m",
    }
    return {
        "state": {**source_reference(record.frame_data_path), "available": frame_rows > 0 and bool(state_fields), "fields": state_fields},
        "action": {**source_reference(record.frame_data_path), "available": frame_rows > 0 and bool(action_fields), "fields": action_fields, "action_mode": record.action_mode},
        "high_rate_state": {**source_reference(record.high_rate_path), "available": high_rows > 0 and bool(high_fields), "fields": high_fields},
        "trajectory": {**source_reference(trajectory_path), "available": trajectory_rows > 0 and bool(trajectory_fields), "fields": trajectory_fields},
        "contact": {
            **source_reference(record.events_path),
            "available": contact_required.issubset(event_fields),
            "fields": event_fields,
            "event_count": event_rows,
            "known_all_negative": contact_required.issubset(event_fields) and event_rows == 0,
        },
    }


def export_wan(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    include_nonrelease: bool = False,
    camera_names: Iterable[str] | None = None,
) -> WanExportSummary:
    """Export fixed 24 FPS/121-frame videos and a versioned JSONL manifest.

    Endpoint frames are repeated only when the event-centered five-second clock
    extends outside the source episode. Source motion is sampled on its original
    physical clock; it is never slowed or duration-rescaled.
    """

    source_root = Path(dataset_root).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination == source_root or source_root in destination.parents:
        raise ValueError("Wan export must use a separate output root")
    records = load_episode_records(source_root)
    info_path = source_root / "meta" / "info.json"
    dataset_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    time_base = dict(dataset_info.get("time_base") or {})
    if include_nonrelease:
        selected = list(records)
    else:
        from .qc import validate_dataset

        qc_report = validate_dataset(source_root, deep_video_checks=True)
        if qc_report.global_failures:
            raise ValueError(
                "Default Wan export refuses a dataset with global hard-QC failures: "
                + "; ".join(qc_report.global_failures)
            )
        hard_qc_pass = {episode.episode_uuid: episode.passed for episode in qc_report.episodes}
        selected = [
            record
            for record in records
            if record.release_eligible and hard_qc_pass.get(record.episode_uuid, False)
        ]
    excluded_count = len(records) - len(selected)
    requested_cameras = {canonical_camera_name(name) for name in camera_names} if camera_names else None
    video_count = 0
    endpoint_padding_seconds = 0.0
    manifest_rows: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}

    with AtomicDirectory(destination) as temporary:
        layout = DatasetLayout(temporary)
        for record in selected:
            event_time = record.event_time_s
            if event_time is None:
                event_time = (record.duration_s or 0.0) / 2
            exported_videos: dict[str, str] = {}
            sampling: dict[str, Any] = {}
            for camera, relative_source in sorted(record.video_paths.items()):
                if requested_cameras is not None and camera not in requested_cameras:
                    continue
                relative_destination = layout.video(camera, record.episode_index)
                output_video = resolve_dataset_path(temporary, relative_destination)
                output_video.parent.mkdir(parents=True, exist_ok=True)
                indices, start_padding, end_padding = _export_one_video(
                    resolve_dataset_path(source_root, relative_source), output_video, event_time
                )
                exported_videos[camera] = relative_destination
                sampling[camera] = {
                    "source_video": relative_source,
                    "source_frame_indices": indices,
                    "endpoint_padding_start_frames": start_padding,
                    "endpoint_padding_end_frames": end_padding,
                }
                endpoint_padding_seconds += (start_padding + end_padding) / 24.0
                checksums[relative_destination] = sha256_file(output_video)
                video_count += 1
            if not exported_videos:
                raise ValueError(f"No requested camera exists for episode {record.episode_uuid}")
            physics, physics_mask, physics_metadata, source_mapping, raw_physics = _physics_and_masks(
                record, time_base
            )
            modalities = _modality_manifest(source_root, record)
            manifest_rows.append(
                {
                    "schema_version": WAN_MANIFEST_VERSION,
                    "episode_uuid": record.episode_uuid,
                    "episode_index": record.episode_index,
                    "counterfactual_bundle_id": record.counterfactual_bundle_id,
                    "physics_counterfactual_family_id": record.physics_counterfactual_family_id,
                    "split_group_id": record.split_group_id,
                    "split": record.split.value,
                    "family": record.family,
                    "subfamily": record.subfamily,
                    "variant": record.variant,
                    "text": record.extras.get("text", f"{record.family}: {record.subfamily}"),
                    "videos": exported_videos,
                    "video": exported_videos.get("observation.images.main", next(iter(exported_videos.values()))),
                    "frame_count": WAN_FRAME_COUNT,
                    "fps": 24,
                    "width": 832,
                    "height": 480,
                    "event_time_s_in_source": event_time,
                    "key_event_name": record.key_event_name,
                    "key_event_time_s_in_source": record.key_event_time_s,
                    "sampling": sampling,
                    "camera_stream_calibration_ids": dict(
                        record.camera_stream_calibration_ids
                    ),
                    "coordinate_convention": dict(
                        dataset_info.get("coordinate_convention") or {}
                    ),
                    **modalities,
                    "physics_schema_version": WAN_PHYSICS_FIELDS_VERSION,
                    "physics": physics,
                    "physics_mask": physics_mask,
                    "physics_metadata": physics_metadata,
                    "physics_source_mapping": source_mapping,
                    "raw_source_physics": raw_physics,
                    "parameter_range_provenance": dict(
                        record.physics.parameter_range_provenance
                    ),
                    "solver_settings": dict(record.physics.solver_settings),
                    "controller_profile": dict(record.controller_profile),
                    "robot_start_provenance": dict(record.robot_start_provenance),
                    "tool_calibration_provenance": dict(
                        record.tool_calibration_provenance
                    ),
                    "assistance": dict(record.assistance),
                    "intended_branch": record.intended_branch,
                    "actual_outcome": record.actual_outcome,
                    "actual_outcome_class": record.actual_outcome_class.value,
                    "task_success": record.task_success,
                    "failure_mode": record.failure_mode,
                    "primary_failure_code": record.primary_failure_code,
                    "failure_tags": list(record.failure_tags),
                    "failure_taxonomy_version": record.failure_taxonomy_version,
                    "label_status": record.label_status.value,
                    "objective_evaluator_id": record.objective_evaluator_id,
                    "objective_evaluator_version": record.objective_evaluator_version,
                    "objective_threshold_set_hash": record.objective_threshold_set_hash,
                    "objective_evidence": record.objective_evidence,
                    "dynamics_mode": record.dynamics_mode.value,
                    "release_tier": record.release_tier.value,
                    "physics_qc_pass": record.physics_qc_pass,
                    "quality_flags": record.quality_flags,
                    "source_generator": record.source_generator,
                    "source_generator_version": record.source_generator_version,
                    "generator_git_commit": record.generator_git_commit,
                    "config_hash": record.config_hash,
                    "source_content_hashes": dict(record.content_hashes),
                }
            )
        manifest_relative = "manifest.jsonl"
        payload = b"".join(
            (json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            for row in manifest_rows
        )
        atomic_write_bytes(temporary / manifest_relative, payload)
        checksums[manifest_relative] = sha256_file(temporary / manifest_relative)
        source_episodes = source_root / "meta" / "episodes.parquet"
        source_root_path = temporary / "source_root.json"
        atomic_write_json(
            source_root_path,
            {
                "schema_version": WAN_MANIFEST_VERSION,
                "source_root_at_export": str(source_root),
                "source_episodes_sha256": sha256_file(source_episodes) if source_episodes.is_file() else None,
                "canonical_paths_remain_relative": True,
                "modality_path_root": "canonical_source_root",
                "modality_sidecars_are_referenced_not_copied": True,
            },
        )
        checksums["source_root.json"] = sha256_file(source_root_path)
        atomic_write_json(temporary / "checksums.json", dict(sorted(checksums.items())))
        unique_derived = len(manifest_rows) * WAN_FRAME_COUNT / 24
        summary = WanExportSummary(
            source_root=str(source_root),
            output_root=str(destination),
            manifest_path=manifest_relative,
            episode_count=len(manifest_rows),
            video_count=video_count,
            excluded_count=excluded_count,
            unique_source_episode_seconds=sum(float(record.duration_s or 0.0) for record in selected),
            encoded_source_stream_seconds=sum(
                float(record.duration_s or 0.0)
                * len(
                    [
                        camera
                        for camera in record.video_paths
                        if requested_cameras is None or camera in requested_cameras
                    ]
                )
                for record in selected
            ),
            unique_derived_clip_seconds=unique_derived,
            encoded_derived_stream_seconds=video_count * WAN_FRAME_COUNT / 24,
            endpoint_padding_seconds=endpoint_padding_seconds,
            total_duration_s=unique_derived,
        )
        atomic_write_json(temporary / "export_summary.json", summary.to_dict())
    return summary
