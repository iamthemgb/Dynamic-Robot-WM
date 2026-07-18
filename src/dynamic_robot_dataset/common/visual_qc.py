"""Shared rendered visual-admission contracts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .hashing import sha256_json

NATIVE_VISUAL_QC_SCHEMA = "native-visual-qc/v1"
NATIVE_VISUAL_THRESHOLDS = {
    "minimum_target_visible_frame_fraction": 0.90,
    "minimum_bbox_margin_px": 8,
    "minimum_key_event_object_area_px": 64,
    "maximum_underexposed_fraction": 0.35,
    "maximum_overexposed_fraction": 0.30,
}

# The owned source_mujoco path uses the same conservative key-event footprint
# as the retired native backend, but it deliberately does not require 64
# target pixels throughout a rollout.  Wide overview cameras may render a
# physically useful target with a smaller footprint.  Whole-trajectory
# presence therefore uses a small, explicit segmentation floor, while the
# key event retains the stronger 64-pixel requirement.  Final and apex frames
# are checked separately so a 90% aggregate cannot hide their disappearance.
SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA = "source-mujoco-visibility-qc/v1"
SOURCE_MUJOCO_VISIBILITY_MEDIA_BINDING_SCHEMA = (
    "source-mujoco-visibility-media-binding/v1"
)
SOURCE_MUJOCO_VISUAL_THRESHOLDS = {
    "minimum_target_visible_frame_fraction": 0.90,
    "minimum_trajectory_object_area_px": 4,
    "minimum_bbox_margin_px": 8,
    "minimum_key_event_object_area_px": 64,
    "minimum_counterpart_area_px": 16,
    "minimum_planned_counterpart_area_px": 8,
    "maximum_underexposed_fraction": 0.35,
    "maximum_overexposed_fraction": 0.30,
}


def source_mujoco_visibility_media_binding(
    *,
    visibility_qc_sha256: str,
    camera_rows: Sequence[Mapping[str, Any]],
    camera_stream_calibration_ids: Mapping[str, str],
    video_paths: Mapping[str, str],
    content_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Bind visibility evidence to the calibrated, encoded camera bytes."""

    required = {
        "observation.images.main",
        "observation.images.secondary",
    }
    if set(camera_stream_calibration_ids) != required or set(video_paths) != required:
        raise ValueError("source visibility binding requires exactly two canonical streams")
    rows_by_stream: dict[str, Mapping[str, Any]] = {}
    for row in camera_rows:
        stream = str(row.get("camera_name") or "")
        if stream in rows_by_stream:
            raise ValueError(f"duplicate camera calibration row for {stream}")
        rows_by_stream[stream] = row
    if set(rows_by_stream) != required:
        raise ValueError("source visibility binding camera rows are incomplete")
    camera_hashes: dict[str, str] = {}
    video_hashes: dict[str, str] = {}
    for stream in sorted(required):
        row = rows_by_stream[stream]
        if str(row.get("camera_id") or "") != str(
            camera_stream_calibration_ids[stream]
        ):
            raise ValueError("camera row ID differs from stream calibration binding")
        camera_hashes[stream] = sha256_json(row)
        relative = str(video_paths[stream])
        digest = str(content_hashes.get(relative) or "")
        if len(digest) != 64:
            raise ValueError(f"encoded video content hash is missing for {stream}")
        video_hashes[stream] = digest
    return {
        "schema_version": SOURCE_MUJOCO_VISIBILITY_MEDIA_BINDING_SCHEMA,
        "visibility_qc_sha256": str(visibility_qc_sha256),
        "camera_calibration_sha256": camera_hashes,
        "camera_stream_calibration_ids": {
            key: str(camera_stream_calibration_ids[key]) for key in sorted(required)
        },
        "video_content_sha256": video_hashes,
    }

__all__ = [
    "NATIVE_VISUAL_QC_SCHEMA",
    "NATIVE_VISUAL_THRESHOLDS",
    "SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA",
    "SOURCE_MUJOCO_VISIBILITY_MEDIA_BINDING_SCHEMA",
    "SOURCE_MUJOCO_VISUAL_THRESHOLDS",
    "source_mujoco_visibility_media_binding",
]
