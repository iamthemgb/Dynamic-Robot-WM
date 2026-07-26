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
#
# v3 deliberately does not accept v2 evidence.  v2 proved only that each
# checkpoint was visible in *some* view, which let a poor task-review camera be
# hidden by its companion stream.  v3 additionally applies the task-specific
# required-view contracts below during independent persisted QC replay.
SOURCE_MUJOCO_VISIBILITY_QC_SCHEMA = "source-mujoco-visibility-qc/v3"
SOURCE_MUJOCO_VISIBILITY_MEDIA_BINDING_SCHEMA = (
    "source-mujoco-visibility-media-binding/v1"
)
SOURCE_MUJOCO_TASK_VISIBILITY_CONTRACT_SCHEMA = (
    "source-mujoco-task-visibility-contract/v1"
)
SOURCE_MUJOCO_REQUIRED_CHECKPOINTS = (
    "initial",
    "apex",
    "key_event",
    "final",
)
# Side-on means that the camera optical axis is perpendicular to the physical
# wall normal.  The repaired fixed P0c wall camera is exactly perpendicular;
# retain a small five-percent cosine tolerance for serialized floating-point
# transforms without admitting an oblique three-quarter view.
SOURCE_MUJOCO_MAXIMUM_SIDE_ON_WALL_NORMAL_DOT = 0.05
SOURCE_MUJOCO_TASK_VISIBILITY_REQUIREMENTS = {
    "P0b/main": {
        "contract_id": "P0b/main-required-checkpoints/v1",
        "required_view": "main",
        "required_checkpoints": SOURCE_MUJOCO_REQUIRED_CHECKPOINTS,
        "require_side_on_wall_normal": False,
    },
    "P0c/wall_rebound/secondary": {
        "contract_id": "P0c/wall-rebound-secondary-required-checkpoints/v1",
        "required_view": "secondary",
        "required_checkpoints": SOURCE_MUJOCO_REQUIRED_CHECKPOINTS,
        "require_side_on_wall_normal": True,
    },
}
SOURCE_MUJOCO_VISUAL_THRESHOLDS = {
    "minimum_target_visible_frame_fraction": 0.90,
    "minimum_trajectory_object_area_px": 4,
    "minimum_bbox_margin_px": 8,
    "minimum_key_event_object_area_px": 64,
    "minimum_counterpart_area_px": 16,
    "minimum_planned_counterpart_area_px": 8,
    # Structural legs are much narrower than task surfaces, especially for
    # the sloped P0d fixture.  Four segmentation pixels in either persisted
    # view is enough to prove that a physical leg is rendered, while the
    # grounded/interface geometry is established independently by compiled
    # MuJoCo distances in the background-clearance contract.
    "minimum_structural_support_area_px": 4,
    "maximum_side_on_wall_normal_dot": (
        SOURCE_MUJOCO_MAXIMUM_SIDE_ON_WALL_NORMAL_DOT
    ),
    "maximum_underexposed_fraction": 0.35,
    "maximum_overexposed_fraction": 0.30,
}


def source_mujoco_task_visibility_requirement(
    source_scenario: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Resolve the strict required-view contract from a saved source spec.

    Relevant task identities are intentionally checked redundantly.  A corrupt
    spec must not avoid the stronger contract merely by dropping or changing
    one of ``corpus_leaf_id``, ``task_variant``, or ``motion_kind``.
    """

    if not isinstance(source_scenario, Mapping):
        raise ValueError("SourceScenarioSpec must be a mapping")
    leaf_id = source_scenario.get("corpus_leaf_id")
    task_variant = source_scenario.get("task_variant")
    initial_state = source_scenario.get("initial_state")
    if not isinstance(initial_state, Mapping):
        if leaf_id in {"P0b", "P0c"}:
            raise ValueError("passive task SourceScenarioSpec lacks initial_state")
        return None
    motion_kind = initial_state.get("motion_kind")

    p0b_variants = {"ballistic_projectile", "angled_projectile"}
    p0b_signaled = (
        leaf_id == "P0b"
        or task_variant in p0b_variants
        or motion_kind == "passive_projectile"
    )
    if p0b_signaled:
        if not (
            leaf_id == "P0b"
            and task_variant in p0b_variants
            and motion_kind == "passive_projectile"
        ):
            raise ValueError("P0b projectile task identity is inconsistent")
        return SOURCE_MUJOCO_TASK_VISIBILITY_REQUIREMENTS["P0b/main"]

    if leaf_id == "P0c":
        if task_variant == "wall_rebound":
            if motion_kind != "passive_wall_rebound":
                raise ValueError("P0c wall-rebound motion identity is inconsistent")
            return SOURCE_MUJOCO_TASK_VISIBILITY_REQUIREMENTS[
                "P0c/wall_rebound/secondary"
            ]
        if task_variant == "table_bounce":
            if motion_kind != "passive_table_bounce":
                raise ValueError("P0c table-bounce motion identity is inconsistent")
            return None
        raise ValueError("P0c task variant is missing or unsupported")
    if motion_kind == "passive_wall_rebound":
        raise ValueError("passive wall-rebound motion is not identified as P0c")
    return None


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
    "SOURCE_MUJOCO_TASK_VISIBILITY_CONTRACT_SCHEMA",
    "SOURCE_MUJOCO_REQUIRED_CHECKPOINTS",
    "SOURCE_MUJOCO_MAXIMUM_SIDE_ON_WALL_NORMAL_DOT",
    "SOURCE_MUJOCO_TASK_VISIBILITY_REQUIREMENTS",
    "SOURCE_MUJOCO_VISUAL_THRESHOLDS",
    "source_mujoco_task_visibility_requirement",
    "source_mujoco_visibility_media_binding",
]
