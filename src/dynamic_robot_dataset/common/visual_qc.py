"""Shared native visual-admission contract."""

from __future__ import annotations

NATIVE_VISUAL_QC_SCHEMA = "native-visual-qc/v1"
NATIVE_VISUAL_THRESHOLDS = {
    "minimum_target_visible_frame_fraction": 0.90,
    "minimum_bbox_margin_px": 8,
    "minimum_key_event_object_area_px": 64,
    "maximum_underexposed_fraction": 0.35,
    "maximum_overexposed_fraction": 0.30,
}

__all__ = ["NATIVE_VISUAL_QC_SCHEMA", "NATIVE_VISUAL_THRESHOLDS"]
