"""Unified, non-fluid dynamic-object robotics dataset generators."""

from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("dynamic-robot-dataset")
except Exception:  # editable source tree before package installation
    __version__ = "0+unknown"

from .common import EpisodeRecord, OutcomeResult, SplitAssigner, export_wan, validate_dataset

__all__ = [
    "EpisodeRecord",
    "OutcomeResult",
    "SplitAssigner",
    "__version__",
    "export_wan",
    "validate_dataset",
]

