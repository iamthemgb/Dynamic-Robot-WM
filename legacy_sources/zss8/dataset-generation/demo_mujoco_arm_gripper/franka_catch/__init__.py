"""Franka ball-catch MuJoCo dataset generation.

Physics-aware robotic world-model data: a Franka Panda intercepts a falling ball
across a variety of scene backgrounds. Each episode produces an external-camera
MP4, an optional wrist RGB + depth stream, an optional close-up MP4, a replayable
GLB (geometry + animated cameras), and a JSON metadata sidecar.
"""

from __future__ import annotations

from .assets import AssetInfo, locate_franka_asset
from .controller import MujocoCatchController
from .scene_builder import EpisodeBundle, EpisodeSample, build_episode, sample_episode
from .variants import VARIANT_NAMES, VARIANTS

__all__ = [
    "AssetInfo",
    "locate_franka_asset",
    "MujocoCatchController",
    "EpisodeBundle",
    "EpisodeSample",
    "build_episode",
    "sample_episode",
    "VARIANT_NAMES",
    "VARIANTS",
]
