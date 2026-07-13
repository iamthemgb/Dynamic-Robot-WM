"""Franka ground-bounce ball-catch MuJoCo dataset generation (subfamily F1_B).

Sibling of ``franka_catch`` (F1_A centered vertical drop). The ball drops,
bounces once off the tabletop with a per-episode restitution, and the Franka
reactively snatches it near the rebound apex. Reuses the franka_catch packaging,
variants, IK and contact machinery; overrides only the bounce physics + the
reactive post-bounce controller.
"""
from __future__ import annotations

from .controller import BounceCatchController
from .scene_builder import BOUNCE_SURFACE_Z, build_bounce_episode, sample_bounce_episode
from .taxonomy import FAMILY, SUBFAMILY

__all__ = [
    "BounceCatchController",
    "build_bounce_episode",
    "sample_bounce_episode",
    "BOUNCE_SURFACE_Z",
    "FAMILY",
    "SUBFAMILY",
]
