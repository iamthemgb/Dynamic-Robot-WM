"""Decision-complete 120-branch smoke matrix.

These requests exercise planners, simulation records, metrics, counterfactual
grouping, three background styles, and named physics sweeps.  Rendering and
dataset serialization are intentionally delegated to the common smoke runner.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from . import get_family
from .base import EpisodePlan, GenerationRequest, SimulationResult, STANDARD_BRANCHES
from .deformable.cloth import CLOTH_TASKS
from .deformable.rope import ROPE_TASKS
from .deformable.soft_body import SOFT_BODY_TASKS
from .legacy_proxy_quarantine.adapter import LEGACY_PROXY_TYPES


EXPECTED_SMOKE_COUNTS = {
    "falling_catch": 24,
    "rolling_interception": 16,
    "projectile_rebound": 30,
    "cloth": 16,
    "rope": 18,
    "soft_body": 12,
    "legacy_proxy_quarantine": 4,
}

_STYLES = (
    "clean_franka_lab",
    "robocasa_kitchen_tabletop",
    "robotwin_cluttered_tabletop",
)


def build_smoke_requests(seed: int = 0) -> list[GenerationRequest]:
    requests: list[GenerationRequest] = []
    request_index = 0

    def add(family: str, **kwargs: object) -> None:
        nonlocal request_index
        requests.append(
            GenerationRequest(
                family=family,
                seed=seed + 10_000 * request_index,
                scene_style=_STYLES[request_index % len(_STYLES)],
                options={"smoke_request_index": request_index},
                **kwargs,
            )
        )
        request_index += 1

    add(
        "falling_catch",
        subfamily="centered_vertical_drop",
        num_bundles=6,
        branches=STANDARD_BRANCHES,
    )
    add(
        "rolling_interception",
        subfamily="rolling_island",
        num_bundles=4,
        branches=STANDARD_BRANCHES,
    )

    # 15 parameter-counterfactual episodes: one five-point family per field.
    for sweep in ("gravity", "friction", "restitution"):
        add(
            "projectile_rebound",
            subfamily="free_contact_rebound",
            num_bundles=1,
            branches=("success_seeking",),
            physics_sweep=sweep,
        )
    # Twelve action counterfactuals and three independent nominal episodes.
    add(
        "projectile_rebound",
        subfamily="free_contact_rebound",
        num_bundles=3,
        branches=STANDARD_BRANCHES,
    )
    add(
        "projectile_rebound",
        subfamily="free_contact_rebound",
        num_bundles=3,
        branches=("success_seeking",),
    )

    for task in CLOTH_TASKS:
        add("cloth", subfamily=task, num_bundles=1, branches=STANDARD_BRANCHES)
    for task in ROPE_TASKS:
        add(
            "rope",
            subfamily=task,
            num_bundles=1,
            branches=("success_seeking", "bad_action"),
        )
    for task in SOFT_BODY_TASKS:
        add(
            "soft_body",
            subfamily=task,
            num_bundles=1,
            branches=("success_seeking",),
            physics_sweep="material",
        )
    for proxy_type in LEGACY_PROXY_TYPES:
        add(
            "legacy_proxy_quarantine",
            subfamily=proxy_type,
            num_bundles=1,
            branches=("bad_action",),
        )
    return requests


def plan_smoke(seed: int = 0) -> list[EpisodePlan]:
    plans = [
        episode
        for request in build_smoke_requests(seed)
        for episode in get_family(request.family).plan(request)
    ]
    counts = Counter(episode.family for episode in plans)
    if dict(counts) != EXPECTED_SMOKE_COUNTS:
        raise AssertionError(f"smoke matrix drift: expected {EXPECTED_SMOKE_COUNTS}, got {dict(counts)}")
    if len(plans) != 120:
        raise AssertionError(f"smoke matrix must have exactly 120 branches, got {len(plans)}")
    if len({episode.episode_uuid for episode in plans}) != len(plans):
        raise AssertionError("smoke matrix produced duplicate episode UUIDs")
    if len({episode.scene_style for episode in plans}) < 3:
        raise AssertionError("smoke matrix must use at least three scene styles")
    return plans


def simulate_smoke(seed: int = 0) -> list[SimulationResult]:
    return [get_family(episode.family).simulate(episode) for episode in plan_smoke(seed)]


__all__ = [
    "EXPECTED_SMOKE_COUNTS",
    "build_smoke_requests",
    "plan_smoke",
    "simulate_smoke",
]
