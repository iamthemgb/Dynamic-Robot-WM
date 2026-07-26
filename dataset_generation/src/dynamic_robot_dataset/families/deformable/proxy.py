"""Shared deterministic geometry utilities for quarantined smoke adapters."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


Vec3 = tuple[float, float, float]


def smoothstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def interpolate_points(start: Sequence[Vec3], end: Sequence[Vec3], alpha: float) -> list[Vec3]:
    blend = smoothstep(alpha)
    return [
        (
            a[0] + (b[0] - a[0]) * blend,
            a[1] + (b[1] - a[1]) * blend,
            a[2] + (b[2] - a[2]) * blend,
        )
        for a, b in zip(start, end)
    ]


def centroid(points: Sequence[Vec3]) -> Vec3:
    count = len(points)
    return tuple(sum(point[axis] for point in points) / count for axis in range(3))  # type: ignore[return-value]


def polyline_length(points: Sequence[Vec3]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def grid_points(nx: int, ny: int, width: float, height: float, z: float) -> list[Vec3]:
    return [
        (
            -width / 2 + width * ix / (nx - 1),
            -height / 2 + height * iy / (ny - 1),
            z,
        )
        for iy in range(ny)
        for ix in range(nx)
    ]

