"""Exact timestamp construction and deterministic cross-stream synchronization."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Sequence


class SynchronizationError(ValueError):
    """Streams cannot be synchronized without violating timing constraints."""


def pts_to_seconds(pts: Iterable[int], time_base_num: int, time_base_den: int) -> list[float]:
    """Convert integer presentation timestamps using an exact rational time base."""

    if time_base_num <= 0 or time_base_den <= 0:
        raise SynchronizationError("Time-base numerator and denominator must be positive")
    time_base = Fraction(time_base_num, time_base_den)
    return [float(int(value) * time_base) for value in pts]


def exact_frame_timestamps(frame_count: int, fps_num: int, fps_den: int = 1) -> list[float]:
    """Construct ideal timestamps with rational arithmetic, never cumulative floats."""

    if frame_count < 0 or fps_num <= 0 or fps_den <= 0:
        raise SynchronizationError("Frame count must be non-negative and FPS must be positive")
    period = Fraction(fps_den, fps_num)
    return [float(index * period) for index in range(frame_count)]


def validate_monotonic_timestamps(
    timestamps: Sequence[float],
    *,
    strictly: bool = True,
    name: str = "timestamps",
) -> None:
    """Require finite, non-negative, monotonically increasing timestamps."""

    if any(not math.isfinite(value) or value < 0 for value in timestamps):
        raise SynchronizationError(f"{name} must be finite and non-negative")
    for previous, current in zip(timestamps, timestamps[1:]):
        if (current <= previous) if strictly else (current < previous):
            relation = "strictly increasing" if strictly else "monotonic"
            raise SynchronizationError(f"{name} must be {relation}")


@dataclass(slots=True, frozen=True)
class SynchronizedSample:
    """A target timestamp mapped to a source sample."""

    target_timestamp: float
    source_index: int
    source_timestamp: float
    error_s: float
    value: Any


def synchronize_nearest(
    target_timestamps: Sequence[float],
    source_timestamps: Sequence[float],
    source_values: Sequence[Any],
    *,
    max_error_s: float | None = None,
) -> list[SynchronizedSample]:
    """Map each target timestamp to the closest source timestamp.

    Ties choose the earlier source sample.  The returned source timestamp and
    error make the alignment auditable rather than implying ``frame/FPS``.
    """

    validate_monotonic_timestamps(target_timestamps, name="target_timestamps")
    validate_monotonic_timestamps(source_timestamps, name="source_timestamps")
    if len(source_timestamps) != len(source_values) or not source_timestamps:
        raise SynchronizationError("Source timestamps and values must be equal-length and non-empty")
    result: list[SynchronizedSample] = []
    for target in target_timestamps:
        right = bisect.bisect_left(source_timestamps, target)
        candidates = [index for index in (right - 1, right) if 0 <= index < len(source_timestamps)]
        index = min(candidates, key=lambda item: (abs(source_timestamps[item] - target), item))
        error = abs(source_timestamps[index] - target)
        if max_error_s is not None and error > max_error_s:
            raise SynchronizationError(
                f"No source sample within {max_error_s}s of target {target}s (nearest error {error}s)"
            )
        result.append(SynchronizedSample(target, index, source_timestamps[index], error, source_values[index]))
    return result


def synchronize_previous(
    target_timestamps: Sequence[float],
    source_timestamps: Sequence[float],
    source_values: Sequence[Any],
) -> list[SynchronizedSample]:
    """Map target frames to the latest available causal controller sample."""

    validate_monotonic_timestamps(target_timestamps, name="target_timestamps")
    validate_monotonic_timestamps(source_timestamps, name="source_timestamps")
    if len(source_timestamps) != len(source_values) or not source_timestamps:
        raise SynchronizationError("Source timestamps and values must be equal-length and non-empty")
    result: list[SynchronizedSample] = []
    for target in target_timestamps:
        index = bisect.bisect_right(source_timestamps, target) - 1
        if index < 0:
            raise SynchronizationError(f"No causal source sample for target timestamp {target}")
        result.append(
            SynchronizedSample(target, index, source_timestamps[index], target - source_timestamps[index], source_values[index])
        )
    return result


def validate_synchronized_streams(
    streams: dict[str, Sequence[float]],
    *,
    tolerance_s: float = 1e-6,
) -> None:
    """Require camera streams to expose the same frame clock within tolerance."""

    if not streams:
        raise SynchronizationError("At least one stream is required")
    reference_name, reference = next(iter(streams.items()))
    validate_monotonic_timestamps(reference, name=reference_name)
    for name, values in list(streams.items())[1:]:
        validate_monotonic_timestamps(values, name=name)
        if len(values) != len(reference):
            raise SynchronizationError(f"Frame-count mismatch: {reference_name}={len(reference)}, {name}={len(values)}")
        worst = max((abs(left - right) for left, right in zip(reference, values)), default=0.0)
        if worst > tolerance_s:
            raise SynchronizationError(
                f"Camera clocks differ by {worst}s, exceeding tolerance {tolerance_s}s"
            )

