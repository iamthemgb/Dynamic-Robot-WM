"""Versioned final-state grasp-retention thresholds.

Transient bilateral contact is not a completed catch.  This contract is
persisted in every actuated rigid SourceScenarioSpec and compared exactly by
the independent evaluator so threshold edits require an evaluator/profile
version change rather than silently relabeling sealed episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


GRASP_RETENTION_SCHEMA_VERSION = "dynamic-robot-grasp-retention/v1"


@dataclass(frozen=True, slots=True)
class GraspRetentionThresholds:
    final_window_s: float = 0.10
    minimum_bilateral_fraction: float = 0.95
    maximum_relative_range_m: float = 0.01
    require_bilateral_at_final_sample: bool = True
    schema_version: str = GRASP_RETENTION_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != GRASP_RETENTION_SCHEMA_VERSION:
            raise ValueError("unsupported grasp-retention threshold schema")
        numeric = (
            self.final_window_s,
            self.minimum_bilateral_fraction,
            self.maximum_relative_range_m,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("grasp-retention thresholds must be finite")
        if not 0.05 <= self.final_window_s <= 0.25:
            raise ValueError("final grasp-retention window must be bounded")
        if not 0.9 <= self.minimum_bilateral_fraction <= 1.0:
            raise ValueError("final bilateral-contact fraction must be strict")
        if not 0.0 < self.maximum_relative_range_m <= 0.02:
            raise ValueError("final object-to-grasp range must be bounded")
        if self.require_bilateral_at_final_sample is not True:
            raise ValueError("final sample must retain opposing bilateral contact")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraspRetentionThresholds":
        try:
            result = cls(
                final_window_s=float(value["final_window_s"]),
                minimum_bilateral_fraction=float(
                    value["minimum_bilateral_fraction"]
                ),
                maximum_relative_range_m=float(value["maximum_relative_range_m"]),
                require_bilateral_at_final_sample=value[
                    "require_bilateral_at_final_sample"
                ],
                schema_version=str(value["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid grasp-retention threshold payload") from error
        result.validate()
        return result


DEFAULT_GRASP_RETENTION = GraspRetentionThresholds()
DEFAULT_GRASP_RETENTION.validate()


__all__ = [
    "DEFAULT_GRASP_RETENTION",
    "GRASP_RETENTION_SCHEMA_VERSION",
    "GraspRetentionThresholds",
]
