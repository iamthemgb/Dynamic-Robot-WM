"""Objective outcome result types and branch-distribution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol

from .schema import (
    ActualOutcomeClass,
    LabelStatus,
    ReleaseTier,
    SchemaValidationError,
    default_failure_tags,
    infer_actual_outcome_class,
    validate_outcome_contract,
)


class ObjectiveEvaluator(Protocol):
    """Protocol implemented by task-specific objective evaluators."""

    def evaluate(self, rollout: Any) -> "OutcomeResult": ...


@dataclass(slots=True)
class OutcomeResult:
    """Measured task outcome; never a restatement of the intended branch."""

    task_success: bool
    failure_mode: str
    actual_outcome: str
    actual_outcome_class: ActualOutcomeClass | None = None
    primary_failure_code: str | None = None
    failure_tags: list[str] = field(default_factory=list)
    partial_success_score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    label_confidence: float | None = None
    label_status: LabelStatus = LabelStatus.VERIFIED

    def __post_init__(self) -> None:
        if not isinstance(self.label_status, LabelStatus):
            self.label_status = LabelStatus(self.label_status)
        if self.primary_failure_code is None:
            self.primary_failure_code = self.failure_mode
        if self.actual_outcome_class is None:
            self.actual_outcome_class = infer_actual_outcome_class(
                task_success=self.task_success,
                actual_outcome=self.actual_outcome,
                failure_code=self.primary_failure_code,
                label_status=self.label_status,
                partial_success_score=self.partial_success_score,
            )
        elif not isinstance(self.actual_outcome_class, ActualOutcomeClass):
            self.actual_outcome_class = ActualOutcomeClass(self.actual_outcome_class)
        if not self.failure_tags and self.primary_failure_code != "none":
            self.failure_tags = default_failure_tags(self.primary_failure_code)

    def validate(self) -> None:
        if not isinstance(self.task_success, bool):
            raise SchemaValidationError("Outcome task_success must be boolean")
        if not self.actual_outcome.strip():
            raise SchemaValidationError("Outcome actual_outcome is required")
        assert self.actual_outcome_class is not None
        assert self.primary_failure_code is not None
        validate_outcome_contract(
            task_success=self.task_success,
            outcome_class=self.actual_outcome_class,
            primary_failure_code=self.primary_failure_code,
            compatibility_failure_mode=self.failure_mode,
            partial_success_score=self.partial_success_score,
        )
        if self.partial_success_score is not None and not 0 <= self.partial_success_score <= 1:
            raise SchemaValidationError("partial_success_score must be in [0, 1]")
        if self.label_confidence is not None and not 0 <= self.label_confidence <= 1:
            raise SchemaValidationError("label_confidence must be in [0, 1]")

    @property
    def release_tier(self) -> ReleaseTier:
        """Default publication tier implied by label status."""

        return ReleaseTier.FREE_CONTACT if self.label_status == LabelStatus.VERIFIED else ReleaseTier.UNVERIFIED

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "actual_outcome_class": self.actual_outcome_class.value,
            "label_status": self.label_status.value,
        }

    @classmethod
    def unverified(cls, reason: str, metrics: dict[str, Any] | None = None) -> "OutcomeResult":
        """Create an explicitly quarantined result when no evaluator exists."""

        return cls(
            task_success=False,
            failure_mode="label_unverified",
            actual_outcome="unverified",
            actual_outcome_class=ActualOutcomeClass.UNVERIFIED,
            primary_failure_code="label_unverified",
            metrics={"unverified_reason": reason, **(metrics or {})},
            label_confidence=0.0,
            label_status=LabelStatus.UNVERIFIED,
        )


DEFAULT_BRANCH_MIX: tuple[tuple[str, float], ...] = (
    ("success", 0.40),
    ("near_miss", 0.30),
    ("contact_failure", 0.20),
    ("bad_action", 0.10),
)


def branch_counts(total: int, mix: Iterable[tuple[str, float]] = DEFAULT_BRANCH_MIX) -> dict[str, int]:
    """Allocate an integer number of branches with largest-remainder rounding."""

    if total < 0:
        raise ValueError("total must be non-negative")
    weights = list(mix)
    if not weights or any(weight < 0 for _, weight in weights):
        raise ValueError("Branch weights must be non-negative")
    denominator = sum(weight for _, weight in weights)
    if denominator <= 0:
        raise ValueError("At least one branch weight must be positive")
    exact = [(name, total * weight / denominator) for name, weight in weights]
    result = {name: int(value) for name, value in exact}
    remaining = total - sum(result.values())
    order = sorted(exact, key=lambda item: (-(item[1] - int(item[1])), item[0]))
    for name, _ in order[:remaining]:
        result[name] += 1
    return result
