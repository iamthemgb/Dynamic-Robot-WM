"""Shared projection from candidate evaluator output to canonical labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import LabelStatus


@dataclass(frozen=True, slots=True)
class CanonicalOutcomeProjection:
    actual_outcome: str
    task_success: bool
    partial_success_score: float | None
    failure_mode: str
    label_confidence: float | None
    diagnostic_candidate_outcome: dict[str, Any] | None


def project_candidate_outcome(
    *,
    actual_outcome: str,
    task_success: bool,
    partial_success_score: float | None,
    failure_mode: str,
    label_confidence: float | None,
    source_label_status: str,
    canonical_label_status: LabelStatus,
) -> CanonicalOutcomeProjection:
    """Keep unverified proxy metrics without presenting them as truth labels."""

    if canonical_label_status == LabelStatus.VERIFIED:
        return CanonicalOutcomeProjection(
            actual_outcome=actual_outcome,
            task_success=task_success,
            partial_success_score=partial_success_score,
            failure_mode=failure_mode,
            label_confidence=label_confidence,
            diagnostic_candidate_outcome=None,
        )
    candidate = {
        "actual_outcome": actual_outcome,
        "task_success": task_success,
        "partial_success_score": partial_success_score,
        "failure_mode": failure_mode,
        "label_confidence": label_confidence,
        "label_status": source_label_status,
        "metrics_preserved_in": "objective_metrics",
    }
    return CanonicalOutcomeProjection(
        actual_outcome="unverified",
        task_success=False,
        partial_success_score=None,
        failure_mode=(
            "legacy_proxy_quarantined"
            if failure_mode == "legacy_proxy_quarantined"
            else "label_unverified"
        ),
        label_confidence=None,
        diagnostic_candidate_outcome=candidate,
    )


__all__ = ["CanonicalOutcomeProjection", "project_candidate_outcome"]
