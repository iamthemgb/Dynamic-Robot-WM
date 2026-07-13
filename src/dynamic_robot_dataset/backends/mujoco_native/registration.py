"""Bridge the native evaluator into the dataset-wide persisted-QC registry.

The native evaluator deliberately consumes only the immutable scenario
contract and committed frame/event tables.  This adapter translates its
result into the common v2 outcome contract without consulting branch intent
or an online simulator result.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...common.contract_v2 import (
    DEFAULT_OBJECTIVE_EVALUATORS,
    ObjectiveRecomputeInput,
    ObjectiveRecomputeResult,
)
from ...common.schema import ActualOutcomeClass
from ..base import ScenarioSpec
from .evaluators import EVALUATOR_ID, EVALUATOR_VERSION, evaluate_saved_native_episode


def _scenario_payload(evidence: ObjectiveRecomputeInput) -> Mapping[str, Any]:
    extras = evidence.record.extras
    raw = extras.get("native_scenario_spec")
    if not isinstance(raw, Mapping):
        provenance = extras.get("backend_provenance")
        if isinstance(provenance, Mapping):
            raw = provenance.get("native_scenario_spec")
    if not isinstance(raw, Mapping):
        raise ValueError(
            "native objective recomputation requires extras.native_scenario_spec"
        )
    return raw


def recompute_native_objective(
    evidence: ObjectiveRecomputeInput,
) -> ObjectiveRecomputeResult:
    """Recompute one native objective solely from persisted artifacts."""

    evaluation = evaluate_saved_native_episode(
        ScenarioSpec.from_dict(_scenario_payload(evidence)),
        evidence.frame_rows,
        evidence.event_rows,
    )
    task_contact_times = [
        float(row["timestamp"])
        for row in evidence.event_rows
        if row.get("object_b") == "native_tool"
        or row.get("contact_role") == "task_contact"
    ]
    if task_contact_times:
        key_event_time = min(task_contact_times)
        key_event_name = "first_task_contact"
    elif evidence.event_rows:
        first_event = min(
            evidence.event_rows, key=lambda row: float(row["timestamp"])
        )
        key_event_time = float(first_event["timestamp"])
        key_event_name = f"first_{first_event.get('object_b', 'physics')}_contact"
    else:
        key_event_time = (
            float(evidence.frame_rows[-1]["timestamp"])
            if evidence.frame_rows
            else 0.0
        )
        key_event_name = "terminal_observation"
    result = ObjectiveRecomputeResult(
        task_success=evaluation.outcome.task_success,
        actual_outcome_class=ActualOutcomeClass(evaluation.actual_outcome_class),
        primary_failure_code=evaluation.outcome.failure_mode,
        partial_success_score=evaluation.outcome.partial_success_score,
        evidence=dict(evaluation.evidence),
        key_event_name=key_event_name,
        key_event_time_s=key_event_time,
    )
    result.validate()
    return result


def register_common_objective_evaluator() -> None:
    """Idempotently install the native evaluator in the common registry."""

    try:
        DEFAULT_OBJECTIVE_EVALUATORS.register(
            EVALUATOR_ID,
            EVALUATOR_VERSION,
            recompute_native_objective,
        )
    except ValueError as error:
        if "already registered" not in str(error):
            raise


__all__ = ["recompute_native_objective", "register_common_objective_evaluator"]
