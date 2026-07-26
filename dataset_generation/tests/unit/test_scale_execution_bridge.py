"""Scale bridge gating, dispatch separation, and fixed-review non-regression."""

import pytest

from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.common.run_orchestration import RunPlanEpisode
from dynamic_robot_dataset.common.scale_execution import (
    SCALE_EXECUTION_BRIDGE_SCHEMA,
    _planned_scale_inputs,
    _require_executable_scale_case,
)
from dynamic_robot_dataset.common.scale_suite import mint_scale_cases
from dynamic_robot_dataset.common.source_execution import (
    SOURCE_EXECUTION_BRIDGE_SCHEMA,
    SourceExecutionBindingError,
    _planned_inputs,
)

# Non-regression: the fork must not change the immutable fixed review plan.
UPSTREAM_REVIEW_PLAN_SHA256 = (
    "48ee988faf69d321b49346c18ef922f4cf7a9d3256bcc7c0cca7d47f1dddc99b"
)


def test_fixed_review_plan_hash_is_unchanged() -> None:
    plan = build_review_suite_plan()
    assert plan.plan_sha256 == UPSTREAM_REVIEW_PLAN_SHA256


def test_scale_case_gate_requires_scale_type() -> None:
    review_case = build_review_suite_plan().cases[24]  # F1a-review-00
    with pytest.raises(TypeError, match="ScaleSuiteCase"):
        _require_executable_scale_case(review_case)


def test_scale_case_gate_admits_minted_cases() -> None:
    case = mint_scale_cases(
        "F1a", episode_start=0, count=1, scale_suite_id="scale-F1a-block-0000"
    )[0]
    _require_executable_scale_case(case)


def test_fixed_executor_rejects_scale_schema() -> None:
    entry = RunPlanEpisode(
        episode_uuid="0" * 8 + "-0000-0000-0000-" + "0" * 12,
        episode_index=0,
        shard_id=0,
        declaration={"schema_version": SCALE_EXECUTION_BRIDGE_SCHEMA},
    )
    with pytest.raises(SourceExecutionBindingError, match="not a canonical source"):
        _planned_inputs(entry)


def test_scale_executor_rejects_review_schema() -> None:
    entry = RunPlanEpisode(
        episode_uuid="0" * 8 + "-0000-0000-0000-" + "0" * 12,
        episode_index=0,
        shard_id=0,
        declaration={"schema_version": SOURCE_EXECUTION_BRIDGE_SCHEMA},
    )
    with pytest.raises(SourceExecutionBindingError, match="not a scale bridge"):
        _planned_scale_inputs(entry)


def test_bridge_schemas_are_distinct() -> None:
    assert SCALE_EXECUTION_BRIDGE_SCHEMA != SOURCE_EXECUTION_BRIDGE_SCHEMA
