"""CLI adapter for immutable fixed review-suite planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common.review_suite import (
    load_review_suite_bundle,
    load_review_suite_config,
    write_review_suite_bundle,
)


def _execute_source_preview(
    review_suite_root: str | Path,
    dataset_root: str | Path,
    *,
    case_ids: tuple[str, ...],
    leaf_ids: tuple[str, ...],
    clean_r0_only: bool,
    resume: bool,
) -> Any:
    from .common.source_preview import execute_source_review_preview

    return execute_source_review_preview(
        review_suite_root,
        dataset_root,
        case_ids=case_ids,
        leaf_ids=leaf_ids,
        clean_r0_only=clean_r0_only,
        resume=resume,
    )


def command_review_suite(arguments: argparse.Namespace) -> int:
    selectors = bool(arguments.case_id or arguments.leaf_id or arguments.clean_r0_only)
    if not arguments.execute and (
        arguments.dataset_output is not None or arguments.resume or selectors
    ):
        raise ValueError(
            "--dataset-output, --resume, and review selectors require --execute"
        )
    if arguments.execute and arguments.dataset_output is None:
        raise ValueError("--execute requires --dataset-output")
    config = load_review_suite_config(arguments.config)
    if arguments.validate_only:
        bundle = load_review_suite_bundle(arguments.output, config=config)
    else:
        bundle = write_review_suite_bundle(arguments.output, config=config)
    quota = bundle.plan.leaf_execution_quota()
    summary = {
        "output_root": str(Path(arguments.output).resolve()),
        "suite_id": bundle.plan.suite_id,
        "plan_sha256": bundle.plan.plan_sha256,
        "request_ledger_sha256": bundle.requests.ledger_sha256,
        "planned_case_count": len(bundle.plan.cases),
        "required_video_count": len(bundle.plan.cases) * 2,
        "required_event_strip_count": len(bundle.plan.cases) * 2,
        "executable_case_count": bundle.plan.executable_case_count,
        "blocked_leaf_count": sum(value == 0 for value in quota.values()),
        "leaf_execution_quota": quota,
        "validated_only": bool(arguments.validate_only),
    }
    execution = None
    if arguments.execute:
        execution = _execute_source_preview(
            bundle.root,
            arguments.dataset_output,
            case_ids=tuple(arguments.case_id or ()),
            leaf_ids=tuple(arguments.leaf_id or ()),
            clean_r0_only=bool(arguments.clean_r0_only),
            resume=bool(arguments.resume),
        )
        summary["execution"] = execution.to_dict()
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if execution is None or execution.strict_qc_passed else 1


def add_review_suite_subcommand(subparsers: Any) -> None:
    review = subparsers.add_parser(
        "review-suite",
        help="write or validate the immutable fixed 20-by-6 acceptance plan",
    )
    review.add_argument("--output", required=True)
    review.add_argument(
        "--config",
        help="strict review policy (defaults to configs/review/acceptance_v1.yaml)",
    )
    review.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing plan/request bundle without writing",
    )
    review.add_argument(
        "--execute",
        action="store_true",
        help="run a selected fixed source_mujoco preview through finalization and strict QC",
    )
    review.add_argument(
        "--dataset-output",
        help="canonical dataset output for --execute (separate from the review plan root)",
    )
    review.add_argument(
        "--case-id",
        action="append",
        help="fixed case ID to execute; repeat to select multiple cases",
    )
    review.add_argument(
        "--leaf-id",
        action="append",
        help="corpus leaf ID to execute; repeat to select multiple leaves",
    )
    review.add_argument(
        "--clean-r0-only",
        action="store_true",
        help="restrict execution to each selected leaf's fixed clean R0 rollout",
    )
    review.add_argument("--resume", action="store_true", help="resume the exact immutable preview")
    review.set_defaults(handler=command_review_suite)


__all__ = ["add_review_suite_subcommand", "command_review_suite"]
