"""CLI adapter for immutable external human-review publication."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .common.review_finalize import finalize_human_review


def command_review_finalize(arguments: argparse.Namespace) -> int:
    result = finalize_human_review(
        arguments.dataset,
        arguments.decisions,
        arguments.output_root,
    )
    print(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if result.status == "approved" else 1


def add_review_finalize_subcommand(subparsers: Any) -> None:
    review = subparsers.add_parser(
        "review-finalize",
        help="publish hash-bound human decisions outside a sealed dataset",
    )
    review.add_argument(
        "--dataset",
        "--dataset-root",
        dest="dataset",
        required=True,
        help="sealed fixed-six review dataset",
    )
    review.add_argument(
        "--decisions",
        required=True,
        help="dynamic-robot-human-review-decisions/v1 JSON",
    )
    review.add_argument(
        "--output-root",
        required=True,
        help="external immutable ledger root (normally dataset_reviews)",
    )
    review.set_defaults(handler=command_review_finalize)


__all__ = ["add_review_finalize_subcommand", "command_review_finalize"]
