"""Short-lived worker for one immutable source-review declaration.

Preparing a RoboCasa-backed declaration compiles provisional and final MuJoCo
models.  Some native allocations are retained by the process even after the
Python objects are released, so the preview parent must never prepare many
cases in-process.  This private module accepts one canonical request on stdin,
prints one declaration on stdout, and exits so the operating system reclaims
all native simulator state.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, Sequence

from .review_suite import ReviewSuiteCase
from .source_execution import prepare_source_review_declaration


SOURCE_PREVIEW_PREPARE_REQUEST_SCHEMA = (
    "dynamic-robot-source-preview-prepare-request/v1"
)


def prepare_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and execute exactly one declaration-preparation request."""

    if value.get("schema_version") != SOURCE_PREVIEW_PREPARE_REQUEST_SCHEMA:
        raise ValueError("unsupported source preview prepare-request schema")
    raw_case = value.get("review_case")
    if not isinstance(raw_case, Mapping):
        raise ValueError("source preview prepare request lacks a review case")
    episode_index = value.get("episode_index")
    if (
        not isinstance(episode_index, int)
        or isinstance(episode_index, bool)
        or episode_index < 0
    ):
        raise ValueError("source preview prepare request has an invalid episode index")
    generator_git_commit = value.get("generator_git_commit")
    if not isinstance(generator_git_commit, str) or not generator_git_commit:
        raise ValueError("source preview prepare request lacks a generator commit")
    case = ReviewSuiteCase.from_dict(raw_case)
    return prepare_source_review_declaration(
        case,
        episode_index=episode_index,
        generator_git_commit=generator_git_commit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Read one JSON request from stdin and emit one canonical declaration."""

    if argv:
        raise ValueError("source preview prepare worker takes no command arguments")
    value = json.load(sys.stdin)
    if not isinstance(value, Mapping):
        raise ValueError("source preview prepare request must be a mapping")
    declaration = prepare_request(value)
    print(
        json.dumps(
            declaration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())


__all__ = [
    "SOURCE_PREVIEW_PREPARE_REQUEST_SCHEMA",
    "main",
    "prepare_request",
]
