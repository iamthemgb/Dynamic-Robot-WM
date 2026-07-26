"""Short-lived worker for one immutable scale declaration.

Preparing a RoboCasa-backed declaration compiles provisional and final MuJoCo
models, and some native allocations outlive the Python objects.  Exactly like
the preview prepare worker, this private module accepts one canonical request
on stdin, prints one declaration on stdout, and exits so the operating system
reclaims all native simulator state.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, Sequence

from .scale_execution import prepare_source_scale_declaration
from .scale_suite import ScaleSuiteCase


SCALE_PREPARE_REQUEST_SCHEMA = "dynamic-robot-scale-prepare-request/v1"


def prepare_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and execute exactly one scale declaration-preparation request."""

    if value.get("schema_version") != SCALE_PREPARE_REQUEST_SCHEMA:
        raise ValueError("unsupported scale prepare-request schema")
    raw_case = value.get("scale_case")
    if not isinstance(raw_case, Mapping):
        raise ValueError("scale prepare request lacks a scale case")
    episode_index = value.get("episode_index")
    if (
        not isinstance(episode_index, int)
        or isinstance(episode_index, bool)
        or episode_index < 0
    ):
        raise ValueError("scale prepare request has an invalid episode index")
    generator_git_commit = value.get("generator_git_commit")
    if not isinstance(generator_git_commit, str) or not generator_git_commit:
        raise ValueError("scale prepare request lacks a generator commit")
    case = ScaleSuiteCase.from_dict(raw_case)
    return prepare_source_scale_declaration(
        case,
        episode_index=episode_index,
        generator_git_commit=generator_git_commit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Read one JSON request from stdin and emit one canonical declaration."""

    if argv:
        raise ValueError("scale prepare worker takes no command arguments")
    value = json.load(sys.stdin)
    if not isinstance(value, Mapping):
        raise ValueError("scale prepare request must be a mapping")
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
    "SCALE_PREPARE_REQUEST_SCHEMA",
    "main",
    "prepare_request",
]
