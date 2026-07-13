#!/usr/bin/env python3
"""Reject ambiguous Cosmos exports until a versioned target contract is supplied.

The migration plan deliberately defines the canonical dataset independently of
Cosmos, but it does not specify a Cosmos resolution, frame clock, manifest
schema, or conditioning interface.  This command exists so automation fails
clearly instead of silently reusing the Wan contract or inventing one.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.parse_args()
    print(
        "error: no versioned Cosmos export contract is defined in plan.md; "
        "canonical data and the implemented Wan export remain model-independent",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
