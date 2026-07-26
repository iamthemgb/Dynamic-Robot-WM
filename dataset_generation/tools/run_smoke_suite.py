#!/usr/bin/env python3
"""Generate and validate the exact non-production 120-branch smoke suite."""

from __future__ import annotations

import argparse
import json

from dynamic_robot_dataset.smoke_runner import run_smoke_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shallow-video-checks", action="store_true")
    args = parser.parse_args()
    result = run_smoke_suite(
        args.config,
        args.output,
        seed=args.seed,
        resume=args.resume,
        deep_video_checks=not args.shallow_video_checks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qc_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
