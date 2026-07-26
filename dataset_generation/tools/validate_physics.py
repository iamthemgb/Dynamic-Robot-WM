#!/usr/bin/env python3
"""Run independent physics checks and print an episode-level JSON summary."""

from __future__ import annotations

import argparse
import json

from dynamic_robot_dataset.common.qc import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    args = parser.parse_args()
    report = validate_dataset(args.dataset, deep_video_checks=False)
    rows = []
    for episode in report.episodes:
        physics_failures = [
            value
            for value in episode.hard_failures
            if value.startswith("physics") or "penetration" in value
        ]
        rows.append(
            {
                "episode_uuid": episode.episode_uuid,
                "episode_index": episode.episode_index,
                "release_eligible": episode.release_eligible,
                "passed": not physics_failures,
                "failures": physics_failures,
                "metrics": {
                    key: value
                    for key, value in episode.metrics.items()
                    if key.startswith("physics.") or "penetration" in key
                },
            }
        )
    payload = {
        "dataset_root": report.dataset_root,
        "episode_count": len(rows),
        "failed_count": sum(not row["passed"] for row in rows),
        "failed_release_eligible_count": sum(
            row["release_eligible"] and not row["passed"] for row in rows
        ),
        "episodes": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["failed_release_eligible_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

