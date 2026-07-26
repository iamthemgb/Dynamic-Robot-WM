#!/usr/bin/env python3
"""Summarize intended branches, measured outcomes, labels, and failure codes."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from dynamic_robot_dataset.common.episode_writer import load_episode_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    args = parser.parse_args()
    records = load_episode_records(args.dataset)
    branch_outcome = Counter((record.intended_branch, record.actual_outcome) for record in records)
    payload = {
        "episode_count": len(records),
        "task_success": dict(Counter(str(record.task_success).lower() for record in records)),
        "label_status": dict(Counter(record.label_status.value for record in records)),
        "failure_mode": dict(Counter(record.failure_mode for record in records)),
        "intended_vs_actual": [
            {"intended_branch": key[0], "actual_outcome": key[1], "count": count}
            for key, count in sorted(branch_outcome.items())
        ],
        "intended_actual_disagreement_count": sum(
            record.intended_branch.replace("_seeking", "") != record.actual_outcome
            for record in records
        ),
        "invalid_failure_code_count": sum(
            not record.task_success and record.failure_mode.strip().lower() in {"", "none", "null"}
            for record in records
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["invalid_failure_code_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

