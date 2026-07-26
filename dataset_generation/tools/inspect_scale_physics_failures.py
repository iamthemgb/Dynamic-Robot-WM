#!/usr/bin/env python3
"""Print backend physics-QC failure reasons for selected episodes of a block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--indices", default="")
    args = parser.parse_args()
    root = Path(args.dataset).resolve(strict=True)
    wanted = (
        {int(v) for v in args.indices.split(",") if v}
        if args.indices
        else None
    )
    table = pq.read_table(root / "meta" / "episodes.parquet").to_pylist()
    for row in table:
        if wanted is not None and row["episode_index"] not in wanted:
            continue
        extras = row.get("extras")
        if isinstance(extras, str):
            extras = json.loads(extras)
        physics_qc = (extras or {}).get("physics_qc") or {}
        failures = list(physics_qc.get("failures") or ())
        failed_checks = [
            name
            for name, passed in (physics_qc.get("checks") or {}).items()
            if passed is False
        ]
        failed_task = [
            name
            for name, passed in (physics_qc.get("task_evidence_checks") or {}).items()
            if passed is False
        ]
        if wanted is None and not (failures or failed_checks or failed_task):
            continue
        online = (extras or {}).get("online_outcome_diagnostic") or {}
        print(
            row["episode_index"],
            str(row.get("tool_type"))[:8],
            str(row.get("intended_branch"))[:24],
            "| qc_pass:",
            row.get("physics_qc_pass"),
            "| actual:",
            row.get("actual_outcome"),
            "| online:",
            online.get("actual_outcome"),
            "| flags:",
            [
                flag
                for flag in (row.get("quality_flags") or ())
                if flag
                not in (
                    "review_only_unreleased_backend",
                    "uncalibrated_physics_profile",
                    "human_review_pending",
                )
            ],
            "| failures:",
            failures + failed_checks + failed_task,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
