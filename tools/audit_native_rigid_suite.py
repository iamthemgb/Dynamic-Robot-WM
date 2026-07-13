#!/usr/bin/env python3
"""Execute the rigid portion of an acceptance suite without rendering.

This is a development physics/outcome audit, not a finalized acceptance
artifact.  It never writes dataset episodes and cannot satisfy visual,
calibration, deformable, or full-native release gates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from dynamic_robot_dataset.backends.mujoco_native import NativeMuJoCoBackend
from dynamic_robot_dataset.common.native_suite import plan_suite_cases
from dynamic_robot_dataset.common.suites import expand_suite


INTENDED_TO_REQUIRED_ACTUAL = {
    "success": "success",
    "success_seeking": "success",
    "near_miss": "near_miss",
    "contact_failure": "contact_failure",
    "no_op": "no_op",
    "bad_action": "wrong_action",
    "wrong_action": "wrong_action",
}


def audit(config: Path, *, width: int, height: int) -> dict[str, Any]:
    planned = [
        item
        for item in plan_suite_cases(expand_suite(config))
        if item.execution_backend == "native_mujoco"
    ]
    backend = NativeMuJoCoBackend(width=width, height=height)
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []
    maximum_joint_acceleration = 0.0
    for native_index, item in enumerate(planned):
        result = backend.run(item.episode_plan, render=False)
        checks = result.simulation.physics_qc["checks"]
        failed_checks = sorted(
            name
            for name, value in checks.items()
            if isinstance(value, bool) and not value
        )
        maximum_joint_acceleration = max(
            maximum_joint_acceleration,
            float(checks["maximum_joint_acceleration_rad_s2"]),
        )
        outcomes[item.case.family][result.simulation.actual_outcome] += 1
        if failed_checks:
            failures.append(
                {
                    "native_index": native_index,
                    "family": item.case.family,
                    "subfamily": item.case.subfamily,
                    "intended_branch": item.case.branch,
                    "actual_outcome": result.simulation.actual_outcome,
                    "failed_checks": failed_checks,
                }
            )

    required: dict[str, set[str]] = defaultdict(set)
    for item in planned:
        required_class = INTENDED_TO_REQUIRED_ACTUAL.get(item.case.branch)
        if required_class is not None:
            required[item.case.family].add(required_class)
    missing = {
        family: sorted(values - set(outcomes[family]))
        for family, values in required.items()
        if values - set(outcomes[family])
    }
    return {
        "schema_version": "native-rigid-no-render-audit/v1",
        "config": str(config.resolve()),
        "native_branch_count": len(planned),
        "physics_failure_count": len(failures),
        "physics_failures": failures,
        "maximum_joint_acceleration_rad_s2": maximum_joint_acceleration,
        "outcomes_by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(outcomes.items())
        },
        "required_actual_outcomes": {
            family: sorted(values) for family, values in sorted(required.items())
        },
        "missing_required_actual_outcomes": missing,
        "passed": len(planned) == 128 and not failures and not missing,
        "limitations": [
            "rendering_not_evaluated",
            "no_dataset_artifact_written",
            "calibration_admission_not_evaluated",
            "deformable_and_negative_control_cases_not_executed",
            "not_a_full_native_160_case_acceptance_result",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/families/native_acceptance_160.yaml"),
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()
    report = audit(args.config, width=args.width, height=args.height)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
