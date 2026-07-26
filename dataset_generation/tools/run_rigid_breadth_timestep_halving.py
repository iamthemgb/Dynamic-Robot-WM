#!/usr/bin/env python3
"""Compare fixed rigid-breadth rollouts at 600 and 1200 Hz.

This diagnostic never changes a seed, label, registry state, or output dataset.
It records the exact semantic/QC comparison used to decide whether 600 Hz is
admitted or the 1200 Hz reference must remain mandatory for a leaf.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from dynamic_robot_dataset.backends.source_mujoco import SourceMujocoBackend
from dynamic_robot_dataset.backends.source_mujoco.profiles import (
    RIGID_REVIEW_PROFILE,
    timestep_comparison_failures,
)
from dynamic_robot_dataset.common.hashing import sha256_file, sha256_json
from dynamic_robot_dataset.common.paths import atomic_write_json
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan


SCHEMA_VERSION = "dynamic-robot-rigid-breadth-timestep-halving/v1"
DEFAULT_LEAVES = ("F2b", "F2d", "F2e", "F2f")
SOURCE_FILES = (
    "configs/backends/capabilities_v1.yaml",
    "configs/corpus/dynamic_manipulation_v2.yaml",
    "src/dynamic_robot_dataset/backends/source_mujoco/backend.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/compiler.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/model.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/profiles.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/source_spec.py",
    "src/dynamic_robot_dataset/common/source_scenario.py",
    "src/dynamic_robot_dataset/common/source_evaluators.py",
    "src/dynamic_robot_dataset/scenarios/_rigid_shared.py",
    "src/dynamic_robot_dataset/scenarios/f2b_ramp_launch.py",
    "src/dynamic_robot_dataset/scenarios/f2d_wall_barrier_rebound.py",
    "src/dynamic_robot_dataset/scenarios/f2e_multi_surface_rebound.py",
    "src/dynamic_robot_dataset/scenarios/f2f_arbitrary_surface_bounce.py",
    "src/dynamic_robot_dataset/scenarios/types.py",
    "tools/run_rigid_breadth_timestep_halving.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _key_event_position(result: Any, key_event_time_s: float) -> list[float]:
    if not result.high_rate_rows:
        raise RuntimeError("rigid rollout has no high-rate state rows")
    row = min(
        result.high_rate_rows,
        key=lambda item: abs(float(item["timestamp"]) - key_event_time_s),
    )
    position = row.get("object.position")
    if (
        not isinstance(position, list)
        or len(position) != 3
        or any(not math.isfinite(float(value)) for value in position)
    ):
        raise RuntimeError("rigid rollout lacks a finite key-event position")
    return [float(value) for value in position]


def _observation(result: Any) -> dict[str, Any]:
    key_event_time_s = float(result.outcome["key_event_time_s"])
    penetration = result.physics_qc["penetration"]["metrics"]
    return {
        "simulation_hz": result.scenario.simulation_hz,
        "outcome": result.outcome["actual_outcome"],
        "task_success": result.outcome["task_success"],
        "physics_qc_pass": result.physics_qc["physics_qc_pass"],
        "saved_artifact_objective_replay_matches": result.outcome[
            "saved_artifact_objective_replay_matches"
        ],
        "key_event_time_s": key_event_time_s,
        "key_event_position_m": _key_event_position(result, key_event_time_s),
        "maximum_gripper_penetration_m": penetration[
            "maximum_gripper_penetration_m"
        ],
        "maximum_task_surface_penetration_m": penetration[
            "maximum_task_surface_penetration_m"
        ],
        "maximum_arm_joint_acceleration_rad_s2": result.physics_qc[
            "maximum_arm_joint_acceleration_rad_s2"
        ],
        "maximum_passive_finger_acceleration_rad_s2": result.physics_qc[
            "maximum_passive_finger_acceleration_rad_s2"
        ],
        "mutation_boundary_violations": result.runtime_audit[
            "mutation_boundary_violations"
        ],
    }


def _case_result(backend: SourceMujocoBackend, case: Any) -> dict[str, Any]:
    base = {
        "case_id": case.case_id,
        "case_sha256": case.case_sha256,
        "embodiment": case.embodiment,
        "task_variant": case.task_variant,
        "branch_role": case.branch_role,
        "intended_outcome": case.intended_outcome,
        "seed_replaced": False,
    }
    try:
        scenario = backend.compile_case(case)
        observations = []
        for rate in (
            RIGID_REVIEW_PROFILE.simulation_hz,
            RIGID_REVIEW_PROFILE.comparison_simulation_hz,
        ):
            observations.append(
                _observation(
                    backend.run(replace(scenario, simulation_hz=rate), render=False)
                )
            )
    except Exception as error:
        return {
            **base,
            "execution": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "reference_profile_passed": False,
            "cheaper_profile_passed": False,
        }
    coarse, fine = observations
    failures = list(timestep_comparison_failures(coarse, fine))
    reference_passed = bool(
        fine["physics_qc_pass"] is True
        and fine["saved_artifact_objective_replay_matches"] is True
        and fine["mutation_boundary_violations"] == 0
    )
    return {
        **base,
        "execution": "completed",
        "coarse_600_hz": coarse,
        "reference_1200_hz": fine,
        "absolute_key_event_time_shift_s": abs(
            coarse["key_event_time_s"] - fine["key_event_time_s"]
        ),
        "key_event_position_shift_m": math.dist(
            coarse["key_event_position_m"], fine["key_event_position_m"]
        ),
        "comparison_failures": failures,
        "reference_profile_passed": reference_passed,
        "cheaper_profile_passed": bool(reference_passed and not failures),
    }


def build_report(leaves: tuple[str, ...]) -> dict[str, Any]:
    unknown = sorted(set(leaves) - set(DEFAULT_LEAVES))
    if unknown:
        raise ValueError(f"unsupported rigid-breadth leaves: {unknown}")
    plan = build_review_suite_plan()
    cases = tuple(case for case in plan.cases if case.corpus_leaf_id in leaves)
    backend = SourceMujocoBackend()
    results = [_case_result(backend, case) for case in cases]
    per_leaf: dict[str, Any] = {}
    for leaf in leaves:
        selected = [item for item in results if item["case_id"].startswith(f"{leaf}-")]
        reference_admitted = bool(
            len(selected) == 6
            and all(item.get("reference_profile_passed") is True for item in selected)
        )
        cheaper_admitted = bool(
            reference_admitted
            and all(item.get("cheaper_profile_passed") is True for item in selected)
        )
        per_leaf[leaf] = {
            "case_count": len(selected),
            "reference_profile_admitted": reference_admitted,
            "cheaper_profile_admitted": cheaper_admitted,
            "selected_simulation_hz": (
                RIGID_REVIEW_PROFILE.simulation_hz
                if cheaper_admitted
                else RIGID_REVIEW_PROFILE.comparison_simulation_hz
                if reference_admitted
                else None
            ),
        }
    root = _repository_root()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "training_eligible": False,
        "fixed_master_seed": plan.fixed_master_seed,
        "review_plan_sha256": plan.plan_sha256,
        "thresholds": {
            "maximum_event_time_shift_s": 1.0 / 30.0,
            "maximum_key_event_position_shift_m": 0.01,
            "rates_hz": [
                RIGID_REVIEW_PROFILE.simulation_hz,
                RIGID_REVIEW_PROFILE.comparison_simulation_hz,
            ],
        },
        "leaves": list(leaves),
        "source_file_sha256": {
            relative: sha256_file(root / relative) for relative in SOURCE_FILES
        },
        "per_leaf": per_leaf,
        "cases": results,
    }
    payload["report_sha256"] = sha256_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leaves", nargs="+", choices=DEFAULT_LEAVES, default=DEFAULT_LEAVES
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = build_report(tuple(arguments.leaves))
    if arguments.output is not None:
        atomic_write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(
        item["reference_profile_admitted"]
        for item in report["per_leaf"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
