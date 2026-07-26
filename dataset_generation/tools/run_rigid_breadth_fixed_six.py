#!/usr/bin/env python3
"""Run the immutable rigid-breadth fixed cases without rendering.

This is a diagnostic verdict generator, not a retry loop and not a dataset
writer.  It always keeps the review-suite seeds, records construction/runtime
failures, replays the registered persisted evaluator when the scenario is
admitted, and emits a content-hashed JSON report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from dynamic_robot_dataset.backends.source_mujoco import SourceMujocoBackend
from dynamic_robot_dataset.backends.source_mujoco.source_spec import prepare_review_case
from dynamic_robot_dataset.common.hashing import sha256_file, sha256_json
from dynamic_robot_dataset.common.paths import atomic_write_json
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.common.source_evaluators import evaluate_source_rows


SCHEMA_VERSION = "dynamic-robot-rigid-breadth-fixed-six-report/v1"
DEFAULT_LEAVES = ("F2b", "F2d", "F2e", "F2f")
SOURCE_FILES = (
    "configs/backends/capabilities_v1.yaml",
    "configs/corpus/dynamic_manipulation_v2.yaml",
    "src/dynamic_robot_dataset/backends/source_mujoco/backend.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/compiler.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/model.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/profiles.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/rigid_breadth.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/source_spec.py",
    "src/dynamic_robot_dataset/common/source_scenario.py",
    "src/dynamic_robot_dataset/common/source_evaluators.py",
    "src/dynamic_robot_dataset/scenarios/_rigid_shared.py",
    "src/dynamic_robot_dataset/scenarios/registry.py",
    "src/dynamic_robot_dataset/scenarios/types.py",
    "src/dynamic_robot_dataset/scenarios/f2b_ramp_launch.py",
    "src/dynamic_robot_dataset/scenarios/f2d_wall_barrier_rebound.py",
    "src/dynamic_robot_dataset/scenarios/f2e_multi_surface_rebound.py",
    "src/dynamic_robot_dataset/scenarios/f2f_arbitrary_surface_bounce.py",
    "tools/run_rigid_breadth_fixed_six.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _false_checks(physics_qc: dict[str, Any]) -> list[str]:
    checks = physics_qc.get("checks")
    if not isinstance(checks, dict):
        return ["checks_missing"]
    return sorted(name for name, passed in checks.items() if passed is False)


def _case_result(backend: SourceMujocoBackend, case: Any) -> dict[str, Any]:
    base = {
        "case_id": case.case_id,
        "case_sha256": case.case_sha256,
        "rollout_index": case.rollout_index,
        "embodiment": case.embodiment,
        "task_variant": case.task_variant,
        "branch_role": case.branch_role,
        "intended_outcome": case.intended_outcome,
        "rng_subseeds": asdict(case.rng_subseeds),
        "rendered": False,
        "seed_replaced": False,
    }
    try:
        result = backend.run(case, render=False)
    except Exception as error:  # diagnostic evidence must preserve construction errors
        return {
            **base,
            "execution": "construction_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "automated_qc_passed": False,
            "canonical_replay": {
                "status": "not_run",
                "matches_online": False,
            },
        }
    qc = dict(result.physics_qc)
    penetration = dict(qc.get("penetration") or {})
    metrics = dict(penetration.get("metrics") or {})
    evidence = dict(qc.get("task_evidence") or {})
    replay: dict[str, Any]
    try:
        spec = prepare_review_case(case, backend=backend)
        measured = evaluate_source_rows(
            evaluator_id=case.evaluator,
            corpus_leaf_id=case.corpus_leaf_id,
            task_variant=case.task_variant,
            source_spec=spec.to_dict(),
            state_rows=result.high_rate_rows,
            event_rows=result.contact_rows,
        )
        replay = {
            "status": "completed",
            "task_success": measured.task_success,
            "actual_outcome_class": measured.actual_outcome_class.value,
            "primary_failure_code": measured.primary_failure_code,
            "evidence_sha256": sha256_json(measured.evidence),
            "matches_online": bool(
                measured.task_success == result.outcome.get("task_success")
                and measured.actual_outcome_class.value
                == result.outcome.get("actual_outcome")
            ),
        }
    except Exception as error:
        replay = {
            "status": "rejected",
            "error_type": type(error).__name__,
            "error": str(error),
            "matches_online": False,
        }
    return {
        **base,
        "execution": "completed",
        "simulation_hz": result.scenario.simulation_hz,
        "online_outcome": dict(result.outcome),
        "automated_qc_passed": qc.get("physics_qc_pass") is True,
        "false_common_checks": _false_checks(qc),
        "task_evidence_failures": list(qc.get("task_evidence_failures") or ()),
        "penetration_failures": list(penetration.get("failures") or ()),
        "maximum_gripper_or_arm_penetration_m": metrics.get(
            "maximum_gripper_penetration_m"
        ),
        "maximum_task_surface_penetration_m": metrics.get(
            "maximum_task_surface_penetration_m"
        ),
        "maximum_arm_joint_acceleration_rad_s2": qc.get(
            "maximum_arm_joint_acceleration_rad_s2"
        ),
        "maximum_passive_finger_acceleration_rad_s2": qc.get(
            "maximum_passive_finger_acceleration_rad_s2"
        ),
        "contract_evidence": {
            name: evidence.get(name)
            for name in (
                "surface_transition_pass",
                "ordered_contact_sequence_pass",
                "sampled_surface_rebound_pass",
                "rolling_or_sliding_slip_within_limit",
                "friction_deceleration_consistent",
            )
            if name in evidence
        },
        "mutation_boundary_violations": result.runtime_audit.get(
            "mutation_boundary_violations"
        ),
        "canonical_replay": replay,
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
        per_leaf[leaf] = {
            "case_count": len(selected),
            "automated_qc_pass_count": sum(
                item["automated_qc_passed"] is True for item in selected
            ),
            "all_six_automated_qc_passed": bool(
                len(selected) == 6
                and all(item["automated_qc_passed"] is True for item in selected)
            ),
            "all_canonical_replays_completed_and_matched": bool(
                len(selected) == 6
                and all(
                    item["canonical_replay"].get("status") == "completed"
                    and item["canonical_replay"].get("matches_online") is True
                    for item in selected
                )
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
    parser.add_argument("--leaves", nargs="+", choices=DEFAULT_LEAVES, default=DEFAULT_LEAVES)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = build_report(tuple(arguments.leaves))
    if arguments.output is not None:
        atomic_write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(
        item["all_six_automated_qc_passed"]
        and item["all_canonical_replays_completed_and_matched"]
        for item in report["per_leaf"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
