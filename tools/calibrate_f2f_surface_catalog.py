#!/usr/bin/env python3
"""Generate or verify immutable F2f sampled-surface admission evidence.

Generation is diagnostic and training-ineligible.  It runs each selected raw
catalog candidate for both embodiments, every fixed R0/R1 background, and the
600/1200 Hz pair.  It never retries for an intended outcome and never changes
the catalog admission flags.  A construction or physics failure is retained
as evidence that blocks that candidate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import multiprocessing
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping, Sequence

from dynamic_robot_dataset.backends.source_mujoco import SourceMujocoBackend
from dynamic_robot_dataset.common.f2f_surface_admission import (
    F2F_CALIBRATION_RATES_HZ,
    F2F_EMBODIMENTS,
    F2F_SCENE_PROFILES,
    F2F_SURFACE_PROBE_SCHEMA,
    build_f2f_surface_admission_evidence,
    candidate_geometry_sha256,
    expected_probe_id,
    f2f_admission_source_hashes,
    repository_root,
    verify_f2f_surface_admission_evidence,
)
from dynamic_robot_dataset.common.hashing import sha256_json
from dynamic_robot_dataset.common.paths import atomic_write_json
from dynamic_robot_dataset.common.review_suite import build_review_suite_plan
from dynamic_robot_dataset.scenarios.f2f_arbitrary_surface_bounce import (
    ArbitrarySurfaceCandidate,
    admitted_surface_catalog,
    sample_surface_candidate,
)


_PROBE_UUID_NAMESPACE = uuid.UUID("9ae1cf88-755c-4aaa-93f9-74d9fd6676db")
_COMMON_CHECK_NAMES = (
    "finite_state",
    "no_solver_warnings",
    "no_tunneling",
    "no_unexplained_velocity_discontinuity",
    "no_mutation_boundary_violation",
    "no_applied_forces",
    "no_object_linked_equality_or_latch_assistance",
    "actuator_forces_within_model_limits",
    "joint_motion_within_model_limits",
    "arm_command_travel_present",
    "arm_arrived_at_commanded_intercept",
)
_PROCESS_LOCAL_BACKEND: SourceMujocoBackend | None = None


def _candidate_seed(candidate: ArbitrarySurfaceCandidate) -> int:
    # This is candidate enumeration for calibration, not outcome-conditioned
    # retry.  The first deterministic PCG64 seed selecting the raw candidate
    # is recorded in every probe.
    for seed in range(1_000_000):
        selected = sample_surface_candidate(
            candidate.task_variant, source_seed=seed
        )
        if selected.candidate_id == candidate.candidate_id:
            return seed
    raise RuntimeError(
        f"could not resolve a deterministic calibration seed for {candidate.candidate_id}"
    )


def _base_case(task_variant: str, embodiment: str):
    return next(
        case
        for case in build_review_suite_plan().cases
        if case.corpus_leaf_id == "F2f"
        and case.task_variant == task_variant
        and case.embodiment == embodiment
    )


def _probe_case(
    candidate: ArbitrarySurfaceCandidate,
    *,
    embodiment: str,
    scene_profile: str,
    source_seed: int,
    probe_purpose: str,
    branch_role: str,
    intended_outcome: str,
):
    base = _base_case(candidate.task_variant, embodiment)
    probe_id = expected_probe_id(
        candidate.candidate_id,
        embodiment,
        scene_profile,
        F2F_CALIBRATION_RATES_HZ[0],
        purpose=probe_purpose,
    ).rsplit("/", 1)[0]
    return replace(
        base,
        case_id=(
            f"F2f-admission-{candidate.candidate_id}-{probe_purpose}-"
            f"{embodiment}-{scene_profile}"
        ),
        episode_uuid=str(uuid.uuid5(_PROBE_UUID_NAMESPACE, probe_id)),
        scene_profile=scene_profile,
        randomization_level="R0" if scene_profile == "clean_R0" else "R1",
        requires_real_robocasa=scene_profile != "clean_R0",
        branch_role=branch_role,
        intended_outcome=intended_outcome,
        rng_subseeds=replace(base.rng_subseeds, physics=source_seed),
        counterfactual_bundle_id=(
            f"f2f-admission-{candidate.candidate_id}-{probe_purpose}-"
            f"{scene_profile}"
        ),
        counterfactual_branch_id=f"{branch_role}-{embodiment}",
        counterfactual_sibling_index=0,
    )


def _rotation_xyz(euler: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    roll, pitch, yaw = (float(value) for value in euler)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _point_on_candidate_face(
    point: Sequence[float], candidate: ArbitrarySurfaceCandidate
) -> bool:
    rotation = _rotation_xyz(candidate.euler_rad)
    delta = tuple(
        float(point[index]) - candidate.position_m[index] for index in range(3)
    )
    local = tuple(
        sum(
            rotation[world_axis][local_axis] * delta[world_axis]
            for world_axis in range(3)
        )
        for local_axis in range(3)
    )
    half = candidate.half_size_m
    tolerance = 0.006
    if candidate.role == "table":
        return bool(
            abs(local[0]) <= half[0] + tolerance
            and abs(local[1]) <= half[1] + tolerance
            and abs(local[2] - half[2]) <= tolerance
        )
    return bool(
        abs(local[0]) <= half[0] + tolerance
        and abs(local[2]) <= half[2] + tolerance
        and abs(local[1] + half[1]) <= tolerance
    )


def _surface_rebound_evidence(
    result: Any, candidate: ArbitrarySurfaceCandidate
) -> tuple[bool, float, list[float]]:
    stable_id = f"owned_arbitrary_surface__{candidate.candidate_id}"
    planned_intercept_time_s = float(result.scenario.ballistic_event_time_s)
    contacts = [
        row
        for row in result.contact_rows
        if row.get("contact_category") == "task_surface"
        # Match the canonical saved-artifact evaluator: contacts after the
        # planned interception are downstream consequences, not part of the
        # sampled rebound identity.  In particular, a correctly rebounded
        # negative barrier rollout may land on the room floor after missing
        # the hand.  Treating that later floor contact as a second sampled
        # surface incorrectly rejected otherwise valid barrier geometry.
        and float(row.get("timestamp", math.inf))
        <= planned_intercept_time_s + 1e-12
    ]
    matching = [
        row for row in contacts if row.get("task_surface_id") == stable_id
    ]
    identity_matches = bool(
        matching
        and {str(row.get("task_surface_id") or "") for row in contacts}
        == {stable_id}
    )
    expected_normal = candidate.normal_world_xyz
    normal_matches = bool(
        matching
        and all(
            isinstance(row.get("normal_world"), Sequence)
            and len(row["normal_world"]) == 3
            and sum(
                float(left) * float(right)
                for left, right in zip(row["normal_world"], expected_normal)
            )
            >= 0.98
            for row in matching
        )
    )
    points_match = bool(
        matching
        and all(
            isinstance(row.get("point_world_m"), Sequence)
            and len(row["point_world_m"]) == 3
            and _point_on_candidate_face(row["point_world_m"], candidate)
            for row in matching
        )
    )
    rebound = dict(result.physics_qc.get("restitution") or {})
    event_time = float(rebound.get("event_time_s", math.nan))
    event_position = rebound.get("event_position_m")
    if (
        not math.isfinite(event_time)
        or not isinstance(event_position, Sequence)
        or isinstance(event_position, (str, bytes, bytearray))
        or len(event_position) != 3
        or any(not math.isfinite(float(value)) for value in event_position)
    ):
        return False, 1e9, [1e9, 1e9, 1e9]
    return (
        bool(
            identity_matches
            and normal_matches
            and points_match
            and rebound.get("rebound_acceptance_pass") is True
        ),
        event_time,
        [float(value) for value in event_position],
    )


def _fixture_measurements(
    result: Any, candidate: ArbitrarySurfaceCandidate
) -> tuple[Mapping[str, Any], float]:
    stable_id = f"owned_arbitrary_surface__{candidate.candidate_id}"
    rows = list(result.background_clearance.get("fixture_rows") or ())
    task = next(row for row in rows if row.get("fixture_id") == stable_id)
    aabb = task.get("world_aabb")
    minimum = aabb.get("minimum_m") if isinstance(aabb, Mapping) else None
    ground_distance = (
        float(minimum[2])
        if isinstance(minimum, Sequence)
        and not isinstance(minimum, (str, bytes, bytearray))
        and len(minimum) == 3
        else 1e9
    )
    return task, ground_distance


def _completed_probe(
    *,
    case: Any,
    candidate: ArbitrarySurfaceCandidate,
    simulation_hz: int,
    source_seed: int,
    result: Any,
    probe_purpose: str,
) -> dict[str, Any]:
    geometry_hash = candidate_geometry_sha256(candidate)
    probe_id = expected_probe_id(
        candidate.candidate_id,
        case.embodiment,
        case.scene_profile,
        simulation_hz,
        purpose=probe_purpose,
    )
    qc = dict(result.physics_qc)
    checks = dict(qc.get("checks") or {})
    false_common = sorted(
        name for name in _COMMON_CHECK_NAMES if checks.get(name) is not True
    )
    penetration = dict(qc.get("penetration") or {})
    penetration_metrics = dict(penetration.get("metrics") or {})
    maximum_gripper = float(
        penetration_metrics.get("maximum_gripper_penetration_m", 1e9)
    )
    maximum_surface = float(
        penetration_metrics.get("maximum_task_surface_penetration_m", 1e9)
    )
    rebound_pass, event_time, event_position = _surface_rebound_evidence(
        result, candidate
    )
    task_evidence = dict(qc.get("task_evidence") or {})
    retained = bool(
        task_evidence.get("sustained_opposing_bilateral_contacts") is True
        and task_evidence.get("stable_object_to_grasp_transform") is True
        and task_evidence.get("retained_through_final_state") is True
    )
    semantic_success = bool(rebound_pass and retained)
    intended_outcome_match = bool(
        semantic_success
        if case.intended_outcome == "success"
        else not semantic_success
    )
    bilateral_duration = float(
        task_evidence.get("maximum_contiguous_bilateral_contact_s") or 0.0
    )
    semantic_outcome = (
        "success"
        if semantic_success
        else "contact_failure"
        if bilateral_duration > 0.0
        else "miss"
    )
    mutation_violations = int(
        result.runtime_audit.get("mutation_boundary_violations", -1)
    )
    strict_physics = bool(
        not false_common
        and rebound_pass
        and maximum_gripper <= 0.002
        and maximum_surface <= 0.003
        and mutation_violations == 0
        and not penetration.get("failures")
    )
    diagnostics = list(result.backend_provenance.get("ik_diagnostics") or ())
    ik_all_success = bool(
        diagnostics and all(row.get("success") is True for row in diagnostics)
    )
    maximum_ik_error = max(
        [float(row.get("position_error_m", 1e9)) for row in diagnostics],
        default=1e9,
    )
    arrival_distance = checks.get("reach_arrival_distance_m")
    arrival_distance_value = (
        float(arrival_distance)
        if isinstance(arrival_distance, (int, float))
        and not isinstance(arrival_distance, bool)
        and math.isfinite(float(arrival_distance))
        else 1e9
    )
    task_fixture, ground_distance = _fixture_measurements(result, candidate)
    clearance = dict(result.background_clearance)
    return {
        "schema_version": F2F_SURFACE_PROBE_SCHEMA,
        "probe_id": probe_id,
        "candidate_id": candidate.candidate_id,
        "task_variant": candidate.task_variant,
        "geometry_sha256": geometry_hash,
        "probe_purpose": probe_purpose,
        "branch_role": case.branch_role,
        "intended_outcome": case.intended_outcome,
        "embodiment": case.embodiment,
        "scene_profile": case.scene_profile,
        "simulation_hz": simulation_hz,
        "source_seed": source_seed,
        "case_sha256": case.case_sha256,
        "execution": "completed",
        "compiled_scene_xml_sha256": str(
            result.source_hashes["compiled_scene_xml"]
        ),
        "runtime_source_hashes_sha256": sha256_json(dict(result.source_hashes)),
        "strict_physics_pass": strict_physics,
        "false_common_checks": false_common,
        "surface_rebound_pass": rebound_pass,
        "semantic_task_success": semantic_success,
        "intended_outcome_match": intended_outcome_match,
        "sustained_opposing_bilateral_contacts": task_evidence.get(
            "sustained_opposing_bilateral_contacts"
        )
        is True,
        "stable_object_to_grasp_transform": task_evidence.get(
            "stable_object_to_grasp_transform"
        )
        is True,
        "retained_through_final_state": task_evidence.get(
            "retained_through_final_state"
        )
        is True,
        "semantic_outcome": semantic_outcome,
        "maximum_gripper_penetration_m": maximum_gripper,
        "maximum_task_surface_penetration_m": maximum_surface,
        "mutation_boundary_violations": mutation_violations,
        "ik_all_success": ik_all_success,
        "maximum_ik_position_error_m": maximum_ik_error,
        "arm_arrived_at_commanded_intercept": checks.get(
            "arm_arrived_at_commanded_intercept"
        )
        is True,
        "reach_arrival_distance_m": arrival_distance_value,
        "task_fixture_anchored": task_fixture.get("anchored") is True,
        "task_fixture_collision_enabled": task_fixture.get("collision_enabled")
        is True,
        "task_fixture_ground_distance_m": ground_distance,
        "structural_support_count": int(
            clearance.get("structural_support_count", -1)
        ),
        "structural_support_chain_pass": clearance.get(
            "structural_support_chain_pass"
        )
        is True,
        "object_swept_volume_clear": clearance.get("object_swept_volume_clear")
        is True,
        "structural_support_swept_volume_clear": clearance.get(
            "structural_support_swept_volume_clear"
        )
        is True,
        "no_tunneling": checks.get("no_tunneling") is True,
        "background_clearance_pass": clearance.get("clearance_pass") is True,
        "all_background_collision_disabled": clearance.get(
            "all_background_collision_disabled"
        )
        is True,
        "all_background_anchored": clearance.get("all_background_anchored")
        is True,
        "background_fixture_intersection_clear": clearance.get(
            "fixture_intersection_clear"
        )
        is True,
        "surface_event_time_s": event_time,
        "surface_event_position_m": event_position,
    }


def _failed_probe(
    *,
    case: Any,
    candidate: ArbitrarySurfaceCandidate,
    simulation_hz: int,
    source_seed: int,
    error: Exception,
    execution: str,
    probe_purpose: str,
) -> dict[str, Any]:
    return {
        "schema_version": F2F_SURFACE_PROBE_SCHEMA,
        "probe_id": expected_probe_id(
            candidate.candidate_id,
            case.embodiment,
            case.scene_profile,
            simulation_hz,
            purpose=probe_purpose,
        ),
        "candidate_id": candidate.candidate_id,
        "task_variant": candidate.task_variant,
        "geometry_sha256": candidate_geometry_sha256(candidate),
        "probe_purpose": probe_purpose,
        "branch_role": case.branch_role,
        "intended_outcome": case.intended_outcome,
        "embodiment": case.embodiment,
        "scene_profile": case.scene_profile,
        "simulation_hz": simulation_hz,
        "source_seed": source_seed,
        "case_sha256": case.case_sha256,
        "execution": execution,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _run_probe_group(
    task: tuple[
        ArbitrarySurfaceCandidate,
        Any,
        tuple[int, ...],
        int,
        str,
    ],
) -> list[dict[str, Any]]:
    """Run one complete immutable branch in a process-local backend."""

    global _PROCESS_LOCAL_BACKEND
    candidate, case, simulation_rates_hz, source_seed, probe_purpose = task
    if _PROCESS_LOCAL_BACKEND is None:
        _PROCESS_LOCAL_BACKEND = SourceMujocoBackend()
    backend = _PROCESS_LOCAL_BACKEND
    destination: list[dict[str, Any]] = []
    try:
        scenario = backend.compile_case(case)
    except Exception as error:
        for simulation_hz in simulation_rates_hz:
            destination.append(
                _failed_probe(
                    case=case,
                    candidate=candidate,
                    simulation_hz=simulation_hz,
                    source_seed=source_seed,
                    error=error,
                    execution="construction_error",
                    probe_purpose=probe_purpose,
                )
            )
        return destination
    for simulation_hz in simulation_rates_hz:
        try:
            result = backend.run(
                replace(scenario, simulation_hz=simulation_hz),
                render=False,
            )
            probe = _completed_probe(
                case=case,
                candidate=candidate,
                simulation_hz=simulation_hz,
                source_seed=source_seed,
                result=result,
                probe_purpose=probe_purpose,
            )
        except Exception as error:
            probe = _failed_probe(
                case=case,
                candidate=candidate,
                simulation_hz=simulation_hz,
                source_seed=source_seed,
                error=error,
                execution="runtime_error",
                probe_purpose=probe_purpose,
            )
        destination.append(probe)
    return destination


def _probe_group_status(rows: Sequence[Mapping[str, Any]]) -> str:
    executions = {str(row.get("execution") or "") for row in rows}
    if executions == {"construction_error"}:
        return "construction-error"
    if "runtime_error" in executions:
        return "runtime-error"
    if all(
        row.get("execution") == "completed"
        and row.get("strict_physics_pass") is True
        and row.get("surface_rebound_pass") is True
        and row.get("intended_outcome_match") is True
        for row in rows
    ):
        return "measured-pass"
    return "measured-failure"


def generate_evidence(
    *,
    candidate_ids: Sequence[str],
    scene_profiles: Sequence[str],
    root: Path,
    jobs: int = 1,
) -> dict[str, Any]:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    catalog = admitted_surface_catalog()
    selected = [candidate for candidate in catalog if candidate.candidate_id in candidate_ids]
    if {candidate.candidate_id for candidate in selected} != set(candidate_ids):
        raise ValueError("candidate selection contains an unknown F2f catalog ID")
    initial_source_hashes = f2f_admission_source_hashes(root)
    probes: dict[str, list[dict[str, Any]]] = {
        candidate.candidate_id: [] for candidate in selected
    }
    tasks: list[
        tuple[
            ArbitrarySurfaceCandidate,
            Any,
            tuple[int, ...],
            int,
            str,
        ]
    ] = []
    for candidate in selected:
        source_seed = _candidate_seed(candidate)
        geometry_branch_role = (
            "nominal_success"
            if candidate.role == "table"
            else "deterministic_negative_initial_state"
        )
        geometry_intended_outcome = (
            "success" if candidate.role == "table" else "failure"
        )
        for scene_profile in scene_profiles:
            for embodiment in F2F_EMBODIMENTS:
                case = _probe_case(
                    candidate,
                    embodiment=embodiment,
                    scene_profile=scene_profile,
                    source_seed=source_seed,
                    probe_purpose="geometry_review",
                    branch_role=geometry_branch_role,
                    intended_outcome=geometry_intended_outcome,
                )
                tasks.append(
                    (
                        candidate,
                        case,
                        tuple(F2F_CALIBRATION_RATES_HZ),
                        source_seed,
                        "geometry_review",
                    )
                )
        if candidate.role == "wall" and "clean_R0" in scene_profiles:
            # The fixed geometry branch is intentionally negative for wall
            # candidates.  Run a separate, clearly identified nominal branch
            # at the reference rate so geometry success cannot masquerade as
            # dual-embodiment retained-catch evidence.
            for embodiment in F2F_EMBODIMENTS:
                case = _probe_case(
                    candidate,
                    embodiment=embodiment,
                    scene_profile="clean_R0",
                    source_seed=source_seed,
                    probe_purpose="positive_controller",
                    branch_role="nominal_success",
                    intended_outcome="success",
                )
                tasks.append(
                    (
                        candidate,
                        case,
                        (1200,),
                        source_seed,
                        "positive_controller",
                    )
                )
    total = sum(len(task[2]) for task in tasks)
    results_by_index: list[list[dict[str, Any]] | None] = [None] * len(tasks)
    completed = 0

    def record(index: int, rows: list[dict[str, Any]]) -> None:
        nonlocal completed
        results_by_index[index] = rows
        completed += len(rows)
        first = rows[0]
        rates = ",".join(str(row["simulation_hz"]) for row in rows)
        print(
            f"[{completed}/{total}] {first['candidate_id']} "
            f"{first['probe_purpose']} {first['embodiment']} "
            f"{first['scene_profile']} {rates}Hz: {_probe_group_status(rows)}",
            file=sys.stderr,
            flush=True,
        )

    if jobs == 1:
        for index, task in enumerate(tasks):
            record(index, _run_probe_group(task))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_run_probe_group, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                record(futures[future], future.result())

    # Worker completion order is intentionally discarded.  Candidate order
    # follows the raw catalog and each candidate's probes are subsequently
    # sorted by the evidence builder's deterministic identity key.
    for task, rows in zip(tasks, results_by_index):
        if rows is None:
            raise RuntimeError("F2f calibration worker returned no probe group")
        probes[task[0].candidate_id].extend(rows)
    final_source_hashes = f2f_admission_source_hashes(root)
    if final_source_hashes != initial_source_hashes:
        raise RuntimeError(
            "F2f generator sources changed during calibration; evidence was not written"
        )
    return build_f2f_surface_admission_evidence(
        probes,
        generated_at=datetime.now(timezone.utc).isoformat(),
        root=root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Process workers for complete probe groups (default: 1; "
            "use 4 for the full matrix)."
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=[candidate.candidate_id for candidate in admitted_surface_catalog()],
        dest="candidates",
        help="Calibrate one candidate; repeat as needed (default: all raw candidates).",
    )
    parser.add_argument(
        "--scene-profile",
        action="append",
        choices=F2F_SCENE_PROFILES,
        dest="scene_profiles",
        help="Run one background; partial matrices remain verifiable but blocked.",
    )
    arguments = parser.parse_args()
    root = repository_root()
    if arguments.verify is not None:
        if arguments.output is not None:
            parser.error("--verify cannot be combined with --output")
        value = json.loads(arguments.verify.read_text(encoding="utf-8"))
        summary = verify_f2f_surface_admission_evidence(value, root=root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if arguments.output is None:
        parser.error("--output is required when generating evidence")
    candidate_ids = arguments.candidates or [
        candidate.candidate_id for candidate in admitted_surface_catalog()
    ]
    scene_profiles = arguments.scene_profiles or list(F2F_SCENE_PROFILES)
    evidence = generate_evidence(
        candidate_ids=candidate_ids,
        scene_profiles=scene_profiles,
        root=root,
        jobs=arguments.jobs,
    )
    # Verify the exact in-memory bytes before committing them atomically.
    summary = verify_f2f_surface_admission_evidence(evidence, root=root)
    atomic_write_json(arguments.output, evidence)
    print(
        json.dumps(
            {
                **summary,
                "output": str(arguments.output.resolve()),
                "probe_count": sum(
                    len(candidate["probes"])
                    for candidate in evidence["candidates"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
