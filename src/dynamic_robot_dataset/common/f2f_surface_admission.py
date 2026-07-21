"""Hash-bound admission evidence for the F2f sampled-surface catalog.

The F2f catalog intentionally remains fail-closed while this evidence is
absent.  This module does not mutate catalog candidates or turn admission
booleans on.  It defines the measured report that a later, reviewed catalog
update may bind and independently verifies its geometry, section hashes, and
generator source files.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
from typing import Any, Mapping, Sequence

from .hashing import sha256_file, sha256_json
from ..scenarios.f2f_arbitrary_surface_bounce import (
    ARBITRARY_SURFACE_CATALOG_VERSION,
    ArbitrarySurfaceCandidate,
    raw_surface_candidate_catalog,
)


F2F_SURFACE_ADMISSION_SCHEMA = "dynamic-robot-f2f-surface-admission-evidence/v1"
F2F_SURFACE_ADMISSION_SECTION_SCHEMA = (
    "dynamic-robot-f2f-surface-admission-section/v1"
)
F2F_SURFACE_PROBE_SCHEMA = "dynamic-robot-f2f-surface-calibration-probe/v1"
F2F_POSITIVE_CONTROLLER_SCHEMA = (
    "dynamic-robot-f2f-positive-controller-evidence/v1"
)

ADMISSION_SECTION_NAMES = (
    "grounded_supported",
    "reachability_checked",
    "swept_volume_clearance_checked",
    "background_clearance_checked",
    "calibrated_600_1200",
)
F2F_SCENE_PROFILES = (
    "clean_R0",
    "robocasa_lab",
    "robocasa_kitchen",
    "robocasa_workbench",
    "robocasa_storage",
    "robocasa_tabletop",
)
F2F_EMBODIMENTS = ("franka_hand", "robotiq_2f85_thick_pad")
F2F_CALIBRATION_RATES_HZ = (600, 1200)
F2F_REFERENCE_RATE_HZ = 1200
F2F_CHEAPER_RATE_HZ = 600

MAXIMUM_IK_POSITION_ERROR_M = 0.003
MAXIMUM_REACH_ARRIVAL_DISTANCE_M = 0.025
MAXIMUM_OBJECT_GRIPPER_PENETRATION_M = 0.002
MAXIMUM_OBJECT_TASK_SURFACE_PENETRATION_M = 0.003
MAXIMUM_TIMESTEP_EVENT_SHIFT_S = 1.0 / 30.0
MAXIMUM_TIMESTEP_POSITION_SHIFT_M = 0.01
GROUND_CONTACT_TOLERANCE_M = 1e-6

# Every file that can change candidate construction, actuation, contact
# measurement, or evidence production is recomputed by the verifier.  Paths
# are repository-relative so reports remain relocatable without trusting an
# absolute GPFS checkout location.
F2F_ADMISSION_SOURCE_FILES = (
    "configs/assets/robocasa_catalog_v1.yaml",
    "configs/backends/capabilities_v1.yaml",
    "configs/corpus/dynamic_manipulation_v2.yaml",
    "src/dynamic_robot_dataset/backends/source_mujoco/backend.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/compiler.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/controller.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/model.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/profiles.py",
    "src/dynamic_robot_dataset/backends/source_mujoco/source_spec.py",
    "src/dynamic_robot_dataset/common/f2f_surface_admission.py",
    "src/dynamic_robot_dataset/common/physics_contract.py",
    "src/dynamic_robot_dataset/common/rebound.py",
    "src/dynamic_robot_dataset/common/review_suite.py",
    "src/dynamic_robot_dataset/common/source_scenario.py",
    "src/dynamic_robot_dataset/common/source_evaluators.py",
    "src/dynamic_robot_dataset/scenarios/_rigid_shared.py",
    "src/dynamic_robot_dataset/scenarios/f2f_arbitrary_surface_bounce.py",
    "src/dynamic_robot_dataset/scenarios/registry.py",
    "src/dynamic_robot_dataset/scenarios/types.py",
    "tools/calibrate_f2f_surface_catalog.py",
)


class F2FSurfaceAdmissionError(ValueError):
    """Raised when an admission report is incomplete, stale, or tampered."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def f2f_admission_source_hashes(
    root: str | Path | None = None,
) -> dict[str, str]:
    resolved = repository_root() if root is None else Path(root).resolve()
    result: dict[str, str] = {}
    for relative in F2F_ADMISSION_SOURCE_FILES:
        path = resolved / relative
        if not path.is_file():
            raise F2FSurfaceAdmissionError(
                f"F2f admission source file is missing: {relative}"
            )
        result[relative] = sha256_file(path)
    return result


def candidate_geometry_descriptor(
    candidate: ArbitrarySurfaceCandidate,
    *,
    catalog_index: int | None = None,
) -> dict[str, Any]:
    candidate.validate()
    raw = raw_surface_candidate_catalog()
    if catalog_index is None:
        matches = [
            index
            for index, item in enumerate(raw)
            if item.candidate_id == candidate.candidate_id
        ]
        if len(matches) != 1:
            raise F2FSurfaceAdmissionError(
                f"candidate {candidate.candidate_id!r} is not unique in the raw catalog"
            )
        catalog_index = matches[0]
    if not 0 <= int(catalog_index) < len(raw):
        raise F2FSurfaceAdmissionError("candidate catalog index is out of range")
    return {
        "catalog_version": ARBITRARY_SURFACE_CATALOG_VERSION,
        "catalog_index": int(catalog_index),
        "candidate_id": candidate.candidate_id,
        "task_variant": candidate.task_variant,
        "role": candidate.role,
        "position_m": list(candidate.position_m),
        "half_size_m": list(candidate.half_size_m),
        "euler_rad": list(candidate.euler_rad),
        "normal_world_xyz": list(candidate.normal_world_xyz),
        "contact_profile": candidate.contact_profile,
    }


def candidate_geometry_sha256(candidate: ArbitrarySurfaceCandidate) -> str:
    return sha256_json(candidate_geometry_descriptor(candidate))


def raw_catalog_geometry_identity() -> dict[str, Any]:
    candidates = raw_surface_candidate_catalog()
    rows = [
        candidate_geometry_descriptor(candidate, catalog_index=index)
        for index, candidate in enumerate(candidates)
    ]
    return {
        "catalog_version": ARBITRARY_SURFACE_CATALOG_VERSION,
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "candidate_geometry_sha256": [sha256_json(row) for row in rows],
        "geometry_catalog_sha256": sha256_json(rows),
    }


def expected_probe_id(
    candidate_id: str,
    embodiment: str,
    scene_profile: str,
    simulation_hz: int,
    *,
    purpose: str = "geometry_review",
) -> str:
    return (
        f"{candidate_id}/{purpose}/{embodiment}/{scene_profile}/"
        f"{simulation_hz}hz"
    )


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _completed_probe_is_well_formed(probe: Mapping[str, Any]) -> bool:
    bool_fields = (
        "strict_physics_pass",
        "surface_rebound_pass",
        "semantic_task_success",
        "intended_outcome_match",
        "sustained_opposing_bilateral_contacts",
        "stable_object_to_grasp_transform",
        "retained_through_final_state",
        "ik_all_success",
        "arm_arrived_at_commanded_intercept",
        "task_fixture_anchored",
        "task_fixture_collision_enabled",
        "structural_support_chain_pass",
        "object_swept_volume_clear",
        "structural_support_swept_volume_clear",
        "no_tunneling",
        "background_clearance_pass",
        "all_background_collision_disabled",
        "all_background_anchored",
        "background_fixture_intersection_clear",
    )
    numeric_fields = (
        "maximum_ik_position_error_m",
        "reach_arrival_distance_m",
        "task_fixture_ground_distance_m",
        "maximum_gripper_penetration_m",
        "maximum_task_surface_penetration_m",
        "surface_event_time_s",
    )
    position = probe.get("surface_event_position_m")
    return bool(
        all(isinstance(probe.get(name), bool) for name in bool_fields)
        and all(_is_number(probe.get(name)) for name in numeric_fields)
        and isinstance(probe.get("structural_support_count"), int)
        and not isinstance(probe.get("structural_support_count"), bool)
        and int(probe["structural_support_count"]) >= 0
        and isinstance(probe.get("mutation_boundary_violations"), int)
        and not isinstance(probe.get("mutation_boundary_violations"), bool)
        and int(probe["mutation_boundary_violations"]) >= 0
        and isinstance(probe.get("semantic_outcome"), str)
        and bool(str(probe.get("semantic_outcome")))
        and isinstance(position, Sequence)
        and not isinstance(position, (str, bytes, bytearray))
        and len(position) == 3
        and all(_is_number(value) for value in position)
        and _is_sha256(probe.get("case_sha256"))
        and _is_sha256(probe.get("compiled_scene_xml_sha256"))
        and _is_sha256(probe.get("runtime_source_hashes_sha256"))
    )


def _validate_probe(
    probe: Mapping[str, Any],
    *,
    candidate: ArbitrarySurfaceCandidate,
) -> dict[str, Any]:
    value = deepcopy(dict(probe))
    if value.get("schema_version") != F2F_SURFACE_PROBE_SCHEMA:
        raise F2FSurfaceAdmissionError("F2f calibration probe schema changed")
    embodiment = str(value.get("embodiment") or "")
    scene_profile = str(value.get("scene_profile") or "")
    purpose = str(value.get("probe_purpose") or "")
    branch_role = str(value.get("branch_role") or "")
    intended_outcome = str(value.get("intended_outcome") or "")
    rate = value.get("simulation_hz")
    if embodiment not in F2F_EMBODIMENTS:
        raise F2FSurfaceAdmissionError("F2f calibration probe has an invalid embodiment")
    if scene_profile not in F2F_SCENE_PROFILES:
        raise F2FSurfaceAdmissionError("F2f calibration probe has an invalid scene profile")
    if purpose not in {"geometry_review", "positive_controller"}:
        raise F2FSurfaceAdmissionError("F2f calibration probe has an invalid purpose")
    if not branch_role or intended_outcome not in {"success", "failure"}:
        raise F2FSurfaceAdmissionError("F2f calibration probe lacks its declared branch")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int)
        or rate not in F2F_CALIBRATION_RATES_HZ
    ):
        raise F2FSurfaceAdmissionError("F2f calibration probe has an invalid rate")
    expected_geometry_branch = (
        ("nominal_success", "success")
        if candidate.role == "table"
        else ("deterministic_negative_initial_state", "failure")
    )
    if purpose == "geometry_review" and (
        branch_role,
        intended_outcome,
    ) != expected_geometry_branch:
        raise F2FSurfaceAdmissionError(
            "F2f geometry evidence changed its declared fixed branch"
        )
    if purpose == "positive_controller" and (
        branch_role != "nominal_success"
        or intended_outcome != "success"
        or scene_profile != "clean_R0"
        or rate != F2F_REFERENCE_RATE_HZ
    ):
        raise F2FSurfaceAdmissionError(
            "F2f positive-controller evidence must be a clean 1200 Hz nominal success"
        )
    expected_id = expected_probe_id(
        candidate.candidate_id,
        embodiment,
        scene_profile,
        int(rate),
        purpose=purpose,
    )
    if (
        value.get("probe_id") != expected_id
        or value.get("candidate_id") != candidate.candidate_id
        or value.get("task_variant") != candidate.task_variant
        or value.get("geometry_sha256") != candidate_geometry_sha256(candidate)
    ):
        raise F2FSurfaceAdmissionError(
            f"F2f calibration probe identity changed for {expected_id}"
        )
    source_seed = value.get("source_seed")
    if (
        isinstance(source_seed, bool)
        or not isinstance(source_seed, int)
        or not 0 <= source_seed < 2**64
    ):
        raise F2FSurfaceAdmissionError("F2f calibration source seed is invalid")
    execution = value.get("execution")
    if execution == "completed":
        if not _completed_probe_is_well_formed(value):
            raise F2FSurfaceAdmissionError(
                f"completed F2f calibration probe is malformed: {expected_id}"
            )
    elif execution in {"construction_error", "runtime_error"}:
        if not str(value.get("error_type") or "") or not str(
            value.get("error") or ""
        ):
            raise F2FSurfaceAdmissionError(
                f"failed F2f calibration probe lacks error evidence: {expected_id}"
            )
    else:
        raise F2FSurfaceAdmissionError(
            f"F2f calibration probe has an invalid execution state: {expected_id}"
        )
    return value


def _probe_key(probe: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(probe["probe_purpose"]),
        str(probe["embodiment"]),
        str(probe["scene_profile"]),
        int(probe["simulation_hz"]),
    )


def _expected_geometry_keys() -> set[tuple[str, str, str, int]]:
    return {
        ("geometry_review", embodiment, scene_profile, rate)
        for embodiment in F2F_EMBODIMENTS
        for scene_profile in F2F_SCENE_PROFILES
        for rate in F2F_CALIBRATION_RATES_HZ
    }


def _reference_rows(probes: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        probe
        for probe in probes
        if probe["probe_purpose"] == "geometry_review"
        if int(probe["simulation_hz"]) == F2F_REFERENCE_RATE_HZ
    ]


def _section(
    name: str,
    *,
    passed: bool,
    measurements: Any,
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": F2F_SURFACE_ADMISSION_SECTION_SCHEMA,
        "section": name,
        "passed": bool(passed),
        "measurements": measurements,
        "blockers": sorted(set(str(value) for value in blockers)),
    }


def _derive_sections(
    candidate: ArbitrarySurfaceCandidate,
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_key = {
        _probe_key(probe): probe
        for probe in probes
        if probe["probe_purpose"] == "geometry_review"
    }
    expected = _expected_geometry_keys()
    missing = sorted(expected - set(by_key))
    reference = _reference_rows(probes)
    expected_reference_count = len(F2F_EMBODIMENTS) * len(F2F_SCENE_PROFILES)

    grounded_measurements = []
    grounded_blockers: list[str] = []
    for row in reference:
        completed = row.get("execution") == "completed"
        anchored = completed and row.get("task_fixture_anchored") is True
        collision = completed and row.get("task_fixture_collision_enabled") is True
        if candidate.role == "table":
            supported = bool(
                completed
                and row.get("structural_support_count") == 4
                and row.get("structural_support_chain_pass") is True
            )
        else:
            supported = bool(
                completed
                and row.get("structural_support_count") == 0
                and _is_number(row.get("task_fixture_ground_distance_m"))
                and abs(float(row["task_fixture_ground_distance_m"]))
                <= GROUND_CONTACT_TOLERANCE_M
            )
        row_pass = bool(anchored and collision and supported)
        grounded_measurements.append(
            {
                "probe_id": row["probe_id"],
                "anchored": anchored,
                "collision_enabled": collision,
                "supported_or_grounded": supported,
                "passed": row_pass,
            }
        )
        if not row_pass:
            grounded_blockers.append(f"{row['probe_id']}:fixture_support_failed")
    if len(reference) != expected_reference_count:
        grounded_blockers.append("reference_probe_matrix_incomplete")
    grounded = _section(
        "grounded_supported",
        passed=(
            len(reference) == expected_reference_count
            and all(row["passed"] for row in grounded_measurements)
        ),
        measurements=grounded_measurements,
        blockers=grounded_blockers,
    )

    reach_measurements = []
    reach_blockers: list[str] = []
    for row in reference:
        row_pass = bool(
            row.get("execution") == "completed"
            and row.get("ik_all_success") is True
            and _is_number(row.get("maximum_ik_position_error_m"))
            and float(row["maximum_ik_position_error_m"])
            <= MAXIMUM_IK_POSITION_ERROR_M
            and row.get("arm_arrived_at_commanded_intercept") is True
            and _is_number(row.get("reach_arrival_distance_m"))
            and float(row["reach_arrival_distance_m"])
            <= MAXIMUM_REACH_ARRIVAL_DISTANCE_M
        )
        reach_measurements.append(
            {
                "probe_id": row["probe_id"],
                "maximum_ik_position_error_m": row.get(
                    "maximum_ik_position_error_m"
                ),
                "reach_arrival_distance_m": row.get("reach_arrival_distance_m"),
                "passed": row_pass,
            }
        )
        if not row_pass:
            reach_blockers.append(f"{row['probe_id']}:reachability_failed")
    if len(reference) != expected_reference_count:
        reach_blockers.append("reference_probe_matrix_incomplete")
    reach = _section(
        "reachability_checked",
        passed=(
            len(reference) == expected_reference_count
            and all(row["passed"] for row in reach_measurements)
        ),
        measurements=reach_measurements,
        blockers=reach_blockers,
    )

    sweep_measurements = []
    sweep_blockers: list[str] = []
    for row in reference:
        row_pass = bool(
            row.get("execution") == "completed"
            and row.get("object_swept_volume_clear") is True
            and row.get("structural_support_swept_volume_clear") is True
            and row.get("no_tunneling") is True
            and row.get("mutation_boundary_violations") == 0
        )
        sweep_measurements.append(
            {
                "probe_id": row["probe_id"],
                "object_swept_volume_clear": row.get(
                    "object_swept_volume_clear"
                ),
                "structural_support_swept_volume_clear": row.get(
                    "structural_support_swept_volume_clear"
                ),
                "no_tunneling": row.get("no_tunneling"),
                "mutation_boundary_violations": row.get(
                    "mutation_boundary_violations"
                ),
                "passed": row_pass,
            }
        )
        if not row_pass:
            sweep_blockers.append(f"{row['probe_id']}:swept_volume_failed")
    if len(reference) != expected_reference_count:
        sweep_blockers.append("reference_probe_matrix_incomplete")
    sweep = _section(
        "swept_volume_clearance_checked",
        passed=(
            len(reference) == expected_reference_count
            and all(row["passed"] for row in sweep_measurements)
        ),
        measurements=sweep_measurements,
        blockers=sweep_blockers,
    )

    background_measurements = []
    background_blockers: list[str] = []
    for row in reference:
        row_pass = bool(
            row.get("execution") == "completed"
            and row.get("background_clearance_pass") is True
            and row.get("all_background_collision_disabled") is True
            and row.get("all_background_anchored") is True
            and row.get("background_fixture_intersection_clear") is True
        )
        background_measurements.append(
            {
                "probe_id": row["probe_id"],
                "scene_profile": row["scene_profile"],
                "clearance_pass": row.get("background_clearance_pass"),
                "collision_disabled": row.get(
                    "all_background_collision_disabled"
                ),
                "anchored": row.get("all_background_anchored"),
                "fixture_intersection_clear": row.get(
                    "background_fixture_intersection_clear"
                ),
                "passed": row_pass,
            }
        )
        if not row_pass:
            background_blockers.append(
                f"{row['probe_id']}:background_clearance_failed"
            )
    reference_scene_coverage = {
        (str(row["embodiment"]), str(row["scene_profile"])) for row in reference
    }
    expected_scene_coverage = {
        (embodiment, scene_profile)
        for embodiment in F2F_EMBODIMENTS
        for scene_profile in F2F_SCENE_PROFILES
    }
    if reference_scene_coverage != expected_scene_coverage:
        background_blockers.append("background_profile_matrix_incomplete")
    background = _section(
        "background_clearance_checked",
        passed=(
            reference_scene_coverage == expected_scene_coverage
            and all(row["passed"] for row in background_measurements)
        ),
        measurements=background_measurements,
        blockers=background_blockers,
    )

    comparisons = []
    calibration_blockers: list[str] = []
    reference_admitted = True
    cheaper_admitted = True
    for embodiment in F2F_EMBODIMENTS:
        for scene_profile in F2F_SCENE_PROFILES:
            coarse = by_key.get(
                (
                    "geometry_review",
                    embodiment,
                    scene_profile,
                    F2F_CHEAPER_RATE_HZ,
                )
            )
            fine = by_key.get(
                (
                    "geometry_review",
                    embodiment,
                    scene_profile,
                    F2F_REFERENCE_RATE_HZ,
                )
            )
            comparison_failures: list[str] = []
            if coarse is None or fine is None:
                comparison_failures.append("rate_pair_missing")
                reference_pass = False
                cheaper_pass = False
                event_shift = None
                position_shift = None
            else:
                reference_pass = bool(
                    fine.get("execution") == "completed"
                    and fine.get("strict_physics_pass") is True
                    and fine.get("surface_rebound_pass") is True
                    and fine.get("intended_outcome_match") is True
                    and float(fine.get("maximum_gripper_penetration_m", math.inf))
                    <= MAXIMUM_OBJECT_GRIPPER_PENETRATION_M
                    and float(
                        fine.get("maximum_task_surface_penetration_m", math.inf)
                    )
                    <= MAXIMUM_OBJECT_TASK_SURFACE_PENETRATION_M
                )
                if coarse.get("execution") != "completed":
                    comparison_failures.append("cheaper_profile_execution_failed")
                    event_shift = None
                    position_shift = None
                else:
                    if coarse.get("surface_rebound_pass") is not True:
                        comparison_failures.append(
                            "cheaper_profile_rebound_contract_failed"
                        )
                    if coarse.get("intended_outcome_match") is not True:
                        comparison_failures.append(
                            "cheaper_profile_intended_outcome_mismatch"
                        )
                    if coarse.get("semantic_outcome") != fine.get("semantic_outcome"):
                        comparison_failures.append("outcome_changed_at_1200_hz")
                    if coarse.get("semantic_task_success") != fine.get(
                        "semantic_task_success"
                    ):
                        comparison_failures.append("task_success_changed_at_1200_hz")
                    if coarse.get("strict_physics_pass") is not True:
                        comparison_failures.append("cheaper_profile_strict_physics_failed")
                    if (
                        float(
                            coarse.get(
                                "maximum_gripper_penetration_m", math.inf
                            )
                        )
                        > MAXIMUM_OBJECT_GRIPPER_PENETRATION_M
                    ):
                        comparison_failures.append(
                            "cheaper_profile_gripper_penetration_failed"
                        )
                    if (
                        float(
                            coarse.get(
                                "maximum_task_surface_penetration_m", math.inf
                            )
                        )
                        > MAXIMUM_OBJECT_TASK_SURFACE_PENETRATION_M
                    ):
                        comparison_failures.append(
                            "cheaper_profile_surface_penetration_failed"
                        )
                    try:
                        event_shift = abs(
                            float(coarse["surface_event_time_s"])
                            - float(fine["surface_event_time_s"])
                        )
                        position_shift = math.dist(
                            [float(value) for value in coarse["surface_event_position_m"]],
                            [float(value) for value in fine["surface_event_position_m"]],
                        )
                    except (KeyError, TypeError, ValueError):
                        event_shift = None
                        position_shift = None
                        comparison_failures.append("surface_event_evidence_missing")
                    else:
                        if event_shift > MAXIMUM_TIMESTEP_EVENT_SHIFT_S:
                            comparison_failures.append(
                                "surface_event_shift_exceeds_one_frame"
                            )
                        if position_shift > MAXIMUM_TIMESTEP_POSITION_SHIFT_M:
                            comparison_failures.append(
                                "surface_event_position_shift_exceeds_1cm"
                            )
                cheaper_pass = bool(reference_pass and not comparison_failures)
            reference_admitted = bool(reference_admitted and reference_pass)
            cheaper_admitted = bool(cheaper_admitted and cheaper_pass)
            if not reference_pass:
                calibration_blockers.append(
                    f"{embodiment}/{scene_profile}:reference_profile_failed"
                )
            comparisons.append(
                {
                    "embodiment": embodiment,
                    "scene_profile": scene_profile,
                    "reference_profile_passed": reference_pass,
                    "cheaper_profile_passed": cheaper_pass,
                    "absolute_surface_event_time_shift_s": event_shift,
                    "surface_event_position_shift_m": position_shift,
                    "cheaper_profile_failures": comparison_failures,
                }
            )
    complete_matrix = set(by_key) == expected
    if not complete_matrix:
        calibration_blockers.append(
            f"probe_matrix_incomplete:missing={missing}"
        )
    if not reference_admitted:
        calibration_blockers.append("1200hz_reference_not_admitted")
    selected_rate = (
        F2F_CHEAPER_RATE_HZ
        if complete_matrix and reference_admitted and cheaper_admitted
        else F2F_REFERENCE_RATE_HZ
        if complete_matrix and reference_admitted
        else None
    )
    calibration = _section(
        "calibrated_600_1200",
        passed=bool(complete_matrix and reference_admitted),
        measurements={
            "reference_simulation_hz": F2F_REFERENCE_RATE_HZ,
            "cheaper_simulation_hz": F2F_CHEAPER_RATE_HZ,
            "reference_profile_admitted": bool(
                complete_matrix and reference_admitted
            ),
            "cheaper_profile_admitted": bool(
                complete_matrix and reference_admitted and cheaper_admitted
            ),
            "selected_simulation_hz": selected_rate,
            "comparisons": comparisons,
        },
        blockers=calibration_blockers,
    )
    return {
        "grounded_supported": grounded,
        "reachability_checked": reach,
        "swept_volume_clearance_checked": sweep,
        "background_clearance_checked": background,
        "calibrated_600_1200": calibration,
    }


def _derive_positive_controller(
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the independent retained-positive gate for pilot eligibility.

    Table candidates reuse their clean 1200 Hz nominal geometry probe.  Wall
    candidates use the separately identified ``positive_controller`` probes,
    because their geometry matrix deliberately preserves the fixed negative
    branch.  Geometry admission therefore never implies controller success.
    """

    measurements: list[dict[str, Any]] = []
    blockers: list[str] = []
    for embodiment in F2F_EMBODIMENTS:
        rows = [
            row
            for row in probes
            if row.get("embodiment") == embodiment
            and row.get("scene_profile") == "clean_R0"
            and row.get("simulation_hz") == F2F_REFERENCE_RATE_HZ
            and row.get("branch_role") == "nominal_success"
            and row.get("intended_outcome") == "success"
        ]
        if len(rows) != 1:
            measurements.append(
                {
                    "embodiment": embodiment,
                    "probe_id": None,
                    "passed": False,
                }
            )
            blocker = (
                "positive_controller_probe_missing"
                if not rows
                else "positive_controller_probe_not_unique"
            )
            blockers.append(f"{embodiment}:{blocker}")
            continue
        row = rows[0]
        row_pass = bool(
            row.get("execution") == "completed"
            and row.get("strict_physics_pass") is True
            and row.get("surface_rebound_pass") is True
            and row.get("semantic_task_success") is True
            and row.get("intended_outcome_match") is True
            and row.get("sustained_opposing_bilateral_contacts") is True
            and row.get("stable_object_to_grasp_transform") is True
            and row.get("retained_through_final_state") is True
            and _is_number(row.get("maximum_gripper_penetration_m"))
            and float(row["maximum_gripper_penetration_m"])
            <= MAXIMUM_OBJECT_GRIPPER_PENETRATION_M
            and _is_number(row.get("maximum_task_surface_penetration_m"))
            and float(row["maximum_task_surface_penetration_m"])
            <= MAXIMUM_OBJECT_TASK_SURFACE_PENETRATION_M
            and row.get("mutation_boundary_violations") == 0
        )
        measurements.append(
            {
                "embodiment": embodiment,
                "probe_id": row["probe_id"],
                "strict_physics_pass": row.get("strict_physics_pass"),
                "surface_rebound_pass": row.get("surface_rebound_pass"),
                "semantic_task_success": row.get("semantic_task_success"),
                "intended_outcome_match": row.get("intended_outcome_match"),
                "sustained_opposing_bilateral_contacts": row.get(
                    "sustained_opposing_bilateral_contacts"
                ),
                "stable_object_to_grasp_transform": row.get(
                    "stable_object_to_grasp_transform"
                ),
                "retained_through_final_state": row.get(
                    "retained_through_final_state"
                ),
                "passed": row_pass,
            }
        )
        if not row_pass:
            blockers.append(f"{row['probe_id']}:retained_positive_failed")
    admitted = bool(
        len(measurements) == len(F2F_EMBODIMENTS)
        and all(row["passed"] for row in measurements)
    )
    return {
        "schema_version": F2F_POSITIVE_CONTROLLER_SCHEMA,
        "required_embodiments": list(F2F_EMBODIMENTS),
        "reference_simulation_hz": F2F_REFERENCE_RATE_HZ,
        "admitted": admitted,
        "measurements": measurements,
        "blockers": sorted(set(blockers)),
    }


def build_f2f_surface_admission_evidence(
    probes_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    generated_at: str,
    root: str | Path | None = None,
) -> dict[str, Any]:
    if not generated_at.strip():
        raise F2FSurfaceAdmissionError("F2f evidence requires a generation timestamp")
    catalog = raw_surface_candidate_catalog()
    by_id = {candidate.candidate_id: candidate for candidate in catalog}
    selected_ids = [
        candidate.candidate_id
        for candidate in catalog
        if candidate.candidate_id in probes_by_candidate
    ]
    if set(selected_ids) != set(probes_by_candidate):
        unknown = sorted(set(probes_by_candidate) - set(by_id))
        raise F2FSurfaceAdmissionError(
            f"F2f evidence contains unknown candidates: {unknown}"
        )
    source_files = f2f_admission_source_hashes(root)
    entries = []
    for candidate_id in selected_ids:
        candidate = by_id[candidate_id]
        probes = [
            _validate_probe(probe, candidate=candidate)
            for probe in probes_by_candidate[candidate_id]
        ]
        keys = [_probe_key(probe) for probe in probes]
        if len(keys) != len(set(keys)):
            raise F2FSurfaceAdmissionError(
                f"F2f evidence repeats probes for {candidate_id}"
            )
        probes.sort(key=lambda row: _probe_key(row))
        sections = _derive_sections(candidate, probes)
        section_hashes = {
            name: sha256_json(sections[name]) for name in ADMISSION_SECTION_NAMES
        }
        positive_controller = _derive_positive_controller(probes)
        positive_controller_sha256 = sha256_json(positive_controller)
        admission_ready = all(
            sections[name]["passed"] is True for name in ADMISSION_SECTION_NAMES
        )
        positive_controller_admitted = positive_controller["admitted"] is True
        entry: dict[str, Any] = {
            "candidate_id": candidate_id,
            "geometry": candidate_geometry_descriptor(candidate),
            "geometry_sha256": candidate_geometry_sha256(candidate),
            "probes": probes,
            "sections": sections,
            "section_sha256": section_hashes,
            "positive_controller": positive_controller,
            "positive_controller_sha256": positive_controller_sha256,
            # ``admission_ready`` is deliberately geometry-only.  A negative
            # fixed branch can prove that a candidate is physically admitted,
            # but cannot qualify it for pilot/production.
            "admission_ready": admission_ready,
            "positive_controller_admitted": positive_controller_admitted,
            "pilot_ready": bool(
                admission_ready and positive_controller_admitted
            ),
        }
        entry["candidate_evidence_sha256"] = sha256_json(entry)
        entries.append(entry)
    result: dict[str, Any] = {
        "schema_version": F2F_SURFACE_ADMISSION_SCHEMA,
        "generated_at": generated_at,
        "immutable": True,
        "training_eligible": False,
        "catalog_geometry": raw_catalog_geometry_identity(),
        "selected_candidate_ids": selected_ids,
        "source_files": source_files,
        "source_files_sha256": sha256_json(source_files),
        "candidates": entries,
    }
    result["report_sha256"] = sha256_json(result)
    return result


def verify_f2f_surface_admission_evidence(
    value: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    report = deepcopy(dict(value))
    stored_report_hash = report.pop("report_sha256", None)
    if not _is_sha256(stored_report_hash) or stored_report_hash != sha256_json(report):
        raise F2FSurfaceAdmissionError("F2f admission report hash changed")
    if (
        report.get("schema_version") != F2F_SURFACE_ADMISSION_SCHEMA
        or report.get("immutable") is not True
        or report.get("training_eligible") is not False
    ):
        raise F2FSurfaceAdmissionError("F2f admission report envelope changed")
    expected_catalog = raw_catalog_geometry_identity()
    if report.get("catalog_geometry") != expected_catalog:
        raise F2FSurfaceAdmissionError("F2f raw catalog geometry fingerprint changed")
    actual_source_files = f2f_admission_source_hashes(root)
    if report.get("source_files") != actual_source_files:
        raise F2FSurfaceAdmissionError("F2f admission source-file hashes changed")
    if report.get("source_files_sha256") != sha256_json(actual_source_files):
        raise F2FSurfaceAdmissionError("F2f admission source-file manifest hash changed")
    selected = report.get("selected_candidate_ids")
    entries = report.get("candidates")
    if (
        not isinstance(selected, Sequence)
        or isinstance(selected, (str, bytes, bytearray))
        or not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes, bytearray))
    ):
        raise F2FSurfaceAdmissionError("F2f admission candidate table is malformed")
    selected_ids = [str(value) for value in selected]
    raw_order = list(expected_catalog["candidate_ids"])
    if selected_ids != [value for value in raw_order if value in selected_ids]:
        raise F2FSurfaceAdmissionError("F2f evidence changed raw candidate ordering")
    if len(selected_ids) != len(set(selected_ids)) or len(entries) != len(selected_ids):
        raise F2FSurfaceAdmissionError("F2f evidence candidate identities are not unique")
    candidates = {
        item.candidate_id: item for item in raw_surface_candidate_catalog()
    }
    ready: list[str] = []
    positive_controller_ready: list[str] = []
    pilot_ready: list[str] = []
    blocked: dict[str, list[str]] = {}
    positive_controller_blocked: dict[str, list[str]] = {}
    verified_section_hashes: dict[str, dict[str, str]] = {}
    for candidate_id, raw_entry in zip(selected_ids, entries):
        if candidate_id not in candidates or not isinstance(raw_entry, Mapping):
            raise F2FSurfaceAdmissionError(
                f"F2f evidence contains an unknown candidate: {candidate_id}"
            )
        candidate = candidates[candidate_id]
        entry = deepcopy(dict(raw_entry))
        stored_candidate_hash = entry.pop("candidate_evidence_sha256", None)
        if not _is_sha256(stored_candidate_hash) or stored_candidate_hash != sha256_json(entry):
            raise F2FSurfaceAdmissionError(
                f"F2f candidate evidence hash changed for {candidate_id}"
            )
        if entry.get("candidate_id") != candidate_id:
            raise F2FSurfaceAdmissionError("F2f candidate evidence order changed")
        geometry = candidate_geometry_descriptor(candidate)
        if (
            entry.get("geometry") != geometry
            or entry.get("geometry_sha256") != sha256_json(geometry)
        ):
            raise F2FSurfaceAdmissionError(
                f"F2f geometry fingerprint changed for {candidate_id}"
            )
        raw_probes = entry.get("probes")
        if not isinstance(raw_probes, Sequence) or isinstance(
            raw_probes, (str, bytes, bytearray)
        ):
            raise F2FSurfaceAdmissionError(
                f"F2f probes are malformed for {candidate_id}"
            )
        probes = [_validate_probe(probe, candidate=candidate) for probe in raw_probes]
        keys = [_probe_key(probe) for probe in probes]
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise F2FSurfaceAdmissionError(
                f"F2f probes changed deterministic order for {candidate_id}"
            )
        sections = _derive_sections(candidate, probes)
        section_hashes = {
            name: sha256_json(sections[name]) for name in ADMISSION_SECTION_NAMES
        }
        if entry.get("sections") != sections:
            raise F2FSurfaceAdmissionError(
                f"F2f derived admission sections changed for {candidate_id}"
            )
        if entry.get("section_sha256") != section_hashes:
            raise F2FSurfaceAdmissionError(
                f"F2f admission section hashes changed for {candidate_id}"
            )
        admission_ready = all(
            sections[name]["passed"] is True for name in ADMISSION_SECTION_NAMES
        )
        if entry.get("admission_ready") is not admission_ready:
            raise F2FSurfaceAdmissionError(
                f"F2f admission verdict changed for {candidate_id}"
            )
        positive_controller = _derive_positive_controller(probes)
        positive_controller_sha256 = sha256_json(positive_controller)
        if entry.get("positive_controller") != positive_controller:
            raise F2FSurfaceAdmissionError(
                f"F2f positive-controller evidence changed for {candidate_id}"
            )
        if entry.get("positive_controller_sha256") != positive_controller_sha256:
            raise F2FSurfaceAdmissionError(
                f"F2f positive-controller hash changed for {candidate_id}"
            )
        positive_admitted = positive_controller["admitted"] is True
        if entry.get("positive_controller_admitted") is not positive_admitted:
            raise F2FSurfaceAdmissionError(
                f"F2f positive-controller verdict changed for {candidate_id}"
            )
        candidate_pilot_ready = bool(admission_ready and positive_admitted)
        if entry.get("pilot_ready") is not candidate_pilot_ready:
            raise F2FSurfaceAdmissionError(
                f"F2f pilot verdict changed for {candidate_id}"
            )
        verified_section_hashes[candidate_id] = section_hashes
        if admission_ready:
            ready.append(candidate_id)
        else:
            blocked[candidate_id] = sorted(
                {
                    blocker
                    for section in sections.values()
                    for blocker in section["blockers"]
                }
            )
        if positive_admitted:
            positive_controller_ready.append(candidate_id)
        else:
            positive_controller_blocked[candidate_id] = list(
                positive_controller["blockers"]
            )
        if candidate_pilot_ready:
            pilot_ready.append(candidate_id)
    return {
        "verified": True,
        "report_sha256": stored_report_hash,
        "admission_ready_candidates": ready,
        "blocked_candidates": blocked,
        "positive_controller_admitted_candidates": positive_controller_ready,
        "positive_controller_blocked_candidates": positive_controller_blocked,
        "pilot_ready_candidates": pilot_ready,
        "verified_section_sha256": verified_section_hashes,
    }


__all__ = [
    "ADMISSION_SECTION_NAMES",
    "F2F_ADMISSION_SOURCE_FILES",
    "F2F_CALIBRATION_RATES_HZ",
    "F2F_CHEAPER_RATE_HZ",
    "F2F_EMBODIMENTS",
    "F2F_REFERENCE_RATE_HZ",
    "F2F_SCENE_PROFILES",
    "F2F_POSITIVE_CONTROLLER_SCHEMA",
    "F2F_SURFACE_ADMISSION_SCHEMA",
    "F2F_SURFACE_PROBE_SCHEMA",
    "F2FSurfaceAdmissionError",
    "build_f2f_surface_admission_evidence",
    "candidate_geometry_descriptor",
    "candidate_geometry_sha256",
    "expected_probe_id",
    "f2f_admission_source_hashes",
    "raw_catalog_geometry_identity",
    "repository_root",
    "verify_f2f_surface_admission_evidence",
]
