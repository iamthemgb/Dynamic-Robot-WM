from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from dynamic_robot_dataset.common.f2f_surface_admission import (
    ADMISSION_SECTION_NAMES,
    F2F_EMBODIMENTS,
    F2F_SCENE_PROFILES,
    F2F_SURFACE_PROBE_SCHEMA,
    F2FSurfaceAdmissionError,
    build_f2f_surface_admission_evidence,
    candidate_geometry_sha256,
    expected_probe_id,
    f2f_admission_source_hashes,
    raw_catalog_geometry_identity,
    repository_root,
    verify_f2f_surface_admission_evidence,
)
from dynamic_robot_dataset.common.hashing import sha256_file, sha256_json
from dynamic_robot_dataset.scenarios.f2f_arbitrary_surface_bounce import (
    ArbitrarySurfaceCandidate,
    admitted_surface_catalog,
    raw_surface_candidate_catalog,
)
from dynamic_robot_dataset.scenarios import f2f_arbitrary_surface_bounce as f2f_catalog


def _probe(
    candidate: ArbitrarySurfaceCandidate,
    *,
    embodiment: str,
    scene_profile: str,
    simulation_hz: int,
    purpose: str = "geometry_review",
    cheaper_strict_failure: bool = False,
) -> dict[str, object]:
    if purpose == "positive_controller" or candidate.role == "table":
        branch_role = "nominal_success"
        intended_outcome = "success"
        semantic_success = True
    else:
        branch_role = "deterministic_negative_initial_state"
        intended_outcome = "failure"
        semantic_success = False
    strict = not (cheaper_strict_failure and simulation_hz == 600)
    identity = {
        "candidate": candidate.candidate_id,
        "purpose": purpose,
        "embodiment": embodiment,
        "scene_profile": scene_profile,
        "simulation_hz": simulation_hz,
    }
    return {
        "schema_version": F2F_SURFACE_PROBE_SCHEMA,
        "probe_id": expected_probe_id(
            candidate.candidate_id,
            embodiment,
            scene_profile,
            simulation_hz,
            purpose=purpose,
        ),
        "candidate_id": candidate.candidate_id,
        "task_variant": candidate.task_variant,
        "geometry_sha256": candidate_geometry_sha256(candidate),
        "probe_purpose": purpose,
        "branch_role": branch_role,
        "intended_outcome": intended_outcome,
        "embodiment": embodiment,
        "scene_profile": scene_profile,
        "simulation_hz": simulation_hz,
        "source_seed": 17,
        "case_sha256": sha256_json({"case": identity}),
        "execution": "completed",
        "compiled_scene_xml_sha256": sha256_json({"xml": identity}),
        "runtime_source_hashes_sha256": sha256_json({"sources": identity}),
        "strict_physics_pass": strict,
        "surface_rebound_pass": True,
        "semantic_task_success": semantic_success,
        "intended_outcome_match": True,
        "sustained_opposing_bilateral_contacts": semantic_success,
        "stable_object_to_grasp_transform": semantic_success,
        "retained_through_final_state": semantic_success,
        "semantic_outcome": "success" if semantic_success else "miss",
        "maximum_gripper_penetration_m": 0.001,
        "maximum_task_surface_penetration_m": 0.0015,
        "mutation_boundary_violations": 0,
        "ik_all_success": True,
        "maximum_ik_position_error_m": 0.001,
        "arm_arrived_at_commanded_intercept": True,
        "reach_arrival_distance_m": 0.01,
        "task_fixture_anchored": True,
        "task_fixture_collision_enabled": True,
        "task_fixture_ground_distance_m": 0.0,
        "structural_support_count": 4 if candidate.role == "table" else 0,
        "structural_support_chain_pass": candidate.role == "table",
        "object_swept_volume_clear": True,
        "structural_support_swept_volume_clear": True,
        "no_tunneling": True,
        "background_clearance_pass": True,
        "all_background_collision_disabled": True,
        "all_background_anchored": True,
        "background_fixture_intersection_clear": True,
        "surface_event_time_s": 0.8,
        "surface_event_position_m": list(candidate.position_m),
    }


def _geometry_matrix(
    candidate: ArbitrarySurfaceCandidate,
    *,
    cheaper_strict_failure: bool = False,
) -> list[dict[str, object]]:
    return [
        _probe(
            candidate,
            embodiment=embodiment,
            scene_profile=scene_profile,
            simulation_hz=simulation_hz,
            cheaper_strict_failure=cheaper_strict_failure,
        )
        for embodiment in F2F_EMBODIMENTS
        for scene_profile in F2F_SCENE_PROFILES
        for simulation_hz in (600, 1200)
    ]


def _report(candidate: ArbitrarySurfaceCandidate, probes=None):
    return build_f2f_surface_admission_evidence(
        {
            candidate.candidate_id: (
                _geometry_matrix(candidate) if probes is None else probes
            )
        },
        generated_at="2026-07-21T12:00:00+00:00",
    )


def _rehash_candidate(entry: dict[str, object]) -> None:
    entry.pop("candidate_evidence_sha256", None)
    entry["candidate_evidence_sha256"] = sha256_json(entry)


def _rehash_report(report: dict[str, object]) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = sha256_json(report)


def test_table_geometry_and_dual_embodiment_positive_controller_verify() -> None:
    candidate = admitted_surface_catalog()[0]
    report = _report(candidate)

    summary = verify_f2f_surface_admission_evidence(report)
    entry = report["candidates"][0]

    assert entry["admission_ready"] is True
    assert entry["positive_controller_admitted"] is True
    assert entry["pilot_ready"] is True
    assert summary["admission_ready_candidates"] == [candidate.candidate_id]
    assert summary["pilot_ready_candidates"] == [candidate.candidate_id]


def test_wall_negative_matrix_can_admit_geometry_but_not_pilot() -> None:
    candidate = next(
        item for item in admitted_surface_catalog() if item.role == "wall"
    )
    report = _report(candidate)

    summary = verify_f2f_surface_admission_evidence(report)
    entry = report["candidates"][0]

    assert entry["admission_ready"] is True
    assert entry["positive_controller_admitted"] is False
    assert entry["pilot_ready"] is False
    assert summary["admission_ready_candidates"] == [candidate.candidate_id]
    assert summary["pilot_ready_candidates"] == []
    assert candidate.candidate_id in summary[
        "positive_controller_blocked_candidates"
    ]


def test_1200_reference_can_admit_when_600_profile_is_rejected() -> None:
    candidate = admitted_surface_catalog()[0]
    report = _report(
        candidate,
        _geometry_matrix(candidate, cheaper_strict_failure=True),
    )
    entry = report["candidates"][0]
    calibration = entry["sections"]["calibrated_600_1200"]

    verify_f2f_surface_admission_evidence(report)

    assert calibration["passed"] is True
    assert calibration["measurements"]["reference_profile_admitted"] is True
    assert calibration["measurements"]["cheaper_profile_admitted"] is False
    assert calibration["measurements"]["selected_simulation_hz"] == 1200


def test_probe_completion_order_does_not_change_report_or_hash() -> None:
    candidate = admitted_surface_catalog()[0]
    probes = _geometry_matrix(candidate)

    forward = _report(candidate, probes)
    reversed_completion = _report(candidate, list(reversed(probes)))

    assert reversed_completion == forward


def test_verifier_recomputes_section_hashes() -> None:
    report = _report(admitted_surface_catalog()[0])
    tampered = deepcopy(report)
    entry = tampered["candidates"][0]
    entry["section_sha256"][ADMISSION_SECTION_NAMES[0]] = "f" * 64
    _rehash_candidate(entry)
    _rehash_report(tampered)

    with pytest.raises(F2FSurfaceAdmissionError, match="section hashes"):
        verify_f2f_surface_admission_evidence(tampered)


def test_verifier_rederives_sections_instead_of_trusting_bound_hashes() -> None:
    report = _report(admitted_surface_catalog()[0])
    tampered = deepcopy(report)
    entry = tampered["candidates"][0]
    section_name = ADMISSION_SECTION_NAMES[0]
    entry["sections"][section_name]["passed"] = False
    entry["section_sha256"][section_name] = sha256_json(
        entry["sections"][section_name]
    )
    _rehash_candidate(entry)
    _rehash_report(tampered)

    with pytest.raises(F2FSurfaceAdmissionError, match="derived admission sections"):
        verify_f2f_surface_admission_evidence(tampered)


def test_verifier_recomputes_actual_source_file_hashes() -> None:
    report = _report(admitted_surface_catalog()[0])
    relative = next(iter(report["source_files"]))
    assert report["source_files"][relative] == sha256_file(
        repository_root() / relative
    )

    tampered = deepcopy(report)
    tampered["source_files"][relative] = "e" * 64
    tampered["source_files_sha256"] = sha256_json(tampered["source_files"])
    _rehash_report(tampered)

    with pytest.raises(F2FSurfaceAdmissionError, match="source-file hashes"):
        verify_f2f_surface_admission_evidence(tampered)


def test_geometry_fingerprint_binds_transform_not_admission_flags() -> None:
    candidate = admitted_surface_catalog()[0]
    moved = replace(
        candidate,
        position_m=(candidate.position_m[0] + 0.001, *candidate.position_m[1:]),
    )

    assert candidate_geometry_sha256(moved) != candidate_geometry_sha256(candidate)

    report = _report(candidate)
    tampered = deepcopy(report)
    entry = tampered["candidates"][0]
    entry["geometry"]["position_m"][0] += 0.001
    entry["geometry_sha256"] = sha256_json(entry["geometry"])
    _rehash_candidate(entry)
    _rehash_report(tampered)
    with pytest.raises(F2FSurfaceAdmissionError, match="geometry fingerprint"):
        verify_f2f_surface_admission_evidence(tampered)


def test_raw_catalog_identity_preserves_order_and_remains_unadmitted() -> None:
    catalog = raw_surface_candidate_catalog()
    overlay = admitted_surface_catalog()
    identity = raw_catalog_geometry_identity()

    assert identity["candidate_ids"] == [item.candidate_id for item in catalog]
    assert all(item.admission.admitted is False for item in catalog)
    assert [item.candidate_id for item in overlay] == identity["candidate_ids"]
    assert {
        item.candidate_id for item in overlay if item.admission.admitted
    } == {
        "plane_mid_285",
        "plane_table_400",
        "barrier_yaw_neg_15",
        "barrier_yaw_neg_25",
    }
    actual_hashes = f2f_admission_source_hashes()
    assert actual_hashes
    assert all(len(digest) == 64 for digest in actual_hashes.values())


def test_catalog_overlays_only_verified_admission_without_reordering(
    tmp_path, monkeypatch
) -> None:
    candidate = admitted_surface_catalog()[0]
    report = _report(candidate)
    evidence_path = tmp_path / "f2f_surface_admission_v1.json"
    evidence_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        f2f_catalog, "F2F_SURFACE_ADMISSION_EVIDENCE_PATH", evidence_path
    )

    catalog = f2f_catalog.admitted_surface_catalog()

    assert [item.candidate_id for item in catalog] == list(
        raw_catalog_geometry_identity()["candidate_ids"]
    )
    assert catalog[0].admission.admitted is True
    assert set(catalog[0].admission_evidence_sha256) == set(
        ADMISSION_SECTION_NAMES
    )
    assert all(item.admission.admitted is False for item in catalog[1:])


def test_catalog_fails_closed_when_bound_evidence_is_tampered(
    tmp_path, monkeypatch
) -> None:
    report = _report(admitted_surface_catalog()[0])
    report["candidates"][0]["sections"][ADMISSION_SECTION_NAMES[0]][
        "passed"
    ] = False
    evidence_path = tmp_path / "f2f_surface_admission_v1.json"
    evidence_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        f2f_catalog, "F2F_SURFACE_ADMISSION_EVIDENCE_PATH", evidence_path
    )

    assert all(
        item.admission.admitted is False
        for item in f2f_catalog.admitted_surface_catalog()
    )


def test_normal_registry_import_remains_cycle_free_in_fresh_process() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "from dynamic_robot_dataset.scenarios.registry import "
            "load_scenario_definition; "
            "assert load_scenario_definition('F2f').leaf_id == 'F2f'",
        ),
        cwd=repository_root(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
