from __future__ import annotations

import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "configs/physics/rigid_600_1200_calibration_v1.yaml"
CAPABILITIES_PATH = ROOT / "configs/backends/capabilities_v1.yaml"


def _mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_rigid_profile_evidence_is_versioned_source_bound_and_fail_closed() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    capabilities = _mapping(CAPABILITIES_PATH)
    source_mujoco = next(
        backend
        for backend in capabilities["backends"]
        if backend["name"] == "source_mujoco"
    )

    assert evidence["schema_version"] == "dynamic-robot-rigid-profile-calibration/v1"
    assert evidence["contract_version"] == "dynamic-robot-dataset/v2"
    assert evidence["evidence_status"] == "exploratory_unbound"
    assert evidence["release_state"] == "blocked"
    assert evidence["release_eligible"] is False
    assert evidence["admission"]["admitted"] is False
    assert evidence["admission"]["preferred_profile"] is None
    assert evidence["admission"]["cheaper_profile_admitted"] is False
    assert evidence["source_binding"]["evidence_artifacts_hash_bound"] is False
    assert evidence["source_binding"]["source_trees_read_only"] is True
    assert (
        evidence["source_binding"]["external_dependency_manifest_sha256"]
        == source_mujoco["source_hashes"]["external_dependency_manifest"]
    )
    assert all(not profile["admitted"] for profile in evidence["profiles"].values())


def test_rigid_profiles_are_exact_600_and_1200_hz_candidates() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    profiles = evidence["profiles"]
    assert set(profiles) == {
        "rigid_600hz_candidate_v4",
        "rigid_1200hz_reference_v4",
    }

    for name, expected_hz in (
        ("rigid_600hz_candidate_v4", 600),
        ("rigid_1200hz_reference_v4", 1200),
    ):
        profile = profiles[name]
        assert profile["simulation_hz"] == expected_hz
        assert math.isclose(
            profile["timestep_s"], 1.0 / expected_hz, rel_tol=0.0, abs_tol=1e-15
        )
        assert expected_hz % profile["control_hz"] == 0
        assert profile["video_hz"] == 30
        contact = profile["robotiq_thick_pad_contact"]
        assert contact["condim"] == 3
        assert contact["friction"] == [0.9, 0.005, 0.0001]
        assert contact["actuator_force_range_n"] == [-0.16, 0.16]
        assert contact["closure_duration_s"] > 0.0


def test_wall_rebound_measurements_are_physical_without_invented_fields() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    threshold = evidence["acceptance_thresholds"][
        "maximum_measured_effective_restitution"
    ]
    trials = evidence["measurements"]["wall_rebound"]["trials"]
    by_profile = {trial["profile"]: trial for trial in trials}
    low = by_profile["rigid_600hz_candidate_v4"]
    reference = by_profile["rigid_1200hz_reference_v4"]

    assert low["incoming_normal_velocity_m_s"] < 0.0
    assert low["outgoing_normal_velocity_m_s"] > 0.0
    assert math.isclose(low["measured_effective_restitution"], 0.18777696222292306)
    assert math.isclose(reference["measured_effective_restitution"], 0.24216913860953423)
    assert all(
        trial["measured_effective_restitution"] <= threshold for trial in trials
    )
    comparison = evidence["measurements"]["wall_rebound"][
        "timestep_halving_comparison"
    ]
    assert comparison["outcome_match"] is True
    assert comparison["within_one_video_frame"] is True
    assert comparison["within_one_centimeter"] is True


def test_robotiq_halving_evidence_meets_local_thresholds_but_not_admission() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    threshold = evidence["acceptance_thresholds"]
    measurement = evidence["measurements"]["robotiq_free_contact_catch"]
    assert measurement["review_case_id"] == "F1a-review-01"
    assert measurement["fixed_master_seed"] == 20260717
    assert measurement["controller_profile"].endswith("-v4")
    trials = measurement["trials"]
    comparison = measurement["timestep_halving_comparison"]

    measured_event_shift = abs(
        trials[0]["first_bilateral_contact_time_s"]
        - trials[1]["first_bilateral_contact_time_s"]
    )
    assert math.isclose(
        comparison["absolute_event_time_shift_s"],
        measured_event_shift,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert (
        comparison["absolute_event_time_shift_s"]
        <= threshold["timestep_halving_maximum_event_shift_s"]
    )
    assert (
        comparison["key_event_position_shift_m"]
        <= threshold["timestep_halving_maximum_key_position_shift_m"]
    )
    assert (
        max(trial["maximum_object_gripper_penetration_m"] for trial in trials)
        <= threshold["maximum_object_gripper_penetration_m"]
    )
    assert comparison["within_one_video_frame"] is True
    assert comparison["within_one_centimeter"] is True
    assert evidence["admission"]["admitted"] is False
    assert "full_leaf_timestep_halving_matrix_pending" in evidence["admission"][
        "blockers"
    ]
    assert "record_hash_bound_human_review_approval" in evidence[
        "required_before_admission"
    ]


def test_reference_rate_exceptions_preserve_strict_thresholds() -> None:
    evidence = _mapping(EVIDENCE_PATH)
    measurement = evidence["measurements"]["reference_rate_required_cases"]
    assert measurement["policy"].endswith("rigid_1200hz_reference_v4")
    cases = {value["review_case_id"]: value for value in measurement["cases"]}
    assert set(cases) == {
        "P0c-review-01",
        "P0c-review-02",
        "F1a-review-04",
        "F1b-review-04",
    }
    for case_id in ("P0c-review-01", "F1a-review-04", "F1b-review-04"):
        case = cases[case_id]
        assert case["rigid_600hz_observation"] > case["threshold"]
        assert case["rigid_1200hz_observation"] <= case["threshold"]
    assert cases["P0c-review-02"]["rigid_600hz_observation"] is False
    assert cases["P0c-review-02"]["rigid_1200hz_observation"] is True


def test_persisted_artifact_repair_keeps_the_failed_seed() -> None:
    evidence = _mapping(EVIDENCE_PATH)["measurements"]["persisted_artifact_repairs"]
    assert evidence["review_case_id"] == "P0c-review-03"
    assert evidence["fixed_seed_preserved"] is True
    assert evidence["failed_profile"].endswith("-v3")
    assert evidence["repaired_profile"].endswith("-v4")
    assert evidence["failure_evidence"]["main_frozen_transition_fraction"] == 1.0
    repaired = evidence["repaired_evidence"]
    assert repaired["main_frozen_transition_fraction"] == 0.0
    assert repaired["finite_difference_velocity_rmse_m_s"] < repaired[
        "finite_difference_velocity_tolerance_m_s"
    ]
    assert repaired["strict_persisted_artifact_qc_pass"] is True


def test_operator_guide_names_the_canonical_lifecycle_and_scale_gates() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs/unified_generator_operator_guide.md").read_text(
        encoding="utf-8"
    )

    assert "docs/unified_generator_operator_guide.md" in readme
    for command in ("plan-run", "run-shard", "finalize-run", "generate", "review-suite"):
        assert f"dynamic-robot-dataset {command}" in guide
    for gate in ("100-episode pilot", "10-unique-hour gate", "100-unique-hour gate"):
        assert gate in guide
    assert "remain fail-closed historical gate" in guide
