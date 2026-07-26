"""Deterministic minting and fail-closed identity of the scale suite."""

import pytest

from dynamic_robot_dataset.common.review_suite import (
    ReviewSuiteValidationError,
    build_review_suite_plan,
)
from dynamic_robot_dataset.common.scale_suite import (
    SCALE_MASTER_SEED,
    ScaleSuiteCase,
    mint_scale_cases,
    scale_block_manifest,
)


def _block(leaf_id: str = "F1a", *, episode_start: int = 0, count: int = 12):
    return mint_scale_cases(
        leaf_id,
        episode_start=episode_start,
        count=count,
        scale_suite_id=f"scale-{leaf_id}-block-0000",
    )


def test_minting_is_deterministic_and_unique() -> None:
    first = _block()
    second = _block()
    assert [case.case_sha256 for case in first] == [
        case.case_sha256 for case in second
    ]
    assert len({case.episode_uuid for case in first}) == len(first)
    assert len({case.case_id for case in first}) == len(first)
    seeds = [case.rng_subseeds.initial_state for case in first]
    assert len(set(seeds)) == len(seeds)


def test_minted_block_is_contiguous_and_honestly_gated() -> None:
    cases = _block(count=8)
    assert [case.episode_index for case in cases] == list(range(8))
    assert [case.scale_index for case in cases] == list(range(8))
    for case in cases:
        assert case.execution_eligible and case.executable_quota == 1
        assert case.initial_state_mode == "sampled_scale"
        assert case.leaf_release_state == "blocked"
        assert case.backend == "source_mujoco"


def test_rotations_cover_embodiments_variants_branches_scenes() -> None:
    cases = _block(count=40)
    assert {case.embodiment for case in cases} == {
        "franka_hand",
        "robotiq_2f85_thick_pad",
    }
    assert len({case.task_variant for case in cases}) == 2
    branches = {case.branch_role for case in cases}
    assert branches == {
        "nominal_success",
        "deterministic_negative_initial_state",
        "deterministic_negative_controller_timing",
    }
    nominal = sum(1 for case in cases if case.branch_role == "nominal_success")
    assert nominal == 26  # 13 of every 20
    scenes = [case.scene_profile for case in cases]
    assert scenes.count("clean_R0") == 2
    # Measured on calibration blocks 9001/9002: the storage scene occludes
    # the F1 catch zone, so F1 leaves rotate without it.
    assert "robocasa_storage" not in scenes
    p0_scenes = {case.scene_profile for case in _block("P0a", count=40)}
    assert "robocasa_storage" in p0_scenes
    assert {case.randomization_level for case in cases} == {"R0", "R1"}
    for case in cases:
        assert case.requires_real_robocasa == (case.scene_profile != "clean_R0")


def test_passive_leaves_stay_passive_with_nominal_profile() -> None:
    cases = _block("P0a", count=6)
    for case in cases:
        assert case.embodiment == "no_robot"
        assert case.branch_role == "passive_observation"
        assert case.intended_outcome == "passive_observation"
        assert case.passive_variation_profile == "nominal"


def test_case_round_trip_and_tamper_detection() -> None:
    case = _block(count=1)[0]
    payload = case.to_dict()
    restored = ScaleSuiteCase.from_dict(payload)
    assert restored == case
    tampered = dict(payload)
    tampered["branch_role"] = "deterministic_negative_controller_timing"
    tampered["intended_outcome"] = "failure"
    with pytest.raises(ReviewSuiteValidationError, match="hash mismatch"):
        ScaleSuiteCase.from_dict(tampered)


def test_scale_identity_is_disjoint_from_fixed_review() -> None:
    review_uuids = {case.episode_uuid for case in build_review_suite_plan().cases}
    scale_uuids = {case.episode_uuid for case in _block(count=20)}
    assert not (review_uuids & scale_uuids)
    assert SCALE_MASTER_SEED != 20260717


def test_block_manifest_binds_cases() -> None:
    cases = _block(count=4)
    manifest = scale_block_manifest(cases)
    assert manifest["episode_count"] == 4
    assert manifest["case_sha256"] == [case.case_sha256 for case in cases]
    assert manifest["training_eligible"] is False
    assert manifest["diagnostic_only"] is True
    with pytest.raises(ReviewSuiteValidationError):
        scale_block_manifest(tuple(reversed(cases)))


def test_f1b_catch_variant_lands_on_the_robotiq() -> None:
    # Calibration block 9000 measured the Panda off-center catch wedging
    # (6.6-8.9 mm gripper penetration in 16/50 nominals); the catch lane
    # moves to the scale-proven Robotiq and the near-miss lane to the Panda.
    cases = _block("F1b", count=20)
    pairs = {(case.task_variant, case.embodiment) for case in cases}
    assert pairs == {
        ("off_center_catch", "robotiq_2f85_thick_pad"),
        ("off_center_near_miss", "franka_hand"),
    }


def test_f2c_scale_minting_samples_the_table_lane_only() -> None:
    # floor_bounce rode the 2 mm gripper-penetration threshold across its
    # sampled lane in calibration block 9000; scale keeps the proven table
    # lane until the floor lane is repaired.
    cases = _block("F2c", count=20)
    assert {case.task_variant for case in cases} == {"table_bounce"}


def test_unsampled_leaves_are_rejected() -> None:
    with pytest.raises(ReviewSuiteValidationError, match="no scale initial-state"):
        _block("F2b")
    with pytest.raises(ReviewSuiteValidationError, match="no scale initial-state"):
        _block("D1")
