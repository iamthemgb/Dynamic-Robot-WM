from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dynamic_robot_dataset.common.cameras import (
    CameraCalibration,
    invert_rigid_transform,
    pose_to_matrix,
    quaternion_xyzw_to_wxyz,
)
from dynamic_robot_dataset.common.contacts import AssistanceSample, AssistanceSummary
from dynamic_robot_dataset.common.episode_writer import EpisodeWriter
from dynamic_robot_dataset.common.paths import (
    ExistingOutputError,
    ResumeGuard,
    ResumeMismatchError,
    ensure_not_source_path,
    portable_relative_path,
)
from dynamic_robot_dataset.common.provenance import environment_hash
from dynamic_robot_dataset.common.schema import (
    DynamicsMode,
    EpisodeRecord,
    LabelStatus,
    ReleaseTier,
    SchemaValidationError,
    Split,
)
from dynamic_robot_dataset.common.splits import SplitAssigner, validate_no_split_leakage
from dynamic_robot_dataset.common.synchronization import (
    exact_frame_timestamps,
    synchronize_previous,
    validate_synchronized_streams,
)


def _episode(index: int, **updates: object) -> EpisodeRecord:
    values: dict[str, object] = {
        "episode_uuid": f"00000000-0000-4000-8000-{index:012d}",
        "episode_index": index,
        "counterfactual_bundle_id": f"bundle-{index}",
        "physics_counterfactual_family_id": f"physics-{index}",
        "split_group_id": f"group-{index}",
        "scene_seed": index,
        "branch_seed": index + 100,
        "family": "falling_catch",
        "subfamily": "centered_vertical_drop",
        "intended_branch": "success_seeking",
        "actual_outcome": "success",
        "task_success": True,
        "failure_mode": "none",
        "source_generator": "test",
        "source_generator_version": "1",
        "config_hash": "a" * 64,
        "simulator_name": "test",
        "simulator_version": "1",
        "renderer": "test",
    }
    values.update(updates)
    return EpisodeRecord(**values)  # type: ignore[arg-type]


def test_environment_hash_does_not_require_pip_and_tracks_lockfile(tmp_path: Path) -> None:
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    first = environment_hash(lockfiles=[lockfile])
    assert len(first) == 64
    assert first != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    lockfile.write_text("version = 2\n", encoding="utf-8")
    assert environment_hash(lockfiles=[lockfile]) != first


def test_failed_outcome_needs_concrete_failure_code() -> None:
    record = _episode(0, task_success=False, actual_outcome="miss", failure_mode="none")
    with pytest.raises(SchemaValidationError, match="concrete failure_mode"):
        record.validate()


def test_failed_outcome_rejects_unknown_failure_code() -> None:
    record = _episode(
        0,
        task_success=False,
        actual_outcome="miss",
        failure_mode="totally_made_up_typo",
    )
    with pytest.raises(SchemaValidationError, match="Unknown failure_mode"):
        record.validate()


def test_branch_intent_does_not_control_measured_label() -> None:
    record = _episode(
        0,
        intended_branch="success_seeking",
        actual_outcome="near_miss",
        task_success=False,
        failure_mode="receptacle_near_miss",
    )
    record.validate()
    assert record.intended_branch != record.actual_outcome
    assert not record.task_success


def test_unverified_and_assisted_records_are_not_default_release() -> None:
    unverified = _episode(0, label_status=LabelStatus.UNVERIFIED, release_tier=ReleaseTier.UNVERIFIED)
    assisted = _episode(
        1,
        dynamics_mode=DynamicsMode.ASSISTED_CONTACT,
        release_tier=ReleaseTier.ASSISTED_CONTACT,
        assistance={
            "assisted_grasp": False,
            "assisted_retention": True,
            "equality_constraint_active": False,
            "latch_active": True,
            "constraint_activation_time": 0.1,
            "constraint_deactivation_time": 0.4,
        },
    )
    unverified.validate()
    assisted.validate()
    assert not unverified.release_eligible
    assert not assisted.release_eligible


def test_portable_paths_and_source_write_protection(tmp_path: Path) -> None:
    assert portable_relative_path("videos/main/chunk-000/file-000.mp4") == "videos/main/chunk-000/file-000.mp4"
    for invalid in ("/absolute/file.mp4", "../escape", "a/../escape", "a\\b"):
        with pytest.raises(ValueError):
            portable_relative_path(invalid)
    assert ensure_not_source_path(tmp_path / "dataset") == (tmp_path / "dataset").resolve()
    with pytest.raises(PermissionError):
        ensure_not_source_path("/gpfs/radev/scratch/sous/mzl7/do-not-write")
    with pytest.raises(PermissionError):
        ensure_not_source_path("/gpfs/radev/project/sous/zss8/dataset-generation/do-not-write")
    with pytest.raises(PermissionError):
        ensure_not_source_path("/gpfs/radev/project/sous/zl664/wan_scripts/checkpoints/do-not-write")
    repository_root = Path(__file__).resolve().parents[2]
    with pytest.raises(PermissionError):
        ensure_not_source_path(repository_root / "legacy_sources" / "mzl7" / "do-not-write")


def test_resume_requires_identical_resolved_configuration(tmp_path: Path) -> None:
    output = tmp_path / "run"
    ResumeGuard(output, {"seed": 7}).initialize()
    assert ResumeGuard(output, {"seed": 7}, resume=True).initialize()
    with pytest.raises(ResumeMismatchError):
        ResumeGuard(output, {"seed": 8}, resume=True).initialize()
    with pytest.raises(ExistingOutputError):
        ResumeGuard(output, {"seed": 7}, resume=False).initialize()


def test_episode_uuid_cannot_escape_writer_staging_root(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path / "run", {"seed": 9})
    sentinel = writer.root / "sentinel"
    sentinel.write_text("must survive", encoding="utf-8")
    malicious = _episode(0, episode_uuid="..")
    with pytest.raises(SchemaValidationError, match="canonical UUID"):
        writer.write_episode(malicious, frame_rows=(), videos={})
    assert sentinel.read_text(encoding="utf-8") == "must survive"
    with pytest.raises(ValueError, match="canonical UUID"):
        writer._transaction_dir("..")


def test_exact_timestamps_and_causal_controller_alignment() -> None:
    frames = exact_frame_timestamps(4, 30)
    assert frames == [0.0, 1 / 30, 2 / 30, 0.1]
    validate_synchronized_streams({"main": frames, "secondary": list(frames)})
    aligned = synchronize_previous(frames, [0.0, 0.05, 0.1], ["a", "b", "c"])
    assert [sample.value for sample in aligned] == ["a", "a", "b", "c"]


def test_camera_roundtrip_and_quaternion_conversion() -> None:
    camera_to_world = pose_to_matrix((0.1, -0.2, 1.0), quaternion_xyzw_to_wxyz((0, 0, 0, 1)))
    calibration = CameraCalibration(
        camera_name="observation.images.main",
        intrinsic_matrix=(500, 0, 416, 0, 500, 240, 0, 0, 1),
        world_to_camera=invert_rigid_transform(camera_to_world),
        camera_to_world=camera_to_world,
    )
    calibration.validate()
    assert CameraCalibration.from_dict(calibration.to_dict()) == calibration
    assert quaternion_xyzw_to_wxyz((0, 0, 0, 1)) == (1.0, 0.0, 0.0, 0.0)
    invalid = replace(calibration, intrinsic_matrix=(0.0,) * 9)
    with pytest.raises(SchemaValidationError, match="focal lengths"):
        invalid.validate()


def test_event_time_must_lie_inside_episode() -> None:
    record = _episode(0, duration_s=1.0, event_time_s=-0.1)
    with pytest.raises(SchemaValidationError, match="event_time_s"):
        record.validate()
    record.event_time_s = 1.1
    with pytest.raises(SchemaValidationError, match="outside"):
        record.validate()


def test_assistance_masks_derive_episode_mode_and_times() -> None:
    summary = AssistanceSummary.from_samples(
        [
            AssistanceSample(0.0),
            AssistanceSample(0.1, assisted_retention=True, latch_active=True),
            AssistanceSample(0.2, assisted_retention=True, latch_active=True),
            AssistanceSample(0.3),
        ]
    )
    assert summary.dynamics_mode == DynamicsMode.ASSISTED_CONTACT
    assert summary.constraint_activation_time == 0.1
    assert summary.constraint_deactivation_time == 0.2


def test_counterfactual_relations_stay_in_one_split() -> None:
    records = [
        _episode(0, counterfactual_bundle_id="action-family", split_group_id="scene-family"),
        _episode(1, counterfactual_bundle_id="action-family", split_group_id="scene-family"),
        _episode(2, physics_counterfactual_family_id="physics-family", split_group_id="scene-family"),
    ]
    assignments = SplitAssigner(seed=11).assign(records)
    assert len({assignment.split for assignment in assignments}) == 1
    assert validate_no_split_leakage(records, assignments) == []


def test_split_validator_catches_scene_and_parent_leakage() -> None:
    parent = _episode(0, split=Split.TRAIN)
    sibling = _episode(
        1,
        counterfactual_bundle_id="other-bundle",
        physics_counterfactual_family_id="other-physics",
        split_group_id="other-declared-group",
        scene_seed=parent.scene_seed,
        parent_episode_uuid=parent.episode_uuid,
        split=Split.TEST,
    )
    problems = validate_no_split_leakage([parent, sibling])
    assert problems and "connected leakage group" in problems[0]
