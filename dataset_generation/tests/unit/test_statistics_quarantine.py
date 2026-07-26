from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_robot_dataset.common import statistics
from dynamic_robot_dataset.common.schema import EpisodeRecord


def _record(
    index: int,
    *,
    duration_s: float,
    view_count: int,
    dynamics_mode: str = "free_contact",
    release_tier: str = "free_contact",
    legacy_namespace: bool = False,
) -> EpisodeRecord:
    unverified = release_tier == "quarantine"
    assistance = {
        "assisted_grasp": dynamics_mode == "assisted_contact",
        "assisted_retention": False,
        "equality_constraint_active": False,
        "latch_active": False,
        "constraint_activation_time": (
            0.0 if dynamics_mode == "assisted_contact" else None
        ),
        "constraint_deactivation_time": (
            1.0 if dynamics_mode == "assisted_contact" else None
        ),
    }
    if dynamics_mode == "assisted_contact":
        assistance["mechanisms"] = [
            {
                "mechanism_id": "legacy-grasp",
                "mechanism_type": "assisted_grasp",
                "source": "simulator_observed",
                "activation_intervals": [
                    {"start_time_s": 0.0, "end_time_s": 1.0}
                ],
                "target_body_ids": ["object"],
            }
        ]
    video_paths = {
        f"observation.images.view_{view}": f"videos/view-{view}/file-{index}.mp4"
        for view in range(view_count)
    }
    return EpisodeRecord(
        episode_uuid=f"00000000-0000-4000-8000-{index:012d}",
        episode_index=index,
        counterfactual_bundle_id=f"bundle-{index}",
        physics_counterfactual_family_id=f"physics-{index}",
        split_group_id=f"split-{index}",
        scene_seed=index,
        branch_seed=100 + index,
        family="legacy_assisted/F1a" if legacy_namespace else "falling_catch",
        subfamily="centered_vertical_drop",
        intended_branch="success",
        actual_outcome="unverified" if unverified else "success",
        task_success=not unverified,
        failure_mode="label_unverified" if unverified else "none",
        label_status="unverified" if unverified else "verified_objective",
        dynamics_mode=dynamics_mode,
        release_tier=release_tier,
        physics_qc_pass=not unverified,
        source_generator="statistics-test",
        source_generator_version="1",
        generator_git_commit="test",
        config_hash="a" * 64,
        simulator_name="test",
        simulator_version="1",
        renderer="test",
        duration_s=duration_s,
        task_index=0,
        video_paths=video_paths,
        camera_stream_calibration_ids={name: name for name in video_paths},
        camera_ids=list(video_paths),
        assistance=assistance,
        extras=(
            {"corpus_namespace": "legacy_assisted"}
            if legacy_namespace
            else {}
        ),
    )


def test_headline_hours_exclude_quarantine_and_preserve_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        _record(0, duration_s=10.0, view_count=2),
        _record(
            1,
            duration_s=20.0,
            view_count=2,
            release_tier="quarantine",
            legacy_namespace=True,
        ),
        _record(
            2,
            duration_s=30.0,
            view_count=1,
            dynamics_mode="assisted_contact",
            release_tier="assisted_contact",
        ),
        _record(
            3,
            duration_s=40.0,
            view_count=3,
            dynamics_mode="scripted_motion",
            release_tier="scripted_motion",
        ),
    ]
    root = tmp_path / "dataset"
    root.mkdir()
    monkeypatch.setattr(statistics, "load_episode_records", lambda _root: records)

    report = statistics.collect_dataset_statistics(root, probe_streams=False)

    assert report.logical_episode_count == 1
    assert report.encoded_source_view_count == 2
    assert report.inventory_logical_episode_count == 4
    assert report.inventory_encoded_source_view_count == 8
    assert report.quarantine_logical_episode_count == 3
    assert report.quarantine_encoded_source_view_count == 6
    # A two-camera rollout is ten seconds of experience, not twenty.
    assert report.durations.unique_source_episode_seconds == pytest.approx(10.0)
    assert report.durations.encoded_source_stream_seconds == pytest.approx(20.0)
    assert report.inventory_durations.unique_source_episode_seconds == pytest.approx(
        100.0
    )
    assert report.inventory_durations.encoded_source_stream_seconds == pytest.approx(
        210.0
    )
    assert report.quarantine_durations.unique_source_episode_seconds == pytest.approx(
        90.0
    )
    assert report.quarantine_durations.encoded_source_stream_seconds == pytest.approx(
        190.0
    )
    payload = report.to_dict()
    assert payload["durations"]["unique_source_episode_hours"] == pytest.approx(
        10.0 / 3600.0
    )
    assert payload["inventory_durations"]["unique_source_episode_hours"] == pytest.approx(
        100.0 / 3600.0
    )
    assert payload["quarantine_durations"]["unique_source_episode_hours"] == pytest.approx(
        90.0 / 3600.0
    )
    assert "each logical rollout counts once" in payload["accounting_semantics"][
        "unique_duration_rule"
    ]


def test_derived_exports_are_partitioned_by_logical_episode_not_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        _record(0, duration_s=10.0, view_count=2),
        _record(
            1,
            duration_s=20.0,
            view_count=2,
            release_tier="quarantine",
            legacy_namespace=True,
        ),
    ]
    root = tmp_path / "dataset"
    wan = tmp_path / "wan"
    root.mkdir()
    wan.mkdir()
    monkeypatch.setattr(statistics, "load_episode_records", lambda _root: records)
    rows = [
        {
            "episode_uuid": record.episode_uuid,
            "frame_count": 48,
            "fps": 24,
            "videos": {"main": "main.mp4", "secondary": "secondary.mp4"},
            "sampling": {
                "main": {
                    "endpoint_padding_start_frames": 1,
                    "endpoint_padding_end_frames": 0,
                }
            },
        }
        for record in records
    ]
    (wan / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    report = statistics.collect_dataset_statistics(
        root, wan_root=wan, probe_streams=False
    )

    assert report.durations.unique_derived_clip_seconds == pytest.approx(2.0)
    assert report.durations.encoded_derived_stream_seconds == pytest.approx(4.0)
    assert report.inventory_durations.unique_derived_clip_seconds == pytest.approx(4.0)
    assert report.inventory_durations.encoded_derived_stream_seconds == pytest.approx(8.0)
    assert report.quarantine_durations.unique_derived_clip_seconds == pytest.approx(2.0)
    assert report.quarantine_durations.encoded_derived_stream_seconds == pytest.approx(4.0)


def test_dataset_level_legacy_assisted_namespace_quarantines_all_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(0, duration_s=12.0, view_count=2)
    root = tmp_path / "legacy-root"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"extras": {"dataset_namespace": "legacy_assisted/v1"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(statistics, "load_episode_records", lambda _root: [record])

    report = statistics.collect_dataset_statistics(root, probe_streams=False)

    assert report.logical_episode_count == 0
    assert report.encoded_source_view_count == 0
    assert report.inventory_logical_episode_count == 1
    assert report.quarantine_logical_episode_count == 1
    assert report.durations.unique_source_episode_seconds == 0.0
    assert report.quarantine_durations.unique_source_episode_seconds == 12.0
