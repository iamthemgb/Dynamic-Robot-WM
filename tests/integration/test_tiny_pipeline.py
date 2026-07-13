"""One real MP4/Parquet/split/QC/Wan pipeline, including finalize-crash resume."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dynamic_robot_dataset.common.episode_writer import EpisodeWriter, load_episode_records
from dynamic_robot_dataset.common.provenance import GenerationProvenance
from dynamic_robot_dataset.common.qc import validate_dataset
from dynamic_robot_dataset.common.schema import DatasetInfo, NamedFeature, Split, TimeBase
from dynamic_robot_dataset.common.splits import SplitAssigner
from dynamic_robot_dataset.common.video_writer import VideoSpec, probe_video
from dynamic_robot_dataset.common.wan_export import export_wan
from dynamic_robot_dataset.families import get_family
from dynamic_robot_dataset.families.base import GenerationRequest
from dynamic_robot_dataset.smoke_runner import (
    DiagnosticRenderer,
    _event_rows,
    _frame_rows,
    _high_rate_rows,
    _record,
    smoke_cameras,
)


pytestmark = pytest.mark.integration


def test_tiny_pipeline_and_finalize_crash_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = GenerationRequest(
        family="falling_catch",
        subfamily="centered_vertical_drop",
        num_bundles=1,
        branches=("success_seeking",),
        views=("main", "secondary"),
        seed=731,
        scene_style="clean_franka_lab",
    )
    result = get_family(request.family).generate(request)[0]
    output = tmp_path / "canonical"
    config = {"test": "tiny-full-pipeline", "seed": request.seed, "non_production": True}
    writer = EpisodeWriter(
        output,
        config,
        video_spec=VideoSpec(preset="ultrafast"),
    )
    record = _record(result, 0, "integration-test")
    record.task_index = 0
    assignment = SplitAssigner(seed=0).assign([record])[0]
    record.split = Split(assignment.split)
    record.split_group_id = assignment.split_group_id
    cameras = smoke_cameras()
    writer.write_episode(
        record,
        frame_rows=_frame_rows(result, 0),
        videos={name: DiagnosticRenderer(result, camera).frames() for name, camera in cameras.items()},
        high_rate_rows=_high_rate_rows(result, 0),
        event_rows=_event_rows(result, 0),
        object_state_rows=(),
    )
    info = DatasetInfo(
        name="tiny_pipeline",
        time_base=TimeBase(240.0, 60.0, 30.0),
        state_features=[NamedFeature("primary_target", "named_columns", "SI")],
        action_features=[NamedFeature("command", "named_columns", "declared_per_column")],
    )
    camera_rows = [{"camera_id": name, **camera.to_dict()} for name, camera in cameras.items()]
    provenance = GenerationProvenance(
        source_generator="integration-test",
        source_generator_version="1",
        generator_git_commit="integration-test",
        config_hash=writer.config_hash,
        simulator_name="family-smoke-solver",
        simulator_version="1",
        renderer="diagnostic_state_renderer",
        asset_hashes={},
        source_hashes={},
    ).to_dict()
    split_rows = [
        {
            "episode_uuid": assignment.episode_uuid,
            "episode_index": assignment.episode_index,
            "split_group_id": assignment.split_group_id,
            "split": assignment.split,
        }
    ]

    original_commit = EpisodeWriter._commit_meta_transaction
    injected = {"pending": True}

    def crash_after_links(self: EpisodeWriter, staging: Path, meta: Path) -> None:
        if not injected["pending"]:
            original_commit(self, staging, meta)
            return
        injected["pending"] = False
        transaction = json.loads((staging / ".transaction.json").read_text(encoding="utf-8"))
        meta.mkdir(parents=True, exist_ok=True)
        for name in transaction["content_hashes"]:
            os.link(staging / name, meta / name)
        raise RuntimeError("injected crash before completion marker")

    monkeypatch.setattr(EpisodeWriter, "_commit_meta_transaction", crash_after_links)
    with pytest.raises(RuntimeError, match="injected crash"):
        writer.finalize(
            info,
            cameras=camera_rows,
            provenance=[provenance],
            splits=split_rows,
        )
    assert not (output / "meta" / ".complete.json").exists()
    assert (output / ".staging" / "_meta" / ".transaction.json").is_file()

    resumed = EpisodeWriter(output, config, resume=True)
    resumed.finalize(info, cameras=camera_rows, provenance=[provenance], splits=split_rows)
    assert (output / "meta" / ".complete.json").is_file()
    loaded = load_episode_records(output)
    assert len(loaded) == 1 and loaded[0].split != Split.UNASSIGNED

    report = validate_dataset(output, deep_video_checks=True, write_reports=True)
    assert report.passed
    assert report.episodes[0].passed
    assert (output / "qc" / "manifests" / "default_training.jsonl").read_text() == ""
    assert (output / "qc" / "manifests" / "free_contact.jsonl").read_text().strip()

    wan_root = tmp_path / "wan"
    summary = export_wan(output, wan_root, include_nonrelease=True)
    assert summary.episode_count == 1 and summary.video_count == 2
    manifest = json.loads((wan_root / "manifest.jsonl").read_text().splitlines()[0])
    assert manifest["action"]["available"] is True
    assert manifest["trajectory"]["available"] is True
    assert manifest["contact"]["available"] is True
    derived = probe_video(wan_root / manifest["video"])
    assert (derived.width, derived.height, derived.frame_count) == (832, 480, 121)
    assert derived.fps == pytest.approx(24.0)
