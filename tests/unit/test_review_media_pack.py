from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import pytest

import dynamic_robot_dataset.common.review as review_module
from dynamic_robot_dataset.common.corpus_registry import (
    load_backend_capability_registry,
)
from dynamic_robot_dataset.common.episode_writer import (
    FINALIZED_METADATA_ARTIFACTS,
    episode_record_to_table_row,
    write_parquet_atomic,
)
from dynamic_robot_dataset.common.hashing import sha256_file, sha256_json
from dynamic_robot_dataset.common.paths import atomic_write_json
from dynamic_robot_dataset.common.review import (
    STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA,
    bind_fixed_review_media_pack_strict,
    write_fixed_review_media_pack,
)
from dynamic_robot_dataset.common.review_suite import write_review_suite_bundle
from dynamic_robot_dataset.common.schema import EpisodeRecord, Split
from dynamic_robot_dataset.common.source_scenario import (
    CounterfactualIdentity,
    EmbodimentSpec,
    PoseSpec,
    RoboCasaAssetManifest,
    SourceCameraSpec,
    SourceScenarioSpec,
)
from dynamic_robot_dataset.common.video_writer import VideoProbe


def _scenario(bundle: Any) -> SourceScenarioSpec:
    case = next(
        value
        for value in bundle.plan.cases
        if value.corpus_leaf_id == "P0a" and value.rollout_index == 0
    )
    backend = load_backend_capability_registry().by_name[case.backend]
    return SourceScenarioSpec(
        scenario_id=case.case_id,
        corpus_leaf_id=case.corpus_leaf_id,
        task_variant=case.task_variant,
        backend=case.backend,
        duration_s=1.0,
        physics={
            "simulation_hz": 1200,
            "gravity_world_m_s2": [0.0, 0.0, -9.81],
        },
        initial_state={
            "object_position_m": [0.0, 0.0, 0.8],
            "object_linear_velocity_m_s": [0.0, 0.0, 0.0],
        },
        embodiment=EmbodimentSpec(
            end_effector="no_robot",
            robot_model="none",
            action_names=(),
            action_semantics="no_actuators/v1",
        ),
        fixtures=(),
        actuator_phases=(),
        cameras=(
            SourceCameraSpec(
                name="main",
                role="main_three_quarter",
                pose=PoseSpec((2.0, -2.0, 1.5)),
                look_at_m=(0.0, 0.0, 0.5),
            ),
            SourceCameraSpec(
                name="secondary",
                role="secondary_side",
                pose=PoseSpec((0.0, 2.0, 1.3)),
                look_at_m=(0.0, 0.0, 0.5),
            ),
        ),
        rng_subseeds=case.rng_subseeds,
        robocasa_manifest=RoboCasaAssetManifest(
            catalog_id="review_robocasa_catalog",
            catalog_version=bundle.plan.robocasa_catalog_version,
            asset_root_id="robocasa_read_only",
            catalog_sha256=bundle.plan.robocasa_catalog_sha256,
            license_manifest_sha256=bundle.plan.robocasa_license_sha256,
            assets=(),
        ),
        counterfactual=CounterfactualIdentity(
            bundle_id=case.counterfactual_bundle_id,
            split_group_id=case.counterfactual_bundle_id,
            branch_id=case.counterfactual_branch_id,
            sibling_index=case.counterfactual_sibling_index,
        ),
        source_hashes={
            **backend.source_hashes,
            "robocasa_catalog": bundle.plan.robocasa_catalog_sha256,
        },
    )


def _write_finalized_fixture(
    tmp_path: Path,
    *,
    key_event_time_s: float = 0.5,
    include_secondary: bool = True,
    dataset_episode_index: int = 0,
) -> dict[str, Any]:
    bundle = write_review_suite_bundle(tmp_path / "suite")
    scenario = _scenario(bundle)
    case = next(
        value for value in bundle.plan.cases if value.case_id == scenario.scenario_id
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    scenario_path = dataset / "spec/source_scenario.json"
    scenario_path.parent.mkdir(parents=True)
    scenario_path.write_text(scenario.to_json(indent=2), encoding="utf-8")
    source_manifest_path = dataset / "provenance/source_manifest.json"
    source_manifest_path.parent.mkdir(parents=True)
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA,
                "backend": scenario.backend,
                "source_hashes": dict(scenario.source_hashes),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    timestamps = tuple(index / 30 for index in range(30))
    frame_path = write_parquet_atomic(
        dataset / "data/frame.parquet",
        [
            {
                "episode_index": dataset_episode_index,
                "frame_index": index,
                "video_frame_index": index,
                "timestamp": timestamp,
            }
            for index, timestamp in enumerate(timestamps)
        ],
    )
    canonical_video_paths = {
        "observation.images.main": "videos/main.mp4",
    }
    if include_secondary:
        canonical_video_paths["observation.images.secondary"] = (
            "videos/secondary.mp4"
        )
    for relative in canonical_video_paths.values():
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    content_hashes = {
        "data/frame.parquet": sha256_file(frame_path),
        **{
            relative: sha256_file(dataset / relative)
            for relative in canonical_video_paths.values()
        },
    }
    record = EpisodeRecord(
        episode_uuid=case.episode_uuid,
        episode_index=dataset_episode_index,
        counterfactual_bundle_id=case.counterfactual_bundle_id,
        split_group_id=case.counterfactual_bundle_id,
        scene_seed=1,
        branch_seed=2,
        family=case.family,
        subfamily=case.subfamily,
        variant=case.task_variant,
        tool_type="no_robot",
        action_mode="no_actuators/v1",
        intended_branch=case.branch_role,
        actual_outcome="miss",
        task_success=False,
        failure_mode="no_contact",
        source_generator="source_mujoco",
        source_generator_version="1",
        config_hash="a" * 64,
        simulator_name="mujoco",
        simulator_version="3.10.0",
        renderer="mujoco.Renderer",
        split=Split.TRAIN,
        task_index=0,
        frame_count=len(timestamps),
        duration_s=scenario.duration_s,
        event_time_s=key_event_time_s,
        key_event_name="freefall_midpoint",
        objective_evaluator_id=case.evaluator,
        objective_evaluator_version="1",
        objective_evidence={
            "independently_recomputed": True,
            "evidence_hash": "a" * 64,
            "source": "persisted_frame_rows",
        },
        randomization={
            "background_style": "clean_franka_lab",
            "randomization_level": case.randomization_level,
        },
        quality_flags=["review_only"],
        video_paths=canonical_video_paths,
        frame_data_path="data/frame.parquet",
        camera_stream_calibration_ids={
            stream: stream for stream in canonical_video_paths
        },
        camera_ids=sorted(canonical_video_paths),
        content_hashes=content_hashes,
        extras={
            "backend": "source_mujoco",
            "end_effector": "no_robot",
            "review_case": case.to_dict(),
            "review_case_sha256": case.case_sha256,
            "review_suite_episode_index": case.episode_index,
            "source_scenario_spec": scenario.to_dict(),
            "source_scenario_spec_sha256": scenario.spec_hash,
        },
    )
    record.validate()

    meta = dataset / "meta"
    meta.mkdir()
    write_parquet_atomic(
        meta / "episodes.parquet", [episode_record_to_table_row(record)]
    )
    write_parquet_atomic(
        meta / "splits.parquet",
        [
            {
                "episode_uuid": record.episode_uuid,
                "episode_index": record.episode_index,
                "split_group_id": record.split_group_id,
                "split": record.split.value,
            }
        ],
    )
    for name in (
        "tasks.parquet",
        "cameras.parquet",
        "provenance.parquet",
        "counterfactual_families.parquet",
    ):
        write_parquet_atomic(meta / name, [{"fixture": name}])
    atomic_write_json(meta / "info.json", {"name": "fixed-review-test"})
    seal_path = atomic_write_json(
        dataset / ".seal.json",
        {
            "schema_version": "dynamic-robot-dataset-seal/v1",
            "config_hash": record.config_hash,
            "run_plan_sha256": bundle.plan.plan_sha256,
        },
    )
    metadata_hashes = {
        name: sha256_file(meta / name) for name in FINALIZED_METADATA_ARTIFACTS
    }
    completion_path = atomic_write_json(
        meta / ".complete.json",
        {
            "config_hash": record.config_hash,
            "content_hashes": metadata_hashes,
            "seal_sha256": sha256_file(seal_path),
        },
    )

    qc_path = dataset / "qc/dataset_report.json"
    qc_path.parent.mkdir(parents=True)
    qc_path.write_text(
        json.dumps(
            {
                "schema_version": "dynamic-robot-qc-report/v2",
                "dataset_root": str(dataset.resolve()),
                "dataset_episodes_sha256": sha256_file(meta / "episodes.parquet"),
                "metadata_complete_manifest_sha256": sha256_file(completion_path),
                "metadata_content_hashes": metadata_hashes,
                "strict_all": True,
                "passed": True,
                "global_failures": [],
                "episodes": [
                    {
                        "episode_uuid": case.episode_uuid,
                        "episode_index": dataset_episode_index,
                        "passed": True,
                        "metrics": {
                            "objective_recompute": {
                                "evaluator_id": case.evaluator,
                                "evidence_version": "dynamic-robot-objective-evidence/v1",
                                "evidence_hash": "a" * 64,
                                "key_event_name": "freefall_midpoint",
                                "key_event_time_s": key_event_time_s,
                                "replay_match": True,
                            }
                        },
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "bundle": bundle,
        "case": case,
        "dataset": dataset,
        "scenario_path": scenario_path,
        "source_manifest_path": source_manifest_path,
        "qc_path": qc_path,
        "timestamps": timestamps,
        "record": record,
    }


def _fake_video_decode(
    monkeypatch: pytest.MonkeyPatch,
    timestamps: tuple[float, ...],
) -> None:
    def probe(path: str | Path) -> VideoProbe:
        return VideoProbe(
            path=str(path),
            width=832,
            height=480,
            codec_name="h264",
            pixel_format="yuv420p",
            fps_num=30,
            fps_den=1,
            time_base_num=1,
            time_base_den=15360,
            frame_count=len(timestamps),
            duration_s=len(timestamps) / 30,
        )

    def frames(path: str | Path, width: int, height: int):
        offset = 100 if "secondary" in str(path) else 0
        for index in range(len(timestamps)):
            yield bytes([(offset + index) % 256]) * (width * height * 3)

    monkeypatch.setattr(review_module, "probe_video", probe)
    monkeypatch.setattr(
        review_module, "probe_frame_timestamps", lambda _path: list(timestamps)
    )
    monkeypatch.setattr(review_module, "iter_rgb_frames", frames)


def _write(value: dict[str, Any]):
    return write_fixed_review_media_pack(
        value["dataset"],
        review_suite_root=value["bundle"].root,
        source_scenario_spec_path=value["scenario_path"],
    )


def test_fixed_media_pack_decodes_finalized_frames_and_binds_strictly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _write_finalized_fixture(tmp_path, dataset_episode_index=0)
    _fake_video_decode(monkeypatch, value["timestamps"])

    publication = _write(value)
    publication.validate()
    manifest = publication.manifest
    assert manifest.review_matrix_episode_index == value["case"].episode_index
    assert manifest.dataset_episode_index == 0
    assert manifest.event_strip_indices == {
        "pre_event": 12,
        "event": 15,
        "post_0p1_s": 18,
        "post_0p3_s": 24,
        "final": 29,
    }
    assert sha256_file(value["dataset"] / publication.manifest_path) == (
        publication.manifest_sha256
    )
    for view, relative in manifest.event_strip_paths.items():
        path = value["dataset"] / relative
        assert sha256_file(path) == manifest.event_strip_sha256[view]
        with Image.open(path) as strip:
            assert strip.size == (832 * 5, 480)
            offset = 100 if view == "secondary" else 0
            expected = [12, 15, 18, 24, 29]
            assert [
                strip.getpixel((panel * 832 + 10, 10))[0]
                for panel in range(5)
            ] == [offset + index for index in expected]

    artifact = bind_fixed_review_media_pack_strict(
        value["dataset"],
        review_suite_root=value["bundle"].root,
        source_scenario_spec_path=value["scenario_path"],
        source_manifest_path=value["source_manifest_path"],
        media_pack_manifest_path=publication.manifest_path,
    )
    assert artifact.event_strip_sha256 == manifest.event_strip_sha256
    assert artifact.video_sha256 == manifest.video_sha256


def test_fixed_media_pack_uses_local_preview_index_not_global_matrix_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _write_finalized_fixture(tmp_path, dataset_episode_index=7)
    _fake_video_decode(monkeypatch, value["timestamps"])

    publication = _write(value)

    assert publication.manifest.dataset_episode_index == 7
    assert publication.manifest.review_matrix_episode_index == value["case"].episode_index


def test_fixed_media_pack_fails_closed_without_both_finalized_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _write_finalized_fixture(tmp_path, include_secondary=False)
    _fake_video_decode(monkeypatch, value["timestamps"])

    with pytest.raises(ValueError, match="missing synchronized main and secondary"):
        _write(value)

    expected_parent = value["dataset"] / Path(
        next(
            request
            for request in value["bundle"].requests.requests
            if request.case_id == value["case"].case_id
        ).event_strip_request["output_paths"]["main"]
    ).parent
    assert not expected_parent.exists()


@pytest.mark.parametrize(
    ("key_event_time_s", "message"),
    [
        (0.05, "pre-event context"),
        (20 / 30, "duplicate, clamped"),
    ],
)
def test_fixed_media_pack_rejects_boundary_clamping_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key_event_time_s: float,
    message: str,
) -> None:
    value = _write_finalized_fixture(
        tmp_path, key_event_time_s=key_event_time_s
    )
    _fake_video_decode(monkeypatch, value["timestamps"])

    with pytest.raises(ValueError, match=message):
        _write(value)


def test_strict_media_pack_binding_detects_post_publish_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _write_finalized_fixture(tmp_path)
    _fake_video_decode(monkeypatch, value["timestamps"])
    publication = _write(value)
    main = value["dataset"] / publication.manifest.event_strip_paths["main"]
    main.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="event-strip hash changed"):
        bind_fixed_review_media_pack_strict(
            value["dataset"],
            review_suite_root=value["bundle"].root,
            source_scenario_spec_path=value["scenario_path"],
            source_manifest_path=value["source_manifest_path"],
            media_pack_manifest_path=publication.manifest_path,
        )


def test_fixed_media_pack_rejects_unfinalized_or_stale_qc_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unfinalized = _write_finalized_fixture(tmp_path / "unfinalized")
    _fake_video_decode(monkeypatch, unfinalized["timestamps"])
    (unfinalized["dataset"] / "meta/.complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        _write(unfinalized)
    assert not (unfinalized["dataset"] / "reviews").exists()

    stale = _write_finalized_fixture(tmp_path / "stale")
    _fake_video_decode(monkeypatch, stale["timestamps"])
    qc = json.loads(stale["qc_path"].read_text(encoding="utf-8"))
    qc["dataset_episodes_sha256"] = "b" * 64
    stale["qc_path"].write_text(json.dumps(qc), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to finalized episode metadata"):
        _write(stale)
    assert not (stale["dataset"] / "reviews").exists()
