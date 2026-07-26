from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from dynamic_robot_dataset import cli
from dynamic_robot_dataset.common.episode_writer import (
    FINALIZED_METADATA_ARTIFACTS,
    DatasetSealedError,
    EpisodeWriter,
)
from dynamic_robot_dataset.common.hashing import sha256_file
from dynamic_robot_dataset.common.paths import ResumeMismatchError, atomic_write_json
from dynamic_robot_dataset.common.qc import validate_dataset
from dynamic_robot_dataset.common.run_orchestration import (
    EpisodeAttemptFailure,
    EpisodeMaterialization,
    TERMINAL_FAILURE_LEDGER_FILE,
    TERMINAL_FAILURE_STAGE,
    finalize_run,
    load_run_plan,
    plan_run,
    run_shard,
)
from dynamic_robot_dataset.common.schema import ActualOutcomeClass, DatasetInfo, EpisodeRecord
from dynamic_robot_dataset.common.video_writer import VideoSpec


def _uuid(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012d}"


def _declarations(count: int) -> list[dict[str, object]]:
    return [
        {"episode_uuid": _uuid(index), "episode_index": index, "seed": 1000 + index}
        for index in range(count)
    ]


def _episode(index: int) -> EpisodeRecord:
    return EpisodeRecord(
        episode_uuid=_uuid(index),
        episode_index=index,
        counterfactual_bundle_id=f"bundle-{index}",
        physics_counterfactual_family_id=f"physics-{index}",
        split_group_id=f"group-{index}",
        scene_seed=index,
        branch_seed=index + 100,
        family="falling_catch",
        subfamily="centered_vertical_drop",
        intended_branch="success_seeking",
        actual_outcome="success",
        task_success=True,
        failure_mode="none",
        source_generator="test",
        source_generator_version="1",
        config_hash="a" * 64,
        simulator_name="test",
        simulator_version="1",
        renderer="test",
        task_index=0,
    )


def _publish_marker(writer: EpisodeWriter, record: EpisodeRecord) -> EpisodeRecord:
    record.config_hash = writer.config_hash
    record.validate()
    atomic_write_json(
        writer._marker(record.episode_uuid),
        {"config_hash": writer.config_hash, "episode": record.to_dict()},
    )
    return record


def _add_canonical_artifacts(root: Path, record: EpisodeRecord) -> None:
    index = record.episode_index
    record.video_paths = {
        "observation.images.main": f"videos/observation.images.main/fake-{index}.mp4",
        "observation.images.secondary": f"videos/observation.images.secondary/fake-{index}.mp4",
    }
    record.frame_data_path = f"data/fake-{index}.parquet"
    record.high_rate_path = f"high_rate/fake-{index}.parquet"
    record.events_path = f"events/fake-{index}.parquet"
    record.transition_events_path = f"transitions/fake-{index}.parquet"
    record.object_states_path = f"object_states/fake-{index}.parquet"
    references = [
        *record.video_paths.values(),
        record.frame_data_path,
        record.high_rate_path,
        record.events_path,
        record.transition_events_path,
        record.object_states_path,
    ]
    for relative in references:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}-{relative}".encode())
    record.content_hashes = {relative: sha256_file(root / relative) for relative in references}


def test_plan_is_content_addressed_deterministic_and_binds_writer_settings(tmp_path: Path) -> None:
    root = tmp_path / "run"
    config = {"seed": 71, "backend": "test_backend"}
    plan = plan_run(
        root,
        config,
        _declarations(5),
        shard_count=3,
        chunk_size=2,
        video_spec=VideoSpec(preset="ultrafast"),
    )
    assert [episode.shard_id for episode in plan.episodes] == [0, 1, 2, 0, 1]
    assert load_run_plan(root) == plan
    assert plan_run(
        root,
        config,
        _declarations(5),
        shard_count=3,
        resume=True,
        chunk_size=2,
        video_spec=VideoSpec(preset="ultrafast"),
    ) == plan

    with pytest.raises(ResumeMismatchError, match="Resolved config changed|settings changed"):
        plan_run(
            root,
            config,
            _declarations(5),
            shard_count=3,
            resume=True,
            chunk_size=3,
            video_spec=VideoSpec(preset="ultrafast"),
        )
    with pytest.raises(ResumeMismatchError, match="Resolved config changed|settings changed"):
        plan_run(
            root,
            config,
            _declarations(5),
            shard_count=3,
            resume=True,
            chunk_size=2,
            video_spec=VideoSpec(preset="medium"),
        )


def test_plan_run_is_exposed_by_public_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "config.json"
    episodes_path = tmp_path / "episodes.json"
    config_path.write_text(json.dumps({"seed": 9}), encoding="utf-8")
    episodes_path.write_text(json.dumps(_declarations(2)), encoding="utf-8")
    output = tmp_path / "cli-run"
    assert cli.main(
        [
            "plan-run",
            "--config",
            str(config_path),
            "--episodes",
            str(episodes_path),
            "--output",
            str(output),
            "--shards",
            "2",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["episode_count"] == 2
    assert report["shard_count"] == 2
    assert (output / ".run_plan.json").is_file()


def test_plan_tampering_and_noncontiguous_membership_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    plan_run(root, {"seed": 1}, _declarations(2), shard_count=1)
    path = root / ".run_plan.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["episodes"][0]["declaration"]["seed"] = 999
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable plan identity|plan_hash"):
        load_run_plan(root)

    with pytest.raises(ValueError, match="contiguous"):
        plan_run(
            tmp_path / "bad",
            {"seed": 1},
            [{"episode_uuid": _uuid(0), "episode_index": 1}],
            shard_count=1,
        )


def test_source_backend_plan_requires_complete_source_scenario_specs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a complete source_scenario_spec"):
        plan_run(
            tmp_path / "source-run",
            {"backend": "source_mujoco"},
            _declarations(1),
            shard_count=1,
        )


def test_run_shard_resumes_interruption_and_preserves_measured_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    plan_run(root, {"seed": 2}, _declarations(4), shard_count=2)

    def marker_only_write(self: EpisodeWriter, record: EpisodeRecord, **_kwargs: object) -> EpisodeRecord:
        return _publish_marker(self, record)

    monkeypatch.setattr(EpisodeWriter, "write_episode", marker_only_write)
    calls: list[int] = []
    interrupt_once = {"pending": True}

    def executor(entry):
        calls.append(entry.episode_index)
        if entry.episode_index == 2 and interrupt_once["pending"]:
            interrupt_once["pending"] = False
            raise RuntimeError("simulated worker interruption")
        return EpisodeMaterialization(_episode(entry.episode_index), (), {})

    with pytest.raises(RuntimeError, match="worker interruption"):
        run_shard(root, 0, executor)
    assert calls == [0, 2]

    result = run_shard(root, 0, executor)
    assert calls == [0, 2, 2]
    assert result.recovered_count == 1
    assert result.committed_count == 1
    assert result.remaining_count == 0
    assert ".staging/workers/shard-00000" in str(
        EpisodeWriter(root, {"seed": 2}, resume=True, worker_id="shard-00000")._transaction_dir(
            _uuid(2)
        )
    )

    failed_calls = 0

    def exhausted_construction(_entry):
        nonlocal failed_calls
        failed_calls += 1
        raise EpisodeAttemptFailure(
            "asset admission exhausted",
            terminal_stage=TERMINAL_FAILURE_STAGE,
            evidence={"attempt_seeds": [31, 32], "admitted": False},
        )

    failed = run_shard(root, 1, exhausted_construction, max_episodes=1)
    assert failed.failed_count == 1
    assert run_shard(root, 1, exhausted_construction, max_episodes=1).failed_count == 2
    assert failed_calls == 2
    assert run_shard(root, 1, exhausted_construction, max_episodes=1).failed_count == 2
    assert failed_calls == 2
    assert {
        record.episode_uuid: record.episode_index
        for record in EpisodeWriter(root, {"seed": 2}, resume=True).records()
    } == load_run_plan(root).membership


def test_outcome_mismatch_is_materialized_and_runtime_exception_semantics_reject_misuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="completed outcome mismatches"):
        EpisodeAttemptFailure(
            "runtime mismatch",
            terminal_stage="runtime_physics",
            evidence={"actual_outcome": "near_miss"},
        )

    root = tmp_path / "outcome-mismatch"
    plan_run(root, {"seed": 18}, _declarations(1), shard_count=1)

    def marker_only_write(self: EpisodeWriter, record: EpisodeRecord, **_kwargs: object) -> EpisodeRecord:
        return _publish_marker(self, record)

    monkeypatch.setattr(EpisodeWriter, "write_episode", marker_only_write)
    mismatch = _episode(0)
    mismatch.actual_outcome = "near_miss"
    mismatch.task_success = False
    mismatch.failure_mode = "spatial_near_miss"
    mismatch.primary_failure_code = "spatial_near_miss"
    mismatch.actual_outcome_class = ActualOutcomeClass.NEAR_MISS

    result = run_shard(
        root,
        0,
        lambda _entry: EpisodeMaterialization(mismatch, (), {}),
    )
    assert result.committed_count == 1 and result.failed_count == 0
    persisted = EpisodeWriter(root, {"seed": 18}, resume=True).records()[0]
    assert persisted.actual_outcome == "near_miss"
    assert "terminal_failure" not in persisted.extras


def test_terminal_failure_finalizes_with_hash_bound_ledger_and_blocks_qc(
    tmp_path: Path,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "terminal-finalize"
    plan = plan_run(root, {"seed": 19}, _declarations(1), shard_count=1)

    def exhausted(_entry):
        raise EpisodeAttemptFailure(
            "RoboCasa asset attempts exhausted",
            terminal_stage=TERMINAL_FAILURE_STAGE,
            evidence={"attempt_seeds": [101, 102, 103], "license_available": True},
        )

    result = run_shard(root, 0, exhausted)
    assert result.failed_count == 1 and result.remaining_count == 0
    records = finalize_run(root, DatasetInfo(name="terminal-failure-test"))
    assert {record.episode_uuid: record.episode_index for record in records} == plan.membership
    assert records[0].release_eligible is False
    assert records[0].video_paths == {} and records[0].content_hashes == {}

    ledger_path = root / TERMINAL_FAILURE_LEDGER_FILE
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["terminal_failure_count"] == 1
    assert ledger["pilot_readiness_blocked"] is True
    failure = ledger["terminal_failures"][0]
    assert failure["evidence_sha256"]
    assert sha256_file(root / failure["receipt_path"]) == failure["receipt_sha256"]

    provenance = pq.read_table(root / "meta" / "provenance.parquet").to_pylist()
    binding = next(
        row
        for row in provenance
        if row["schema_version"] == "dynamic-robot-terminal-failure-provenance/v1"
    )
    assert binding["artifact_sha256"] == sha256_file(ledger_path)
    assert binding["pilot_readiness_blocked"] is True
    assert (root / ".seal.json").is_file()

    qc = validate_dataset(root, strict_all=True, write_reports=False)
    assert qc.passed is False


def test_terminal_receipt_tampering_prevents_seal(tmp_path: Path) -> None:
    root = tmp_path / "terminal-tamper"
    plan_run(root, {"seed": 20}, _declarations(1), shard_count=1)

    def exhausted(_entry):
        raise EpisodeAttemptFailure(
            "scene build exhausted",
            terminal_stage=TERMINAL_FAILURE_STAGE,
            evidence={"attempts": 4},
        )

    run_shard(root, 0, exhausted)
    receipt = next((root / ".shards").glob("shard-*/attempts/*.json"))
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["evidence_sha256"] = "0" * 64
    receipt.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RuntimeError, match="receipt or evidence binding differs"):
        finalize_run(root, DatasetInfo(name="tampered-terminal"))
    assert not (root / ".seal.json").exists()


def test_finalize_requires_exact_membership_seals_and_hashes_all_metadata(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "run"
    config = {"seed": 3}
    plan = plan_run(root, config, _declarations(2), shard_count=2)
    writer = EpisodeWriter(root, config, resume=True, worker_id="test-publisher")
    first = _episode(0)
    _add_canonical_artifacts(root, first)
    _publish_marker(writer, first)
    info = DatasetInfo(name="orchestration-test")

    with pytest.raises(RuntimeError, match="partial or changed plan membership"):
        finalize_run(root, info)
    assert not (root / ".seal.json").exists()

    second = _episode(1)
    _add_canonical_artifacts(root, second)
    _publish_marker(writer, second)
    records = finalize_run(root, info)
    assert [record.episode_index for record in records] == [0, 1]
    seal = json.loads((root / ".seal.json").read_text(encoding="utf-8"))
    assert seal["run_plan_sha256"]
    assert seal["expected_episode_membership"] == [
        {"episode_uuid": episode_uuid, "episode_index": index}
        for episode_uuid, index in sorted(plan.membership.items())
    ]
    completion = json.loads((root / "meta" / ".complete.json").read_text(encoding="utf-8"))
    assert set(completion["content_hashes"]) == FINALIZED_METADATA_ARTIFACTS
    assert completion["seal_sha256"]
    episode_schema = pq.ParquetFile(root / "meta" / "episodes.parquet").schema_arrow
    assert episode_schema.metadata[b"contract"] == b"dynamic-robot-episodes/v2"
    assert str(episode_schema.field("episode_index").type) == "int64"
    assert str(episode_schema.field("failure_tags").type) == "list<element: string>"

    resumed = EpisodeWriter(root, config, resume=True)
    with pytest.raises(DatasetSealedError):
        resumed.write_episode(_episode(2), frame_rows=(), videos={})
    # A semantic resume may reconstruct volatile DatasetInfo UUID/time values.
    assert len(finalize_run(root, DatasetInfo(name="orchestration-test"))) == 2


def test_exclusive_finalize_waits_for_an_active_episode_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "locked-run"
    config = {"seed": 88}
    writer = EpisodeWriter(root, config)
    _publish_marker(writer, _episode(0))
    entered = threading.Event()
    release = threading.Event()
    finalized = threading.Event()
    errors: list[BaseException] = []

    def slow_publish(self: EpisodeWriter, record: EpisodeRecord, **_kwargs: object) -> EpisodeRecord:
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test publication was never released")
        return _publish_marker(self, record)

    monkeypatch.setattr(EpisodeWriter, "_write_episode_unlocked", slow_publish)

    def publish() -> None:
        try:
            writer.write_episode(_episode(1), frame_rows=(), videos={})
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def seal() -> None:
        try:
            EpisodeWriter(root, config, resume=True, worker_id="finalizer-test").finalize(
                DatasetInfo(name="lock-test"),
                expected_episode_membership={_uuid(0): 0, _uuid(1): 1},
            )
            finalized.set()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    publisher = threading.Thread(target=publish)
    finalizer = threading.Thread(target=seal)
    publisher.start()
    assert entered.wait(timeout=2)
    finalizer.start()
    assert not finalized.wait(timeout=0.05)
    release.set()
    publisher.join(timeout=5)
    finalizer.join(timeout=5)
    assert not publisher.is_alive() and not finalizer.is_alive()
    assert errors == []
    assert finalized.is_set()
    assert (root / ".seal.json").is_file()
