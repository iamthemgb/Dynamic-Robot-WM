"""Positive and adversarial tests for release-gate artifact binding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_robot_dataset.common import readiness
from dynamic_robot_dataset.common.hashing import sha256_file, sha256_json
from dynamic_robot_dataset.common.schema import EpisodeRecord


def _record(index: int, *, native: bool = True) -> EpisodeRecord:
    return EpisodeRecord(
        episode_uuid=f"70000000-0000-4000-8000-{index:012d}",
        episode_index=index,
        counterfactual_bundle_id=f"bundle-{index}",
        split_group_id=f"split-{index}",
        scene_seed=index,
        branch_seed=index,
        family="falling_catch",
        subfamily="centered_vertical_drop",
        intended_branch="success_seeking",
        actual_outcome="success",
        task_success=True,
        failure_mode="none",
        source_generator="readiness-test",
        source_generator_version="1",
        config_hash="a" * 64,
        simulator_name="mujoco" if native else "diagnostic-surrogate",
        simulator_version="3.3.0" if native else "1",
        renderer="mujoco",
        video_paths={
            "observation.images.main": f"videos/main/{index}.mp4",
            "observation.images.secondary": f"videos/secondary/{index}.mp4",
        },
        randomization={"background_style": "clean_franka_lab"},
        extras={
            "native_mujoco": native,
            "backend_provenance": {
                "backend": "native_mujoco" if native else "diagnostic_quarantine",
                "scenario_hash": "d" * 64 if native else None,
            },
        },
    )


def _acceptance_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path, list[EpisodeRecord]]:
    root = tmp_path / "acceptance"
    (root / "meta").mkdir(parents=True)
    (root / ".suite_attempts").mkdir()
    for name in (".complete.json", "info.json", "episodes.parquet"):
        (root / "meta" / name).write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "mini_acceptance.yaml"
    config_path.write_text(
        "schema_version: dynamic-robot-suite/v1\n"
        "name: mini_acceptance\n"
        "requirements: {require_measured_success_and_failure: []}\n",
        encoding="utf-8",
    )
    records = [_record(index) for index in range(4)]
    cases = [
        {
            "suite_name": "mini_acceptance",
            "case_index": index,
            "category_id": "rigid",
            "views": ["main", "secondary"],
            "scene_style": "clean_franka_lab",
        }
        for index in range(4)
    ]
    writer_hash = "b" * 64
    plan = {
        "schema_version": "dynamic-robot-suite-plan-ledger/v1",
        "suite_name": "mini_acceptance",
        "source_config_sha256": sha256_file(config_path),
        "planned_case_count": 4,
        "execution_backend_counts": {"native_mujoco": 4},
        "full_native": True,
        "writer_config_hash": writer_hash,
        "counterfactual_families": [],
        "planned_episodes": [
            {
                "case": case,
                "episode_uuid": record.episode_uuid,
                "execution_backend": "native_mujoco",
                "scenario_spec_hash": "d" * 64,
                "episode_plan": {
                    "family": record.family,
                    "intended_branch": record.intended_branch,
                },
            }
            for case, record in zip(cases, records)
        ],
    }
    plan_path = root / ".suite_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    (root / ".generation.json").write_text(
        json.dumps({"config_hash": writer_hash}), encoding="utf-8"
    )
    for index, record in enumerate(records):
        (root / ".suite_attempts" / f"case-{index:06d}.json").write_text(
            json.dumps(
                {
                    "episode_uuid": record.episode_uuid,
                    "case_index": index,
                    "config_hash": writer_hash,
                    "status": "committed",
                }
            ),
            encoding="utf-8",
        )
    report = {
        "schema_version": "dynamic-robot-suite-execution/v1",
        "suite_name": "mini_acceptance",
        "dataset_root": str(root),
        "writer_config_hash": writer_hash,
        "suite_plan_sha256": sha256_file(plan_path),
        "source_config_sha256": sha256_file(config_path),
        "committed_episode_uuid_set_sha256": sha256_json(
            sorted(record.episode_uuid for record in records)
        ),
        "planned_case_count": 4,
        "committed_episode_count": 4,
        "planned_execution_backend_counts": {"native_mujoco": 4},
        "committed_execution_backend_counts": {"native_mujoco": 4},
        "full_native": True,
        "passed_execution": True,
    }
    report_path = root / ".suite_execution.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(readiness, "_canonical_suite_config", lambda *_: config_path)
    monkeypatch.setattr(
        readiness,
        "expand_suite",
        lambda _: [SimpleNamespace(to_dict=lambda value=value: value) for value in cases],
    )
    monkeypatch.setattr(readiness, "plan_suite_cases", lambda _: [])
    monkeypatch.setattr(readiness, "load_episode_records", lambda _: records)
    monkeypatch.setattr(
        readiness,
        "_hard_qc_map",
        lambda *_: ({record.episode_uuid: True for record in records}, [], "c" * 64),
    )
    monkeypatch.setattr(
        readiness,
        "_negative_control_evidence",
        lambda *_: (True, {"controls": {}, "problems": []}),
    )
    monkeypatch.setattr(
        readiness, "validate_counterfactual_family_records", lambda *_args, **_kwargs: []
    )
    requirement = {
        "acceptance_suite": {
            "suite_name": "mini_acceptance",
            "branch_count": 4,
            "require_full_native": True,
            "require_passed_execution": True,
        }
    }
    return requirement, report_path.resolve(), records


def test_acceptance_recomputation_positive_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, report_path, _ = _acceptance_fixture(tmp_path, monkeypatch)
    checks, hashes = readiness._acceptance_checks(config, report_path)

    assert checks
    assert all(check.passed for check in checks), {
        check.name: check.measured for check in checks if not check.passed
    }
    assert hashes["acceptance_report_sha256"] == sha256_file(report_path)


def test_acceptance_rejects_reported_native_claim_when_record_is_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, report_path, records = _acceptance_fixture(tmp_path, monkeypatch)
    records[1] = _record(1, native=False)

    checks, _ = readiness._acceptance_checks(config, report_path)
    by_name = {check.name: check for check in checks}
    assert not by_name["acceptance_suite.backend_counts_recomputed"].passed
    assert not by_name["acceptance_suite.report_claims_match_recomputed"].passed
    assert not by_name["acceptance_suite.full_native"].passed
    assert not by_name["acceptance_suite.passed_execution"].passed


def test_acceptance_recomputes_required_sweep_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, report_path, _ = _acceptance_fixture(tmp_path, monkeypatch)
    config_path = tmp_path / "mini_acceptance.yaml"
    config_path.write_text(
        "schema_version: dynamic-robot-suite/v1\n"
        "name: mini_acceptance\n"
        "requirements:\n"
        "  require_measured_success_and_failure: []\n"
        "  require_measured_physics_sweeps:\n"
        "    restitution_sweep:\n"
        "      requested_field: effective_restitution_target\n"
        "      response_metric: measured_effective_restitution\n"
        "      sample_count_metric: restitution_measurement_sample_count\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        readiness,
        "sweep_acceptance_evidence",
        lambda *_: {
            "passed": False,
            "families": {
                "restitution_sweep": {
                    "passed": False,
                    "problems": ["measured response is not strictly increasing"],
                    "measurements": [],
                }
            },
        },
    )

    checks, _ = readiness._acceptance_checks(config, report_path)
    by_name = {check.name: check for check in checks}
    core = by_name["acceptance_suite.core_gates_recomputed"]
    assert not core.passed
    assert not core.measured["checks"][
        "required_physics_sweeps_measured_and_monotonic"
    ]
    assert not by_name["acceptance_suite.passed_execution"].passed


def test_qc_binding_verifies_episode_media_content_hashes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "qc").mkdir()
    media = root / "videos" / "main.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"first media payload")
    episodes = root / "meta" / "episodes.parquet"
    episodes.write_bytes(b"episode metadata")
    completion = root / "meta" / ".complete.json"
    metadata_hashes = {"episodes.parquet": sha256_file(episodes)}
    completion.write_text(
        json.dumps({"config_hash": "a" * 64, "content_hashes": metadata_hashes}),
        encoding="utf-8",
    )
    record = _record(0)
    record.content_hashes = {"videos/main.mp4": sha256_file(media)}
    report = {
        "schema_version": "dynamic-robot-qc-report/v2",
        "dataset_root": str(root),
        "dataset_episodes_sha256": sha256_file(episodes),
        "metadata_complete_manifest_sha256": sha256_file(completion),
        "metadata_content_hashes": metadata_hashes,
        "passed": True,
        "global_failures": [],
        "episodes": [
            {
                "episode_uuid": record.episode_uuid,
                "passed": True,
                "release_eligible": record.release_eligible,
            }
        ],
    }
    qc_path = root / "qc" / "dataset_report.json"
    qc_path.write_text(json.dumps(report), encoding="utf-8")

    _, failures, _ = readiness._hard_qc_map(root.resolve(), [record])
    assert failures == []

    media.write_bytes(b"tampered media payload")
    _, failures, _ = readiness._hard_qc_map(root.resolve(), [record])
    assert any("artifact hash mismatch" in failure for failure in failures)


def test_model_evaluation_binds_real_files_and_rejects_nonfinite_metric(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.safetensors"
    manifest = tmp_path / "evaluation.jsonl"
    model.write_bytes(b"model")
    manifest.write_text("{}\n", encoding="utf-8")
    value = {
        "metrics": {"trajectory": 0.2},
        "provenance": {
            "dataset_episodes_sha256": "1" * 64,
            "dataset_qc_report_sha256": "2" * 64,
            "model_artifact_path": str(model),
            "model_artifact_sha256": sha256_file(model),
            "evaluation_manifest_path": str(manifest),
            "evaluation_manifest_sha256": sha256_file(manifest),
        },
    }
    config = {"model_evaluation": {"trajectory": {"minimum": 0.1}}}
    checks = readiness._model_checks(
        config,
        value,
        dataset_episodes_sha256="1" * 64,
        dataset_qc_report_sha256="2" * 64,
    )
    assert all(check.passed for check in checks)

    value["metrics"]["trajectory"] = float("nan")
    checks = readiness._model_checks(
        config,
        value,
        dataset_episodes_sha256="1" * 64,
        dataset_qc_report_sha256="2" * 64,
    )
    assert not next(check for check in checks if check.name == "model.trajectory").passed


def test_prerequisite_rejects_moved_claimed_pass_report(tmp_path: Path) -> None:
    dataset = tmp_path / "prior-dataset"
    dataset.mkdir()
    forged = tmp_path / "copied-readiness.json"
    forged.write_text(
        json.dumps(
            {
                "schema_version": "dynamic-robot-readiness-report/v1",
                "gate_id": "native_10h_v1",
                "dataset_root": str(dataset),
                "passed": True,
                "checks": [
                    {"name": "minimum_unique_release_hours", "passed": True},
                    {"name": "zero_global_or_release_qc_failures", "passed": True},
                    {
                        "name": "acceptance_suite.core_gates_recomputed",
                        "passed": True,
                    },
                    {"name": "acceptance_suite.full_native", "passed": True},
                    {"name": "acceptance_suite.passed_execution", "passed": True},
                ],
                "blockers": [],
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )

    checks, _ = readiness._prerequisite_checks(
        {"gate_id": "native_10h_v1", "passed": True}, forged.resolve()
    )
    by_name = {check.name: check for check in checks}
    assert not by_name["prerequisite_gate.canonical_report_path"].passed
    assert not by_name["prerequisite_gate.artifacts_content_bound"].passed
    assert not by_name["prerequisite_gate_passed"].passed
