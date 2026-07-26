from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from dynamic_robot_dataset.common.corpus_registry import load_backend_capability_registry
from dynamic_robot_dataset.common.review import (
    STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA,
    bind_review_artifacts_strict,
    event_strip_frame_indices,
)
from dynamic_robot_dataset.common.review_suite import write_review_suite_bundle
from dynamic_robot_dataset.common.source_scenario import (
    CounterfactualIdentity,
    EmbodimentSpec,
    PoseSpec,
    RoboCasaAssetManifest,
    SourceCameraSpec,
    SourceScenarioSpec,
)


def _scenario_for_clean_p0_case(bundle) -> SourceScenarioSpec:
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
            "evaluator": case.evaluator,
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


def _strict_binding_inputs(tmp_path: Path):
    bundle = write_review_suite_bundle(tmp_path / "suite")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    scenario = _scenario_for_clean_p0_case(bundle)
    case = next(value for value in bundle.plan.cases if value.case_id == scenario.scenario_id)

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

    qc_path = dataset / "qc/dataset_report.json"
    qc_path.parent.mkdir(parents=True)
    qc_path.write_text(
        json.dumps(
            {
                "schema_version": "dynamic-robot-qc-report/v2",
                "strict_all": True,
                "passed": True,
                "global_failures": [],
                "episodes": [
                    {
                        "episode_uuid": case.episode_uuid,
                        "episode_index": case.episode_index,
                        "passed": True,
                        "metrics": {
                            "objective_recompute": {
                                "evaluator_id": case.evaluator,
                                "evidence_version": "objective-evidence/v1",
                                "evidence_hash": "a" * 64,
                                "key_event_name": "freefall_midpoint",
                                "key_event_time_s": 0.5,
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

    video_paths = {
        "main": "videos/main.mp4",
        "secondary": "videos/secondary.mp4",
    }
    event_strip_paths = {
        "main": "reviews/main.png",
        "secondary": "reviews/secondary.png",
    }
    for relative in (*video_paths.values(), *event_strip_paths.values()):
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    return {
        "bundle": bundle,
        "dataset": dataset,
        "scenario": scenario,
        "scenario_path": scenario_path,
        "source_manifest_path": source_manifest_path,
        "qc_path": qc_path,
        "video_paths": video_paths,
        "event_strip_paths": event_strip_paths,
    }


def _bind(value):
    return bind_review_artifacts_strict(
        value["dataset"],
        review_suite_root=value["bundle"].root,
        source_scenario_spec_path=value["scenario_path"],
        source_manifest_path=value["source_manifest_path"],
        qc_report_path=value["qc_path"],
        video_paths=value["video_paths"],
        event_strip_paths=value["event_strip_paths"],
        frame_timestamps_s=[index / 30 for index in range(31)],
    )


def test_event_strip_rejects_missing_context_and_duplicate_clamping() -> None:
    timestamps = [index / 30 for index in range(31)]
    with pytest.raises(ValueError, match="0.1 s.*pre-event"):
        event_strip_frame_indices(timestamps, 0.05)
    with pytest.raises(ValueError, match="0.3 s.*post-event"):
        event_strip_frame_indices(timestamps, 0.8)
    with pytest.raises(ValueError, match="duplicate, clamped"):
        event_strip_frame_indices([0.0, 0.1, 0.2, 0.5], 0.1)


def test_strict_binder_derives_every_identity_and_event_time(tmp_path: Path) -> None:
    value = _strict_binding_inputs(tmp_path)
    artifact = _bind(value)
    case = next(
        item
        for item in value["bundle"].plan.cases
        if item.case_id == value["scenario"].scenario_id
    )
    request = next(
        item
        for item in value["bundle"].requests.requests
        if item.case_id == case.case_id
    )

    assert artifact.episode_uuid == case.episode_uuid
    assert artifact.leaf_id == case.corpus_leaf_id
    assert artifact.rollout_index == case.rollout_index
    assert artifact.fixed_master_seed == value["bundle"].plan.fixed_master_seed
    assert artifact.review_plan_sha256 == value["bundle"].plan.plan_sha256
    assert artifact.review_case_sha256 == case.case_sha256
    assert artifact.review_request_sha256 == request.request_sha256
    assert artifact.scenario_spec_sha256 == value["scenario"].spec_hash
    assert artifact.key_event_time_s == 0.5
    assert artifact.event_strip_indices == {
        "pre_event": 12,
        "event": 15,
        "post_0p1_s": 18,
        "post_0p3_s": 24,
        "final": 30,
    }


def test_strict_binder_uses_episode_qc_when_another_leaf_fails(
    tmp_path: Path,
) -> None:
    value = _strict_binding_inputs(tmp_path)
    qc = json.loads(value["qc_path"].read_text(encoding="utf-8"))
    qc["passed"] = False
    qc["episodes"].append(
        {
            "episode_uuid": "00000000-0000-4000-8000-000000000001",
            "episode_index": 6,
            "passed": False,
            "hard_failures": ["failure from another leaf"],
        }
    )
    value["qc_path"].write_text(
        json.dumps(qc, sort_keys=True), encoding="utf-8"
    )

    artifact = _bind(value)

    assert artifact.automated_qc_passed is True


def test_strict_binder_keeps_dataset_global_failures_fail_closed(
    tmp_path: Path,
) -> None:
    value = _strict_binding_inputs(tmp_path)
    qc = json.loads(value["qc_path"].read_text(encoding="utf-8"))
    qc["passed"] = False
    qc["global_failures"] = ["finalized metadata binding changed"]
    value["qc_path"].write_text(
        json.dumps(qc, sort_keys=True), encoding="utf-8"
    )

    artifact = _bind(value)

    assert artifact.automated_qc_passed is False


def test_strict_binder_rejects_missing_global_failure_evidence(
    tmp_path: Path,
) -> None:
    value = _strict_binding_inputs(tmp_path)
    qc = json.loads(value["qc_path"].read_text(encoding="utf-8"))
    del qc["global_failures"]
    value["qc_path"].write_text(
        json.dumps(qc, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="global failure evidence"):
        _bind(value)


def test_strict_binder_rejects_replacement_seed_source_pin_and_evaluator(
    tmp_path: Path,
) -> None:
    value = _strict_binding_inputs(tmp_path)
    scenario = value["scenario"]
    replacement_seed = replace(
        scenario.rng_subseeds,
        physics=scenario.rng_subseeds.physics + 1,
    )
    value["scenario_path"].write_text(
        replace(scenario, rng_subseeds=replacement_seed).to_json(indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rng_subseeds"):
        _bind(value)

    value["scenario_path"].write_text(scenario.to_json(indent=2), encoding="utf-8")
    source_manifest = json.loads(
        value["source_manifest_path"].read_text(encoding="utf-8")
    )
    source_manifest["source_hashes"]["external_dependency_manifest"] = "b" * 64
    value["source_manifest_path"].write_text(
        json.dumps(source_manifest, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source pins"):
        _bind(value)

    value["source_manifest_path"].write_text(
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
    qc = json.loads(value["qc_path"].read_text(encoding="utf-8"))
    qc["episodes"][0]["metrics"]["objective_recompute"]["evaluator_id"] = (
        "replacement-evaluator"
    )
    value["qc_path"].write_text(json.dumps(qc, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="different evaluator"):
        _bind(value)
