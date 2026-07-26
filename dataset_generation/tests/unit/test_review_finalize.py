from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
import shutil

import pytest

from dynamic_robot_dataset.common.hashing import sha256_file, sha256_json
from dynamic_robot_dataset.common.paths import ExistingOutputError
from dynamic_robot_dataset.common.review import (
    HumanReviewLedger,
    REVIEW_CHECKS,
    REVIEW_SCENE_SEQUENCE,
    ReviewArtifactManifest,
    ReviewItem,
)
from dynamic_robot_dataset.common import review_finalize


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _artifact(index: int) -> ReviewArtifactManifest:
    return ReviewArtifactManifest(
        leaf_id="F2b",
        episode_uuid=f"00000000-0000-4000-8000-{index:012d}",
        rollout_index=index,
        fixed_master_seed=20260717,
        review_plan_sha256=HEX_A,
        review_case_sha256=HEX_B,
        review_request_ledger_sha256=HEX_C,
        review_request_sha256=HEX_A,
        scenario_spec_sha256=HEX_B,
        qc_report_sha256=HEX_C,
        qc_report_schema="dynamic-robot-qc-report/v2",
        qc_episode_result_sha256=HEX_A,
        qc_strict_all=True,
        automated_qc_passed=True,
        source_manifest_sha256=HEX_B,
        frame_timestamps_sha256=HEX_C,
        key_event_time_s=0.5,
        video_paths={
            "main": f"videos/main/{index}.mp4",
            "secondary": f"videos/secondary/{index}.mp4",
        },
        video_sha256={"main": HEX_A, "secondary": HEX_B},
        event_strip_paths={
            "main": f"reviews/main/{index}.png",
            "secondary": f"reviews/secondary/{index}.png",
        },
        event_strip_sha256={"main": HEX_B, "secondary": HEX_C},
        event_strip_indices={
            "pre_event": 3,
            "event": 6,
            "post_0p1_s": 9,
            "post_0p3_s": 15,
            "final": 29,
        },
    )


def _pending() -> HumanReviewLedger:
    return HumanReviewLedger(
        items=tuple(
            ReviewItem(index, scene, _artifact(index), True)
            for index, scene in enumerate(REVIEW_SCENE_SEQUENCE)
        ),
        review_plan_sha256=HEX_A,
        review_request_ledger_sha256=HEX_C,
        extras={"state": "pending_human_review", "activation_forbidden": True},
    )


def _materialized_pending(dataset: Path) -> HumanReviewLedger:
    qc_path = dataset / "qc/dataset_report.json"
    _write_json(
        qc_path,
        {
            "strict_all": True,
            "passed": True,
            "episode_count": 6,
            "failed_episode_count": 0,
            "release_eligible_count": 6,
        },
    )
    items: list[ReviewItem] = []
    for index, scene in enumerate(REVIEW_SCENE_SEQUENCE):
        case_id = f"F2b-review-{index:02d}"
        video_paths = {
            "main": f"videos/main/{index}.mp4",
            "secondary": f"videos/secondary/{index}.mp4",
        }
        strip_paths = {
            "main": f"reviews/strips/main/{index}.png",
            "secondary": f"reviews/strips/secondary/{index}.png",
        }
        for relative in (*video_paths.values(), *strip_paths.values()):
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode("utf-8"))
        source_manifest = dataset / f"reviews/source_manifests/{case_id}.json"
        source_spec = dataset / f"reviews/source_specs/{case_id}.json"
        _write_json(source_manifest, {"case_id": case_id, "kind": "manifest"})
        _write_json(source_spec, {"case_id": case_id, "kind": "scenario"})
        artifact = replace(
            _artifact(index),
            qc_report_sha256=sha256_file(qc_path),
            scenario_spec_sha256=sha256_json(
                {"case_id": case_id, "kind": "scenario"}
            ),
            source_manifest_sha256=sha256_file(source_manifest),
            video_paths=video_paths,
            video_sha256={
                view: sha256_file(dataset / relative)
                for view, relative in video_paths.items()
            },
            event_strip_paths=strip_paths,
            event_strip_sha256={
                view: sha256_file(dataset / relative)
                for view, relative in strip_paths.items()
            },
        )
        artifact_path = dataset / f"reviews/artifacts/F2b/{case_id}.json"
        _write_json(
            artifact_path,
            {**asdict(artifact), "binding_sha256": artifact.binding_sha256},
        )
        items.append(ReviewItem(index, scene, artifact, True))
    return HumanReviewLedger(
        items=tuple(items),
        review_plan_sha256=HEX_A,
        review_request_ledger_sha256=HEX_C,
        extras={"state": "pending_human_review", "activation_forbidden": True},
    )


def _decision_document(ledger: HumanReviewLedger) -> dict[str, object]:
    return {
        "schema_version": review_finalize.REVIEW_DECISIONS_SCHEMA,
        "reviewer": "reviewer@example.org",
        "reviewed_at": "2026-07-20T16:00:00+00:00",
        "items": [
            {
                "leaf_id": item.artifacts.leaf_id,
                "rollout_index": item.rollout_index,
                "artifact_binding_sha256": item.artifacts.binding_sha256,
                "checks": {name: True for name in REVIEW_CHECKS},
                "notes": "reviewed both synchronized views and event strips",
            }
            for item in ledger.items
        ],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _catalog_module():
    script = Path(__file__).resolve().parents[2] / "tools" / "catalog_datasets.py"
    spec = importlib.util.spec_from_file_location("review_catalog_tool", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decisions_require_exact_membership_checks_notes_and_binding(
    tmp_path: Path,
) -> None:
    ledger = _pending()
    decision_path = tmp_path / "decisions.json"
    value = _decision_document(ledger)
    _write_json(decision_path, value)
    _, _, decisions, _ = review_finalize._load_decisions(decision_path, ledger)
    assert len(decisions) == 6

    incomplete = dict(value)
    incomplete["items"] = list(value["items"])[:-1]
    _write_json(tmp_path / "incomplete.json", incomplete)
    with pytest.raises(ValueError, match="cover every pending rollout"):
        review_finalize._load_decisions(tmp_path / "incomplete.json", ledger)

    duplicate = dict(value)
    duplicate["items"] = [*list(value["items"]), list(value["items"])[0]]
    _write_json(tmp_path / "duplicate.json", duplicate)
    with pytest.raises(ValueError, match="duplicate human-review decision"):
        review_finalize._load_decisions(tmp_path / "duplicate.json", ledger)

    tampered = json.loads(json.dumps(value))
    tampered["items"][0]["artifact_binding_sha256"] = HEX_A
    _write_json(tmp_path / "tampered.json", tampered)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        review_finalize._load_decisions(tmp_path / "tampered.json", ledger)


def test_external_review_is_immutable_catalogued_and_never_releases_review_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "sealed-review-run"
    _write_json(dataset / ".seal.json", {"sealed": True})
    _write_json(dataset / ".run_plan.json", {"episodes": ["episode-0"]})
    _write_json(dataset / "meta/.complete.json", {"complete": True})
    _write_json(dataset / "meta/info.json", {"schema_version": "fixture/v1"})
    _write_json(
        dataset / "qc/dataset_report.json",
        {
            "passed": True,
            "episode_count": 6,
            "failed_episode_count": 0,
            "release_eligible_count": 6,
        },
    )
    pending_path = dataset / "reviews/human_review_ledger.pending.json"
    ledger = _materialized_pending(dataset)
    _write_json(pending_path, ledger.to_dict())
    marker = dataset / "sealed-bytes.bin"
    marker.write_bytes(b"must remain untouched")
    before = sha256_file(marker)
    seal_sha256 = sha256_file(dataset / ".seal.json")
    monkeypatch.setattr(
        review_finalize,
        "_verify_finalized_dataset",
        lambda _root: {
            "schema_version": review_finalize.DATASET_REVIEW_BINDING_SCHEMA,
            "dataset_name": dataset.name,
            "dataset_path": str(dataset),
            "dataset_seal_sha256": seal_sha256,
            "dataset_seal_payload_sha256": HEX_A,
            "run_plan_sha256": sha256_file(dataset / ".run_plan.json"),
            "finalized_completion_sha256": sha256_file(
                dataset / "meta/.complete.json"
            ),
            "finalized_metadata_sha256": {
                "meta/info.json": sha256_file(dataset / "meta/info.json")
            },
        },
    )
    monkeypatch.setattr(
        review_finalize,
        "_load_pending_ledger",
        lambda _root: (
            pending_path,
            ledger,
            {
                item.artifacts.binding_sha256: f"F2b-review-{item.rollout_index:02d}"
                for item in ledger.items
            },
        ),
    )
    decisions = tmp_path / "decisions.json"
    _write_json(decisions, _decision_document(ledger))
    with pytest.raises(ValueError, match="cannot be inside the sealed dataset"):
        review_finalize.finalize_human_review(
            dataset, decisions, dataset / "external-reviews"
        )
    output_root = tmp_path / "dataset_reviews"
    result = review_finalize.finalize_human_review(dataset, decisions, output_root)
    assert result.status == "approved"
    assert result.approved_leaf_ids == ("F2b",)
    validated = review_finalize.validate_external_review_publication(
        Path(result.output_path) / "activation_report.json", corpus_id="F2b"
    )
    assert validated.status == "approved"
    assert validated.report_sha256 == json.loads(
        (Path(result.output_path) / "activation_report.json").read_text(
            encoding="utf-8"
        )
    )["report_sha256"]
    assert sha256_file(marker) == before
    assert not (dataset / "human_review_ledger.json").exists()
    with pytest.raises(ExistingOutputError):
        review_finalize.finalize_human_review(dataset, decisions, output_root)

    catalog = _catalog_module()
    index, errors = catalog._review_publication_index(output_root)
    assert errors == []
    record = catalog._canonical_run(dataset, index)
    assert record["human_review"] == "approved"
    assert record["approved_leaf_ids"] == "F2b"
    assert record["release_eligible"] is False
    assert record["category"] == "runs/sealed_qc_pass_human_approved"

    publication_manifest = Path(result.output_path) / "manifest.json"
    original_manifest = publication_manifest.read_bytes()
    mutable_manifest = json.loads(original_manifest)
    mutable_manifest["status"] = "failed"
    _write_json(publication_manifest, mutable_manifest)
    poisoned_index, poisoned_errors = catalog._review_publication_index(output_root)
    assert poisoned_errors
    assert poisoned_index[seal_sha256]["status"] == "failed"
    with pytest.raises(ValueError, match="manifest verdict"):
        review_finalize.validate_external_review_publication(
            Path(result.output_path) / "activation_report.json", corpus_id="F2b"
        )
    publication_manifest.write_bytes(original_manifest)

    reviewed_video = dataset / "videos/main/0.mp4"
    original_video = reviewed_video.read_bytes()
    reviewed_video.write_bytes(b"post-review tampering")
    tampered = catalog._canonical_run(dataset, index)
    assert tampered["human_review"] == "failed"
    assert tampered["release_eligible"] is False
    assert "no longer matches" in tampered["notes"]
    reviewed_video.write_bytes(original_video)

    shutil.copytree(Path(result.output_path), output_root / "duplicate-publication")
    duplicate_index, duplicate_errors = catalog._review_publication_index(output_root)
    assert duplicate_errors
    assert duplicate_index[seal_sha256]["status"] == "failed"


def test_persisted_ledger_rejects_derived_hash_tampering() -> None:
    value = _pending().to_dict()
    value["extras"]["state"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        HumanReviewLedger.from_dict(value)
