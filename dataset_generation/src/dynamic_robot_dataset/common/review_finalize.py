"""Immutable, external publication of completed fixed-rollout reviews."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .episode_writer import (
    FINALIZED_METADATA_ARTIFACTS,
    load_episode_records,
    read_parquet_rows,
)
from .hashing import sha256_file, sha256_json
from .paths import AtomicDirectory, atomic_write_json, portable_relative_path
from .review import (
    HumanReview,
    HumanReviewLedger,
    REVIEW_CHECKS,
    ReviewArtifactManifest,
    ReviewItem,
    ReviewMediaPackManifest,
    STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA,
    _content_bound_episode_file,
    _strict_qc_episode,
    _validate_finalized_qc_report,
)
from .source_scenario import SourceScenarioSpec


REVIEW_DECISIONS_SCHEMA = "dynamic-robot-human-review-decisions/v1"
DATASET_REVIEW_BINDING_SCHEMA = "dynamic-robot-dataset-review-binding/v1"
ACTIVATION_REPORT_SCHEMA = "dynamic-robot-leaf-activation-report/v1"
EXTERNAL_REVIEW_PUBLICATION_SCHEMA = (
    "dynamic-robot-external-review-publication/v1"
)

_SHA256_LENGTH = 64


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not canonical JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON mapping: {path}")
    return dict(value)


def _dataset_file(root: Path, relative: str) -> Path:
    normalized = portable_relative_path(relative)
    path = (root / normalized).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"review-bound artifact escapes dataset root: {relative}") from error
    if not path.is_file():
        raise ValueError(f"review-bound artifact is not a regular file: {relative}")
    return path


def _expect_hash(path: Path, expected: str, *, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, observed {actual}"
        )


def _verify_finalized_dataset(root: Path) -> dict[str, Any]:
    seal_path = _dataset_file(root, ".seal.json")
    plan_path = _dataset_file(root, ".run_plan.json")
    complete_path = _dataset_file(root, "meta/.complete.json")
    seal = _mapping_file(seal_path, label="dataset seal")
    complete = _mapping_file(complete_path, label="finalized completion marker")
    if seal.get("schema_version") != "dynamic-robot-dataset-seal/v1":
        raise ValueError("human review requires a canonical sealed dataset")
    seal_file_sha256 = sha256_file(seal_path)
    if complete.get("seal_sha256") != seal_file_sha256:
        raise ValueError("finalized completion marker is bound to a different seal")
    if complete.get("config_hash") != seal.get("config_hash"):
        raise ValueError("finalized completion marker uses a different configuration")
    run_plan_sha256 = sha256_file(plan_path)
    if seal.get("run_plan_sha256") != run_plan_sha256:
        raise ValueError("dataset seal is bound to a different run plan")
    content_hashes = complete.get("content_hashes")
    if not isinstance(content_hashes, Mapping) or set(content_hashes) != set(
        FINALIZED_METADATA_ARTIFACTS
    ):
        raise ValueError("finalized completion marker has non-canonical metadata membership")
    verified_metadata: dict[str, str] = {}
    for name, digest in sorted(content_hashes.items()):
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("finalized metadata hash map is malformed")
        relative = f"meta/{portable_relative_path(name)}"
        path = _dataset_file(root, relative)
        _expect_hash(path, digest, label=f"finalized metadata {name}")
        verified_metadata[relative] = digest
    records = load_episode_records(root)
    membership = seal.get("expected_episode_membership")
    if not isinstance(membership, Sequence) or isinstance(
        membership, (str, bytes, bytearray)
    ):
        raise ValueError("dataset seal lacks exact episode membership")
    try:
        sealed_membership = {
            (str(item["episode_uuid"]), int(item["episode_index"]))
            for item in membership
            if isinstance(item, Mapping)
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dataset seal contains malformed episode membership") from error
    actual_membership = {
        (record.episode_uuid, record.episode_index) for record in records
    }
    if (
        not records
        or len(membership) != len(sealed_membership)
        or sealed_membership != actual_membership
    ):
        raise ValueError("dataset seal membership differs from finalized episodes")
    return {
        "schema_version": DATASET_REVIEW_BINDING_SCHEMA,
        "dataset_name": root.name,
        "dataset_path": str(root),
        "dataset_seal_sha256": seal_file_sha256,
        "dataset_seal_payload_sha256": sha256_json(seal),
        "run_plan_sha256": run_plan_sha256,
        "finalized_completion_sha256": sha256_file(complete_path),
        "finalized_metadata_sha256": verified_metadata,
    }


def _artifact_files_by_binding(root: Path) -> dict[str, tuple[Path, ReviewArtifactManifest]]:
    artifact_root = root / "reviews" / "artifacts"
    if not artifact_root.is_dir():
        raise ValueError("sealed review dataset lacks bound artifact manifests")
    result: dict[str, tuple[Path, ReviewArtifactManifest]] = {}
    for path in sorted(artifact_root.glob("*/*.json")):
        raw = _mapping_file(path, label="review artifact manifest")
        artifact = ReviewArtifactManifest.from_dict(raw)
        binding = artifact.binding_sha256
        if binding in result:
            raise ValueError(f"duplicate review artifact binding: {binding}")
        result[binding] = (path, artifact)
    return result


def _verify_review_artifact(
    root: Path,
    expected: ReviewArtifactManifest,
    artifact_file: Path,
    persisted: ReviewArtifactManifest,
    *,
    qc_report: Mapping[str, Any],
    record_by_uuid: Mapping[str, Any],
    seal_path: Path,
    completion_path: Path,
) -> str:
    if persisted != expected:
        raise ValueError("pending ledger artifact differs from its published manifest")
    case_id = artifact_file.stem
    if artifact_file.parent.name != expected.leaf_id:
        raise ValueError("review artifact path disagrees with its leaf identity")
    source_spec = _dataset_file(root, f"reviews/source_specs/{case_id}.json")
    source_manifest = _dataset_file(root, f"reviews/source_manifests/{case_id}.json")
    scenario = SourceScenarioSpec.from_dict(
        _mapping_file(source_spec, label=f"source scenario {case_id}")
    )
    if scenario.spec_hash != expected.scenario_spec_sha256:
        raise ValueError(f"source scenario semantic hash mismatch for {case_id}")
    if scenario.scenario_id != case_id or scenario.corpus_leaf_id != expected.leaf_id:
        raise ValueError(f"source scenario identity mismatch for {case_id}")
    source_manifest_value = _mapping_file(
        source_manifest, label=f"source manifest {case_id}"
    )
    _expect_hash(
        source_manifest,
        expected.source_manifest_sha256,
        label=f"source manifest {case_id}",
    )
    if (
        source_manifest_value.get("schema_version")
        != STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA
        or source_manifest_value.get("case_id") != case_id
        or source_manifest_value.get("episode_uuid") != expected.episode_uuid
        or source_manifest_value.get("backend") != scenario.backend
        or source_manifest_value.get("review_plan_sha256")
        != expected.review_plan_sha256
        or source_manifest_value.get("review_case_sha256")
        != expected.review_case_sha256
        or source_manifest_value.get("source_scenario_spec_sha256")
        != scenario.spec_hash
        or source_manifest_value.get("source_hashes") != dict(scenario.source_hashes)
        or source_manifest_value.get("source_hashes_sha256")
        != sha256_json(dict(scenario.source_hashes))
    ):
        raise ValueError(f"source manifest semantic binding mismatch for {case_id}")
    qc = _dataset_file(root, "qc/dataset_report.json")
    _expect_hash(qc, expected.qc_report_sha256, label="strict-all QC report")
    record = record_by_uuid.get(expected.episode_uuid)
    if record is None:
        raise ValueError(f"review episode is absent from finalized metadata: {case_id}")
    if (
        record.extras.get("source_scenario_spec") != scenario.to_dict()
        or record.extras.get("source_scenario_spec_sha256") != scenario.spec_hash
        or record.extras.get("review_case_sha256") != expected.review_case_sha256
    ):
        raise ValueError(f"finalized episode identity mismatch for {case_id}")
    evaluator_id = str(scenario.physics.get("evaluator") or "")
    episode_qc, key_event_time_s = _strict_qc_episode(
        qc_report,
        episode_uuid=expected.episode_uuid,
        dataset_episode_index=record.episode_index,
        evaluator_id=evaluator_id,
    )
    automated_qc_passed = bool(
        episode_qc.get("passed") is True and not qc_report.get("global_failures")
    )
    if (
        expected.qc_episode_result_sha256 != sha256_json(episode_qc)
        or expected.key_event_time_s != key_event_time_s
        or expected.automated_qc_passed != automated_qc_passed
    ):
        raise ValueError(f"review artifact QC claim mismatch for {case_id}")
    for view, relative in expected.video_paths.items():
        record_key = f"observation.images.{view}"
        if record.video_paths.get(record_key) != relative:
            raise ValueError(f"{case_id} {view} video is not the finalized stream")
        _expect_hash(
            _content_bound_episode_file(
                root, record, relative, label=f"{case_id} finalized {view} video"
            ),
            expected.video_sha256[view],
            label=f"{case_id} {view} review video",
        )
    for view, relative in expected.event_strip_paths.items():
        _expect_hash(
            _dataset_file(root, relative),
            expected.event_strip_sha256[view],
            label=f"{case_id} {view} event strip",
        )
    pack_path = _dataset_file(
        root,
        f"reviews/event_strips/{expected.leaf_id}/{case_id}/media_pack.json",
    )
    pack = ReviewMediaPackManifest.from_dict(
        _mapping_file(pack_path, label="review media pack")
    )
    if record.frame_data_path is None:
        raise ValueError(f"finalized review episode lacks frame data: {case_id}")
    frame_path = _content_bound_episode_file(
        root, record, record.frame_data_path, label=f"{case_id} finalized frame data"
    )
    timestamps = tuple(
        float(row["timestamp"]) for row in read_parquet_rows(frame_path)
    )
    if (
        pack.episode_uuid != expected.episode_uuid
        or pack.leaf_id != expected.leaf_id
        or pack.case_id != case_id
        or pack.dataset_seal_sha256 != sha256_file(seal_path)
        or pack.finalized_completion_sha256 != sha256_file(completion_path)
        or pack.finalized_episode_record_sha256 != sha256_json(record.to_dict())
        or pack.dataset_episode_index != record.episode_index
        or pack.frame_data_path != record.frame_data_path
        or pack.frame_data_sha256 != sha256_file(frame_path)
        or pack.frame_timestamps_s != timestamps
        or pack.qc_report_path != "qc/dataset_report.json"
        or pack.qc_report_sha256 != expected.qc_report_sha256
        or pack.qc_episode_result_sha256 != expected.qc_episode_result_sha256
        or pack.key_event_time_s != expected.key_event_time_s
        or dict(pack.video_sha256) != dict(expected.video_sha256)
        or dict(pack.video_paths) != dict(expected.video_paths)
        or dict(pack.event_strip_sha256) != dict(expected.event_strip_sha256)
        or dict(pack.event_strip_paths) != dict(expected.event_strip_paths)
        or dict(pack.event_strip_indices) != dict(expected.event_strip_indices)
        or pack.frame_timestamps_sha256 != expected.frame_timestamps_sha256
    ):
        raise ValueError("review media pack differs from the artifact shown to the reviewer")
    return case_id


def _load_pending_ledger(root: Path) -> tuple[Path, HumanReviewLedger, dict[str, str]]:
    path = _dataset_file(root, "reviews/human_review_ledger.pending.json")
    ledger = HumanReviewLedger.from_dict(
        _mapping_file(path, label="pending human-review ledger")
    )
    if ledger.extras.get("state") != "pending_human_review":
        raise ValueError("input review ledger is not in pending-human-review state")
    if any(item.review is not None for item in ledger.items):
        raise ValueError("pending review ledger already contains a human decision")
    leaves = tuple(sorted({item.artifacts.leaf_id for item in ledger.items}))
    ledger.validate(required_leaf_ids=leaves)
    if ledger.extras.get("activation_forbidden") is not True:
        raise ValueError("pending review ledger does not forbid premature activation")
    if tuple(sorted(ledger.extras.get("complete_leaf_ids") or ())) != leaves:
        raise ValueError("pending review ledger complete-leaf membership is inconsistent")
    seal_path = _dataset_file(root, ".seal.json")
    completion_path = _dataset_file(root, "meta/.complete.json")
    records = load_episode_records(root)
    record_by_uuid = {record.episode_uuid: record for record in records}
    if len(record_by_uuid) != len(records):
        raise ValueError("finalized dataset contains duplicate episode UUIDs")
    qc_path = _dataset_file(root, "qc/dataset_report.json")
    qc_report = _mapping_file(qc_path, label="strict-all QC report")
    _validate_finalized_qc_report(
        root,
        qc_report,
        completion_path=completion_path,
        records=records,
    )
    available = _artifact_files_by_binding(root)
    case_ids: dict[str, str] = {}
    for item in ledger.items:
        binding = item.artifacts.binding_sha256
        match = available.get(binding)
        if match is None:
            raise ValueError(f"pending ledger artifact is not published: {binding}")
        case_ids[binding] = _verify_review_artifact(
            root,
            item.artifacts,
            match[0],
            match[1],
            qc_report=qc_report,
            record_by_uuid=record_by_uuid,
            seal_path=seal_path,
            completion_path=completion_path,
        )
    return path, ledger, case_ids


def _load_decisions(
    path: Path,
    ledger: HumanReviewLedger,
) -> tuple[str, str, dict[tuple[str, int], HumanReview], dict[str, Any]]:
    raw = _mapping_file(path, label="human-review decisions")
    if raw.get("schema_version") != REVIEW_DECISIONS_SCHEMA:
        raise ValueError(f"decisions must use {REVIEW_DECISIONS_SCHEMA}")
    if set(raw) != {"schema_version", "reviewer", "reviewed_at", "items"}:
        raise ValueError("decisions contain missing or unknown top-level fields")
    reviewer = raw.get("reviewer")
    reviewed_at = raw.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("decisions require a non-empty reviewer identity")
    if not isinstance(reviewed_at, str) or not reviewed_at:
        raise ValueError("decisions require an ISO-8601 review timestamp")
    raw_items = raw.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(
        raw_items, (str, bytes, bytearray)
    ):
        raise ValueError("decisions items must be a sequence")
    decisions: dict[tuple[str, int], HumanReview] = {}
    expected = {
        (item.artifacts.leaf_id, item.rollout_index): item
        for item in ledger.items
    }
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("each human-review decision must be a mapping")
        if set(raw_item) != {
            "leaf_id",
            "rollout_index",
            "artifact_binding_sha256",
            "checks",
            "notes",
        }:
            raise ValueError("decision item contains missing or unknown fields")
        identity = (
            str(raw_item.get("leaf_id", "")),
            int(raw_item.get("rollout_index", -1)),
        )
        if identity in decisions:
            raise ValueError(f"duplicate human-review decision: {identity}")
        source_item = expected.get(identity)
        if source_item is None:
            raise ValueError(f"decision is not a member of the pending ledger: {identity}")
        binding = raw_item.get("artifact_binding_sha256")
        if binding != source_item.artifacts.binding_sha256:
            raise ValueError(f"decision artifact hash mismatch for {identity}")
        checks = raw_item.get("checks")
        if not isinstance(checks, Mapping):
            raise ValueError(f"decision lacks checks for {identity}")
        if "notes" not in raw_item or not isinstance(raw_item.get("notes"), str):
            raise ValueError(f"decision requires a notes string for {identity}")
        decision = HumanReview(
            artifact_binding_sha256=str(binding),
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            checks={str(key): value for key, value in checks.items()},
            notes=str(raw_item["notes"]),
        )
        decision.validate()
        decisions[identity] = decision
    missing = sorted(set(expected) - set(decisions))
    if missing:
        raise ValueError(f"decisions do not cover every pending rollout: {missing}")
    return reviewer, reviewed_at, decisions, raw


@dataclass(frozen=True, slots=True)
class ExternalReviewPublication:
    output_path: str
    publication_id: str
    status: str
    approved_leaf_ids: tuple[str, ...]
    failed_leaf_ids: tuple[str, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidatedExternalReviewPublication:
    publication_path: str
    activation_report_path: str
    dataset_root: str
    publication_id: str
    status: str
    approved_leaf_ids: tuple[str, ...]
    failed_leaf_ids: tuple[str, ...]
    report_sha256: str
    dataset_seal_sha256: str


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} fields differ: missing={missing}, extra={extra}")


def validate_external_review_publication(
    activation_report_path: str | Path,
    *,
    corpus_id: str | None = None,
) -> ValidatedExternalReviewPublication:
    """Validate an external review bundle and all dataset bytes it authorizes.

    The activation report is deliberately accepted by path only. Its sibling
    manifest, completed ledger, dataset binding, and bound sealed dataset are
    all required; a self-authored report mapping is not activation evidence.
    """

    activation_path = Path(activation_report_path).resolve(strict=True)
    if not activation_path.is_file() or activation_path.name != "activation_report.json":
        raise ValueError("pilot activation must name a publication activation_report.json")
    publication_root = activation_path.parent
    manifest_path = publication_root / "manifest.json"
    binding_path = publication_root / "dataset_binding.json"
    ledger_path = publication_root / "human_review_ledger.json"
    for path, label in (
        (manifest_path, "external review manifest"),
        (binding_path, "external dataset binding"),
        (ledger_path, "completed human-review ledger"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    manifest = _mapping_file(manifest_path, label="external review manifest")
    binding = _mapping_file(binding_path, label="external dataset binding")
    ledger_raw = _mapping_file(ledger_path, label="completed human-review ledger")
    activation = _mapping_file(activation_path, label="pilot activation report")
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "publication_id",
            "status",
            "approved_leaf_ids",
            "failed_leaf_ids",
            "dataset_seal_sha256",
            "dataset_binding_sha256",
            "dataset_binding_file_sha256",
            "human_review_ledger_sha256",
            "human_review_ledger_file_sha256",
            "activation_report_sha256",
            "activation_report_file_sha256",
            "fixed_review_artifacts_training_eligible",
        },
        label="external review manifest",
    )
    if manifest.get("schema_version") != EXTERNAL_REVIEW_PUBLICATION_SCHEMA:
        raise ValueError("external review publication schema is unsupported")
    for path, key in (
        (binding_path, "dataset_binding_file_sha256"),
        (ledger_path, "human_review_ledger_file_sha256"),
        (activation_path, "activation_report_file_sha256"),
    ):
        _expect_hash(path, str(manifest.get(key) or ""), label=key)

    _require_exact_keys(
        binding,
        {
            "schema_version",
            "dataset_name",
            "dataset_path",
            "dataset_seal_sha256",
            "dataset_seal_payload_sha256",
            "run_plan_sha256",
            "finalized_completion_sha256",
            "finalized_metadata_sha256",
            "pending_ledger_file_sha256",
            "pending_ledger_sha256",
            "decisions_file_sha256",
            "decisions_payload_sha256",
            "artifact_binding_sha256",
            "binding_sha256",
        },
        label="external dataset binding",
    )
    binding_payload = dict(binding)
    binding_sha256 = binding_payload.pop("binding_sha256", None)
    if not _valid_sha256(binding_sha256) or binding_sha256 != sha256_json(
        binding_payload
    ):
        raise ValueError("external dataset binding hash mismatch")
    for key in ("decisions_file_sha256", "decisions_payload_sha256"):
        if not _valid_sha256(binding.get(key)):
            raise ValueError(f"external dataset binding has malformed {key}")

    dataset_path = binding.get("dataset_path")
    if not isinstance(dataset_path, str) or not dataset_path:
        raise ValueError("external dataset binding lacks its dataset path")
    dataset_root = Path(dataset_path).resolve(strict=True)
    if not dataset_root.is_dir():
        raise ValueError("external dataset binding target is not a directory")
    live_binding = _verify_finalized_dataset(dataset_root)
    for key, value in live_binding.items():
        if binding.get(key) != value:
            raise ValueError(f"external dataset binding is stale for {key}")
    pending_path, pending, case_ids = _load_pending_ledger(dataset_root)
    if (
        binding.get("pending_ledger_file_sha256") != sha256_file(pending_path)
        or binding.get("pending_ledger_sha256") != pending.ledger_sha256
        or binding.get("artifact_binding_sha256")
        != sorted(item.artifacts.binding_sha256 for item in pending.items)
    ):
        raise ValueError("external dataset binding differs from current review evidence")

    _require_exact_keys(
        ledger_raw,
        {
            "schema_version",
            "review_plan_sha256",
            "review_request_ledger_sha256",
            "expected_rollouts_per_leaf",
            "items",
            "extras",
            "ledger_sha256",
        },
        label="completed human-review ledger",
    )
    completed = HumanReviewLedger.from_dict(ledger_raw)
    leaf_ids = tuple(sorted({item.artifacts.leaf_id for item in completed.items}))
    completed.validate(required_leaf_ids=leaf_ids)
    if (
        completed.review_plan_sha256 != pending.review_plan_sha256
        or completed.review_request_ledger_sha256
        != pending.review_request_ledger_sha256
        or len(completed.items) != len(pending.items)
    ):
        raise ValueError("completed ledger is bound to different pending evidence")
    pending_by_identity = {
        (item.artifacts.leaf_id, item.rollout_index): item for item in pending.items
    }
    reviewers: set[str] = set()
    reviewed_at: set[str] = set()
    for item in completed.items:
        identity = (item.artifacts.leaf_id, item.rollout_index)
        pending_item = pending_by_identity.get(identity)
        if (
            pending_item is None
            or item.artifacts != pending_item.artifacts
            or item.scene_profile != pending_item.scene_profile
            or item.automated_qc_passed != pending_item.automated_qc_passed
            or item.review is None
        ):
            raise ValueError("completed ledger differs from its pending artifact membership")
        reviewers.add(item.review.reviewer)
        reviewed_at.add(item.review.reviewed_at)
        expected_case_id = f"{item.artifacts.leaf_id}-review-{item.rollout_index:02d}"
        if case_ids.get(item.artifacts.binding_sha256) != expected_case_id:
            raise ValueError("completed ledger uses a non-canonical fixed review case ID")
    if len(reviewers) != 1 or len(reviewed_at) != 1:
        raise ValueError("completed ledger must contain one reviewer and timestamp")
    extras = completed.extras
    if (
        extras.get("state") != "human_review_completed"
        or extras.get("activation_forbidden") is not False
        or extras.get("fixed_review_artifacts_training_eligible") is not False
        or extras.get("dataset_binding_sha256") != binding_sha256
        or extras.get("reviewer") != next(iter(reviewers))
        or extras.get("reviewed_at") != next(iter(reviewed_at))
    ):
        raise ValueError("completed ledger activation metadata is inconsistent")

    _require_exact_keys(
        activation,
        {
            "schema_version",
            "dataset_binding_sha256",
            "human_review_ledger_sha256",
            "fixed_review_artifacts_training_eligible",
            "leaves",
            "report_sha256",
        },
        label="pilot activation report",
    )
    activation_payload = dict(activation)
    report_sha256 = activation_payload.pop("report_sha256", None)
    if (
        activation.get("schema_version") != ACTIVATION_REPORT_SCHEMA
        or not _valid_sha256(report_sha256)
        or report_sha256 != sha256_json(activation_payload)
        or activation.get("dataset_binding_sha256") != binding_sha256
        or activation.get("human_review_ledger_sha256") != completed.ledger_sha256
        or activation.get("fixed_review_artifacts_training_eligible") is not False
    ):
        raise ValueError("pilot activation report cross-document binding mismatch")
    leaves = activation.get("leaves")
    if not isinstance(leaves, Mapping) or set(leaves) != set(leaf_ids):
        raise ValueError("pilot activation report leaf membership is inconsistent")
    failures = {
        leaf_id: completed.activation_failures(leaf_id) for leaf_id in leaf_ids
    }
    approved = tuple(leaf_id for leaf_id in leaf_ids if not failures[leaf_id])
    failed = tuple(leaf_id for leaf_id in leaf_ids if failures[leaf_id])
    for leaf_id in leaf_ids:
        leaf = leaves.get(leaf_id)
        if not isinstance(leaf, Mapping):
            raise ValueError(f"pilot activation report leaf is malformed: {leaf_id}")
        _require_exact_keys(
            leaf,
            {
                "review_status",
                "pilot_eligible",
                "production_released",
                "failures",
                "case_ids",
            },
            label=f"pilot activation report leaf {leaf_id}",
        )
        expected_case_ids = [f"{leaf_id}-review-{index:02d}" for index in range(6)]
        expected_approved = not failures[leaf_id]
        if (
            leaf.get("review_status")
            != ("approved" if expected_approved else "failed")
            or leaf.get("pilot_eligible") is not expected_approved
            or leaf.get("production_released") is not False
            or leaf.get("failures") != failures[leaf_id]
            or leaf.get("case_ids") != expected_case_ids
        ):
            raise ValueError(f"pilot activation verdict mismatch for {leaf_id}")

    expected_status = "approved" if not failed else "failed"
    publication_id = sha256_json(
        {
            "dataset_binding_sha256": binding_sha256,
            "human_review_ledger_sha256": completed.ledger_sha256,
            "activation_report_sha256": report_sha256,
        }
    )
    if (
        manifest.get("publication_id") != publication_id
        or manifest.get("status") != expected_status
        or manifest.get("approved_leaf_ids") != list(approved)
        or manifest.get("failed_leaf_ids") != list(failed)
        or manifest.get("dataset_seal_sha256")
        != binding.get("dataset_seal_sha256")
        or manifest.get("dataset_binding_sha256") != binding_sha256
        or manifest.get("human_review_ledger_sha256") != completed.ledger_sha256
        or manifest.get("activation_report_sha256") != report_sha256
        or manifest.get("fixed_review_artifacts_training_eligible") is not False
    ):
        raise ValueError("external review manifest verdict or binding mismatch")
    expected_directory = f"{binding['dataset_name']}--{publication_id[:16]}"
    if publication_root.name != expected_directory:
        raise ValueError("external review publication directory is not content-addressed")
    if corpus_id is not None and corpus_id not in approved:
        raise ValueError(
            f"publication has no complete approved fixed-six verdict for {corpus_id}"
        )
    return ValidatedExternalReviewPublication(
        publication_path=str(publication_root),
        activation_report_path=str(activation_path),
        dataset_root=str(dataset_root),
        publication_id=publication_id,
        status=expected_status,
        approved_leaf_ids=approved,
        failed_leaf_ids=failed,
        report_sha256=str(report_sha256),
        dataset_seal_sha256=str(binding["dataset_seal_sha256"]),
    )


def finalize_human_review(
    dataset_root: str | Path,
    decisions_path: str | Path,
    output_root: str | Path,
) -> ExternalReviewPublication:
    """Verify and publish completed reviews without modifying the sealed run."""

    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    binding = _verify_finalized_dataset(root)
    pending_path, pending, case_ids = _load_pending_ledger(root)
    decisions_file = Path(decisions_path).resolve(strict=True)
    try:
        decisions_file.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("human-review decisions must remain outside the sealed dataset")
    reviewer, reviewed_at, decisions, raw_decisions = _load_decisions(
        decisions_file, pending
    )
    binding.update(
        {
            "pending_ledger_file_sha256": sha256_file(pending_path),
            "pending_ledger_sha256": pending.ledger_sha256,
            "decisions_file_sha256": sha256_file(decisions_file),
            "decisions_payload_sha256": sha256_json(raw_decisions),
            "artifact_binding_sha256": sorted(
                item.artifacts.binding_sha256 for item in pending.items
            ),
        }
    )
    binding["binding_sha256"] = sha256_json(binding)

    completed_items = tuple(
        replace(
            item,
            review=decisions[(item.artifacts.leaf_id, item.rollout_index)],
        )
        for item in pending.items
    )
    completed = HumanReviewLedger(
        items=completed_items,
        review_plan_sha256=pending.review_plan_sha256,
        review_request_ledger_sha256=pending.review_request_ledger_sha256,
        expected_rollouts_per_leaf=pending.expected_rollouts_per_leaf,
        extras={
            **dict(pending.extras),
            "state": "human_review_completed",
            "activation_forbidden": False,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "dataset_binding_sha256": binding["binding_sha256"],
            "fixed_review_artifacts_training_eligible": False,
        },
    )
    leaf_ids = tuple(sorted({item.artifacts.leaf_id for item in completed.items}))
    completed.validate(required_leaf_ids=leaf_ids)
    failures = {
        leaf_id: completed.activation_failures(leaf_id) for leaf_id in leaf_ids
    }
    approved = tuple(leaf_id for leaf_id in leaf_ids if not failures[leaf_id])
    failed = tuple(leaf_id for leaf_id in leaf_ids if failures[leaf_id])
    status = "approved" if not failed else "failed"
    activation = {
        "schema_version": ACTIVATION_REPORT_SCHEMA,
        "dataset_binding_sha256": binding["binding_sha256"],
        "human_review_ledger_sha256": completed.ledger_sha256,
        "fixed_review_artifacts_training_eligible": False,
        "leaves": {
            leaf_id: {
                "review_status": "approved" if not failures[leaf_id] else "failed",
                "pilot_eligible": not failures[leaf_id],
                "production_released": False,
                "failures": failures[leaf_id],
                "case_ids": sorted(
                    case_ids[item.artifacts.binding_sha256]
                    for item in completed.items
                    if item.artifacts.leaf_id == leaf_id
                ),
            }
            for leaf_id in leaf_ids
        },
    }
    activation["report_sha256"] = sha256_json(activation)
    publication_id = sha256_json(
        {
            "dataset_binding_sha256": binding["binding_sha256"],
            "human_review_ledger_sha256": completed.ledger_sha256,
            "activation_report_sha256": activation["report_sha256"],
        }
    )
    external_root = Path(output_root).resolve()
    try:
        external_root.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("external review output cannot be inside the sealed dataset")
    destination = external_root / (
        f"{root.name}--{publication_id[:16]}"
    )
    with AtomicDirectory(destination) as staging:
        binding_path = atomic_write_json(staging / "dataset_binding.json", binding)
        ledger_path = atomic_write_json(
            staging / "human_review_ledger.json", completed.to_dict()
        )
        activation_path = atomic_write_json(
            staging / "activation_report.json", activation
        )
        manifest = {
            "schema_version": EXTERNAL_REVIEW_PUBLICATION_SCHEMA,
            "publication_id": publication_id,
            "status": status,
            "approved_leaf_ids": list(approved),
            "failed_leaf_ids": list(failed),
            "dataset_seal_sha256": binding["dataset_seal_sha256"],
            "dataset_binding_sha256": binding["binding_sha256"],
            "dataset_binding_file_sha256": sha256_file(binding_path),
            "human_review_ledger_sha256": completed.ledger_sha256,
            "human_review_ledger_file_sha256": sha256_file(ledger_path),
            "activation_report_sha256": activation["report_sha256"],
            "activation_report_file_sha256": sha256_file(activation_path),
            "fixed_review_artifacts_training_eligible": False,
        }
        atomic_write_json(staging / "manifest.json", manifest)
    manifest_sha256 = sha256_file(destination / "manifest.json")
    return ExternalReviewPublication(
        output_path=str(destination),
        publication_id=publication_id,
        status=status,
        approved_leaf_ids=approved,
        failed_leaf_ids=failed,
        manifest_sha256=manifest_sha256,
    )


__all__ = [
    "ACTIVATION_REPORT_SCHEMA",
    "DATASET_REVIEW_BINDING_SCHEMA",
    "EXTERNAL_REVIEW_PUBLICATION_SCHEMA",
    "ExternalReviewPublication",
    "REVIEW_DECISIONS_SCHEMA",
    "ValidatedExternalReviewPublication",
    "finalize_human_review",
    "validate_external_review_publication",
]
