"""Single-worker canonical preview execution for fixed source review cases.

This module is intentionally orchestration-only.  It prepares the same
immutable run plan used by sharded generation, executes that plan through the
owned source dispatcher, seals the dataset, runs strict persisted-artifact QC,
and derives review media from the finalized MP4/Parquet bytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid

from .episode_writer import load_episode_records
from .hashing import sha256_file, sha256_json
from .paths import ExistingOutputError, atomic_write_json, ensure_not_source_path
from .provenance import get_git_commit
from .qc import validate_dataset
from .review import (
    HumanReviewLedger,
    ReviewItem,
    STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA,
    bind_fixed_review_media_pack_strict,
    write_fixed_review_media_pack,
)
from .review_suite import (
    REVIEW_SCENE_SEQUENCE,
    ReviewSuiteBundle,
    ReviewSuiteCase,
    load_review_suite_bundle,
)
from .run_orchestration import finalize_run, plan_run
from .schema import DatasetInfo, TimeBase
from .source_execution import source_finalization_rows
from .source_scenario import SourceScenarioSpec
from .video_writer import VideoSpec


SOURCE_PREVIEW_RUN_SCHEMA = "dynamic-robot-source-preview-run/v2"
SOURCE_REVIEW_INPUT_PROVENANCE_SCHEMA = (
    "dynamic-robot-source-review-input-provenance/v1"
)
SOURCE_PREVIEW_PREPARE_REQUEST_SCHEMA = (
    "dynamic-robot-source-preview-prepare-request/v1"
)


def _canonical_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    normalized = _canonical_copy(dict(value))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ExistingOutputError(f"immutable preview artifact differs: {path}")
        return
    atomic_write_json(path, normalized)


def select_source_review_cases(
    bundle: ReviewSuiteBundle,
    *,
    case_ids: Sequence[str] = (),
    leaf_ids: Sequence[str] = (),
    clean_r0_only: bool = False,
) -> tuple[ReviewSuiteCase, ...]:
    """Select an ordered executable subset without changing any fixed seed."""

    requested_cases = tuple(dict.fromkeys(str(value) for value in case_ids))
    requested_leaves = tuple(dict.fromkeys(str(value) for value in leaf_ids))
    known_cases = {case.case_id for case in bundle.plan.cases}
    known_leaves = {case.corpus_leaf_id for case in bundle.plan.cases}
    unknown_cases = sorted(set(requested_cases) - known_cases)
    unknown_leaves = sorted(set(requested_leaves) - known_leaves)
    if unknown_cases or unknown_leaves:
        raise ValueError(
            f"unknown fixed review selectors: cases={unknown_cases}, leaves={unknown_leaves}"
        )
    selected = [
        case
        for case in bundle.plan.cases
        if (not requested_cases or case.case_id in requested_cases)
        and (not requested_leaves or case.corpus_leaf_id in requested_leaves)
        and (not clean_r0_only or case.rollout_index == 0)
    ]
    if not selected:
        raise ValueError("source preview selection is empty")
    blocked = {
        case.case_id: list(case.execution_blockers)
        for case in selected
        if not case.execution_eligible or case.executable_quota != 1
    }
    if blocked:
        raise ValueError(
            "blocked review cases cannot consume preview quota: "
            + json.dumps(blocked, sort_keys=True, separators=(",", ":"))
        )
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class SourcePreviewResult:
    dataset_root: str
    run_id: str
    plan_hash: str
    review_suite_id: str
    case_ids: tuple[str, ...]
    episode_count: int
    strict_qc_passed: bool
    passed_case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]
    qc_report_path: str
    media_pack_paths: Mapping[str, str]
    review_artifact_paths: Mapping[str, str]
    pending_human_review_ledger_path: str | None
    schema_version: str = SOURCE_PREVIEW_RUN_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return _canonical_copy(asdict(self))


def _write_review_inputs(
    root: Path,
    bundle: ReviewSuiteBundle,
    declarations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    scenario_paths: dict[str, str] = {}
    provenance: list[dict[str, Any]] = []
    for declaration in declarations:
        raw_case = declaration.get("review_case")
        raw_spec = declaration.get("source_scenario_spec")
        if not isinstance(raw_case, Mapping) or not isinstance(raw_spec, Mapping):
            raise ValueError("source declaration lacks review case or scenario spec")
        case_id = str(raw_case["case_id"])
        scenario = SourceScenarioSpec.from_dict(raw_spec)
        relative_spec = f"reviews/source_specs/{case_id}.json"
        relative_manifest = f"reviews/source_manifests/{case_id}.json"
        spec_path = root / relative_spec
        manifest_path = root / relative_manifest
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_or_validate_json(spec_path, scenario.to_dict())
        source_manifest = {
            "schema_version": STRICT_REVIEW_SOURCE_MANIFEST_SCHEMA,
            "backend": scenario.backend,
            "case_id": case_id,
            "episode_uuid": str(declaration["episode_uuid"]),
            "review_plan_sha256": bundle.plan.plan_sha256,
            "review_case_sha256": str(declaration["review_case_sha256"]),
            "source_scenario_spec_sha256": scenario.spec_hash,
            "source_hashes": dict(scenario.source_hashes),
            "source_hashes_sha256": sha256_json(scenario.source_hashes),
        }
        _write_or_validate_json(manifest_path, source_manifest)
        scenario_paths[case_id] = relative_spec
        provenance.append(
            {
                "schema_version": SOURCE_REVIEW_INPUT_PROVENANCE_SCHEMA,
                "case_id": case_id,
                "episode_uuid": str(declaration["episode_uuid"]),
                "review_suite_id": bundle.plan.suite_id,
                "review_plan_sha256": bundle.plan.plan_sha256,
                "review_request_ledger_sha256": bundle.requests.ledger_sha256,
                "source_scenario_spec_path": relative_spec,
                "source_scenario_file_sha256": sha256_file(spec_path),
                "source_scenario_spec_sha256": scenario.spec_hash,
                "source_manifest_path": relative_manifest,
                "source_manifest_sha256": sha256_file(manifest_path),
            }
        )
    return scenario_paths, provenance


def _task_rows(records: Iterable[Any]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.task_index is None:
            raise ValueError("source preview record lacks its registry task index")
        row = {
            "task_index": int(record.task_index),
            "family": record.family,
            "subfamily": record.subfamily,
            "corpus_leaf_id": str(
                record.extras.get("source_scenario_spec", {}).get(
                    "corpus_leaf_id", ""
                )
            ),
        }
        existing = by_index.get(int(record.task_index))
        if existing is not None and existing != row:
            raise ValueError("registry task index maps to multiple source leaves")
        by_index[int(record.task_index)] = row
    return [by_index[index] for index in sorted(by_index)]


def _run_preview_shards_isolated(root: Path, shard_count: int) -> None:
    """Execute each preview episode in a fresh owned worker process.

    MuJoCo render contexts and imported asset trees can retain native memory
    after Python objects are released.  A long single-process 48-case review
    therefore risks being killed after several otherwise valid episodes.  The
    immutable plan already defines worker-private shards, so execute one case
    per short-lived process through the ordinary owned ``run-shard`` CLI.  No
    arbitrary executor is imported and the parent still performs exact-member
    finalization.
    """

    if shard_count <= 0:
        raise ValueError("source preview requires at least one isolated shard")
    repository_root = Path(__file__).resolve().parents[3]
    for shard_id in range(shard_count):
        command = (
            sys.executable,
            "-m",
            "dynamic_robot_dataset.cli",
            "run-shard",
            "--dataset",
            str(root),
            "--shard-id",
            str(shard_id),
        )
        completed = subprocess.run(
            command,
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(
                f"isolated source preview shard {shard_id} failed: {detail}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"isolated source preview shard {shard_id} returned malformed JSON"
            ) from error
        if (
            int(result.get("failed_count", -1)) != 0
            or int(result.get("remaining_count", -1)) != 0
            or int(result.get("committed_count", 0))
            + int(result.get("recovered_count", 0))
            != int(result.get("planned_count", -1))
        ):
            raise RuntimeError(
                "source preview cannot finalize an incomplete isolated shard: "
                f"{json.dumps(result, sort_keys=True)}"
            )


def _prepare_source_review_declarations_isolated(
    cases: Sequence[ReviewSuiteCase],
    *,
    generator_git_commit: str,
) -> list[dict[str, Any]]:
    """Prepare each model-backed declaration in its own short-lived process.

    MuJoCo and RoboCasa retain native model allocations beyond Python object
    lifetime.  Eagerly preparing six F1 cases in the preview parent can leave
    more than 1.5 GiB resident before the first render worker starts, causing
    the parent to be killed while otherwise valid shards are committing.  A
    one-case worker preserves the exact declaration contract while bounding
    native lifetime independently from both planning and execution.
    """

    if not cases:
        raise ValueError("source preview requires at least one declaration")
    if not generator_git_commit:
        raise ValueError("source preview declaration preparation lacks a commit")
    repository_root = Path(__file__).resolve().parents[3]
    declarations: list[dict[str, Any]] = []
    for episode_index, case in enumerate(cases):
        request = {
            "schema_version": SOURCE_PREVIEW_PREPARE_REQUEST_SCHEMA,
            "review_case": case.to_dict(),
            "episode_index": episode_index,
            "generator_git_commit": generator_git_commit,
        }
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "dynamic_robot_dataset.common.source_preview_prepare_worker",
            ),
            cwd=repository_root,
            input=json.dumps(
                request,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(
                "isolated source preview declaration preparation failed for "
                f"{case.case_id}: {detail}"
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "isolated source preview declaration preparation returned "
                f"malformed JSON for {case.case_id}"
            ) from error
        if not isinstance(value, Mapping):
            raise RuntimeError(
                "isolated source preview declaration preparation returned a "
                f"non-mapping for {case.case_id}"
            )
        declaration = dict(value)
        raw_case = declaration.get("review_case")
        if (
            declaration.get("episode_index") != episode_index
            or declaration.get("episode_uuid") != case.episode_uuid
            or declaration.get("review_suite_episode_index") != case.episode_index
            or declaration.get("review_case_sha256") != case.case_sha256
            or declaration.get("generator_git_commit") != generator_git_commit
            or raw_case != case.to_dict()
        ):
            raise RuntimeError(
                "isolated source preview declaration identity differs for "
                f"{case.case_id}"
            )
        raw_spec = declaration.get("source_scenario_spec")
        if not isinstance(raw_spec, Mapping):
            raise RuntimeError(
                f"isolated source preview declaration lacks a spec for {case.case_id}"
            )
        scenario = SourceScenarioSpec.from_dict(raw_spec)
        if (
            scenario.scenario_id != case.case_id
            or declaration.get("source_scenario_spec_sha256") != scenario.spec_hash
        ):
            raise RuntimeError(
                "isolated source preview declaration scenario binding differs for "
                f"{case.case_id}"
            )
        declarations.append(declaration)
    return declarations


def execute_source_review_preview(
    review_suite_root: str | Path,
    dataset_root: str | Path,
    *,
    case_ids: Sequence[str] = (),
    leaf_ids: Sequence[str] = (),
    clean_r0_only: bool = False,
    resume: bool = False,
    deep_video_checks: bool = True,
) -> SourcePreviewResult:
    """Plan, execute, finalize, validate, and package fixed review rollouts."""

    bundle = load_review_suite_bundle(review_suite_root)
    cases = select_source_review_cases(
        bundle,
        case_ids=case_ids,
        leaf_ids=leaf_ids,
        clean_r0_only=clean_r0_only,
    )
    root = ensure_not_source_path(dataset_root)
    generator_git_commit = get_git_commit(Path(__file__).resolve().parents[3])
    declarations = _prepare_source_review_declarations_isolated(
        cases,
        generator_git_commit=generator_git_commit,
    )
    simulation_rates = sorted(
        {
            int(declaration["source_scenario_spec"]["physics"]["simulation_hz"])
            for declaration in declarations
        }
    )
    resolved_config = {
        "schema_version": SOURCE_PREVIEW_RUN_SCHEMA,
        "backend": "source_mujoco",
        "purpose": "fixed_review_preview",
        "generator_git_commit": generator_git_commit,
        "review_suite_id": bundle.plan.suite_id,
        "review_plan_sha256": bundle.plan.plan_sha256,
        "review_request_ledger_sha256": bundle.requests.ledger_sha256,
        "corpus_leaf_ids": [case.corpus_leaf_id for case in cases],
        "case_ids": [case.case_id for case in cases],
        "simulation_hz_values": simulation_rates,
        "control_hz": 60,
        "video_hz": 30,
        "views": ["main", "secondary"],
        "release_state": "review_only_blocked",
    }
    # One episode per shard keeps renderer/native simulator lifetime bounded;
    # execution remains sequential and deterministic for this preview wrapper.
    shard_count = len(declarations)
    plan = plan_run(
        root,
        resolved_config,
        declarations,
        shard_count=shard_count,
        resume=resume,
        video_spec=VideoSpec(),
        split_settings={
            "strategy": "leakage_aware_stratified",
            "seed": 20260717,
            "train_fraction": 0.80,
            "validation_fraction": 0.10,
            "test_fraction": 0.10,
        },
    )
    scenario_paths, input_provenance = _write_review_inputs(
        root, bundle, declarations
    )
    _run_preview_shards_isolated(root, shard_count)
    records = load_episode_records(root)
    finalization = source_finalization_rows(records)
    info = DatasetInfo(
        name=f"source-review-{bundle.plan.suite_id}",
        dataset_uuid=str(
            uuid.uuid5(
                uuid.UUID("24998d67-a140-50d8-a753-2fbb37425389"),
                plan.run_id,
            )
        ),
        description=(
            "Fixed, review-only source_mujoco rollouts; blocked from training until "
            "profile calibration and human acceptance."
        ),
        # Per-episode SourceScenarioSpec is authoritative when a review bundle
        # mixes the admitted 600 Hz candidate with 1200 Hz reference cases.
        # The dataset-level base records the finest shared simulation clock.
        time_base=TimeBase(
            sim_hz=float(max(simulation_rates)),
            control_hz=60.0,
            video_hz=30.0,
        ),
        generator_version=generator_git_commit,
        extras={
            "run_id": plan.run_id,
            "run_plan_sha256": sha256_file(root / ".run_plan.json"),
            "review_suite_id": bundle.plan.suite_id,
            "review_plan_sha256": bundle.plan.plan_sha256,
            "review_request_ledger_sha256": bundle.requests.ledger_sha256,
            "simulation_hz_values": simulation_rates,
            "episode_simulation_hz_source": "source_scenario_spec.physics.simulation_hz",
            "release_state": "blocked",
        },
    )
    finalize_run(
        root,
        info,
        tasks=_task_rows(records),
        cameras=finalization.cameras,
        provenance=(*finalization.provenance, *input_provenance),
    )

    qc_path = root / "qc" / "dataset_report.json"
    if qc_path.exists():
        stored_qc = json.loads(qc_path.read_text(encoding="utf-8"))
        report = validate_dataset(
            root,
            deep_video_checks=deep_video_checks,
            strict_all=True,
        )
        if stored_qc.get("episodes") != [item.to_dict() for item in report.episodes]:
            raise ExistingOutputError("stored strict QC differs from current validation")
    else:
        report = validate_dataset(
            root,
            deep_video_checks=deep_video_checks,
            write_reports=True,
            report_dir=root / "qc",
            strict_all=True,
        )

    qc_by_uuid = {item.episode_uuid: item for item in report.episodes}
    media_paths: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    review_items: list[ReviewItem] = []
    for case in cases:
        relative_spec = scenario_paths[case.case_id]
        relative_manifest = f"reviews/source_manifests/{case.case_id}.json"
        relative_pack = (
            f"reviews/event_strips/{case.corpus_leaf_id}/"
            f"{case.case_id}/media_pack.json"
        )
        if not (root / relative_pack).is_file():
            publication = write_fixed_review_media_pack(
                root,
                review_suite_root=bundle.root,
                source_scenario_spec_path=relative_spec,
            )
            if publication.manifest_path != relative_pack:
                raise RuntimeError("review media pack used a non-canonical path")
        artifact = bind_fixed_review_media_pack_strict(
            root,
            review_suite_root=bundle.root,
            source_scenario_spec_path=relative_spec,
            source_manifest_path=relative_manifest,
            media_pack_manifest_path=relative_pack,
        )
        relative_artifact = (
            f"reviews/artifacts/{case.corpus_leaf_id}/{case.case_id}.json"
        )
        artifact_path = root / relative_artifact
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        _write_or_validate_json(
            artifact_path,
            {**asdict(artifact), "binding_sha256": artifact.binding_sha256},
        )
        media_paths[case.case_id] = relative_pack
        artifact_paths[case.case_id] = relative_artifact
        review_items.append(
            ReviewItem(
                rollout_index=case.rollout_index,
                scene_profile=case.scene_profile,
                artifacts=artifact,
                automated_qc_passed=artifact.automated_qc_passed,
                review=None,
            )
        )

    # A ledger is meaningful only for complete six-rollout leaves.  Publish a
    # hash-bound pending ledger as soon as such leaves exist, but never invent
    # a human decision or let a partial preview look activation-ready.
    complete_leaf_ids = tuple(
        sorted(
            leaf_id
            for leaf_id in {item.artifacts.leaf_id for item in review_items}
            if tuple(
                item.rollout_index
                for item in sorted(
                    (
                        value
                        for value in review_items
                        if value.artifacts.leaf_id == leaf_id
                    ),
                    key=lambda value: value.rollout_index,
                )
            )
            == tuple(range(len(REVIEW_SCENE_SEQUENCE)))
        )
    )
    pending_ledger_path: str | None = None
    if complete_leaf_ids:
        complete_items = tuple(
            item
            for item in review_items
            if item.artifacts.leaf_id in complete_leaf_ids
        )
        ledger = HumanReviewLedger(
            items=complete_items,
            review_plan_sha256=bundle.plan.plan_sha256,
            review_request_ledger_sha256=bundle.requests.ledger_sha256,
            extras={
                "state": "pending_human_review",
                "activation_forbidden": True,
                "complete_leaf_ids": list(complete_leaf_ids),
            },
        )
        ledger.validate(required_leaf_ids=complete_leaf_ids)
        pending_ledger_path = "reviews/human_review_ledger.pending.json"
        _write_or_validate_json(root / pending_ledger_path, ledger.to_dict())

    passed = tuple(
        case.case_id
        for case in cases
        if qc_by_uuid[case.episode_uuid].passed
    )
    failed = tuple(
        case.case_id
        for case in cases
        if not qc_by_uuid[case.episode_uuid].passed
    )
    result = SourcePreviewResult(
        dataset_root=str(root.resolve()),
        run_id=plan.run_id,
        plan_hash=plan.plan_hash,
        review_suite_id=bundle.plan.suite_id,
        case_ids=tuple(case.case_id for case in cases),
        episode_count=len(records),
        strict_qc_passed=report.passed,
        passed_case_ids=passed,
        failed_case_ids=failed,
        qc_report_path="qc/dataset_report.json",
        media_pack_paths=media_paths,
        review_artifact_paths=artifact_paths,
        pending_human_review_ledger_path=pending_ledger_path,
    )
    _write_or_validate_json(root / "reviews" / "preview_result.json", result.to_dict())
    return result


__all__ = [
    "SOURCE_PREVIEW_RUN_SCHEMA",
    "SourcePreviewResult",
    "_prepare_source_review_declarations_isolated",
    "execute_source_review_preview",
    "select_source_review_cases",
]
