"""Immutable run planning, deterministic sharding, and exact finalization.

The simulator-facing code supplies serializable episode declarations to
``plan_run`` and an executor callback to ``run_shard``.  This module owns all
storage identity and membership decisions; workers cannot choose different
episodes, paths, chunking, encoding, or split settings at runtime.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .episode_writer import EpisodeWriter
from .hashing import canonical_json_bytes, sha256_file, sha256_json
from .paths import (
    ExistingOutputError,
    ResumeMismatchError,
    atomic_write_json,
    ensure_not_source_path,
    resolve_dataset_path,
)
from .schema import DatasetInfo, EpisodeRecord
from .source_scenario import SourceScenarioSpec
from .video_writer import VideoSpec


RUN_PLAN_SCHEMA_VERSION = "dynamic-robot-run-plan/v1"
RUN_PLAN_FILE = ".run_plan.json"
TERMINAL_FAILURE_STAGE = "scene_construction_exhausted"
TERMINAL_FAILURE_LEDGER_FILE = "terminal_failures.json"
TERMINAL_FAILURE_LEDGER_SCHEMA = "dynamic-robot-terminal-failures/v1"
TERMINAL_FAILURE_RECORD_SCHEMA = "dynamic-robot-terminal-failure-record/v1"
TERMINAL_FAILURE_PROVENANCE_SCHEMA = (
    "dynamic-robot-terminal-failure-provenance/v1"
)


def _canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _validate_uuid(value: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"Run-plan episode UUID is invalid: {value!r}") from exc
    if str(parsed) != value:
        raise ValueError("Run-plan UUIDs must use canonical lowercase text")


@dataclass(slots=True, frozen=True)
class RunPlanEpisode:
    """One immutable episode declaration and its deterministic partition."""

    episode_uuid: str
    episode_index: int
    shard_id: int
    declaration: dict[str, Any] = field(default_factory=dict)
    source_scenario_spec_sha256: str | None = None

    def validate(self, shard_count: int) -> None:
        _validate_uuid(self.episode_uuid)
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.shard_id != self.episode_index % shard_count:
            raise ValueError(
                f"Episode {self.episode_index} is assigned to the wrong deterministic shard"
            )
        if not isinstance(self.declaration, dict):
            raise ValueError("Run-plan declaration must be a JSON mapping")
        if self.source_scenario_spec_sha256 is not None and (
            len(self.source_scenario_spec_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_scenario_spec_sha256
            )
        ):
            raise ValueError("source scenario binding must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_uuid": self.episode_uuid,
            "episode_index": self.episode_index,
            "shard_id": self.shard_id,
            "declaration": _canonical_copy(self.declaration),
            "source_scenario_spec_sha256": self.source_scenario_spec_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunPlanEpisode":
        return cls(
            episode_uuid=str(value["episode_uuid"]),
            episode_index=int(value["episode_index"]),
            shard_id=int(value["shard_id"]),
            declaration=dict(value.get("declaration") or {}),
            source_scenario_spec_sha256=(
                None
                if value.get("source_scenario_spec_sha256") is None
                else str(value.get("source_scenario_spec_sha256"))
            ),
        )


@dataclass(slots=True, frozen=True)
class RunPlan:
    """Content-addressed episode/shard ledger persisted before simulation."""

    run_id: str
    plan_hash: str
    writer_config_hash: str
    writer_settings: dict[str, Any]
    shard_count: int
    episodes: tuple[RunPlanEpisode, ...]
    schema_version: str = RUN_PLAN_SCHEMA_VERSION

    def _identity_dict(self, *, include_run_id: bool) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "writer_config_hash": self.writer_config_hash,
            "writer_settings": _canonical_copy(self.writer_settings),
            "shard_count": self.shard_count,
            "expected_episode_count": len(self.episodes),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }
        if include_run_id:
            value["run_id"] = self.run_id
        return value

    def validate(self) -> None:
        if self.schema_version != RUN_PLAN_SCHEMA_VERSION:
            raise ValueError(f"Unsupported run-plan schema: {self.schema_version}")
        if self.shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if len(self.writer_config_hash) != 64:
            raise ValueError("writer_config_hash must be a SHA-256 digest")
        indices = [episode.episode_index for episode in self.episodes]
        uuids = [episode.episode_uuid for episode in self.episodes]
        if indices != list(range(len(self.episodes))):
            raise ValueError("Run-plan episode indices must be contiguous and ordered from zero")
        if len(uuids) != len(set(uuids)):
            raise ValueError("Run-plan episode UUIDs must be unique")
        for episode in self.episodes:
            episode.validate(self.shard_count)
        expected_run_id = "run-" + sha256_json(self._identity_dict(include_run_id=False))[:24]
        if self.run_id != expected_run_id:
            raise ValueError("run_id does not match the immutable plan identity")
        expected_plan_hash = sha256_json(self._identity_dict(include_run_id=True))
        if self.plan_hash != expected_plan_hash:
            raise ValueError("plan_hash does not match run-plan contents")

    @property
    def membership(self) -> dict[str, int]:
        return {episode.episode_uuid: episode.episode_index for episode in self.episodes}

    def episodes_for_shard(self, shard_id: int) -> tuple[RunPlanEpisode, ...]:
        if shard_id < 0 or shard_id >= self.shard_count:
            raise ValueError(f"shard_id must be in [0, {self.shard_count})")
        return tuple(episode for episode in self.episodes if episode.shard_id == shard_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._identity_dict(include_run_id=True), "plan_hash": self.plan_hash}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunPlan":
        plan = cls(
            run_id=str(value["run_id"]),
            plan_hash=str(value["plan_hash"]),
            writer_config_hash=str(value["writer_config_hash"]),
            writer_settings=dict(value["writer_settings"]),
            shard_count=int(value["shard_count"]),
            episodes=tuple(
                RunPlanEpisode.from_dict(item) for item in value.get("episodes", ())
            ),
            schema_version=str(value.get("schema_version", "")),
        )
        if int(value.get("expected_episode_count", -1)) != len(plan.episodes):
            raise ValueError("Run-plan expected_episode_count is inconsistent")
        plan.validate()
        return plan


def _write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    normalized = _canonical_copy(value)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ExistingOutputError(f"Immutable orchestration artifact differs: {path}")
        return
    atomic_write_json(path, normalized)


def plan_run(
    dataset_root: str | Path,
    resolved_config: Mapping[str, Any],
    episode_declarations: Iterable[Mapping[str, Any]],
    *,
    shard_count: int,
    resume: bool = False,
    chunk_size: int | None = None,
    video_spec: VideoSpec | None = None,
    split_settings: Mapping[str, Any] | None = None,
) -> RunPlan:
    """Write the generation marker and immutable episode/shard ledger."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    root = ensure_not_source_path(dataset_root)
    writer = EpisodeWriter(
        root,
        resolved_config,
        resume=resume,
        chunk_size=chunk_size,
        video_spec=video_spec,
        split_settings=split_settings,
        worker_id="planner",
    )
    entries: list[RunPlanEpisode] = []
    configured_backend = str(resolved_config.get("backend") or "")
    requires_source_scenario = configured_backend.startswith("source_")
    for expected_index, raw in enumerate(episode_declarations):
        declaration = _canonical_copy(dict(raw))
        episode_index = int(declaration.get("episode_index", expected_index))
        if episode_index != expected_index:
            raise ValueError("Episode declarations must be ordered with contiguous indices")
        episode_uuid = str(declaration.get("episode_uuid", ""))
        source_scenario_raw = declaration.get("source_scenario_spec")
        if requires_source_scenario and not isinstance(source_scenario_raw, Mapping):
            raise ValueError(
                f"backend {configured_backend} requires a complete source_scenario_spec "
                f"for episode {expected_index}"
            )
        source_scenario_hash: str | None = None
        if source_scenario_raw is not None:
            if not isinstance(source_scenario_raw, Mapping):
                raise ValueError("source_scenario_spec must be a mapping")
            source_scenario = SourceScenarioSpec.from_dict(source_scenario_raw)
            if configured_backend and source_scenario.backend != configured_backend:
                raise ValueError(
                    "source scenario backend differs from resolved run configuration"
                )
            source_scenario_hash = source_scenario.spec_hash
            declaration["source_scenario_spec"] = source_scenario.to_dict()
            derived_declaration = {
                "backend": source_scenario.backend,
                "corpus_leaf_id": source_scenario.corpus_leaf_id,
                "task_variant": source_scenario.task_variant,
                "embodiment": source_scenario.embodiment.end_effector,
                "duration_s": source_scenario.duration_s,
            }
            for name, expected_value in derived_declaration.items():
                if name in declaration and declaration[name] != expected_value:
                    raise ValueError(
                        f"episode declaration {name} differs from source_scenario_spec"
                    )
                declaration[name] = expected_value
            declaration["source_scenario_spec_sha256"] = source_scenario_hash
        entry = RunPlanEpisode(
            episode_uuid=episode_uuid,
            episode_index=episode_index,
            shard_id=episode_index % shard_count,
            declaration=declaration,
            source_scenario_spec_sha256=source_scenario_hash,
        )
        entry.validate(shard_count)
        entries.append(entry)
    if not entries:
        raise ValueError("A run plan must contain at least one episode declaration")
    identity = {
        "schema_version": RUN_PLAN_SCHEMA_VERSION,
        "writer_config_hash": writer.config_hash,
        "writer_settings": writer.writer_settings,
        "shard_count": shard_count,
        "expected_episode_count": len(entries),
        "episodes": [entry.to_dict() for entry in entries],
    }
    run_id = "run-" + sha256_json(identity)[:24]
    plan_hash = sha256_json({**identity, "run_id": run_id})
    plan = RunPlan(
        run_id=run_id,
        plan_hash=plan_hash,
        writer_config_hash=writer.config_hash,
        writer_settings=_canonical_copy(writer.writer_settings),
        shard_count=shard_count,
        episodes=tuple(entries),
    )
    _write_or_validate_json(root / RUN_PLAN_FILE, plan.to_dict())
    return plan


def load_run_plan(dataset_root: str | Path) -> RunPlan:
    """Load a run plan and verify its content and generation-root binding."""

    root = Path(dataset_root).resolve(strict=True)
    plan_path = root / RUN_PLAN_FILE
    plan = RunPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
    generation = json.loads((root / ".generation.json").read_text(encoding="utf-8"))
    if generation.get("config_hash") != plan.writer_config_hash:
        raise ResumeMismatchError("Run plan is bound to a different generation configuration")
    if generation.get("identity_settings") != plan.writer_settings:
        raise ResumeMismatchError("Run plan is bound to different writer settings")
    return plan


def _writer_for_plan(root: Path, plan: RunPlan, *, worker_id: str) -> EpisodeWriter:
    generation = json.loads((root / ".generation.json").read_text(encoding="utf-8"))
    settings = plan.writer_settings
    writer = EpisodeWriter(
        root,
        dict(generation["resolved_config"]),
        resume=True,
        chunk_size=int(settings["chunk_size"]),
        video_spec=VideoSpec(**dict(settings["video"])),
        split_settings=dict(settings["split_settings"]),
        layout_version=str(settings["layout_version"]),
        worker_id=worker_id,
    )
    if writer.config_hash != plan.writer_config_hash:
        raise ResumeMismatchError("Writer configuration no longer matches the run plan")
    return writer


@dataclass(slots=True)
class EpisodeMaterialization:
    """In-memory/persisted rollout artifacts returned by a shard executor."""

    record: EpisodeRecord
    frame_rows: Iterable[Mapping[str, Any]]
    videos: Mapping[str, str | Path | Iterable[Any]]
    high_rate_rows: Iterable[Mapping[str, Any]] = ()
    event_rows: Iterable[Mapping[str, Any]] = ()
    transition_rows: Iterable[Mapping[str, Any]] = ()
    object_state_rows: Iterable[Mapping[str, Any]] = ()
    camera_calibration_ids: Mapping[str, str] | None = None


class EpisodeAttemptFailure(RuntimeError):
    """Terminal *pre-rollout* construction exhaustion with no artifacts.

    This exception is deliberately unavailable for a completed rollout,
    solver/QC failure, or intended/actual outcome mismatch.  Those attempts
    must return :class:`EpisodeMaterialization` so their measured videos,
    states, contacts, and negative label are preserved normally.  Callers must
    explicitly acknowledge the sole accepted terminal stage.
    """

    def __init__(
        self,
        reason: str,
        *,
        terminal_stage: str,
        evidence: Mapping[str, Any] | None = None,
    ):
        if terminal_stage != TERMINAL_FAILURE_STAGE:
            raise ValueError(
                "EpisodeAttemptFailure is only valid for exhausted scene construction "
                "before rollout/artifact creation; completed outcome mismatches must "
                "return EpisodeMaterialization"
            )
        if not str(reason).strip():
            raise ValueError("Terminal scene-construction failure requires a reason")
        super().__init__(reason)
        self.reason = str(reason)
        self.terminal_stage = terminal_stage
        self.evidence = _canonical_copy(dict(evidence or {}))


def _planned_task_key(entry: RunPlanEpisode) -> tuple[str, str]:
    return (
        str(entry.declaration.get("family") or "terminal_failure"),
        str(
            entry.declaration.get("subfamily")
            or "scene_construction_exhausted"
        ),
    )


def _terminal_task_index(plan: RunPlan, entry: RunPlanEpisode) -> int:
    declared = entry.declaration.get("task_index")
    if declared is not None:
        return int(declared)
    keys = sorted({_planned_task_key(candidate) for candidate in plan.episodes})
    return keys.index(_planned_task_key(entry))


def _terminal_failure_payload(
    plan: RunPlan,
    entry: RunPlanEpisode,
    failure: EpisodeAttemptFailure,
) -> dict[str, Any]:
    evidence_hash = sha256_json(failure.evidence)
    return {
        "schema_version": TERMINAL_FAILURE_RECORD_SCHEMA,
        "plan_hash": plan.plan_hash,
        "episode_uuid": entry.episode_uuid,
        "episode_index": entry.episode_index,
        "shard_id": entry.shard_id,
        "terminal_stage": failure.terminal_stage,
        "reason": failure.reason,
        "evidence": failure.evidence,
        "evidence_sha256": evidence_hash,
        "rollout_started": False,
        "artifacts_created": False,
        "retry_policy": "new run plan or generator version required",
    }


def _terminal_failure_record(
    writer: EpisodeWriter,
    plan: RunPlan,
    entry: RunPlanEpisode,
    failure: EpisodeAttemptFailure,
) -> EpisodeRecord:
    """Create an honest no-artifact record for exact planned membership."""

    declaration = entry.declaration
    family, subfamily = _planned_task_key(entry)
    payload = _terminal_failure_payload(plan, entry, failure)
    record = EpisodeRecord(
        episode_uuid=entry.episode_uuid,
        episode_index=entry.episode_index,
        counterfactual_bundle_id=str(
            declaration.get("counterfactual_bundle_id")
            or declaration.get("bundle_id")
            or f"terminal-{entry.episode_uuid}"
        ),
        physics_counterfactual_family_id=str(
            declaration.get("physics_counterfactual_family_id")
            or declaration.get("counterfactual_bundle_id")
            or f"terminal-physics-{entry.episode_uuid}"
        ),
        split_group_id=str(
            declaration.get("split_group_id")
            or declaration.get("counterfactual_bundle_id")
            or f"terminal-split-{entry.episode_uuid}"
        ),
        scene_seed=int(declaration.get("scene_seed", declaration.get("seed", 0))),
        branch_seed=int(declaration.get("branch_seed", declaration.get("seed", 0))),
        family=family,
        subfamily=subfamily,
        intended_branch=str(declaration.get("intended_branch") or "unmaterialized"),
        actual_outcome="unverified",
        task_success=False,
        failure_mode="label_unverified",
        variant=str(declaration.get("variant") or "terminal_failure"),
        robot_model=str(declaration.get("robot_model") or "not_instantiated"),
        tool_type=str(declaration.get("tool_type") or "not_instantiated"),
        action_mode="no_artifact_terminal_failure",
        label_confidence=0.0,
        label_status="unverified",
        release_tier="quarantine",
        physics_qc_pass=False,
        source_generator="dynamic_robot_dataset.run_orchestration",
        source_generator_version="terminal-failure/v1",
        generator_git_commit=str(
            writer.config.get("generator_git_commit") or "unknown"
        ),
        config_hash=writer.config_hash,
        simulator_name="not_started",
        simulator_version="not_started",
        renderer="not_started",
        task_index=_terminal_task_index(plan, entry),
        objective_evaluator_id="not_run_terminal_failure",
        objective_evaluator_version="v1",
        objective_threshold_set_hash=sha256_json(
            {"terminal_stage": TERMINAL_FAILURE_STAGE, "evaluator": "not_run"}
        ),
        objective_evidence={
            "stored_objective_success": None,
            "independently_recomputed": False,
            "source": "scene_construction_exhausted_before_rollout",
            "terminal_failure_sha256": sha256_json(payload),
        },
        quality_flags=[
            "terminal_scene_construction_failure",
            "no_rollout_artifacts",
        ],
        extras={
            "terminal_failure": payload,
            "terminal_failure_sha256": sha256_json(payload),
            "planned_source_scenario_spec_sha256": (
                entry.source_scenario_spec_sha256
            ),
        },
    )
    record.validate()
    return record


def _terminal_payload_from_record(
    plan: RunPlan,
    entry: RunPlanEpisode,
    record: EpisodeRecord,
) -> dict[str, Any] | None:
    raw = record.extras.get("terminal_failure")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise RuntimeError("Terminal-failure record payload is not a mapping")
    payload = _canonical_copy(dict(raw))
    expected_identity = {
        "schema_version": TERMINAL_FAILURE_RECORD_SCHEMA,
        "plan_hash": plan.plan_hash,
        "episode_uuid": entry.episode_uuid,
        "episode_index": entry.episode_index,
        "shard_id": entry.shard_id,
        "terminal_stage": TERMINAL_FAILURE_STAGE,
        "rollout_started": False,
        "artifacts_created": False,
    }
    for name, expected in expected_identity.items():
        if payload.get(name) != expected:
            raise ResumeMismatchError(
                f"Terminal failure {name} differs from the run plan: {entry.episode_uuid}"
            )
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeError("Terminal failure evidence must be a mapping")
    if payload.get("evidence_sha256") != sha256_json(evidence):
        raise RuntimeError("Terminal failure evidence hash mismatch")
    payload_hash = sha256_json(payload)
    if record.extras.get("terminal_failure_sha256") != payload_hash:
        raise RuntimeError("Terminal failure record hash mismatch")
    if record.content_hashes or record.video_paths or any(
        value is not None
        for value in (
            record.frame_data_path,
            record.high_rate_path,
            record.events_path,
            record.transition_events_path,
            record.object_states_path,
        )
    ):
        raise RuntimeError("Terminal failure record must not claim rollout artifacts")
    if record.release_eligible or record.physics_qc_pass:
        raise RuntimeError("Terminal failure record cannot be release/QC eligible")
    return payload


def _terminal_receipt(
    plan: RunPlan,
    entry: RunPlanEpisode,
    record: EpisodeRecord,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "dynamic-robot-shard-attempt/v2",
        "plan_hash": plan.plan_hash,
        "episode_uuid": entry.episode_uuid,
        "episode_index": entry.episode_index,
        "shard_id": entry.shard_id,
        "status": "terminal_failure",
        "terminal_stage": TERMINAL_FAILURE_STAGE,
        "evidence_sha256": payload["evidence_sha256"],
        "terminal_failure_sha256": sha256_json(payload),
        "episode_record_sha256": sha256_json(record.to_dict()),
        "retry_policy": "new run plan or generator version required",
    }


@dataclass(slots=True, frozen=True)
class ShardRunResult:
    shard_id: int
    planned_count: int
    committed_count: int
    recovered_count: int
    failed_count: int
    remaining_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "shard_id": self.shard_id,
            "planned_count": self.planned_count,
            "committed_count": self.committed_count,
            "recovered_count": self.recovered_count,
            "failed_count": self.failed_count,
            "remaining_count": self.remaining_count,
        }


ShardExecutor = Callable[[RunPlanEpisode], EpisodeMaterialization]


def _validate_source_materialization_binding(
    entry: RunPlanEpisode,
    record: EpisodeRecord,
) -> None:
    """Require a worker to materialize the exact planned source scenario."""

    expected_hash = entry.source_scenario_spec_sha256
    if expected_hash is None:
        return
    raw = record.extras.get("source_scenario_spec")
    if not isinstance(raw, Mapping):
        raise ResumeMismatchError(
            "Executor record lacks its planned source_scenario_spec"
        )
    scenario = SourceScenarioSpec.from_dict(raw)
    if scenario.spec_hash != expected_hash:
        raise ResumeMismatchError(
            "Executor source_scenario_spec differs from the immutable run plan"
        )
    persisted_hash = record.extras.get("source_scenario_spec_sha256")
    if persisted_hash != expected_hash:
        raise ResumeMismatchError(
            "Executor record lacks the planned source-scenario hash binding"
        )
    expected_fields = {
        "backend": scenario.backend,
        "corpus_leaf_id": scenario.corpus_leaf_id,
        "task_variant": scenario.task_variant,
        "embodiment": scenario.embodiment.end_effector,
    }
    for name, expected_value in expected_fields.items():
        if entry.declaration.get(name) != expected_value:
            raise ResumeMismatchError(
                f"Run-plan {name} differs from its source scenario"
            )


def run_shard(
    dataset_root: str | Path,
    shard_id: int,
    executor: ShardExecutor,
    *,
    max_episodes: int | None = None,
) -> ShardRunResult:
    """Execute exactly one deterministic partition with private staging.

    Ordinary exceptions model interruption and are intentionally not converted
    into retry markers. ``EpisodeAttemptFailure`` is restricted to exhausted
    scene construction before rollout/artifact creation. Completed negative or
    mismatched outcomes return ``EpisodeMaterialization`` and are committed as
    ordinary measured episodes.
    """

    root = Path(dataset_root).resolve(strict=True)
    plan = load_run_plan(root)
    targets = plan.episodes_for_shard(shard_id)
    if max_episodes is not None and max_episodes < 0:
        raise ValueError("max_episodes must be non-negative")
    writer = _writer_for_plan(root, plan, worker_id=f"shard-{shard_id:05d}")
    records = writer.records()
    actual = {record.episode_uuid: record for record in records}
    unexpected = sorted(set(actual) - set(plan.membership))
    if unexpected:
        raise ExistingOutputError(f"Generation root contains unplanned episodes: {unexpected}")
    for episode_uuid, record in actual.items():
        if record.episode_index != plan.membership[episode_uuid]:
            raise ResumeMismatchError(f"Committed episode index changed: {episode_uuid}")

    attempt_root = root / ".shards" / f"shard-{shard_id:05d}" / "attempts"
    attempt_root.mkdir(parents=True, exist_ok=True)
    committed = recovered = failed = processed = 0
    for entry in targets:
        receipt = attempt_root / f"{entry.episode_uuid}.json"
        if receipt.is_file():
            value = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                value.get("plan_hash") != plan.plan_hash
                or value.get("episode_uuid") != entry.episode_uuid
                or int(value.get("episode_index", -1)) != entry.episode_index
            ):
                raise ResumeMismatchError(f"Shard attempt receipt identity changed: {receipt}")
            if value.get("status") == "committed":
                if entry.episode_uuid not in actual:
                    raise RuntimeError(f"Committed receipt has no episode marker: {receipt}")
                record = actual[entry.episode_uuid]
                if _terminal_payload_from_record(plan, entry, record) is not None:
                    raise RuntimeError(
                        f"Committed receipt points to a terminal-failure record: {receipt}"
                    )
                receipt_hashes = value.get("content_hashes")
                if receipt_hashes is not None and dict(receipt_hashes) != record.content_hashes:
                    raise RuntimeError(f"Committed receipt content hashes changed: {receipt}")
                recovered += 1
                continue
            if value.get("status") == "terminal_failure":
                record = actual.get(entry.episode_uuid)
                if record is None:
                    raise RuntimeError(
                        f"Terminal-failure receipt has no episode marker: {receipt}"
                    )
                payload = _terminal_payload_from_record(plan, entry, record)
                if payload is None:
                    raise RuntimeError(
                        f"Terminal-failure receipt points to an ordinary episode: {receipt}"
                    )
                _write_or_validate_json(
                    receipt, _terminal_receipt(plan, entry, record, payload)
                )
                failed += 1
                continue
            if value.get("status") == "failed":
                raise RuntimeError(
                    "Legacy failed receipt has no exact EpisodeRecord membership; "
                    "create a new run plan and use terminal scene-construction failures"
                )
            raise RuntimeError(f"Unknown shard attempt status: {receipt}")
        if entry.episode_uuid in actual:
            existing_record = actual[entry.episode_uuid]
            terminal_payload = _terminal_payload_from_record(
                plan, entry, existing_record
            )
            if terminal_payload is not None:
                _write_or_validate_json(
                    receipt,
                    _terminal_receipt(
                        plan, entry, existing_record, terminal_payload
                    ),
                )
                failed += 1
                continue
            _write_or_validate_json(
                receipt,
                {
                    "schema_version": "dynamic-robot-shard-attempt/v1",
                    "plan_hash": plan.plan_hash,
                    "episode_uuid": entry.episode_uuid,
                    "episode_index": entry.episode_index,
                    "shard_id": shard_id,
                    "status": "committed",
                    "recovered_existing_commit": True,
                },
            )
            recovered += 1
            continue
        if max_episodes is not None and processed >= max_episodes:
            break
        processed += 1
        try:
            materialization = executor(entry)
        except EpisodeAttemptFailure as exc:
            record = _terminal_failure_record(writer, plan, entry, exc)
            marker_value = {
                "config_hash": writer.config_hash,
                "episode": record.to_dict(),
            }
            _write_or_validate_json(
                writer._marker(entry.episode_uuid), marker_value
            )
            actual[record.episode_uuid] = record
            payload = _terminal_payload_from_record(plan, entry, record)
            assert payload is not None
            _write_or_validate_json(
                receipt, _terminal_receipt(plan, entry, record, payload)
            )
            failed += 1
            continue
        if not isinstance(materialization, EpisodeMaterialization):
            raise TypeError("Shard executor must return EpisodeMaterialization")
        if (
            materialization.record.episode_uuid != entry.episode_uuid
            or materialization.record.episode_index != entry.episode_index
        ):
            raise ResumeMismatchError("Executor returned a different episode identity than planned")
        _validate_source_materialization_binding(entry, materialization.record)
        record = writer.write_episode(
            materialization.record,
            frame_rows=materialization.frame_rows,
            videos=materialization.videos,
            high_rate_rows=materialization.high_rate_rows,
            event_rows=materialization.event_rows,
            transition_rows=materialization.transition_rows,
            object_state_rows=materialization.object_state_rows,
            camera_calibration_ids=materialization.camera_calibration_ids,
        )
        actual[record.episode_uuid] = record
        _write_or_validate_json(
            receipt,
            {
                "schema_version": "dynamic-robot-shard-attempt/v1",
                "plan_hash": plan.plan_hash,
                "episode_uuid": entry.episode_uuid,
                "episode_index": entry.episode_index,
                "shard_id": shard_id,
                "status": "committed",
                "content_hashes": record.content_hashes,
            },
        )
        committed += 1
    completed_ids = {
        entry.episode_uuid
        for entry in targets
        if (attempt_root / f"{entry.episode_uuid}.json").is_file()
    }
    return ShardRunResult(
        shard_id=shard_id,
        planned_count=len(targets),
        committed_count=committed,
        recovered_count=recovered,
        failed_count=failed,
        remaining_count=len(targets) - len(completed_ids),
    )


def _attempt_receipt_path(root: Path, entry: RunPlanEpisode) -> Path:
    return (
        root
        / ".shards"
        / f"shard-{entry.shard_id:05d}"
        / "attempts"
        / f"{entry.episode_uuid}.json"
    )


def _terminal_failure_ledger(
    root: Path,
    plan: RunPlan,
    record_by_uuid: Mapping[str, EpisodeRecord],
) -> tuple[dict[str, Any], str]:
    """Verify all receipts and write the immutable terminal-failure ledger."""

    entry_by_uuid = {entry.episode_uuid: entry for entry in plan.episodes}
    expected_receipt_paths = {
        _attempt_receipt_path(root, entry).resolve(): entry
        for entry in plan.episodes
    }
    attempts_root = root / ".shards"
    if attempts_root.exists():
        for path in attempts_root.glob("shard-*/attempts/*.json"):
            if path.resolve() not in expected_receipt_paths:
                raise ExistingOutputError(
                    f"Generation root contains an unplanned attempt receipt: {path}"
                )

    failures: list[dict[str, Any]] = []
    for episode_uuid, record in sorted(
        record_by_uuid.items(), key=lambda item: item[1].episode_index
    ):
        entry = entry_by_uuid[episode_uuid]
        payload = _terminal_payload_from_record(plan, entry, record)
        receipt_path = _attempt_receipt_path(root, entry)
        receipt: dict[str, Any] | None = None
        if receipt_path.is_file():
            raw = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise RuntimeError(f"Attempt receipt is not a mapping: {receipt_path}")
            receipt = dict(raw)
            expected_identity = {
                "plan_hash": plan.plan_hash,
                "episode_uuid": entry.episode_uuid,
                "episode_index": entry.episode_index,
                "shard_id": entry.shard_id,
            }
            for name, expected in expected_identity.items():
                if receipt.get(name) != expected:
                    raise ResumeMismatchError(
                        f"Attempt receipt {name} differs from run plan: {receipt_path}"
                    )
        if payload is None:
            if receipt is not None:
                if receipt.get("status") != "committed":
                    raise RuntimeError(
                        f"Ordinary episode has a non-committed receipt: {receipt_path}"
                    )
                receipt_hashes = receipt.get("content_hashes")
                if receipt_hashes is not None and dict(receipt_hashes) != record.content_hashes:
                    raise RuntimeError(
                        f"Committed receipt content hashes differ: {receipt_path}"
                    )
            continue

        if receipt is None:
            raise RuntimeError(
                f"Terminal-failure record lacks its immutable receipt: {receipt_path}"
            )
        expected_receipt = _terminal_receipt(plan, entry, record, payload)
        if receipt != expected_receipt:
            raise RuntimeError(
                f"Terminal-failure receipt or evidence binding differs: {receipt_path}"
            )
        failures.append(
            {
                "episode_uuid": entry.episode_uuid,
                "episode_index": entry.episode_index,
                "shard_id": entry.shard_id,
                "terminal_stage": TERMINAL_FAILURE_STAGE,
                "reason": payload["reason"],
                "evidence": payload["evidence"],
                "evidence_sha256": payload["evidence_sha256"],
                "terminal_failure_sha256": sha256_json(payload),
                "episode_record_sha256": sha256_json(record.to_dict()),
                "receipt_path": receipt_path.relative_to(root).as_posix(),
                "receipt_sha256": sha256_file(receipt_path),
            }
        )

    run_plan_sha256 = sha256_file(root / RUN_PLAN_FILE)
    ledger = {
        "schema_version": TERMINAL_FAILURE_LEDGER_SCHEMA,
        "plan_hash": plan.plan_hash,
        "run_plan_sha256": run_plan_sha256,
        "planned_episode_count": len(plan.episodes),
        "terminal_failure_count": len(failures),
        "terminal_failures": failures,
        "terminal_failure_set_sha256": sha256_json(failures),
        "pilot_readiness_blocked": bool(failures),
        "failure_semantics": (
            "pre-rollout scene-construction exhaustion only; completed outcome "
            "mismatches are canonical episodes"
        ),
    }
    ledger_path = root / TERMINAL_FAILURE_LEDGER_FILE
    _write_or_validate_json(ledger_path, ledger)
    return ledger, sha256_file(ledger_path)


def finalize_run(
    dataset_root: str | Path,
    info: DatasetInfo,
    *,
    tasks: Iterable[Mapping[str, Any]] | None = None,
    cameras: Iterable[Mapping[str, Any]] = (),
    provenance: Iterable[Mapping[str, Any]] = (),
    counterfactual_families: Iterable[Mapping[str, Any]] | None = None,
) -> list[EpisodeRecord]:
    """Finalize exact plan membership, including honest terminal placeholders."""

    root = Path(dataset_root).resolve(strict=True)
    plan = load_run_plan(root)
    writer = _writer_for_plan(root, plan, worker_id="finalizer")
    records = writer.records()
    record_by_uuid = {record.episode_uuid: record for record in records}
    actual_membership = {
        record.episode_uuid: record.episode_index for record in records
    }
    if actual_membership != plan.membership:
        missing = sorted(set(plan.membership) - set(actual_membership))
        unexpected = sorted(set(actual_membership) - set(plan.membership))
        raise RuntimeError(
            "Cannot finalize partial or changed plan membership: "
            f"missing={missing}, unexpected={unexpected}"
        )
    entry_by_uuid = {entry.episode_uuid: entry for entry in plan.episodes}
    required_streams = set(plan.writer_settings["required_camera_streams"])
    for episode_uuid, record in record_by_uuid.items():
        entry = entry_by_uuid[episode_uuid]
        terminal_payload = _terminal_payload_from_record(plan, entry, record)
        for field_name in ("family", "subfamily", "variant", "robot_model", "tool_type"):
            planned_value = entry.declaration.get(field_name)
            if planned_value is not None and str(planned_value) != str(getattr(record, field_name)):
                raise ResumeMismatchError(
                    f"Committed {field_name} differs from run plan: {episode_uuid}"
                )
        if terminal_payload is not None:
            continue
        if set(record.video_paths) != required_streams:
            raise RuntimeError(
                f"Planned episode lacks exact canonical camera streams: {episode_uuid}"
            )
        _validate_source_materialization_binding(entry, record)
        if entry.source_scenario_spec_sha256 is not None:
            scenario = SourceScenarioSpec.from_dict(
                record.extras["source_scenario_spec"]
            )
            if record.duration_s is None or abs(
                float(record.duration_s) - scenario.duration_s
            ) > 1e-9:
                raise RuntimeError(
                    f"Persisted duration differs from planned scenario: {episode_uuid}"
                )
        referenced = {
            *record.video_paths.values(),
            record.frame_data_path,
            record.high_rate_path,
            record.events_path,
            record.transition_events_path,
            record.object_states_path,
        }
        if None in referenced or set(record.content_hashes) != referenced:
            raise RuntimeError(
                f"Planned episode has incomplete artifact membership: {episode_uuid}"
            )
        for relative, expected_hash in record.content_hashes.items():
            path = resolve_dataset_path(root, relative)
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"Planned episode artifact is missing or corrupt: {path}")

    ledger, ledger_sha256 = _terminal_failure_ledger(root, plan, record_by_uuid)
    provenance_rows = [dict(row) for row in provenance]
    if any(
        row.get("schema_version") == TERMINAL_FAILURE_PROVENANCE_SCHEMA
        for row in provenance_rows
    ):
        raise ValueError("Caller provenance cannot replace the terminal-failure binding")
    provenance_rows.append(
        {
            "schema_version": TERMINAL_FAILURE_PROVENANCE_SCHEMA,
            "artifact_path": TERMINAL_FAILURE_LEDGER_FILE,
            "artifact_sha256": ledger_sha256,
            "plan_hash": plan.plan_hash,
            "run_plan_sha256": ledger["run_plan_sha256"],
            "terminal_failure_count": ledger["terminal_failure_count"],
            "pilot_readiness_blocked": ledger["pilot_readiness_blocked"],
        }
    )
    return writer.finalize(
        info,
        tasks=tasks,
        cameras=cameras,
        provenance=provenance_rows,
        counterfactual_families=counterfactual_families,
        expected_episode_membership=plan.membership,
        run_plan_sha256=sha256_file(root / RUN_PLAN_FILE),
    )
