"""Block-level planning, execution, and finalization for scale generation.

One *block* is one immutable canonical dataset (plan-run → run-shard →
finalize-run) holding a contiguous slice of a leaf's sampled-scale episodes.
Blocks stay small enough that each ``.run_plan.json`` loads in seconds and
each finalize hash-verifies in minutes, while the runs-manifest binds all
blocks of one campaign together.  Everything produced here is diagnostic and
training-ineligible; release accounting is structurally zero.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

import yaml

from .episode_writer import load_episode_records
from .hashing import sha256_file, sha256_json
from .paths import atomic_write_json, ensure_not_source_path
from .provenance import get_git_commit
from .qc import validate_dataset
from .run_orchestration import RunPlan, finalize_run, plan_run
from .scale_execution import SCALE_EXECUTION_BRIDGE_SCHEMA
from .scale_prepare_worker import SCALE_PREPARE_REQUEST_SCHEMA
from .scale_suite import (
    SCALE_MASTER_SEED,
    ScaleSuiteCase,
    mint_scale_cases,
    scale_block_manifest,
)
from .schema import DatasetInfo, TimeBase
from .source_execution import source_finalization_rows
from .source_preview import SOURCE_PREVIEW_ARROW_EXTRA_FIELDS
from .source_scenario import SourceScenarioSpec
from .video_writer import VideoSpec


SCALE_RUN_SCHEMA = "dynamic-robot-scale-run/v1"
SCALE_BLOCK_MANIFEST_FILE = "scale_suite_manifest.json"
_SCALE_DATASET_UUID_NAMESPACE = uuid.UUID("5d1f7c37-9a54-5e0b-8d0e-1c2a90b7f3d4")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HOURS_CONFIG = _REPOSITORY_ROOT / "configs/scale/hours_v1.yaml"


@dataclass(frozen=True, slots=True)
class ScaleHoursPlan:
    """Resolved campaign targets: hours → episode counts → blocks."""

    episode_seconds: Mapping[str, float]
    target_hours: Mapping[str, float]
    block_episodes: Mapping[str, int]
    shards_per_block: int

    def episode_target(self, leaf_id: str) -> int:
        return math.ceil(
            float(self.target_hours[leaf_id])
            * 3600.0
            / float(self.episode_seconds[leaf_id])
        )

    def block_size(self, leaf_id: str) -> int:
        return int(self.block_episodes.get(leaf_id, self.block_episodes["default"]))

    def blocks(self, leaf_id: str) -> list[dict[str, int]]:
        target = self.episode_target(leaf_id)
        size = self.block_size(leaf_id)
        result = []
        start = 0
        block_index = 0
        while start < target:
            count = min(size, target - start)
            result.append(
                {
                    "block_index": block_index,
                    "episode_start": start,
                    "episode_count": count,
                }
            )
            start += count
            block_index += 1
        return result

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for leaf_id in self.target_hours:
            target = self.episode_target(leaf_id)
            rows.append(
                {
                    "leaf_id": leaf_id,
                    "target_hours": float(self.target_hours[leaf_id]),
                    "episode_seconds": float(self.episode_seconds[leaf_id]),
                    "episode_target": target,
                    "block_size": self.block_size(leaf_id),
                    "block_count": len(self.blocks(leaf_id)),
                }
            )
        return rows


def load_hours_plan(path: str | Path | None = None) -> ScaleHoursPlan:
    source = _DEFAULT_HOURS_CONFIG if path is None else Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("scale hours config must be a mapping")
    if raw.get("schema_version") != "dynamic-robot-scale-hours/v1":
        raise ValueError("unsupported scale hours config schema")
    return ScaleHoursPlan(
        episode_seconds={str(k): float(v) for k, v in raw["episode_seconds"].items()},
        target_hours={str(k): float(v) for k, v in raw["target_hours"].items()},
        block_episodes={str(k): int(v) for k, v in raw["block_episodes"].items()},
        shards_per_block=int(raw["shards_per_block"]),
    )


def block_dataset_root(output_root: str | Path, leaf_id: str, block_index: int) -> Path:
    return Path(output_root) / leaf_id / f"block-{block_index:04d}"


def _prepare_one_declaration(
    case: ScaleSuiteCase,
    episode_index: int,
    generator_git_commit: str,
) -> dict[str, Any]:
    request = {
        "schema_version": SCALE_PREPARE_REQUEST_SCHEMA,
        "scale_case": case.to_dict(),
        "episode_index": episode_index,
        "generator_git_commit": generator_git_commit,
    }
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "dynamic_robot_dataset.common.scale_prepare_worker",
        ),
        cwd=_REPOSITORY_ROOT,
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
            f"isolated scale declaration preparation failed for {case.case_id}: {detail}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"scale declaration preparation returned a non-mapping for {case.case_id}"
        )
    declaration = dict(value)
    if (
        declaration.get("schema_version") != SCALE_EXECUTION_BRIDGE_SCHEMA
        or declaration.get("episode_index") != episode_index
        or declaration.get("episode_uuid") != case.episode_uuid
        or declaration.get("scale_suite_episode_index") != case.episode_index
        or declaration.get("scale_case_sha256") != case.case_sha256
        or declaration.get("generator_git_commit") != generator_git_commit
        or declaration.get("scale_case") != case.to_dict()
    ):
        raise RuntimeError(f"scale declaration identity differs for {case.case_id}")
    raw_spec = declaration.get("source_scenario_spec")
    if not isinstance(raw_spec, Mapping):
        raise RuntimeError(f"scale declaration lacks a spec for {case.case_id}")
    scenario = SourceScenarioSpec.from_dict(raw_spec)
    if (
        scenario.scenario_id != case.case_id
        or declaration.get("source_scenario_spec_sha256") != scenario.spec_hash
    ):
        raise RuntimeError(
            f"scale declaration scenario binding differs for {case.case_id}"
        )
    return declaration


def prepare_scale_declarations_isolated(
    cases: Sequence[ScaleSuiteCase],
    *,
    generator_git_commit: str,
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Prepare every declaration in short-lived worker processes.

    Same native-memory rationale as the preview prepare worker; scale blocks
    additionally parallelize across processes because a block holds thousands
    of cases.  Results are returned in exact case order.
    """

    if not cases:
        raise ValueError("scale preparation requires at least one case")
    if not generator_git_commit:
        raise ValueError("scale declaration preparation lacks a commit")
    if workers <= 0:
        raise ValueError("scale preparation requires at least one worker")
    declarations: list[dict[str, Any] | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _prepare_one_declaration, case, index, generator_git_commit
            ): index
            for index, case in enumerate(cases)
        }
        for future, index in futures.items():
            declarations[index] = future.result()
    assert all(value is not None for value in declarations)
    return [dict(value) for value in declarations if value is not None]


def plan_scale_block(
    leaf_id: str,
    *,
    block_index: int,
    episode_start: int,
    episode_count: int,
    output_root: str | Path,
    shard_count: int = 16,
    prepare_workers: int = 8,
    resume: bool = False,
) -> dict[str, Any]:
    """Mint, prepare, and immutably plan one scale block."""

    root = ensure_not_source_path(block_dataset_root(output_root, leaf_id, block_index))
    scale_suite_id = f"scale-{leaf_id}-block-{block_index:04d}"
    cases = mint_scale_cases(
        leaf_id,
        episode_start=episode_start,
        count=episode_count,
        scale_suite_id=scale_suite_id,
    )
    manifest = scale_block_manifest(cases)
    generator_git_commit = get_git_commit(_REPOSITORY_ROOT)
    started = time.monotonic()
    declarations = prepare_scale_declarations_isolated(
        cases,
        generator_git_commit=generator_git_commit,
        workers=prepare_workers,
    )
    prepare_seconds = time.monotonic() - started
    simulation_rates = sorted(
        {
            int(declaration["source_scenario_spec"]["physics"]["simulation_hz"])
            for declaration in declarations
        }
    )
    resolved_config = {
        "schema_version": SCALE_RUN_SCHEMA,
        "backend": "source_mujoco",
        "purpose": "scale_generation",
        "generator_git_commit": generator_git_commit,
        "scale_suite_id": scale_suite_id,
        "scale_suite_manifest_sha256": manifest["manifest_sha256"],
        "scale_master_seed": SCALE_MASTER_SEED,
        "corpus_leaf_id": leaf_id,
        "block_index": block_index,
        "episode_start": episode_start,
        "episode_count": episode_count,
        "simulation_hz_values": simulation_rates,
        "control_hz": 60,
        "video_hz": 30,
        "views": ["main", "secondary"],
        "arrow_extra_fields": SOURCE_PREVIEW_ARROW_EXTRA_FIELDS,
        "release_state": "review_only_blocked",
        "training_eligible": False,
        "diagnostic_only": True,
    }
    plan = plan_run(
        root,
        resolved_config,
        declarations,
        shard_count=shard_count,
        resume=resume,
        video_spec=VideoSpec(),
        split_settings={
            "strategy": "leakage_aware_stratified",
            "seed": SCALE_MASTER_SEED,
            "train_fraction": 0.80,
            "validation_fraction": 0.10,
            "test_fraction": 0.10,
        },
    )
    manifest_path = root / SCALE_BLOCK_MANIFEST_FILE
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"immutable scale block manifest differs: {manifest_path}")
    else:
        atomic_write_json(manifest_path, manifest)
    return {
        "dataset_root": str(root.resolve()),
        "scale_suite_id": scale_suite_id,
        "run_id": plan.run_id,
        "plan_hash": plan.plan_hash,
        "episode_count": len(plan.episodes),
        "shard_count": plan.shard_count,
        "prepare_seconds": prepare_seconds,
        "prepare_seconds_per_episode": prepare_seconds / len(plan.episodes),
    }


def run_scale_shard_isolated(
    dataset_root: str | Path,
    shard_id: int,
    *,
    max_episodes_per_process: int = 1,
) -> dict[str, Any]:
    """Drive one shard to completion, one fresh worker process per episode.

    Mirrors the preview's per-process isolation rationale.  Terminal
    scene-construction failures are honest committed records; only an
    unchanged ``remaining_count`` across an iteration is an error.
    """

    root = Path(dataset_root).resolve(strict=True)
    last_remaining: int | None = None
    committed_total = 0
    while True:
        command = (
            sys.executable,
            "-m",
            "dynamic_robot_dataset.cli",
            "run-shard",
            "--dataset",
            str(root),
            "--shard-id",
            str(shard_id),
            "--max-episodes",
            str(max_episodes_per_process),
        )
        completed = subprocess.run(
            command,
            cwd=_REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(
                f"scale shard {shard_id} returned malformed JSON: {detail}"
            ) from error
        remaining = int(result.get("remaining_count", -1))
        if remaining < 0:
            raise RuntimeError(f"scale shard {shard_id} reported no progress state")
        # Each isolated invocation reports only its own fresh commits, so the
        # driver-level total is their sum.  Recovered/terminal counts are
        # complete in the final invocation, which walks every receipt once.
        committed_total += int(result.get("committed_count", 0))
        if remaining == 0:
            summary = dict(result)
            summary["committed_count"] = committed_total
            return summary
        if last_remaining is not None and remaining >= last_remaining:
            detail = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(
                f"scale shard {shard_id} stalled at {remaining} remaining: {detail}"
            )
        last_remaining = remaining


def _task_rows(records: Sequence[Any]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.task_index is None:
            raise ValueError("scale record lacks its registry task index")
        row = {
            "task_index": int(record.task_index),
            "family": record.family,
            "subfamily": record.subfamily,
            "corpus_leaf_id": str(
                record.extras.get("source_scenario_spec", {}).get("corpus_leaf_id", "")
            ),
        }
        existing = by_index.get(int(record.task_index))
        if existing is not None and existing != row:
            raise ValueError("registry task index maps to multiple source leaves")
        by_index[int(record.task_index)] = row
    return [by_index[index] for index in sorted(by_index)]


def finalize_scale_block(
    dataset_root: str | Path,
    *,
    deep_video_checks: bool = False,
) -> dict[str, Any]:
    """Finalize, seal, and strict-QC one completed scale block."""

    root = Path(dataset_root).resolve(strict=True)
    generation = json.loads((root / ".generation.json").read_text(encoding="utf-8"))
    resolved = generation["resolved_config"]
    if resolved.get("schema_version") != SCALE_RUN_SCHEMA:
        raise ValueError("dataset root is not a scale block")
    leaf_id = str(resolved["corpus_leaf_id"])
    block_index = int(resolved["block_index"])
    records = load_episode_records(root)
    finalization = source_finalization_rows(records)
    run_plan_sha256 = sha256_file(root / ".run_plan.json")
    simulation_rates = [int(v) for v in resolved["simulation_hz_values"]]
    info = DatasetInfo(
        name=f"scale-{leaf_id}-block-{block_index:04d}",
        dataset_uuid=str(
            uuid.uuid5(_SCALE_DATASET_UUID_NAMESPACE, str(resolved["scale_suite_id"]))
        ),
        description=(
            "Diagnostic sampled-scale source_mujoco rollouts; review-only and "
            "blocked from training until profile calibration and human "
            "acceptance."
        ),
        time_base=TimeBase(
            sim_hz=float(max(simulation_rates)),
            control_hz=60.0,
            video_hz=30.0,
        ),
        generator_version=str(resolved["generator_git_commit"]),
        extras={
            "run_plan_sha256": run_plan_sha256,
            "scale_suite_id": resolved["scale_suite_id"],
            "scale_suite_manifest_sha256": resolved["scale_suite_manifest_sha256"],
            "simulation_hz_values": simulation_rates,
            "episode_simulation_hz_source": "source_scenario_spec.physics.simulation_hz",
            "release_state": "blocked",
            "training_eligible": False,
        },
    )
    finalize_run(
        root,
        info,
        tasks=_task_rows(records),
        cameras=finalization.cameras,
        provenance=finalization.provenance,
    )
    qc_path = root / "qc" / "dataset_report.json"
    if qc_path.exists():
        report = validate_dataset(
            root,
            deep_video_checks=deep_video_checks,
            strict_all=True,
        )
    else:
        report = validate_dataset(
            root,
            deep_video_checks=deep_video_checks,
            write_reports=True,
            report_dir=root / "qc",
            strict_all=True,
        )
    passed = sum(1 for item in report.episodes if item.passed)
    # Campaign gate: hard physics validity.  A completed rollout whose
    # measured outcome differs from its intended branch is a canonical,
    # honest episode; strict QC still fails it (fixed-review semantics), so
    # separate that class from genuine physics/visibility defects.
    mismatch_only_prefixes = (
        "physics_qc_pass=false",
        "strict physics QC: backend physics_qc_pass is not true",
    )
    record_by_uuid = {record.episode_uuid: record for record in records}
    outcome_mismatch_only = 0
    hard_invalid = 0
    for item in report.episodes:
        if item.passed:
            continue
        record = record_by_uuid[item.episode_uuid]
        mismatch = "intended_outcome_mismatch_preserved" in (
            record.quality_flags or ()
        )
        failures = list(getattr(item, "hard_failures", ()) or ())
        only_mismatch_driven = all(
            any(text.startswith(prefix) for prefix in mismatch_only_prefixes)
            for text in failures
        )
        if mismatch and failures and only_mismatch_driven:
            outcome_mismatch_only += 1
        else:
            hard_invalid += 1
    nominal_by_tool: dict[str, list[int]] = {}
    for record in records:
        if record.intended_branch != "nominal_success":
            continue
        bucket = nominal_by_tool.setdefault(str(record.tool_type), [0, 0])
        bucket[0] += 1
        bucket[1] += int(bool(record.task_success))
    return {
        "dataset_root": str(root),
        "episode_count": len(records),
        "strict_qc_passed": report.passed,
        "episodes_qc_passed": passed,
        "episodes_qc_failed": len(report.episodes) - passed,
        "episodes_outcome_mismatch_only": outcome_mismatch_only,
        "episodes_hard_invalid": hard_invalid,
        "hard_validity_rate": (
            (len(report.episodes) - hard_invalid) / len(report.episodes)
            if report.episodes
            else None
        ),
        "nominal_catch_rate_by_tool": {
            tool: {"attempts": counts[0], "successes": counts[1]}
            for tool, counts in sorted(nominal_by_tool.items())
        },
        "unique_seconds": sum(
            float(record.duration_s or 0.0) for record in records
        ),
    }


def load_block_plan(dataset_root: str | Path) -> RunPlan:
    from .run_orchestration import load_run_plan

    return load_run_plan(Path(dataset_root).resolve(strict=True))


__all__ = [
    "SCALE_BLOCK_MANIFEST_FILE",
    "SCALE_RUN_SCHEMA",
    "ScaleHoursPlan",
    "block_dataset_root",
    "finalize_scale_block",
    "load_block_plan",
    "load_hours_plan",
    "plan_scale_block",
    "prepare_scale_declarations_isolated",
    "run_scale_shard_isolated",
]
