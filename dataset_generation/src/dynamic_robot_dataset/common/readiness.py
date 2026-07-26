"""Machine-readable release gates for staged dataset generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .contract_v2 import (
    CounterfactualFamilyRecord,
    validate_counterfactual_family_records,
)
from .episode_writer import load_episode_records, read_parquet_rows
from .embodiments import (
    FORBIDDEN_CUSTOM_ATTACHMENTS,
    validate_production_end_effector,
)
from .hashing import sha256_file, sha256_json
from .native_suite import plan_suite_cases
from .paths import resolve_dataset_path
from .physics_sweeps import sweep_acceptance_evidence
from .schema import DynamicsMode, EpisodeRecord, ReleaseTier
from .statistics import collect_dataset_statistics
from .suites import expand_suite


@dataclass(slots=True, frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    measured: Any
    required: Any
    message: str = ""


@dataclass(slots=True)
class ReadinessReport:
    schema_version: str
    gate_id: str
    dataset_root: str
    passed: bool
    checks: list[ReadinessCheck]
    denominators: dict[str, Any]
    provenance: dict[str, Any]
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "checks": [asdict(check) for check in self.checks],
        }


def load_gate_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Readiness gate config must contain a mapping")
    config = dict(value)
    if config.get("schema_version") != "dynamic-robot-readiness-gate/v1":
        raise ValueError("Unsupported readiness gate schema_version")
    if not config.get("gate_id"):
        raise ValueError("Readiness gate requires gate_id")
    acceptance = config.get("acceptance_suite")
    if isinstance(acceptance, Mapping):
        for value in acceptance.get("required_end_effectors", ()):
            validate_production_end_effector(str(value))
    return config


def _hard_qc_map(
    root: Path, records: Sequence[EpisodeRecord]
) -> tuple[dict[str, bool], list[str], str | None]:
    """Load a canonical QC report and bind it to finalized metadata.

    ``passed`` flags are useful only after the report is proven to describe the
    exact finalized metadata transaction.  The completion marker binds every
    canonical metadata table, not just ``episodes.parquet``; this prevents a
    stale QC artifact from being reused after cameras, provenance, tasks, or
    split assignments change.
    """

    path = root / "qc" / "dataset_report.json"
    if not path.is_file():
        return {}, ["qc/dataset_report.json is missing"], None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return {}, ["QC report root must be a mapping"], sha256_file(path)
    report = dict(value)
    failures: list[str] = []
    if report.get("schema_version") != "dynamic-robot-qc-report/v2":
        failures.append("QC report does not use dynamic-robot-qc-report/v2")
    reported_root_raw = report.get("dataset_root")
    try:
        if not isinstance(reported_root_raw, str) or not reported_root_raw.strip():
            raise ValueError("dataset_root is missing")
        reported_root = Path(reported_root_raw).resolve(strict=True)
    except (FileNotFoundError, OSError, ValueError):
        reported_root = None
    if reported_root != root:
        failures.append("QC report dataset_root does not match the evaluated dataset")
    episodes_path = root / "meta" / "episodes.parquet"
    if not episodes_path.is_file():
        failures.append("Finalized episodes.parquet is missing")
        episodes_hash = None
    else:
        episodes_hash = sha256_file(episodes_path)
        if report.get("dataset_episodes_sha256") != episodes_hash:
            failures.append("QC report is not bound to the current episodes.parquet")
    completion_path = root / "meta" / ".complete.json"
    if not completion_path.is_file():
        completion: Mapping[str, Any] = {}
        failures.append("Finalized metadata completion marker is missing")
    else:
        raw_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion = raw_completion if isinstance(raw_completion, Mapping) else {}
        if not completion:
            failures.append("Finalized metadata completion marker must be a mapping")
        if report.get("metadata_complete_manifest_sha256") != sha256_file(
            completion_path
        ):
            failures.append(
                "QC report is not bound to the current metadata completion manifest"
            )
    completion_hashes = completion.get("content_hashes")
    expected_metadata_hashes = (
        dict(completion_hashes) if isinstance(completion_hashes, Mapping) else {}
    )
    reported_metadata_hashes = report.get("metadata_content_hashes")
    reported_metadata_hashes = (
        dict(reported_metadata_hashes)
        if isinstance(reported_metadata_hashes, Mapping)
        else {}
    )
    if reported_metadata_hashes != expected_metadata_hashes:
        failures.append("QC report metadata content manifest is stale or incomplete")
    for name, expected_hash in sorted(expected_metadata_hashes.items()):
        if Path(name).name != name:
            failures.append(f"metadata completion manifest has unsafe path: {name!r}")
            continue
        metadata_artifact = root / "meta" / name
        if not metadata_artifact.is_file():
            failures.append(f"finalized metadata artifact is missing: {name}")
        elif sha256_file(metadata_artifact) != expected_hash:
            failures.append(f"finalized metadata artifact hash mismatch: {name}")
    expected_uuids = {record.episode_uuid for record in records}
    raw_episode_items = report.get("episodes")
    episode_items = (
        list(raw_episode_items)
        if isinstance(raw_episode_items, Sequence)
        and not isinstance(raw_episode_items, (str, bytes))
        else []
    )
    if not episode_items:
        failures.append("QC report episodes must be a non-empty sequence")
    if any(not isinstance(item, Mapping) for item in episode_items):
        failures.append("QC report episodes must contain mappings")
    valid_items = [item for item in episode_items if isinstance(item, Mapping)]
    reported_uuids = [
        str(item.get("episode_uuid"))
        for item in valid_items
        if item.get("episode_uuid")
    ]
    if len(reported_uuids) != len(set(reported_uuids)):
        failures.append("QC report contains duplicate episode UUIDs")
    if set(reported_uuids) != expected_uuids:
        failures.append("QC report episode membership differs from episodes.parquet")
    episode_map = {
        str(item["episode_uuid"]): bool(item.get("passed", False))
        for item in valid_items
        if item.get("episode_uuid")
    }
    record_by_uuid = {record.episode_uuid: record for record in records}
    for record in records:
        if not record.content_hashes:
            failures.append(
                f"episode {record.episode_uuid} has no content-addressed artifact manifest"
            )
            continue
        for relative, expected_hash in sorted(record.content_hashes.items()):
            try:
                artifact = resolve_dataset_path(root, relative)
            except (TypeError, ValueError) as error:
                failures.append(
                    f"episode {record.episode_uuid} artifact path is unsafe: {relative!r}: {error}"
                )
                continue
            if not artifact.is_file():
                failures.append(
                    f"episode {record.episode_uuid} artifact is missing: {relative}"
                )
            elif sha256_file(artifact) != expected_hash:
                failures.append(
                    f"episode {record.episode_uuid} artifact hash mismatch: {relative}"
                )
    for item in valid_items:
        episode_uuid = str(item.get("episode_uuid") or "")
        record = record_by_uuid.get(episode_uuid)
        if record is not None and item.get("release_eligible") is not record.release_eligible:
            failures.append(
                f"QC report release eligibility disagrees for episode {episode_uuid}"
            )
    raw_global_failures = report.get("global_failures")
    global_failures = (
        list(raw_global_failures)
        if isinstance(raw_global_failures, Sequence)
        and not isinstance(raw_global_failures, (str, bytes))
        else ["QC report global_failures must be a sequence"]
    )
    failures.extend(str(value) for value in global_failures)
    failures.extend(
        f"episode {item.get('episode_uuid')} failed hard QC"
        for item in valid_items
        if item.get("release_eligible") is True and not item.get("passed", False)
    )
    recomputed_report_pass = not global_failures and all(
        bool(item.get("passed", False))
        or item.get("release_eligible") is not True
        for item in valid_items
    )
    if report.get("passed") is not recomputed_report_pass:
        failures.append("QC report top-level passed claim disagrees with episode results")
    return episode_map, failures, sha256_file(path)


def _selector_matches(record: EpisodeRecord, selector: Mapping[str, Any]) -> bool:
    families = {str(value) for value in selector.get("families", ())}
    subfamilies = {str(value) for value in selector.get("subfamilies", ())}
    tiers = {str(value) for value in selector.get("release_tiers", ())}
    if families and record.family not in families:
        return False
    if subfamilies and record.subfamily not in subfamilies:
        return False
    tier = record.release_tier.value if hasattr(record.release_tier, "value") else str(record.release_tier)
    if tiers and tier not in tiers:
        return False
    return True


def _native_record(record: EpisodeRecord) -> bool:
    simulator = record.simulator_name.lower()
    marker = record.extras.get("native_mujoco")
    if marker is None:
        marker = record.extras.get("backend") == "native_mujoco"
    if marker is None or marker is False:
        provenance = record.extras.get("backend_provenance")
        if isinstance(provenance, Mapping):
            marker = provenance.get("backend") == "native_mujoco"
    return "mujoco" in simulator and bool(marker)


def _model_checks(
    config: Mapping[str, Any],
    model_evaluation: Mapping[str, Any] | None,
    *,
    dataset_episodes_sha256: str,
    dataset_qc_report_sha256: str | None,
) -> list[ReadinessCheck]:
    requirements = config.get("model_evaluation") or {}
    if not requirements:
        return []
    if model_evaluation is None:
        return [
            ReadinessCheck(
                "model_evaluation_present",
                False,
                False,
                True,
                "The next scale stage requires an external model-evaluation artifact",
            )
        ]
    checks = [ReadinessCheck("model_evaluation_present", True, True, True)]
    provenance = dict(model_evaluation.get("provenance") or {})
    required_bindings = {
        "dataset_episodes_sha256": dataset_episodes_sha256,
        "dataset_qc_report_sha256": dataset_qc_report_sha256,
    }
    for name, expected in required_bindings.items():
        checks.append(
            ReadinessCheck(
                f"model.provenance.{name}",
                expected is not None and provenance.get(name) == expected,
                provenance.get(name),
                expected,
                "Model evaluation must be bound to this exact dataset and QC report",
            )
        )
    for name in ("model_artifact_sha256", "evaluation_manifest_sha256"):
        value = provenance.get(name)
        valid = isinstance(value, str) and len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )
        checks.append(
            ReadinessCheck(
                f"model.provenance.{name}", valid, value, "lowercase SHA-256"
            )
        )
        path_name = name.removesuffix("_sha256") + "_path"
        raw_path = provenance.get(path_name)
        resolved_path: Path | None = None
        path_error: str | None = None
        if isinstance(raw_path, str) and raw_path.strip():
            try:
                resolved_path = Path(raw_path).resolve(strict=True)
            except (FileNotFoundError, OSError) as error:
                path_error = str(error)
        else:
            path_error = "path is missing"
        actual_hash = (
            sha256_file(resolved_path)
            if resolved_path is not None and resolved_path.is_file()
            else None
        )
        checks.append(
            ReadinessCheck(
                f"model.provenance.{path_name}.content_bound",
                valid and actual_hash == value,
                {
                    "path": raw_path,
                    "declared_sha256": value,
                    "actual_sha256": actual_hash,
                    "error": path_error,
                },
                "existing regular file matching the declared SHA-256",
            )
        )
    metrics = dict(model_evaluation.get("metrics") or model_evaluation)
    for name, requirement in requirements.items():
        if name == "required":
            continue
        measured = metrics.get(name)
        if isinstance(requirement, Mapping):
            if "minimum" in requirement:
                passed = (
                    not isinstance(measured, bool)
                    and isinstance(measured, (int, float))
                    and math.isfinite(float(measured))
                    and float(measured) >= float(requirement["minimum"])
                )
            elif "maximum" in requirement:
                passed = (
                    not isinstance(measured, bool)
                    and isinstance(measured, (int, float))
                    and math.isfinite(float(measured))
                    and float(measured) <= float(requirement["maximum"])
                )
            elif "equals" in requirement:
                expected = requirement["equals"]
                passed = (
                    measured is expected
                    if isinstance(expected, bool)
                    else measured == expected
                    and (
                        not isinstance(measured, (int, float))
                        or isinstance(measured, bool)
                        or math.isfinite(float(measured))
                    )
                )
            else:
                raise ValueError(f"Unsupported model gate requirement for {name}")
        else:
            passed = measured == requirement
        checks.append(ReadinessCheck(f"model.{name}", passed, measured, requirement))
    return checks


def _canonical_suite_config(
    requirement: Mapping[str, Any], suite_name: str
) -> Path | None:
    """Resolve an independently configured suite definition when available."""

    configured = requirement.get("source_config_path")
    candidates: list[Path] = []
    if configured:
        raw = Path(str(configured))
        candidates.append(raw if raw.is_absolute() else Path.cwd() / raw)
    repository_root = Path(__file__).resolve().parents[3]
    candidates.append(repository_root / "configs" / "families" / f"{suite_name}.yaml")
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if resolved.is_file():
            return resolved
    return None


def _negative_control_evidence(
    root: Path,
    planned_rows: Sequence[Mapping[str, Any]],
    records_by_uuid: Mapping[str, EpisodeRecord],
) -> tuple[bool, dict[str, Any]]:
    """Validate the four quarantined controls without demanding ordinary QC.

    These controls exist to prove that assisted and scripted dynamics are
    detected.  They must remain non-release data, carry the correct tier, and
    contain frame-level evidence for the declared mechanism/motion mode.
    """

    controls = [
        item
        for item in planned_rows
        if isinstance(item.get("case"), Mapping)
        and item["case"].get("category_id") == "negative_controls"
    ]
    expected = {
        "scripted_bounce": (DynamicsMode.SCRIPTED_MOTION, ReleaseTier.SCRIPTED_MOTION),
        "legacy_motion_proxy": (
            DynamicsMode.SCRIPTED_MOTION,
            ReleaseTier.SCRIPTED_MOTION,
        ),
        "assisted_latch": (DynamicsMode.ASSISTED_CONTACT, ReleaseTier.ASSISTED_CONTACT),
        "equality_grasp": (DynamicsMode.ASSISTED_CONTACT, ReleaseTier.ASSISTED_CONTACT),
    }
    problems: list[str] = []
    measured: dict[str, Any] = {}
    names = [str(item["case"].get("subfamily")) for item in controls]
    if len(controls) != 4 or Counter(names) != Counter(expected):
        problems.append(
            f"negative-control plan must contain each required singleton once, got {names}"
        )
    for item in controls:
        episode_uuid = str(item.get("episode_uuid") or "")
        name = str(item["case"].get("subfamily"))
        record = records_by_uuid.get(episode_uuid)
        if record is None:
            problems.append(f"negative control {name} is not committed")
            continue
        expected_pair = expected.get(name)
        if expected_pair is None:
            continue
        expected_mode, expected_tier = expected_pair
        item_problems: list[str] = []
        if record.release_eligible:
            item_problems.append("is release eligible")
        if record.dynamics_mode != expected_mode:
            item_problems.append(
                f"dynamics_mode={record.dynamics_mode.value}, expected {expected_mode.value}"
            )
        if record.release_tier != expected_tier:
            item_problems.append(
                f"release_tier={record.release_tier.value}, expected {expected_tier.value}"
            )
        if not record.quality_flags:
            item_problems.append("lacks an explicit quarantine/non-production quality flag")
        frame_rows: list[dict[str, Any]] = []
        if not record.frame_data_path:
            item_problems.append("lacks frame data")
        else:
            try:
                frame_path = resolve_dataset_path(root, record.frame_data_path)
                frame_rows = read_parquet_rows(frame_path)
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                item_problems.append(f"frame evidence could not be loaded: {error}")
        if frame_rows and not all(bool(row.get("legacy.quarantine")) for row in frame_rows):
            item_problems.append("frame rows do not all carry legacy.quarantine=true")
        if expected_mode == DynamicsMode.SCRIPTED_MOTION:
            if any(bool(row.get("assistance.active", False)) for row in frame_rows):
                item_problems.append("scripted control unexpectedly activates assistance")
            if frame_rows and not all(
                bool(row.get("dynamics.scripted_active", False))
                for row in frame_rows
            ):
                item_problems.append(
                    "frame rows do not continuously evidence scripted dynamics"
                )
        else:
            flag = "latch_active" if name == "assisted_latch" else "equality_constraint_active"
            if not bool(record.assistance.get(flag, False)):
                item_problems.append(f"assistance summary does not declare {flag}")
            if record.assistance.get("constraint_activation_time") is None:
                item_problems.append("assistance activation time is missing")
            if record.assistance.get("constraint_deactivation_time") is None:
                item_problems.append("assistance deactivation time is missing")
            if frame_rows and not any(
                bool(row.get("assistance.active", False))
                and bool(row.get(f"assistance.{flag}", False))
                for row in frame_rows
            ):
                item_problems.append(f"frame rows never observe active {flag}")
        measured[name] = {
            "episode_uuid": episode_uuid,
            "dynamics_mode": record.dynamics_mode.value,
            "release_tier": record.release_tier.value,
            "release_eligible": record.release_eligible,
            "frame_count": len(frame_rows),
            "problems": item_problems,
        }
        problems.extend(f"{name}: {problem}" for problem in item_problems)
    return not problems, {"controls": measured, "problems": problems}


def _acceptance_checks(
    config: Mapping[str, Any], report_path: Path | None
) -> tuple[list[ReadinessCheck], dict[str, Any]]:
    """Recompute suite acceptance from immutable plan and finalized records.

    The execution report is an index into the evidence chain, not an authority:
    its ``passed_execution``, ``full_native``, counts, backend maps, and gate
    booleans are compared with values derived here.
    """

    requirement = config.get("acceptance_suite")
    if not isinstance(requirement, Mapping):
        return [], {}
    if report_path is None:
        return [
            ReadinessCheck(
                "acceptance_suite_present",
                False,
                None,
                dict(requirement),
                "A content-addressed native acceptance report is required",
            )
        ], {}
    raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    report = raw_report if isinstance(raw_report, Mapping) else {}
    suite_name = str(requirement.get("suite_name") or "")
    branch_count = int(requirement.get("branch_count", 160))
    dataset_root_raw = report.get("dataset_root")
    try:
        if not isinstance(dataset_root_raw, str) or not dataset_root_raw.strip():
            raise ValueError("dataset_root is missing")
        acceptance_root = Path(dataset_root_raw).resolve(strict=True)
    except (FileNotFoundError, OSError, ValueError):
        acceptance_root = None
    canonical_report = acceptance_root / ".suite_execution.json" if acceptance_root else None
    plan_path = acceptance_root / ".suite_plan.json" if acceptance_root else None
    plan: Mapping[str, Any] = {}
    if plan_path and plan_path.is_file():
        raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if isinstance(raw_plan, Mapping):
            plan = raw_plan
    planned_rows = [
        item for item in plan.get("planned_episodes", ()) if isinstance(item, Mapping)
    ]
    planned_uuid_list = [str(item.get("episode_uuid") or "") for item in planned_rows]
    planned_uuids = {value for value in planned_uuid_list if value}
    case_indices: list[int] = []
    invalid_case_index = False
    for item in planned_rows:
        if not isinstance(item.get("case"), Mapping):
            invalid_case_index = True
            continue
        try:
            case_indices.append(int(item["case"].get("case_index", -1)))
        except (TypeError, ValueError):
            invalid_case_index = True
    recomputed_plan_backend_counts = Counter(
        str(item.get("execution_backend") or "<missing>") for item in planned_rows
    )
    recomputed_plan_full_native = bool(planned_rows) and set(
        recomputed_plan_backend_counts
    ) == {"native_mujoco"}

    suite_config_path = _canonical_suite_config(requirement, suite_name)
    expected_config_hash = (
        sha256_file(suite_config_path) if suite_config_path is not None else None
    )
    expected_cases: list[dict[str, Any]] | None = None
    expected_planned: list[Any] = []
    suite_requirements: dict[str, Any] = {}
    suite_config_error: str | None = None
    if suite_config_path is not None:
        try:
            expanded_cases = expand_suite(suite_config_path)
            expected_cases = [case.to_dict() for case in expanded_cases]
            expected_planned = plan_suite_cases(expanded_cases)
            suite_value = yaml.safe_load(suite_config_path.read_text(encoding="utf-8"))
            if isinstance(suite_value, Mapping):
                suite_requirements = dict(suite_value.get("requirements") or {})
        except (OSError, ValueError, TypeError, KeyError) as error:
            suite_config_error = str(error)
    planned_cases = [
        dict(item["case"]) for item in planned_rows if isinstance(item.get("case"), Mapping)
    ]

    acceptance_records: list[EpisodeRecord] = []
    dataset_error = ""
    qc_pass_map: dict[str, bool] = {}
    qc_failures: list[str] = []
    qc_hash: str | None = None
    finalized_paths = [] if acceptance_root is None else [
        acceptance_root / "meta" / ".complete.json",
        acceptance_root / "meta" / "info.json",
        acceptance_root / "meta" / "episodes.parquet",
    ]
    finalized = bool(finalized_paths) and all(path.is_file() for path in finalized_paths)
    if finalized and acceptance_root is not None:
        try:
            acceptance_records = load_episode_records(acceptance_root)
            qc_pass_map, qc_failures, qc_hash = _hard_qc_map(
                acceptance_root, acceptance_records
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            dataset_error = str(error)
    else:
        dataset_error = "acceptance dataset is not atomically finalized"
    records_by_uuid = {record.episode_uuid: record for record in acceptance_records}
    committed_uuids = set(records_by_uuid)
    exact_membership = (
        len(planned_rows) == len(planned_uuid_list)
        == len(planned_uuids)
        == branch_count
        and committed_uuids == planned_uuids
    )

    backend_problems: list[str] = []
    committed_backend_counts: Counter[str] = Counter()
    for item in planned_rows:
        episode_uuid = str(item.get("episode_uuid") or "")
        expected_backend = str(item.get("execution_backend") or "")
        record = records_by_uuid.get(episode_uuid)
        if record is None:
            continue
        actual_backend = "native_mujoco" if _native_record(record) else "diagnostic_quarantine"
        committed_backend_counts[actual_backend] += 1
        if actual_backend != expected_backend:
            backend_problems.append(
                f"{episode_uuid}: planned {expected_backend}, measured {actual_backend}"
            )
        expected_scenario_hash = item.get("scenario_spec_hash")
        backend_provenance = record.extras.get("backend_provenance")
        actual_scenario_hash = (
            backend_provenance.get("scenario_hash")
            if isinstance(backend_provenance, Mapping)
            else None
        )
        if expected_backend == "native_mujoco" and (
            not isinstance(expected_scenario_hash, str)
            or actual_scenario_hash != expected_scenario_hash
        ):
            backend_problems.append(
                f"{episode_uuid}: compiled scenario hash differs from immutable plan"
            )
    recomputed_full_native = (
        exact_membership
        and recomputed_plan_full_native
        and bool(acceptance_records)
        and all(_native_record(record) for record in acceptance_records)
    )

    required_end_effectors = {
        str(value) for value in requirement.get("required_end_effectors", ())
    }
    forbid_custom_attachments = bool(
        requirement.get("forbid_custom_flange_attachments", False)
    )
    enforce_embodiment_contract = bool(required_end_effectors) or forbid_custom_attachments
    observed_end_effectors: set[str] = set()
    embodiment_problems: list[str] = []
    for record in (acceptance_records if enforce_embodiment_contract else ()):
        raw_end_effector = str(record.tool_type or "").strip()
        try:
            end_effector = validate_production_end_effector(raw_end_effector)
        except ValueError as error:
            embodiment_problems.append(f"{record.episode_uuid}: {error}")
            continue
        observed_end_effectors.add(end_effector)
        if required_end_effectors and end_effector not in required_end_effectors:
            embodiment_problems.append(
                f"{record.episode_uuid}: undeclared end effector {end_effector!r}"
            )
        backend_provenance = record.extras.get("backend_provenance")
        if isinstance(backend_provenance, Mapping):
            serialized = json.dumps(backend_provenance, sort_keys=True).lower()
            forbidden = sorted(
                value for value in FORBIDDEN_CUSTOM_ATTACHMENTS if value in serialized
            )
            if forbidden:
                embodiment_problems.append(
                    f"{record.episode_uuid}: custom attachment provenance {forbidden}"
                )
    missing_end_effectors = sorted(required_end_effectors - observed_end_effectors)
    if missing_end_effectors:
        embodiment_problems.append(
            f"acceptance coverage is missing end effectors {missing_end_effectors}"
        )
    embodiment_contract_passed = not enforce_embodiment_contract or (
        bool(acceptance_records)
        and not embodiment_problems
        and (
            not forbid_custom_attachments
            or all(
                str(record.tool_type or "") not in FORBIDDEN_CUSTOM_ATTACHMENTS
                for record in acceptance_records
            )
        )
    )

    target_qc_uuids = {
        record.episode_uuid
        for record in acceptance_records
        if record.release_eligible
    }
    target_qc_uuids.update(
        str(item.get("episode_uuid"))
        for item in planned_rows
        if item.get("execution_backend") == "native_mujoco"
    )
    failed_target_qc = sorted(
        episode_uuid
        for episode_uuid in target_qc_uuids
        if qc_pass_map.get(episode_uuid) is not True
    )
    target_qc_passed = (
        not dataset_error
        and not qc_failures
        and set(qc_pass_map) == committed_uuids
        and not failed_target_qc
    )

    negative_passed, negative_evidence = (
        _negative_control_evidence(acceptance_root, planned_rows, records_by_uuid)
        if acceptance_root is not None
        else (False, {"problems": ["acceptance dataset root is unavailable"]})
    )

    view_style_problems: list[str] = []
    for item in planned_rows:
        episode_uuid = str(item.get("episode_uuid") or "")
        case = item.get("case") if isinstance(item.get("case"), Mapping) else {}
        record = records_by_uuid.get(episode_uuid)
        if record is None:
            continue
        expected_views = {
            value if str(value).startswith("observation.") else f"observation.images.{value}"
            for value in case.get("views", ())
        }
        if set(record.video_paths) != expected_views:
            view_style_problems.append(
                f"{episode_uuid}: video streams {sorted(record.video_paths)} != {sorted(expected_views)}"
            )
        measured_style = str(
            record.randomization.get("background_style")
            or record.randomization.get("scene_style")
            or ""
        )
        if measured_style != str(case.get("scene_style") or ""):
            view_style_problems.append(
                f"{episode_uuid}: scene style {measured_style!r} != {case.get('scene_style')!r}"
            )

    measured_outcomes: dict[str, list[str]] = {}
    outcome_problems: list[str] = []
    required_outcome_families: Sequence[str] = tuple(
        str(value)
        for value in suite_requirements.get(
            "require_measured_success_and_failure", ()
        )
    )
    intended_to_actual = {
        "success": "success",
        "success_seeking": "success",
        "near_miss": "near_miss",
        "contact_failure": "contact_failure",
        "no_op": "no_op",
        "bad_action": "wrong_action",
        "wrong_action": "wrong_action",
    }
    for family in required_outcome_families:
        actual = {
            record.actual_outcome_class.value
            for record in acceptance_records
            if record.family == family
        }
        measured_outcomes[family] = sorted(actual)
        expected = {
            intended_to_actual[str(item["episode_plan"].get("intended_branch"))]
            for item in planned_rows
            if isinstance(item.get("episode_plan"), Mapping)
            and item["episode_plan"].get("family") == family
            and str(item["episode_plan"].get("intended_branch")) in intended_to_actual
        }
        if "success" not in actual or not (actual - {"success", "unverified"}):
            outcome_problems.append(f"{family}: lacks measured success and failure")
        missing = sorted(expected - actual)
        if missing:
            outcome_problems.append(f"{family}: missing planned classes {missing}")

    counterfactual_problems: list[str] = []
    try:
        declarations = [
            CounterfactualFamilyRecord.from_dict(item)
            for item in plan.get("counterfactual_families", ())
            if isinstance(item, Mapping)
        ]
        derived = {
            record.episode_uuid: {
                "derived_initial_state_hash": record.extras.get(
                    "derived_initial_state_hash"
                ),
                "derived_action_hash": record.extras.get("derived_action_hash"),
            }
            for record in acceptance_records
        }
        counterfactual_problems = validate_counterfactual_family_records(
            declarations, acceptance_records, derived_by_uuid=derived
        )
    except (ValueError, TypeError, KeyError) as error:
        counterfactual_problems = [f"counterfactual evidence could not be validated: {error}"]

    generation: Mapping[str, Any] = {}
    generation_path = acceptance_root / ".generation.json" if acceptance_root else None
    if generation_path and generation_path.is_file():
        raw_generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if isinstance(raw_generation, Mapping):
            generation = raw_generation
    writer_hash = plan.get("writer_config_hash")
    identity_ok = (
        plan.get("schema_version") == "dynamic-robot-suite-plan-ledger/v1"
        and plan.get("suite_name") == suite_name
        and plan.get("planned_case_count") == branch_count
        and len(planned_rows) == branch_count
        and len(planned_uuids) == branch_count
        and not invalid_case_index
        and sorted(case_indices) == list(range(branch_count))
        and all(
            isinstance(item.get("case"), Mapping)
            and item["case"].get("suite_name") == suite_name
            for item in planned_rows
        )
    )
    config_bound = (
        suite_config_path is not None
        and suite_config_error is None
        and plan.get("source_config_sha256") == expected_config_hash
        and report.get("source_config_sha256") == expected_config_hash
        and expected_cases == planned_cases
    )
    plan_hash = sha256_file(plan_path) if plan_path and plan_path.is_file() else None
    content_bound = (
        plan_hash is not None
        and report.get("suite_plan_sha256") == plan_hash
        and report.get("writer_config_hash") == writer_hash
        and generation.get("config_hash") == writer_hash
        and report.get("committed_episode_uuid_set_sha256")
        == sha256_json(sorted(committed_uuids))
    )

    attempts: list[Mapping[str, Any]] = []
    attempt_problems: list[str] = []
    attempt_root = acceptance_root / ".suite_attempts" if acceptance_root else None
    if attempt_root and attempt_root.is_dir():
        for path in sorted(attempt_root.glob("case-*.json")):
            raw_attempt = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw_attempt, Mapping):
                attempts.append(raw_attempt)
            else:
                attempt_problems.append(f"{path.name}: root is not a mapping")
    attempt_uuids = [str(item.get("episode_uuid") or "") for item in attempts]
    if len(attempts) != branch_count or set(attempt_uuids) != planned_uuids:
        attempt_problems.append("attempt membership does not exactly match the plan")
    if len(attempt_uuids) != len(set(attempt_uuids)):
        attempt_problems.append("attempt records contain duplicate episode UUIDs")
    for item in attempts:
        if item.get("status") != "committed":
            attempt_problems.append(
                f"{item.get('episode_uuid')}: attempt status is {item.get('status')!r}"
            )
        if item.get("config_hash") != writer_hash:
            attempt_problems.append(
                f"{item.get('episode_uuid')}: attempt writer config hash differs"
            )

    native_physics_failures = sorted(
        record.episode_uuid
        for record in acceptance_records
        if record.episode_uuid in target_qc_uuids and _native_record(record)
        and not record.physics_qc_pass
    )
    required_physics_sweeps = {
        str(name): dict(value)
        for name, value in dict(
            suite_requirements.get("require_measured_physics_sweeps") or {}
        ).items()
    }
    sweep_evidence = sweep_acceptance_evidence(
        expected_planned,
        acceptance_records,
        required_physics_sweeps,
    )
    core_gate_map = {
        "plan_identity": identity_ok,
        "configured_suite_exactly_matches_plan": config_bound,
        "report_plan_generation_content_chain": content_bound,
        "exact_finalized_membership": exact_membership and not dataset_error,
        "planned_and_measured_backends_match": not backend_problems,
        "real_gripper_embodiment_contract": embodiment_contract_passed,
        "attempts_complete_without_retry_or_failure": not attempt_problems,
        "views_and_styles_match_plan": not view_style_problems,
        "native_physics_qc_pass": not native_physics_failures,
        "required_physics_sweeps_measured_and_monotonic": (
            not required_physics_sweeps or sweep_evidence["passed"]
        ),
        "target_records_pass_bound_qc": target_qc_passed,
        "negative_controls_quarantined_and_evidenced": negative_passed,
        "required_rigid_outcomes_observed": not outcome_problems,
        "counterfactual_membership_and_invariants": not counterfactual_problems,
    }
    recomputed_core_pass = all(core_gate_map.values())
    # The suite's execution contract is intentionally stronger than a caller
    # that merely wants to inspect it: the acceptance suite passes only when
    # every case is native.  Gate requirements decide whether that status is a
    # blocking prerequisite for the caller.
    recomputed_execution_pass = recomputed_core_pass and recomputed_full_native
    require_full_native = bool(requirement.get("require_full_native", False))
    require_passed_execution = bool(
        requirement.get("require_passed_execution", False)
    )

    checks = [
        ReadinessCheck(
            "acceptance_suite.report_schema",
            report.get("schema_version") == "dynamic-robot-suite-execution/v1",
            report.get("schema_version"),
            "dynamic-robot-suite-execution/v1",
        ),
        ReadinessCheck(
            "acceptance_suite.canonical_report_path",
            canonical_report == report_path,
            str(report_path),
            None if canonical_report is None else str(canonical_report),
        ),
        ReadinessCheck(
            "acceptance_suite.plan_present",
            bool(plan_path and plan_path.is_file()),
            None if plan_path is None else str(plan_path),
            "existing immutable .suite_plan.json",
        ),
        ReadinessCheck(
            "acceptance_suite.plan_schema_and_identity",
            identity_ok,
            {
                "schema_version": plan.get("schema_version"),
                "suite_name": plan.get("suite_name"),
                "declared_count": plan.get("planned_case_count"),
                "row_count": len(planned_rows),
                "unique_uuid_count": len(planned_uuids),
                "case_indices": sorted(case_indices),
            },
            {"suite_name": suite_name, "branch_count": branch_count},
        ),
        ReadinessCheck(
            "acceptance_suite.source_config_bound",
            config_bound,
            {
                "path": None if suite_config_path is None else str(suite_config_path),
                "expected_sha256": expected_config_hash,
                "plan_sha256": plan.get("source_config_sha256"),
                "report_sha256": report.get("source_config_sha256"),
                "case_rows_match": expected_cases == planned_cases,
                "error": suite_config_error,
            },
            "current configured suite identity, hash, and expanded case list",
        ),
        ReadinessCheck(
            "acceptance_suite.artifact_chain_bound",
            content_bound,
            {
                "plan_sha256": plan_hash,
                "report_plan_sha256": report.get("suite_plan_sha256"),
                "plan_writer_config_hash": writer_hash,
                "report_writer_config_hash": report.get("writer_config_hash"),
                "generation_config_hash": generation.get("config_hash"),
                "committed_episode_uuid_set_sha256": sha256_json(
                    sorted(committed_uuids)
                ),
            },
            "report, plan, generation marker, and finalized episode membership",
        ),
        ReadinessCheck(
            "acceptance_suite.finalized_dataset",
            finalized and not dataset_error,
            {"paths": [str(path) for path in finalized_paths], "error": dataset_error or None},
            "atomically finalized dataset metadata",
        ),
        ReadinessCheck(
            "acceptance_suite.exact_episode_membership",
            exact_membership,
            {
                "planned_count": len(planned_rows),
                "planned_unique_count": len(planned_uuids),
                "committed_count": len(committed_uuids),
                "missing": sorted(planned_uuids - committed_uuids),
                "extra": sorted(committed_uuids - planned_uuids),
            },
            f"exactly {branch_count} unique planned/finalized UUIDs",
        ),
        ReadinessCheck(
            "acceptance_suite.backend_counts_recomputed",
            not backend_problems
            and dict(recomputed_plan_backend_counts)
            == dict(plan.get("execution_backend_counts") or {})
            and dict(recomputed_plan_backend_counts)
            == dict(report.get("planned_execution_backend_counts") or {})
            and dict(committed_backend_counts)
            == dict(report.get("committed_execution_backend_counts") or {}),
            {
                "planned_recomputed": dict(recomputed_plan_backend_counts),
                "committed_recomputed": dict(committed_backend_counts),
                "problems": backend_problems,
            },
            "backend counts and per-record provenance agree with plan and report",
        ),
        ReadinessCheck(
            "acceptance_suite.real_gripper_embodiments",
            embodiment_contract_passed,
            {
                "observed_end_effectors": sorted(observed_end_effectors),
                "problems": embodiment_problems,
            },
            {
                "required_end_effectors": sorted(required_end_effectors),
                "forbid_custom_flange_attachments": forbid_custom_attachments,
            },
            "acceptance data must use only declared real Franka/Robotiq end effectors",
        ),
        ReadinessCheck(
            "acceptance_suite.qc_v2_bound_and_target_records_pass",
            target_qc_passed,
            {
                "qc_report_sha256": qc_hash,
                "artifact_failures": qc_failures,
                "target_episode_count": len(target_qc_uuids),
                "failed_target_episode_uuids": failed_target_qc,
            },
            "all release-eligible or planned-native candidates pass bound QC; quarantined diagnostics need not",
        ),
        ReadinessCheck(
            "acceptance_suite.qc_v2_bound_and_passed",
            target_qc_passed,
            {
                "qc_report_sha256": qc_hash,
                "artifact_failures": qc_failures,
                "failed_target_episode_uuids": failed_target_qc,
            },
            "compatibility alias: bound QC passes for release/native targets",
        ),
        ReadinessCheck(
            "acceptance_suite.negative_controls",
            negative_passed,
            negative_evidence,
            "four nonrelease scripted/assisted controls with frame evidence",
        ),
        ReadinessCheck(
            "acceptance_suite.core_gates_recomputed",
            recomputed_core_pass,
            {
                "checks": core_gate_map,
                "attempt_problems": attempt_problems,
                "view_style_problems": view_style_problems,
                "native_physics_failures": native_physics_failures,
                "physics_sweep_evidence": sweep_evidence,
                "outcome_problems": outcome_problems,
                "measured_outcomes": measured_outcomes,
                "counterfactual_problems": counterfactual_problems,
            },
            "all record/plan-derived core acceptance gates",
        ),
        ReadinessCheck(
            "acceptance_suite.report_claims_match_recomputed",
            report.get("full_native") is recomputed_full_native
            and report.get("passed_execution") is recomputed_execution_pass
            and report.get("planned_case_count") == len(planned_rows)
            and report.get("committed_episode_count") == len(acceptance_records),
            {
                "reported_full_native": report.get("full_native"),
                "recomputed_full_native": recomputed_full_native,
                "reported_passed_execution": report.get("passed_execution"),
                "recomputed_passed_execution": recomputed_execution_pass,
            },
            "execution report claims equal record/plan-derived values",
        ),
        ReadinessCheck(
            "acceptance_suite.full_native",
            (not require_full_native) or recomputed_full_native,
            recomputed_full_native,
            require_full_native,
            "require_full_native is evaluated from plan and record provenance",
        ),
        ReadinessCheck(
            "acceptance_suite.passed_execution",
            (not require_passed_execution) or recomputed_execution_pass,
            recomputed_execution_pass,
            require_passed_execution,
            "require_passed_execution is evaluated from recomputed evidence",
        ),
    ]
    hashes = {
        "acceptance_dataset_root": (
            None if acceptance_root is None else str(acceptance_root)
        ),
        "acceptance_report_path": str(report_path),
        "acceptance_report_sha256": sha256_file(report_path),
        "acceptance_plan_sha256": plan_hash,
        "acceptance_episodes_sha256": (
            sha256_file(acceptance_root / "meta" / "episodes.parquet")
            if acceptance_root is not None
            and (acceptance_root / "meta" / "episodes.parquet").is_file()
            else None
        ),
        "acceptance_qc_report_sha256": qc_hash,
        "acceptance_source_config_sha256": expected_config_hash,
    }
    return checks, hashes


def canonical_readiness_report_path(dataset_root: Path, gate_id: str) -> Path:
    """Canonical immutable location for a readiness artifact used downstream."""

    return dataset_root / "qc" / "readiness" / f"{gate_id}.json"


def _prerequisite_checks(
    requirement: Mapping[str, Any], report_path: Path | None
) -> tuple[list[ReadinessCheck], dict[str, Any]]:
    """Validate and content-bind a prerequisite readiness report."""

    required_gate_id = str(requirement.get("gate_id") or "")
    if report_path is None:
        return [
            ReadinessCheck(
                "prerequisite_gate_present",
                False,
                None,
                dict(requirement),
                "A canonical passed prerequisite readiness artifact is required",
            )
        ], {}
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    prior = raw if isinstance(raw, Mapping) else {}
    prior_root_raw = prior.get("dataset_root")
    try:
        if not isinstance(prior_root_raw, str) or not prior_root_raw.strip():
            raise ValueError("dataset_root is missing")
        prior_root = Path(prior_root_raw).resolve(strict=True)
    except (FileNotFoundError, OSError, ValueError):
        prior_root = None
    canonical_path = (
        canonical_readiness_report_path(prior_root, required_gate_id)
        if prior_root is not None
        else None
    )
    raw_prior_checks = prior.get("checks")
    prior_check_sequence = (
        list(raw_prior_checks)
        if isinstance(raw_prior_checks, Sequence)
        and not isinstance(raw_prior_checks, (str, bytes))
        else []
    )
    checks_raw = [item for item in prior_check_sequence if isinstance(item, Mapping)]
    check_names = [str(item.get("name") or "") for item in checks_raw]
    failed_check_names = [
        str(item.get("name") or "")
        for item in checks_raw
        if item.get("passed") is not True
    ]
    blockers = [str(value) for value in prior.get("blockers", ())]
    internal_consistency = (
        len(checks_raw) == len(prior_check_sequence)
        and len(check_names) == len(set(check_names))
        and all(name for name in check_names)
        and all(isinstance(item.get("passed"), bool) for item in checks_raw)
        and sorted(blockers) == sorted(failed_check_names)
        and prior.get("passed") == (not failed_check_names)
    )

    provenance = prior.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    artifact_problems: list[str] = []
    episodes_hash: str | None = None
    qc_hash: str | None = None
    prior_records: list[EpisodeRecord] = []
    if prior_root is None:
        artifact_problems.append("prerequisite dataset_root is unavailable")
    else:
        try:
            prior_records = load_episode_records(prior_root)
            episodes_hash = sha256_file(prior_root / "meta" / "episodes.parquet")
            _, prior_qc_failures, qc_hash = _hard_qc_map(prior_root, prior_records)
            artifact_problems.extend(prior_qc_failures)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            artifact_problems.append(f"prerequisite dataset/QC could not be loaded: {error}")
    if provenance.get("dataset_episodes_sha256") != episodes_hash:
        artifact_problems.append("prerequisite episodes hash is stale")
    if provenance.get("qc_report_sha256") != qc_hash:
        artifact_problems.append("prerequisite QC report hash is stale")
    if prior_root is not None:
        completion_path = prior_root / "meta" / ".complete.json"
        completion_hash = (
            sha256_file(completion_path) if completion_path.is_file() else None
        )
        if provenance.get("metadata_complete_manifest_sha256") != completion_hash:
            artifact_problems.append(
                "prerequisite finalized metadata manifest hash is stale"
            )

    repository_root = Path(__file__).resolve().parents[3]
    gate_matches: list[Path] = []
    for candidate in sorted((repository_root / "configs" / "release_gates").glob("*.yaml")):
        try:
            candidate_config = load_gate_config(candidate)
        except (OSError, ValueError, TypeError):
            continue
        if candidate_config.get("gate_id") == required_gate_id:
            gate_matches.append(candidate)
    gate_path = gate_matches[0] if len(gate_matches) == 1 else None
    gate_hash = sha256_file(gate_path) if gate_path is not None else None
    if gate_path is None:
        artifact_problems.append(
            "exactly one installed prerequisite gate config must match gate_id"
        )
    if provenance.get("gate_config_sha256") != gate_hash:
        artifact_problems.append("prerequisite gate config hash is stale")

    acceptance_report_raw = provenance.get("acceptance_report_path")
    acceptance_path: Path | None = None
    if acceptance_report_raw:
        try:
            acceptance_path = Path(str(acceptance_report_raw)).resolve(strict=True)
        except (FileNotFoundError, OSError):
            artifact_problems.append("prerequisite acceptance report path is unavailable")
    else:
        artifact_problems.append("prerequisite provenance lacks acceptance_report_path")
    acceptance_checks: list[ReadinessCheck] = []
    acceptance_hashes: dict[str, Any] = {}
    if gate_path is not None:
        prior_gate_config = load_gate_config(gate_path)
        if bool(prior_gate_config.get("require_no_sparse_strata", False)):
            split_diagnostics_path = (
                prior_root / "meta" / "split_diagnostics.json"
                if prior_root is not None
                else None
            )
            splits_path = (
                prior_root / "meta" / "splits.parquet"
                if prior_root is not None
                else None
            )
            split_diagnostics_hash = (
                sha256_file(split_diagnostics_path)
                if split_diagnostics_path is not None
                and split_diagnostics_path.is_file()
                else None
            )
            splits_hash = (
                sha256_file(splits_path)
                if splits_path is not None and splits_path.is_file()
                else None
            )
            if provenance.get("split_diagnostics_sha256") != split_diagnostics_hash:
                artifact_problems.append(
                    "prerequisite split diagnostics hash is stale"
                )
            if provenance.get("splits_parquet_sha256") != splits_hash:
                artifact_problems.append("prerequisite split table hash is stale")
            if split_diagnostics_path is None or not split_diagnostics_path.is_file():
                artifact_problems.append("prerequisite split diagnostics are missing")
            else:
                split_diagnostics = json.loads(
                    split_diagnostics_path.read_text(encoding="utf-8")
                )
                if (
                    not isinstance(split_diagnostics, Mapping)
                    or split_diagnostics.get("schema_version")
                    != "dynamic-robot-split-diagnostics/v2"
                    or split_diagnostics.get("dataset_episodes_sha256")
                    != episodes_hash
                    or split_diagnostics.get("splits_parquet_sha256") != splits_hash
                    or split_diagnostics.get("sparse_strata")
                ):
                    artifact_problems.append(
                        "prerequisite split diagnostics are stale or sparse"
                    )
        acceptance_checks, acceptance_hashes = _acceptance_checks(
            prior_gate_config, acceptance_path
        )
        if any(not check.passed for check in acceptance_checks):
            artifact_problems.append(
                "prerequisite acceptance evidence no longer passes recomputation"
            )
        for name in (
            "acceptance_report_sha256",
            "acceptance_plan_sha256",
            "acceptance_episodes_sha256",
            "acceptance_qc_report_sha256",
            "acceptance_source_config_sha256",
        ):
            if provenance.get(name) != acceptance_hashes.get(name):
                artifact_problems.append(f"prerequisite {name} is stale")
        if prior_gate_config.get("model_evaluation"):
            model_evaluation_raw = provenance.get("model_evaluation_path")
            model_evaluation_path: Path | None = None
            if isinstance(model_evaluation_raw, str) and model_evaluation_raw:
                try:
                    model_evaluation_path = Path(model_evaluation_raw).resolve(
                        strict=True
                    )
                except (FileNotFoundError, OSError):
                    pass
            if model_evaluation_path is None or not model_evaluation_path.is_file():
                artifact_problems.append(
                    "prerequisite model-evaluation artifact is unavailable"
                )
            elif provenance.get("model_evaluation_sha256") != sha256_file(
                model_evaluation_path
            ):
                artifact_problems.append(
                    "prerequisite model-evaluation artifact hash is stale"
                )
            else:
                model_value = json.loads(
                    model_evaluation_path.read_text(encoding="utf-8")
                )
                if not isinstance(model_value, Mapping):
                    artifact_problems.append(
                        "prerequisite model-evaluation root is not a mapping"
                    )
                else:
                    model_checks = _model_checks(
                        prior_gate_config,
                        model_value,
                        dataset_episodes_sha256=episodes_hash or "",
                        dataset_qc_report_sha256=qc_hash,
                    )
                    if any(not check.passed for check in model_checks):
                        artifact_problems.append(
                            "prerequisite model evaluation no longer passes recomputation"
                        )

    # Reports used as prerequisites must contain the release-critical checks
    # produced by the canonical evaluator, not an arbitrary list of green rows.
    required_check_names = {
        "minimum_unique_release_hours",
        "zero_global_or_release_qc_failures",
        "acceptance_suite.core_gates_recomputed",
        "acceptance_suite.full_native",
        "acceptance_suite.passed_execution",
    }
    missing_checks = sorted(required_check_names - set(check_names))
    if missing_checks:
        artifact_problems.append(
            f"prerequisite report lacks required checks: {missing_checks}"
        )
    bound = not artifact_problems
    checks = [
        ReadinessCheck(
            "prerequisite_gate.schema_and_identity",
            prior.get("schema_version") == "dynamic-robot-readiness-report/v1"
            and prior.get("gate_id") == required_gate_id,
            {
                "schema_version": prior.get("schema_version"),
                "gate_id": prior.get("gate_id"),
            },
            {
                "schema_version": "dynamic-robot-readiness-report/v1",
                "gate_id": required_gate_id,
            },
        ),
        ReadinessCheck(
            "prerequisite_gate.canonical_report_path",
            canonical_path == report_path,
            str(report_path),
            None if canonical_path is None else str(canonical_path),
        ),
        ReadinessCheck(
            "prerequisite_gate.internal_consistency",
            internal_consistency,
            {
                "reported_passed": prior.get("passed"),
                "failed_checks": failed_check_names,
                "blockers": blockers,
            },
            "passed iff every unique typed check passes and blockers equal failures",
        ),
        ReadinessCheck(
            "prerequisite_gate.artifacts_content_bound",
            bound,
            {
                "problems": artifact_problems,
                "dataset_episodes_sha256": episodes_hash,
                "qc_report_sha256": qc_hash,
                "gate_config_sha256": gate_hash,
                "acceptance_check_failures": [
                    check.name for check in acceptance_checks if not check.passed
                ],
            },
            "current dataset, QC, gate config, and recomputed acceptance evidence",
        ),
        ReadinessCheck(
            "prerequisite_gate_passed",
            prior.get("passed") is True and internal_consistency and bound,
            {"gate_id": prior.get("gate_id"), "passed": prior.get("passed")},
            {"gate_id": required_gate_id, "passed": True},
        ),
    ]
    return checks, {
        "prerequisite_report_path": str(report_path),
        "prerequisite_report_sha256": sha256_file(report_path),
        "prerequisite_dataset_episodes_sha256": episodes_hash,
        "prerequisite_qc_report_sha256": qc_hash,
        "prerequisite_gate_config_sha256": gate_hash,
    }


def evaluate_readiness(
    dataset_root: str | Path,
    gate_config: str | Path | Mapping[str, Any],
    *,
    wan_root: str | Path | None = None,
    model_evaluation_path: str | Path | None = None,
    prerequisite_report_path: str | Path | None = None,
    acceptance_report_path: str | Path | None = None,
) -> ReadinessReport:
    """Evaluate a stage gate without modifying the dataset."""

    root = Path(dataset_root).resolve(strict=True)
    config = (
        load_gate_config(gate_config)
        if isinstance(gate_config, (str, Path))
        else dict(gate_config)
    )
    records = load_episode_records(root)
    episodes_path = root / "meta" / "episodes.parquet"
    episodes_hash = sha256_file(episodes_path)
    qc_pass, qc_failures, qc_hash = _hard_qc_map(root, records)
    eligible = [
        record
        for record in records
        if record.release_eligible and qc_pass.get(record.episode_uuid, False)
    ]
    stats = collect_dataset_statistics(root, wan_root=wan_root, probe_streams=False)
    checks: list[ReadinessCheck] = []
    minimum_hours = float(config.get("minimum_unique_release_hours", 0.0))
    measured_hours = sum(float(record.duration_s or 0.0) for record in eligible) / 3600.0
    checks.append(
        ReadinessCheck(
            "minimum_unique_release_hours",
            measured_hours + 1e-12 >= minimum_hours,
            measured_hours,
            minimum_hours,
        )
    )
    checks.append(ReadinessCheck("zero_global_or_release_qc_failures", not qc_failures, qc_failures, []))

    split_diagnostics_path = root / "meta" / "split_diagnostics.json"
    if bool(config.get("require_no_sparse_strata", False)):
        if split_diagnostics_path.is_file():
            split_diagnostics = json.loads(split_diagnostics_path.read_text(encoding="utf-8"))
            sparse = list(split_diagnostics.get("sparse_strata") or [])
            splits_path = root / "meta" / "splits.parquet"
            split_rows = read_parquet_rows(splits_path) if splits_path.is_file() else []
            split_uuids = [str(row.get("episode_uuid") or "") for row in split_rows]
            record_uuids = sorted(record.episode_uuid for record in records)
            split_binding_passed = (
                split_diagnostics.get("schema_version")
                == "dynamic-robot-split-diagnostics/v2"
                and split_diagnostics.get("dataset_episodes_sha256") == episodes_hash
                and splits_path.is_file()
                and split_diagnostics.get("splits_parquet_sha256")
                == sha256_file(splits_path)
                and split_diagnostics.get("episode_uuid_set_sha256")
                == sha256_json(record_uuids)
                and int(split_diagnostics.get("assignment_count", -1)) == len(records)
                and len(split_uuids) == len(set(split_uuids)) == len(records)
                and sorted(split_uuids) == record_uuids
            )
            checks.append(
                ReadinessCheck(
                    "split.diagnostics_content_bound",
                    split_binding_passed,
                    {
                        "schema_version": split_diagnostics.get("schema_version"),
                        "dataset_episodes_sha256": split_diagnostics.get(
                            "dataset_episodes_sha256"
                        ),
                        "expected_dataset_episodes_sha256": episodes_hash,
                        "splits_parquet_sha256": split_diagnostics.get(
                            "splits_parquet_sha256"
                        ),
                        "actual_splits_parquet_sha256": (
                            sha256_file(splits_path) if splits_path.is_file() else None
                        ),
                        "assignment_count": split_diagnostics.get("assignment_count"),
                        "unique_split_uuid_count": len(set(split_uuids)),
                    },
                    "diagnostics bound to current episodes.parquet and splits.parquet",
                )
            )
            checks.append(
                ReadinessCheck(
                    "split.no_sparse_strata",
                    not sparse,
                    sparse,
                    [],
                    "Sparse strata must block readiness rather than split connected families",
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    "split.diagnostics_present",
                    False,
                    False,
                    True,
                    "meta/split_diagnostics.json is required",
                )
            )

    if bool(config.get("require_native_mujoco", False)):
        non_native = sorted(record.episode_uuid for record in eligible if not _native_record(record))
        checks.append(ReadinessCheck("native_mujoco_only", not non_native, non_native, []))
    if bool(config.get("free_contact_only", False)):
        wrong_tier = sorted(
            record.episode_uuid
            for record in eligible
            if str(record.release_tier.value if hasattr(record.release_tier, "value") else record.release_tier)
            != "free_contact"
        )
        checks.append(ReadinessCheck("free_contact_only", not wrong_tier, wrong_tier, []))

    for allocation in config.get("allocations", ()):
        selector = dict(allocation.get("selector") or {})
        selected = [record for record in eligible if _selector_matches(record, selector)]
        hours = sum(float(record.duration_s or 0.0) for record in selected) / 3600.0
        required = float(allocation["minimum_unique_hours"])
        checks.append(
            ReadinessCheck(
                f"allocation.{allocation['id']}",
                hours + 1e-12 >= required,
                {"hours": hours, "episodes": len(selected)},
                {"minimum_unique_hours": required},
                "Allocation is never silently reassigned to another family",
            )
        )

    coverage = dict(config.get("coverage") or {})
    styles = {
        str(record.randomization.get("background_style") or record.randomization.get("scene_style"))
        for record in eligible
    }
    checks.append(
        ReadinessCheck(
            "coverage.background_styles",
            len(styles - {"None", ""}) >= int(coverage.get("minimum_background_styles", 0)),
            sorted(styles - {"None", ""}),
            int(coverage.get("minimum_background_styles", 0)),
        )
    )
    minimum_outcomes = int(coverage.get("minimum_actual_outcome_classes_per_rigid_family", 0))
    for family in coverage.get("rigid_families", ()):
        outcomes = {
            record.actual_outcome_class.value
            for record in eligible
            if record.family == family
        }
        checks.append(
            ReadinessCheck(
                f"coverage.{family}.actual_outcomes",
                len(outcomes) >= minimum_outcomes,
                sorted(outcomes),
                minimum_outcomes,
            )
        )
    required_splits = set(coverage.get("required_splits", ()))
    for family in coverage.get("families_in_every_split", ()):
        splits = {
            record.split.value if hasattr(record.split, "value") else str(record.split)
            for record in eligible
            if record.family == family
        }
        checks.append(
            ReadinessCheck(
                f"coverage.{family}.splits",
                required_splits.issubset(splits),
                sorted(splits),
                sorted(required_splits),
            )
        )

    model_path = None if model_evaluation_path is None else Path(model_evaluation_path).resolve(strict=True)
    model_evaluation: Mapping[str, Any] | None = None
    if model_path is not None:
        raw_model_evaluation = json.loads(model_path.read_text(encoding="utf-8"))
        if not isinstance(raw_model_evaluation, Mapping):
            raise ValueError("Model-evaluation artifact root must be a mapping")
        model_evaluation = raw_model_evaluation
    checks.extend(
        _model_checks(
            config,
            model_evaluation,
            dataset_episodes_sha256=episodes_hash,
            dataset_qc_report_sha256=qc_hash,
        )
    )
    acceptance_path = (
        None
        if acceptance_report_path is None
        else Path(acceptance_report_path).resolve(strict=True)
    )
    acceptance_checks, acceptance_hashes = _acceptance_checks(
        config, acceptance_path
    )
    checks.extend(acceptance_checks)
    prerequisite = config.get("prerequisite_gate")
    prerequisite_path = (
        None
        if prerequisite_report_path is None
        else Path(prerequisite_report_path).resolve(strict=True)
    )
    prerequisite_hashes: dict[str, Any] = {}
    if isinstance(prerequisite, Mapping):
        prerequisite_checks, prerequisite_hashes = _prerequisite_checks(
            prerequisite, prerequisite_path
        )
        checks.extend(prerequisite_checks)
    if config.get("hard_blocked") is True:
        checks.append(
            ReadinessCheck(
                "stage_explicitly_blocked",
                False,
                config.get("blocked_reason"),
                "positive scaling curve and explicit gate revision",
            )
        )
    blockers = [check.name for check in checks if not check.passed]
    warnings = list(stats.warnings)
    return ReadinessReport(
        schema_version="dynamic-robot-readiness-report/v1",
        gate_id=str(config["gate_id"]),
        dataset_root=str(root),
        passed=not blockers,
        checks=checks,
        denominators={
            "all_logical_episodes": len(records),
            "schema_release_eligible_episodes": sum(record.release_eligible for record in records),
            "hard_qc_passed_release_episodes": len(eligible),
            "hard_qc_passed_unique_source_hours": measured_hours,
        },
        provenance={
            "dataset_episodes_sha256": episodes_hash,
            "qc_report_sha256": qc_hash,
            "metadata_complete_manifest_sha256": sha256_file(
                root / "meta" / ".complete.json"
            ),
            "gate_config_sha256": (
                sha256_file(Path(gate_config).resolve(strict=True))
                if isinstance(gate_config, (str, Path))
                else sha256_json(config)
            ),
            "resolved_gate_config_sha256": sha256_json(config),
            "model_evaluation_sha256": None if model_path is None else sha256_file(model_path),
            "model_evaluation_path": None if model_path is None else str(model_path),
            "model_artifact_path": (
                None
                if model_evaluation is None
                else (model_evaluation.get("provenance") or {}).get(
                    "model_artifact_path"
                )
            ),
            "model_artifact_sha256": (
                None
                if model_evaluation is None
                else (model_evaluation.get("provenance") or {}).get(
                    "model_artifact_sha256"
                )
            ),
            "evaluation_manifest_path": (
                None
                if model_evaluation is None
                else (model_evaluation.get("provenance") or {}).get(
                    "evaluation_manifest_path"
                )
            ),
            "evaluation_manifest_sha256": (
                None
                if model_evaluation is None
                else (model_evaluation.get("provenance") or {}).get(
                    "evaluation_manifest_sha256"
                )
            ),
            **prerequisite_hashes,
            **acceptance_hashes,
            "split_diagnostics_sha256": (
                sha256_file(split_diagnostics_path)
                if split_diagnostics_path.is_file()
                else None
            ),
            "splits_parquet_sha256": (
                sha256_file(root / "meta" / "splits.parquet")
                if (root / "meta" / "splits.parquet").is_file()
                else None
            ),
            "canonical_report_path": str(
                canonical_readiness_report_path(root, str(config["gate_id"]))
            ),
            "wan_root": None if wan_root is None else str(Path(wan_root).resolve()),
        },
        blockers=blockers,
        warnings=warnings,
    )
