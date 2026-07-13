"""Common command-line interface for inventory, generation, QC, and export."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .common.episode_writer import EpisodeWriter, load_episode_records, read_parquet_rows, write_parquet_atomic
from .common.cameras import CameraCalibration
from .common.episode_writer import canonical_camera_name
from .common.contract_v2 import (
    CounterfactualFamilyRecord,
    CounterfactualRelation,
    build_counterfactual_family_records,
    counterfactual_fixed_hashes,
    derived_action_hash_from_rows,
    derived_initial_state_hash_from_rows,
    validate_counterfactual_family_records,
)
from .common.hashing import sha256_file, sha256_json
from .common.labels import project_candidate_outcome
from .common.native_suite import (
    PlannedSuiteEpisode,
    build_planned_counterfactual_family_records,
    plan_suite_cases,
)
from .common.contacts import (
    assistance_interval_contains,
    normalize_assistance,
    select_task_event_time,
)
from .common.paths import (
    ExistingOutputError,
    ResumeMismatchError,
    atomic_write_json,
    ensure_not_source_path,
)
from .common.physics_sweeps import sweep_acceptance_evidence
from .common.provenance import get_git_commit, inventory_source
from .common.qc import validate_dataset
from .common.calibration import calibrate_physics_catalog
from .common.readiness import evaluate_readiness
from .common.statistics import collect_dataset_statistics
from .common.suites import expand_suite
from .common.schema import (
    DatasetInfo,
    DynamicsMode,
    EpisodeRecord,
    LabelStatus,
    NamedFeature,
    PhysicsMetadata,
    PhysicsValue,
    PhysicsValueKind,
    ReleaseTier,
    TimeBase,
)
from .common.splits import SplitAssigner
from .common.wan_export import export_wan

# Backward-compatible private alias retained for focused suite-contract tests;
# readiness imports the shared implementation directly.
_sweep_acceptance_evidence = sweep_acceptance_evidence

DEFAULT_SOURCE_ROOTS = (
    "/gpfs/radev/scratch/sous/mzl7",
    "/gpfs/radev/scratch/sous/zl664",
    "/gpfs/radev/scratch/sous/zss8",
)


def _comma_list(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value")
    return result


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _load_renderer(specification: str | None) -> Callable[..., Mapping[str, Any]]:
    value = specification or os.environ.get("DYNAMIC_ROBOT_RENDERER")
    if not value:
        raise ValueError(
            "Non-dry generation requires --renderer module:function or "
            "DYNAMIC_ROBOT_RENDERER; the diagnostic smoke renderer is never a default"
        )
    module_name, separator, attribute = value.partition(":")
    if not separator:
        raise ValueError("Renderer must use module:function syntax")
    renderer = getattr(importlib.import_module(module_name), attribute)
    if not callable(renderer):
        raise TypeError(f"Renderer is not callable: {value}")
    return renderer


def _physics_metadata(raw_fields: Mapping[str, Any]) -> PhysicsMetadata:
    kinds = {
        "physical": PhysicsValueKind.PHYSICAL,
        "calibrated_effective": PhysicsValueKind.CALIBRATED_EFFECTIVE,
        "proxy": PhysicsValueKind.SIMULATOR_PROXY,
        "simulator_proxy": PhysicsValueKind.SIMULATOR_PROXY,
        "unknown": PhysicsValueKind.UNKNOWN,
        "not_implemented": PhysicsValueKind.NOT_IMPLEMENTED,
    }
    parameters: dict[str, PhysicsValue] = {}
    gravity = (0.0, 0.0, -9.81)
    gravity_valid = False
    timestep: float | None = None
    substeps: int | None = None
    for name, raw in raw_fields.items():
        field = dict(raw) if isinstance(raw, Mapping) else {
            "value": raw,
            "unit": "unknown",
            "valid": raw is not None,
            "implemented": True,
            "interpretation": "physical",
        }
        value = field.get("value")
        interpretation = str(field.get("interpretation", "physical"))
        implemented = bool(field.get("implemented", True))
        kind = kinds.get(interpretation, PhysicsValueKind.UNKNOWN)
        if not implemented and interpretation == "not_implemented":
            kind = PhysicsValueKind.NOT_IMPLEMENTED
        valid = bool(field.get("valid", value is not None))
        if kind == PhysicsValueKind.UNKNOWN and valid:
            # A known numeric simulator setting with unknown physical semantics
            # is a proxy, not an invalid "unknown" physical measurement.
            kind = PhysicsValueKind.SIMULATOR_PROXY
        parameters[name] = PhysicsValue(
            name=name,
            value=value,
            unit=str(field.get("unit", "unknown")),
            valid=valid,
            implemented=implemented,
            kind=kind,
            source="family_adapter_config",
        )
        if name == "gravity" and valid and isinstance(value, Sequence):
            gravity = tuple(float(component) for component in value)  # type: ignore[assignment]
            gravity_valid = implemented
        elif name == "simulation_timestep" and valid:
            timestep = float(value)
        elif name == "substeps" and valid:
            substeps = int(value)
    return PhysicsMetadata(
        parameters=parameters,
        gravity_world_m_s2=gravity,
        gravity_valid=gravity_valid,
        simulation_timestep_s=timestep,
        substeps=substeps,
    )


def _physics_range_provenance(rendered: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize backend range provenance without treating a label as calibration."""

    backend = dict(rendered.get("backend_provenance") or {})
    raw = dict(backend.get("physics_range_provenance") or {})
    declared_artifact_hash = raw.get("calibration_artifact_hash") or raw.get(
        "calibration_artifact"
    )
    artifact_path_raw = raw.get("calibration_artifact_path")
    artifact_verified = False
    artifact_hash = None
    if artifact_path_raw:
        try:
            artifact_path = Path(str(artifact_path_raw)).resolve(strict=True)
            artifact_hash = sha256_file(artifact_path)
            report = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact_verified = (
                artifact_hash == declared_artifact_hash
                and report.get("schema_version")
                == "dynamic-robot-calibration-report/v1"
                and report.get("passed") is True
                and report.get("release_eligible") is True
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            artifact_verified = False
    requested_calibrated = raw.get("calibrated") is True
    calibrated = requested_calibrated and artifact_verified
    version = str(raw.get("profile_version") or raw.get("range_version") or "unversioned")
    return {
        "profile_id": str(raw.get("profile_id") or "native_rigid_ranges"),
        "profile_version": version,
        "partition": str(raw.get("partition") or "uncalibrated"),
        "source": str(raw.get("source") or "configs/physics/rigid_ranges_v1.yaml"),
        "calibrated": calibrated,
        "calibration_verified": artifact_verified,
        "calibration_artifact_hash": artifact_hash if calibrated else None,
        "requested_calibrated": requested_calibrated,
    }


def _label_status(value: str) -> LabelStatus:
    if value in {"verified", "verified_objective"}:
        return LabelStatus.VERIFIED
    if value == "proxy":
        return LabelStatus.PROXY
    return LabelStatus.UNVERIFIED


def _component_quality_flags(
    result: Any,
    rendered: Mapping[str, Any],
) -> list[str]:
    """Return component-specific flags without conflating them with release gates.

    Native backend ``production_eligible`` is intentionally an aggregate release
    verdict: calibration, visual-style admission, tool calibration, physics QC,
    and rendering all contribute to it.  Treating that aggregate as a verdict on
    the simulator adapter or renderer adds misleading flags whenever any
    independent release gate is still closed.  Native payloads instead prove the
    two components directly by backend identity and by a non-empty integrated
    MuJoCo render payload.  Diagnostic adapters retain the conservative legacy
    fallback because they do not provide component-level provenance.
    """

    flags = [str(value) for value in rendered.get("quality_flags", ())]
    native_backend = bool(result.simulator.get("native_mujoco", False))
    if (
        not native_backend
        and result.simulator.get("production_eligible") is not True
    ):
        flags.append("non_production_family_adapter")

    backend_provenance = dict(rendered.get("backend_provenance") or {})
    renderer_name = str(
        rendered.get("renderer") or backend_provenance.get("renderer") or ""
    )
    videos = rendered.get("videos")
    native_renderer_verified = bool(
        native_backend
        and renderer_name == "mujoco.Renderer"
        and isinstance(videos, Mapping)
        and videos
    )
    renderer_component_verified = (
        native_renderer_verified
        if native_backend
        else rendered.get("production_eligible") is True
    )
    if not renderer_component_verified:
        flags.append("renderer_not_production_verified")
    return sorted(set(flags))


def _episode_record(
    result: Any,
    episode_index: int,
    git_commit: str,
    rendered: Mapping[str, Any],
) -> EpisodeRecord:
    persisted_frame_rows = list(
        rendered.get("frame_rows") or _default_frame_rows(result, episode_index)
    )
    persisted_action_rows = list(
        rendered.get("high_rate_rows") or result.high_rate_states or persisted_frame_rows
    )
    if not any(
        any(name.startswith("action.") for name in row)
        for row in persisted_action_rows
    ):
        # Older diagnostic adapters persist state-only controller-rate tables.
        # Their frame table still contains the timestamped commanded action;
        # hashing the state-only table would collapse every action sibling to
        # the same timestamp-only digest.
        persisted_action_rows = persisted_frame_rows
    derived_action_hash = derived_action_hash_from_rows(persisted_action_rows)
    derived_initial_state_hash = derived_initial_state_hash_from_rows(
        persisted_frame_rows
    )
    status = _label_status(str(result.outcome.label_status))
    outcome_projection = project_candidate_outcome(
        actual_outcome=result.actual_outcome,
        task_success=result.outcome.task_success,
        partial_success_score=result.outcome.partial_success_score,
        failure_mode=result.outcome.failure_mode,
        label_confidence=result.outcome.label_confidence,
        source_label_status=str(result.outcome.label_status),
        canonical_label_status=status,
    )
    dynamics = DynamicsMode(str(result.dynamics_mode))
    release = ReleaseTier(str(result.release_tier))
    if dynamics == DynamicsMode.SCRIPTED_MOTION:
        release = ReleaseTier.SCRIPTED_MOTION
    elif dynamics == DynamicsMode.ASSISTED_CONTACT:
        release = ReleaseTier.ASSISTED_CONTACT
    elif status != LabelStatus.VERIFIED and release == ReleaseTier.FREE_CONTACT:
        release = ReleaseTier.UNVERIFIED
    event_time = (
        float(rendered["key_event_time_s"])
        if rendered.get("key_event_time_s") is not None
        else float(rendered["task_event_time_s"])
        if rendered.get("task_event_time_s") is not None
        else select_task_event_time(result.contacts)
    )
    native_backend = bool(result.simulator.get("native_mujoco", False))
    backend_provenance = dict(rendered.get("backend_provenance") or {})
    quality_flags = _component_quality_flags(result, rendered)
    physics = _physics_metadata(result.plan.physics)
    if native_backend:
        physics.parameter_range_provenance = _physics_range_provenance(rendered)
        physics.solver_settings = {
            "semantics": "MuJoCo solver settings; not literal physical restitution",
            "model_hash": result.simulator.get("model_hash"),
            "contact_resolution": result.simulator.get("contact_resolution"),
        }
    event_time = event_time
    objective_evidence = dict(rendered.get("objective_evidence") or {})
    native_spec_dict: dict[str, Any] = {}
    if native_backend:
        raw_spec = result.plan.options.get("native_scenario_spec")
        if not isinstance(raw_spec, Mapping):
            raise ValueError("Native result is missing its immutable ScenarioSpec")
        native_spec_dict = dict(raw_spec)
        from .backends.base import ScenarioSpec
        from .backends.mujoco_native.evaluators import evaluate_saved_native_episode

        independent = evaluate_saved_native_episode(
            ScenarioSpec.from_dict(raw_spec),
            list(rendered.get("frame_rows") or ()),
            list(rendered.get("event_rows") or result.contacts),
            list(rendered.get("transition_rows") or ()),
        )
        if (
            independent.outcome.task_success != result.outcome.task_success
            or independent.outcome.failure_mode != result.outcome.failure_mode
            or independent.actual_outcome_class != result.actual_outcome
        ):
            raise ValueError("Native online label disagrees with persisted-artifact evaluator")
        objective_evidence = {
            **dict(independent.evidence),
            "stored_objective_success": result.outcome.task_success,
            "independently_recomputed": True,
            "source": "registered_persisted_state_event_evaluator",
            "evidence_hash": sha256_json(independent.evidence),
        }
    if not objective_evidence and native_backend:
        objective_evidence = {
            "stored_objective_success": result.outcome.task_success,
            "independently_recomputed": False,
            "source": "native_rollout_not_yet_recomputed_from_persisted_tables",
        }
    if outcome_projection.diagnostic_candidate_outcome is not None:
        objective_evidence = {
            **objective_evidence,
            "stored_objective_success": None,
            "independently_recomputed": False,
            "source": "diagnostic_candidate_outcome_unverified",
            "diagnostic_candidate_outcome": outcome_projection.diagnostic_candidate_outcome,
        }
    threshold_set = dict(rendered.get("objective_thresholds") or {})
    first_state = dict(result.states[0]) if result.states else {}
    initial_state_payload = {
        key: value
        for key, value in first_state.items()
        if key != "timestamp"
        and not key.startswith("task.")
        and not key.startswith("assistance.")
        and not key.endswith("_flag")
    }
    key_event_name = rendered.get("key_event_name")
    if key_event_name is None and event_time is not None:
        key_event_name = "first_task_contact"
    return EpisodeRecord(
        episode_uuid=result.plan.episode_uuid,
        episode_index=episode_index,
        counterfactual_bundle_id=result.plan.counterfactual_bundle_id,
        physics_counterfactual_family_id=result.plan.physics_counterfactual_family_id,
        split_group_id=result.plan.split_group_id,
        scene_seed=result.plan.scene_seed,
        branch_seed=result.plan.branch_seed,
        family=result.plan.family,
        subfamily=result.plan.subfamily,
        variant=result.plan.variant,
        robot_model=result.plan.robot_model,
        tool_type=result.plan.tool_type,
        action_mode=str(result.plan.options.get("action_mode", "family_specific_named_command")),
        intended_branch=result.plan.intended_branch,
        actual_outcome=outcome_projection.actual_outcome,
        actual_outcome_class=outcome_projection.actual_outcome_class,
        task_success=outcome_projection.task_success,
        partial_success_score=outcome_projection.partial_success_score,
        failure_mode=outcome_projection.failure_mode,
        label_confidence=outcome_projection.label_confidence,
        label_status=status,
        dynamics_mode=dynamics,
        release_tier=release,
        physics_qc_pass=bool(result.physics_qc.get("physics_qc_pass", False)),
        source_generator=result.plan.source_generator,
        source_generator_version=result.plan.source_generator_version,
        generator_git_commit=git_commit,
        config_hash=result.plan.config_hash,
        simulator_name=str(result.simulator.get("name", "unknown")),
        simulator_version=str(result.simulator.get("version", "unknown")),
        renderer=str(rendered.get("renderer") or "external_native_renderer"),
        asset_ids=[str(value) for value in rendered.get("asset_ids", ())],
        asset_hashes={str(key): str(value) for key, value in dict(rendered.get("asset_hashes") or {}).items()},
        event_time_s=event_time,
        key_event_name=None if key_event_name is None else str(key_event_name),
        key_event_time_s=event_time,
        objective_evaluator_id=str(
            rendered.get("objective_evaluator_id")
            or ("native_rigid_state_event" if native_backend else "legacy_embedded")
        ),
        objective_evaluator_version=str(
            rendered.get("objective_evaluator_version")
            or ("1.2.0" if native_backend else "unversioned")
        ),
        objective_threshold_set_hash=str(
            rendered.get("objective_threshold_set_hash") or sha256_json(threshold_set)
        ),
        objective_evidence=objective_evidence,
        physics=physics,
        assistance=normalize_assistance(result.assistance),
        objective_metrics=dict(result.outcome.metrics),
        controller_profile={
            "profile_id": str(
                backend_provenance.get("controller_profile_id")
                or ("franka_cartesian_dls_joint_position" if native_backend else "legacy_unspecified")
            ),
            "profile_version": str(
                backend_provenance.get("controller_profile_version")
                or ("1.1.0" if native_backend else "unversioned")
            ),
            "control_latency_s": (
                backend_provenance.get("control_latency_s") if native_backend else None
            ),
            "camera_latency_s": (
                backend_provenance.get("camera_latency_s") if native_backend else None
            ),
        },
        robot_start_provenance=(
            {
                "source": "native_mujoco_initial_state",
                "initial_state_hash": derived_initial_state_hash,
                "state_write_phase": "initialization_only",
                "joint_offsets_rad": backend_provenance.get(
                    "robot_start_joint_offsets_rad"
                ),
            }
            if native_backend
            else {}
        ),
        tool_calibration_provenance=(
            {
                "calibration_id": backend_provenance.get("tool_calibration_id"),
                "calibrated": bool(backend_provenance.get("tool_calibrated", False)),
                "source": "native_backend_scenario_spec",
                "artifact_path": backend_provenance.get(
                    "tool_calibration_artifact_path"
                ),
                "artifact_sha256": backend_provenance.get(
                    "tool_calibration_artifact_sha256"
                ),
                "actual_artifact_sha256": backend_provenance.get(
                    "tool_calibration_actual_sha256"
                ),
            }
            if native_backend
            else {}
        ),
        randomization={
            **(
                dict(result.plan.scene_parameters["randomization"])
                if isinstance(result.plan.scene_parameters.get("randomization"), Mapping)
                else {}
            ),
            "scene_style": result.plan.scene_style,
            "background_style": result.plan.scene_parameters.get(
                "background_style", result.plan.scene_style
            ),
            "randomization_level": result.plan.randomization_level,
            "visual_seed": result.plan.scene_parameters.get("visual_seed", result.plan.scene_seed),
            **(
                {
                    "object_asset_id": f"procedural_{dict(native_spec_dict.get('object') or {}).get('shape', 'unknown')}",
                    "object_rgba": list(
                        dict(native_spec_dict.get("object") or {}).get("rgba", ())
                    ),
                    "object_material_id": "native_object_mat",
                    "lighting_style": result.plan.scene_style,
                    "camera_preset_id": sha256_json(
                        native_spec_dict.get("cameras", ())
                    )[:16],
                    "tool_geometry_id": sha256_json(
                        native_spec_dict.get("tool", {})
                    )[:16],
                    "tool_type": result.plan.tool_type,
                }
                if native_spec_dict
                else {}
            ),
        },
        quality_flags=sorted(set(quality_flags)),
        extras={
            "family_plan_config_hash": result.plan.config_hash,
            "action_hash": result.plan.action_hash,
            "physics_hash": result.plan.physics_hash,
            "counterfactual_invariant_hash": result.plan.invariant_hash,
            "physics_variant": result.plan.physics_variant,
            "scene_parameters": dict(result.plan.scene_parameters),
            "branch_parameters": dict(result.plan.branch_parameters),
            "physics_qc": dict(result.physics_qc),
            "planned_physics": dict(result.plan.physics),
            "planned_counterfactual_fixed_hashes": {
                str(relation): dict(values)
                for relation, values in dict(
                    result.plan.options.get("planned_counterfactual_fixed_hashes")
                    or {}
                ).items()
                if isinstance(values, Mapping)
            },
            "initial_state_hash": sha256_json(initial_state_payload),
            "derived_initial_state_hash": derived_initial_state_hash,
            "derived_action_hash": derived_action_hash,
            "backend_provenance": backend_provenance,
            **(
                {"native_scenario_spec": dict(result.plan.options["native_scenario_spec"])}
                if isinstance(result.plan.options.get("native_scenario_spec"), Mapping)
                else {}
            ),
            "notes": list(result.notes),
            "event_time_semantics": str(
                rendered.get("key_event_semantics")
                or "first_non_fixture_task_contact"
            ),
            **(
                {"visibility_qc": dict(rendered["visibility_qc"])}
                if isinstance(rendered.get("visibility_qc"), Mapping)
                else {}
            ),
        },
    )


def _default_frame_rows(result: Any, episode_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contact_times = [float(item["timestamp"]) for item in result.contacts]
    half_frame = 0.5 / float(result.plan.rates_hz["video"])
    for frame_index, (timestamp, state) in enumerate(zip(result.frame_times_s, result.states)):
        action = result.actions[min(frame_index, len(result.actions) - 1)] if result.actions else {}
        assistance_active = bool(state.get("assistance.active", False))
        mechanism_ids = [
            str(mechanism["mechanism_id"])
            for mechanism in result.assistance.get("mechanisms", ())
            if any(
                assistance_interval_contains(interval, float(timestamp))
                for interval in mechanism.get("activation_intervals", ())
            )
        ]
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_index,
            "video_frame_index": frame_index,
            "timestamp": float(timestamp),
            "contact.active": any(abs(float(timestamp) - event) <= half_frame for event in contact_times),
            "event.contact": any(abs(float(timestamp) - event) <= half_frame for event in contact_times),
            "assistance.active": assistance_active,
            "assistance.assisted_grasp": assistance_active and bool(result.assistance.get("assisted_grasp", False)),
            "assistance.assisted_retention": assistance_active and bool(result.assistance.get("assisted_retention", False)),
            "assistance.equality_constraint_active": assistance_active and bool(result.assistance.get("equality_constraint_active", False)),
            "assistance.latch_active": assistance_active and bool(result.assistance.get("latch_active", False)),
            "assistance.mechanism_ids": mechanism_ids,
        }
        row.update({key: value for key, value in state.items() if key != "timestamp"})
        row.update({f"action.{key}": value for key, value in action.items() if key != "timestamp"})
        rows.append(row)
    return rows


def _register_camera_calibrations(
    rendered: Mapping[str, Any],
    camera_rows: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Register task-specific calibrations and return stream→calibration IDs.

    A stream role is stable (for example ``observation.images.main``), while
    its pose may legitimately vary by task. Therefore calibration IDs include
    a content digest instead of incorrectly using the stream name as identity.
    """

    raw_cameras = rendered.get("cameras") or rendered.get("camera_calibrations")
    if not raw_cameras:
        raise ValueError("Renderer result must contain camera calibrations")
    camera_values = raw_cameras.items() if isinstance(raw_cameras, Mapping) else (
        (getattr(value, "camera_name", f"camera-{position}"), value)
        for position, value in enumerate(raw_cameras)
    )
    mapping: dict[str, str] = {}
    for camera_name, calibration in camera_values:
        row = calibration.to_dict() if hasattr(calibration, "to_dict") else dict(calibration)
        row.pop("camera_id", None)
        stream = canonical_camera_name(str(row.get("camera_name") or camera_name))
        row["camera_name"] = stream
        CameraCalibration.from_dict(row)
        identifier = f"{stream}@{sha256_json(row)[:16]}"
        table_row = {"camera_id": identifier, **row}
        existing = camera_rows.get(identifier)
        if existing is not None and existing != table_row:
            raise ValueError(f"Camera calibration ID collision: {identifier}")
        camera_rows[identifier] = table_row
        if stream in mapping and mapping[stream] != identifier:
            raise ValueError(f"Multiple calibrations provided for stream {stream}")
        mapping[stream] = identifier
    return mapping


def _infer_physics_interventions(records: Sequence[EpisodeRecord]) -> dict[str, tuple[str, ...]]:
    """Name exactly the persisted physics fields that vary within each family."""

    groups: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        if record.physics_counterfactual_family_id is not None:
            groups.setdefault(record.physics_counterfactual_family_id, []).append(record)
    result: dict[str, tuple[str, ...]] = {}
    for family_id, siblings in groups.items():
        if len(siblings) < 2:
            continue
        names = sorted({name for sibling in siblings for name in sibling.physics.parameters})
        changed = tuple(
            name
            for name in names
            if len(
                {
                    sha256_json(asdict(sibling.physics.parameters[name]))
                    if name in sibling.physics.parameters
                    else "missing"
                    for sibling in siblings
                }
            )
            > 1
        )
        if changed:
            result[family_id] = changed
    return result


def _planned_episode_declarations(plans: Sequence[Any]) -> list[CounterfactualFamilyRecord]:
    """Declare exact expected membership before an ordinary generation run."""

    action_groups: dict[str, list[Any]] = {}
    physics_groups: dict[str, list[Any]] = {}
    for plan in plans:
        action_groups.setdefault(str(plan.counterfactual_bundle_id), []).append(plan)
        if plan.physics_counterfactual_family_id:
            physics_groups.setdefault(
                str(plan.physics_counterfactual_family_id), []
            ).append(plan)
    declarations: list[CounterfactualFamilyRecord] = []
    for family_id, siblings in sorted(action_groups.items()):
        if len(siblings) < 2:
            continue
        split_ids = {str(value.split_group_id) for value in siblings}
        if len(split_ids) != 1 or len({value.physics_hash for value in siblings}) != 1:
            raise ValueError(f"planned action family {family_id} changes split group or physics")
        declaration = CounterfactualFamilyRecord(
            family_id=family_id,
            relation=CounterfactualRelation.ACTION,
            split_group_id=next(iter(split_ids)),
            expected_member_count=len(siblings),
            expected_episode_uuids=sorted(value.episode_uuid for value in siblings),
            intervention_fields=["action"],
            fixed_field_hashes={
                "scene_seed": sha256_json(siblings[0].scene_seed),
                "appearance": sha256_json(siblings[0].scene_parameters),
                "split_group_id": sha256_json(siblings[0].split_group_id),
                "initial_state": sha256_json(siblings[0].scene_parameters),
                "physics": sha256_json(siblings[0].physics),
            },
            expected_member_plan_hashes={
                value.episode_uuid: value.config_hash for value in siblings
            },
        )
        declaration.validate()
        declarations.append(declaration)
    for family_id, siblings in sorted(physics_groups.items()):
        if len(siblings) < 2:
            continue
        split_ids = {str(value.split_group_id) for value in siblings}
        if len(split_ids) != 1 or len({value.action_hash for value in siblings}) != 1:
            raise ValueError(f"planned physics family {family_id} changes split group or action")
        field_names = sorted({name for value in siblings for name in value.physics})
        interventions = [
            name
            for name in field_names
            if len(
                {
                    sha256_json(value.physics.get(name, "__missing__"))
                    for value in siblings
                }
            )
            > 1
        ]
        if not interventions:
            raise ValueError(f"planned physics family {family_id} has no named intervention")
        nonintervened = {
            sha256_json(
                {
                    name: value
                    for name, value in sibling.physics.items()
                    if name not in interventions
                }
            )
            for sibling in siblings
        }
        if len(nonintervened) != 1:
            raise ValueError(f"planned physics family {family_id} changes undeclared physics")
        declaration = CounterfactualFamilyRecord(
            family_id=family_id,
            relation=CounterfactualRelation.PHYSICS,
            split_group_id=next(iter(split_ids)),
            expected_member_count=len(siblings),
            expected_episode_uuids=sorted(value.episode_uuid for value in siblings),
            intervention_fields=interventions,
            fixed_field_hashes={
                "scene_seed": sha256_json(siblings[0].scene_seed),
                "appearance": sha256_json(siblings[0].scene_parameters),
                "split_group_id": sha256_json(siblings[0].split_group_id),
                "initial_state": sha256_json(siblings[0].scene_parameters),
                "action": sha256_json(siblings[0].action_hash),
                "nonintervened_physics": next(iter(nonintervened)),
            },
            expected_member_plan_hashes={
                value.episode_uuid: value.config_hash for value in siblings
            },
        )
        declaration.validate()
        declarations.append(declaration)
    return sorted(declarations, key=lambda value: (value.relation.value, value.family_id))


def _annotate_plans_with_counterfactual_hashes(
    plans: Sequence[Any],
    declarations: Sequence[CounterfactualFamilyRecord],
) -> list[Any]:
    """Persist the immutable declaration hashes into ordinary run records."""

    by_key = {
        (declaration.relation, declaration.family_id): declaration
        for declaration in declarations
    }
    annotated: list[Any] = []
    for plan in plans:
        hashes: dict[str, dict[str, str]] = {}
        action = by_key.get(
            (CounterfactualRelation.ACTION, str(plan.counterfactual_bundle_id))
        )
        if action is not None:
            hashes[CounterfactualRelation.ACTION.value] = dict(
                action.fixed_field_hashes
            )
        if plan.physics_counterfactual_family_id:
            physics = by_key.get(
                (
                    CounterfactualRelation.PHYSICS,
                    str(plan.physics_counterfactual_family_id),
                )
            )
            if physics is not None:
                hashes[CounterfactualRelation.PHYSICS.value] = dict(
                    physics.fixed_field_hashes
                )
        annotated.append(
            replace(
                plan,
                options={
                    **dict(plan.options),
                    "planned_counterfactual_fixed_hashes": hashes,
                },
            )
        )
    return annotated


def _command_inventory(arguments: argparse.Namespace) -> int:
    inventories = [inventory_source(root).to_dict() for root in arguments.roots]
    payload = {"source_roots": inventories}
    if arguments.output:
        atomic_write_json(arguments.output, payload)
    else:
        _json_print(payload)
    return 0


def _generation_config(arguments: argparse.Namespace) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "subfamily": "default",
        "variant": "clean_v1",
        "robot_model": "abstract",
        "tool_type": "task_default",
        "backend": "diagnostic",
        "num_bundles": 1,
        "branches": ["success", "near_miss", "contact_failure", "bad_action"],
        "views": ["main", "secondary"],
        "seed": 0,
        "randomization_level": "R1",
        "scene_style": "clean_franka_lab",
        "sim_hz": 240,
        "control_hz": 60,
        "video_hz": 30,
        "duration_s": None,
        "physics_sweep": None,
    }
    loaded: dict[str, Any] = {}
    if arguments.config:
        config_path = Path(arguments.config).resolve(strict=True)
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            value = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML generation configs require PyYAML") from exc
            value = yaml.safe_load(text)
        if not isinstance(value, Mapping):
            raise ValueError("Generation config must contain a mapping")
        loaded = dict(value.get("generation", value))
    config = {**defaults, **loaded}
    for key in defaults.keys() | {"family"}:
        override = getattr(arguments, key, None)
        if override is not None:
            config[key] = list(override) if key in {"branches", "views"} else override
    if not config.get("family"):
        raise ValueError("Generation requires --family or a family field in --config")
    return config


def _native_plans_from_adapter(
    adapter_plans: Sequence[Any], config: Mapping[str, Any]
) -> list[Any]:
    """Rebind adapter identities to typed native specs and real action hashes.

    The legacy analytical adapters remain useful deterministic request
    expanders. Their trajectories and action hashes are not valid native
    MuJoCo commands, however, so the native path must construct ScenarioSpec
    records before dry-run or execution instead of reconstructing them late in
    the backend.
    """

    from .backends import IntendedBranch
    from .backends.mujoco_native import (
        make_scenario_spec_for_subfamily,
        scenario_to_episode_plan,
    )

    views = tuple(str(value) for value in config.get("views", ("main", "secondary")))
    if {canonical_camera_name(value) for value in views} != {
        "observation.images.main",
        "observation.images.secondary",
    }:
        raise ValueError("native_mujoco generation requires exactly main and secondary views")
    branch_aliases = {
        "success_seeking": IntendedBranch.SUCCESS,
        "near_miss": IntendedBranch.NEAR_MISS,
        "contact_failure": IntendedBranch.CONTACT_FAILURE,
        "no_op": IntendedBranch.NO_OP,
        "bad_action": IntendedBranch.WRONG_ACTION,
        "wrong_action": IntendedBranch.WRONG_ACTION,
    }
    typed_plans: list[Any] = []
    for source in adapter_plans:
        try:
            branch = branch_aliases[source.intended_branch]
        except KeyError as error:
            raise ValueError(
                f"native_mujoco does not support branch {source.intended_branch!r}"
            ) from error
        randomization = (
            dict(source.scene_parameters.get("randomization") or {})
            if isinstance(source.scene_parameters, Mapping)
            else {}
        )
        options = {
            **dict(source.options),
            "randomization": randomization,
            "randomization_level": source.randomization_level,
            "physics_variant": source.physics_variant,
            "requested_variant": source.variant,
        }
        spec = make_scenario_spec_for_subfamily(
            source.family,
            source.subfamily,
            seed=source.scene_seed,
            branch=branch,
            scene_style=source.scene_style,
            physics_variant=source.physics_variant,
            split_group_id=source.split_group_id,
            counterfactual_bundle_id=source.counterfactual_bundle_id,
            physics_counterfactual_family_id=source.physics_counterfactual_family_id,
            options=options,
        )
        requested_duration = config.get("duration_s")
        spec = replace(
            spec,
            sim_hz=int(source.rates_hz["simulation"]),
            control_hz=int(source.rates_hz["control"]),
            video_hz=int(source.rates_hz["video"]),
            maximum_duration_s=(
                spec.maximum_duration_s
                if requested_duration is None
                else float(requested_duration)
            ),
            controller_latency_s=float(
                config.get("controller_delay_s", spec.controller_latency_s)
            ),
        )
        spec.validate()
        typed = scenario_to_episode_plan(spec)
        typed_plans.append(
            replace(
                typed,
                variant=source.variant,
                randomization_level=source.randomization_level,
                views=("main", "secondary"),
            )
        )
    return typed_plans


def _command_generate(arguments: argparse.Namespace) -> int:
    from .families import get_family

    config = _generation_config(arguments)
    output = arguments.output or config.pop("output", None)
    adapter = get_family(str(config["family"]))
    backend_name = str(config.get("backend") or "diagnostic").strip().lower().replace("-", "_")
    plans = adapter.plan(config)
    if backend_name == "native_mujoco":
        plans = _native_plans_from_adapter(plans, config)
    planned_declarations = _planned_episode_declarations(plans)
    plans = _annotate_plans_with_counterfactual_hashes(
        plans, planned_declarations
    )
    if arguments.dry_run:
        _json_print(
            {
                "dry_run": True,
                "backend": backend_name,
                "episode_count": len(plans),
                "plans": [plan.to_dict() for plan in plans],
                "counterfactual_families": [
                    value.to_dict() for value in planned_declarations
                ],
            }
        )
        return 0
    if not output:
        raise ValueError("Generation requires --output or an output field in --config")
    renderer_spec: str | None
    renderer: Callable[..., Mapping[str, Any]] | None = None
    native_backend: Any | None = None
    if backend_name == "diagnostic":
        renderer_spec = arguments.renderer or os.environ.get("DYNAMIC_ROBOT_RENDERER")
        renderer = _load_renderer(renderer_spec)
    elif backend_name == "native_mujoco":
        if arguments.renderer:
            raise ValueError("--renderer is diagnostic-only; native_mujoco renders its own rollout")
        from .backends import get_backend

        native_backend = get_backend("native_mujoco")
        renderer_spec = "integrated:mujoco.Renderer"
    else:
        raise ValueError("--backend must be diagnostic or native_mujoco")
    git_commit = get_git_commit(Path(__file__).parents[2])
    asset_roots = {
        name: os.environ[name]
        for name in ("ROBOCASA_ROOT", "ROBOTWIN_ROOT", "ROBOTWIN_2_ROOT", "MUJOCO_MENAGERIE_ROOT")
        if os.environ.get(name)
    }
    writer_config = {
        **config,
        "provenance": {
            "renderer_backend": renderer_spec,
            "simulation_backend": backend_name,
            "generator_git_commit": git_commit,
            "asset_roots": asset_roots,
        },
    }
    writer = EpisodeWriter(output, writer_config, resume=arguments.resume)
    _write_or_validate_immutable_json(
        Path(output) / ".generation_plan.json",
        {
            "schema_version": "dynamic-robot-generation-plan/v1",
            "config_hash": writer.config_hash,
            "backend": backend_name,
            "expected_episode_count": len(plans),
            "expected_episode_uuids": [plan.episode_uuid for plan in plans],
            "plans": [plan.to_dict() for plan in plans],
            "counterfactual_families": [
                declaration.to_dict() for declaration in planned_declarations
            ],
        },
    )
    task_index_by_key = {
        key: index
        for index, key in enumerate(
            sorted({(plan.family, plan.subfamily) for plan in plans})
        )
    }
    existing_context_path = Path(output) / ".finalize_context.json"
    existing_context = (
        json.loads(existing_context_path.read_text(encoding="utf-8"))
        if existing_context_path.is_file()
        else None
    )
    if existing_context is not None and existing_context.get("config_hash") != writer.config_hash:
        raise ResumeMismatchError("Existing finalize context has a different configuration")
    camera_rows: dict[str, dict[str, Any]] = {
        str(row["camera_id"]): dict(row)
        for row in ((existing_context or {}).get("cameras") or ())
    }
    provenance_rows = {
        _suite_provenance_key(row): dict(row)
        for row in ((existing_context or {}).get("provenance") or ())
    }
    committed_records: list[EpisodeRecord] = []
    committed_by_uuid = {record.episode_uuid: record for record in writer.records()}
    if existing_context is not None:
        expected_uuids = {plan.episode_uuid for plan in plans}
        if set(committed_by_uuid) != expected_uuids:
            raise ResumeMismatchError(
                "Finalize context exists but committed episode membership is incomplete or changed"
            )
    for index, plan in enumerate(plans):
        if existing_context is not None and plan.episode_uuid in committed_by_uuid:
            existing_record = committed_by_uuid[plan.episode_uuid]
            if existing_record.episode_index != index:
                raise ResumeMismatchError(
                    f"Committed episode index changed: {plan.episode_uuid}"
                )
            committed_records.append(existing_record)
            continue
        if native_backend is not None:
            backend_result = native_backend.run(plan)
            result = backend_result.simulation
            rendered = dict(backend_result.as_renderer_payload())
            rendered["backend_provenance"] = dict(backend_result.backend_provenance)
            rendered["transition_rows"] = list(backend_result.transition_events)
        else:
            result = adapter.simulate(plan)
            assert renderer is not None
            rendered = dict(renderer(result=result, episode_index=index, views=tuple(config["views"])))
        if "videos" not in rendered:
            raise ValueError("Renderer result must contain a videos mapping")
        calibration_mapping = _register_camera_calibrations(rendered, camera_rows)
        provenance_row = {
            "source_generator": result.plan.source_generator,
            "source_generator_version": result.plan.source_generator_version,
            "generator_git_commit": git_commit,
            "config_hash": writer.config_hash,
            "simulator_name": str(result.simulator.get("name", "unknown")),
            "simulator_version": str(result.simulator.get("version", "unknown")),
            "renderer": str(rendered.get("renderer") or renderer_spec),
            "execution_backend": backend_name,
            "integrated_native_backend": backend_name == "native_mujoco",
            "release_eligibility_is_per_episode": True,
            "asset_roots": asset_roots,
        }
        provenance_key = _suite_provenance_key(provenance_row)
        previous_provenance = provenance_rows.get(provenance_key)
        if previous_provenance is not None and previous_provenance != provenance_row:
            raise ExistingOutputError(f"Conflicting provenance row: {provenance_key}")
        provenance_rows[provenance_key] = provenance_row
        record = _episode_record(result, index, git_commit, rendered)
        record.task_index = task_index_by_key[(record.family, record.subfamily)]
        committed_records.append(
            writer.write_episode(
                record,
                frame_rows=rendered.get("frame_rows") or _default_frame_rows(result, index),
                videos=rendered["videos"],
                high_rate_rows=rendered.get("high_rate_rows") or result.high_rate_states,
                event_rows=rendered.get("event_rows") or result.contacts,
                transition_rows=rendered.get("transition_rows") or (),
                object_state_rows=rendered.get("object_state_rows") or (),
                camera_calibration_ids=calibration_mapping,
            )
        )
    counterfactual_rows = _expected_counterfactual_rows(
        planned_declarations,
        committed_records,
    )
    finalize_context = {
        "config_hash": writer.config_hash,
        "cameras": [camera_rows[key] for key in sorted(camera_rows)],
        "provenance": [provenance_rows[key] for key in sorted(provenance_rows)],
        "counterfactual_families": counterfactual_rows,
    }
    context_path = Path(output) / ".finalize_context.json"
    if context_path.exists():
        if json.loads(context_path.read_text(encoding="utf-8")) != finalize_context:
            raise ExistingOutputError(f"Finalize context differs and will not be overwritten: {context_path}")
    else:
        atomic_write_json(context_path, finalize_context)
    _json_print(
        {
            "dataset_root": str(Path(output).resolve()),
            "episode_count": len(plans),
            "backend": backend_name,
        }
    )
    return 0


def _command_build_splits(arguments: argparse.Namespace) -> int:
    root = Path(arguments.dataset).resolve(strict=True)
    records = load_episode_records(root)
    split_config: dict[str, Any] = {}
    config_path = getattr(arguments, "config", None)
    if config_path:
        import yaml

        split_config = dict(
            yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        )
        if split_config.get("schema_version") != "dynamic-robot-splits/v2":
            raise ValueError("Unsupported split configuration schema_version")
    fractions = dict(split_config.get("fractions") or {})
    requested_seed = getattr(arguments, "seed", None)
    assigner = SplitAssigner(
        seed=(
            requested_seed
            if requested_seed is not None
            else split_config.get("seed", 0)
        ),
        train_fraction=float(fractions.get("train", 0.80)),
        validation_fraction=float(fractions.get("validation", 0.10)),
        test_fraction=float(fractions.get("test", 0.10)),
    )
    assignments, diagnostics = assigner.assign_with_diagnostics(records)
    problems = []
    split_path = ensure_not_source_path(root / "meta" / "splits.parquet")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(value) for value in assignments]
    if split_path.exists():
        existing = read_parquet_rows(split_path)
        if existing != rows:
            raise ExistingOutputError(
                f"A different split assignment already exists and will not be overwritten: {split_path}"
            )
    else:
        write_parquet_atomic(split_path, rows)
    counts: dict[str, int] = {}
    for value in assignments:
        counts[value.split] = counts.get(value.split, 0) + 1
    diagnostics_path = root / "meta" / "split_diagnostics.json"
    episodes_path = root / "meta" / "episodes.parquet"
    diagnostics_payload = {
        **asdict(diagnostics),
        "schema_version": "dynamic-robot-split-diagnostics/v2",
        "dataset_episodes_sha256": (
            sha256_file(episodes_path) if episodes_path.is_file() else None
        ),
        "splits_parquet_sha256": sha256_file(split_path),
        "episode_uuid_set_sha256": sha256_json(
            sorted(value.episode_uuid for value in assignments)
        ),
        "assignment_count": len(assignments),
        "split_config_sha256": (
            sha256_file(Path(config_path).resolve(strict=True))
            if config_path
            else sha256_json(
                {
                    "schema_version": "dynamic-robot-splits/v2",
                    "seed": assigner.seed,
                    "fractions": diagnostics.requested_fractions,
                }
            )
        ),
    }
    if diagnostics_path.exists():
        if json.loads(diagnostics_path.read_text(encoding="utf-8")) != diagnostics_payload:
            raise ExistingOutputError(
                f"A different split diagnostic already exists and will not be overwritten: {diagnostics_path}"
            )
    else:
        atomic_write_json(diagnostics_path, diagnostics_payload)
    _json_print({
        "split_path": str(split_path),
        "episode_count": len(assignments),
        "counts": counts,
        "diagnostics_path": str(diagnostics_path),
        "sparse_strata": diagnostics.sparse_strata,
    })
    return 0


def _command_stats(arguments: argparse.Namespace) -> int:
    report = collect_dataset_statistics(
        arguments.dataset,
        wan_root=arguments.wan_root,
        probe_streams=not arguments.no_probe,
    )
    payload = report.to_dict()
    if arguments.output:
        atomic_write_json(arguments.output, payload)
    _json_print(payload)
    return 0


def _command_calibrate_physics(arguments: argparse.Namespace) -> int:
    report = calibrate_physics_catalog(
        arguments.config,
        observations=arguments.observations,
    )
    payload = report.to_dict()
    if arguments.output:
        atomic_write_json(arguments.output, payload)
    _json_print(payload)
    return 0 if report.passed else 1


def _command_evaluate_readiness(arguments: argparse.Namespace) -> int:
    report = evaluate_readiness(
        arguments.dataset,
        arguments.gate,
        wan_root=arguments.wan_root,
        model_evaluation_path=arguments.model_evaluation,
        prerequisite_report_path=arguments.prerequisite_report,
        acceptance_report_path=getattr(arguments, "acceptance_report", None),
    )
    payload = report.to_dict()
    if arguments.output:
        requested_output = Path(arguments.output).resolve()
        canonical_output = Path(
            str(report.provenance["canonical_report_path"])
        ).resolve()
        if requested_output != canonical_output:
            raise ValueError(
                "Persisted readiness reports must use the canonical path "
                f"{canonical_output}; omit --output for an exploratory stdout-only evaluation"
            )
        atomic_write_json(canonical_output, payload)
    _json_print(payload)
    return 0 if report.passed else 1


def _normalized_json(value: Any) -> Any:
    """Return the exact JSON value that immutable sidecars persist."""

    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )


def _write_or_validate_immutable_json(path: Path, value: Any) -> None:
    """Publish an immutable JSON sidecar, accepting only byte-equivalent resumes."""

    normalized = _normalized_json(value)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ExistingOutputError(
                f"Immutable suite sidecar differs and will not be overwritten: {path}"
            )
        return
    atomic_write_json(path, normalized)


def _suite_plan_ledger(
    config_path: Path,
    planned: Sequence[PlannedSuiteEpisode],
    declarations: Sequence[CounterfactualFamilyRecord],
    *,
    generator_git_commit: str,
) -> dict[str, Any]:
    backend_counts = Counter(item.execution_backend for item in planned)
    full_native = bool(planned) and all(
        item.execution_backend == "native_mujoco" for item in planned
    )
    return {
        "schema_version": "dynamic-robot-suite-plan-ledger/v1",
        "suite_name": planned[0].case.suite_name if planned else None,
        "source_config_path": str(config_path),
        "source_config_sha256": sha256_file(config_path),
        "generator_git_commit": generator_git_commit,
        "planned_case_count": len(planned),
        "execution_backend_counts": dict(sorted(backend_counts.items())),
        "full_native": full_native,
        "execution_scope": (
            ", ".join(
                f"{count} cases use {backend}"
                for backend, count in sorted(backend_counts.items())
            )
        ),
        "retry_policy": {
            "retry_on_outcome_mismatch": False,
            "retry_failed_case_within_suite_version": False,
        },
        "planned_episodes": [
            {
                "case": item.case.to_dict(),
                "episode_uuid": item.episode_uuid,
                "execution_backend": item.execution_backend,
                "episode_plan": item.episode_plan.to_dict(),
                "scenario_spec_hash": (
                    item.scenario_spec.spec_hash if item.scenario_spec is not None else None
                ),
            }
            for item in planned
        ],
        "counterfactual_families": [value.to_dict() for value in declarations],
    }


def _execute_suite_item(
    item: PlannedSuiteEpisode,
    native_backend: Any | None,
) -> tuple[Any, dict[str, Any]]:
    """Execute one planned route exactly once; measured outcome is never retried."""

    if item.execution_backend == "native_mujoco":
        if native_backend is None:
            raise RuntimeError("native_mujoco backend initialization failed")
        backend_result = native_backend.run(item.episode_plan)
        result = backend_result.simulation
        rendered = dict(backend_result.as_renderer_payload())
        rendered["backend_provenance"] = dict(backend_result.backend_provenance)
        rendered["transition_rows"] = list(backend_result.transition_events)
    else:
        from .families import get_family
        from .smoke_runner import render_result

        result = get_family(item.episode_plan.family).simulate(item.episode_plan)
        rendered = dict(
            render_result(
                result=result,
                episode_index=item.case.case_index,
                views=item.case.views,
            )
        )
        rendered["backend_provenance"] = {
            "execution_backend": "diagnostic_quarantine",
            "production_eligible": False,
            "suite_case_index": item.case.case_index,
        }
        rendered["transition_rows"] = []
    if result.plan.episode_uuid != item.episode_uuid:
        raise ValueError("Executed result changed the immutable planned episode UUID")
    if "videos" not in rendered:
        raise ValueError("Suite execution did not return synchronized videos")
    return result, rendered


_SuiteProvenanceKey = tuple[str, str, str, str, str, str]


def _suite_provenance_key(row: Mapping[str, Any]) -> _SuiteProvenanceKey:
    """Return the identity of one truthful suite execution lineage.

    A source adapter can expose multiple simulators (notably the quarantined
    legacy proxies), so adapter name/version/backend alone is not a unique
    provenance identity. Keep those simulator and renderer lineages separate
    while still rejecting inconsistent duplicate rows for the same identity.
    """

    return (
        str(row["source_generator"]),
        str(row["source_generator_version"]),
        str(row.get("simulator_name", "unknown")),
        str(row.get("simulator_version", "unknown")),
        str(row.get("renderer", "unknown")),
        str(
            row.get("execution_backend")
            or (
                "native_mujoco"
                if row.get("integrated_native_backend") is True
                else "diagnostic"
            )
        ),
    )


def _merge_suite_runtime_rows(
    root: Path,
    *,
    config_hash: str,
) -> tuple[dict[str, dict[str, Any]], dict[_SuiteProvenanceKey, dict[str, Any]]]:
    """Recover cameras/provenance from immutable per-case resume sidecars."""

    cameras: dict[str, dict[str, Any]] = {}
    provenance: dict[_SuiteProvenanceKey, dict[str, Any]] = {}
    runtime_root = root / ".suite_runtime"
    if not runtime_root.is_dir():
        return cameras, provenance
    for path in sorted(runtime_root.glob("case-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("config_hash") != config_hash:
            raise ResumeMismatchError(f"Suite runtime sidecar has a different config: {path}")
        for row in value.get("cameras", ()):
            item = dict(row)
            identifier = str(item["camera_id"])
            existing = cameras.get(identifier)
            if existing is not None and existing != item:
                raise ExistingOutputError(f"Conflicting camera calibration: {identifier}")
            cameras[identifier] = item
        for row in value.get("provenance", ()):
            item = dict(row)
            key = _suite_provenance_key(item)
            existing = provenance.get(key)
            if existing is not None and existing != item:
                raise ExistingOutputError(f"Conflicting suite provenance row: {key}")
            provenance[key] = item
    return cameras, provenance


def _expected_counterfactual_rows(
    expected: Sequence[CounterfactualFamilyRecord],
    committed_records: Sequence[EpisodeRecord],
) -> list[dict[str, Any]]:
    """Return the immutable pre-simulation declarations without substitution.

    Runtime evidence is validated against these hashes; it must never replace
    the planned values merely because all committed siblings agree with one
    another.
    """

    del committed_records
    return [declaration.to_table_row() for declaration in expected]


def _suite_acceptance_gates(
    planned: Sequence[PlannedSuiteEpisode],
    committed_records: Sequence[EpisodeRecord],
    *,
    required_rigid_outcome_families: Sequence[str],
    required_physics_sweeps: Mapping[str, Mapping[str, Any]] | None = None,
    counterfactual_problems: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate immutable 160-branch structural and measured-outcome gates."""

    planned_by_uuid = {item.episode_uuid: item for item in planned}
    planned_categories = Counter(item.case.category_id for item in planned)
    committed_categories = Counter(
        planned_by_uuid[record.episode_uuid].case.category_id
        for record in committed_records
        if record.episode_uuid in planned_by_uuid
    )
    planned_styles = Counter(item.case.scene_style for item in planned)
    committed_styles = Counter(
        planned_by_uuid[record.episode_uuid].case.scene_style
        for record in committed_records
        if record.episode_uuid in planned_by_uuid
    )
    required_streams = {
        "observation.images.main",
        "observation.images.secondary",
    }
    planned_two_views = all(
        {canonical_camera_name(view) for view in item.case.views} == required_streams
        for item in planned
    )
    committed_two_views = all(
        set(record.video_paths) == required_streams for record in committed_records
    ) and bool(committed_records)
    full_native = bool(planned) and all(
        item.execution_backend == "native_mujoco" for item in planned
    )
    native_episode_uuids = {
        item.episode_uuid
        for item in planned
        if item.execution_backend == "native_mujoco"
    }
    committed_native_records = [
        record
        for record in committed_records
        if record.episode_uuid in native_episode_uuids
    ]
    native_physics_qc_pass = (
        len(committed_native_records) == len(native_episode_uuids)
        and bool(native_episode_uuids)
        and all(record.physics_qc_pass for record in committed_native_records)
    )
    sweep_evidence = sweep_acceptance_evidence(
        planned,
        committed_records,
        required_physics_sweeps or {},
    )
    outcome_classes: dict[str, list[str]] = {}
    missing_outcome_families: list[str] = []
    required_outcome_classes: dict[str, list[str]] = {}
    missing_outcome_classes: dict[str, list[str]] = {}
    intended_to_actual_class = {
        "success": "success",
        "success_seeking": "success",
        "near_miss": "near_miss",
        "contact_failure": "contact_failure",
        "no_op": "no_op",
        "bad_action": "wrong_action",
        "wrong_action": "wrong_action",
    }
    for family in required_rigid_outcome_families:
        classes = sorted(
            {
                (
                    record.actual_outcome_class.value
                    if hasattr(record.actual_outcome_class, "value")
                    else str(record.actual_outcome_class)
                )
                for record in committed_records
                if record.family == family
            }
        )
        outcome_classes[family] = classes
        has_success = "success" in classes
        has_measured_failure = any(
            value not in {"success", "unverified"} for value in classes
        )
        if not (has_success and has_measured_failure):
            missing_outcome_families.append(family)
        expected = sorted(
            {
                intended_to_actual_class[item.episode_plan.intended_branch]
                for item in planned
                if item.episode_plan.family == family
                and item.episode_plan.intended_branch in intended_to_actual_class
            }
        )
        required_outcome_classes[family] = expected
        missing = sorted(set(expected) - set(classes))
        if missing:
            missing_outcome_classes[family] = missing
    checks = {
        "exact_160_planned": len(planned) == 160,
        "exact_160_committed": len(committed_records) == 160,
        "category_counts_match_plan": committed_categories == planned_categories,
        "at_least_three_planned_styles": len(planned_styles) >= 3,
        "all_planned_styles_committed": set(committed_styles) == set(planned_styles),
        "planned_two_synchronized_views": planned_two_views,
        "committed_two_synchronized_views": committed_two_views,
        "all_acceptance_cases_use_native_mujoco": full_native,
        "all_native_rigid_physics_qc_pass": native_physics_qc_pass,
        "all_required_physics_sweeps_measured_and_monotonic": sweep_evidence[
            "passed"
        ],
        "required_rigid_measured_success_and_failure": not missing_outcome_families,
        "all_planned_rigid_actual_outcome_classes_observed": not missing_outcome_classes,
        "counterfactual_membership_and_invariants": not counterfactual_problems,
    }
    blockers: list[str] = []
    for name, passed in checks.items():
        if not passed:
            blockers.append(name)
    blockers.extend(
        f"missing_measured_success_or_failure:{family}"
        for family in missing_outcome_families
    )
    blockers.extend(
        f"missing_actual_outcome_classes:{family}:{','.join(classes)}"
        for family, classes in sorted(missing_outcome_classes.items())
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "blockers": blockers,
        "planned_category_counts": dict(sorted(planned_categories.items())),
        "committed_category_counts": dict(sorted(committed_categories.items())),
        "planned_style_counts": dict(sorted(planned_styles.items())),
        "committed_style_counts": dict(sorted(committed_styles.items())),
        "full_native": full_native,
        "required_rigid_outcome_families": list(required_rigid_outcome_families),
        "required_actual_outcome_classes": required_outcome_classes,
        "measured_actual_outcome_classes": outcome_classes,
        "missing_required_outcome_families": missing_outcome_families,
        "missing_required_actual_outcome_classes": missing_outcome_classes,
        "counterfactual_problems": list(counterfactual_problems),
        "physics_sweep_evidence": sweep_evidence,
    }


def _command_generate_suite(arguments: argparse.Namespace) -> int:
    config_path = Path(arguments.config).resolve(strict=True)
    cases = expand_suite(config_path)
    planned = plan_suite_cases(cases)
    declarations = build_planned_counterfactual_family_records(planned)
    planned_backend_counts = Counter(item.execution_backend for item in planned)
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - base dependency is pinned
        raise RuntimeError("generate-suite requires PyYAML") from error
    raw_suite_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required_outcome_families = tuple(
        str(value)
        for value in dict(raw_suite_config.get("requirements") or {}).get(
            "require_measured_success_and_failure",
            (),
        )
    )
    required_physics_sweeps = {
        str(name): dict(value)
        for name, value in dict(
            dict(raw_suite_config.get("requirements") or {}).get(
                "require_measured_physics_sweeps", {}
            )
        ).items()
    }
    if arguments.dry_run:
        _json_print(
            {
                "dry_run": True,
                "suite": cases[0].suite_name if cases else None,
                "case_count": len(cases),
                "execution_backend_counts": dict(sorted(planned_backend_counts.items())),
                "full_native": bool(planned)
                and all(
                    item.execution_backend == "native_mujoco" for item in planned
                ),
                "cases": [case.to_dict() for case in cases],
                "planned_episode_uuids": [item.episode_uuid for item in planned],
                "counterfactual_families": [value.to_dict() for value in declarations],
            }
        )
        return 0
    if not arguments.output:
        raise ValueError("generate-suite requires --output unless --dry-run is used")

    output = ensure_not_source_path(arguments.output)
    repository_root = Path(__file__).resolve().parents[2]
    git_commit = get_git_commit(repository_root)
    ledger_base = _suite_plan_ledger(
        config_path,
        planned,
        declarations,
        generator_git_commit=git_commit,
    )
    ledger_hash = sha256_json(ledger_base)
    asset_roots = {
        name: os.environ[name]
        for name in (
            "ROBOCASA_ROOT",
            "ROBOTWIN_ROOT",
            "ROBOTWIN_2_ROOT",
            "MUJOCO_MENAGERIE_ROOT",
        )
        if os.environ.get(name)
    }
    writer_config = {
        "schema_version": "dynamic-robot-suite-run/v1",
        "suite_name": cases[0].suite_name if cases else None,
        "suite_config_sha256": sha256_file(config_path),
        "planned_ledger_hash": ledger_hash,
        "planned_case_count": len(planned),
        "execution_backend_counts": dict(sorted(planned_backend_counts.items())),
        "retry_on_outcome_mismatch": False,
        "retry_failed_case_within_suite_version": False,
        "sim_hz": 240,
        "control_hz": 60,
        "video_hz": 30,
        "provenance": {
            "generator_git_commit": git_commit,
            "asset_roots": asset_roots,
            "native_renderer": "integrated:mujoco.Renderer",
            "quarantine_renderer": "dynamic_robot_dataset.diagnostic_state_renderer/v1",
        },
    }
    writer = EpisodeWriter(output, writer_config, resume=arguments.resume)
    ledger = {**ledger_base, "writer_config_hash": writer.config_hash}
    _write_or_validate_immutable_json(output / ".suite_plan.json", ledger)
    (output / ".suite_attempts").mkdir(parents=True, exist_ok=True)
    (output / ".suite_runtime").mkdir(parents=True, exist_ok=True)

    task_index_by_key = {
        key: index
        for index, key in enumerate(
            sorted({(item.episode_plan.family, item.episode_plan.subfamily) for item in planned})
        )
    }
    planned_by_uuid = {item.episode_uuid: item for item in planned}
    committed_by_uuid = {record.episode_uuid: record for record in writer.records()}
    unexpected = sorted(set(committed_by_uuid) - set(planned_by_uuid))
    if unexpected:
        raise ExistingOutputError(f"Suite output contains unplanned episode UUIDs: {unexpected}")
    for episode_uuid, record in committed_by_uuid.items():
        item = planned_by_uuid[episode_uuid]
        if record.episode_index != item.case.case_index:
            raise ResumeMismatchError(f"Committed suite episode index changed: {episode_uuid}")
        expected_task = task_index_by_key[(record.family, record.subfamily)]
        if record.task_index != expected_task:
            raise ResumeMismatchError(f"Committed suite task index changed: {episode_uuid}")

    camera_rows, provenance_rows = _merge_suite_runtime_rows(
        output,
        config_hash=writer.config_hash,
    )

    native_backend: Any | None = None
    native_initialization_error: Exception | None = None
    if any(
        item.execution_backend == "native_mujoco"
        and item.episode_uuid not in committed_by_uuid
        and not (output / ".suite_attempts" / f"case-{item.case.case_index:06d}.json").exists()
        for item in planned
    ):
        try:
            from .backends import get_backend

            native_backend = get_backend("native_mujoco")
        except Exception as error:  # recorded once per immutable planned case below
            native_initialization_error = error

    for item in planned:
        index = item.case.case_index
        attempt_path = output / ".suite_attempts" / f"case-{index:06d}.json"
        if attempt_path.exists():
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            if (
                attempt.get("config_hash") != writer.config_hash
                or attempt.get("episode_uuid") != item.episode_uuid
                or int(attempt.get("case_index", -1)) != index
            ):
                raise ResumeMismatchError(f"Suite attempt identity changed: {attempt_path}")
            if attempt.get("status") == "committed" and item.episode_uuid not in committed_by_uuid:
                raise RuntimeError(f"Committed suite attempt has no episode marker: {attempt_path}")
            if attempt.get("status") == "failed" and item.episode_uuid in committed_by_uuid:
                raise RuntimeError(f"Failed suite attempt unexpectedly has an episode marker: {attempt_path}")
            if attempt.get("status") not in {"committed", "failed"}:
                raise RuntimeError(f"Unknown suite attempt status: {attempt_path}")
            continue
        if item.episode_uuid in committed_by_uuid:
            _write_or_validate_immutable_json(
                attempt_path,
                {
                    "schema_version": "dynamic-robot-suite-attempt/v1",
                    "config_hash": writer.config_hash,
                    "case_index": index,
                    "episode_uuid": item.episode_uuid,
                    "execution_backend": item.execution_backend,
                    "status": "committed",
                    "recovered_existing_commit": True,
                    "outcome_retry_count": 0,
                },
            )
            continue

        try:
            if item.execution_backend == "native_mujoco" and native_initialization_error:
                raise RuntimeError(
                    f"native backend initialization failed: {native_initialization_error}"
                ) from native_initialization_error
            result, rendered = _execute_suite_item(item, native_backend)
            case_camera_rows: dict[str, dict[str, Any]] = {}
            calibration_mapping = _register_camera_calibrations(
                rendered,
                case_camera_rows,
            )
            case_camera_rows = {
                identifier: _normalized_json(row)
                for identifier, row in case_camera_rows.items()
            }
            for identifier, row in case_camera_rows.items():
                previous_camera = camera_rows.get(identifier)
                if previous_camera is not None and previous_camera != row:
                    raise ExistingOutputError(f"Conflicting camera calibration: {identifier}")
            provenance_row = {
                "source_generator": result.plan.source_generator,
                "source_generator_version": result.plan.source_generator_version,
                "generator_git_commit": git_commit,
                "config_hash": writer.config_hash,
                "simulator_name": str(result.simulator.get("name", "unknown")),
                "simulator_version": str(result.simulator.get("version", "unknown")),
                "renderer": str(rendered.get("renderer") or "unknown"),
                "execution_backend": item.execution_backend,
                "integrated_native_backend": item.execution_backend == "native_mujoco",
                "release_eligibility_is_per_episode": True,
                "asset_roots": asset_roots,
            }
            provenance_key = _suite_provenance_key(provenance_row)
            previous_provenance = provenance_rows.get(provenance_key)
            if previous_provenance is not None and previous_provenance != provenance_row:
                raise ExistingOutputError(f"Conflicting provenance row: {provenance_key}")
            runtime_path = output / ".suite_runtime" / f"case-{index:06d}.json"
            _write_or_validate_immutable_json(
                runtime_path,
                {
                    "schema_version": "dynamic-robot-suite-runtime-context/v1",
                    "config_hash": writer.config_hash,
                    "case_index": index,
                    "episode_uuid": item.episode_uuid,
                    "cameras": [
                        case_camera_rows[identifier]
                        for identifier in sorted(set(calibration_mapping.values()))
                    ],
                    "provenance": [provenance_row],
                },
            )
            camera_rows.update(case_camera_rows)
            provenance_rows[provenance_key] = provenance_row
            record = _episode_record(result, index, git_commit, rendered)
            record.task_index = task_index_by_key[(record.family, record.subfamily)]
            committed = writer.write_episode(
                record,
                frame_rows=rendered.get("frame_rows") or _default_frame_rows(result, index),
                videos=rendered["videos"],
                high_rate_rows=rendered.get("high_rate_rows") or result.high_rate_states,
                event_rows=rendered.get("event_rows") or result.contacts,
                transition_rows=rendered.get("transition_rows") or (),
                object_state_rows=rendered.get("object_state_rows") or (),
                camera_calibration_ids=calibration_mapping,
            )
            committed_by_uuid[item.episode_uuid] = committed
            _write_or_validate_immutable_json(
                attempt_path,
                {
                    "schema_version": "dynamic-robot-suite-attempt/v1",
                    "config_hash": writer.config_hash,
                    "case_index": index,
                    "episode_uuid": item.episode_uuid,
                    "execution_backend": item.execution_backend,
                    "status": "committed",
                    "actual_outcome": committed.actual_outcome,
                    "actual_outcome_class": committed.actual_outcome_class.value,
                    "task_success": committed.task_success,
                    "outcome_retry_count": 0,
                },
            )
        except (ExistingOutputError, ResumeMismatchError, PermissionError):
            raise
        except Exception as error:
            _write_or_validate_immutable_json(
                attempt_path,
                {
                    "schema_version": "dynamic-robot-suite-attempt/v1",
                    "config_hash": writer.config_hash,
                    "case_index": index,
                    "episode_uuid": item.episode_uuid,
                    "execution_backend": item.execution_backend,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "outcome_retry_count": 0,
                    "retry_policy": "new generator/suite version required",
                },
            )

    committed_records = sorted(
        committed_by_uuid.values(), key=lambda value: value.episode_index
    )
    counterfactual_rows = _expected_counterfactual_rows(
        declarations,
        committed_records,
    )
    finalized_declarations = [
        CounterfactualFamilyRecord.from_dict(row) for row in counterfactual_rows
    ]
    derived_by_uuid = {
        record.episode_uuid: {
            "derived_initial_state_hash": record.extras.get(
                "derived_initial_state_hash"
            ),
            "derived_action_hash": record.extras.get("derived_action_hash"),
        }
        for record in committed_records
    }
    counterfactual_problems = validate_counterfactual_family_records(
        finalized_declarations,
        committed_records,
        derived_by_uuid=derived_by_uuid,
    )
    finalize_context = {
        "config_hash": writer.config_hash,
        "suite_plan_ledger": ".suite_plan.json",
        "cameras": [camera_rows[key] for key in sorted(camera_rows)],
        "provenance": [provenance_rows[key] for key in sorted(provenance_rows)],
        "counterfactual_families": counterfactual_rows,
    }
    _write_or_validate_immutable_json(output / ".finalize_context.json", finalize_context)

    attempts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / ".suite_attempts").glob("case-*.json"))
    ]
    attempt_by_uuid = {str(value["episode_uuid"]): value for value in attempts}
    unexpected_attempts = sorted(set(attempt_by_uuid) - set(planned_by_uuid))
    if unexpected_attempts:
        raise ExistingOutputError(
            f"Suite output contains attempt records for unplanned UUIDs: {unexpected_attempts}"
        )
    missing_attempts = sorted(set(planned_by_uuid) - set(attempt_by_uuid))
    if missing_attempts:
        raise RuntimeError(f"Suite ended without attempt records: {missing_attempts}")
    failed = [value for value in attempts if value["status"] == "failed"]
    committed_backend_counts = Counter(
        planned_by_uuid[record.episode_uuid].execution_backend
        for record in committed_records
    )
    acceptance_gates = _suite_acceptance_gates(
        planned,
        committed_records,
        required_rigid_outcome_families=required_outcome_families,
        required_physics_sweeps=required_physics_sweeps,
        counterfactual_problems=counterfactual_problems,
    )
    report = {
        "schema_version": "dynamic-robot-suite-execution/v1",
        "suite_name": cases[0].suite_name if cases else None,
        "dataset_root": str(output.resolve()),
        "writer_config_hash": writer.config_hash,
        "suite_plan_sha256": sha256_file(output / ".suite_plan.json"),
        "source_config_sha256": sha256_file(config_path),
        "committed_episode_uuid_set_sha256": sha256_json(
            sorted(record.episode_uuid for record in committed_records)
        ),
        "planned_case_count": len(planned),
        "planned_execution_backend_counts": dict(sorted(planned_backend_counts.items())),
        "committed_episode_count": len(committed_records),
        "committed_execution_backend_counts": dict(sorted(committed_backend_counts.items())),
        "failed_case_count": len(failed),
        "failed_cases": [
            {
                "case_index": value["case_index"],
                "episode_uuid": value["episode_uuid"],
                "execution_backend": value["execution_backend"],
                "error_type": value.get("error_type"),
                "error": value.get("error"),
            }
            for value in failed
        ],
        "counterfactual_family_count": len(declarations),
        "counterfactual_expected_members_preserved": True,
        "counterfactual_validation_problems": counterfactual_problems,
        "outcome_mismatch_retry_count": 0,
        "full_native": acceptance_gates["full_native"],
        "execution_scope": (
            "native_mujoco rigid acceptance plus explicitly quarantined diagnostic "
            "cloth/rope/legacy controls"
        ),
        "release_eligible_episode_count": sum(
            record.release_eligible for record in committed_records
        ),
        "actual_outcome_class_counts": dict(
            sorted(
                Counter(
                    record.actual_outcome_class.value for record in committed_records
                ).items()
            )
        ),
        "acceptance_gates": acceptance_gates,
        "passed_execution": (
            not failed
            and len(committed_records) == len(planned)
            and acceptance_gates["passed"]
        ),
    }
    _write_or_validate_immutable_json(output / ".suite_execution.json", report)
    _json_print(report)
    return 0 if report["passed_execution"] else 1


def _feature_unit(name: str) -> str:
    """Return an explicit unit for canonical state/action column names."""

    lowered = name.lower()
    if "joint_acceleration" in lowered:
        return "rad/s^2"
    if "joint_velocity" in lowered or lowered.endswith(".dq"):
        return "rad/s"
    if "joint_position" in lowered or "target_rpy" in lowered or lowered.endswith(".rpy"):
        return "rad"
    if "angular_velocity" in lowered:
        return "rad/s"
    if "linear_velocity" in lowered or "twist" in lowered:
        return "m/s"
    if "quaternion" in lowered:
        return "1 (WXYZ)"
    if "force" in lowered:
        return "N"
    if "torque" in lowered or "tau" in lowered:
        return "N*m"
    if any(
        token in lowered
        for token in ("task_phase", "task.phase", "motion_mode", "active_surface", "contact_role")
    ):
        return "categorical"
    if any(token in lowered for token in ("position", "vertices", ".points", "centroid", "width")):
        return "m"
    if lowered.endswith(".enabled") or lowered.endswith(".active") or lowered.endswith(".flag"):
        return "bool"
    return "unknown_explicitly_unmapped"


def _command_finalize(arguments: argparse.Namespace) -> int:
    root = Path(arguments.dataset).resolve(strict=True)
    generation = json.loads((root / ".generation.json").read_text(encoding="utf-8"))
    config = generation["resolved_config"]
    writer = EpisodeWriter(root, config, resume=True)
    records = load_episode_records(root)
    if not records:
        raise ValueError("No committed episodes to finalize")
    first = records[0]
    feature_types: dict[str, str] = {}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Finalization requires pyarrow") from exc
    for record in records:
        if not record.frame_data_path:
            continue
        schema = pq.ParquetFile(root / record.frame_data_path).schema_arrow
        for field in schema:
            field_type = str(field.type)
            existing_type = feature_types.get(field.name)
            if existing_type is not None and existing_type != field_type:
                raise ValueError(
                    f"Frame field {field.name!r} changes type across episodes: "
                    f"{existing_type} != {field_type}"
                )
            feature_types[field.name] = field_type
    required_frame_fields = {
        "episode_index",
        "frame_index",
        "video_frame_index",
        "task_index",
        "timestamp",
    }
    missing_frame_fields = sorted(required_frame_fields - set(feature_types))
    if missing_frame_fields:
        raise ValueError(f"Canonical frame schema is missing fields: {missing_frame_fields}")
    action_names = sorted(name for name in feature_types if name.startswith("action."))
    excluded_prefixes = ("action.", "contact.", "event.", "assistance.")
    excluded_names = {"episode_index", "frame_index", "video_frame_index", "timestamp", "task_index"}
    state_names = sorted(
        name
        for name in feature_types
        if name not in excluded_names and not name.startswith(excluded_prefixes)
    )

    info = DatasetInfo(
        name=arguments.name or root.name,
        description=arguments.description,
        time_base=TimeBase(
            sim_hz=float(config.get("sim_hz", 240)),
            control_hz=float(config.get("control_hz", 60)),
            video_hz=float(config.get("video_hz", 30)),
        ),
        state_features=[
            NamedFeature(name, feature_types[name], _feature_unit(name), description="Named frame-table state column")
            for name in state_names
        ],
        action_features=[
            NamedFeature(name, feature_types[name], _feature_unit(name), description="Named frame-table action column")
            for name in action_names
        ],
        generator_version=first.generator_git_commit,
        extras={"generation_config_hash": generation["config_hash"]},
    )
    context_path = root / ".finalize_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.is_file() else {}
    if "counterfactual_families" not in context:
        plan_path = root / ".generation_plan.json"
        if plan_path.is_file():
            planned_context = json.loads(plan_path.read_text(encoding="utf-8"))
            context["counterfactual_families"] = [
                CounterfactualFamilyRecord.from_dict(value).to_table_row()
                for value in planned_context.get("counterfactual_families", ())
            ]
    writer.finalize(
        info,
        cameras=context.get("cameras", ()),
        provenance=context.get("provenance", ()),
        counterfactual_families=context.get("counterfactual_families"),
    )
    _json_print({"dataset_root": str(root), "episode_count": len(records), "finalized": True})
    return 0


def _command_validate(arguments: argparse.Namespace) -> int:
    report = validate_dataset(arguments.dataset, deep_video_checks=not arguments.shallow)
    _json_print(report.to_dict())
    return 0 if report.passed else 1


def _command_qc(arguments: argparse.Namespace) -> int:
    report = validate_dataset(
        arguments.dataset,
        deep_video_checks=True,
        write_reports=True,
        report_dir=arguments.report_dir,
    )
    _json_print(report.to_dict())
    return 0 if report.passed else 1


def _command_export_wan(arguments: argparse.Namespace) -> int:
    summary = export_wan(
        arguments.dataset,
        arguments.output,
        include_nonrelease=arguments.include_nonrelease,
        camera_names=arguments.views,
    )
    _json_print(summary.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the public argument parser without importing family backends."""

    parser = argparse.ArgumentParser(prog="dynamic-robot-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="read-only source inventory")
    inventory.add_argument("--roots", nargs="+", default=DEFAULT_SOURCE_ROOTS)
    inventory.add_argument("--output")
    inventory.set_defaults(handler=_command_inventory)

    generate = subparsers.add_parser("generate", help="plan/simulate an episode family")
    generate.add_argument("--config", help="YAML/JSON generation configuration")
    generate.add_argument("--family")
    generate.add_argument("--subfamily")
    generate.add_argument("--variant")
    generate.add_argument("--robot-model")
    generate.add_argument("--tool-type")
    generate.add_argument("--backend", choices=("diagnostic", "native_mujoco"))
    generate.add_argument("--num-bundles", type=int)
    generate.add_argument("--branches", type=_comma_list)
    generate.add_argument("--views", type=_comma_list)
    generate.add_argument("--seed", type=int)
    generate.add_argument("--randomization-level")
    generate.add_argument("--scene-style")
    generate.add_argument("--sim-hz", type=int)
    generate.add_argument("--control-hz", type=int)
    generate.add_argument("--video-hz", type=int)
    generate.add_argument("--duration-s", type=float)
    generate.add_argument("--physics-sweep", choices=("gravity", "friction", "restitution"))
    generate.add_argument("--output")
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument(
        "--renderer",
        help="diagnostic/backward-compatible renderer callable as module:function",
    )
    generate.set_defaults(handler=_command_generate)

    finalize = subparsers.add_parser("finalize", help="write finalized metadata tables")
    finalize.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    finalize.add_argument("--name")
    finalize.add_argument("--description", default="")
    finalize.set_defaults(handler=_command_finalize)

    validate = subparsers.add_parser("validate", help="validate a canonical dataset")
    validate.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    validate.add_argument("--shallow", action="store_true", help="skip perceptual/frozen-video checks")
    validate.set_defaults(handler=_command_validate)

    splits = subparsers.add_parser("build-splits", help="create deterministic stratified 80/10/10 splits")
    splits.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    splits.add_argument("--config")
    splits.add_argument("--seed", type=int)
    splits.set_defaults(handler=_command_build_splits)

    qc = subparsers.add_parser("qc", help="validate and write QC reports")
    qc.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    qc.add_argument("--report-dir")
    qc.set_defaults(handler=_command_qc)

    wan = subparsers.add_parser("export-wan", help="export 24 FPS, 121-frame Wan clips")
    wan.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    wan.add_argument("--output", required=True)
    wan.add_argument("--include-nonrelease", action="store_true")
    wan.add_argument("--views", type=_comma_list)
    wan.set_defaults(handler=_command_export_wan)

    suite = subparsers.add_parser("generate-suite", help="expand or run a versioned multi-family suite")
    suite.add_argument("--config", required=True)
    suite.add_argument("--output")
    suite.add_argument("--resume", action="store_true")
    suite.add_argument("--dry-run", action="store_true")
    suite.set_defaults(handler=_command_generate_suite)

    stats = subparsers.add_parser("stats", help="report logical, stream, and derived-export hours")
    stats.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    stats.add_argument("--wan-root")
    stats.add_argument("--no-probe", action="store_true", help="use metadata durations instead of probing source streams")
    stats.add_argument("--output")
    stats.set_defaults(handler=_command_stats)

    calibrate = subparsers.add_parser("calibrate-physics", help="validate physics ranges against native calibration evidence")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--observations")
    calibrate.add_argument("--output")
    calibrate.set_defaults(handler=_command_calibrate_physics)

    readiness = subparsers.add_parser("evaluate-readiness", help="evaluate a staged unique-hour release gate")
    readiness.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    readiness.add_argument("--gate", required=True)
    readiness.add_argument("--wan-root")
    readiness.add_argument("--model-evaluation")
    readiness.add_argument("--prerequisite-report")
    readiness.add_argument("--acceptance-report")
    readiness.add_argument("--output")
    readiness.set_defaults(handler=_command_evaluate_readiness)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning a process-compatible status code."""

    arguments = build_parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (ExistingOutputError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
