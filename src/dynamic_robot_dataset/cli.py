"""Common command-line interface for inventory, generation, QC, and export."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .common.episode_writer import EpisodeWriter, load_episode_records, read_parquet_rows, write_parquet_atomic
from .common.cameras import CameraCalibration
from .common.contacts import normalize_assistance, select_task_event_time
from .common.paths import ExistingOutputError, atomic_write_json, ensure_not_source_path
from .common.provenance import get_git_commit, inventory_source
from .common.qc import validate_dataset
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


def _label_status(value: str) -> LabelStatus:
    if value in {"verified", "verified_objective"}:
        return LabelStatus.VERIFIED
    if value == "proxy":
        return LabelStatus.PROXY
    return LabelStatus.UNVERIFIED


def _episode_record(
    result: Any,
    episode_index: int,
    git_commit: str,
    rendered: Mapping[str, Any],
) -> EpisodeRecord:
    status = _label_status(str(result.outcome.label_status))
    dynamics = DynamicsMode(str(result.dynamics_mode))
    release = ReleaseTier(str(result.release_tier))
    if dynamics == DynamicsMode.SCRIPTED_MOTION:
        release = ReleaseTier.SCRIPTED_MOTION
    elif dynamics == DynamicsMode.ASSISTED_CONTACT:
        release = ReleaseTier.ASSISTED_CONTACT
    elif status != LabelStatus.VERIFIED and release == ReleaseTier.FREE_CONTACT:
        release = ReleaseTier.UNVERIFIED
    event_time = (
        float(rendered["task_event_time_s"])
        if rendered.get("task_event_time_s") is not None
        else select_task_event_time(result.contacts)
    )
    simulator_production_eligible = result.simulator.get("production_eligible") is True
    renderer_production_eligible = rendered.get("production_eligible") is True
    quality_flags = [str(value) for value in rendered.get("quality_flags", ())]
    if not simulator_production_eligible:
        quality_flags.append("non_production_family_adapter")
    if not renderer_production_eligible:
        quality_flags.append("renderer_not_production_verified")
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
        actual_outcome=result.actual_outcome,
        task_success=result.outcome.task_success,
        partial_success_score=result.outcome.partial_success_score,
        failure_mode=result.outcome.failure_mode,
        label_confidence=result.outcome.label_confidence,
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
        physics=_physics_metadata(result.plan.physics),
        assistance=normalize_assistance(result.assistance),
        objective_metrics=dict(result.outcome.metrics),
        randomization={
            "scene_style": result.plan.scene_style,
            "randomization_level": result.plan.randomization_level,
            "visual_seed": result.plan.scene_parameters.get("visual_seed", result.plan.scene_seed),
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
            "notes": list(result.notes),
            "event_time_semantics": "first_non_fixture_task_contact",
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
        }
        row.update({key: value for key, value in state.items() if key != "timestamp"})
        row.update({f"action.{key}": value for key, value in action.items() if key != "timestamp"})
        rows.append(row)
    return rows


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


def _command_generate(arguments: argparse.Namespace) -> int:
    from .families import get_family

    config = _generation_config(arguments)
    output = arguments.output or config.pop("output", None)
    adapter = get_family(str(config["family"]))
    if arguments.dry_run:
        plans = [plan.to_dict() for plan in adapter.plan(config)]
        _json_print({"dry_run": True, "episode_count": len(plans), "plans": plans})
        return 0
    if not output:
        raise ValueError("Generation requires --output or an output field in --config")
    renderer_spec = arguments.renderer or os.environ.get("DYNAMIC_ROBOT_RENDERER")
    renderer = _load_renderer(renderer_spec)
    results = adapter.generate(config)
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
            "generator_git_commit": git_commit,
            "asset_roots": asset_roots,
        },
    }
    writer = EpisodeWriter(output, writer_config, resume=arguments.resume)
    task_index_by_key = {
        key: index
        for index, key in enumerate(
            sorted({(result.plan.family, result.plan.subfamily) for result in results})
        )
    }
    camera_rows: dict[str, dict[str, Any]] = {}
    provenance_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for index, result in enumerate(results):
        rendered = dict(renderer(result=result, episode_index=index, views=tuple(config["views"])))
        if "videos" not in rendered:
            raise ValueError("Renderer result must contain a videos mapping")
        raw_cameras = rendered.get("cameras") or rendered.get("camera_calibrations")
        if not raw_cameras:
            raise ValueError("Renderer result must contain camera calibrations")
        camera_values = raw_cameras.items() if isinstance(raw_cameras, Mapping) else (
            (getattr(value, "camera_name", f"camera-{position}"), value)
            for position, value in enumerate(raw_cameras)
        )
        for camera_name, calibration in camera_values:
            row = calibration.to_dict() if hasattr(calibration, "to_dict") else dict(calibration)
            identifier = str(row.get("camera_name") or camera_name)
            row["camera_id"] = identifier
            calibration_payload = dict(row)
            calibration_payload.pop("camera_id", None)
            CameraCalibration.from_dict(calibration_payload)
            existing = camera_rows.get(identifier)
            if existing is not None and existing != row:
                raise ValueError(f"Camera calibration changed within the generation run: {identifier}")
            camera_rows[identifier] = row
        provenance_rows[(result.plan.source_generator, result.plan.source_generator_version)] = {
            "source_generator": result.plan.source_generator,
            "source_generator_version": result.plan.source_generator_version,
            "generator_git_commit": git_commit,
            "config_hash": writer.config_hash,
            "simulator_name": str(result.simulator.get("name", "unknown")),
            "simulator_version": str(result.simulator.get("version", "unknown")),
            "renderer": str(rendered.get("renderer") or renderer_spec),
            "asset_roots": asset_roots,
        }
        record = _episode_record(result, index, git_commit, rendered)
        record.task_index = task_index_by_key[(record.family, record.subfamily)]
        writer.write_episode(
            record,
            frame_rows=rendered.get("frame_rows") or _default_frame_rows(result, index),
            videos=rendered["videos"],
            high_rate_rows=rendered.get("high_rate_rows") or result.high_rate_states,
            event_rows=rendered.get("event_rows") or result.contacts,
            object_state_rows=rendered.get("object_state_rows") or (),
        )
    finalize_context = {
        "config_hash": writer.config_hash,
        "cameras": [camera_rows[key] for key in sorted(camera_rows)],
        "provenance": [provenance_rows[key] for key in sorted(provenance_rows)],
    }
    context_path = Path(output) / ".finalize_context.json"
    if context_path.exists():
        if json.loads(context_path.read_text(encoding="utf-8")) != finalize_context:
            raise ExistingOutputError(f"Finalize context differs and will not be overwritten: {context_path}")
    else:
        atomic_write_json(context_path, finalize_context)
    _json_print({"dataset_root": str(Path(output).resolve()), "episode_count": len(results)})
    return 0


def _command_build_splits(arguments: argparse.Namespace) -> int:
    root = Path(arguments.dataset).resolve(strict=True)
    records = load_episode_records(root)
    assignments = SplitAssigner(seed=arguments.seed).assign(records)
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
    _json_print({"split_path": str(split_path), "episode_count": len(assignments), "counts": counts})
    return 0


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

    def unit_for(name: str) -> str:
        lowered = name.lower()
        if any(token in lowered for token in ("position", "vertices", ".points", "centroid", "width")):
            return "m"
        if "angular_velocity" in lowered or "dq" in lowered:
            return "rad/s"
        if "linear_velocity" in lowered or "twist" in lowered:
            return "m/s"
        if "force" in lowered:
            return "N"
        if "torque" in lowered or "tau" in lowered:
            return "N*m"
        if any(token in lowered for token in ("enabled", "active", "flag")):
            return "bool"
        return "unknown_explicitly_unmapped"

    info = DatasetInfo(
        name=arguments.name or root.name,
        description=arguments.description,
        time_base=TimeBase(
            sim_hz=float(config.get("sim_hz", 240)),
            control_hz=float(config.get("control_hz", 60)),
            video_hz=float(config.get("video_hz", 30)),
        ),
        state_features=[
            NamedFeature(name, feature_types[name], unit_for(name), description="Named frame-table state column")
            for name in state_names
        ],
        action_features=[
            NamedFeature(name, feature_types[name], unit_for(name), description="Named frame-table action column")
            for name in action_names
        ],
        generator_version=first.generator_git_commit,
        extras={"generation_config_hash": generation["config_hash"]},
    )
    context_path = root / ".finalize_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.is_file() else {}
    writer.finalize(
        info,
        cameras=context.get("cameras", ()),
        provenance=context.get("provenance", ()),
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
    generate.add_argument("--physics-sweep", choices=("gravity", "friction", "restitution", "material"))
    generate.add_argument("--output")
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--renderer", help="native renderer callable as module:function")
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

    splits = subparsers.add_parser("build-splits", help="create deterministic 90/5/5 splits")
    splits.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    splits.add_argument("--seed", type=int, default=0)
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
