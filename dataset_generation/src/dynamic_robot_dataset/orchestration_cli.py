"""Thin CLI adapters for the canonical run-orchestration APIs."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .common.run_orchestration import finalize_run, load_run_plan, plan_run, run_shard
from .common.schema import DatasetInfo
from .common.video_writer import VideoSpec


def _read_data(path: str | Path) -> Any:
    source = Path(path).resolve(strict=True)
    text = source.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    return value


def _read_mapping(path: str | Path) -> dict[str, Any]:
    value = _read_data(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a mapping in {path}")
    return dict(value)


def _read_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    value = _read_data(path)
    if isinstance(value, Mapping):
        for key in (
            "rows",
            "episodes",
            "plans",
            "declarations",
            "tasks",
            "cameras",
            "provenance",
            "counterfactual_families",
        ):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"Expected a list of mappings in {path}")
    return [dict(row) for row in value]


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _load_executor(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator:
        raise ValueError("Executor must use module:function syntax")
    executor = getattr(importlib.import_module(module_name), attribute)
    if not callable(executor):
        raise TypeError(f"Executor is not callable: {specification}")
    return executor


def _owned_shard_executor(dataset_root: str | Path) -> Callable[..., Any]:
    """Resolve an owned executor from the immutable run configuration.

    Importing a user-supplied ``module:function`` is arbitrary code execution,
    so the normal path only dispatches a backend that this package owns and
    whose identity is bound into both the generation marker and run plan.
    """

    root = Path(dataset_root).resolve(strict=True)
    plan = load_run_plan(root)
    generation = _read_mapping(root / ".generation.json")
    resolved_config = generation.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise ValueError("Generation marker lacks its resolved configuration")
    backend = str(resolved_config.get("backend") or "").strip().lower().replace("-", "_")
    declaration_backends = {
        str(entry.declaration.get("backend") or "").strip().lower().replace("-", "_")
        for entry in plan.episodes
    }
    declaration_schemas = {
        str(entry.declaration.get("schema_version") or "")
        for entry in plan.episodes
    }
    if backend == "source_mujoco" and declaration_backends == {"source_mujoco"}:
        from .common.scale_execution import SCALE_EXECUTION_BRIDGE_SCHEMA
        from .common.source_execution import SOURCE_EXECUTION_BRIDGE_SCHEMA

        if declaration_schemas == {SCALE_EXECUTION_BRIDGE_SCHEMA}:
            from .common.scale_execution import (
                execute_source_mujoco_scale_episode,
            )

            return execute_source_mujoco_scale_episode
        if declaration_schemas != {SOURCE_EXECUTION_BRIDGE_SCHEMA}:
            raise ValueError(
                "run plan mixes or lacks owned declaration schemas: "
                f"{sorted(declaration_schemas)!r}"
            )
        from .common.source_execution import execute_source_mujoco_episode

        return execute_source_mujoco_episode
    raise ValueError(
        "No owned run-shard executor is registered for immutable backend "
        f"{backend!r} with declaration backends {sorted(declaration_backends)!r}; "
        "use --executor together with --unsafe-executor only for diagnostics"
    )


def _resolve_shard_executor(arguments: argparse.Namespace) -> Callable[..., Any]:
    executor_spec = getattr(arguments, "executor", None)
    unsafe = bool(getattr(arguments, "unsafe_executor", False))
    if executor_spec is not None:
        if not unsafe:
            raise ValueError(
                "Arbitrary module:function execution requires the explicit "
                "--unsafe-executor opt-in"
            )
        return _load_executor(executor_spec)
    if unsafe:
        raise ValueError("--unsafe-executor requires --executor module:function")
    return _owned_shard_executor(arguments.dataset)


def command_plan_run(arguments: argparse.Namespace) -> int:
    config = _read_mapping(arguments.config)
    declarations = _read_rows(arguments.episodes)
    split_settings = {
        "strategy": "leakage_aware_stratified",
        "seed": arguments.split_seed,
        "train_fraction": arguments.train_fraction,
        "validation_fraction": arguments.validation_fraction,
        "test_fraction": arguments.test_fraction,
    }
    plan = plan_run(
        arguments.output,
        config,
        declarations,
        shard_count=arguments.shards,
        resume=arguments.resume,
        chunk_size=arguments.chunk_size,
        video_spec=VideoSpec(
            width=arguments.width,
            height=arguments.height,
            fps_num=arguments.fps_num,
            fps_den=arguments.fps_den,
            codec=arguments.codec,
            pixel_format=arguments.pixel_format,
            crf=arguments.crf,
            preset=arguments.preset,
        ),
        split_settings=split_settings,
    )
    _print_json(
        {
            "dataset_root": str(Path(arguments.output).resolve()),
            "run_id": plan.run_id,
            "plan_hash": plan.plan_hash,
            "episode_count": len(plan.episodes),
            "shard_count": plan.shard_count,
        }
    )
    return 0


def command_run_shard(arguments: argparse.Namespace) -> int:
    result = run_shard(
        arguments.dataset,
        arguments.shard_id,
        _resolve_shard_executor(arguments),
        max_episodes=arguments.max_episodes,
    )
    _print_json(result.to_dict())
    return 0 if result.failed_count == 0 else 1


def command_finalize_run(arguments: argparse.Namespace) -> int:
    info = DatasetInfo.from_dict(_read_mapping(arguments.info))
    records = finalize_run(
        arguments.dataset,
        info,
        tasks=None if arguments.tasks is None else _read_rows(arguments.tasks),
        cameras=_read_rows(arguments.cameras),
        provenance=_read_rows(arguments.provenance),
        counterfactual_families=(
            None
            if arguments.counterfactual_families is None
            else _read_rows(arguments.counterfactual_families)
        ),
    )
    _print_json(
        {
            "dataset_root": str(Path(arguments.dataset).resolve()),
            "episode_count": len(records),
            "finalized": True,
        }
    )
    return 0


def add_orchestration_subcommands(subparsers: Any) -> None:
    """Register plan-run, run-shard, and finalize-run on the main parser."""

    plan = subparsers.add_parser("plan-run", help="write an immutable episode/shard ledger")
    plan.add_argument("--config", required=True, help="resolved YAML/JSON run configuration")
    plan.add_argument("--episodes", required=True, help="JSON/YAML episode declaration list")
    plan.add_argument("--output", required=True)
    plan.add_argument("--shards", type=int, required=True)
    plan.add_argument("--resume", action="store_true")
    plan.add_argument("--chunk-size", type=int, default=1000)
    plan.add_argument("--width", type=int, default=832)
    plan.add_argument("--height", type=int, default=480)
    plan.add_argument("--fps-num", type=int, default=30)
    plan.add_argument("--fps-den", type=int, default=1)
    plan.add_argument("--codec", default="libx264")
    plan.add_argument("--pixel-format", default="yuv420p")
    plan.add_argument("--crf", type=int, default=18)
    plan.add_argument("--preset", default="medium")
    plan.add_argument("--split-seed", type=int, default=0)
    plan.add_argument("--train-fraction", type=float, default=0.80)
    plan.add_argument("--validation-fraction", type=float, default=0.10)
    plan.add_argument("--test-fraction", type=float, default=0.10)
    plan.set_defaults(handler=command_plan_run)

    shard = subparsers.add_parser("run-shard", help="execute one deterministic run partition")
    shard.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    shard.add_argument("--shard-id", type=int, required=True)
    shard.add_argument(
        "--executor",
        help="diagnostic-only module:function returning EpisodeMaterialization",
    )
    shard.add_argument(
        "--unsafe-executor",
        action="store_true",
        help="explicitly allow importing the arbitrary callable passed to --executor",
    )
    shard.add_argument("--max-episodes", type=int)
    shard.set_defaults(handler=command_run_shard)

    finalize = subparsers.add_parser(
        "finalize-run", help="seal and finalize an exactly complete immutable run plan"
    )
    finalize.add_argument("--dataset", "--dataset-root", dest="dataset", required=True)
    finalize.add_argument("--info", required=True, help="canonical DatasetInfo JSON/YAML")
    finalize.add_argument("--tasks")
    finalize.add_argument("--cameras")
    finalize.add_argument("--provenance")
    finalize.add_argument("--counterfactual-families")
    finalize.set_defaults(handler=command_finalize_run)
