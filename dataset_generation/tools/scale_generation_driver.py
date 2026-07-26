#!/usr/bin/env python3
"""Operator driver for diagnostic sampled-scale block generation.

Subcommands:

- ``plan-hours``: print the campaign table (hours to episodes to blocks).
- ``mint-plan``: mint one block's cases, prepare declarations in isolated
  workers, and write its immutable run plan.
- ``run-shards``: drive one/all shards of a planned block to completion,
  one fresh worker process per episode.
- ``finalize``: seal one completed block and run strict persisted QC.
- ``calibrate``: end-to-end small block (mint, run, finalize) with timing
  and size measurements for SLURM sizing.
- ``report``: aggregate finalized blocks under an output root.

Every artifact is diagnostic and training-ineligible; release-hour
accounting remains structurally zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from dynamic_robot_dataset.common.scale_generation import (
    finalize_scale_block,
    load_block_plan,
    load_hours_plan,
    plan_scale_block,
    run_scale_shard_isolated,
)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def command_plan_hours(arguments: argparse.Namespace) -> int:
    plan = load_hours_plan(arguments.hours_config)
    table = plan.table()
    _print_json(
        {
            "leaves": table,
            "total_hours": sum(row["target_hours"] for row in table),
            "total_episodes": sum(row["episode_target"] for row in table),
            "total_blocks": sum(row["block_count"] for row in table),
            "shards_per_block": plan.shards_per_block,
        }
    )
    return 0


def command_mint_plan(arguments: argparse.Namespace) -> int:
    hours = load_hours_plan(arguments.hours_config)
    blocks = hours.blocks(arguments.leaf)
    selected = [b for b in blocks if b["block_index"] == arguments.block]
    if not selected:
        raise SystemExit(
            f"{arguments.leaf} has no block {arguments.block}; "
            f"valid blocks are 0..{len(blocks) - 1}"
        )
    block = selected[0]
    episode_count = (
        arguments.episodes
        if arguments.episodes is not None
        else block["episode_count"]
    )
    result = plan_scale_block(
        arguments.leaf,
        block_index=block["block_index"],
        episode_start=block["episode_start"],
        episode_count=episode_count,
        output_root=arguments.output_root,
        shard_count=arguments.shards or hours.shards_per_block,
        prepare_workers=arguments.prepare_workers,
        resume=arguments.resume,
    )
    _print_json(result)
    return 0


def command_run_shards(arguments: argparse.Namespace) -> int:
    plan = load_block_plan(arguments.dataset)
    shard_ids = (
        [arguments.shard_id]
        if arguments.shard_id is not None
        else list(range(plan.shard_count))
    )
    results = []
    started = time.monotonic()
    for shard_id in shard_ids:
        results.append(
            run_scale_shard_isolated(
                arguments.dataset,
                shard_id,
                max_episodes_per_process=arguments.max_episodes_per_process,
            )
        )
    elapsed = time.monotonic() - started
    committed = sum(int(r["committed_count"]) for r in results)
    _print_json(
        {
            "dataset_root": str(Path(arguments.dataset).resolve()),
            "shards": results,
            "elapsed_seconds": elapsed,
            "committed_count": committed,
            "seconds_per_committed_episode": (
                elapsed / committed if committed else None
            ),
        }
    )
    return 0


def command_finalize(arguments: argparse.Namespace) -> int:
    result = finalize_scale_block(
        arguments.dataset,
        deep_video_checks=arguments.deep_video_checks,
    )
    _print_json(result)
    return 0


def command_calibrate(arguments: argparse.Namespace) -> int:
    """Small end-to-end block through the identical code path, with timing."""

    planned = plan_scale_block(
        arguments.leaf,
        block_index=arguments.block,
        episode_start=arguments.episode_start,
        episode_count=arguments.episodes,
        output_root=arguments.output_root,
        shard_count=arguments.shards,
        prepare_workers=arguments.prepare_workers,
    )
    root = Path(planned["dataset_root"])
    run_started = time.monotonic()
    shard_results = [
        run_scale_shard_isolated(root, shard_id, max_episodes_per_process=1)
        for shard_id in range(arguments.shards)
    ]
    run_seconds = time.monotonic() - run_started
    finalized = finalize_scale_block(
        root, deep_video_checks=arguments.deep_video_checks
    )
    committed = sum(int(r["committed_count"]) for r in shard_results)
    total_bytes = _tree_bytes(root)
    _print_json(
        {
            "planned": planned,
            "run_seconds": run_seconds,
            "committed_count": committed,
            "run_seconds_per_episode": run_seconds / committed if committed else None,
            "finalized": finalized,
            "dataset_bytes": total_bytes,
            "bytes_per_episode": (
                total_bytes // finalized["episode_count"]
                if finalized["episode_count"]
                else None
            ),
        }
    )
    return 0


def command_report(arguments: argparse.Namespace) -> int:
    root = Path(arguments.output_root).resolve(strict=True)
    rows = []
    total_episodes = 0
    for qc_report in sorted(root.glob("*/block-*/qc/dataset_report.json")):
        block_root = qc_report.parent.parent
        generation = json.loads(
            (block_root / ".generation.json").read_text(encoding="utf-8")
        )
        resolved = generation["resolved_config"]
        report = json.loads(qc_report.read_text(encoding="utf-8"))
        episodes = report.get("episodes", [])
        passed = sum(1 for item in episodes if item.get("passed"))
        rows.append(
            {
                "dataset_root": str(block_root),
                "corpus_leaf_id": resolved["corpus_leaf_id"],
                "block_index": resolved["block_index"],
                "episode_count": resolved["episode_count"],
                "episodes_qc_passed": passed,
                "episodes_qc_failed": len(episodes) - passed,
            }
        )
        total_episodes += int(resolved["episode_count"])
    _print_json(
        {
            "output_root": str(root),
            "finalized_blocks": rows,
            "total_finalized_episodes": total_episodes,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan_hours = commands.add_parser("plan-hours", help="print the campaign table")
    plan_hours.add_argument("--hours-config", default=None)
    plan_hours.set_defaults(handler=command_plan_hours)

    mint = commands.add_parser("mint-plan", help="mint and plan one block")
    mint.add_argument("--leaf", required=True)
    mint.add_argument("--block", type=int, required=True)
    mint.add_argument("--episodes", type=int, default=None)
    mint.add_argument("--output-root", required=True)
    mint.add_argument("--hours-config", default=None)
    mint.add_argument("--shards", type=int, default=None)
    mint.add_argument("--prepare-workers", type=int, default=8)
    mint.add_argument("--resume", action="store_true")
    mint.set_defaults(handler=command_mint_plan)

    run = commands.add_parser("run-shards", help="execute planned shards")
    run.add_argument("--dataset", required=True)
    run.add_argument("--shard-id", type=int, default=None)
    run.add_argument("--max-episodes-per-process", type=int, default=1)
    run.set_defaults(handler=command_run_shards)

    finalize = commands.add_parser("finalize", help="seal and QC one block")
    finalize.add_argument("--dataset", required=True)
    finalize.add_argument("--deep-video-checks", action="store_true")
    finalize.set_defaults(handler=command_finalize)

    calibrate = commands.add_parser(
        "calibrate", help="small end-to-end block with measurements"
    )
    calibrate.add_argument("--leaf", required=True)
    calibrate.add_argument("--episodes", type=int, default=100)
    calibrate.add_argument("--block", type=int, default=9000)
    calibrate.add_argument("--episode-start", type=int, default=9000000)
    calibrate.add_argument("--output-root", required=True)
    calibrate.add_argument("--shards", type=int, default=4)
    calibrate.add_argument("--prepare-workers", type=int, default=8)
    calibrate.add_argument("--deep-video-checks", action="store_true")
    calibrate.set_defaults(handler=command_calibrate)

    report = commands.add_parser("report", help="aggregate finalized blocks")
    report.add_argument("--output-root", required=True)
    report.set_defaults(handler=command_report)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
