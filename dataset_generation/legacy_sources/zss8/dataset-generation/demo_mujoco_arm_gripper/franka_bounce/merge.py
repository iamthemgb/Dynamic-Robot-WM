"""Merge per-shard F1_B outputs. Thin wrapper over franka_catch.merge that
fixes up the task label for the bounce subfamily.

Usage:
    python -m franka_bounce.merge --root <dataset_root> --shards N
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from franka_catch.merge import merge as _merge

from .generate import TASK


def merge(root: Path, shards: int) -> dict:
    summary = _merge(root, shards)
    root = Path(root)
    with (root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--shards", type=int, required=True)
    args = p.parse_args()
    print(json.dumps(merge(args.root, args.shards), indent=2))


if __name__ == "__main__":
    main()
