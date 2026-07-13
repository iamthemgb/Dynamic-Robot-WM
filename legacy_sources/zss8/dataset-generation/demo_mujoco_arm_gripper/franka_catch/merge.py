"""Merge per-shard LeRobot-v3-style outputs into a single dataset.

Each shard wrote ``<root>/shardNNN/{data,videos,meta}`` using GLOBALLY unique
episode indices, so merging is a union (no renumbering). Files are moved into
``<root>/{data,videos,meta}`` and the meta tables are rebuilt.

Usage:
    python -m franka_catch.merge --root <dataset_root> --shards N
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .lerobot_export import VIEW_KEYS, LeRobotWriter


def merge(root: Path, shards: int) -> dict:
    root = Path(root)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "rich").mkdir(parents=True, exist_ok=True)
    for key in VIEW_KEYS:
        (root / "videos" / key / "chunk-000").mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    info_template = None
    for s in range(shards):
        sd = root / f"shard{s:03d}"
        if not sd.exists():
            print(f"[merge] shard {s} missing: {sd}")
            continue
        ep_file = sd / "meta" / "episodes.jsonl"
        if ep_file.exists():
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    episodes.append(json.loads(line))
        info_path = sd / "meta" / "info.json"
        if info_path.exists() and info_template is None:
            info_template = json.loads(info_path.read_text(encoding="utf-8"))
        # move files
        for pq in (sd / "data" / "chunk-000").glob("*.parquet"):
            shutil.move(str(pq), str(root / "data" / "chunk-000" / pq.name))
        for key in VIEW_KEYS:
            src = sd / "videos" / key / "chunk-000"
            if src.exists():
                for mp4 in src.glob("*.mp4"):
                    shutil.move(str(mp4), str(root / "videos" / key / "chunk-000" / mp4.name))
        rich = sd / "meta" / "rich"
        if rich.exists():
            for j in rich.glob("*.json"):
                shutil.move(str(j), str(root / "meta" / "rich" / j.name))

    episodes.sort(key=lambda e: e["episode_index"])
    n_ep = len(episodes)
    total_frames = int(sum(e.get("length", 0) for e in episodes))
    if info_template is None:
        info_template = {}
    info_template.update({
        "total_episodes": n_ep,
        "total_frames": total_frames,
        "total_videos": n_ep * len(VIEW_KEYS),
        "splits": {"train": f"0:{n_ep}"},
    })
    (root / "meta" / "info.json").write_text(json.dumps(info_template, indent=2), encoding="utf-8")
    with (root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e) + "\n")
    with (root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"task_index": 0, "task": "Catch the falling ball with the Franka gripper."}) + "\n")

    summary = {"total_episodes": n_ep, "total_frames": total_frames,
               "by_branch": {}, "by_outcome": {}, "by_variant": {}, "by_failure_mode": {}}
    for e in episodes:
        for k, field in (("by_branch", "branch"), ("by_outcome", "outcome"),
                         ("by_variant", "scene_variant"), ("by_failure_mode", "failure_mode")):
            v = e.get(field, "?")
            summary[k][v] = summary[k].get(v, 0) + 1
    (root / "meta" / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # clean up now-empty shard dirs
    for s in range(shards):
        sd = root / f"shard{s:03d}"
        if sd.exists():
            shutil.rmtree(sd, ignore_errors=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--shards", type=int, required=True)
    args = p.parse_args()
    summary = merge(args.root, args.shards)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
