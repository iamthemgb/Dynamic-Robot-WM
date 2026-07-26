from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .lerobot_export import VIEW_KEYS, summarize_episodes
from .taxonomy import SUBFAMILIES


def _move_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))


def merge(root: Path, shards: int) -> dict:
    root = Path(root)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "rich").mkdir(parents=True, exist_ok=True)
    for key in VIEW_KEYS:
        (root / "videos" / key / "chunk-000").mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    info_template: dict | None = None
    for shard in range(shards):
        shard_dir = root / f"shard{shard:03d}"
        if not shard_dir.exists():
            print(f"[merge] missing shard: {shard_dir}")
            continue
        info_path = shard_dir / "meta" / "info.json"
        if info_path.exists() and info_template is None:
            info_template = json.loads(info_path.read_text(encoding="utf-8"))
        episodes_path = shard_dir / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            for line in episodes_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    episodes.append(json.loads(line))

        for parquet_path in (shard_dir / "data" / "chunk-000").glob("*.parquet"):
            _move_replace(parquet_path, root / "data" / "chunk-000" / parquet_path.name)
        for key in VIEW_KEYS:
            for video_path in (shard_dir / "videos" / key / "chunk-000").glob("*.mp4"):
                _move_replace(video_path, root / "videos" / key / "chunk-000" / video_path.name)
        rich_dir = shard_dir / "meta" / "rich"
        if rich_dir.exists():
            for rich_path in rich_dir.glob("*.json"):
                _move_replace(rich_path, root / "meta" / "rich" / rich_path.name)

    episodes.sort(key=lambda e: int(e["episode_index"]))
    total_frames = int(sum(e.get("length", 0) for e in episodes))
    if info_template is None:
        info_template = {}
    info_template.update(
        {
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": len(SUBFAMILIES),
            "total_videos": len(episodes) * len(VIEW_KEYS),
            "splits": {"train": f"0:{len(episodes)}"},
        }
    )
    (root / "meta" / "info.json").write_text(json.dumps(info_template, indent=2), encoding="utf-8")
    with (root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
        for episode in episodes:
            fh.write(json.dumps(episode) + "\n")
    with (root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        for info in sorted(SUBFAMILIES.values(), key=lambda x: int(x["task_index"])):
            fh.write(json.dumps({"task_index": int(info["task_index"]), "task": info["task"]}) + "\n")

    summary = summarize_episodes(episodes, total_frames=total_frames)
    (root / "meta" / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for shard in range(shards):
        shutil.rmtree(root / f"shard{shard:03d}", ignore_errors=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Robotiq catch LeRobot shards.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    summary = merge(args.root, args.shards)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

