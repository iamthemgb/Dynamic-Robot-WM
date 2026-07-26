"""Aggregate per-episode JSON sidecars into a single dataset manifest.

Usage:
    python -m franka_catch.summarize --dir outputs/dataset
Writes ``dataset_manifest.json`` (every episode + global stats) next to the data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(data_dir: Path) -> dict:
    episodes = []
    for jp in sorted(data_dir.glob("episode_*.json")):
        try:
            md = json.loads(jp.read_text())
        except Exception:
            continue
        episodes.append(
            {
                "json": jp.name,
                "video": Path(md.get("output_video_path", "")).name or None,
                "glb": Path(md["output_glb_path"]).name if md.get("output_glb_path") else None,
                "wrist_rgb": Path(md["output_wrist_rgb_video_path"]).name if md.get("output_wrist_rgb_video_path") else None,
                "wrist_depth": Path(md["output_wrist_depth_path"]).name if md.get("output_wrist_depth_path") else None,
                "closeup": Path(md["output_closeup_video_path"]).name if md.get("output_closeup_video_path") else None,
                "episode_index": md.get("episode_index"),
                "seed": md.get("seed"),
                "scene_variant": md.get("scene_variant"),
                "difficulty": md.get("difficulty"),
                "randomized": md.get("randomized"),
                "success": md.get("success"),
                "frame_count": md.get("frame_count"),
                "fps": md.get("fps"),
                "ball_initial_position": md.get("ball_initial_position"),
                "ball_initial_velocity": md.get("ball_initial_velocity"),
                "first_contact_time_s": md.get("contact_information", {}).get("first_contact_time_s"),
            }
        )

    by_variant: dict[str, Counter] = defaultdict(Counter)
    by_difficulty: dict[str, Counter] = defaultdict(Counter)
    n_success = 0
    for e in episodes:
        by_variant[e["scene_variant"]]["total"] += 1
        by_variant[e["scene_variant"]]["success" if e["success"] else "fail"] += 1
        by_difficulty[str(e["difficulty"])]["total"] += 1
        by_difficulty[str(e["difficulty"])]["success" if e["success"] else "fail"] += 1
        n_success += int(bool(e["success"]))

    manifest = {
        "dataset": "franka_ball_catch_mujoco",
        "num_episodes": len(episodes),
        "num_success": n_success,
        "num_failure": len(episodes) - n_success,
        "success_rate": (n_success / len(episodes)) if episodes else 0.0,
        "per_variant": {k: dict(v) for k, v in sorted(by_variant.items())},
        "per_difficulty": {k: dict(v) for k, v in sorted(by_difficulty.items())},
        "episodes": episodes,
    }
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = summarize(args.dir)
    out = args.dir / "dataset_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[summarize] {manifest['num_episodes']} episodes, "
          f"{manifest['num_success']} success / {manifest['num_failure']} fail "
          f"({manifest['success_rate']:.0%}) -> {out}")
    for v, c in manifest["per_variant"].items():
        print(f"  {v:16s} total={c.get('total',0):3d} success={c.get('success',0):3d} fail={c.get('fail',0):3d}")


if __name__ == "__main__":
    main()
