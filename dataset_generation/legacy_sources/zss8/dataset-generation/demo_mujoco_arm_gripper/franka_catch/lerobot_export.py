"""LeRobotDataset v3-style packaging for the ball-catch dataset.

Layout written under ``root``::

    meta/info.json          feature schema, fps, control_hz, counts
    meta/tasks.jsonl        task table (single task here)
    meta/episodes.jsonl     one rich record per episode (branch/outcome/...)
    data/chunk-000/episode_XXXXXX.parquet     per-frame state/action @ control_hz
    videos/observation.images.main/chunk-000/episode_XXXXXX.mp4   (832x480, 30 fps)
    videos/observation.images.side/chunk-000/episode_XXXXXX.mp4

Video is 30 fps (two synchronized oblique views); the parquet proprio stream runs
at ``control_hz`` (60-120 Hz) with a ``video_frame_index`` column mapping each
control row to its video frame. This is a pragmatic v3-*style* dataset: it keeps
the meta/data/videos split and per-frame parquet, while honoring the separate
30 fps video / 60-120 Hz control-rate requirement.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Two synchronized oblique third-person views. (A wrist eye-in-hand camera exists
# in the scene and the writer is N-view ready, but the current wrist placement
# frames mostly gripper/void, so it is left out until its pose is tuned.)
VIEW_KEYS = ("observation.images.main", "observation.images.side")
CHUNK = "chunk-000"


class LeRobotWriter:
    def __init__(self, root: Path, *, fps: int, control_hz: int, state_dim: int, action_dim: int,
                 resolution=(832, 480), task: str = "Catch the falling ball with the Franka gripper."):
        self.root = Path(root)
        self.fps = fps
        self.control_hz = control_hz
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.width, self.height = resolution
        self.task = task
        self.episodes: list[dict] = []
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / CHUNK).mkdir(parents=True, exist_ok=True)
        for key in VIEW_KEYS:
            (self.root / "videos" / key / CHUNK).mkdir(parents=True, exist_ok=True)

    def _write_video(self, path: Path, frames: list[np.ndarray]) -> None:
        writer = imageio.get_writer(
            str(path), fps=self.fps, codec="libx264", quality=8,
            macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        try:
            for f in frames:
                if f.dtype != np.uint8:
                    f = np.clip(f, 0, 255).astype(np.uint8)
                writer.append_data(f)
        finally:
            writer.close()

    def add_episode(self, episode_index: int, *, frames_by_view: dict,
                    state, action, timestamps, video_frame_index, rich_meta: dict) -> dict:
        tag = f"episode_{episode_index:06d}"
        # --- videos (N synchronized views) ---
        view_paths = {}
        n_video_frames = None
        for key in VIEW_KEYS:
            frames = frames_by_view[key]
            path = self.root / "videos" / key / CHUNK / f"{tag}.mp4"
            self._write_video(path, frames)
            view_paths[key] = str(path.relative_to(self.root))
            if n_video_frames is None:
                n_video_frames = len(frames)

        # --- parquet proprio @ control_hz ---
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        n = state.shape[0]
        done = np.zeros(n, dtype=bool)
        if n:
            done[-1] = True
        table = pa.table({
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "timestamp": pa.array(np.asarray(timestamps, dtype=np.float32)),
            "video_frame_index": pa.array(np.asarray(video_frame_index, dtype=np.int64)),
            "task_index": pa.array(np.zeros(n, dtype=np.int64)),
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), self.state_dim)),
            "action": pa.array(list(action), type=pa.list_(pa.float32(), self.action_dim)),
            "next.done": pa.array(done),
        })
        parquet_path = self.root / "data" / CHUNK / f"{tag}.parquet"
        pq.write_table(table, parquet_path)

        rec = dict(rich_meta)
        rec.update({
            "episode_index": episode_index,
            "length": int(n),
            "video_frames": int(n_video_frames or 0),
            "paths": {**view_paths, "data": str(parquet_path.relative_to(self.root))},
        })
        self.episodes.append(rec)
        return rec

    def finalize(self) -> dict:
        n_ep = len(self.episodes)
        total_frames = int(sum(e["length"] for e in self.episodes))
        info = {
            "codebase_version": "v3.0-style",
            "robot_type": "franka_panda",
            "total_episodes": n_ep,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": n_ep * len(VIEW_KEYS),
            "chunks_size": 1000,
            "fps": self.fps,
            "control_hz": self.control_hz,
            "video_fps": self.fps,
            "splits": {"train": f"0:{n_ep}"},
            "data_path": "data/" + CHUNK + "/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/" + CHUNK + "/episode_{episode_index:06d}.mp4",
            "resolution": [self.width, self.height],
            "features": {
                **{key: {"dtype": "video", "shape": [self.height, self.width, 3],
                         "names": ["height", "width", "channel"],
                         "info": {"video.fps": self.fps, "video.codec": "h264"}}
                   for key in VIEW_KEYS},
                "observation.state": {"dtype": "float32", "shape": [self.state_dim],
                                      "names": STATE_NAMES},
                "action": {"dtype": "float32", "shape": [self.action_dim], "names": ACTION_NAMES},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "video_frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "next.done": {"dtype": "bool", "shape": [1]},
            },
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        with (self.root / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_index": 0, "task": self.task}) + "\n")
        with (self.root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for e in self.episodes:
                fh.write(json.dumps(e) + "\n")

        # split / outcome summary
        summary = {"total_episodes": n_ep, "total_frames": total_frames, "by_branch": {}, "by_outcome": {}, "by_variant": {}}
        for e in self.episodes:
            for k, field in (("by_branch", "branch"), ("by_outcome", "outcome"), ("by_variant", "scene_variant")):
                v = e.get(field, "?")
                summary[k][v] = summary[k].get(v, 0) + 1
        (self.root / "meta" / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


STATE_NAMES = (
    [f"arm_q{i}" for i in range(1, 8)] + ["finger1", "finger2"]
    + ["ee_x", "ee_y", "ee_z"] + ["ball_x", "ball_y", "ball_z"] + ["ball_vx", "ball_vy", "ball_vz"]
)
ACTION_NAMES = [f"arm_q{i}_cmd" for i in range(1, 8)] + ["gripper_cmd"]
