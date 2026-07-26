from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VIEW_KEYS = ("observation.images.main", "observation.images.side")
CHUNK = "chunk-000"

STATE_NAMES = (
    [f"arm_q{i}" for i in range(1, 8)]
    + [
        "rq_right_driver",
        "rq_right_coupler",
        "rq_right_spring_link",
        "rq_right_follower",
        "rq_left_driver",
        "rq_left_coupler",
        "rq_left_spring_link",
        "rq_left_follower",
        "grasp_x",
        "grasp_y",
        "grasp_z",
        "ball_x",
        "ball_y",
        "ball_z",
        "ball_vx",
        "ball_vy",
        "ball_vz",
    ]
)
ACTION_NAMES = (
    [f"arm_q{i}_cmd" for i in range(1, 8)]
    + [
        "rq_right_driver_cmd",
        "rq_right_coupler_cmd",
        "rq_right_spring_link_cmd",
        "rq_right_follower_cmd",
        "rq_left_driver_cmd",
        "rq_left_coupler_cmd",
        "rq_left_spring_link_cmd",
        "rq_left_follower_cmd",
    ]
)


class LeRobotWriter:
    def __init__(
        self,
        root: Path,
        *,
        fps: int,
        control_hz: int,
        resolution=(832, 480),
        tasks: dict[int, str],
    ):
        self.root = Path(root)
        self.fps = int(fps)
        self.control_hz = int(control_hz)
        self.width, self.height = resolution
        self.tasks = dict(sorted((int(k), str(v)) for k, v in tasks.items()))
        self.episodes: list[dict] = []
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / CHUNK).mkdir(parents=True, exist_ok=True)
        for key in VIEW_KEYS:
            (self.root / "videos" / key / CHUNK).mkdir(parents=True, exist_ok=True)

    def _write_video(self, path: Path, frames: list[np.ndarray]) -> None:
        writer = imageio.get_writer(
            str(path),
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        try:
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                writer.append_data(frame)
        finally:
            writer.close()

    def add_episode(
        self,
        episode_index: int,
        *,
        task_index: int,
        main_frames,
        side_frames,
        state,
        action,
        timestamps,
        video_frame_index,
        rich_meta: dict,
    ) -> dict:
        tag = f"episode_{episode_index:06d}"
        main_path = self.root / "videos" / VIEW_KEYS[0] / CHUNK / f"{tag}.mp4"
        side_path = self.root / "videos" / VIEW_KEYS[1] / CHUNK / f"{tag}.mp4"
        self._write_video(main_path, main_frames)
        self._write_video(side_path, side_frames)

        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        n = int(state.shape[0])
        done = np.zeros(n, dtype=bool)
        if n:
            done[-1] = True
        table = pa.table(
            {
                "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
                "frame_index": pa.array(np.arange(n, dtype=np.int64)),
                "timestamp": pa.array(np.asarray(timestamps, dtype=np.float32)),
                "video_frame_index": pa.array(np.asarray(video_frame_index, dtype=np.int64)),
                "task_index": pa.array(np.full(n, int(task_index), dtype=np.int64)),
                "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), len(STATE_NAMES))),
                "action": pa.array(list(action), type=pa.list_(pa.float32(), len(ACTION_NAMES))),
                "next.done": pa.array(done),
            }
        )
        parquet_path = self.root / "data" / CHUNK / f"{tag}.parquet"
        pq.write_table(table, parquet_path)

        rec = dict(rich_meta)
        rec.update(
            {
                "episode_index": int(episode_index),
                "task_index": int(task_index),
                "length": n,
                "video_frames": int(len(main_frames)),
                "paths": {
                    VIEW_KEYS[0]: str(main_path.relative_to(self.root)),
                    VIEW_KEYS[1]: str(side_path.relative_to(self.root)),
                    "data": str(parquet_path.relative_to(self.root)),
                },
            }
        )
        self.episodes.append(rec)
        return rec

    def finalize(self) -> dict:
        n_ep = len(self.episodes)
        total_frames = int(sum(e["length"] for e in self.episodes))
        info = {
            "codebase_version": "v3.0-style",
            "robot_type": "franka_panda_robotiq_2f85_thick_pad",
            "total_episodes": n_ep,
            "total_frames": total_frames,
            "total_tasks": len(self.tasks),
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
                VIEW_KEYS[0]: {
                    "dtype": "video",
                    "shape": [self.height, self.width, 3],
                    "names": ["height", "width", "channel"],
                    "info": {"video.fps": self.fps, "video.codec": "h264"},
                },
                VIEW_KEYS[1]: {
                    "dtype": "video",
                    "shape": [self.height, self.width, 3],
                    "names": ["height", "width", "channel"],
                    "info": {"video.fps": self.fps, "video.codec": "h264"},
                },
                "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": list(STATE_NAMES)},
                "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": list(ACTION_NAMES)},
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
            for task_index, task in self.tasks.items():
                fh.write(json.dumps({"task_index": task_index, "task": task}) + "\n")
        with (self.root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for episode in sorted(self.episodes, key=lambda e: e["episode_index"]):
                fh.write(json.dumps(episode) + "\n")

        summary = summarize_episodes(self.episodes, total_frames=total_frames)
        (self.root / "meta" / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def summarize_episodes(episodes: list[dict], *, total_frames: int | None = None) -> dict:
    summary = {
        "total_episodes": len(episodes),
        "total_frames": int(sum(e.get("length", 0) for e in episodes) if total_frames is None else total_frames),
        "by_subfamily": {},
        "by_branch": {},
        "by_outcome": {},
        "by_variant": {},
        "by_failure_mode": {},
        "by_task_index": {},
        "by_scene_randomization_level": {},
    }
    fields = (
        ("by_subfamily", "subfamily"),
        ("by_branch", "branch"),
        ("by_outcome", "outcome"),
        ("by_variant", "scene_variant"),
        ("by_failure_mode", "failure_mode"),
        ("by_task_index", "task_index"),
        ("by_scene_randomization_level", "scene_randomization_level"),
    )
    for episode in episodes:
        for bucket, field in fields:
            value = str(episode.get(field, "?"))
            summary[bucket][value] = summary[bucket].get(value, 0) + 1
    return summary

