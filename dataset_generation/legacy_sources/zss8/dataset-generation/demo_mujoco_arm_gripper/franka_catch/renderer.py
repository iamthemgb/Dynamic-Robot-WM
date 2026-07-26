from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


class EpisodeRenderer:
    def __init__(self, model: mujoco.MjModel, *, width: int, height: int, fps: int, camera_name: str, save_frames: bool = False, frames_dir: Path | None = None):
        self.model = model
        self.width = width + (width % 2)
        self.height = height + (height % 2)
        self.fps = fps
        self.camera_name = camera_name
        self.save_frames = save_frames
        self.frames_dir = frames_dir
        self.frames: list[np.ndarray] = []
        if save_frames:
            assert frames_dir is not None
            frames_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def add_frame(self, data: mujoco.MjData) -> None:
        self.renderer.update_scene(data, camera=self.camera_name)
        frame = self.renderer.render()
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if self.save_frames and self.frames_dir is not None:
            imageio.imwrite(self.frames_dir / f"frame_{len(self.frames):05d}.png", frame)
        self.frames.append(frame.copy())

    def close(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=self.fps, codec="libx264", quality=8, macro_block_size=1)
        try:
            for frame in self.frames:
                writer.append_data(frame)
        finally:
            writer.close()
            self.renderer.close()


class EpisodeDepthRenderer:
    def __init__(self, model: mujoco.MjModel, *, width: int, height: int, camera_name: str):
        self.model = model
        self.width = width + (width % 2)
        self.height = height + (height % 2)
        self.camera_name = camera_name
        self.frames: list[np.ndarray] = []
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.renderer.enable_depth_rendering()

    def add_frame(self, data: mujoco.MjData) -> None:
        self.renderer.update_scene(data, camera=self.camera_name)
        self.frames.append(self.renderer.render().astype(np.float32, copy=True))

    def close(self, path: Path | None = None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self.frames:
                frames = np.stack(self.frames, axis=0)
            else:
                frames = np.empty((0, self.height, self.width), dtype=np.float32)
            np.save(path, frames)
        self.renderer.close()
