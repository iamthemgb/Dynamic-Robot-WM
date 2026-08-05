from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .controller import MujocoInterceptionController
from .metadata import build_metadata, write_metadata
from .renderer import MultiEpisodeRenderer
from .scene_builder import build_episode, sample_episode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a seeded MuJoCo projectile-catch rollout in a RoboCasa train scene.")
    parser.add_argument("--seed", type=int, required=True, help="Environment seed.")
    parser.add_argument("--layout-id", type=int, help="Override the RoboCasa kitchen layout ID.")
    parser.add_argument("--style-id", type=int, help="Override the RoboCasa kitchen style ID.")
    parser.add_argument("--camera-variant", type=str, help="Optional camera variant override.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where video and metadata will be written.",
    )
    parser.add_argument("--width", type=int, default=832, help="Video width.")
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--fps", type=int, default=30, help="Rendered video FPS.")
    parser.add_argument("--duration", type=float, default=2.5, help="Episode duration in seconds.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a faster smoke-test configuration: 320x240, 10 FPS, 1.0 second.",
    )
    return parser.parse_args()


def run_episode(
    *,
    output_dir: Path,
    sample,
    width: int,
    height: int,
    fps: int,
    episode_index: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    view_video_paths = {
        "main_camera": output_dir / "main.mp4",
        "side_camera": output_dir / "side.mp4",
    }
    metadata_path = output_dir / "metadata.json"
    bundle = build_episode(sample)
    controller = MujocoInterceptionController(bundle.model, bundle.data, sample)
    controller.prepare()
    renderer = MultiEpisodeRenderer(
        bundle.model,
        width=width,
        height=height,
        fps=fps,
        camera_names=["main_camera", "side_camera"],
    )

    frame_every = max(1, round((1.0 / sample.timestep) / sample.fps))
    num_steps = int(round(sample.duration_s / sample.timestep))

    start = time.time()
    renderer.add_frame(bundle.data)
    for step in range(num_steps):
        sim_t = step * sample.timestep
        controller.before_step(sim_t)
        mujoco.mj_step(bundle.model, bundle.data)
        controller.after_step(sim_t + sample.timestep)
        if (step + 1) % frame_every == 0:
            renderer.add_frame(bundle.data)
    frame_counts = renderer.close(view_video_paths)
    elapsed = time.time() - start

    controller_result = controller.result()
    metadata = build_metadata(
        episode_index=episode_index,
        sample=sample,
        asset_info=bundle.asset_info,
        renderer_requested="main_camera",
        renderer_used="main_camera,side_camera",
        frame_counts=frame_counts,
        controller_result=controller_result,
        view_video_paths=view_video_paths,
        wrist_rgb_video_path=None,
        wrist_depth_path=None,
        glb_path=None,
        closeup_video_path=None,
        camera_metadata=bundle.camera_pose,
        elapsed_wall_time_s=elapsed,
        video_error=None,
        wrist_rgb_video_error="not_rendered",
        wrist_depth_error="not_rendered",
        closeup_error="not_rendered",
    )
    write_metadata(metadata_path, metadata)
    return metadata


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.width = 320
        args.height = 240
        args.fps = 10
        args.duration = 1.0

    rng = np.random.default_rng(args.seed)
    sample = sample_episode(
        rng,
        "robocasa_kitchen",
        args.seed,
        fps=args.fps,
        duration=args.duration,
        layout_id_override=args.layout_id,
        style_id_override=args.style_id,
        camera_variant=args.camera_variant,
    )
    sample = replace(sample, offscreen_width=args.width, offscreen_height=args.height)
    metadata = run_episode(
        output_dir=args.output_dir,
        sample=sample,
        width=args.width,
        height=args.height,
        fps=args.fps,
        episode_index=0,
    )

    print(args.output_dir / "main.mp4")
    print(args.output_dir / "side.mp4")
    print(args.output_dir / "metadata.json")
    print(metadata["robocasa_environment_index"])
    print(metadata["robocasa_layout_id"], metadata["robocasa_style_id"])
    print(metadata["scene_resolution_backend"])
    print(metadata["outcomes"]["catch_success"])


if __name__ == "__main__":
    main()
