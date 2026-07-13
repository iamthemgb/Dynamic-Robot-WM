from __future__ import annotations

import argparse
import time
from collections.abc import Callable
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
        "--demo-mode",
        type=str,
        default="ballistic_intercept",
        choices=("ballistic_intercept", "rolling_island_catch"),
        help="Rollout behavior mode.",
    )
    parser.add_argument(
        "--success-retry-limit",
        type=int,
        default=1,
        help="Retry with new seeds until success or this many attempts are exhausted.",
    )
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
    camera_names: list[str] | None = None,
    video_filenames: dict[str, str] | None = None,
    model_hook: Callable[[mujoco.MjModel], None] | None = None,
) -> dict:
    camera_names = list(camera_names) if camera_names is not None else ["main_camera", "side_camera"]
    video_filenames = dict(video_filenames) if video_filenames is not None else {
        "main_camera": "main.mp4",
        "side_camera": "side.mp4",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    view_video_paths = {name: output_dir / video_filenames[name] for name in camera_names}
    metadata_path = output_dir / "metadata.json"
    bundle = build_episode(sample)
    if model_hook is not None:
        model_hook(bundle.model)
    controller = MujocoInterceptionController(bundle.model, bundle.data, sample)
    controller.prepare()
    renderer = MultiEpisodeRenderer(
        bundle.model,
        width=width,
        height=height,
        fps=fps,
        camera_names=camera_names,
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
        renderer_used=",".join(camera_names),
        frame_counts=frame_counts,
        controller_result=controller_result,
        view_video_paths=view_video_paths,
        wrist_rgb_video_path=None,
        wrist_depth_path=None,
        glb_path=None,
        closeup_video_path=view_video_paths.get("closeup_camera"),
        camera_metadata=bundle.camera_pose,
        elapsed_wall_time_s=elapsed,
        video_error=None,
        wrist_rgb_video_error="not_rendered",
        wrist_depth_error="not_rendered",
        closeup_error=None if "closeup_camera" in view_video_paths else "not_rendered",
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

    if args.success_retry_limit <= 0:
        raise ValueError("--success-retry-limit must be positive.")

    metadata = None
    for attempt_index in range(args.success_retry_limit):
        attempt_seed = int(args.seed + 9973 * attempt_index)
        rng = np.random.default_rng(attempt_seed)
        sample = sample_episode(
            rng,
            "robocasa_kitchen",
            attempt_seed,
            fps=args.fps,
            duration=args.duration,
            layout_id_override=args.layout_id,
            style_id_override=args.style_id,
            camera_variant=args.camera_variant,
            demo_mode=args.demo_mode,
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
        metadata["attempt_index"] = attempt_index
        metadata["attempt_seed"] = attempt_seed
        write_metadata(args.output_dir / "metadata.json", metadata)
        if metadata["success"]:
            break
    if metadata is None:
        raise RuntimeError("No rollout attempts were executed.")

    print(args.output_dir / "main.mp4")
    print(args.output_dir / "side.mp4")
    print(args.output_dir / "metadata.json")
    print(metadata["robocasa_environment_index"])
    print(metadata["robocasa_layout_id"], metadata["robocasa_style_id"])
    print(metadata["scene_resolution_backend"])
    print(metadata["success"])


if __name__ == "__main__":
    main()
