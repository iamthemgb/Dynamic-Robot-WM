from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .run_rollout import run_episode
from .scene_builder import sample_episode


TARGET_SCENES = (
    (38, 42),
    (48, 41),
    (51, 34),
)
DEFAULT_OUTPUT_ROOT = Path("/gpfs/radev/scratch/sous/mzl7/robocasa_empty_center_layout_demos_2026-07-03")
CAMERA_VARIANT = "arm_relative_island_demo"
DEMO_MODE = "rolling_island_catch"
SUCCESS_RETRY_LIMIT = 60
WIDTH = 960
HEIGHT = 540
FPS = 30
DURATION = 2.5
BASE_SEED = 38042

# The rolling-intercept controller tracks the ball continuously with the
# fingers hovering only millimeters above it, using a fixed default close
# lead time (see MujocoInterceptionController._prepare_rolling_surface_intercept).
# For these island demo scenes, the robot base placement puts the scripted
# catch point near the edge of joint4's range: as the arm nears full
# extension it can no longer track the ball, the fingertip lags behind, and
# the still-open gripper visibly clips/collides with the ball well before
# the scheduled grasp. Closing earlier -- while the arm still has joint
# headroom and tracks tightly -- grasps the ball cleanly instead. Each scene
# needs its own lead since how much reach margin is available depends on
# that scene's robot-to-catch-point geometry.
CLOSE_LEAD_TIME_OFFSET_OVERRIDES_S = {
    (38, 42): -0.28,
    (48, 41): -0.30,
    (51, 34): -0.20,
}

# Absolute sim time at which the arm starts reacting/tracking the ball (see
# EpisodeSample.arm_reaction_delay_s / before_step's parked phase); the ball
# rolls on its own, untracked, from release_time_s (0.3s) until then. Purely
# a staging choice for these demo videos -- later values leave less time for
# the arm to converge before the close deadline above, and for (38, 42) that
# deadline is tight (see the joint-limit note), so 0.5s is close to the
# latest reaction time that still lets tracking converge before the
# scheduled grasp.
ARM_REACTION_DELAY_OVERRIDES_S = {
    (38, 42): 0.50,
    (48, 41): 0.75,
    (51, 34): 0.65,
}

# The wrist-mounted "wrist_rgb" camera (see scene_builder.py) sits only
# ~10cm from the grasped ball. MuJoCo's near clip plane is znear * stat.extent
# -- for this kitchen-scale scene extent is ~15.6, so the default znear=0.03
# clips everything closer than ~47cm, silently discarding the ball at every
# camera angle regardless of aim (confirmed by aiming dead-on, sub-mm
# precision, at 170deg FOV, and still getting zero ball pixels; dropping
# znear alone made it render immediately). Shrink it just for this render so
# the wrist view isn't clipped, without touching the shared model default
# used by main/closeup cameras and the wider dataset pipeline.
SECOND_CAMERA_NAME = "wrist_rgb"
SECOND_CAMERA_FILENAME = "wrist.mp4"
WRIST_CAMERA_ZNEAR = 0.0005


def _shrink_znear_for_wrist_camera(model) -> None:
    model.vis.map.znear = WRIST_CAMERA_ZNEAR


def render_successful_scene(
    *,
    output_dir: Path,
    layout_id: int,
    style_id: int,
    base_seed: int,
    retry_limit: int,
) -> dict:
    last_metadata = None
    for attempt_index in range(retry_limit):
        attempt_seed = int(base_seed + 1009 * attempt_index)
        rng = np.random.default_rng(attempt_seed)
        sample = sample_episode(
            rng,
            "robocasa_kitchen",
            attempt_seed,
            fps=FPS,
            duration=DURATION,
            layout_id_override=layout_id,
            style_id_override=style_id,
            camera_variant=CAMERA_VARIANT,
            demo_mode=DEMO_MODE,
        )
        close_lead_time_offset_s = CLOSE_LEAD_TIME_OFFSET_OVERRIDES_S.get((layout_id, style_id), 0.0)
        arm_reaction_delay_s = ARM_REACTION_DELAY_OVERRIDES_S.get((layout_id, style_id), 0.0)
        sample = replace(
            sample,
            offscreen_width=WIDTH,
            offscreen_height=HEIGHT,
            close_lead_time_offset_s=close_lead_time_offset_s,
            arm_reaction_delay_s=arm_reaction_delay_s,
        )
        metadata = run_episode(
            output_dir=output_dir,
            sample=sample,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            episode_index=0,
            camera_names=["main_camera", SECOND_CAMERA_NAME],
            video_filenames={"main_camera": "main.mp4", SECOND_CAMERA_NAME: SECOND_CAMERA_FILENAME},
            model_hook=_shrink_znear_for_wrist_camera,
        )
        metadata["attempt_index"] = attempt_index
        metadata["attempt_seed"] = attempt_seed
        last_metadata = metadata
        if metadata["success"]:
            break
    if last_metadata is None or not last_metadata["success"]:
        raise RuntimeError(
            f"Failed to generate a successful catch for layout {layout_id}, style {style_id} "
            f"within {retry_limit} attempts."
        )
    return last_metadata


def main() -> None:
    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generation_mode": DEMO_MODE,
        "camera_variant": CAMERA_VARIANT,
        "output_root": str(output_root),
        "videos": [],
    }
    for scene_index, (layout_id, style_id) in enumerate(TARGET_SCENES):
        scene_dir = output_root / f"layout{layout_id:02d}_style{style_id:02d}"
        metadata = render_successful_scene(
            output_dir=scene_dir,
            layout_id=layout_id,
            style_id=style_id,
            base_seed=BASE_SEED + 100_000 * scene_index,
            retry_limit=SUCCESS_RETRY_LIMIT,
        )
        manifest["videos"].append(
            {
                "layout_id": layout_id,
                "style_id": style_id,
                "success": metadata["success"],
                "attempt_index": metadata["attempt_index"],
                "attempt_seed": metadata["attempt_seed"],
                "main_video_path": metadata["output_video_path"],
                "wrist_video_path": metadata["output_view_video_paths"][SECOND_CAMERA_NAME],
                "metadata_path": str(scene_dir / "metadata.json"),
            }
        )
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
