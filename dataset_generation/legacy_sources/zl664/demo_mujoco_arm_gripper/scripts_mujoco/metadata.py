from __future__ import annotations

import json
from pathlib import Path


def build_metadata(
    *,
    episode_index: int,
    sample,
    asset_info,
    renderer_requested: str,
    renderer_used: str,
    frame_count: int,
    controller_result: dict,
    video_path: Path,
    wrist_rgb_video_path: Path | None,
    wrist_depth_path: Path | None,
    glb_path: Path | None,
    closeup_video_path: Path | None,
    camera_metadata: dict | None,
    elapsed_wall_time_s: float,
    video_error: str | None,
    wrist_rgb_video_error: str | None,
    wrist_depth_error: str | None,
    closeup_error: str | None,
) -> dict:
    return {
        "episode_index": episode_index,
        "seed": sample.seed,
        "simulator": "MuJoCo",
        "mujoco_asset_source": asset_info.source,
        "mujoco_asset_path": str(asset_info.robot_xml),
        "renderer_requested": renderer_requested,
        "renderer_used": renderer_used,
        "scene_variant": sample.scene_variant,
        "gravity": list(sample.gravity),
        "timestep": sample.timestep,
        "fps": sample.fps,
        "frame_count": frame_count,
        "robot_base_position": list(sample.robot_base_position),
        "robot_base_euler": list(sample.robot_base_euler),
        "tabletop_height": sample.tabletop_height,
        "ball_radius": sample.ball_radius,
        "ball_mass": sample.ball_mass,
        "ball_color": list(sample.ball_color),
        "ball_initial_position": list(sample.ball_initial_position),
        "ball_initial_velocity": list(sample.ball_initial_velocity),
        "projectile": getattr(sample, "projectile", None),
        "intercept_position": controller_result["intercept_position"],
        "predicted_close_time_s": controller_result["predicted_close_time_s"],
        "gripper_close_time": controller_result["gripper_close_time"],
        "success": controller_result["success"],
        "final_ball_position": controller_result["final_ball_position"],
        "contact_information": {
            "first_contact_time_s": controller_result["first_contact_time_s"],
            "contact_frames": controller_result["contact_frames"],
            "max_retained_time_s": controller_result["max_retained_time_s"],
        },
        "cameras": camera_metadata or {},
        "output_video_path": str(video_path),
        "output_external_video_path": str(video_path),
        "output_wrist_rgb_video_path": str(wrist_rgb_video_path) if wrist_rgb_video_path else None,
        "output_wrist_depth_path": str(wrist_depth_path) if wrist_depth_path else None,
        "output_closeup_video_path": str(closeup_video_path) if closeup_video_path else None,
        "output_glb_path": str(glb_path) if glb_path else None,
        "elapsed_wall_time_s": elapsed_wall_time_s,
        "video_error": video_error,
        "external_video_error": video_error,
        "wrist_rgb_video_error": wrist_rgb_video_error,
        "wrist_depth_error": wrist_depth_error,
        "closeup_error": closeup_error,
        "controller_result": controller_result,
    }


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
