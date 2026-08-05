from __future__ import annotations

import json
from pathlib import Path

from .scene_builder import BALL_FRICTION, BALL_SOLIMP, BALL_SOLREF


def build_metadata(
    *,
    episode_index: int,
    sample,
    asset_info,
    renderer_requested: str,
    renderer_used: str,
    frame_counts: dict[str, int],
    controller_result: dict,
    view_video_paths: dict[str, Path],
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
    visual_settings = sample.visual_settings or {}
    dataset_tags = sample.dataset_tags or {}
    frame_count = int(frame_counts.get("main_camera", 0))
    return {
        "episode_index": episode_index,
        "episode_id": dataset_tags.get("episode_id", f"episode_{episode_index:06d}"),
        "family": dataset_tags.get("family"),
        "subfamily": dataset_tags.get("subfamily"),
        "branch": dataset_tags.get("branch"),
        "outcome": dataset_tags.get("outcome"),
        "failure_mode": dataset_tags.get("failure_mode"),
        "group": dataset_tags.get("group"),
        "seed": sample.seed,
        "simulator": "MuJoCo",
        "mujoco_asset_source": asset_info.source,
        "mujoco_asset_path": str(asset_info.robot_xml),
        "renderer_requested": renderer_requested,
        "renderer_used": renderer_used,
        "scene_variant": sample.scene_variant,
        "visual_settings": sample.visual_settings,
        "robocasa_environment_index": visual_settings.get("kitchen_env_index"),
        "robocasa_environment_count": visual_settings.get("kitchen_env_count"),
        "robocasa_layout_id": visual_settings.get("kitchen_layout_id"),
        "robocasa_style_id": visual_settings.get("kitchen_style_id"),
        "scene_resolution_backend": visual_settings.get("scene_resolution_backend"),
        "scene_resolution_warning": visual_settings.get("scene_resolution_warning"),
        "timestep": sample.timestep,
        "fps": sample.fps,
        "frame_count": frame_count,
        "frame_counts": {key: int(value) for key, value in frame_counts.items()},
        "duration_sec": sample.duration_s,
        "robot_base_position": list(sample.robot_base_position),
        "robot_base_euler": list(sample.robot_base_euler),
        "robot_counter_name": visual_settings.get("robot_counter_name"),
        "robot_counter_edge": visual_settings.get("robot_counter_edge"),
        "tabletop_height": sample.tabletop_height,
        "ball_color": list(sample.ball_color),
        "scene_catch_position": visual_settings.get("catch_position_xyz"),
        "scene_launch_position_xy": visual_settings.get("launch_position_xy"),
        "physics_tokens": {
            "gravity_mps2": list(sample.gravity),
            "drag": {
                "model": "none",
                "air_density_kgpm3": 0.0,
                "viscosity_pas": 0.0,
                "wind_mps": [0.0, 0.0, 0.0],
                "note": "MuJoCo fluid forces are disabled; free flight is ballistic and mass-independent.",
            },
            "friction": {
                "ball_sliding_torsional_rolling": list(BALL_FRICTION),
                "combination_rule": "elementwise max with the contacting geom's friction",
            },
            "restitution_compliance": {
                "ball_solref_timeconst_dampratio": list(BALL_SOLREF),
                "ball_solimp": list(BALL_SOLIMP),
                "note": "damping ratio 1.0 gives critically damped contact, effectively zero restitution",
            },
            "ball_mass_kg": sample.ball_mass,
            "ball_mass_note": "included because mass scales contact impulse and penetration; with drag disabled it does not affect free flight",
        },
        "state_context": {
            "frame": "world",
            "ball_initial_position_m": list(sample.ball_initial_position),
            "ball_initial_velocity_mps": list(sample.ball_initial_velocity),
            "release_time_s": sample.release_time_s,
        },
        "geometry_context": {
            "ball_radius_m": sample.ball_radius,
        },
        "action_context": {
            "joint_trajectory": controller_result["joint_trajectory"],
            "gripper_close_command_time_s": controller_result["gripper_close_time"],
            "controller_mode": sample.controller_mode,
        },
        "outcomes": {
            "intercept_time_s": controller_result.get("ballistic_intercept_time_s"),
            "intercept_position_m": controller_result["intercept_position"],
            "impact_speed_mps": controller_result["impact_speed_mps"],
            "first_contact_time_s": controller_result["first_contact_time_s"],
            "total_contact_time_s": controller_result["total_contact_time_s"],
            "catch_success": controller_result["success"],
            "max_penetration_depth_m": controller_result["max_penetration_depth_m"],
            "contact_impulse_until_grasp_ns": controller_result["contact_impulse_until_grasp_ns"],
            "total_contact_impulse_ns": controller_result["total_contact_impulse_ns"],
            "peak_contact_force_n": controller_result["peak_contact_force_n"],
        },
        "events": {
            "release_frame": int(round(sample.release_time_s * sample.fps)),
            "first_contact_frame": None
            if controller_result["first_contact_time_s"] is None
            else int(round(float(controller_result["first_contact_time_s"]) * sample.fps)),
            "entered_tray_frame": None,
            "exit_frame": None,
        },
        "views": ["main_camera", "side_camera"],
        "cameras": camera_metadata or {},
        "output_video_path": str(view_video_paths["main_camera"]),
        "output_external_video_path": str(view_video_paths["main_camera"]),
        "output_side_video_path": str(view_video_paths["side_camera"]),
        "output_view_video_paths": {key: str(value) for key, value in view_video_paths.items()},
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
        "side_video_error": video_error,
        "controller_result": {key: value for key, value in controller_result.items() if key != "joint_trajectory"},
    }


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
