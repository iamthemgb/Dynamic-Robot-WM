from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from .canonical_writer import (
    SCHEMA_VERSION,
    FAILURE_TAXONOMY_VERSION,
    VideoSpec,
    canonical_camera_name,
    episode_uuid,
    fixed_duration_frame_timestamps,
    sha256_json,
    write_episode_artifacts,
    write_episode_marker,
)
from .renderer import MultiEpisodeRenderer
from .scene_builder import (
    BALL_FRICTION,
    BALL_SOLIMP,
    BALL_SOLREF,
    build_episode,
    randomize_initial_velocity,
    sample_episode,
)

ACTION_MODE = "passive_no_actuation"
BALL_GEOM_NAME = "catch_ball_geom"
BALL_OBJECT_ID = "catch_ball"
BALL_JOINT_NAME = "ball_freejoint"
DEFAULT_CAMERA_NAMES = ["main_camera", "closeup_camera"]
DEFAULT_CAMERA_STREAMS = {"main_camera": "main", "closeup_camera": "secondary"}


@dataclass
class EpisodeRollout:
    """One persisted passive rollout: synchronized frames plus sidecar tables."""

    sample: object
    frames: dict[str, list[np.ndarray]]
    frame_timestamps: list[float]
    frame_rows: list[dict]
    high_rate_rows: list[dict]
    event_rows: list[dict]
    transition_rows: list[dict]
    object_state_rows: list[dict]
    roll_metrics: dict
    camera_pose: dict
    elapsed_wall_time_s: float
    frame_counts: dict[str, int] = field(default_factory=dict)


def _geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        return name
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
    return body or f"geom_{geom_id}"


def _ball_contact_rows(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ball_geom: int,
    timestamp: float,
) -> list[dict]:
    rows: list[dict] = []
    force = np.zeros(6, dtype=np.float64)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 != ball_geom and contact.geom2 != ball_geom:
            continue
        counterpart = contact.geom2 if contact.geom1 == ball_geom else contact.geom1
        mujoco.mj_contactForce(model, data, index, force)
        rows.append(
            {
                "timestamp": float(timestamp),
                "object_a": BALL_OBJECT_ID,
                "object_b": _geom_label(model, int(counterpart)),
                "point_world_m": [float(v) for v in contact.pos],
                "normal_world": [float(v) for v in contact.frame[:3]],
                "penetration_depth_m": float(max(0.0, -contact.dist)),
                "contact_category": "fixture_ball",
                "counterpart_geom_id": int(counterpart),
                "normal_force_n": float(force[0]),
                "normal_impulse_n_s": None,
                "relative_velocity_world_m_s": None,
                "expected_fixture_contact": True,
                "snag": False,
            }
        )
    return rows


def _ball_contact_active(data: mujoco.MjData, ball_geom: int) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 == ball_geom or contact.geom2 == ball_geom:
            return True
    return False


def rollout_episode(
    *,
    sample,
    width: int,
    height: int,
    fps: int,
    camera_names: list[str] | None = None,
) -> EpisodeRollout:
    """Run one passive rollout, rendering round(duration x fps) frames at k/fps.

    The ball is free from t=0 with its sampled velocity and matching rolling
    spin; nothing else in the scene moves on its own.
    """

    camera_names = list(camera_names) if camera_names is not None else list(DEFAULT_CAMERA_NAMES)
    bundle = build_episode(sample)
    model, data = bundle.model, bundle.data

    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, BALL_JOINT_NAME)
    ball_qadr = int(model.jnt_qposadr[ball_joint])
    ball_vadr = int(model.jnt_dofadr[ball_joint])
    data.qpos[ball_qadr : ball_qadr + 3] = sample.ball_initial_position
    data.qpos[ball_qadr + 3 : ball_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[ball_vadr : ball_vadr + 3] = sample.ball_initial_velocity
    data.qvel[ball_vadr + 3 : ball_vadr + 6] = sample.ball_initial_angular_velocity
    mujoco.mj_forward(model, data)

    renderer = MultiEpisodeRenderer(
        model,
        width=width,
        height=height,
        fps=fps,
        camera_names=camera_names,
    )

    timestamps = fixed_duration_frame_timestamps(sample.duration_s, sample.fps)
    frame_count = len(timestamps)
    steps_per_frame = max(1, round((1.0 / sample.timestep) / sample.fps))
    num_steps = int(round(sample.duration_s / sample.timestep))
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BALL_GEOM_NAME)
    start_xy = np.asarray(sample.ball_initial_position[:2], dtype=np.float64)
    fell_off_threshold_z = sample.tabletop_height - 2.0 * sample.ball_radius

    frame_rows: list[dict] = []
    event_rows: list[dict] = []
    object_state_rows: list[dict] = []
    fell_off_time: float | None = None
    max_penetration = 0.0

    def capture_frame(frame_index: int, simulation_timestamp: float, contact_in_interval: bool) -> None:
        renderer.add_frame(data)
        target = timestamps[frame_index]
        contact_now = _ball_contact_active(data, ball_geom)
        frame_rows.append(
            {
                "frame_index": frame_index,
                "video_frame_index": frame_index,
                "timestamp": target,
                "simulation_timestamp": float(simulation_timestamp),
                "synchronization_error_s": abs(float(simulation_timestamp) - target),
                "action.actuator_command": [],
                "simulator.applied_actuator_ctrl": [],
                "action.mode": ACTION_MODE,
                "contact.active": contact_now,
                "event.contact": bool(contact_in_interval or contact_now),
                "assistance.active": False,
                "assistance.assisted_grasp": False,
                "assistance.assisted_retention": False,
                "assistance.equality_constraint_active": False,
                "assistance.latch_active": False,
                "assistance.mechanism_ids": [],
            }
        )
        object_state_rows.append(
            {
                "timestamp": target,
                "object_id": BALL_OBJECT_ID,
                "object.position": [float(v) for v in data.qpos[ball_qadr : ball_qadr + 3]],
                "object.quaternion_wxyz": [float(v) for v in data.qpos[ball_qadr + 3 : ball_qadr + 7]],
                "object.linear_velocity": [float(v) for v in data.qvel[ball_vadr : ball_vadr + 3]],
                "object.angular_velocity": [float(v) for v in data.qvel[ball_vadr + 3 : ball_vadr + 6]],
            }
        )

    start = time.time()
    capture_frame(0, 0.0, contact_in_interval=False)
    frames_captured = 1
    contact_since_last_frame = False
    for step in range(num_steps):
        mujoco.mj_step(model, data)
        t_after = (step + 1) * sample.timestep
        step_contacts = _ball_contact_rows(model, data, ball_geom, t_after)
        event_rows.extend(step_contacts)
        contact_since_last_frame = contact_since_last_frame or bool(step_contacts)
        for row in step_contacts:
            max_penetration = max(max_penetration, row["penetration_depth_m"])
        ball_z = float(data.qpos[ball_qadr + 2])
        if fell_off_time is None and ball_z < fell_off_threshold_z:
            fell_off_time = float(t_after)
        if (step + 1) % steps_per_frame == 0 and frames_captured < frame_count:
            capture_frame(frames_captured, t_after, contact_since_last_frame)
            frames_captured += 1
            contact_since_last_frame = False
    elapsed = time.time() - start
    if frames_captured != frame_count:
        raise RuntimeError(f"Captured {frames_captured} frames, expected {frame_count}")

    final_position = [float(v) for v in data.qpos[ball_qadr : ball_qadr + 3]]
    final_velocity = [float(v) for v in data.qvel[ball_vadr : ball_vadr + 3]]
    roll_metrics = {
        "initial_speed_mps": float(np.linalg.norm(np.asarray(sample.ball_initial_velocity[:2]))),
        "final_speed_mps": float(np.linalg.norm(np.asarray(final_velocity[:2]))),
        "travel_distance_m": float(
            np.linalg.norm(np.asarray(final_position[:2], dtype=np.float64) - start_xy)
        ),
        "final_position_m": final_position,
        "fell_off_counter": fell_off_time is not None,
        "fell_off_time_s": fell_off_time,
        "max_penetration_depth_m": float(max_penetration),
        "contact_event_count": len(event_rows),
    }

    transition_rows = [
        {
            "timestamp": 0.0,
            "event_type": "roll_start",
            "from": "at_rest",
            "to": "rolling",
            "active_surface": "island_counter",
        }
    ]
    if fell_off_time is not None:
        transition_rows.append(
            {
                "timestamp": fell_off_time,
                "event_type": "left_counter_surface",
                "from": "rolling",
                "to": "falling",
                "active_surface": None,
            }
        )

    frames = {name: renderer.renderers[name].frames for name in camera_names}
    renderer.release()
    return EpisodeRollout(
        sample=sample,
        frames=frames,
        frame_timestamps=timestamps,
        frame_rows=frame_rows,
        high_rate_rows=[],
        event_rows=event_rows,
        transition_rows=transition_rows,
        object_state_rows=object_state_rows,
        roll_metrics=roll_metrics,
        camera_pose=bundle.camera_pose,
        elapsed_wall_time_s=elapsed,
        frame_counts={name: len(view) for name, view in frames.items()},
    )


def write_canonical_episode(
    rollout: EpisodeRollout,
    *,
    root: Path,
    episode_index: int,
    task_index: int,
    camera_streams: dict[str, str],
    video_spec: VideoSpec,
    overwrite: bool = False,
) -> dict:
    """Publish one rollout in the canonical sharded layout; return artifact info."""

    streams = {camera: canonical_camera_name(stream) for camera, stream in camera_streams.items()}
    if set(streams) != set(rollout.frames):
        raise ValueError(f"camera_streams must map every rendered camera: {sorted(rollout.frames)}")
    videos = {streams[camera]: frames for camera, frames in rollout.frames.items()}

    def bound(rows: list[dict]) -> list[dict]:
        return [{"episode_index": episode_index, **row} for row in rows]

    frame_rows = [
        {"episode_index": episode_index, "task_index": task_index, **row}
        for row in rollout.frame_rows
    ]
    artifacts = write_episode_artifacts(
        root,
        episode_index,
        frame_rows=frame_rows,
        high_rate_rows=bound(rollout.high_rate_rows),
        event_rows=bound(rollout.event_rows),
        transition_rows=bound(rollout.transition_rows),
        object_state_rows=bound(rollout.object_state_rows),
        videos=videos,
        video_spec=video_spec,
        overwrite=overwrite,
    )
    artifacts["camera_streams"] = streams
    return artifacts


def build_episode_record(
    rollout: EpisodeRollout,
    *,
    artifacts: dict,
    episode_uuid_text: str,
    episode_index: int,
    task_index: int,
    family: str,
    subfamily: str,
    scene_seed: int,
    branch_seed: int,
    config_hash: str,
) -> dict:
    """Assemble the v2 episode-metadata record for one passive rollout."""

    sample = rollout.sample
    metrics = rollout.roll_metrics
    visual_settings = dict(sample.visual_settings or {})
    calibration_ids = {stream: stream for stream in artifacts["video_paths"]}
    key_event_time = metrics.get("fell_off_time_s")
    key_event_name = None if key_event_time is None else "left_counter_surface"
    return {
        "episode_uuid": episode_uuid_text,
        "episode_index": episode_index,
        "counterfactual_bundle_id": f"{family}_{episode_index:06d}",
        "scene_seed": int(scene_seed),
        "branch_seed": int(branch_seed),
        "family": family,
        "subfamily": subfamily,
        "intended_branch": "free_roll",
        "actual_outcome": "success",
        "task_success": True,
        "failure_mode": "none",
        "schema_version": SCHEMA_VERSION,
        "actual_outcome_class": "success",
        "primary_failure_code": "none",
        "failure_tags": [],
        "failure_taxonomy_version": FAILURE_TAXONOMY_VERSION,
        "variant": sample.scene_variant,
        "robot_model": "none",
        "tool_type": "none",
        "action_mode": ACTION_MODE,
        "partial_success_score": None,
        "label_confidence": None,
        "label_status": "verified_objective",
        "dynamics_mode": "free_contact",
        "release_tier": "free_contact",
        "physics_qc_pass": True,
        "split": "unassigned",
        "physics_counterfactual_family_id": None,
        "split_group_id": f"{family}_{episode_index:06d}",
        "parent_episode_uuid": None,
        "source_generator": "ball_rolling_dynamics_scripts",
        "source_generator_version": "v1",
        "generator_git_commit": "unknown",
        "config_hash": config_hash,
        "asset_ids": [
            f"robocasa_layout{visual_settings.get('kitchen_layout_id')}"
            f"_style{visual_settings.get('kitchen_style_id')}"
        ],
        "asset_hashes": {},
        "simulator_name": "mujoco",
        "simulator_version": mujoco.__version__,
        "renderer": "mujoco_offscreen_egl",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_data_path": artifacts["frame_data_path"],
        "high_rate_path": artifacts["high_rate_path"],
        "events_path": artifacts["events_path"],
        "transition_events_path": artifacts["transition_events_path"],
        "object_states_path": artifacts["object_states_path"],
        "camera_ids": sorted(calibration_ids.values()),
        "task_index": task_index,
        "frame_count": int(artifacts["frame_count"]),
        "duration_s": float(sample.duration_s),
        "event_time_s": key_event_time,
        "key_event_name": key_event_name,
        "key_event_time_s": key_event_time,
        "objective_evaluator_id": "legacy_embedded",
        "objective_evaluator_version": "unversioned",
        "objective_threshold_set_hash": sha256_json(
            {"metric": "passive_rollout_completed", "hard_failures": "none_defined"}
        ),
        "quality_flags": [],
        "release_eligible": False,
        "physics": {
            "gravity_m_s2": list(sample.gravity),
            "simulation_hz": 1.0 / sample.timestep,
            "timestep_s": sample.timestep,
            "ball_radius_m": sample.ball_radius,
            "ball_mass_kg": sample.ball_mass,
            "ball_friction": list(BALL_FRICTION),
            "ball_solref": list(BALL_SOLREF),
            "ball_solimp": list(BALL_SOLIMP),
            "ball_initial_position_m": list(sample.ball_initial_position),
            "ball_initial_velocity_m_s": list(sample.ball_initial_velocity),
            "ball_initial_angular_velocity_rad_s": list(sample.ball_initial_angular_velocity),
            "dynamics": "passive_free_contact_no_robot",
        },
        "assistance": {
            "assisted_grasp": False,
            "assisted_retention": False,
            "equality_constraint_active": False,
            "latch_active": False,
            "constraint_activation_time": None,
            "constraint_deactivation_time": None,
        },
        "objective_metrics": {
            "task_success": True,
            **metrics,
        },
        "objective_evidence": {
            "stored_objective_success": True,
            "independently_recomputed": False,
            "source": "legacy_embedded_metrics",
        },
        "randomization": {
            "randomized_fields": ["ball_initial_velocity"],
            "rolling_speed_mps": visual_settings.get("rolling_speed_mps"),
            "rolling_speed_cap_mps": visual_settings.get("rolling_speed_cap_mps"),
            "rolling_runway_m": visual_settings.get("rolling_runway_m"),
            "rolling_heading_offset_rad": visual_settings.get("rolling_heading_offset_rad"),
            "rolling_direction_xy": visual_settings.get("rolling_direction_xy"),
        },
        "content_hashes": artifacts["content_hashes"],
        "video_paths": artifacts["video_paths"],
        "camera_stream_calibration_ids": calibration_ids,
        "controller_profile": {
            "profile_id": "passive_no_controller",
            "profile_version": "v1",
            "control_latency_s": None,
            "camera_latency_s": None,
        },
        "robot_start_provenance": {},
        "tool_calibration_provenance": {},
        "extras": {
            "camera_streams": artifacts["camera_streams"],
            "camera_poses": rollout.camera_pose,
            "scene_resolution_backend": visual_settings.get("scene_resolution_backend"),
            "scene_resolution_warning": visual_settings.get("scene_resolution_warning"),
            "visual_settings": visual_settings,
            "roll_metrics": metrics,
            "elapsed_wall_time_s": rollout.elapsed_wall_time_s,
            "frame_counts": rollout.frame_counts,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a seeded passive ball-rolling rollout in a RoboCasa island scene.")
    parser.add_argument("--seed", type=int, required=True, help="Velocity-sampling seed.")
    parser.add_argument("--layout-id", type=int, required=True, help="RoboCasa kitchen layout ID.")
    parser.add_argument("--style-id", type=int, required=True, help="RoboCasa kitchen style ID.")
    parser.add_argument("--camera-variant", type=str, default="island_rolling_view", help="Camera variant.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Canonical dataset root where the rollout artifacts will be written.",
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


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.width = 320
        args.height = 240
        args.fps = 10
        args.duration = 1.0

    spec = VideoSpec(width=args.width, height=args.height, fps_num=args.fps)
    family = f"run_rollout_layout{args.layout_id}_style{args.style_id}"
    config = {
        "seed": args.seed,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "camera_variant": args.camera_variant,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "duration": args.duration,
    }
    sample = sample_episode(
        "robocasa_kitchen",
        args.seed,
        fps=args.fps,
        duration=args.duration,
        layout_id_override=args.layout_id,
        style_id_override=args.style_id,
        camera_variant=args.camera_variant,
    )
    sample = replace(sample, offscreen_width=args.width, offscreen_height=args.height)
    sample = randomize_initial_velocity(sample, np.random.default_rng(args.seed))
    rollout = rollout_episode(
        sample=sample,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    artifacts = write_canonical_episode(
        rollout,
        root=args.output_dir,
        episode_index=0,
        task_index=0,
        camera_streams=DEFAULT_CAMERA_STREAMS,
        video_spec=spec,
        overwrite=True,
    )
    record = build_episode_record(
        rollout,
        artifacts=artifacts,
        episode_uuid_text=episode_uuid(family, 0),
        episode_index=0,
        task_index=0,
        family=family,
        subfamily=family,
        scene_seed=args.seed,
        branch_seed=args.seed,
        config_hash=sha256_json(config),
    )
    write_episode_marker(args.output_dir, record, sha256_json(config))

    for stream in sorted(record["video_paths"]):
        print(args.output_dir / record["video_paths"][stream])
    print(args.output_dir / record["frame_data_path"])
    print(record["extras"]["scene_resolution_backend"])
    print(
        "initial_speed_mps",
        round(record["objective_metrics"]["initial_speed_mps"], 3),
        "heading_offset_deg",
        round(float(np.degrees(record["randomization"]["rolling_heading_offset_rad"] or 0.0)), 1),
    )


if __name__ == "__main__":
    main()
