"""Episode driver + dataset CLI for the Franka ground-bounce catch data (F1_B).

The ball is released at t=0, falls, bounces once off the tabletop (per-episode
restitution), and rebounds into the catch zone; the Franka waits at home and
reactively snatches it near the rebound apex (see ``controller``). Each episode
is drawn from the success/failure taxonomy across three environments
(clean_lab, office, robocasa_kitchen), rendered from two synchronized oblique
views at 832x480 @ 30 fps with a 120 Hz proprio stream, and packaged as a
LeRobotDataset-v3-style dataset.

Headless rendering uses EGL on a GPU node (set MUJOCO_GL=egl).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

from franka_catch.lerobot_export import ACTION_NAMES, STATE_NAMES, VIEW_KEYS, LeRobotWriter
from franka_catch.variants import VARIANT_NAMES

from .controller import BounceCatchController
from .scene_builder import build_bounce_episode, sample_bounce_episode
from .taxonomy import FAILURE_MODE, FAMILY, SUBFAMILY, build_plan, choose_branch

HOME_Q = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 1.6, 0.78, 0.04, 0.04], dtype=np.float64)
# Small perturbation of the catch-ready starting pose (kept small so it never
# inflates the post-bounce dip past the Franka-feasible window; see controller).
ARM_JITTER = np.array([0.03, 0.025, 0.03, 0.025, 0.035, 0.03, 0.04], dtype=np.float64)
STATE_DIM = 18
ACTION_DIM = 8
# MuJoCo camera name backing each exported view key (parallel to VIEW_KEYS).
VIEW_CAMERAS = ("main_camera", "side_camera")
VIEWS = VIEW_CAMERAS
TASK = "Catch the ball after it bounces off the table with the Franka gripper."


def _state_vector(ids, data) -> np.ndarray:
    arm = data.qpos[ids.qpos_adrs[:7]]
    fing = data.qpos[ids.qpos_adrs[7:9]]
    ee = data.xpos[ids.hand_body]
    ball = data.xpos[ids.ball_body]
    ball_v = data.qvel[ids.ball_qvel_adr:ids.ball_qvel_adr + 3]
    return np.concatenate([arm, fing, ee, ball, ball_v]).astype(np.float32)


def _f(t, fps):
    return None if t is None else int(round(t * fps))


def run_episode(writer: LeRobotWriter, *, episode_index: int, variant: str, seed: int,
                fps: int, control_hz: int, duration: float, width: int, height: int,
                extra_out: Path | None = None) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    sample = sample_bounce_episode(rng, variant, seed, fps=fps, duration=duration)
    sample = dataclasses.replace(sample, offscreen_width=max(width, 1280), offscreen_height=max(height, 720))
    branch = choose_branch(rng)
    plan = build_plan(branch, rng)
    # Variety is applied to the catch-ready starting pose (see controller.prepare).
    plan.start_jitter = tuple(float(v) for v in rng.normal(0.0, ARM_JITTER))

    bundle = build_bounce_episode(sample)
    model, data = bundle.model, bundle.data
    ctrl = BounceCatchController(model, data, sample, plan)
    ctrl.prepare()
    ids = ctrl.ids
    # True initial arm configuration is the (jittered) catch-ready pose.
    home_q = np.asarray(ctrl.home_q, dtype=np.float64).copy()

    renderers = {v: mujoco.Renderer(model, height=height, width=width) for v in VIEWS}
    view_frames = {v: [] for v in VIEWS}

    ts = sample.timestep
    total_steps = int(round(duration / ts))
    frame_every = max(1, int(round((1.0 / fps) / ts)))
    ctrl_every = max(1, int(round((1.0 / control_hz) / ts)))

    states, actions, timestamps = [], [], []
    n_video = 0
    for step in range(total_steps + 1):
        t = step * ts
        if step > 0:
            ctrl.before_step(t)
            mujoco.mj_step(model, data)
            ctrl.after_step(t)
        if step % frame_every == 0:
            for v in VIEWS:
                renderers[v].update_scene(data, camera=v)
                view_frames[v].append(renderers[v].render().copy())
            n_video += 1
        if step % ctrl_every == 0:
            states.append(_state_vector(ids, data))
            actions.append(np.concatenate([ctrl.last_cmd[:7], [ctrl.gripper_cmd]]).astype(np.float32))
            timestamps.append(float(t))
    for v in VIEWS:
        renderers[v].close()
    vfi = [min(int(round(tt * fps)), max(0, n_video - 1)) for tt in timestamps]

    res = ctrl.result()
    outcome = "success" if res["success"] else "failure"
    cams_meta = {k: v for k, v in bundle.camera_pose.items() if k in VIEWS}

    rich = {
        "episode_id": f"{variant}_{episode_index:06d}",
        "scene_seed_id": int(seed),
        "family": FAMILY,
        "subfamily": SUBFAMILY,
        "branch": branch,
        "outcome": outcome,
        "failure_mode": FAILURE_MODE[branch] if outcome == "failure" else "none",
        "intended_failure_mode": FAILURE_MODE[branch],
        "views": list(VIEWS),
        "scene_variant": variant,
        "object_asset_id": "sphere_ball",
        "background_asset_id": bundle.background_asset_ids or [variant],
        "tool_type": "franka_hand",
        "robot_model": "franka_panda",
        "simulator": "MuJoCo",
        "fps": fps,
        "control_hz": control_hz,
        "duration_sec": float(duration),
        "gravity": list(sample.gravity),
        "timestep": sample.timestep,
        "robot_base_position": list(sample.robot_base_position),
        "catch_center_z": sample.catch_center_z,
        "drop_height": sample.drop_height,
        "bounce": {
            "surface_z": res.get("surface_z"),
            "detected": res.get("bounce_detected"),
            "bounce_time_s": res.get("bounce_time_s"),
            "bounce_position": res.get("bounce_position"),
            "pre_impact_vz": res.get("pre_impact_vz"),
            "post_bounce_velocity": res.get("post_bounce_velocity"),
            "nominal_restitution": res.get("nominal_restitution"),
            "measured_restitution": res.get("measured_restitution"),
            "measured_apex_z": res.get("measured_apex_z"),
            "catch_time_s": res.get("catch_time_s"),
        },
        "object_physics": {
            "mass": sample.ball_mass,
            "radius": sample.ball_radius,
            "friction": sample.ball_friction,
            "restitution": sample.ball_restitution,
        },
        "ball_initial_position": list(sample.ball_initial_position),
        "ball_initial_velocity": list(sample.ball_initial_velocity),
        "ball_initial_angular_velocity": list(sample.ball_initial_angular_velocity),
        "randomization": {
            "lighting_id": round(sample.lighting_intensity, 4),
            "camera_pose_id": {k: {"pos": v.get("pos"), "azimuth_deg": v.get("azimuth_deg"),
                                    "elevation_deg": v.get("elevation_deg"), "fovy": v.get("fovy")}
                               for k, v in cams_meta.items()},
            "table_texture_id": "robocasa_marble" if variant == "robocasa_kitchen" else "procedural",
            "ball_color": sample.ball_color_name,
            "arm_initial_qpos": [float(v) for v in home_q],
        },
        "events": {
            "release_frame": _f(sample.release_time_s, fps),
            "bounce_frame": _f(res.get("bounce_time_s"), fps),
            "first_contact_frame": _f(res["first_contact_time_s"], fps),
            "entered_grasp_frame": _f(res["grasp_time_s"], fps),
            "exit_frame": _f(res["exit_time_s"], fps),
            "catch_frame": _f(res.get("catch_time_s"), fps),
            "arm_arrival_frame": _f(res["arm_arrival_time_s"], fps),
        },
        "metrics": {
            "min_dist": res["min_dist"],
            "success": bool(res["success"]),
            "grasped": bool(res["grasped"]),
            "bounce_detected": bool(res.get("bounce_detected")),
            "measured_restitution": res.get("measured_restitution"),
            "stayed_inside_until_end": bool(res["success"]),
            "contact_frames": res["contact_frames"],
            "max_retained_time_s": res["max_retained_time_s"],
        },
        "controller_result": res,
        "elapsed_wall_time_s": time.time() - t0,
    }

    frames_by_view = {key: view_frames[cam] for key, cam in zip(VIEW_KEYS, VIEW_CAMERAS)}
    rec = writer.add_episode(
        episode_index, frames_by_view=frames_by_view,
        state=np.stack(states), action=np.stack(actions), timestamps=timestamps,
        video_frame_index=vfi, rich_meta=rich,
    )
    if extra_out is not None:
        extra_out.mkdir(parents=True, exist_ok=True)
        (extra_out / f"episode_{episode_index:06d}.json").write_text(json.dumps(rich, indent=2), encoding="utf-8")
    return rec


def _plan_indices(n_episodes: int, num_shards: int, shard: int) -> list[int]:
    return [i for i in range(n_episodes) if i % num_shards == shard]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Franka ground-bounce catch episodes (LeRobot v3-style).")
    p.add_argument("--out", type=Path, required=True, help="dataset root (under scratch)")
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--variants", nargs="*", default=list(VARIANT_NAMES))
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--control-hz", type=int, default=120)
    p.add_argument("--duration", type=float, default=4.5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--rich-json", action="store_true", help="also dump per-episode rich JSON under meta/rich/")
    args = p.parse_args()

    variants = args.variants
    indices = _plan_indices(args.episodes, args.num_shards, args.shard)
    root = args.out if args.num_shards == 1 else args.out / f"shard{args.shard:03d}"
    writer = LeRobotWriter(root, fps=args.fps, control_hz=args.control_hz,
                           state_dim=STATE_DIM, action_dim=ACTION_DIM,
                           resolution=(args.width, args.height), task=TASK)
    extra_out = (root / "meta" / "rich") if args.rich_json else None

    print(f"[bounce-generate] out={root} episodes={len(indices)}/{args.episodes} "
          f"shard={args.shard}/{args.num_shards} variants={variants} "
          f"MUJOCO_GL={os.environ.get('MUJOCO_GL','<unset>')}", flush=True)
    ok = 0
    for local_i, gi in enumerate(indices):
        variant = variants[gi % len(variants)]
        seed = args.seed + gi
        try:
            rec = run_episode(writer, episode_index=gi, variant=variant, seed=seed,
                              fps=args.fps, control_hz=args.control_hz, duration=args.duration,
                              width=args.width, height=args.height, extra_out=extra_out)
            ok += 1
            b = rec["bounce"]
            mr = b.get("measured_restitution")
            print(f"[ep {gi:06d}] {variant:16s} branch={rec['branch']:17s} outcome={rec['outcome']:7s} "
                  f"bounce={str(b.get('detected')):5s} e_meas={mr if mr is None else round(mr,2)} "
                  f"min_dist={rec['metrics']['min_dist']:.3f} len={rec['length']} "
                  f"t={rec['elapsed_wall_time_s']:.1f}s", flush=True)
        except Exception:
            print(f"[ep {gi:06d}] {variant} FAILED:\n{traceback.format_exc()}", flush=True)

    summary = writer.finalize()
    print(f"[bounce-generate] done {ok}/{len(indices)} -> {root}\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
