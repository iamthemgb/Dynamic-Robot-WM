from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
import traceback
import types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ROBOCASA_ASSETS_ROOT", "/gpfs/radev/project/sous/mzl7/robocasa/robocasa/models/assets")

ROBOCASA_CHECKOUT = Path("/gpfs/radev/project/sous/mzl7/robocasa")
if ROBOCASA_CHECKOUT.exists() and str(ROBOCASA_CHECKOUT) not in sys.path:
    sys.path.insert(0, str(ROBOCASA_CHECKOUT))
ROBOSUITE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "robosuite"
if ROBOSUITE_ROOT.exists() and str(ROBOSUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOSUITE_ROOT))
try:
    import termcolor  # noqa: F401
except ModuleNotFoundError:
    fallback = types.ModuleType("termcolor")
    fallback.colored = lambda text, *args, **kwargs: text
    sys.modules["termcolor"] = fallback

import mujoco
import numpy as np

from scripts_mujoco.robocasa_assets import robocasa_assets_root
from scripts_mujoco.robotiq_verification import robotiq_rollout_check
from scripts_mujoco.scene_builder import build_robotiq_thick_pad_episode, sample_episode
from scripts_mujoco.utils import sphere_mass

from .controller import PlannedRobotiqCatchController
from .lerobot_export import ACTION_NAMES, STATE_NAMES, LeRobotWriter
from .taxonomy import FAILURE_MODE, FAMILY, SUBFAMILIES, branch_sequence, build_plan


SCENE_VARIANTS = (
    "robocasa_tabletop",
    "robocasa_lab",
    "robocasa_workbench",
    "robocasa_storage",
    "robocasa_kitchen",
    "robocasa_official_kitchen",
)
SCENE_RANDOMIZATION_LEVELS = ("clean", "balanced", "cluttered")
SUBFAMILY_ORDER = ("centered_vertical_drop", "direct_projectile_interception")
VIEWS = ("main_camera", "side_camera")
HOME_Q = np.array(
    [
        0.0,
        -0.7,
        0.0,
        -2.2,
        0.0,
        1.6,
        0.78,
        0.002273890874386094,
        0.0001364909715520716,
        0.0024731211244206548,
        -0.00267025473687781,
        0.002273890874386094,
        0.0001364909715520716,
        0.0024731211244206548,
        -0.00267025473687781,
    ],
    dtype=np.float64,
)
ARM_JITTER = np.array([0.10, 0.10, 0.12, 0.10, 0.14, 0.12, 0.18], dtype=np.float64)
BALL_COLORS = {
    "orange": (0.95, 0.45, 0.10, 1.0),
    "green": (0.15, 0.65, 0.25, 1.0),
    "purple": (0.55, 0.20, 0.75, 1.0),
    "red": (0.85, 0.15, 0.12, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
    "yellow": (0.93, 0.82, 0.12, 1.0),
    "cyan": (0.10, 0.72, 0.78, 1.0),
    "magenta": (0.86, 0.16, 0.62, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Robotiq thick-pad LeRobot ball-catch data.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-per-subfamily", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--control-hz", type=int, default=120)
    parser.add_argument("--duration", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--ball-radius", type=float, default=0.025)
    parser.add_argument("--scene-variants", default=",".join(SCENE_VARIANTS))
    parser.add_argument("--scene-randomization-levels", default=",".join(SCENE_RANDOMIZATION_LEVELS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--retry-limit", type=int, default=4)
    parser.add_argument("--rich-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-require-verification", action="store_true")
    return parser.parse_args()


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _episode_indices(total_episodes: int, num_shards: int, shard: int) -> list[int]:
    return [i for i in range(total_episodes) if i % num_shards == shard]


def _state_vector(ids, data) -> np.ndarray:
    robot_q = data.qpos[ids.qpos_adrs]
    grasp = data.site_xpos[ids.grasp_site]
    ball = data.xpos[ids.ball_body]
    ball_v = data.qvel[ids.ball_qvel_adr : ids.ball_qvel_adr + 3]
    return np.concatenate([robot_q, grasp, ball, ball_v]).astype(np.float32)


def _frame(t: float | None, fps: int) -> int | None:
    return None if t is None else int(round(float(t) * int(fps)))


def _base_sample(
    rng: np.random.Generator,
    *,
    seed: int,
    fps: int,
    duration: float,
    ball_radius: float,
    scene_variant: str,
    scene_randomization_level: str,
) -> tuple[object, str]:
    sample = sample_episode(rng, scene_variant, seed=seed, fps=fps, duration=duration)
    density = sample.ball_mass / sphere_mass(sample.ball_radius, 1.0)
    color_name = str(rng.choice(list(BALL_COLORS)))
    sample = dataclasses.replace(
        sample,
        ball_radius=float(ball_radius),
        ball_mass=sphere_mass(float(ball_radius), density),
        ball_color=BALL_COLORS[color_name],
        release_time_s=0.0,
        scene_randomization_level=scene_randomization_level,
    )
    return sample, color_name


def _centered_vertical_sample(sample, rng: np.random.Generator):
    close_z = sample.catch_center_z + 0.055
    total_time = float(rng.uniform(0.400, 0.460))
    target_xy = rng.uniform((0.43, -0.055), (0.52, 0.055))
    start_offset = rng.normal(0.0, (0.008, 0.008))
    start_xy = target_xy + start_offset
    gravity = abs(float(sample.gravity[2]))
    start_z = close_z + 0.5 * gravity * total_time * total_time
    velocity_xy = (target_xy - start_xy) / total_time
    return dataclasses.replace(
        sample,
        ball_initial_position=(float(start_xy[0]), float(start_xy[1]), float(start_z)),
        ball_initial_velocity=(float(velocity_xy[0]), float(velocity_xy[1]), 0.0),
        planned_intercept_time_s=total_time,
        planned_intercept_position=(float(target_xy[0]), float(target_xy[1]), sample.catch_center_z),
        interception_subfamily=SUBFAMILIES["centered_vertical_drop"]["label"],
    )


def _direct_projectile_sample(sample, rng: np.random.Generator):
    close_z = sample.catch_center_z + 0.055
    target_xy = rng.uniform((0.43, -0.055), (0.52, 0.055))
    total_time = float(rng.uniform(0.400, 0.470))
    start_offset = rng.uniform((-0.095, -0.070), (0.095, 0.070))
    if np.linalg.norm(start_offset) < 0.055:
        start_offset[0] = 0.085 if start_offset[0] >= 0.0 else -0.085
    start_xy = target_xy + start_offset
    gravity = abs(float(sample.gravity[2]))
    start_z = close_z + 0.5 * gravity * total_time * total_time
    velocity_xy = (target_xy - start_xy) / total_time
    return dataclasses.replace(
        sample,
        ball_initial_position=(float(start_xy[0]), float(start_xy[1]), float(start_z)),
        ball_initial_velocity=(float(velocity_xy[0]), float(velocity_xy[1]), 0.0),
        planned_intercept_time_s=total_time,
        planned_intercept_position=(float(target_xy[0]), float(target_xy[1]), sample.catch_center_z),
        interception_subfamily=SUBFAMILIES["direct_projectile_interception"]["label"],
    )


def _make_sample(
    rng: np.random.Generator,
    *,
    seed: int,
    fps: int,
    duration: float,
    ball_radius: float,
    subfamily_key: str,
    scene_variant: str,
    scene_randomization_level: str,
):
    sample, color_name = _base_sample(
        rng,
        seed=seed,
        fps=fps,
        duration=duration,
        ball_radius=ball_radius,
        scene_variant=scene_variant,
        scene_randomization_level=scene_randomization_level,
    )
    if subfamily_key == "centered_vertical_drop":
        return _centered_vertical_sample(sample, rng), color_name
    if subfamily_key == "direct_projectile_interception":
        return _direct_projectile_sample(sample, rng), color_name
    raise ValueError(f"Unsupported subfamily: {subfamily_key}")


def _background_assets(camera_pose: dict) -> tuple[list[dict], list[dict], list[str], dict]:
    main = camera_pose.get("main_camera", {}) if isinstance(camera_pose, dict) else {}
    robocasa_assets = list(main.get("robocasa_background_assets", []) or [])
    robotwin_assets = list(main.get("robotwin_background_assets", []) or [])
    ids: list[str] = []
    for asset in robocasa_assets + robotwin_assets:
        ids.append(
            str(
                asset.get("model_xml")
                or asset.get("model_file")
                or asset.get("asset_source")
                or asset.get("slot")
                or "background_asset"
            )
        )
    scene_randomization = dict(main.get("scene_randomization", {}) or {})
    return robocasa_assets, robotwin_assets, ids, scene_randomization


def run_episode(
    writer: LeRobotWriter,
    *,
    episode_index: int,
    local_index: int,
    subfamily_key: str,
    branch: str,
    seed: int,
    scene_variant: str,
    scene_randomization_level: str,
    fps: int,
    control_hz: int,
    duration: float,
    width: int,
    height: int,
    ball_radius: float,
    extra_out: Path | None,
    require_verification: bool,
) -> dict:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    sample, ball_color_name = _make_sample(
        rng,
        seed=seed,
        fps=fps,
        duration=duration,
        ball_radius=ball_radius,
        subfamily_key=subfamily_key,
        scene_variant=scene_variant,
        scene_randomization_level=scene_randomization_level,
    )
    sample = dataclasses.replace(sample, offscreen_width=max(width, 1280), offscreen_height=max(height, 720))
    plan = build_plan(branch, rng)

    bundle = build_robotiq_thick_pad_episode(sample)
    ctrl = PlannedRobotiqCatchController(bundle.model, bundle.data, sample, plan)
    ctrl.home_q = HOME_Q.copy()
    ctrl.home_q[:7] += rng.normal(0.0, ARM_JITTER)
    ctrl.prepare()
    ids = ctrl.ids

    renderers = {view: mujoco.Renderer(bundle.model, height=height, width=width) for view in VIEWS}
    view_frames = {view: [] for view in VIEWS}
    states, actions, timestamps = [], [], []
    n_video = 0
    timestep = float(sample.timestep)
    total_steps = int(round(duration / timestep))
    frame_every = max(1, int(round((1.0 / fps) / timestep)))
    ctrl_every = max(1, int(round((1.0 / control_hz) / timestep)))

    try:
        for step in range(total_steps + 1):
            t = step * timestep
            if step > 0:
                ctrl.before_step(float(t))
                mujoco.mj_step(bundle.model, bundle.data)
                ctrl.after_step(float(t))
            if step % frame_every == 0:
                for view in VIEWS:
                    renderers[view].update_scene(bundle.data, camera=view)
                    view_frames[view].append(renderers[view].render().copy())
                n_video += 1
            if step % ctrl_every == 0:
                states.append(_state_vector(ids, bundle.data))
                actions.append(np.asarray(ctrl.last_cmd, dtype=np.float32).copy())
                timestamps.append(float(t))
    finally:
        for renderer in renderers.values():
            renderer.close()

    video_frame_index = [min(int(round(t * fps)), max(0, n_video - 1)) for t in timestamps]
    result = ctrl.result()
    expected_success = branch == "success"
    verification = robotiq_rollout_check(
        bundle.model,
        bundle.data,
        result,
        expected_success=expected_success,
        ball_radius=sample.ball_radius,
    )
    if require_verification and not verification["passed"]:
        raise RuntimeError(json.dumps({"verification": verification, "controller_result": result}, indent=2))

    outcome = "success" if result["success"] else "failure"
    subfamily = SUBFAMILIES[subfamily_key]
    robocasa_assets, robotwin_assets, background_ids, scene_randomization = _background_assets(bundle.camera_pose)
    failure_mode = "none" if outcome == "success" else FAILURE_MODE.get(branch, "unexpected_failure")

    rich = {
        "episode_id": f"{subfamily_key}_{episode_index:06d}",
        "scene_seed_id": int(seed),
        "family": FAMILY,
        "subfamily": subfamily["label"],
        "subfamily_key": subfamily_key,
        "task": subfamily["task"],
        "task_index": int(subfamily["task_index"]),
        "branch": branch,
        "branch_mode": plan.branch_mode,
        "outcome": outcome,
        "failure_mode": failure_mode,
        "intended_failure_mode": FAILURE_MODE.get(branch, "none"),
        "views": list(VIEWS),
        "scene_variant": scene_variant,
        "scene_randomization_level": scene_randomization_level,
        "object_asset_id": "sphere_ball",
        "background_asset_id": background_ids or [scene_variant],
        "robocasa_background_assets": robocasa_assets,
        "robotwin_background_assets": robotwin_assets,
        "tool_type": "robotiq_2f85_thick_pad",
        "robot_model": "franka_panda_nohand_plus_robotiq_2f85",
        "simulator": "MuJoCo",
        "fps": int(fps),
        "control_hz": int(control_hz),
        "duration_sec": float(duration),
        "gravity": list(sample.gravity),
        "timestep": float(sample.timestep),
        "robot_base_position": list(sample.robot_base_position),
        "catch_center_z": float(sample.catch_center_z),
        "object_physics": {
            "mass": float(sample.ball_mass),
            "radius": float(sample.ball_radius),
            "friction": 5.0,
            "restitution": 0.0,
        },
        "ball_initial_position": list(sample.ball_initial_position),
        "ball_initial_velocity": list(sample.ball_initial_velocity),
        "planned_intercept_time_s": float(sample.planned_intercept_time_s or 0.0),
        "planned_intercept_position": list(sample.planned_intercept_position or ()),
        "randomization": {
            "lighting_id": round(float(sample.lighting_intensity), 4),
            "camera_pose_id": {view: bundle.camera_pose.get(view, {}) for view in VIEWS},
            "table_texture_id": scene_randomization.get("texture_profile", {}),
            "ball_color": ball_color_name,
            "arm_initial_qpos": [float(v) for v in ctrl.home_q],
            "scene_randomization": scene_randomization,
        },
        "events": {
            "release_frame": _frame(sample.release_time_s, fps),
            "first_contact_frame": _frame(result.get("first_contact_time_s"), fps),
            "first_pad_contact_frame": _frame(result.get("first_pad_contact_time_s"), fps),
            "first_two_pad_contact_frame": _frame(result.get("first_two_pad_contact_time_s"), fps),
            "entered_grasp_frame": _frame(result.get("grasp_time_s"), fps),
            "exit_frame": None,
            "ballistic_intercept_frame": _frame(result.get("ballistic_intercept_time_s"), fps),
            "arm_arrival_frame": _frame(result.get("arm_arrival_time_s"), fps),
        },
        "metrics": {
            "min_dist": result.get("min_dist"),
            "success": bool(result["success"]),
            "grasped": bool(result["grasped"]),
            "stayed_inside_until_end": bool(result["success"]),
            "contact_frames": int(result.get("contact_frames", 0)),
            "left_pad_contact_frames": int(result.get("left_pad_contact_frames", 0)),
            "right_pad_contact_frames": int(result.get("right_pad_contact_frames", 0)),
            "two_pad_contact_frames": int(result.get("two_pad_contact_frames", 0)),
            "max_retained_time_s": float(result.get("max_retained_time_s", 0.0)),
        },
        "controller_result": result,
        "robotiq_verification": verification,
        "pad_orientation": "vertical_opposing_faces",
        "clasp_opening_orientation": "lateral_horizontal",
        "robotiq_mount_quat": [0.5, 0.5, -0.5, -0.5],
        "asset_roots": {
            "robocasa_assets_root": str(robocasa_assets_root()),
            "robocasa_checkout": str(ROBOCASA_CHECKOUT),
            "robotwin_1_0_models": str(Path(__file__).resolve().parents[1] / "third_party" / "robotwin_1_0" / "models"),
            "robotwin_2_0_objects": str(Path(__file__).resolve().parents[1] / "third_party" / "robotwin_2_0" / "assets" / "objects"),
        },
        "state_dim": len(STATE_NAMES),
        "action_dim": len(ACTION_NAMES),
        "elapsed_wall_time_s": time.perf_counter() - start,
    }

    rec = writer.add_episode(
        episode_index,
        task_index=subfamily["task_index"],
        main_frames=view_frames["main_camera"],
        side_frames=view_frames["side_camera"],
        state=np.stack(states),
        action=np.stack(actions),
        timestamps=timestamps,
        video_frame_index=video_frame_index,
        rich_meta=rich,
    )
    if extra_out is not None:
        extra_out.mkdir(parents=True, exist_ok=True)
        (extra_out / f"episode_{episode_index:06d}.json").write_text(json.dumps(rich, indent=2), encoding="utf-8")
    return rec


def main() -> None:
    args = parse_args()
    if args.episodes_per_subfamily <= 0:
        raise ValueError("--episodes-per-subfamily must be positive")
    if args.retry_limit <= 0:
        raise ValueError("--retry-limit must be positive")
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError("--shard must be in [0, --num-shards)")

    scene_variants = _split_csv(args.scene_variants)
    scene_levels = _split_csv(args.scene_randomization_levels)
    total_episodes = args.episodes_per_subfamily * len(SUBFAMILY_ORDER)
    indices = _episode_indices(total_episodes, args.num_shards, args.shard)
    root = args.out if args.num_shards == 1 else args.out / f"shard{args.shard:03d}"
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    writer = LeRobotWriter(
        root,
        fps=args.fps,
        control_hz=args.control_hz,
        resolution=(args.width, args.height),
        tasks={info["task_index"]: info["task"] for info in SUBFAMILIES.values()},
    )
    extra_out = root / "meta" / "rich" if args.rich_json else None
    branches = {
        key: branch_sequence(args.episodes_per_subfamily, args.seed + 101 * task_i)
        for task_i, key in enumerate(SUBFAMILY_ORDER)
    }

    print(
        f"[robotiq-generate] out={root} shard={args.shard}/{args.num_shards} "
        f"episodes={len(indices)}/{total_episodes} scenes={scene_variants} MUJOCO_GL={os.environ.get('MUJOCO_GL')}",
        flush=True,
    )
    ok = 0
    for episode_index in indices:
        task_i = episode_index // args.episodes_per_subfamily
        local_index = episode_index % args.episodes_per_subfamily
        subfamily_key = SUBFAMILY_ORDER[task_i]
        branch = branches[subfamily_key][local_index]
        scene_variant = scene_variants[local_index % len(scene_variants)]
        level = scene_levels[(local_index + local_index // len(scene_variants)) % len(scene_levels)]
        last_error: Exception | None = None
        rec = None
        for attempt in range(args.retry_limit):
            seed = int(args.seed + episode_index + 1_000_003 * attempt)
            try:
                rec = run_episode(
                    writer,
                    episode_index=episode_index,
                    local_index=local_index,
                    subfamily_key=subfamily_key,
                    branch=branch,
                    seed=seed,
                    scene_variant=scene_variant,
                    scene_randomization_level=level,
                    fps=args.fps,
                    control_hz=args.control_hz,
                    duration=args.duration,
                    width=args.width,
                    height=args.height,
                    ball_radius=args.ball_radius,
                    extra_out=extra_out,
                    require_verification=not args.no_require_verification,
                )
                rec["retry_attempt"] = int(attempt)
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"[ep {episode_index:06d}] attempt={attempt} {subfamily_key}/{branch}/{scene_variant} FAILED:\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
        if rec is None:
            raise RuntimeError(f"Episode {episode_index} failed after {args.retry_limit} attempts") from last_error
        ok += 1
        print(
            f"[ep {episode_index:06d}] {subfamily_key:31s} {scene_variant:25s} "
            f"branch={branch:17s} outcome={rec['outcome']:7s} "
            f"verify={rec['robotiq_verification']['passed']} len={rec['length']} frames={rec['video_frames']} "
            f"t={rec['elapsed_wall_time_s']:.1f}s",
            flush=True,
        )

    summary = writer.finalize()
    print(f"[robotiq-generate] done {ok}/{len(indices)} -> {root}\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
