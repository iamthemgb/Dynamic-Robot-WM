#!/usr/bin/env python3
"""Render development examples with the real Panda and Robotiq grippers.

This tool is deliberately separate from the release generator.  It proves the
corrected control boundary before ``source_mujoco`` is admitted to the v2
pipeline:

* robot qpos/qvel may be initialized once, then motion is actuator-only;
* object qpos/qvel may be initialized once, then the object remains free;
* no equality/latch is introduced by this tool;
* runtime guards compare state immediately before and after every controller
  call and abort on any rewrite.

Outputs are development/QC evidence, not release-eligible training data.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "examples"
    / "free_contact_real_grippers_v1.yaml"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "examples"
    / "free_contact_real_grippers_v1"
)


@dataclass(frozen=True)
class Scenario:
    episode_id: str
    embodiment: str
    motion: str
    intended_branch: str
    seed: int
    scene_variant: str
    initialize_robot: str
    target_offset_m: tuple[float, float, float]
    close_control: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], list[Scenario]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("release_eligible") is not False:
        raise ValueError("The example suite must remain release_eligible: false")
    scenarios = []
    for raw in config.get("scenarios", []):
        scenarios.append(
            Scenario(
                episode_id=str(raw["id"]),
                embodiment=str(raw["embodiment"]),
                motion=str(raw["motion"]),
                intended_branch=str(raw["intended_branch"]),
                seed=int(raw["seed"]),
                scene_variant=str(raw["scene_variant"]),
                initialize_robot=str(raw["initialize_robot"]),
                target_offset_m=tuple(
                    float(value) for value in raw.get("target_offset_m", (0, 0, 0))
                ),
                close_control=float(raw["close_control"]),
            )
        )
    if not scenarios:
        raise ValueError("Example suite has no scenarios")
    return config, scenarios


def _install_source_root(source_root: Path) -> None:
    source_root = source_root.resolve()
    required = (
        "scripts_mujoco/scene_builder.py",
        "scripts_mujoco/controller.py",
        "robotiq_catch/controller.py",
        "third_party/mujoco_menagerie/franka_emika_panda/panda.xml",
        "third_party/mujoco_menagerie/robotiq_2f85/2f85.xml",
    )
    missing = [name for name in required if not (source_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source root is missing: {', '.join(missing)}")
    sys.path.insert(0, str(source_root))


def _sample_for_scenario(scenario: Scenario, *, fps: int, duration_s: float):
    from scripts_mujoco.scene_builder import sample_episode
    from scripts_mujoco.utils import sphere_mass

    rng = np.random.default_rng(scenario.seed)
    sample = sample_episode(
        rng,
        scenario.scene_variant,
        seed=scenario.seed,
        fps=fps,
        duration=duration_s,
    )
    radius = 0.025
    density = sample.ball_mass / sphere_mass(sample.ball_radius, 1.0)
    target_xy = rng.uniform((0.44, -0.035), (0.50, 0.035))
    flight_time = float(rng.uniform(0.42, 0.46))
    close_z = sample.catch_center_z + 0.055
    gravity = abs(float(sample.gravity[2]))
    if scenario.motion == "vertical_drop":
        start_xy = target_xy.copy()
    elif scenario.motion == "lateral_projectile":
        direction = -1.0 if scenario.seed % 2 else 1.0
        start_xy = target_xy + np.array((0.090 * direction, -0.055 * direction))
    else:
        raise ValueError(f"unknown motion {scenario.motion!r}")
    velocity_xy = (target_xy - start_xy) / flight_time
    start_z = close_z + 0.5 * gravity * flight_time**2
    return dataclasses.replace(
        sample,
        ball_radius=radius,
        ball_mass=sphere_mass(radius, density),
        ball_initial_position=(float(start_xy[0]), float(start_xy[1]), float(start_z)),
        ball_initial_velocity=(float(velocity_xy[0]), float(velocity_xy[1]), 0.0),
        release_time_s=0.0,
        planned_intercept_time_s=flight_time,
        planned_intercept_position=(
            float(target_xy[0]),
            float(target_xy[1]),
            float(sample.catch_center_z),
        ),
        interception_subfamily=scenario.motion,
        offscreen_width=1280,
        offscreen_height=720,
        scene_randomization_level="clean",
    )


def _controller_classes():
    from robotiq_catch.controller import (
        PlannedRobotiqCatchController,
        RobotiqCatchPlan,
    )
    from scripts_mujoco.controller import MujocoInterceptionController
    from scripts_mujoco.utils import smoothstep

    class ActuatedPandaController(MujocoInterceptionController):
        def __init__(self, *args, scenario: Scenario, **kwargs):
            super().__init__(*args, **kwargs)
            self.scenario = scenario
            self.last_ctrl = np.zeros(8, dtype=np.float64)

        def prepare(self) -> None:
            super().prepare()
            offset = np.asarray(self.scenario.target_offset_m, dtype=np.float64)
            if np.any(offset):
                arm_q = self._solve_ik(
                    self.intercept_position + offset + np.array((0.0, 0.0, -0.075))
                )
                self.open_q = np.r_[arm_q, (0.04, 0.04)]
                closed_finger = float(
                    np.clip(0.85 * self.sample.ball_radius, 0.002, 0.04)
                )
                self.closed_q = np.r_[arm_q, (closed_finger, closed_finger)]
            initial_q = (
                self.open_q
                if self.scenario.initialize_robot == "intercept_pose"
                else self.home_q
            )
            self.data.qpos[self.ids.qpos_adrs] = initial_q
            self.data.qvel[self.ids.qvel_adrs] = 0.0
            self._set_ball_pose(
                self.sample.ball_initial_position, self.sample.ball_initial_velocity
            )
            self.data.ctrl[:7] = initial_q[:7]
            self.data.ctrl[7] = 255.0
            self.last_ctrl = self.data.ctrl.copy()
            mujoco.mj_forward(self.model, self.data)

        def before_step(self, t: float) -> None:
            assert self.open_q is not None and self.closed_q is not None
            if self.scenario.intended_branch == "no_op":
                arm_target = self.home_q[:7]
                finger_target = 255.0
            else:
                should_close = bool(
                    self.predicted_close_time_s is not None
                    and t >= self.predicted_close_time_s
                )
                self.closed = self.closed or should_close
                if self.scenario.initialize_robot == "intercept_pose":
                    arm_target = self.open_q[:7]
                else:
                    arrival = self.arm_arrival_time_s or 0.45
                    alpha = smoothstep(t / arrival)
                    arm_target = (
                        (1.0 - alpha) * self.home_q[:7]
                        + alpha * self.open_q[:7]
                    )
                finger_target = (
                    self.scenario.close_control if self.closed else 255.0
                )
            self.data.ctrl[:7] = arm_target
            self.data.ctrl[7] = finger_target
            self.last_ctrl = self.data.ctrl.copy()

    class ActuatedRobotiqController(PlannedRobotiqCatchController):
        def __init__(self, *args, close_control: float, **kwargs):
            self.close_control = float(close_control)
            super().__init__(*args, **kwargs)

        def _apply_qpos_command(self, q_cmd: np.ndarray, *, closed: bool) -> None:
            self.last_cmd = np.asarray(q_cmd, dtype=np.float64).copy()
            self.data.ctrl[:7] = self.last_cmd[:7]
            self.data.ctrl[7] = self.close_control if closed else 0.0

        def prepare(self) -> None:
            super().prepare()
            initial_q = (
                self.open_q
                if self.plan.branch != "wrong_action"
                and self.plan.branch_mode == "prepositioned"
                else self.home_q
            )
            self.data.qpos[self.ids.qpos_adrs] = initial_q
            self.data.qvel[self.ids.qvel_adrs] = 0.0
            self._set_ball_pose(
                self.sample.ball_initial_position, self.sample.ball_initial_velocity
            )
            self.data.ctrl[:7] = initial_q[:7]
            self.data.ctrl[7] = 0.0
            self.last_cmd = np.asarray(initial_q, dtype=np.float64).copy()
            mujoco.mj_forward(self.model, self.data)

    return ActuatedPandaController, ActuatedRobotiqController, RobotiqCatchPlan


def _build_rollout(scenario: Scenario, sample):
    from scripts_mujoco.scene_builder import (
        build_episode,
        build_robotiq_thick_pad_episode,
    )

    Panda, Robotiq, RobotiqPlan = _controller_classes()
    if scenario.embodiment == "franka_hand":
        bundle = build_episode(sample)
        controller = Panda(
            bundle.model,
            bundle.data,
            sample,
            reach_lead_time_s=0.12,
            close_lead_time_s=0.06,
            scenario=scenario,
        )
        secondary_camera = "closeup_camera"
    elif scenario.embodiment == "robotiq_2f85_thick_pad":
        bundle = build_robotiq_thick_pad_episode(sample)
        if scenario.intended_branch == "no_op":
            plan = RobotiqPlan(
                branch="wrong_action", branch_mode="idle", disable_latch=True
            )
        else:
            plan = RobotiqPlan(
                branch=scenario.intended_branch,
                branch_mode="actuator_only",
                lateral_miss=(
                    scenario.target_offset_m[0], scenario.target_offset_m[1]
                ),
                target_offset=(0.0, 0.0, scenario.target_offset_m[2]),
                close_lead_time_s=0.04,
                reach_lead_time_s=0.12,
                disable_latch=True,
            )
        controller = Robotiq(
            bundle.model,
            bundle.data,
            sample,
            plan,
            close_control=scenario.close_control,
        )
        secondary_camera = "side_camera"
    else:
        raise ValueError(f"unsupported embodiment {scenario.embodiment!r}")
    controller.prepare()
    return bundle, controller, secondary_camera


def _contact_observation(bundle, controller, scenario: Scenario) -> dict[str, Any]:
    model, data = bundle.model, bundle.data
    ball_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "catch_ball_geom"
    )
    left = right = total = 0
    maximum_penetration = 0.0
    if scenario.embodiment == "franka_hand":
        left_bodies = {int(controller.ids.left_finger_body)}
        right_bodies = {int(controller.ids.right_finger_body)}
        center = controller._fingertip_pad_center()
        retention_xy, retention_z = 0.04, 0.05
    else:
        left_geoms = set(int(value) for value in controller.ids.left_pad_geoms)
        right_geoms = set(int(value) for value in controller.ids.right_pad_geoms)
        center = controller._pad_center()
        retention_xy, retention_z = 0.075, 0.075
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 != ball_geom and contact.geom2 != ball_geom:
            continue
        total += 1
        other = int(contact.geom2 if contact.geom1 == ball_geom else contact.geom1)
        if scenario.embodiment == "franka_hand":
            body = int(model.geom_bodyid[other])
            is_left, is_right = body in left_bodies, body in right_bodies
        else:
            is_left, is_right = other in left_geoms, other in right_geoms
        left += int(is_left)
        right += int(is_right)
        if is_left or is_right:
            maximum_penetration = max(
                maximum_penetration, max(0.0, -float(contact.dist))
            )
    ball_position = data.xpos[controller.ids.ball_body].copy()
    distance = float(np.linalg.norm(ball_position - center))
    retained = bool(
        np.linalg.norm(ball_position[:2] - center[:2]) <= retention_xy
        and abs(ball_position[2] - center[2]) <= retention_z
    )
    return {
        "total_contacts": total,
        "left_contacts": left,
        "right_contacts": right,
        "bilateral_contact": bool(left and right),
        "maximum_finger_penetration_m": maximum_penetration,
        "grasp_center_distance_m": distance,
        "retained": retained,
    }


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.table(
        {
            "timestamp_s": [row["timestamp_s"] for row in rows],
            "robot_qpos": [row["robot_qpos"] for row in rows],
            "robot_qvel": [row["robot_qvel"] for row in rows],
            "action_ctrl": [row["action_ctrl"] for row in rows],
            "object_position_m": [row["object_position_m"] for row in rows],
            "object_quaternion_wxyz": [row["object_quaternion_wxyz"] for row in rows],
            "object_linear_velocity_m_s": [
                row["object_linear_velocity_m_s"] for row in rows
            ],
            "object_angular_velocity_rad_s": [
                row["object_angular_velocity_rad_s"] for row in rows
            ],
            "left_finger_contacts": [row["left_finger_contacts"] for row in rows],
            "right_finger_contacts": [row["right_finger_contacts"] for row in rows],
            "bilateral_contact": [row["bilateral_contact"] for row in rows],
            "maximum_finger_penetration_m": [
                row["maximum_finger_penetration_m"] for row in rows
            ],
            "grasp_center_distance_m": [
                row["grasp_center_distance_m"] for row in rows
            ],
            "retained": [row["retained"] for row in rows],
        }
    )
    pq.write_table(table, path, compression="zstd")


def _run_scenario(
    scenario: Scenario,
    output_root: Path,
    *,
    fps: int,
    duration_s: float,
    width: int,
    height: int,
) -> tuple[dict[str, Any], np.ndarray]:
    sample = _sample_for_scenario(scenario, fps=fps, duration_s=duration_s)
    bundle, controller, secondary_camera = _build_rollout(scenario, sample)
    model, data = bundle.model, bundle.data
    episode_root = output_root / "episodes" / scenario.episode_id
    episode_root.mkdir(parents=True)
    renderers = {
        "main": mujoco.Renderer(model, height=height, width=width),
        "secondary": mujoco.Renderer(model, height=height, width=width),
    }
    cameras = {"main": "main_camera", "secondary": secondary_camera}
    frames: dict[str, list[np.ndarray]] = {"main": [], "secondary": []}
    rows: list[dict[str, Any]] = []
    timestep = float(sample.timestep)
    frame_every = max(1, int(round(1.0 / (fps * timestep))))
    total_steps = int(round(duration_s / timestep))
    object_rewrites = robot_rewrites = 0
    max_retained_steps = retained_steps = bilateral_steps = 0
    min_distance = float("inf")
    max_penetration = 0.0

    def record(timestamp: float) -> dict[str, Any]:
        nonlocal retained_steps, max_retained_steps, bilateral_steps
        nonlocal min_distance, max_penetration
        contact = _contact_observation(bundle, controller, scenario)
        retained_steps = retained_steps + 1 if contact["retained"] else 0
        max_retained_steps = max(max_retained_steps, retained_steps)
        bilateral_steps += int(contact["bilateral_contact"])
        min_distance = min(min_distance, contact["grasp_center_distance_m"])
        max_penetration = max(
            max_penetration, contact["maximum_finger_penetration_m"]
        )
        qadr, vadr = controller.ids.ball_qpos_adr, controller.ids.ball_qvel_adr
        rows.append(
            {
                "timestamp_s": float(timestamp),
                "robot_qpos": data.qpos[controller.ids.qpos_adrs].tolist(),
                "robot_qvel": data.qvel[controller.ids.qvel_adrs].tolist(),
                "action_ctrl": data.ctrl.tolist(),
                "object_position_m": data.qpos[qadr : qadr + 3].tolist(),
                "object_quaternion_wxyz": data.qpos[qadr + 3 : qadr + 7].tolist(),
                "object_linear_velocity_m_s": data.qvel[vadr : vadr + 3].tolist(),
                "object_angular_velocity_rad_s": data.qvel[vadr + 3 : vadr + 6].tolist(),
                "left_finger_contacts": contact["left_contacts"],
                "right_finger_contacts": contact["right_contacts"],
                "bilateral_contact": contact["bilateral_contact"],
                "maximum_finger_penetration_m": contact[
                    "maximum_finger_penetration_m"
                ],
                "grasp_center_distance_m": contact["grasp_center_distance_m"],
                "retained": contact["retained"],
            }
        )
        return contact

    try:
        record(0.0)
        for view, renderer in renderers.items():
            renderer.update_scene(data, camera=cameras[view])
            frames[view].append(renderer.render().copy())
        for step in range(1, total_steps + 1):
            timestamp = step * timestep
            qadr, vadr = controller.ids.ball_qpos_adr, controller.ids.ball_qvel_adr
            object_before = np.r_[
                data.qpos[qadr : qadr + 7], data.qvel[vadr : vadr + 6]
            ].copy()
            robot_before = np.r_[
                data.qpos[controller.ids.qpos_adrs],
                data.qvel[controller.ids.qvel_adrs],
            ].copy()
            controller.before_step(float(timestamp))
            object_after_control = np.r_[
                data.qpos[qadr : qadr + 7], data.qvel[vadr : vadr + 6]
            ]
            robot_after_control = np.r_[
                data.qpos[controller.ids.qpos_adrs],
                data.qvel[controller.ids.qvel_adrs],
            ]
            object_rewrites += int(
                not np.array_equal(object_before, object_after_control)
            )
            robot_rewrites += int(not np.array_equal(robot_before, robot_after_control))
            if object_rewrites or robot_rewrites:
                raise RuntimeError(
                    f"post-initialization state rewrite in {scenario.episode_id}: "
                    f"object={object_rewrites}, robot={robot_rewrites}"
                )
            mujoco.mj_step(model, data)
            if scenario.embodiment == "robotiq_2f85_thick_pad":
                controller.after_step(float(timestamp))
            record(float(timestamp))
            if step % frame_every == 0:
                for view, renderer in renderers.items():
                    renderer.update_scene(data, camera=cameras[view])
                    frames[view].append(renderer.render().copy())
    finally:
        for renderer in renderers.values():
            renderer.close()

    retained_duration = max_retained_steps * timestep
    objective_success = bool(
        retained_duration >= 0.40
        and bilateral_steps >= int(round(0.05 / timestep))
        and max_penetration <= 0.002
    )
    if scenario.intended_branch == "no_op":
        actual_outcome = "no-op"
    elif objective_success:
        actual_outcome = "success"
    elif bilateral_steps or any(
        row["left_finger_contacts"] or row["right_finger_contacts"] for row in rows
    ):
        actual_outcome = "contact failure"
    elif min_distance <= 0.16:
        actual_outcome = "near miss"
    else:
        actual_outcome = "miss"

    video_paths = {}
    for view in ("main", "secondary"):
        path = episode_root / f"{view}.mp4"
        _write_video(path, frames[view], fps)
        video_paths[view] = {
            "path": str(path),
            "sha256": _sha256(path),
            "camera": cameras[view],
            "frame_count": len(frames[view]),
        }
    parquet_path = episode_root / "trajectory.parquet"
    _write_parquet(parquet_path, rows)
    metadata = {
        "schema_version": "dynamic-robot-free-contact-example/v1",
        "episode_id": scenario.episode_id,
        "development_only": True,
        "release_eligible": False,
        "release_blocker": (
            "canonical source_mujoco v2 normalizer, objective replay, and admission "
            "artifacts are not yet approved"
        ),
        "embodiment": scenario.embodiment,
        "robot_model": (
            "franka_panda"
            if scenario.embodiment == "franka_hand"
            else "franka_panda_nohand_plus_robotiq_2f85"
        ),
        "motion": scenario.motion,
        "intended_branch": scenario.intended_branch,
        "actual_outcome_class": actual_outcome,
        "objective_success": objective_success,
        "objective_evaluator": "free_contact_retention_v1",
        "objective_evidence": {
            "maximum_contiguous_retention_s": retained_duration,
            "bilateral_contact_steps": bilateral_steps,
            "minimum_grasp_center_distance_m": min_distance,
            "maximum_finger_penetration_m": max_penetration,
        },
        "controller": {
            "post_initialization_robot_motion": "MuJoCo actuator controls only",
            "post_initialization_object_motion": "native MuJoCo freejoint dynamics",
            "object_state_rewrite_count": object_rewrites,
            "robot_state_rewrite_count": robot_rewrites,
            "latch_or_weld": False,
            "equality_capture": False,
            "close_control": scenario.close_control,
        },
        "seed": scenario.seed,
        "scene_variant": scenario.scene_variant,
        "target_offset_m": list(scenario.target_offset_m),
        "initial_object_position_m": list(sample.ball_initial_position),
        "initial_object_velocity_m_s": list(sample.ball_initial_velocity),
        "physics": {
            "gravity_m_s2": list(sample.gravity),
            "object_mass_kg": sample.ball_mass,
            "object_radius_m": sample.ball_radius,
            "timestep_s": timestep,
        },
        "videos": video_paths,
        "trajectory": {
            "path": str(parquet_path),
            "sha256": _sha256(parquet_path),
            "row_count": len(rows),
            "rate_hz": round(1.0 / timestep),
        },
    }
    metadata_path = episode_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    preview_frame = frames["main"][min(len(frames["main"]) - 1, int(0.55 * fps))]
    return metadata, preview_frame


def _contact_sheet(
    output_path: Path, results: list[dict[str, Any]], previews: list[np.ndarray]
) -> None:
    tile_width, tile_height, label_height = 416, 240, 42
    columns = 2
    rows = (len(previews) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (metadata, frame) in enumerate(zip(results, previews)):
        image = Image.fromarray(frame).resize((tile_width, tile_height))
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + label_height)
        canvas.paste(image, (x, y))
        label = (
            f"{metadata['episode_id']}\n"
            f"actual={metadata['actual_outcome_class']}  "
            f"success={metadata['objective_success']}"
        )
        draw.text((x + 6, y + tile_height + 3), label, fill="black")
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    config, scenarios = _load_config(arguments.config)
    source_root = Path(arguments.source_root or config["source_root"]).resolve()
    output = arguments.output.resolve()
    if output.exists():
        if not arguments.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _install_source_root(source_root)
    fps = int(config["fps"])
    duration_s = float(config["duration_s"])
    width, height = (int(value) for value in config["resolution"])
    source_files = (
        source_root / "scripts_mujoco" / "scene_builder.py",
        source_root / "scripts_mujoco" / "controller.py",
        source_root / "robotiq_catch" / "controller.py",
        source_root
        / "third_party"
        / "mujoco_menagerie"
        / "franka_emika_panda"
        / "panda.xml",
        source_root
        / "third_party"
        / "mujoco_menagerie"
        / "robotiq_2f85"
        / "2f85.xml",
    )
    before_hashes = {str(path): _sha256(path) for path in source_files}
    results: list[dict[str, Any]] = []
    previews: list[np.ndarray] = []
    for index, scenario in enumerate(scenarios, 1):
        print(
            f"[{index}/{len(scenarios)}] {scenario.episode_id} "
            f"({scenario.embodiment})",
            flush=True,
        )
        metadata, preview = _run_scenario(
            scenario,
            output,
            fps=fps,
            duration_s=duration_s,
            width=width,
            height=height,
        )
        results.append(metadata)
        previews.append(preview)
        print(
            f"  actual={metadata['actual_outcome_class']} "
            f"objective_success={metadata['objective_success']} "
            f"max_penetration_mm="
            f"{1000 * metadata['objective_evidence']['maximum_finger_penetration_m']:.3f}",
            flush=True,
        )
    after_hashes = {str(path): _sha256(path) for path in source_files}
    if before_hashes != after_hashes:
        raise RuntimeError("source files changed during generation")
    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in results),
        encoding="utf-8",
    )
    _contact_sheet(output / "contact_sheet.png", results, previews)
    suite_report = {
        "schema_version": config["schema_version"],
        "suite_id": config["suite_id"],
        "development_only": True,
        "release_eligible": False,
        "episode_count": len(results),
        "logical_duration_s": len(results) * duration_s,
        "encoded_stream_duration_s": len(results) * duration_s * 2,
        "objective_success_count": sum(
            int(result["objective_success"]) for result in results
        ),
        "actual_outcome_counts": {
            outcome: sum(
                result["actual_outcome_class"] == outcome for result in results
            )
            for outcome in sorted({result["actual_outcome_class"] for result in results})
        },
        "all_controller_rewrite_guards_passed": all(
            result["controller"]["object_state_rewrite_count"] == 0
            and result["controller"]["robot_state_rewrite_count"] == 0
            for result in results
        ),
        "source_tree_unchanged": before_hashes == after_hashes,
        "source_sha256": before_hashes,
        "manifest": str(manifest_path),
        "contact_sheet": str(output / "contact_sheet.png"),
        "reproduction_command": (
            f"MUJOCO_GL=egl {Path(sys.executable)} {Path(__file__).resolve()} "
            f"--config {arguments.config.resolve()} --output {output} --overwrite"
        ),
    }
    (output / "suite_report.json").write_text(
        json.dumps(suite_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(suite_report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
