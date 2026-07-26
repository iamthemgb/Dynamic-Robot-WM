#!/usr/bin/env python3
"""Generate actuator-only, free-object examples for the other rigid scenarios.

The suite covers catch/retain, rolling/sliding, rebound, finite-surface mode
transitions, and robot-induced deflection.  It is development evidence, not a
release generator.  Runtime guards abort if a controller changes robot or
object qpos/qvel after initialization.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/examples/dynamic_scenarios_real_grippers_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/examples/dynamic_scenarios_real_grippers_v1"
PANDA_QPOS_LOWER = np.asarray(
    (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0)
)
PANDA_QPOS_UPPER = np.asarray(
    (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04)
)


@dataclass(frozen=True)
class Scenario:
    episode_id: str
    scenario_type: str
    category: str
    seed: int
    scene_variant: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], list[Scenario]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("release_eligible") is not False:
        raise ValueError("scenario examples must remain release_eligible: false")
    scenarios = [
        Scenario(
            episode_id=str(raw["id"]),
            scenario_type=str(raw["type"]),
            category=str(raw["category"]),
            seed=int(raw["seed"]),
            scene_variant=str(raw["scene_variant"]),
        )
        for raw in config.get("scenarios", [])
    ]
    if not scenarios:
        raise ValueError("no scenarios configured")
    return config, scenarios


def install_source_root(source_root: Path) -> tuple[Path, ...]:
    files = (
        source_root / "scripts_mujoco/scene_builder.py",
        source_root / "scripts_mujoco/controller.py",
        source_root / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml",
    )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {', '.join(missing)}")
    sys.path.insert(0, str(source_root))
    return files


def surface(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    *,
    euler: tuple[float, float, float] | None = None,
    friction: str = "1.0 0.02 0.001",
    solref: str = "0.004 0.7",
    color: tuple[float, float, float, float] = (0.24, 0.36, 0.50, 1.0),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "type": "box",
        "pos": pos,
        "size": size,
        "friction": friction,
        "solref": solref,
        "solimp": "0.96 0.995 0.0005",
        "priority": "2",
        "rgba": color,
    }
    if euler is not None:
        result["euler"] = euler
    return result


def scenario_parameters(scenario: Scenario) -> dict[str, Any]:
    table = surface(
        "motion_table",
        (0.52, 0.0, 0.80),
        (0.54, 0.36, 0.02),
        friction="0.65 0.015 0.001",
        color=(0.20, 0.34, 0.46, 1.0),
    )
    params: dict[str, Any] = {
        "radius": 0.025,
        "ball_friction": (0.65, 0.01, 0.001),
        "surfaces": (table,),
        "surface_names": ("motion_table",),
        "controller_mode": "observer",
        "initial_grasp_center": None,
        "target_grasp_center": None,
        "action_start_s": 0.0,
        "action_end_s": 0.6,
        "close_control": 150.0,
        "focus": (0.52, 0.0, 0.92),
        "camera_distance": 1.55,
        "camera_azimuth": 125.0,
        "camera_elevation": -24.0,
        "transport_offset": (0.0, 0.0, 0.0),
        "transport_start_s": 0.85,
        "transport_end_s": 1.55,
        "table_top_z": 0.82,
        "edge_x": None,
        "catch_center_z": None,
        "timestep_s": 1.0 / 240.0,
        "joint_start_delta": (0.0,) * 7,
        "joint_target_delta": (0.0,) * 7,
    }
    kind = scenario.scenario_type
    if kind in {"catch_smooth_transport", "catch_abrupt_brake"}:
        params.update(
            controller_mode="catch_transport",
            ball_position=(0.47, 0.0, 2.18),
            ball_velocity=(0.0, 0.0, 0.0),
            ball_angular_velocity=(0.0, 0.0, 0.0),
            surfaces=(),
            surface_names=(),
            transport_offset=(0.13, 0.07 if kind.endswith("brake") else 0.0, 0.04),
            transport_start_s=0.82,
            transport_end_s=1.12 if kind.endswith("brake") else 1.62,
            focus=(0.50, 0.0, 1.28),
            catch_center_z=1.24,
        )
    elif kind == "rolling_interception":
        params.update(
            ball_position=(0.84, 0.0, 1.277),
            ball_velocity=(-0.50, 0.0, 0.0),
            ball_angular_velocity=(0.0, -20.0, 0.0),
            ball_friction=(0.12, 0.005, 0.0005),
            surfaces=(
                surface(
                    "motion_table",
                    (0.62, 0.0, 1.232),
                    (0.22, 0.08, 0.02),
                    friction="0.12 0.005 0.0005",
                    color=(0.20, 0.34, 0.46, 1.0),
                ),
            ),
            controller_mode="joint_sweep",
            initial_grasp_center=None,
            target_grasp_center=None,
            joint_start_delta=(-0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            joint_target_delta=(0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            action_start_s=0.50,
            action_end_s=0.85,
            focus=(0.55, 0.0, 1.30),
        )
    elif kind == "sliding_redirect":
        params.update(
            ball_position=(0.84, 0.0, 1.277),
            ball_velocity=(-0.55, 0.0, 0.0),
            ball_angular_velocity=(0.0, 0.0, 0.0),
            ball_friction=(0.06, 0.003, 0.0003),
            surfaces=(
                surface(
                    "motion_table",
                    (0.62, 0.0, 1.232),
                    (0.22, 0.08, 0.02),
                    friction="0.06 0.003 0.0003",
                    color=(0.28, 0.30, 0.42, 1.0),
                ),
            ),
            controller_mode="joint_sweep",
            initial_grasp_center=None,
            target_grasp_center=None,
            joint_start_delta=(-0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            joint_target_delta=(0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            action_start_s=0.45,
            action_end_s=0.75,
            focus=(0.55, 0.0, 1.30),
        )
    elif kind == "table_rebound":
        params.update(
            ball_position=(0.50, 0.0, 1.52),
            ball_velocity=(0.08, 0.0, -0.10),
            ball_angular_velocity=(0.0, 0.0, 0.0),
            surfaces=(
                surface(
                    "bounce_pad",
                    (0.50, 0.0, 0.80),
                    (0.40, 0.34, 0.02),
                    friction="0.55 0.01 0.001",
                    solref="0.002 0.70",
                    color=(0.48, 0.30, 0.22, 1.0),
                ),
            ),
            surface_names=("bounce_pad",),
            focus=(0.50, 0.0, 1.10),
            timestep_s=1.0 / 1000.0,
        )
    elif kind == "wall_rebound":
        params.update(
            ball_position=(0.30, 0.0, 0.846),
            ball_velocity=(0.92, 0.0, 0.0),
            ball_angular_velocity=(0.0, 0.0, 0.0),
            surfaces=(
                table,
                surface(
                    "rebound_wall",
                    (0.82, 0.0, 1.03),
                    (0.02, 0.34, 0.22),
                    friction="0.45 0.01 0.001",
                    solref="0.004 0.25",
                    color=(0.55, 0.24, 0.20, 1.0),
                ),
            ),
            surface_names=("motion_table", "rebound_wall"),
        )
    elif kind == "roll_off_edge":
        edge_x = 0.82
        params.update(
            ball_position=(0.36, 0.0, 0.986),
            ball_velocity=(0.78, 0.0, 0.0),
            ball_angular_velocity=(0.0, 31.2, 0.0),
            ball_friction=(0.20, 0.005, 0.0005),
            surfaces=(
                surface(
                    "finite_platform",
                    (0.50, 0.0, 0.94),
                    (0.32, 0.30, 0.02),
                    friction="0.20 0.005 0.0005",
                    color=(0.24, 0.42, 0.30, 1.0),
                ),
            ),
            surface_names=("finite_platform",),
            table_top_z=0.96,
            edge_x=edge_x,
            focus=(0.54, 0.0, 1.00),
        )
    elif kind == "ramp_to_flight":
        theta = -0.15
        center_x, center_z, half_x, half_z = 0.50, 0.96, 0.34, 0.02
        start_x = 0.22
        top_z = center_z + (half_z - math.sin(theta) * (start_x - center_x)) / math.cos(theta)
        speed = 1.55
        params.update(
            ball_position=(start_x, 0.0, top_z + 0.025 / math.cos(theta) + 0.001),
            ball_velocity=(speed * math.cos(theta), 0.0, -speed * math.sin(theta)),
            ball_angular_velocity=(0.0, speed / 0.025, 0.0),
            ball_friction=(0.28, 0.005, 0.0005),
            surfaces=(
                surface(
                    "launch_ramp",
                    (center_x, 0.0, center_z),
                    (half_x, 0.28, half_z),
                    euler=(0.0, theta, 0.0),
                    friction="0.28 0.005 0.0005",
                    color=(0.44, 0.32, 0.18, 1.0),
                ),
            ),
            surface_names=("launch_ramp",),
            edge_x=center_x + half_x * math.cos(theta),
            focus=(0.55, 0.0, 1.00),
        )
    elif kind in {"hand_deflect_left", "hand_deflect_right"}:
        reverse = kind.endswith("right")
        params.update(
            ball_position=(0.84, 0.0, 1.277),
            ball_velocity=(-0.55, 0.0, 0.0),
            ball_angular_velocity=(0.0, 0.0, 0.0),
            ball_friction=(0.06, 0.003, 0.0003),
            surfaces=(
                surface(
                    "motion_table",
                    (0.62, 0.0, 1.232),
                    (0.22, 0.08, 0.02),
                    friction="0.06 0.003 0.0003",
                    color=(0.25, 0.38, 0.48, 1.0),
                ),
            ),
            controller_mode="joint_sweep",
            initial_grasp_center=None,
            target_grasp_center=None,
            joint_start_delta=((0.12 if reverse else -0.12), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            joint_target_delta=((-0.12 if reverse else 0.12), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            action_start_s=0.45,
            action_end_s=0.75,
            focus=(0.55, 0.0, 1.30),
        )
    else:
        raise ValueError(f"unsupported scenario type {kind!r}")
    return params


def make_sample(scenario: Scenario, params: dict[str, Any], *, fps: int, duration_s: float):
    from scripts_mujoco.scene_builder import sample_episode
    from scripts_mujoco.utils import sphere_mass

    rng = np.random.default_rng(scenario.seed)
    sample = sample_episode(
        rng, scenario.scene_variant, seed=scenario.seed, fps=fps, duration=duration_s
    )
    radius = float(params["radius"])
    density = sample.ball_mass / sphere_mass(sample.ball_radius, 1.0)
    return dataclasses.replace(
        sample,
        ball_radius=radius,
        ball_mass=sphere_mass(radius, density),
        ball_initial_position=tuple(params["ball_position"]),
        ball_initial_velocity=tuple(params["ball_velocity"]),
        release_time_s=0.0,
        catch_center_z=(
            float(params["ball_position"][2])
            if params["catch_center_z"] is None
            else float(params["catch_center_z"])
        ),
        interception_surface_specs=tuple(params["surfaces"]),
        surface_contact_groups=tuple(
            {"name": name, "geoms": [name]} for name in params["surface_names"]
        ),
        expected_contact_sequence=tuple(params["surface_names"]),
        scene_randomization_level="clean",
        offscreen_width=1280,
        offscreen_height=720,
        timestep=float(params["timestep_s"]),
    )


class GenericPandaController:
    def __init__(self, model, data, sample, params):
        from scripts_mujoco.controller import MujocoCatchController

        self.base = MujocoCatchController(model, data, sample)
        self.model, self.data, self.sample = model, data, sample
        self.ids = self.base.ids
        self.params = params
        self.home_q = self.base.home_q.copy()
        self.start_q = self.home_q[:7].copy()
        self.target_q = self.home_q[:7].copy()
        self.last_ctrl = np.zeros(model.nu, dtype=np.float64)

    def _solve(self, grasp_center) -> np.ndarray:
        target = np.asarray(grasp_center, dtype=np.float64) - np.array((0.0, 0.0, 0.075))
        return self.base._solve_ik(target)

    def prepare(self) -> None:
        if self.params["initial_grasp_center"] is not None:
            self.start_q = self._solve(self.params["initial_grasp_center"])
        if self.params["target_grasp_center"] is not None:
            self.target_q = self._solve(self.params["target_grasp_center"])
        if self.params["controller_mode"] == "joint_sweep":
            self.start_q = self.home_q[:7] + np.asarray(
                self.params["joint_start_delta"], dtype=np.float64
            )
            self.target_q = self.home_q[:7] + np.asarray(
                self.params["joint_target_delta"], dtype=np.float64
            )
        initial_arm = (
            self.start_q
            if self.params["controller_mode"] in {"sweep", "joint_sweep"}
            else self.home_q[:7]
        )
        finger_q = 0.04 * float(self.params["close_control"]) / 255.0
        initial_q = np.r_[initial_arm, (finger_q, finger_q)]
        self.data.qpos[self.ids.qpos_adrs] = initial_q
        self.data.qvel[self.ids.qvel_adrs] = 0.0
        self.base._set_ball_pose(
            self.sample.ball_initial_position, self.sample.ball_initial_velocity
        )
        vadr = self.ids.ball_qvel_adr
        self.data.qvel[vadr + 3 : vadr + 6] = self.params["ball_angular_velocity"]
        self.data.ctrl[:7] = initial_arm
        self.data.ctrl[7] = float(self.params["close_control"])
        self.last_ctrl = self.data.ctrl.copy()
        mujoco.mj_forward(self.model, self.data)

    def before_step(self, t: float) -> None:
        if self.params["controller_mode"] in {"sweep", "joint_sweep"}:
            start, end = self.params["action_start_s"], self.params["action_end_s"]
            alpha = float(np.clip((t - start) / max(end - start, 1e-6), 0.0, 1.0))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target = (1.0 - alpha) * self.start_q + alpha * self.target_q
        else:
            target = self.home_q[:7]
        self.data.ctrl[:7] = target
        self.data.ctrl[7] = float(self.params["close_control"])
        self.last_ctrl = self.data.ctrl.copy()

    def grasp_center(self) -> np.ndarray:
        return self.base._fingertip_pad_center()


class CatchTransportController:
    def __init__(self, model, data, sample, params):
        from scripts_mujoco.controller import MujocoInterceptionController

        self.base = MujocoInterceptionController(
            model, data, sample, reach_lead_time_s=0.12, close_lead_time_s=0.06
        )
        self.model, self.data, self.sample = model, data, sample
        self.ids = self.base.ids
        self.params = params
        self.last_ctrl = np.zeros(model.nu, dtype=np.float64)
        self.transport_q: np.ndarray | None = None

    def prepare(self) -> None:
        self.base.prepare()
        assert self.base.open_q is not None
        target_center = self.base.intercept_position + np.asarray(
            self.params["transport_offset"], dtype=np.float64
        )
        self.transport_q = self.base._solve_ik(
            target_center - np.array((0.0, 0.0, 0.075))
        )
        self.data.qpos[self.ids.qpos_adrs] = self.base.open_q
        self.data.qvel[self.ids.qvel_adrs] = 0.0
        self.base._set_ball_pose(
            self.sample.ball_initial_position, self.sample.ball_initial_velocity
        )
        self.data.ctrl[:7] = self.base.open_q[:7]
        self.data.ctrl[7] = 255.0
        self.last_ctrl = self.data.ctrl.copy()
        mujoco.mj_forward(self.model, self.data)

    def before_step(self, t: float) -> None:
        assert self.base.open_q is not None and self.transport_q is not None
        start, end = self.params["transport_start_s"], self.params["transport_end_s"]
        alpha = float(np.clip((t - start) / max(end - start, 1e-6), 0.0, 1.0))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        arm_target = (1.0 - alpha) * self.base.open_q[:7] + alpha * self.transport_q
        close = bool(
            self.base.predicted_close_time_s is not None
            and t >= self.base.predicted_close_time_s
        )
        self.data.ctrl[:7] = arm_target
        self.data.ctrl[7] = float(self.params["close_control"] if close else 255.0)
        self.last_ctrl = self.data.ctrl.copy()

    def grasp_center(self) -> np.ndarray:
        return self.base._fingertip_pad_center()


def configure_model(model, params) -> tuple[int, dict[str, int]]:
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "catch_ball_geom")
    model.geom_friction[ball_geom] = np.asarray(params["ball_friction"])
    surface_ids = {}
    for name in params["surface_names"]:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise KeyError(f"surface geom missing: {name}")
        surface_ids[name] = int(geom_id)
    return int(ball_geom), surface_ids


def observe(model, data, controller, ball_geom: int, surface_ids: dict[str, int]):
    left_body = int(controller.ids.left_finger_body)
    right_body = int(controller.ids.right_finger_body)
    hand_body = int(controller.ids.hand_body)
    left = right = hand = 0
    surface_contacts: list[str] = []
    maximum_finger_penetration = maximum_surface_penetration = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 != ball_geom and contact.geom2 != ball_geom:
            continue
        other = int(contact.geom2 if contact.geom1 == ball_geom else contact.geom1)
        body = int(model.geom_bodyid[other])
        is_left, is_right, is_hand = (
            body == left_body,
            body == right_body,
            body == hand_body,
        )
        left += int(is_left)
        right += int(is_right)
        hand += int(is_hand)
        if is_left or is_right or is_hand:
            maximum_finger_penetration = max(
                maximum_finger_penetration, max(0.0, -float(contact.dist))
            )
        for name, geom_id in surface_ids.items():
            if other == geom_id:
                surface_contacts.append(name)
                maximum_surface_penetration = max(
                    maximum_surface_penetration, max(0.0, -float(contact.dist))
                )
    qadr, vadr = controller.ids.ball_qpos_adr, controller.ids.ball_qvel_adr
    ball_position = data.qpos[qadr : qadr + 3].copy()
    center = controller.grasp_center()
    retained = bool(
        np.linalg.norm(ball_position[:2] - center[:2]) <= 0.04
        and abs(ball_position[2] - center[2]) <= 0.05
    )
    return {
        "ball_position": ball_position,
        "ball_quaternion": data.qpos[qadr + 3 : qadr + 7].copy(),
        "ball_linear_velocity": data.qvel[vadr : vadr + 3].copy(),
        "ball_angular_velocity": data.qvel[vadr + 3 : vadr + 6].copy(),
        "grasp_center": center.copy(),
        "left_contacts": left,
        "right_contacts": right,
        "hand_contacts": hand,
        "end_effector_contact": bool(left or right or hand),
        "bilateral_contact": bool(left and right),
        "surface_contacts": sorted(set(surface_contacts)),
        "maximum_finger_penetration_m": maximum_finger_penetration,
        "maximum_surface_penetration_m": maximum_surface_penetration,
        "retained": retained,
    }


def evaluate(scenario: Scenario, params, rows: list[dict[str, Any]]) -> dict[str, Any]:
    velocities = np.asarray([row["ball_linear_velocity"] for row in rows])
    positions = np.asarray([row["ball_position"] for row in rows])
    robot_qpos = np.asarray([row["robot_qpos"] for row in rows])
    joint_limit_violation = np.maximum(PANDA_QPOS_LOWER - robot_qpos, 0.0) + np.maximum(
        robot_qpos - PANDA_QPOS_UPPER, 0.0
    )
    maximum_joint_limit_violation = float(joint_limit_violation.max())
    maximum_joint_step = float(np.abs(np.diff(robot_qpos[:, :7], axis=0)).max())
    finger_steps = sum(bool(row["left_contacts"] or row["right_contacts"]) for row in rows)
    end_effector_steps = sum(bool(row["end_effector_contact"]) for row in rows)
    bilateral_steps = sum(bool(row["bilateral_contact"]) for row in rows)
    surface_steps = {
        name: sum(name in row["surface_contacts"] for row in rows)
        for name in params["surface_names"]
    }
    max_finger_pen = max(row["maximum_finger_penetration_m"] for row in rows)
    max_surface_pen = max(row["maximum_surface_penetration_m"] for row in rows)
    max_retained = current = 0
    for row in rows:
        current = current + 1 if row["retained"] else 0
        max_retained = max(max_retained, current)
    dt = float(rows[1]["timestamp_s"] - rows[0]["timestamp_s"])
    retained_s = max_retained * dt
    initial_xy = velocities[0, :2]
    final_xy = velocities[-1, :2]
    initial_speed = float(np.linalg.norm(initial_xy))
    final_speed = float(np.linalg.norm(final_xy))
    surface_any = any(value > 0 for value in surface_steps.values())
    kind = scenario.scenario_type
    evidence: dict[str, Any] = {
        "finger_contact_steps": finger_steps,
        "end_effector_contact_steps": end_effector_steps,
        "bilateral_contact_steps": bilateral_steps,
        "surface_contact_steps": surface_steps,
        "maximum_finger_penetration_m": max_finger_pen,
        "maximum_surface_penetration_m": max_surface_pen,
        "maximum_contiguous_retention_s": retained_s,
        "initial_planar_speed_m_s": initial_speed,
        "final_planar_speed_m_s": final_speed,
        "minimum_z_m": float(positions[:, 2].min()),
        "maximum_z_m": float(positions[:, 2].max()),
        "minimum_x_m": float(positions[:, 0].min()),
        "maximum_x_m": float(positions[:, 0].max()),
        "maximum_upward_velocity_m_s": float(velocities[:, 2].max()),
        "minimum_vx_m_s": float(velocities[:, 0].min()),
        "maximum_vx_m_s": float(velocities[:, 0].max()),
        "maximum_joint_limit_violation_rad": maximum_joint_limit_violation,
        "maximum_joint_step_rad": maximum_joint_step,
    }
    if kind in {"catch_smooth_transport", "catch_abrupt_brake"}:
        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        evidence["object_displacement_m"] = displacement
        success = retained_s >= 0.40 and rows[-1]["retained"] and bilateral_steps > 10
    elif kind == "rolling_interception":
        success = end_effector_steps > 3 and final_speed < 0.55 * max(initial_speed, 1e-6)
    elif kind == "sliding_redirect":
        before = initial_xy / max(initial_speed, 1e-6)
        after_index = min(len(rows) - 1, next((i + 20 for i, row in enumerate(rows) if row["end_effector_contact"]), len(rows) - 1))
        after = velocities[after_index, :2]
        after /= max(float(np.linalg.norm(after)), 1e-6)
        turn_angle = math.degrees(math.acos(float(np.clip(np.dot(before, after), -1, 1))))
        evidence["direction_change_deg"] = turn_angle
        success = end_effector_steps > 3 and turn_angle >= 20.0
    elif kind == "table_rebound":
        success = surface_any and float(velocities[:, 2].max()) > 0.25
    elif kind == "wall_rebound":
        success = surface_steps.get("rebound_wall", 0) > 0 and float(velocities[:, 0].min()) < -0.15
    elif kind == "roll_off_edge":
        success = surface_any and float(positions[:, 0].max()) > float(params["edge_x"]) + 0.03 and float(positions[:, 2].min()) < float(params["table_top_z"]) - 0.08
    elif kind == "ramp_to_flight":
        success = surface_any and float(positions[:, 0].max()) > float(params["edge_x"]) + 0.04
    else:
        direction_changed = (
            np.sign(velocities[0, 0]) != np.sign(velocities[-1, 0])
            or abs(float(velocities[:, 1].max())) > 0.18
        )
        success = end_effector_steps > 3 and direction_changed
    physics_qc = (
        max_finger_pen <= 0.002
        and max_surface_pen <= 0.006
        and maximum_joint_limit_violation <= 1e-4
        and maximum_joint_step <= 0.10
    )
    objective_success = bool(success and physics_qc)
    if objective_success:
        outcome = "success"
    elif success:
        outcome = "partial success"
    elif end_effector_steps:
        outcome = "contact failure"
    elif surface_any:
        outcome = "partial success"
    else:
        outcome = "miss"
    return {
        "objective_success": objective_success,
        "actual_outcome_class": outcome,
        "physics_qc_pass": physics_qc,
        "evidence": evidence,
    }


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(
        pa.table(
            {
                "timestamp_s": [row["timestamp_s"] for row in rows],
                "robot_qpos": [row["robot_qpos"] for row in rows],
                "robot_qvel": [row["robot_qvel"] for row in rows],
                "action_ctrl": [row["action_ctrl"] for row in rows],
                "object_position_m": [row["ball_position"] for row in rows],
                "object_quaternion_wxyz": [row["ball_quaternion"] for row in rows],
                "object_linear_velocity_m_s": [row["ball_linear_velocity"] for row in rows],
                "object_angular_velocity_rad_s": [row["ball_angular_velocity"] for row in rows],
                "grasp_center_m": [row["grasp_center"] for row in rows],
                "left_finger_contacts": [row["left_contacts"] for row in rows],
                "right_finger_contacts": [row["right_contacts"] for row in rows],
                "hand_contacts": [row["hand_contacts"] for row in rows],
                "end_effector_contact": [row["end_effector_contact"] for row in rows],
                "bilateral_contact": [row["bilateral_contact"] for row in rows],
                "surface_contacts": [row["surface_contacts"] for row in rows],
                "maximum_finger_penetration_m": [row["maximum_finger_penetration_m"] for row in rows],
                "maximum_surface_penetration_m": [row["maximum_surface_penetration_m"] for row in rows],
                "retained": [row["retained"] for row in rows],
            }
        ),
        path,
        compression="zstd",
    )


def free_camera(params) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = params["focus"]
    camera.distance = float(params["camera_distance"])
    camera.azimuth = float(params["camera_azimuth"])
    camera.elevation = float(params["camera_elevation"])
    return camera


def run_episode(scenario, output_root, *, fps, duration_s, width, height):
    from scripts_mujoco.scene_builder import build_episode

    params = scenario_parameters(scenario)
    sample = make_sample(scenario, params, fps=fps, duration_s=duration_s)
    bundle = build_episode(sample)
    model, data = bundle.model, bundle.data
    ball_geom, surface_ids = configure_model(model, params)
    controller = (
        CatchTransportController(model, data, sample, params)
        if params["controller_mode"] == "catch_transport"
        else GenericPandaController(model, data, sample, params)
    )
    controller.prepare()
    output = output_root / "episodes" / scenario.episode_id
    output.mkdir(parents=True)
    renderers = {
        "main": mujoco.Renderer(model, height=height, width=width),
        "secondary": mujoco.Renderer(model, height=height, width=width),
    }
    cameras: dict[str, Any] = {"main": "main_camera", "secondary": free_camera(params)}
    frames = {"main": [], "secondary": []}
    rows: list[dict[str, Any]] = []
    timestep = float(sample.timestep)
    total_steps = int(round(duration_s / timestep))
    frame_every = max(1, int(round(1.0 / (fps * timestep))))
    object_rewrites = robot_rewrites = 0

    def record(timestamp: float) -> None:
        row = observe(model, data, controller, ball_geom, surface_ids)
        row.update(
            timestamp_s=float(timestamp),
            robot_qpos=data.qpos[controller.ids.qpos_adrs].tolist(),
            robot_qvel=data.qvel[controller.ids.qvel_adrs].tolist(),
            action_ctrl=data.ctrl.tolist(),
        )
        rows.append(row)

    try:
        record(0.0)
        for view, renderer in renderers.items():
            renderer.update_scene(data, camera=cameras[view])
            frames[view].append(renderer.render().copy())
        for step in range(1, total_steps + 1):
            timestamp = step * timestep
            qadr, vadr = controller.ids.ball_qpos_adr, controller.ids.ball_qvel_adr
            object_before = np.r_[data.qpos[qadr : qadr + 7], data.qvel[vadr : vadr + 6]].copy()
            robot_before = np.r_[data.qpos[controller.ids.qpos_adrs], data.qvel[controller.ids.qvel_adrs]].copy()
            controller.before_step(float(timestamp))
            object_after = np.r_[data.qpos[qadr : qadr + 7], data.qvel[vadr : vadr + 6]]
            robot_after = np.r_[data.qpos[controller.ids.qpos_adrs], data.qvel[controller.ids.qvel_adrs]]
            object_rewrites += int(not np.array_equal(object_before, object_after))
            robot_rewrites += int(not np.array_equal(robot_before, robot_after))
            if object_rewrites or robot_rewrites:
                raise RuntimeError(f"state rewrite detected: object={object_rewrites}, robot={robot_rewrites}")
            mujoco.mj_step(model, data)
            record(float(timestamp))
            if step % frame_every == 0:
                for view, renderer in renderers.items():
                    renderer.update_scene(data, camera=cameras[view])
                    frames[view].append(renderer.render().copy())
    finally:
        for renderer in renderers.values():
            renderer.close()

    result = evaluate(scenario, params, rows)
    videos = {}
    for view in ("main", "secondary"):
        path = output / f"{view}.mp4"
        write_video(path, frames[view], fps)
        videos[view] = {
            "path": str(path), "sha256": sha256(path), "frame_count": len(frames[view]),
            "camera": "main_camera" if view == "main" else "scenario_free_camera",
        }
    trajectory = output / "trajectory.parquet"
    write_parquet(trajectory, rows)
    metadata = {
        "schema_version": "dynamic-robot-scenario-example/v1",
        "episode_id": scenario.episode_id,
        "development_only": True,
        "release_eligible": False,
        "category": scenario.category,
        "scenario_type": scenario.scenario_type,
        "embodiment": "franka_hand",
        "actual_outcome_class": result["actual_outcome_class"],
        "objective_success": result["objective_success"],
        "physics_qc_pass": result["physics_qc_pass"],
        "objective_evidence": result["evidence"],
        "controller": {
            "post_initialization_robot_motion": "MuJoCo actuator controls only",
            "post_initialization_object_motion": "native MuJoCo freejoint dynamics",
            "robot_state_rewrite_count": robot_rewrites,
            "object_state_rewrite_count": object_rewrites,
            "latch_or_weld": False,
            "object_capture_equality": False,
            "custom_tray_bin_or_paddle": False,
        },
        "seed": scenario.seed,
        "scene_variant": scenario.scene_variant,
        "physics": {
            "gravity_m_s2": list(sample.gravity),
            "object_radius_m": sample.ball_radius,
            "object_mass_kg": sample.ball_mass,
            "object_friction": list(params["ball_friction"]),
            "timestep_s": timestep,
        },
        "surface_specs": list(params["surfaces"]),
        "videos": videos,
        "trajectory": {"path": str(trajectory), "sha256": sha256(trajectory), "rows": len(rows)},
        "release_blocker": "development example; canonical-v2 admission and full acceptance are not complete",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    preview = frames["main"][min(len(frames["main"]) - 1, int(0.75 * fps))]
    return metadata, preview


def contact_sheet(path: Path, results, previews) -> None:
    tile_w, tile_h, label_h, columns = 416, 240, 44, 2
    rows = (len(results) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, result in enumerate(results):
        reader = imageio.get_reader(result["videos"]["main"]["path"])
        try:
            frame_count = int(result["videos"]["main"]["frame_count"])
            frame = reader.get_data(min(frame_count - 1, int(round(0.30 * frame_count))))
        finally:
            reader.close()
        x, y = (index % columns) * tile_w, (index // columns) * (tile_h + label_h)
        canvas.paste(Image.fromarray(frame).resize((tile_w, tile_h)), (x, y))
        draw.text(
            (x + 5, y + tile_h + 3),
            f"{result['episode_id']}\nactual={result['actual_outcome_class']}  qc={result['physics_qc_pass']}",
            fill="black",
        )
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config, scenarios = load_config(args.config)
    source_root = Path(args.source_root or config["source_root"]).resolve()
    source_files = install_source_root(source_root)
    before = {str(path): sha256(path) for path in source_files}
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    width, height = (int(value) for value in config["resolution"])
    fps, duration_s = int(config["fps"]), float(config["duration_s"])
    results, previews = [], []
    for index, scenario in enumerate(scenarios, 1):
        print(f"[{index}/{len(scenarios)}] {scenario.episode_id}", flush=True)
        result, preview = run_episode(
            scenario, output, fps=fps, duration_s=duration_s, width=width, height=height
        )
        results.append(result)
        previews.append(preview)
        print(
            f"  actual={result['actual_outcome_class']} success={result['objective_success']} "
            f"physics_qc={result['physics_qc_pass']}", flush=True,
        )
    after = {str(path): sha256(path) for path in source_files}
    if before != after:
        raise RuntimeError("source files changed during generation")
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(result, sort_keys=True) + "\n" for result in results), encoding="utf-8")
    contact_sheet(output / "contact_sheet.png", results, previews)
    report = {
        "schema_version": config["schema_version"],
        "suite_id": config["suite_id"],
        "development_only": True,
        "release_eligible": False,
        "episode_count": len(results),
        "logical_duration_s": len(results) * duration_s,
        "encoded_stream_duration_s": len(results) * duration_s * 2,
        "category_counts": {
            category: sum(result["category"] == category for result in results)
            for category in sorted({result["category"] for result in results})
        },
        "objective_success_count": sum(result["objective_success"] for result in results),
        "physics_qc_pass_count": sum(result["physics_qc_pass"] for result in results),
        "all_rewrite_guards_passed": all(
            result["controller"]["robot_state_rewrite_count"] == 0
            and result["controller"]["object_state_rewrite_count"] == 0
            for result in results
        ),
        "source_tree_unchanged": before == after,
        "source_sha256": before,
        "manifest": str(manifest),
        "contact_sheet": str(output / "contact_sheet.png"),
        "reproduction_command": (
            f"MUJOCO_GL=egl {sys.executable} {Path(__file__).resolve()} "
            f"--config {args.config.resolve()} --output {output} --overwrite"
        ),
    }
    (output / "suite_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
