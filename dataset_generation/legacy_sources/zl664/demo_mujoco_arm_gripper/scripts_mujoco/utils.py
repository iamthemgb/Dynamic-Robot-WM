from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


AXIS_CONVERSION_NOTE = "MuJoCo Z-up to glTF Y-up: glTF_position = [x, z, -y]."
MJ_TO_GLTF = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def as_float_list(values) -> list[float]:
    return [float(v) for v in np.asarray(values).reshape(-1)]


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def camera_xyaxes(pos, lookat, up=(0.0, 0.0, 1.0)) -> str:
    pos = np.asarray(pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = lookat - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return " ".join(f"{v:.8f}" for v in [*right, *cam_up])


def mujoco_pos_to_gltf(pos) -> np.ndarray:
    return MJ_TO_GLTF @ np.asarray(pos, dtype=np.float64)


def mujoco_mat_to_gltf_quat_xyzw(mat3x3) -> np.ndarray:
    mat = np.asarray(mat3x3, dtype=np.float64).reshape(3, 3)
    converted = MJ_TO_GLTF @ mat @ MJ_TO_GLTF.T
    quat = Rotation.from_matrix(converted).as_quat()
    quat /= np.linalg.norm(quat)
    return quat


def mat_to_quat_xyzw(mat3x3) -> np.ndarray:
    mat = np.asarray(mat3x3, dtype=np.float64).reshape(3, 3)
    quat = Rotation.from_matrix(mat).as_quat()
    quat /= np.linalg.norm(quat)
    return quat


def mujoco_vertices_to_gltf(vertices) -> np.ndarray:
    return (MJ_TO_GLTF @ np.asarray(vertices, dtype=np.float64).T).T.astype(np.float32)


def sphere_mass(radius: float, density: float) -> float:
    return float((4.0 / 3.0) * math.pi * radius**3 * density)
