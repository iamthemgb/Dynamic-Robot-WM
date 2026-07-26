from __future__ import annotations

import mujoco
import numpy as np


ROBOTIQ_LEFT_PAD_SITE = "robotiq_left_pad_site"
ROBOTIQ_RIGHT_PAD_SITE = "robotiq_right_pad_site"
ROBOTIQ_LEFT_THICK_PAD_GEOM = "rq_left_pad_thick_collision_pad"
ROBOTIQ_RIGHT_THICK_PAD_GEOM = "rq_right_pad_thick_collision_pad"
BALL_BODY = "catch_ball"
LATERAL_AXIS_ABS_Z_TOL = 0.25


def _name2id(model: mujoco.MjModel, objtype, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return idx


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _geom_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    geom_id = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xpos[geom_id].copy()


def _geom_axis(model: mujoco.MjModel, data: mujoco.MjData, name: str, axis_index: int) -> np.ndarray:
    geom_id = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return data.geom_xmat[geom_id].reshape(3, 3)[:, axis_index].copy()


def robotiq_geometry_check(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Check that the thick-pad Robotiq is a lateral parallel gripper."""
    mujoco.mj_forward(model, data)
    left_site = _name2id(model, mujoco.mjtObj.mjOBJ_SITE, ROBOTIQ_LEFT_PAD_SITE)
    right_site = _name2id(model, mujoco.mjtObj.mjOBJ_SITE, ROBOTIQ_RIGHT_PAD_SITE)
    left_pos = data.site_xpos[left_site].copy()
    right_pos = data.site_xpos[right_site].copy()
    jaw_axis = _unit(left_pos - right_pos)

    left_geom_pos = _geom_pos(model, data, ROBOTIQ_LEFT_THICK_PAD_GEOM)
    right_geom_pos = _geom_pos(model, data, ROBOTIQ_RIGHT_THICK_PAD_GEOM)
    left_normal = _unit(_geom_axis(model, data, ROBOTIQ_LEFT_THICK_PAD_GEOM, 1))
    right_normal = _unit(_geom_axis(model, data, ROBOTIQ_RIGHT_THICK_PAD_GEOM, 1))
    left_to_right = _unit(right_geom_pos - left_geom_pos)
    right_to_left = -left_to_right
    left_inward = left_normal if float(np.dot(left_normal, left_to_right)) >= 0.0 else -left_normal
    right_inward = right_normal if float(np.dot(right_normal, right_to_left)) >= 0.0 else -right_normal

    left_face_vertical = abs(float(left_inward[2])) <= LATERAL_AXIS_ABS_Z_TOL
    right_face_vertical = abs(float(right_inward[2])) <= LATERAL_AXIS_ABS_Z_TOL
    jaw_lateral = abs(float(jaw_axis[2])) <= LATERAL_AXIS_ABS_Z_TOL
    normals_opposed = float(np.dot(left_inward, right_inward)) <= -0.95

    return {
        "left_pad_position": [float(v) for v in left_pos],
        "right_pad_position": [float(v) for v in right_pos],
        "jaw_axis_left_to_right": [float(v) for v in jaw_axis],
        "jaw_axis_abs_z": abs(float(jaw_axis[2])),
        "left_inward_normal": [float(v) for v in left_inward],
        "right_inward_normal": [float(v) for v in right_inward],
        "left_inward_normal_abs_z": abs(float(left_inward[2])),
        "right_inward_normal_abs_z": abs(float(right_inward[2])),
        "inward_normal_dot": float(np.dot(left_inward, right_inward)),
        "pad_faces_vertical": bool(left_face_vertical and right_face_vertical),
        "jaw_axis_lateral": bool(jaw_lateral),
        "inward_normals_face_each_other": bool(normals_opposed),
        "lateral_axis_abs_z_tolerance": float(LATERAL_AXIS_ABS_Z_TOL),
        "passed": bool(left_face_vertical and right_face_vertical and jaw_lateral and normals_opposed),
    }


def robotiq_rollout_check(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller_result: dict,
    *,
    expected_success: bool,
    ball_radius: float,
) -> dict:
    """Check success/failure semantics against the corrected side-pad geometry."""
    mujoco.mj_forward(model, data)
    geometry = robotiq_geometry_check(model, data)
    ball_body = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, BALL_BODY)
    ball_pos = data.xpos[ball_body].copy()
    left_site = _name2id(model, mujoco.mjtObj.mjOBJ_SITE, ROBOTIQ_LEFT_PAD_SITE)
    right_site = _name2id(model, mujoco.mjtObj.mjOBJ_SITE, ROBOTIQ_RIGHT_PAD_SITE)
    left_pos = data.site_xpos[left_site].copy()
    right_pos = data.site_xpos[right_site].copy()
    jaw_axis = _unit(left_pos - right_pos)
    pad_center = 0.5 * (left_pos + right_pos)
    pad_center_separation = float(np.linalg.norm(left_pos - right_pos))
    ball_lateral_offset = abs(float(np.dot(ball_pos - pad_center, jaw_axis)))
    ball_center_error = float(np.linalg.norm(ball_pos - pad_center))
    ball_between_side_pads = (
        ball_lateral_offset <= 0.5 * pad_center_separation + 0.20 * float(ball_radius)
        and ball_center_error <= max(0.075, 2.5 * float(ball_radius))
    )

    two_pad_contact = int(controller_result.get("two_pad_contact_frames", 0))
    left_pad_contact = int(controller_result.get("left_pad_contact_frames", 0))
    right_pad_contact = int(controller_result.get("right_pad_contact_frames", 0))
    success_conditions = [
        bool(controller_result.get("success")),
        bool(controller_result.get("grasped")),
        two_pad_contact > 0,
        bool(ball_between_side_pads),
        ball_center_error <= max(0.075, 2.5 * float(ball_radius)),
    ]
    failure_conditions = [
        not bool(controller_result.get("success")),
        not bool(controller_result.get("grasped")),
        two_pad_contact == 0 or not bool(ball_between_side_pads),
    ]
    if left_pad_contact > 0 or right_pad_contact > 0:
        failure_reason = "single-pad glancing contact or slip caused by target misalignment"
    else:
        failure_reason = "ball missed the corrected side-pad gap"

    outcome_passed = all(success_conditions) if expected_success else all(failure_conditions)
    branch = str(controller_result.get("branch", ""))
    branch_mode = str(controller_result.get("branch_mode", ""))
    geometry_required = bool(expected_success or branch in {"spatial_near_miss", "contact_failure"})
    geometry_passed = bool(geometry["passed"] or not geometry_required)
    return {
        "geometry": geometry,
        "expected_success": bool(expected_success),
        "ball_position": [float(v) for v in ball_pos],
        "pad_center": [float(v) for v in pad_center],
        "pad_center_separation": pad_center_separation,
        "ball_lateral_offset_along_jaw_axis": ball_lateral_offset,
        "ball_center_error": ball_center_error,
        "ball_between_side_pads": bool(ball_between_side_pads),
        "left_pad_contact_frames": left_pad_contact,
        "right_pad_contact_frames": right_pad_contact,
        "two_pad_contact_frames": two_pad_contact,
        "failure_reason": None if expected_success else failure_reason,
        "geometry_required": bool(geometry_required),
        "geometry_passed": bool(geometry_passed),
        "branch": branch,
        "branch_mode": branch_mode,
        "passed": bool(geometry_passed and outcome_passed),
    }
