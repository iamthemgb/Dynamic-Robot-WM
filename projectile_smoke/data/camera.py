"""Main-camera geometry: world->camera transform and pinhole projection.

The metadata gives per-episode main_camera {pos, lookat, fovy}. Orientation is
reconstructed as a standard look-at with world +Z up (MuJoCo/OpenGL camera
convention: camera looks along its -Z, +X right, +Y up). The reprojection
go/no-go check (check_reprojection.py) is the ground-truth validation of this
reconstruction on all 4 family dirs - a wrong transform here is invisible to
every training-side signal, so nothing trains until that check passes.
"""

import math

import numpy as np

IMG_W, IMG_H = 832, 480


def camera_rotation(pos, lookat):
    """3x3 matrix whose COLUMNS are the camera axes (right, up, backward) in
    world coordinates. p_cam = R.T @ (p_world - pos)."""
    pos = np.asarray(pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    fwd = lookat - pos
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    n = np.linalg.norm(right)
    assert n > 1e-8, f"camera looks straight up/down, roll undefined: {pos} {lookat}"
    right /= n
    up = np.cross(right, fwd)
    return np.stack([right, up, -fwd], axis=1)


def world_to_camera(points, pos, lookat):
    """points [N,3] world -> camera frame (x right, y up, z backward)."""
    R = camera_rotation(pos, lookat)
    return (np.asarray(points, dtype=np.float64) - np.asarray(pos)) @ R


def world_dir_to_camera(vectors, pos, lookat):
    """Direction vectors (velocities): rotation only, no translation."""
    R = camera_rotation(pos, lookat)
    return np.asarray(vectors, dtype=np.float64) @ R


def project(points_cam, fovy_deg, img_w=IMG_W, img_h=IMG_H):
    """Camera-frame points [N,3] -> pixel coords [N,2] (col, row) + depth [N].

    Points in front of the camera have z_cam < 0; depth = -z_cam.
    Square pixels: fx = fy = (H/2) / tan(fovy/2).
    """
    p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
    f = (img_h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    depth = -p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = img_w / 2.0 + f * p[:, 0] / depth
        v = img_h / 2.0 - f * p[:, 1] / depth
    return np.stack([u, v], axis=1), depth


def ballistic_positions(meta, t_flight):
    """Analytic p(t) = p0 + v0*t + 0.5*g*t^2 in WORLD frame, t_flight [N] in
    seconds since release (metadata events.release_frame / fps)."""
    p0 = np.asarray(meta["ball_initial_position"], dtype=np.float64)
    v0 = np.asarray(meta["ball_initial_velocity"], dtype=np.float64)
    g = np.asarray(meta["gravity"], dtype=np.float64)
    t = np.asarray(t_flight, dtype=np.float64)[:, None]
    return p0 + v0 * t + 0.5 * g * t * t


def project_trajectory(meta, frame_indices, fps=30):
    """Project the analytic ballistic arc for the given video frame indices
    through the episode's main camera. Returns ([N,2] pixels, [N] depth,
    [N] t_flight). Frames before release get t_flight clamped to 0 (ball at
    rest at p0)."""
    cam = meta["cameras"]["main_camera"]
    release = meta["events"]["release_frame"] or 0
    t_flight = np.maximum((np.asarray(frame_indices) - release) / fps, 0.0)
    pts_w = ballistic_positions(meta, t_flight)
    pts_c = world_to_camera(pts_w, cam["pos"], cam["lookat"])
    px, depth = project(pts_c, cam["fovy"])
    return px, depth, t_flight
