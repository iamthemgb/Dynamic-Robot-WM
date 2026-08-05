"""pbc_state_groups episode -> TypedRecord conditioning bundles.

The bundle is the sibling's INITIAL-CONDITION plan and nothing else: ball
state at release, intended catch target, planned gripper close time, and the
branch's controller intent flags. Realized outcomes (intercept position,
contact observables, catch_success) are deliberately excluded -- they are
evaluation data, not teacher inputs. Physics is constant dataset-wide by
construction (validated by gate 3), so no physics keys are emitted at all;
under v3 the state tokens ARE the experiment.

State is expressed in the MAIN camera's frame, mirroring ``f1_records``: the
camera is what relates tokens to pixels, and the cache stores main-view
latents. Within a group the camera is bit-identical across siblings, so the
correct/wrong comparison is unaffected by the frame choice; across groups it
tracks the per-group camera jitter the way f1 tracked its per-episode rigs.

Wrong-state negatives swap this ENTIRE record set between same-group
siblings (``GroupBatcher.paired_batch``): commands are a deterministic
function of the IC, so state and action-plan keys must travel together --
a bundle that mixed sibling A's state with sibling B's plan would be
internally inconsistent and rejectable without watching the video.
"""

import math

import numpy as np

from ..models.metadata_records import TypedRecord

GRAVITY_Z = -9.81

#: key -> (scope, unit); every key must exist in metadata_registry.yaml.
FIELD_SPECS = {
    "log10_ball_mass": ("primary_object", "canonical_sim_unit"),
    "ball_radius": ("primary_object", "m"),
    "p0_cam_x": ("primary_object", "m"),
    "p0_cam_y": ("primary_object", "m"),
    "p0_cam_z": ("primary_object", "m"),
    "v0_cam_x": ("primary_object", "m_per_s"),
    "v0_cam_y": ("primary_object", "m_per_s"),
    "v0_cam_z": ("primary_object", "m_per_s"),
    "log10_speed": ("primary_object", "canonical_sim_unit"),
    "icpt_cam_x": ("global", "m"),
    "icpt_cam_y": ("global", "m"),
    "icpt_cam_z": ("global", "m"),
    "ballistic_intercept_time_s": ("global", "s"),
    "gripper_close_time": ("robot", "s"),
    "release_time_s": ("global", "s"),
    "controller_hold_home": ("robot", "dimensionless"),
    "grasp_capture_enabled": ("robot", "dimensionless"),
}


def camera_matrix(cam: dict):
    """world->camera rotation R (rows: right, up, forward) and translation.

    Built from the recorded pos/lookat with world +z as up. The exact axis
    convention matters less than its consistency: every episode's records use
    this same construction, so distances and comparisons are coherent.
    """
    pos = np.asarray(cam["pos"], dtype=np.float64)
    lookat = np.asarray(cam["lookat"], dtype=np.float64)
    f = lookat - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, f])
    return R, -R @ pos


def _to_cam(R, t, p, rotation_only=False):
    p = np.asarray(p, dtype=np.float64)
    return R @ p if rotation_only else R @ p + t


def _ballistic_intercept(p0, v0, z_plane, g=GRAVITY_Z):
    """Smallest positive t with 0.5*g*t^2 + v0z*t + (p0z - z_plane) = 0."""
    a, b, c = 0.5 * g, float(v0[2]), float(p0[2]) - float(z_plane)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None
    sq = math.sqrt(disc)
    roots = sorted(r for r in ((-b + sq) / (2 * a), (-b - sq) / (2 * a))
                   if r > 0)
    if not roots:
        return None, None
    t = roots[0]
    return t, np.array([p0[0] + v0[0] * t, p0[1] + v0[1] * t, z_plane])


def build_records(metadata: dict, branch: str):
    """-> list[TypedRecord] with RAW (un-whitened) SI values."""
    R, t = camera_matrix(metadata["cameras"]["main_camera"])
    state = metadata["state_context"]
    p0 = np.asarray(state["ball_initial_position_m"], dtype=np.float64)
    v0 = np.asarray(state["ball_initial_velocity_mps"], dtype=np.float64)
    catch = np.asarray(metadata["visual_settings"]["catch_position_xyz"],
                       dtype=np.float64)

    p0c = _to_cam(R, t, p0)
    v0c = _to_cam(R, t, v0, rotation_only=True)
    catch_c = _to_cam(R, t, catch)
    speed = float(np.linalg.norm(v0))

    t_icpt, _ = _ballistic_intercept(p0, v0, catch[2])
    if t_icpt is None:                      # never reaches the catch plane
        t_icpt = 2.0

    vals = {
        "log10_ball_mass": math.log10(
            max(float(metadata["physics_tokens"]["ball_mass_kg"]), 1e-12)),
        "ball_radius": float(
            metadata["geometry_context"]["ball_radius_m"]),
        "p0_cam_x": p0c[0], "p0_cam_y": p0c[1], "p0_cam_z": p0c[2],
        "v0_cam_x": v0c[0], "v0_cam_y": v0c[1], "v0_cam_z": v0c[2],
        "log10_speed": math.log10(max(speed, 1e-6)),
        "icpt_cam_x": catch_c[0], "icpt_cam_y": catch_c[1],
        "icpt_cam_z": catch_c[2],
        "ballistic_intercept_time_s": float(min(max(t_icpt, 0.0), 2.5)),
        "gripper_close_time": float(
            metadata["action_context"]["gripper_close_command_time_s"]),
        "release_time_s": float(state["release_time_s"]),
        "controller_hold_home":
            1.0 if metadata["action_context"]["controller_mode"]
            == "hold_home" else 0.0,
        "grasp_capture_enabled": 1.0 if branch == "success" else 0.0,
    }
    return [TypedRecord(k, FIELD_SPECS[k][0], FIELD_SPECS[k][1], float(v))
            for k, v in vals.items() if v is not None and np.isfinite(v)]
