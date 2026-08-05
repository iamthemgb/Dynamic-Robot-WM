"""f1_10h episode -> variable-length TypedRecord sets.

What is identifiable here is NOT material constants. Verified across the
corpus: gravity is (0,0,-9.81) for every episode, ``object_mass`` spans 6%
(and is a single constant for half the episodes), ``object_radius`` spans 2%.
What varies is the scene initial condition, the embodiment, and the contact
outcome -- so that is what the records encode. The 14 keys from
``projectile/records.py`` already describe exactly that shape and are reused
verbatim; v2 of the registry adds seven more.

Record sets are deliberately variable length: contact records are absent on
``miss`` episodes. The teacher is a DeepSets over a padded/masked batch, so
that is the intended representation rather than a problem to pad away.
"""

import math

import numpy as np

from ..models.metadata_records import TypedRecord
from . import f1_episode as F1

GRAVITY_Z = -9.81

#: key -> (scope, unit). The first 14 mirror projectile/records.FIELD_SPECS.
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
    # v2 additions
    "key_event_time_s": ("global", "s"),
    "intercept_offset_m": ("global", "m"),
    "tool_is_parallel_pad": ("robot", "dimensionless"),
    "impact_speed": ("primary_object--secondary_object", "m_per_s"),
    "impact_normal_impulse": ("primary_object--secondary_object",
                              "newton_second"),
    "max_penetration_depth": ("primary_object--secondary_object", "m"),
    "contact_duration": ("primary_object--secondary_object", "s"),
}

#: Always present regardless of outcome; the rest are contact-conditional.
BASE_KEYS = tuple(list(FIELD_SPECS)[:17])
CONTACT_KEYS = ("impact_speed", "impact_normal_impulse",
                "max_penetration_depth", "contact_duration")


def _to_cam(world_to_camera, p, rotation_only=False):
    R = world_to_camera[:3, :3]
    if rotation_only:
        return R @ np.asarray(p, dtype=np.float64)
    return R @ np.asarray(p, dtype=np.float64) + world_to_camera[:3, 3]


def _ballistic_intercept(p0, v0, z_plane, g=GRAVITY_Z):
    """Time and point at which a ballistic ball first reaches z = z_plane.

    Solves 0.5*g*t^2 + v0z*t + (p0z - z_plane) = 0 for the smallest positive
    root. Returns (t, xyz) or (None, None) when the ball never gets there.
    """
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
    xyz = np.array([p0[0] + v0[0] * t, p0[1] + v0[1] * t, z_plane])
    return t, xyz


def build_records(ref: F1.EpisodeRef, row, fps: float = 30.0):
    """-> list[TypedRecord] with RAW (un-whitened) SI values."""
    cam = F1.read_camera(ref)["world_to_camera"]
    frames = F1.read_stream(ref, "data", [
        "object.position", "object.linear_velocity", "grasp.center_position",
        "action.actuator_command", "contact.active"]).to_pydict()

    p0 = np.asarray(frames["object.position"][0], dtype=np.float64)
    v0 = np.asarray(frames["object.linear_velocity"][0], dtype=np.float64)
    grasp0 = np.asarray(frames["grasp.center_position"][0], dtype=np.float64)

    p0c, v0c = _to_cam(cam, p0), _to_cam(cam, v0, rotation_only=True)
    speed = float(np.linalg.norm(v0))

    t_icpt, icpt_w = _ballistic_intercept(p0, v0, grasp0[2])
    if icpt_w is None:                       # never reaches the grasp plane
        t_icpt, icpt_w = 2.0, p0 + v0 * 2.0
    icpt_c = _to_cam(cam, icpt_w)
    offset = float(np.linalg.norm(icpt_w[:2] - grasp0[:2]))

    vals = {
        "log10_ball_mass": math.log10(max(float(row.object_mass), 1e-12)),
        "ball_radius": float(row.object_radius),
        "p0_cam_x": p0c[0], "p0_cam_y": p0c[1], "p0_cam_z": p0c[2],
        "v0_cam_x": v0c[0], "v0_cam_y": v0c[1], "v0_cam_z": v0c[2],
        "log10_speed": math.log10(max(speed, 1e-6)),
        "icpt_cam_x": icpt_c[0], "icpt_cam_y": icpt_c[1],
        "icpt_cam_z": icpt_c[2],
        "ballistic_intercept_time_s": float(min(max(t_icpt, 0.0), 2.0)),
        "gripper_close_time": _gripper_close_time(
            frames["action.actuator_command"], row, fps),
        "key_event_time_s": float(row.key_event_time_s)
        if np.isfinite(getattr(row, "key_event_time_s", np.nan)) else 0.0,
        "intercept_offset_m": offset,
        "tool_is_parallel_pad":
            1.0 if row.tool_type == "robotiq_2f85_thick_pad" else 0.0,
    }
    vals.update(_contact_values(ref, frames, fps))

    return [TypedRecord(k, FIELD_SPECS[k][0], FIELD_SPECS[k][1], float(v))
            for k, v in vals.items() if v is not None and np.isfinite(v)]


def _gripper_close_time(commands, row, fps):
    """First frame the gripper channel crosses its mid-range, in seconds."""
    g = np.asarray([c[7] for c in commands], dtype=np.float64)
    lo, hi = float(g.min()), float(g.max())
    if hi - lo > 1e-9:
        mid = 0.5 * (hi + lo)
        below = np.nonzero(g < mid)[0]
        if below.size:
            return float(below[0]) / fps
    t = getattr(row, "key_event_time_s", np.nan)
    return float(t) if np.isfinite(t) else 0.0


def _contact_values(ref, frames, fps):
    """Contact observables; all None when the episode never makes contact.

    ``events.object_b`` holds raw simulator geom ids (``unnamed_geom_70``),
    which the registry's scope regex rejects by design -- so only numeric
    aggregates cross into records, never an identifier.
    """
    out = dict.fromkeys(CONTACT_KEYS)
    try:
        ev = F1.read_stream(ref, "events", [
            "timestamp", "normal_impulse_n_s", "penetration_depth_m"]).to_pydict()
    except Exception:
        ev = None
    if ev and ev.get("timestamp"):
        ts = np.asarray(ev["timestamp"], dtype=np.float64)
        imp = np.asarray(ev["normal_impulse_n_s"], dtype=np.float64)
        pen = np.asarray(ev["penetration_depth_m"], dtype=np.float64)
        total = float(np.nansum(imp))
        if total > 0:
            out["impact_normal_impulse"] = total
        depth = float(np.nanmax(pen)) if pen.size else 0.0
        if depth > 0:
            out["max_penetration_depth"] = depth
        out["contact_duration"] = float(ts.max() - ts.min())

        active = frames.get("contact.active") or []
        first = next((i for i, a in enumerate(active) if a), None)
        if first:
            v = np.asarray(frames["object.linear_velocity"][first - 1],
                           dtype=np.float64)
            s = float(np.linalg.norm(v))
            if s > 0:
                out["impact_speed"] = s
    return out


def fit_normalizer(registry, record_sets):
    """Fit on the TRAIN split only, dropping keys with no spread.

    A constant key yields sigma=0; the normalizer's eps keeps that finite but
    the resulting record is pure padding that the teacher must learn to ignore.
    Dropping it is both cheaper and more honest.
    """
    from ..data.normalize_metadata import MetadataNormalizer

    norm = MetadataNormalizer(registry).fit(record_sets)
    dropped = [k for k, s in norm.stats.items() if s["sigma"] < 1e-9]
    for k in dropped:
        norm.stats.pop(k)
    return norm, dropped
