"""Synthetic multi-family counterfactual environments.

Two 2D ball-world families with DIFFERENT metadata schemas, so the
variable-record path is exercised end to end (plan: "define the first two
or three environment families"):

  projectile_impact (4 records: gravity_z, mass, restitution,
      dynamic_friction) — ball free-falls through the whole prefix (theta-
      invariant, so variants share a pixel-identical prefix), restitution
      becomes identifiable at first impact, mass at the scripted push,
      friction while sliding.

  push_slide (5 records: gravity_z, mass, dynamic_friction,
      static_friction, actuator_scale) — ball starts resting on the floor;
      the scripted push is scaled by a hidden actuator gain; static friction
      sets the breakaway threshold, dynamic friction the deceleration.

A counterfactual group shares initial state, appearance, and action script;
only the physics records differ. Change-point episodes switch
dynamic_friction mid-episode. Frames are float32 grayscale [T, 1, H, W];
actions are the commanded horizontal force [T-1, 1] (the actuator gain is
hidden physics, not an observed action).
"""

from dataclasses import dataclass, field

import numpy as np

from ..models.metadata_records import TypedRecord

WORLD = 8.0
BALL_R = 0.4
WALL_E = 0.5          # fixed for all episodes: walls must not leak theta
REST_VY = 0.35
GRAVITY = 9.8

FAMILIES = ("projectile_impact", "push_slide")


@dataclass
class Episode:
    family: str
    frames: np.ndarray            # [T, 1, H, W]
    actions: np.ndarray           # [T-1, 1] commanded force
    records: list                 # list[TypedRecord] — simulation privilege
    events: dict = field(default_factory=dict)
    change_frame: int | None = None
    records_after: list | None = None


def _records(family, p):
    pair = "primary_object--support_surface"
    recs = [
        TypedRecord("gravity_z", "global", "m_per_s2", -p["gravity"]),
        TypedRecord("mass", "primary_object", "kg", p["mass"]),
        TypedRecord("dynamic_friction", pair, "dimensionless", p["mu_d"]),
    ]
    if family == "projectile_impact":
        recs.append(TypedRecord("restitution", pair, "dimensionless", p["e"]))
    elif family == "push_slide":
        recs.append(TypedRecord("static_friction", pair, "dimensionless",
                                p["mu_s"]))
        recs.append(TypedRecord("actuator_scale", "robot", "dimensionless",
                                p["act"]))
    else:
        raise KeyError(family)
    return recs


def _render(xs, ys, H, ball_lum, bg_lum):
    px = WORLD / H
    T = len(xs)
    jj, ii = np.meshgrid(np.arange(H), np.arange(H), indexing="xy")
    cx = xs[:, None, None] / px
    cy = H - 1 - ys[:, None, None] / px
    dist = np.sqrt((jj[None] - cx) ** 2 + (ii[None] - cy) ** 2)
    disc = 1.0 / (1.0 + np.exp((dist - BALL_R / px) / 0.5))
    frames = np.full((T, H, H), bg_lum, dtype=np.float64)
    frames[:, -2:, :] = 0.3
    frames = np.maximum(frames, disc * ball_lum)
    return frames[:, None].astype(np.float32)


def _simulate(family, p, x0, y0, vx0, script, n_frames, dt,
              change_frame=None, p_after=None):
    x, y, vx, vy = x0, y0, vx0, 0.0
    xs, ys = [x], [y]
    impact = slide = None
    q = dict(p)
    for t in range(n_frames - 1):
        if change_frame is not None and t >= change_frame:
            q = dict(p_after)
        g = q["gravity"]
        F = script[t] * q.get("act", 1.0)
        on_floor = y <= BALL_R + 1e-6 and abs(vy) < 1e-6
        ax = F / q["mass"]
        if on_floor:
            static_hold = (abs(vx) < 1e-4 and
                           abs(F) <= q.get("mu_s", q["mu_d"]) *
                           q["mass"] * g)
            if static_hold:
                ax = 0.0
            elif abs(vx) > 1e-4:
                ax -= q["mu_d"] * g * np.sign(vx)
                if slide is None:
                    slide = t
        else:
            vy -= g * dt
        vx += ax * dt
        if on_floor and F == 0.0 and abs(vx) < q["mu_d"] * g * dt:
            vx = 0.0
        x += vx * dt
        y += vy * dt
        if y <= BALL_R and vy < 0:
            y = BALL_R
            vy = -q.get("e", 0.0) * vy
            if impact is None:
                impact = t + 1
            if abs(vy) < REST_VY:
                vy = 0.0
        if x <= BALL_R and vx < 0:
            x, vx = BALL_R, -WALL_E * vx
        if x >= WORLD - BALL_R and vx > 0:
            x, vx = WORLD - BALL_R, -WALL_E * vx
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys), impact, slide


def _sample_params(rng, family):
    p = {"gravity": GRAVITY,
         "mass": float(rng.uniform(0.5, 2.0)),
         "mu_d": float(rng.uniform(0.05, 0.6))}
    if family == "projectile_impact":
        p["e"] = float(rng.uniform(0.3, 0.9))
    else:
        p["mu_s"] = float(rng.uniform(p["mu_d"], 0.8))
        p["act"] = float(rng.uniform(0.5, 2.0))
    return p


def make_group(rng, family, n_frames=33, image_size=64, variants=4,
               dt=1.0 / 16.0, change_frame=None):
    """One counterfactual group: shared prefix/script/appearance, differing
    physics records."""
    ball_lum = rng.uniform(0.7, 1.0)
    bg_lum = rng.uniform(0.05, 0.15)
    if family == "projectile_impact":
        x0, y0 = rng.uniform(2.5, 5.5), rng.uniform(6.2, 7.4)
        vx0 = rng.uniform(-0.4, 0.4)
        push_start = int(rng.integers(20, 25))
    else:
        x0, y0 = rng.uniform(1.5, 3.0), BALL_R
        vx0 = 0.0
        push_start = int(rng.integers(4, 9))
    push_len = int(rng.integers(5, 8))
    F = rng.uniform(3.0, 6.0) * (rng.choice([-1.0, 1.0])
                                 if family == "projectile_impact" else 1.0)
    script = np.zeros(n_frames - 1, dtype=np.float32)
    script[push_start:push_start + push_len] = F

    episodes = []
    for _ in range(variants):
        p = _sample_params(rng, family)
        p_after = None
        if change_frame is not None:
            p_after = dict(p)
            p_after["mu_d"] = float(rng.uniform(0.05, 0.6))
            if family == "push_slide":
                p_after["mu_s"] = float(rng.uniform(p_after["mu_d"], 0.8))
        xs, ys, impact, slide = _simulate(
            family, p, x0, y0, vx0, script, n_frames, dt,
            change_frame=change_frame, p_after=p_after)
        episodes.append(Episode(
            family=family,
            frames=_render(xs, ys, image_size, ball_lum, bg_lum),
            actions=script[:, None].copy(),
            records=_records(family, p),
            events={"impact": impact, "push_start": push_start,
                    "slide_start": slide},
            change_frame=change_frame,
            records_after=_records(family, p_after) if p_after else None,
        ))
    return episodes


def make_dataset(n_groups, seed=0, families=FAMILIES, change_frame=None,
                 **kw):
    """List of counterfactual groups, alternating environment families."""
    rng = np.random.default_rng(seed)
    return [make_group(rng, families[i % len(families)],
                       change_frame=change_frame, **kw)
            for i in range(n_groups)]
