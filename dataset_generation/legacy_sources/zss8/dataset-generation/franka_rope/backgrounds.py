"""Scene backgrounds (visual domain randomization) for the Franka-rope family.

The original preview pipeline hard-coded a single studio look (one skybox
gradient, one checker floor, one table colour, two lights). To train a
world-class deformable-manipulation model we need broad *visual* variety
while keeping the *physics* identical, so a policy/world-model learns rope
dynamics rather than a fixed backdrop.

Design contract
---------------
A ``Background`` produces only cosmetic MJCF:

  * ``asset_xml``      extra ``<texture>`` / ``<material>`` definitions
  * ``headlight_xml``  the ``<headlight>`` inside ``<visual>``
  * ``floor_xml``      the floor plane geom(s)
  * ``lights_xml``     ``<light>`` elements
  * ``backdrop_xml``   visual-only walls / props (``contype=0 conaffinity=0``
                       so they can NEVER collide with the rope or arm)
  * ``table_material`` the material name the table geom should use

It never touches the table geometry, table friction, rope, arm, cameras or
gravity — those are physics and stay fixed across every theme, so a
counterfactual bundle rendered under "kitchen" vs "plain" has bit-identical
dynamics. Only pixels change.

Themes are sampled per episode; within a theme every colour, texture repeat,
light position and intensity is jittered so no two episodes look the same.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# The table top sits at z=0 (box pos z=-0.02, half-height 0.02); the floor
# plane is at z=-0.75. Backdrop walls span from ~floor to ~0.8 m above table.
FLOOR_Z = -0.75
WALL_BACK_X = -0.78          # far wall on the -x side (behind the scene)
WALL_SIDE_Y = 1.02           # side wall on the +y side
WALL_TOP_Z = 0.85            # walls rise to here


# --------------------------------------------------------------------- utils

def _fmt(*vals) -> str:
    flat = []
    for v in vals:
        for x in np.atleast_1d(v):
            flat.append(f"{float(x):.5g}")
    return " ".join(flat)


def _jit(rng, base: float, amt: float, lo=None, hi=None) -> float:
    x = base + rng.uniform(-amt, amt)
    if lo is not None or hi is not None:
        x = float(np.clip(x, lo if lo is not None else -1e9,
                          hi if hi is not None else 1e9))
    return float(x)


def _jrgb(rng, rgb, amt: float) -> Tuple[float, float, float]:
    return tuple(float(np.clip(c + rng.uniform(-amt, amt), 0.0, 1.0)) for c in rgb)


def _shift_rgb(rgb, d) -> Tuple[float, float, float]:
    return tuple(float(np.clip(c + d, 0.0, 1.0)) for c in rgb)


@dataclass
class Background:
    theme: str
    asset_xml: str
    headlight_xml: str
    floor_xml: str
    lights_xml: str
    backdrop_xml: str
    table_material: str
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------- theme spec

@dataclass
class Theme:
    name: str
    sky1: Tuple[float, float, float]
    sky2: Tuple[float, float, float]
    floor1: Tuple[float, float, float]
    floor2: Tuple[float, float, float]
    floor_repeat: float
    floor_reflect: float
    table_rgba: Tuple[float, float, float, float]
    table_spec: float
    table_shin: float
    # table surface texture: None (flat colour) or (rgb1, rgb2, repeat) checker
    table_tex: Optional[Tuple[tuple, tuple, float]]
    headlight: Tuple[float, float, float]     # diffuse, ambient, specular (grey)
    lights: List[dict]
    backdrop: Optional[str] = None            # key into _BACKDROPS
    sky_builtin: str = "gradient"
    color_jitter: float = 0.05
    weight: float = 1.0                       # sampling weight


# Each light: pos, dir, diffuse (grey scalar), castshadow.
def _L(pos, dir, diff, shadow=False):
    return {"pos": pos, "dir": dir, "diff": diff, "shadow": shadow}


THEMES: Dict[str, Theme] = {
    # ---- plain neutral studio (kept as an explicit "plain" option) ---------
    "plain_studio": Theme(
        name="plain_studio",
        sky1=(0.46, 0.54, 0.62), sky2=(0.86, 0.90, 0.94),
        floor1=(0.23, 0.24, 0.26), floor2=(0.29, 0.30, 0.32),
        floor_repeat=8, floor_reflect=0.0,
        table_rgba=(0.46, 0.43, 0.39, 1.0), table_spec=0.2, table_shin=0.2,
        table_tex=None,
        headlight=(0.55, 0.32, 0.10),
        lights=[_L((0.4, -0.6, 1.6), (-0.2, 0.4, -1), 0.5, True),
                _L((-0.6, 0.8, 1.4), (0.3, -0.4, -1), 0.35, False)],
        weight=1.3,
    ),
    # ---- seamless photo-studio cyclorama (very clean, bright) --------------
    "white_cyclorama": Theme(
        name="white_cyclorama",
        sky1=(0.93, 0.94, 0.96), sky2=(0.99, 0.99, 1.0),
        floor1=(0.82, 0.83, 0.85), floor2=(0.88, 0.89, 0.91),
        floor_repeat=4, floor_reflect=0.15,
        table_rgba=(0.90, 0.90, 0.92, 1.0), table_spec=0.35, table_shin=0.4,
        table_tex=None,
        headlight=(0.72, 0.42, 0.12),
        lights=[_L((0.3, -0.5, 1.8), (-0.15, 0.3, -1), 0.55, True),
                _L((-0.5, 0.6, 1.6), (0.25, -0.3, -1), 0.45, False),
                _L((0.0, 0.0, 2.0), (0, 0, -1), 0.3, False)],
        backdrop="cyclorama", color_jitter=0.03, weight=1.0,
    ),
    # ---- kitchen: warm, marble counter, tiled backsplash + floor ----------
    "kitchen": Theme(
        name="kitchen",
        sky1=(0.80, 0.76, 0.66), sky2=(0.93, 0.90, 0.83),
        floor1=(0.70, 0.66, 0.58), floor2=(0.62, 0.57, 0.49),
        floor_repeat=10, floor_reflect=0.05,
        table_rgba=(0.86, 0.84, 0.80, 1.0), table_spec=0.5, table_shin=0.6,
        table_tex=((0.90, 0.88, 0.84), (0.80, 0.77, 0.72), 3),   # marble-ish
        headlight=(0.50, 0.34, 0.12),
        lights=[_L((0.5, -0.4, 1.5), (-0.25, 0.2, -1), 0.55, True),
                _L((-0.4, 0.7, 1.3), (0.2, -0.35, -1), 0.30, False)],
        backdrop="kitchen", color_jitter=0.05, weight=1.2,
    ),
    # ---- industrial warehouse: cool concrete, dim, big room ---------------
    "warehouse": Theme(
        name="warehouse",
        sky1=(0.30, 0.33, 0.37), sky2=(0.55, 0.58, 0.62),
        floor1=(0.34, 0.35, 0.37), floor2=(0.40, 0.41, 0.43),
        floor_repeat=6, floor_reflect=0.08,
        table_rgba=(0.42, 0.44, 0.47, 1.0), table_spec=0.25, table_shin=0.25,
        table_tex=None,
        headlight=(0.42, 0.30, 0.08),
        lights=[_L((0.2, -0.3, 2.0), (-0.1, 0.15, -1), 0.5, True),
                _L((-0.8, 0.9, 1.8), (0.4, -0.45, -1), 0.28, False)],
        backdrop="warehouse", color_jitter=0.04, weight=1.0,
    ),
    # ---- wood workshop bench: warm plank table + planks behind ------------
    "wood_workshop": Theme(
        name="wood_workshop",
        sky1=(0.52, 0.44, 0.34), sky2=(0.72, 0.64, 0.52),
        floor1=(0.40, 0.30, 0.20), floor2=(0.46, 0.35, 0.24),
        floor_repeat=7, floor_reflect=0.0,
        table_rgba=(0.55, 0.38, 0.22, 1.0), table_spec=0.15, table_shin=0.3,
        table_tex=((0.58, 0.40, 0.23), (0.48, 0.32, 0.18), 6),    # planks
        headlight=(0.50, 0.32, 0.10),
        lights=[_L((0.45, -0.5, 1.5), (-0.2, 0.25, -1), 0.5, True),
                _L((-0.5, 0.6, 1.3), (0.25, -0.3, -1), 0.32, False)],
        backdrop="workshop", color_jitter=0.05, weight=1.0,
    ),
    # ---- clean science lab: bright, cool white, epoxy floor ---------------
    "lab": Theme(
        name="lab",
        sky1=(0.78, 0.82, 0.86), sky2=(0.90, 0.93, 0.96),
        floor1=(0.55, 0.60, 0.62), floor2=(0.60, 0.65, 0.67),
        floor_repeat=5, floor_reflect=0.2,
        table_rgba=(0.80, 0.82, 0.84, 1.0), table_spec=0.45, table_shin=0.5,
        table_tex=None,
        headlight=(0.68, 0.42, 0.14),
        lights=[_L((0.3, -0.4, 1.9), (-0.15, 0.2, -1), 0.55, True),
                _L((-0.5, 0.7, 1.7), (0.25, -0.35, -1), 0.42, False),
                _L((0.6, 0.5, 1.7), (-0.3, -0.25, -1), 0.3, False)],
        backdrop="lab", color_jitter=0.03, weight=1.0,
    ),
    # ---- home office desk: wood desk, warm, carpet floor ------------------
    "office_desk": Theme(
        name="office_desk",
        sky1=(0.58, 0.56, 0.52), sky2=(0.78, 0.76, 0.72),
        floor1=(0.36, 0.33, 0.30), floor2=(0.42, 0.39, 0.35),
        floor_repeat=12, floor_reflect=0.0,
        table_rgba=(0.48, 0.36, 0.26, 1.0), table_spec=0.3, table_shin=0.45,
        table_tex=((0.50, 0.38, 0.27), (0.44, 0.32, 0.22), 5),
        headlight=(0.52, 0.34, 0.12),
        lights=[_L((0.4, -0.5, 1.5), (-0.2, 0.25, -1), 0.5, True),
                _L((-0.4, 0.6, 1.4), (0.2, -0.3, -1), 0.35, False)],
        backdrop="office", color_jitter=0.05, weight=0.9,
    ),
    # ---- outdoor patio: blue sky, stone tiles, sun --------------------------
    "outdoor_patio": Theme(
        name="outdoor_patio",
        sky1=(0.35, 0.55, 0.82), sky2=(0.80, 0.88, 0.96),
        floor1=(0.60, 0.58, 0.54), floor2=(0.52, 0.50, 0.46),
        floor_repeat=8, floor_reflect=0.05,
        table_rgba=(0.58, 0.56, 0.52, 1.0), table_spec=0.2, table_shin=0.3,
        table_tex=((0.60, 0.58, 0.54), (0.54, 0.52, 0.48), 4),   # stone
        headlight=(0.60, 0.40, 0.16),
        lights=[_L((0.7, -0.7, 1.8), (-0.35, 0.35, -1), 0.65, True),   # sun
                _L((-0.4, 0.5, 1.5), (0.2, -0.25, -1), 0.25, False)],
        backdrop=None, color_jitter=0.05, weight=0.9,
    ),
    # ---- dim garage: gray concrete, single warm bulb, high contrast -------
    "garage": Theme(
        name="garage",
        sky1=(0.18, 0.19, 0.21), sky2=(0.34, 0.35, 0.38),
        floor1=(0.30, 0.30, 0.31), floor2=(0.35, 0.35, 0.36),
        floor_repeat=6, floor_reflect=0.03,
        table_rgba=(0.38, 0.37, 0.36, 1.0), table_spec=0.2, table_shin=0.2,
        table_tex=None,
        headlight=(0.34, 0.24, 0.06),
        lights=[_L((0.3, -0.2, 1.7), (-0.12, 0.08, -1), 0.62, True),
                _L((-0.7, 0.8, 1.5), (0.35, -0.4, -1), 0.15, False)],
        backdrop="warehouse", color_jitter=0.04, weight=0.8,
    ),
    # ---- vivid tabletop: colourful mat, bright even light -----------------
    "bright_tabletop": Theme(
        name="bright_tabletop",
        sky1=(0.60, 0.70, 0.80), sky2=(0.90, 0.94, 0.98),
        floor1=(0.20, 0.35, 0.45), floor2=(0.25, 0.42, 0.52),
        floor_repeat=6, floor_reflect=0.0,
        table_rgba=(0.30, 0.45, 0.55, 1.0), table_spec=0.35, table_shin=0.4,
        table_tex=None,
        headlight=(0.66, 0.42, 0.14),
        lights=[_L((0.3, -0.5, 1.8), (-0.15, 0.3, -1), 0.5, True),
                _L((-0.5, 0.6, 1.6), (0.25, -0.35, -1), 0.45, False),
                _L((0.0, 0.2, 1.9), (0, -0.1, -1), 0.35, False)],
        backdrop=None, color_jitter=0.07, weight=0.9,
    ),
}

# Default sampling pool (name -> weight). "plain" is included, as requested.
THEME_NAMES = list(THEMES.keys())
_WEIGHTS = np.array([THEMES[n].weight for n in THEME_NAMES], dtype=float)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def choose_theme(rng) -> str:
    return str(rng.choice(THEME_NAMES, p=_WEIGHTS))


# ------------------------------------------------------------- backdrop bank
# All backdrop geoms are visual-only: contype="0" conaffinity="0" group="2".
# They exist purely for pixels and are guaranteed collision-free.

_VIS = 'contype="0" conaffinity="0" group="2"'


def _wall(name, size, pos, material=None, rgba=None):
    mat = f'material="{material}"' if material else ""
    col = f'rgba="{_fmt(rgba)}"' if rgba is not None else ""
    return (f'<geom name="{name}" type="box" size="{_fmt(size)}" '
            f'pos="{_fmt(pos)}" {mat} {col} {_VIS}/>')


def _prop_box(name, size, pos, rgba):
    return (f'<geom name="{name}" type="box" size="{_fmt(size)}" '
            f'pos="{_fmt(pos)}" rgba="{_fmt(rgba)}" {_VIS}/>')


def _prop_cyl(name, r, h, pos, rgba):
    return (f'<geom name="{name}" type="cylinder" size="{r:.4f} {h:.4f}" '
            f'pos="{_fmt(pos)}" rgba="{_fmt(rgba)}" {_VIS}/>')


def _prop_ell(name, size, pos, rgba):
    return (f'<geom name="{name}" type="ellipsoid" size="{_fmt(size)}" '
            f'pos="{_fmt(pos)}" rgba="{_fmt(rgba)}" {_VIS}/>')


def _backdrop_cyclorama(rng, th):
    c = _jrgb(rng, th.floor1, 0.02)
    back = _wall("bd_back", (0.02, 1.4, WALL_TOP_Z + 0.4),
                 (WALL_BACK_X, 0.15, WALL_TOP_Z - 0.4), rgba=(*c, 1.0))
    return back


def _backdrop_kitchen(rng, th):
    # tiled backsplash behind + a light cabinet block on the +y side, plus a
    # couple of visual-only counter props well outside the workspace.
    tile = _jrgb(rng, (0.86, 0.88, 0.90), 0.04)
    cab = _jrgb(rng, (0.80, 0.74, 0.62), 0.05)
    bowl = _jrgb(rng, (0.75, 0.35, 0.25), 0.08)
    board = _jrgb(rng, (0.55, 0.40, 0.24), 0.05)
    parts = [
        _wall("bd_backsplash", (0.02, 1.3, 0.55),
              (WALL_BACK_X, 0.15, 0.30), rgba=(*tile, 1.0)),
        _wall("bd_upper_cab", (0.18, 1.3, 0.22),
              (WALL_BACK_X - 0.02, 0.15, WALL_TOP_Z - 0.05), rgba=(*cab, 1.0)),
        _wall("bd_side_cab", (1.2, 0.16, 0.55),
              (0.1, WALL_SIDE_Y, 0.15), rgba=(*cab, 1.0)),
        # props (out of the x in [0.3,0.6], y in [-0.25,0.25] workspace):
        _prop_ell("bd_bowl", (0.10, 0.10, 0.055),
                  (0.30, -0.62, 0.055), (*bowl, 1.0)),
        _prop_box("bd_board", (0.14, 0.09, 0.008),
                  (0.55, 0.55, 0.008), (*board, 1.0)),
        _prop_cyl("bd_bottle", 0.028, 0.11,
                  (0.62, -0.50, 0.11), _jrgb(rng, (0.25, 0.45, 0.35), 0.05) + (1.0,)),
    ]
    return "\n    ".join(parts)


def _backdrop_warehouse(rng, th):
    # concrete back wall + a couple of tall shelving/pallet blocks behind.
    wall = _jrgb(rng, (0.40, 0.42, 0.45), 0.04)
    shelf = _jrgb(rng, (0.28, 0.30, 0.34), 0.04)
    crate = _jrgb(rng, (0.55, 0.42, 0.22), 0.05)
    parts = [
        _wall("bd_wall", (0.03, 1.5, WALL_TOP_Z + 0.5),
              (WALL_BACK_X, 0.15, WALL_TOP_Z - 0.5), rgba=(*wall, 1.0)),
        _wall("bd_shelf_l", (0.10, 0.10, WALL_TOP_Z),
              (WALL_BACK_X + 0.15, -0.9, WALL_TOP_Z - 0.75), rgba=(*shelf, 1.0)),
        _wall("bd_shelf_r", (0.10, 0.10, WALL_TOP_Z),
              (WALL_BACK_X + 0.15, 1.1, WALL_TOP_Z - 0.75), rgba=(*shelf, 1.0)),
        _prop_box("bd_crate", (0.14, 0.14, 0.14),
                  (WALL_BACK_X + 0.35, 0.9, 0.12), (*crate, 1.0)),
    ]
    return "\n    ".join(parts)


def _backdrop_workshop(rng, th):
    plank = _jrgb(rng, (0.50, 0.35, 0.20), 0.05)
    tool = _jrgb(rng, (0.30, 0.32, 0.36), 0.05)
    parts = [
        _wall("bd_pegboard", (0.02, 1.2, 0.5),
              (WALL_BACK_X, 0.15, 0.35), rgba=(*plank, 1.0)),
        # a couple of "tools" hanging (thin boxes) on the pegboard
        _prop_box("bd_tool1", (0.012, 0.03, 0.10),
                  (WALL_BACK_X + 0.03, -0.2, 0.5), (*tool, 1.0)),
        _prop_box("bd_tool2", (0.012, 0.03, 0.08),
                  (WALL_BACK_X + 0.03, 0.1, 0.55), (*tool, 1.0)),
        _prop_cyl("bd_can", 0.035, 0.06,
                  (0.62, 0.55, 0.06), _jrgb(rng, (0.6, 0.5, 0.2), 0.05) + (1.0,)),
    ]
    return "\n    ".join(parts)


def _backdrop_lab(rng, th):
    wall = _jrgb(rng, (0.86, 0.89, 0.92), 0.03)
    equip = _jrgb(rng, (0.75, 0.78, 0.82), 0.04)
    parts = [
        _wall("bd_wall", (0.02, 1.3, WALL_TOP_Z + 0.4),
              (WALL_BACK_X, 0.15, WALL_TOP_Z - 0.4), rgba=(*wall, 1.0)),
        _prop_box("bd_instrument", (0.10, 0.16, 0.14),
                  (0.58, 0.58, 0.14), (*equip, 1.0)),
    ]
    return "\n    ".join(parts)


def _backdrop_office(rng, th):
    wall = _jrgb(rng, (0.62, 0.60, 0.56), 0.04)
    mon = _jrgb(rng, (0.10, 0.11, 0.13), 0.02)
    parts = [
        _wall("bd_wall", (0.02, 1.3, WALL_TOP_Z + 0.3),
              (WALL_BACK_X, 0.15, WALL_TOP_Z - 0.3), rgba=(*wall, 1.0)),
        _prop_box("bd_monitor", (0.02, 0.20, 0.13),
                  (WALL_BACK_X + 0.30, -0.5, 0.20), (*mon, 1.0)),
    ]
    return "\n    ".join(parts)


_BACKDROPS: Dict[str, Callable] = {
    "cyclorama": _backdrop_cyclorama,
    "kitchen": _backdrop_kitchen,
    "warehouse": _backdrop_warehouse,
    "workshop": _backdrop_workshop,
    "lab": _backdrop_lab,
    "office": _backdrop_office,
}


# ------------------------------------------------------------------- builder

def build_background(theme_name: str, rng) -> Background:
    """Sample a concrete, jittered Background for one episode."""
    th = THEMES[theme_name]
    cj = th.color_jitter

    sky1 = _jrgb(rng, th.sky1, cj)
    sky2 = _jrgb(rng, th.sky2, cj)
    fl1 = _jrgb(rng, th.floor1, cj)
    fl2 = _jrgb(rng, th.floor2, cj)
    frep = max(2.0, _jit(rng, th.floor_repeat, th.floor_repeat * 0.25))
    freflect = _jit(rng, th.floor_reflect, 0.03, lo=0.0, hi=0.6)

    # ---- assets: skybox, floor, table ----
    if th.sky_builtin == "gradient":
        sky = (f'<texture name="bg_sky" type="skybox" builtin="gradient" '
               f'rgb1="{_fmt(sky1)}" rgb2="{_fmt(sky2)}" width="256" height="256"/>')
    else:
        sky = (f'<texture name="bg_sky" type="skybox" builtin="flat" '
               f'rgb1="{_fmt(sky1)}" rgb2="{_fmt(sky1)}" width="256" height="256"/>')

    floor_tex = (f'<texture name="bg_floor_tex" type="2d" builtin="checker" '
                 f'rgb1="{_fmt(fl1)}" rgb2="{_fmt(fl2)}" width="512" height="512"/>')
    floor_mat = (f'<material name="bg_floor_mat" texture="bg_floor_tex" '
                 f'texrepeat="{frep:.3f} {frep:.3f}" reflectance="{freflect:.3f}" '
                 f'specular="0.1" shininess="0.1"/>')

    table_rgba = _jrgb(rng, th.table_rgba[:3], cj * 0.6) + (1.0,)
    tspec = _jit(rng, th.table_spec, 0.08, lo=0.0, hi=1.0)
    tshin = _jit(rng, th.table_shin, 0.1, lo=0.0, hi=1.0)
    if th.table_tex is not None:
        t1 = _jrgb(rng, th.table_tex[0], cj * 0.5)
        t2 = _jrgb(rng, th.table_tex[1], cj * 0.5)
        trep = max(1.0, _jit(rng, th.table_tex[2], th.table_tex[2] * 0.2))
        table_tex = (f'<texture name="bg_table_tex" type="2d" builtin="checker" '
                     f'rgb1="{_fmt(t1)}" rgb2="{_fmt(t2)}" '
                     f'width="512" height="512"/>')
        table_mat = (f'<material name="bg_table_mat" texture="bg_table_tex" '
                     f'texrepeat="{trep:.3f} {trep:.3f}" rgba="{_fmt(table_rgba)}" '
                     f'specular="{tspec:.3f}" shininess="{tshin:.3f}"/>')
        table_asset = table_tex + "\n    " + table_mat
    else:
        table_asset = (f'<material name="bg_table_mat" rgba="{_fmt(table_rgba)}" '
                       f'specular="{tspec:.3f}" shininess="{tshin:.3f}"/>')

    asset_xml = "\n    ".join([sky, floor_tex, floor_mat, table_asset])

    # ---- headlight ----
    hd, ha, hs = th.headlight
    hd = _jit(rng, hd, 0.06, lo=0.05, hi=0.95)
    ha = _jit(rng, ha, 0.05, lo=0.05, hi=0.6)
    headlight_xml = (f'<headlight diffuse="{hd:.3f} {hd:.3f} {hd:.3f}" '
                     f'ambient="{ha:.3f} {ha:.3f} {ha:.3f}" '
                     f'specular="{hs:.3f} {hs:.3f} {hs:.3f}"/>')

    # ---- floor geom ----
    floor_xml = (f'<geom name="floor" type="plane" size="6 6 0.1" '
                 f'pos="0 0 {FLOOR_Z}" material="bg_floor_mat"/>')

    # ---- lights (jittered position + intensity) ----
    light_lines = []
    for i, L in enumerate(th.lights):
        pos = np.asarray(L["pos"], float) + rng.uniform(-0.12, 0.12, 3)
        diff = _jit(rng, L["diff"], 0.08, lo=0.05, hi=0.95)
        light_lines.append(
            f'<light pos="{_fmt(pos)}" dir="{_fmt(L["dir"])}" '
            f'diffuse="{diff:.3f} {diff:.3f} {diff:.3f}" '
            f'castshadow="{"true" if L["shadow"] else "false"}"/>')
    lights_xml = "\n    ".join(light_lines)

    # ---- backdrop ----
    backdrop_xml = ""
    if th.backdrop and th.backdrop in _BACKDROPS:
        backdrop_xml = _BACKDROPS[th.backdrop](rng, th)

    summary = {
        "theme": theme_name,
        "skybox_rgb1": list(sky1), "skybox_rgb2": list(sky2),
        "floor_rgb1": list(fl1), "floor_rgb2": list(fl2),
        "floor_texrepeat": round(frep, 3), "floor_reflectance": round(freflect, 3),
        "table_rgba": list(table_rgba),
        "headlight_diffuse": round(hd, 3), "headlight_ambient": round(ha, 3),
        "n_lights": len(th.lights),
        "has_backdrop": bool(backdrop_xml),
        "note": ("cosmetic only; table geometry+friction, rope, arm, cameras "
                 "and gravity are identical across all themes"),
    }

    return Background(theme=theme_name, asset_xml=asset_xml,
                      headlight_xml=headlight_xml, floor_xml=floor_xml,
                      lights_xml=lights_xml, backdrop_xml=backdrop_xml,
                      table_material="bg_table_mat", summary=summary)
