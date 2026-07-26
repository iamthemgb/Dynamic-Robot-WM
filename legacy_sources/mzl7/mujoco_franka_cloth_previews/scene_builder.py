"""MJCF scene generation: bland-table + flexcomp cloth + attached Franka(s) +
cameras, with per-episode domain randomization of the table color, background
(skybox) color, floor color and light intensity.

Uses the MuJoCo Menagerie franka_emika_panda via MJCF <attach> with per-arm
prefixes (this namespaces bodies/joints/actuators/keyframes automatically, so
full dual-Franka works without name collisions).

Grasp equality constraints reference cloth vertex *body* names, which only
exist after flexcomp expansion — so scenes are compiled in two passes:
  pass 1: compile without grasp equalities, find nearest vertex body names;
  pass 2: regenerate XML with <connect ... active="false"/> equalities,
          add wrist cameras via MjSpec, compile the final model.
"""

import colorsys
import os
from pathlib import Path

import mujoco
import numpy as np

# Local Menagerie checkout (searched paths; first hit wins).
_MENAGERIE_CANDIDATES = [
    "/gpfs/radev/project/sous/zl664/demo_mujoco_arm_gripper/third_party/mujoco_menagerie",
    os.path.expanduser("~/world_model/assets/mujoco_menagerie"),
]


def find_panda_xml() -> str:
    for root in _MENAGERIE_CANDIDATES:
        p = Path(root) / "franka_emika_panda" / "panda.xml"
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        "franka_emika_panda/panda.xml not found in any of: "
        + ", ".join(_MENAGERIE_CANDIDATES))


PANDA_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TCP_OFFSET = 0.103  # m from hand frame origin to fingertip pinch point, along hand +z


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def lookat_quat(campos, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (wxyz) for a MuJoCo camera at `campos` looking at `target`.

    MuJoCo cameras look along -z with +y up.
    """
    campos, target, up = map(np.asarray, (campos, target, up))
    z = campos - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:      # view parallel to up: pick another up
        x = np.cross([0.0, 1.0, 0.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


# --------------------------------------------------- scene appearance (DR)

# Default appearance == the original hardcoded bland look, so build_scene_xml
# with no scene_appearance reproduces the pre-randomization scene exactly.
DEFAULT_SCENE_APPEARANCE = {
    "table_rgba": (0.45, 0.42, 0.38, 1.0),
    "skybox_rgb1": (0.45, 0.55, 0.65),
    "skybox_rgb2": (0.85, 0.90, 0.95),
    "floor_rgb1": (0.22, 0.24, 0.26),
    "floor_rgb2": (0.28, 0.30, 0.32),
    "light_scale": 1.0,
}


def _muted_rgb(rng, s_range, v_range):
    h = rng.uniform(0.0, 1.0)
    s = rng.uniform(*s_range)
    v = rng.uniform(*v_range)
    return tuple(float(c) for c in colorsys.hsv_to_rgb(h, s, v))


def random_scene_appearance(rng) -> dict:
    """Per-episode table / background / floor colors and light intensity.

    Table: muted painted/wood-ish surface. Skybox: a soft two-tint gradient
    (background color). Floor: subtle checker near-neutral. Light: +-15%.
    """
    table = (*_muted_rgb(rng, (0.10, 0.45), (0.30, 0.70)), 1.0)
    sky1 = _muted_rgb(rng, (0.10, 0.55), (0.35, 0.70))
    sky2 = _muted_rgb(rng, (0.05, 0.35), (0.75, 0.98))
    base_floor = _muted_rgb(rng, (0.02, 0.20), (0.18, 0.35))
    floor1 = base_floor
    floor2 = tuple(float(min(1.0, c * 1.25 + 0.03)) for c in base_floor)
    return {
        "table_rgba": table,
        "skybox_rgb1": sky1,
        "skybox_rgb2": sky2,
        "floor_rgb1": floor1,
        "floor_rgb2": floor2,
        "light_scale": float(rng.uniform(0.85, 1.15)),
    }


def _cameras_xml(center, dual: bool) -> str:
    """Fixed front + top cameras aimed at the workspace center."""
    c = np.asarray(center, dtype=float)
    c[2] = 0.05
    if dual:
        front_pos = c + np.array([0.0, -0.95, 0.56])
    else:
        front_pos = c + np.array([0.60, -0.48, 0.38])
    top_pos = c + np.array([0.0, 0.0, 1.45])
    fq = lookat_quat(front_pos, c)
    tq = lookat_quat(top_pos, c, up=(0.0, 1.0, 0.0))
    return (
        f'<camera name="front" pos="{_fmt(front_pos)}" quat="{_fmt(fq)}" fovy="45"/>\n'
        f'<camera name="top" pos="{_fmt(top_pos)}" quat="{_fmt(tq)}" fovy="42"/>\n'
    )


def _cloth_appearance_xml(cloth):
    """(asset_xml, flexcomp_attr) for the cloth's color/texture."""
    if cloth.texture_kind == "flat":
        return "", f'rgba="{_fmt(cloth.rgba)}"'
    builtin = "checker" if cloth.texture_kind == "checker" else "gradient"
    rep = float(cloth.texrepeat)
    asset = (
        f'<texture name="cloth_tex" type="2d" builtin="{builtin}" '
        f'rgb1="{_fmt(cloth.rgba[:3])}" rgb2="{_fmt(cloth.rgba2[:3])}" '
        f'width="64" height="64"/>\n'
        f'<material name="cloth_mat" texture="cloth_tex" '
        f'texrepeat="{rep:.3g} {rep:.3g}"/>\n')
    return asset, 'material="cloth_mat"'


def build_scene_xml(task, material, dt_sim: float, panda_xml: str,
                    grasp_connects=None, scene_appearance=None) -> str:
    """Full scene MJCF. `grasp_connects` is None (pass 1) or a list of
    dicts(name, vertex_body, hand_body) for pass 2. `scene_appearance` (None ->
    DEFAULT_SCENE_APPEARANCE) sets table/skybox/floor colors + light scale."""
    ap = {**DEFAULT_SCENE_APPEARANCE, **(scene_appearance or {})}
    cloth = task.cloth
    nx, ny = cloth.count
    dual = len(task.arms) > 1

    attach_xml = ""
    for arm in task.arms:
        quat = np.array([np.cos(arm.base_yaw / 2), 0, 0, np.sin(arm.base_yaw / 2)])
        attach_xml += (
            f'<frame pos="{_fmt(arm.base_pos)}" quat="{_fmt(quat)}">\n'
            f'  <attach model="panda" body="link0" prefix="{arm.prefix}"/>\n'
            f'</frame>\n')

    eq_xml = ""
    if grasp_connects:
        for gc in grasp_connects:
            eq_xml += (
                f'<connect name="{gc["name"]}" body1="{gc["vertex_body"]}" '
                f'body2="{gc["hand_body"]}" anchor="0 0 0" active="false" '
                f'solref="0.01 1"/>\n')

    # friction: sliding, torsional, rolling. Cloth-side friction; effective
    # pair friction in MuJoCo defaults to elementwise max, so the cloth value
    # dominates against the default-friction table/gripper geoms.
    fric = max(material.fric_table, material.fric_grip)

    cloth_asset_xml, cloth_appearance_attr = _cloth_appearance_xml(cloth)

    ls = ap["light_scale"]
    d1 = _fmt(np.array([0.5, 0.5, 0.5]) * ls)
    d2 = _fmt(np.array([0.35, 0.35, 0.35]) * ls)

    return f"""
<mujoco model="franka_cloth_{task.task_id}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt_sim}" integrator="implicitfast" solver="CG"
          tolerance="1e-6" gravity="0 0 -9.81"/>
  <extension>
    <plugin plugin="mujoco.elasticity.shell"/>
  </extension>

  <visual>
    <global offwidth="1920" offheight="1088"/>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.32 0.32 0.32" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
  </visual>

  <asset>
    <model name="panda" file="{panda_xml}"/>
    <texture type="skybox" builtin="gradient" rgb1="{_fmt(ap['skybox_rgb1'])}"
             rgb2="{_fmt(ap['skybox_rgb2'])}" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="{_fmt(ap['floor_rgb1'])}" rgb2="{_fmt(ap['floor_rgb2'])}"
             width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8"/>
    <material name="table_mat" rgba="{_fmt(ap['table_rgba'])}" specular="0.2" shininess="0.2"/>
    {cloth_asset_xml}
  </asset>

  <worldbody>
    <light pos="0.4 -0.6 1.6" dir="-0.2 0.4 -1" diffuse="{d1}" castshadow="true"/>
    <light pos="-0.6 0.8 1.4" dir="0.3 -0.4 -1" diffuse="{d2}" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.75" material="floor_mat"/>
    <geom name="table" type="box" size="1.1 1.1 0.02" pos="0.15 0.3 -0.02"
          material="table_mat" friction="1.0 0.005 0.0001"/>

    {_cameras_xml(task.workspace_center, dual)}

    {attach_xml}

    <flexcomp name="cloth" type="grid" count="{nx} {ny} 1"
              spacing="{cloth.spacing[0]} {cloth.spacing[1]} {min(cloth.spacing)}"
              pos="{_fmt(cloth.center)}" radius="0.002"
              mass="{material.mass}" {cloth_appearance_attr}>
      <contact condim="3" solref="0.005 1" solimp="0.95 0.99 0.0001"
               friction="{fric} 0.005 0.0001" selfcollide="auto"/>
      <edge equality="true" damping="{material.edge_damping}"/>
      <plugin plugin="mujoco.elasticity.shell">
        <config key="young" value="{material.young}"/>
        <config key="poisson" value="{material.poisson}"/>
        <config key="thickness" value="{material.thickness}"/>
      </plugin>
    </flexcomp>

    {task.extra_worldbody_xml}
  </worldbody>

  <equality>
    {eq_xml}
  </equality>
</mujoco>
"""


def resolve_grasp_connects(model, task):
    """Pass-1 helper: nearest cloth vertex body for each grasp event.

    Uses the *initial* (grid-exact) vertex positions, so vertex choice is
    identical across physics variants of the same seed."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    verts = data.flexvert_xpos.copy()      # (nvert, 3)
    connects = []
    for i, g in enumerate(task.grasps):
        d2 = np.sum((verts[:, :2] - np.asarray(g.point)[:2]) ** 2, axis=1)
        vidx = int(np.argmin(d2))
        body_id = int(model.flex_vertbodyid[vidx])
        vbody = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        prefix = next(a.prefix for a in task.arms if a.name == g.arm)
        connects.append({
            "name": f"grasp_{g.arm}_{i}",
            "vertex_body": vbody,
            "vertex_index": vidx,
            "hand_body": f"{prefix}hand",
            "grasp_event": g,
        })
    return connects


def add_wrist_cameras(spec: mujoco.MjSpec, task) -> list:
    """Add a wrist camera to each arm's hand body via MjSpec. Returns the
    camera names added (empty list on failure -> wrist_camera=False)."""
    names = []
    for arm in task.arms:
        try:
            body = spec.body(f"{arm.prefix}hand")
            cam = body.add_camera()
            cam.name = f"{arm.prefix}wrist"
            # mounted behind/beside the hand, looking at the TCP pinch point
            campos = np.array([0.11, 0.0, -0.03])
            target = np.array([0.0, 0.0, TCP_OFFSET + 0.04])
            cam.pos = campos
            cam.quat = lookat_quat(campos, target, up=(0.0, 0.0, 1.0))
            cam.fovy = 70
            names.append(cam.name)
        except Exception as e:  # noqa: BLE001 - preview robustness
            print(f"[warn] wrist camera for {arm.prefix}hand failed: {e}")
    return names


def compile_episode_model(task, material, dt_sim, panda_xml, scene_xml_path=None,
                          scene_appearance=None):
    """Two-pass compile. Returns (model, final_xml, grasp_connects, wrist_cams,
    scene_meta). `scene_appearance` (None -> default look) sets the randomized
    table/background/floor colors and light intensity."""
    ap = {**DEFAULT_SCENE_APPEARANCE, **(scene_appearance or {})}

    xml1 = build_scene_xml(task, material, dt_sim, panda_xml,
                           scene_appearance=ap)
    model1 = mujoco.MjModel.from_xml_string(xml1)
    connects = resolve_grasp_connects(model1, task)

    xml2 = build_scene_xml(task, material, dt_sim, panda_xml,
                           grasp_connects=connects, scene_appearance=ap)
    spec = mujoco.MjSpec.from_string(xml2)
    wrist_cams = add_wrist_cameras(spec, task)
    model = spec.compile()

    if scene_xml_path is not None:
        Path(scene_xml_path).write_text(spec.to_xml())
    return model, xml2, connects, wrist_cams, ap
