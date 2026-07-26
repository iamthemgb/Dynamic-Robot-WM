"""MJCF scene generation: table + composite-cable rope + attached Franka.

Rope model: MuJoCo composite ``type="cable"`` (rope_model_type =
"mujoco_composite_cable") — rigid capsule links + ball joints, bend/twist
elasticity from the ``mujoco.elasticity.cable`` plugin. The composite must
live at worldbody top level because ``initial="free"`` adds a free joint to
the first link; placement is via ``offset`` and the rope extends along +x.

The endpoint grasp is a pre-declared inactive ``connect`` equality between
the endpoint body (``RB_first``) and the hand, toggled at runtime.
"""

import os
from pathlib import Path

import mujoco
import numpy as np

from .backgrounds import Background, build_background

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
TCP_OFFSET = 0.103        # m, hand frame origin -> fingertip pinch point
ARM_PREFIX = "A0_"
ROPE_PREFIX = "R"
ROPE_SEGMENTS = 41        # composite count (bodies RB_first, RB_1..., RB_last)


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def lookat_quat(campos, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (wxyz) for a MuJoCo camera at `campos` looking at `target`."""
    campos, target, up = map(np.asarray, (campos, target, up))
    z = campos - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross([0.0, 1.0, 0.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat


def _cameras_xml(center) -> str:
    c = np.asarray(center, dtype=float)
    c[2] = 0.05
    front_pos = c + np.array([0.60, -0.48, 0.38])
    top_pos = c + np.array([0.0, 0.0, 1.35])
    fq = lookat_quat(front_pos, c)
    tq = lookat_quat(top_pos, c, up=(0.0, 1.0, 0.0))
    return (
        f'<camera name="front" pos="{_fmt(front_pos)}" quat="{_fmt(fq)}" fovy="45"/>\n'
        f'<camera name="top" pos="{_fmt(top_pos)}" quat="{_fmt(tq)}" fovy="42"/>\n'
    )


def build_scene_xml(task, material, dt_sim: float, panda_xml: str,
                    background: Background) -> str:
    rope_z = material.radius + 0.001
    offset = task.rope.offset.copy()
    offset[2] = rope_z

    extra = task.extra_worldbody_xml
    extra = extra.replace("__POST_FRICTION__", str(material.fric_post))
    extra = extra.replace("__RING_FRICTION__", str(material.fric_ring))

    bg = background
    # The table geom's *friction* is FIXED (1.0 ...) regardless of the visual
    # theme, so dynamics are bit-identical across backgrounds; only its
    # material (appearance) comes from the background.
    return f"""
<mujoco model="franka_rope_{task.task_id}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt_sim}" integrator="implicitfast" solver="CG"
          tolerance="1e-6" gravity="0 0 -9.81"/>
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>

  <visual>
    <global offwidth="1920" offheight="1088"/>
    {bg.headlight_xml}
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
  </visual>

  <asset>
    <model name="panda" file="{panda_xml}"/>
    {bg.asset_xml}
  </asset>

  <worldbody>
    {bg.lights_xml}
    {bg.floor_xml}
    <geom name="table" type="box" size="1.1 1.1 0.02" pos="0.15 0 -0.02"
          material="{bg.table_material}" friction="1.0 0.005 0.0001"/>
    {bg.backdrop_xml}

    {_cameras_xml(task.workspace_center)}

    <attach model="panda" body="link0" prefix="{ARM_PREFIX}"/>

    <composite type="cable" curve="s" count="{ROPE_SEGMENTS} 1 1"
               size="{material.length}" prefix="{ROPE_PREFIX}"
               initial="free" offset="{_fmt(offset)}">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="{material.shear_twist}"/>
        <config key="bend" value="{material.young_bend}"/>
      </plugin>
      <joint kind="main" damping="{material.joint_damping}"/>
      <geom type="capsule" size="{material.radius}" density="{material.density}"
            friction="{material.fric_table} 0.005 0.0001" condim="3"
            rgba="{_fmt(material.rgba)}"/>
    </composite>

    {extra}
  </worldbody>

  <equality>
    <connect name="grasp_endpoint" body1="{ROPE_PREFIX}B_first"
             body2="{ARM_PREFIX}hand" anchor="0 0 0" active="false"
             solref="0.01 1"/>
  </equality>
</mujoco>
"""


def add_wrist_camera(spec: mujoco.MjSpec) -> list:
    """Wrist camera on the hand body via MjSpec; [] on failure."""
    try:
        body = spec.body(f"{ARM_PREFIX}hand")
        cam = body.add_camera()
        cam.name = f"{ARM_PREFIX}wrist"
        campos = np.array([0.11, 0.0, -0.03])
        target = np.array([0.0, 0.0, TCP_OFFSET + 0.04])
        cam.pos = campos
        cam.quat = lookat_quat(campos, target, up=(0.0, 0.0, 1.0))
        cam.fovy = 70
        return [cam.name]
    except Exception as e:  # noqa: BLE001 - preview robustness
        print(f"[warn] wrist camera failed: {e}")
        return []


def compile_episode_model(task, material, dt_sim, panda_xml, background,
                          scene_xml_path=None):
    """Returns (model, wrist_cam_names). Saves final MJCF if path given."""
    xml = build_scene_xml(task, material, dt_sim, panda_xml, background)
    spec = mujoco.MjSpec.from_string(xml)
    wrist_cams = add_wrist_camera(spec)
    model = spec.compile()
    if scene_xml_path is not None:
        Path(scene_xml_path).write_text(spec.to_xml())
    return model, wrist_cams


def rope_body_ids(model) -> list:
    """Ordered rope segment body ids: RB_first, RB_1, ..., RB_last."""
    ids = []
    first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{ROPE_PREFIX}B_first")
    assert first >= 0, "rope first body not found"
    ids.append(first)
    i = 1
    while True:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{ROPE_PREFIX}B_{i}")
        if b < 0:
            break
        ids.append(b)
        i += 1
    last = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{ROPE_PREFIX}B_last")
    assert last >= 0, "rope last body not found"
    ids.append(last)
    return ids


def rope_geom_ids(model) -> set:
    """Geom ids of rope capsules (names 'RG0', 'RG1', ...)."""
    out = set()
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(f"{ROPE_PREFIX}G"):
            out.add(g)
    return out
