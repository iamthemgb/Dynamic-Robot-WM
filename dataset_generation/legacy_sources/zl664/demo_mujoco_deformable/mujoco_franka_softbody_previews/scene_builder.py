"""MJCF scene generation: table + volumetric flexcomp soft object + attached
Franka Panda, plus per-task box / hinged-lid fixtures.

Soft body: ``flexcomp type="grid" dim="3"`` (softbody_model_type =
"mujoco_flexcomp_grid") — tetrahedral FEM with built-in continuum elasticity.
Elasticity damping is 0 (unstable on MuJoCo 3.3.1, see materials.py); damping
comes from flex edge dashpots.
"""

import os
from pathlib import Path

import mujoco
import numpy as np

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
FLEX_NAME = "sponge"
GRIPPER_MAX_WIDTH = 0.08  # m, Panda finger travel 2 x 0.04


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
    front_pos = c + np.array([0.55, -0.50, 0.35])
    top_pos = c + np.array([0.0, 0.0, 1.30])
    fq = lookat_quat(front_pos, c)
    tq = lookat_quat(top_pos, c, up=(0.0, 1.0, 0.0))
    return (
        f'<camera name="front" pos="{_fmt(front_pos)}" quat="{_fmt(fq)}" fovy="45"/>\n'
        f'<camera name="top" pos="{_fmt(top_pos)}" quat="{_fmt(tq)}" fovy="42"/>\n'
    )


def episode_dt(task, material) -> float:
    return material.dt_sim * task.obj.dt_scale


def flexcomp_xml(obj, material, dt_sim: float) -> str:
    """Volumetric soft-body XML. obj = tasks.ObjectSpec.

    Contact solref timeconst = 2*dt_sim (MuJoCo stability floor): contact
    stiffness is inertia-scaled and flex vertices weigh ~0.1 g, so the
    default 0.01 s timeconst lets rigid bodies tunnel through the flex.
    """
    mass = material.mass_for_volume(obj.volume)
    return f"""
    <flexcomp name="{FLEX_NAME}" type="grid" dim="3"
              count="{obj.count[0]} {obj.count[1]} {obj.count[2]}"
              spacing="{_fmt(obj.spacing)}" pos="{_fmt(obj.pos)}"
              radius="{obj.radius}" mass="{mass:.5f}"
              rgba="{_fmt(material.rgba)}">
      <elasticity young="{material.young}" poisson="{material.poisson}"
                  damping="0"/>
      <contact solref="{2 * dt_sim:.6g} 1" selfcollide="none" internal="true"
               friction="{material.friction} 0.005 0.0001" condim="3"/>
      <edge damping="{material.edge_damping}"/>
    </flexcomp>
"""


def build_scene_xml(task, material, panda_xml: str) -> str:
    dt_sim = episode_dt(task, material)
    return f"""
<mujoco model="franka_softbody_{task.task_id}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt_sim}" integrator="implicitfast" solver="CG"
          tolerance="1e-6" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1920" offheight="1088"/>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.32 0.32 0.32" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
    <map znear="0.01"/>
  </visual>

  <asset>
    <model name="panda" file="{panda_xml}"/>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.55 0.65"
             rgb2="0.85 0.9 0.95" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.22 0.24 0.26"
             rgb2="0.28 0.30 0.32" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8"/>
    <material name="table_mat" rgba="0.45 0.42 0.38 1" specular="0.2" shininess="0.2"/>
    <material name="box_mat" rgba="0.55 0.40 0.22 1" specular="0.15" shininess="0.15"/>
    <material name="lid_mat" rgba="0.65 0.48 0.28 1" specular="0.15" shininess="0.15"/>
  </asset>

  <worldbody>
    <light pos="0.4 -0.6 1.6" dir="-0.2 0.4 -1" diffuse="0.5 0.5 0.5" castshadow="true"/>
    <light pos="-0.6 0.8 1.4" dir="0.3 -0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.75" material="floor_mat"/>
    <geom name="table" type="box" size="1.1 1.1 0.02" pos="0.15 0 -0.02"
          material="table_mat" friction="1.0 0.005 0.0001"/>

    {_cameras_xml(task.workspace_center)}

    <attach model="panda" body="link0" prefix="{ARM_PREFIX}"/>

    {flexcomp_xml(task.obj, material, dt_sim)}

    {task.extra_worldbody_xml}
  </worldbody>

  {task.extra_actuator_xml}
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


GRIPPER_KP_SCALE = 12.0   # see strengthen_gripper()


def strengthen_gripper(spec: mujoco.MjSpec, scale: float = GRIPPER_KP_SCALE):
    """Scale the Menagerie gripper position servo (actuator8) gains.

    The stock servo (kp=100 N/m on the 'split' tendon) saturates at ~2.4 N
    pinch — enough to hold rigid parts but too weak to visibly squeeze foam
    (the real Panda gripper sustains ~70 N). Scaling gainprm/biasprm
    together preserves the ctrl(0..255) -> width mapping and raises squeeze
    authority to ~8 N per finger, still well under hardware capability.
    forcerange (+-100 N) is untouched. Recorded in robot metadata as
    gripper_servo_modification.
    """
    for act in spec.actuators:
        if act.name.endswith("actuator8"):
            act.gainprm[0] *= scale
            act.biasprm[1] *= scale
            act.biasprm[2] *= scale
            return True
    print("[warn] actuator8 not found; gripper servo left stock")
    return False


def compile_episode_model(task, material, panda_xml, scene_xml_path=None):
    """Returns (model, wrist_cam_names). Saves final MJCF if path given."""
    xml = build_scene_xml(task, material, panda_xml)
    spec = mujoco.MjSpec.from_string(xml)
    wrist_cams = add_wrist_camera(spec)
    strengthen_gripper(spec)
    model = spec.compile()
    if scene_xml_path is not None:
        Path(scene_xml_path).write_text(spec.to_xml())
    return model, wrist_cams


# ------------------------------------------------------------------ flex API

def flex_id(model) -> int:
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, FLEX_NAME)
    assert fid >= 0, f"flex {FLEX_NAME!r} not found"
    return fid


def flex_vertices(data) -> np.ndarray:
    """(nvert, 3) world vertex positions (single flex in scene)."""
    return data.flexvert_xpos


def flex_elements(model) -> np.ndarray:
    """(nelem, 4) tetrahedron vertex indices."""
    return model.flex_elem.reshape(-1, 4)


def tet_volume(verts: np.ndarray, elem: np.ndarray) -> float:
    """Total volume of the tetrahedral mesh [m^3]."""
    a = verts[elem[:, 0]]
    b = verts[elem[:, 1]]
    c = verts[elem[:, 2]]
    d = verts[elem[:, 3]]
    return float(np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a),
                                  d - a)).sum() / 6.0)


KEYPOINT_NAMES = ["top_center", "bottom_center", "left_y", "right_y",
                  "front_x", "back_x", "center"]


def keypoint_indices(verts0: np.ndarray) -> np.ndarray:
    """Vertex indices of the 6 face centers, chosen from the initial cloud.

    Order matches KEYPOINT_NAMES[:-1]; the 7th keypoint ("center") is the
    vertex-cloud mean, computed per frame (equal vertex masses -> true COM).
    """
    c = verts0.mean(axis=0)
    half = (verts0.max(axis=0) - verts0.min(axis=0)) / 2.0
    targets = [c + [0, 0, half[2]], c - [0, 0, half[2]],
               c - [0, half[1], 0], c + [0, half[1], 0],
               c - [half[0], 0, 0], c + [half[0], 0, 0]]
    return np.array([np.linalg.norm(verts0 - t, axis=1).argmin()
                     for t in targets])
