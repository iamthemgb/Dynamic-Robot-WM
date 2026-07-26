from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    UNSIGNED_INT,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Asset,
    Attributes,
    Buffer,
    BufferView,
    Camera,
    GLTF2,
    Material,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Perspective,
    Primitive,
    Scene,
)

from .utils import (
    AXIS_CONVERSION_NOTE,
    as_float_list,
    mat_to_quat_xyzw,
    mujoco_mat_to_gltf_quat_xyzw,
    mujoco_pos_to_gltf,
    mujoco_vertices_to_gltf,
)


@dataclass
class RecordedGeom:
    geom_id: int
    name: str
    geom_type: int
    rgba: tuple[float, float, float, float]
    positions: list[np.ndarray]
    rotations_xyzw: list[np.ndarray]


@dataclass
class RecordedCamera:
    camera_id: int
    name: str
    parent_body: str
    fovy: float
    resolution: tuple[int, int] | None
    local_pos: np.ndarray
    local_mat: np.ndarray
    local_quat_xyzw: np.ndarray
    positions: list[np.ndarray]
    rotations: list[np.ndarray]
    rotations_xyzw: list[np.ndarray]
    positions_gltf: list[np.ndarray]
    rotations_gltf_xyzw: list[np.ndarray]


def compute_pinhole_intrinsics(width: int, height: int, fovy_degrees: float) -> dict:
    fy = float(height) / (2.0 * math.tan(math.radians(float(fovy_degrees)) / 2.0))
    fx = fy
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "matrix": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        "model": "pinhole",
        "assumes_square_pixels": True,
        "source": "computed_from_vertical_fovy_and_resolution",
    }


class TrajectoryRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        fps: int,
        *,
        camera_names: list[str] | tuple[str, ...] = (),
        camera_resolutions: dict[str, tuple[int, int]] | None = None,
    ):
        self.model = model
        self.fps = fps
        self.times: list[float] = []
        self.geom_ids = self._visible_geom_ids(model)
        self.geoms: dict[int, RecordedGeom] = {}
        self.cameras: dict[str, RecordedCamera] = {}
        for gid in self.geom_ids:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
            mat_id = int(model.geom_matid[gid])
            rgba_source = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[gid]
            rgba = tuple(float(v) for v in rgba_source)
            self.geoms[gid] = RecordedGeom(
                geom_id=gid,
                name=name,
                geom_type=int(model.geom_type[gid]),
                rgba=rgba,
                positions=[],
                rotations_xyzw=[],
            )
        camera_resolutions = camera_resolutions or {}
        for name in camera_names:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if camera_id < 0:
                raise KeyError(f"MuJoCo camera not found: {name}")
            parent_body_id = int(model.cam_bodyid[camera_id])
            parent_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id) or f"body_{parent_body_id}"
            local_mat_flat = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(local_mat_flat, model.cam_quat[camera_id])
            local_mat = local_mat_flat.reshape(3, 3)
            self.cameras[name] = RecordedCamera(
                camera_id=int(camera_id),
                name=name,
                parent_body=parent_body,
                fovy=float(model.cam_fovy[camera_id]),
                resolution=camera_resolutions.get(name),
                local_pos=model.cam_pos[camera_id].copy(),
                local_mat=local_mat,
                local_quat_xyzw=mat_to_quat_xyzw(local_mat).astype(np.float32),
                positions=[],
                rotations=[],
                rotations_xyzw=[],
                positions_gltf=[],
                rotations_gltf_xyzw=[],
            )

    @staticmethod
    def _visible_geom_ids(model: mujoco.MjModel) -> list[int]:
        ids: list[int] = []
        for gid in range(model.ngeom):
            group = int(model.geom_group[gid])
            alpha = float(model.geom_rgba[gid][3])
            if alpha <= 0.01:
                continue
            if group == 3:
                continue
            ids.append(gid)
        return ids

    @property
    def frame_count(self) -> int:
        return len(self.times)

    def record(self, data: mujoco.MjData, *, frame_time: float) -> None:
        self.times.append(float(frame_time))
        for gid in self.geom_ids:
            geom = self.geoms[gid]
            geom.positions.append(mujoco_pos_to_gltf(data.geom_xpos[gid]).astype(np.float32))
            geom.rotations_xyzw.append(mujoco_mat_to_gltf_quat_xyzw(data.geom_xmat[gid]).astype(np.float32))
        for camera in self.cameras.values():
            mat = data.cam_xmat[camera.camera_id].reshape(3, 3).copy()
            camera.positions.append(data.cam_xpos[camera.camera_id].copy())
            camera.rotations.append(mat)
            camera.rotations_xyzw.append(mat_to_quat_xyzw(mat).astype(np.float32))
            camera.positions_gltf.append(mujoco_pos_to_gltf(data.cam_xpos[camera.camera_id]).astype(np.float32))
            camera.rotations_gltf_xyzw.append(mujoco_mat_to_gltf_quat_xyzw(mat).astype(np.float32))

    def camera_metadata(self) -> dict:
        metadata = {}
        for name, camera in self.cameras.items():
            width_height = camera.resolution
            local_optical_axis = -camera.local_mat[:, 2]
            camera_entry = {
                "camera_name": camera.name,
                "parent_body": camera.parent_body,
                "local_pose_relative_to_parent_body": {
                    "position": as_float_list(camera.local_pos),
                    "rotation_matrix": camera.local_mat.astype(float).tolist(),
                    "quaternion_xyzw": as_float_list(camera.local_quat_xyzw),
                    "optical_axis": as_float_list(local_optical_axis),
                },
                "world_pose_per_frame": [
                    {
                        "time_s": float(t),
                        "position": as_float_list(pos),
                        "rotation_matrix": rot.astype(float).tolist(),
                        "quaternion_xyzw": as_float_list(quat),
                    }
                    for t, pos, rot, quat in zip(self.times, camera.positions, camera.rotations, camera.rotations_xyzw)
                ],
                "fovy": camera.fovy,
                "resolution": list(width_height) if width_height else None,
                "intrinsics": compute_pinhole_intrinsics(*width_height, camera.fovy) if width_height else None,
            }
            metadata[name] = camera_entry
        return metadata

    def camera_gltf_extras(self) -> dict:
        extras = {}
        for name, camera in self.cameras.items():
            extras[name] = {
                "camera_name": camera.name,
                "source_parent_body": camera.parent_body,
                "fovy": camera.fovy,
                "resolution": list(camera.resolution) if camera.resolution else None,
                "axis_conversion": AXIS_CONVERSION_NOTE,
                "world_pose_gltf_per_frame": [
                    {
                        "time_s": float(t),
                        "translation": as_float_list(pos),
                        "rotation_xyzw": as_float_list(rot),
                    }
                    for t, pos, rot in zip(self.times, camera.positions_gltf, camera.rotations_gltf_xyzw)
                ],
            }
        return extras


class BufferBuilder:
    def __init__(self):
        self.blob = bytearray()
        self.views: list[BufferView] = []
        self.accessors: list[Accessor] = []

    def _align(self) -> None:
        while len(self.blob) % 4:
            self.blob.append(0)

    def add_array(self, array: np.ndarray, *, target: int | None, accessor_type: str, component_type: int, name: str | None = None) -> int:
        self._align()
        offset = len(self.blob)
        contiguous = np.ascontiguousarray(array)
        raw = contiguous.tobytes()
        self.blob.extend(raw)
        view_index = len(self.views)
        self.views.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(raw), target=target, name=name))
        count = int(contiguous.shape[0])
        kwargs = {}
        if accessor_type in {"VEC2", "VEC3", "VEC4"} and contiguous.size:
            kwargs["min"] = contiguous.reshape(count, -1).min(axis=0).astype(float).tolist()
            kwargs["max"] = contiguous.reshape(count, -1).max(axis=0).astype(float).tolist()
        elif accessor_type == "SCALAR" and contiguous.size:
            kwargs["min"] = [float(contiguous.min())]
            kwargs["max"] = [float(contiguous.max())]
        accessor_index = len(self.accessors)
        self.accessors.append(
            Accessor(
                bufferView=view_index,
                byteOffset=0,
                componentType=component_type,
                count=count,
                type=accessor_type,
                name=name,
                **kwargs,
            )
        )
        return accessor_index


def _mesh_from_mujoco_geom(model: mujoco.MjModel, gid: int) -> trimesh.Trimesh:
    geom_type = int(model.geom_type[gid])
    size = model.geom_size[gid].copy()

    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        mesh = trimesh.creation.uv_sphere(radius=float(size[0]), count=[16, 16])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        mesh = trimesh.creation.box(extents=2.0 * size[:3])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(2.0 * size[1]), sections=24)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(max(0.001, 2.0 * size[1])), count=[12, 12])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        extent_x = float(size[0] if size[0] > 0 else 3.0)
        extent_y = float(size[1] if size[1] > 0 else 3.0)
        mesh = trimesh.creation.box(extents=(2.0 * extent_x, 2.0 * extent_y, 0.01))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            mesh = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
        else:
            vadr = int(model.mesh_vertadr[mesh_id])
            vnum = int(model.mesh_vertnum[mesh_id])
            fadr = int(model.mesh_faceadr[mesh_id])
            fnum = int(model.mesh_facenum[mesh_id])
            vertices = model.mesh_vert[vadr : vadr + vnum].copy()
            vertices *= model.mesh_scale[mesh_id]
            faces = model.mesh_face[fadr : fadr + fnum].copy()
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    else:
        mesh = trimesh.creation.box(extents=(0.04, 0.04, 0.04))

    vertices = mujoco_vertices_to_gltf(mesh.vertices)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)


def _add_material(gltf: GLTF2, rgba: tuple[float, float, float, float]) -> int:
    key = tuple(round(float(v), 4) for v in rgba)
    cache = getattr(gltf, "_material_cache", None)
    if cache is None:
        cache = {}
        setattr(gltf, "_material_cache", cache)
    if key in cache:
        return cache[key]
    mat = Material(
        name=f"rgba_{len(gltf.materials):03d}",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[float(v) for v in rgba],
            metallicFactor=0.0,
            roughnessFactor=0.78,
        ),
        alphaMode="BLEND" if rgba[3] < 0.99 else "OPAQUE",
        doubleSided=False,
    )
    idx = len(gltf.materials)
    gltf.materials.append(mat)
    cache[key] = idx
    return idx


def _add_mesh(gltf: GLTF2, builder: BufferBuilder, mesh: trimesh.Trimesh, material_index: int, name: str) -> int:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    indices = np.asarray(mesh.faces.reshape(-1), dtype=np.uint32)
    pos_acc = builder.add_array(vertices, target=ARRAY_BUFFER, accessor_type="VEC3", component_type=FLOAT, name=f"{name}_POSITION")
    normal_acc = builder.add_array(normals, target=ARRAY_BUFFER, accessor_type="VEC3", component_type=FLOAT, name=f"{name}_NORMAL")
    idx_acc = builder.add_array(indices, target=ELEMENT_ARRAY_BUFFER, accessor_type="SCALAR", component_type=UNSIGNED_INT, name=f"{name}_INDICES")
    primitive = Primitive(attributes=Attributes(POSITION=pos_acc, NORMAL=normal_acc), indices=idx_acc, material=material_index)
    mesh_index = len(gltf.meshes)
    gltf.meshes.append(Mesh(primitives=[primitive], name=name))
    return mesh_index


def _add_recorded_cameras(
    gltf: GLTF2,
    builder: BufferBuilder,
    animation: Animation,
    time_accessor: int,
    recorder: TrajectoryRecorder,
) -> None:
    if not recorder.cameras:
        return
    if gltf.cameras is None:
        gltf.cameras = []

    for recorded in recorder.cameras.values():
        if not recorded.positions_gltf:
            continue
        aspect_ratio = None
        if recorded.resolution is not None:
            aspect_ratio = float(recorded.resolution[0]) / float(recorded.resolution[1])

        camera_index = len(gltf.cameras)
        gltf.cameras.append(
            Camera(
                name=recorded.name,
                type="perspective",
                perspective=Perspective(
                    yfov=math.radians(recorded.fovy),
                    znear=0.03,
                    zfar=20.0,
                    aspectRatio=aspect_ratio,
                ),
                extras={
                    "source_simulator": "MuJoCo",
                    "source_parent_body": recorded.parent_body,
                    "fovy_degrees": recorded.fovy,
                    "resolution": list(recorded.resolution) if recorded.resolution else None,
                },
            )
        )

        node_index = len(gltf.nodes)
        gltf.nodes.append(
            Node(
                name=f"{recorded.name}_camera",
                camera=camera_index,
                translation=as_float_list(recorded.positions_gltf[0]),
                rotation=as_float_list(recorded.rotations_gltf_xyzw[0]),
                extras={
                    "animated_from_mujoco_camera": recorded.name,
                    "source_parent_body": recorded.parent_body,
                },
            )
        )
        gltf.scenes[0].nodes.append(node_index)

        translations = np.asarray(recorded.positions_gltf, dtype=np.float32)
        rotations = np.asarray(recorded.rotations_gltf_xyzw, dtype=np.float32)
        trans_acc = builder.add_array(translations, target=None, accessor_type="VEC3", component_type=FLOAT, name=f"{recorded.name}_camera_translation")
        rot_acc = builder.add_array(rotations, target=None, accessor_type="VEC4", component_type=FLOAT, name=f"{recorded.name}_camera_rotation")

        trans_sampler = len(animation.samplers)
        animation.samplers.append(AnimationSampler(input=time_accessor, output=trans_acc, interpolation="LINEAR"))
        animation.channels.append(AnimationChannel(sampler=trans_sampler, target=AnimationChannelTarget(node=node_index, path="translation")))
        rot_sampler = len(animation.samplers)
        animation.samplers.append(AnimationSampler(input=time_accessor, output=rot_acc, interpolation="LINEAR"))
        animation.channels.append(AnimationChannel(sampler=rot_sampler, target=AnimationChannelTarget(node=node_index, path="rotation")))


def export_glb(*, path: Path, model: mujoco.MjModel, recorder: TrajectoryRecorder, episode_index: int, sample, success: bool) -> None:
    if recorder.frame_count == 0:
        raise ValueError("Cannot export GLB without recorded frames.")

    gltf = GLTF2(asset=Asset(version="2.0", generator="franka_ball_catch_mujoco"))
    gltf.extras = {
        "source_simulator": "MuJoCo",
        "episode_index": episode_index,
        "fps": recorder.fps,
        "frame_count": recorder.frame_count,
        "success": bool(success),
        "scene_variant": sample.scene_variant,
        "seed": sample.seed,
        "robot_base_position": [float(v) for v in sample.robot_base_position],
        "robot_base_euler": [float(v) for v in sample.robot_base_euler],
        "tabletop_height": sample.tabletop_height,
        "axis_conversion": AXIS_CONVERSION_NOTE,
        "note": "Import GLB in Blender and play timeline.",
    }
    if recorder.cameras:
        gltf.extras["cameras"] = recorder.camera_gltf_extras()
    builder = BufferBuilder()
    gltf.scenes = [Scene(name="episode_scene", nodes=[])]
    gltf.scene = 0

    animation = Animation(name="episode_replay", samplers=[], channels=[])
    time_values = np.arange(recorder.frame_count, dtype=np.float32) / float(recorder.fps)
    time_accessor = builder.add_array(time_values, target=None, accessor_type="SCALAR", component_type=FLOAT, name="episode_time")

    for gid in recorder.geom_ids:
        recorded = recorder.geoms[gid]
        try:
            mesh = _mesh_from_mujoco_geom(model, gid)
        except Exception:
            mesh = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
            mesh.vertices = mujoco_vertices_to_gltf(mesh.vertices)
        material = _add_material(gltf, recorded.rgba)
        mesh_index = _add_mesh(gltf, builder, mesh, material, recorded.name)

        node_index = len(gltf.nodes)
        first_pos = as_float_list(recorded.positions[0])
        first_rot = as_float_list(recorded.rotations_xyzw[0])
        gltf.nodes.append(Node(name=recorded.name, mesh=mesh_index, translation=first_pos, rotation=first_rot))
        gltf.scenes[0].nodes.append(node_index)

        translations = np.asarray(recorded.positions, dtype=np.float32)
        rotations = np.asarray(recorded.rotations_xyzw, dtype=np.float32)
        trans_acc = builder.add_array(translations, target=None, accessor_type="VEC3", component_type=FLOAT, name=f"{recorded.name}_translation")
        rot_acc = builder.add_array(rotations, target=None, accessor_type="VEC4", component_type=FLOAT, name=f"{recorded.name}_rotation")

        trans_sampler = len(animation.samplers)
        animation.samplers.append(AnimationSampler(input=time_accessor, output=trans_acc, interpolation="LINEAR"))
        animation.channels.append(AnimationChannel(sampler=trans_sampler, target=AnimationChannelTarget(node=node_index, path="translation")))
        rot_sampler = len(animation.samplers)
        animation.samplers.append(AnimationSampler(input=time_accessor, output=rot_acc, interpolation="LINEAR"))
        animation.channels.append(AnimationChannel(sampler=rot_sampler, target=AnimationChannelTarget(node=node_index, path="rotation")))

    _add_recorded_cameras(gltf, builder, animation, time_accessor, recorder)

    gltf.animations = [animation]
    builder._align()
    gltf.bufferViews = builder.views
    gltf.accessors = builder.accessors
    gltf.buffers = [Buffer(byteLength=len(builder.blob))]
    gltf.set_binary_blob(bytes(builder.blob))
    path.parent.mkdir(parents=True, exist_ok=True)
    gltf.save_binary(str(path))
