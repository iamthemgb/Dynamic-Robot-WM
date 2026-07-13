"""Generate MP4 previews + metadata for the Franka soft-body packing family.

Usage (from the directory containing the package):

  MUJOCO_GL=egl python -m mujoco_franka_softbody_previews.generate_previews \\
      --out mujoco_franka_softbody_previews/outputs/preview_001 \\
      --num-seeds 2 --physics-variants 3 \\
      --width 640 --height 480 --fps 24 --duration 5.0
"""

import argparse
import json
import os
import sys
import time as _time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .control import LidController, PandaArm, interp_waypoints
from .materials import PARAMETER_IMPLEMENTATION_NOTES, PRESETS
from .scene_builder import (ARM_PREFIX, GRIPPER_MAX_WIDTH, KEYPOINT_NAMES,
                            compile_episode_model, episode_dt, find_panda_xml,
                            flex_elements, flex_vertices, keypoint_indices,
                            tet_volume)
from .tasks import TASK_IDS, build_task_spec

DT_CTRL = 1.0e-2
PREROLL = 0.8
MAX_CONTACTS_PER_FRAME = 32
EXCESSIVE_TAU = 60.0          # N*m commanded joint torque
EXCESSIVE_CONTACT_F = 100.0   # N single-contact force

SOFTBODY_MODEL_TYPE = "mujoco_flexcomp_grid"
EMBODIMENT = "single_franka"  # full Franka everywhere (no proxy)
OBJECT_FAMILY = "plush_sponge_softbody"
PLUSH_PROXY = True            # rectangular sponge proxy, not a plush mesh
SQUEEZE_PROXY = False         # real finger-pad contact
LID_PROXY = False             # real hinged dynamic lid


def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _classify_other(model, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if not name:
        bid = model.geom_bodyid[geom_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    low = name.lower()
    if "finger" in low or "hand" in low:
        return "gripper"
    if "lid" in low:
        return "lid"
    if "box_" in low:
        return "box"
    if "table" in low or "floor" in low:
        return "table"
    return name or f"geom_{geom_id}"


_CONTACT_PRIORITY = {"gripper": 0, "lid": 1, "box": 2, "table": 4}


def _scan_flex_contacts(model, data, t, frame_idx):
    """Contacts involving the flex object (internal/self contacts skipped).

    All flex-vs-geom contacts are collected, then the per-frame cap keeps
    gripper/lid/box contacts ahead of the (numerous) table contacts.
    """
    out = []
    force6 = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        fl = int(con.flex[0]), int(con.flex[1])
        if fl[0] < 0 and fl[1] < 0:
            continue
        if fl[0] >= 0 and fl[1] >= 0:
            continue                      # flex-internal
        other_geom = int(con.geom[1] if fl[0] >= 0 else con.geom[0])
        if other_geom < 0:
            continue
        other = _classify_other(model, other_geom)
        mujoco.mj_contactForce(model, data, i, force6)
        frame = np.array(con.frame).reshape(3, 3)
        normal = frame[0].copy()
        out.append((t, frame_idx, con.pos.copy(), normal,
                    frame.T @ force6[:3], other))
    out.sort(key=lambda c: _CONTACT_PRIORITY.get(c[5], 3))
    return out                            # caller caps for storage


def _label_tile(img, caption, tile_w=420):
    h = int(img.shape[0] * tile_w / img.shape[1])
    tile = Image.fromarray(img).resize((tile_w, h))
    canvas = Image.new("RGB", (tile_w, h + 18), (18, 18, 22))
    canvas.paste(tile, (0, 0))
    ImageDraw.Draw(canvas).text((4, h + 3), caption, fill=(230, 230, 230),
                                font=ImageFont.load_default())
    return canvas


def _procrustes_yaw(xy0: np.ndarray, xy1: np.ndarray) -> float:
    """Best-fit planar rotation angle (rad) mapping centered xy0 -> xy1."""
    a = xy0 - xy0.mean(axis=0)
    b = xy1 - xy1.mean(axis=0)
    H = a.T @ b
    return float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))


# ------------------------------------------------------------------ metrics

def compute_metrics(task, rec, kp, com, bbox, comp_ratio, lid_angle,
                    tau_max, contact_f_max, track_err_max):
    h0 = kp[0, 0, 2]                       # top_center keypoint height
    hf = kp[-1, 0, 2]
    max_compression = float(np.max(comp_ratio))
    recovery = float(hf / max(h0, 1e-9))
    excessive = bool(tau_max > EXCESSIVE_TAU or
                     contact_f_max > EXCESSIVE_CONTACT_F)

    m = {
        "max_compression": max_compression,
        "final_recovery_ratio": recovery,
        "recovery_ratio_after_release": recovery,
        "max_deformation": None,           # filled by caller
        "squeeze_success": None,
        "insertion_depth": None,
        "box_entry_success": None,
        "lid_closure_success": None,
        "object_escape_or_popout": None,
        "jam_flag": None,
        "slip_flag": None,
        "rebound_flag": None,
        "object_rotation_yaw_final": None,
        "excessive_force_flag": excessive,
        "max_commanded_joint_torque": float(tau_max),
        "max_flex_contact_force": float(contact_f_max),
        "max_tcp_tracking_error": float(track_err_max),
    }

    # Success = the scripted interaction visibly happened and the object
    # behaved like an elastic solid. Absolute compression naturally varies
    # with material stiffness inside a bundle - that is the counterfactual
    # signal, not a failure - so thresholds are material-independent floors.
    tid = task.task_id
    if tid == "press_release":
        strain_target = task.press_depth / max(h0, 1e-9)
        ok_press = max_compression >= 0.04
        ok_rec = recovery >= 0.70
        ok = bool(ok_press and ok_rec)
        fail = "none" if ok else (
            "insufficient_compression" if not ok_press else "poor_recovery")
        m["commanded_strain"] = float(strain_target)

    elif tid == "squeeze_with_gripper":
        widths = np.asarray(rec["gripper_width"])
        width_min = float(widths.min())
        ext_x = bbox[:, 1, 0] - bbox[:, 0, 0]
        squeeze_ratio = float(1.0 - ext_x.min() / max(ext_x[0], 1e-9))
        band_ext = rec.get("band_ext_x")
        band_ratio = (float(1.0 - band_ext.min() / max(band_ext[0], 1e-9))
                      if band_ext is not None else None)
        ejected = bool(np.linalg.norm(com[-1, :2] - com[0, :2]) > 0.06)
        touched = bool(rec["gripper_contact_any"])
        squeezed = (band_ratio if band_ratio is not None
                    else squeeze_ratio) >= 0.08
        ok = bool(touched and squeezed and not ejected)
        m["squeeze_success"] = ok
        m["min_gripper_width"] = width_min
        m["squeeze_ratio_along_finger_axis"] = squeeze_ratio
        m["squeeze_ratio_pad_band"] = band_ratio
        m["object_ejected"] = ejected
        m["gripper_contact_made"] = touched
        fail = "none" if ok else (
            "object_ejected" if ejected else
            "no_contact" if not touched else "no_squeeze")

    elif tid == "push_toward_box_opening":
        opening_x = task.box_interior["x"][0]
        cx = com[:, 0]
        insertion = float(cx[-1] - opening_x)
        entered = bool(cx[-1] > opening_x + 0.01)
        rebound = bool(cx.max() - cx[-1] > 0.03)
        slip = bool(abs(com[-1, 1] - com[0, 1]) > 0.045)
        yaw = _procrustes_yaw(rec["verts"][0][:, :2], rec["verts"][-1][:, :2])
        rotated = bool(abs(yaw) > 0.5)
        jam = bool(not entered and rec["box_contact_any"]
                   and track_err_max > 0.05)
        ok = bool(entered and not rebound)
        m.update(insertion_depth=insertion, box_entry_success=entered,
                 jam_flag=jam, slip_flag=slip, rebound_flag=rebound,
                 object_rotation_yaw_final=float(yaw),
                 object_rotated_flag=rotated)
        fail = "none" if ok else (
            "jam" if jam else "rebound" if rebound else
            "slipped_aside" if slip else "insufficient_push")

    else:  # compress_and_close_lid
        angle_final = float(lid_angle[-1])
        closed = bool(abs(angle_final) < 0.15)
        inter = task.box_interior
        inside_xy = (inter["x"][0] < com[-1, 0] < inter["x"][1] and
                     inter["y"][0] < com[-1, 1] < inter["y"][1])
        rim_z = inter["z"][1]
        popout = bool((not inside_xy) or kp[-1, 0, 2] > rim_z + 0.035)
        ok = bool(closed and not popout)
        m["lid_closure_success"] = closed
        m["lid_angle_final"] = angle_final
        m["object_escape_or_popout"] = popout
        fail = "none" if ok else (
            "object_popped_out" if popout else "lid_blocked_open")

    m["success_label"] = "success" if ok else "failure"
    m["failure_mode_label"] = fail
    return m


# ------------------------------------------------------------------ episode

def run_episode(task_id, seed, variant, args, panda_xml, out_dirs):
    material = PRESETS[variant % len(PRESETS)]
    task = build_task_spec(task_id, seed, args.duration)
    dt_sim = episode_dt(task, material)
    episode_id = f"{task_id}__seed{seed:03d}__var{variant:02d}"
    bundle_id = f"{task_id}__seed{seed:03d}"
    scene_xml_path = out_dirs["scenes"] / f"{episode_id}.xml"

    model, wrist_cams = compile_episode_model(task, material, panda_xml,
                                              scene_xml_path)
    data = mujoco.MjData(model)

    arm = PandaArm(model, ARM_PREFIX, "arm0")
    arm.reset_home(data, grip=task.waypoints[0].grip)
    lid = LidController(model, task.lid) if task.lid else None
    if lid:
        lid.reset(model, data, task.lid["open_angle"])
    mujoco.mj_forward(model, data)
    arm.capture_down_orientation(data)

    elem = flex_elements(model)
    n_ctrl = int(round(DT_CTRL / dt_sim))
    prev_time = data.time

    def _step():
        nonlocal prev_time
        mujoco.mj_step(model, data)
        if data.time < prev_time:
            raise RuntimeError(
                f"flex instability: MuJoCo BADQACC auto-reset at "
                f"sim t~{prev_time:.4f}s (material={material.label})")
        prev_time = data.time

    for step in range(int(PREROLL / dt_sim)):
        if step % n_ctrl == 0:
            pos, grip = interp_waypoints(task.waypoints, 0.0)
            arm.set_targets(pos, grip)
            arm.update(data, DT_CTRL)
            if lid:
                lid.update(data, 0.0)
        _step()
    t0 = data.time

    verts0 = flex_vertices(data).copy()
    kp_idx = keypoint_indices(verts0)

    # 480p panels, 24 FPS, fps*duration+1 = 121 frames (dataset convention)
    F = int(round(args.duration * args.fps)) + 1
    frame_times = np.arange(F) / args.fps
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam_list = ["front", "top"] + wrist_cams

    video_path = out_dirs["videos"] / f"{episode_id}.mp4"
    writer = imageio.get_writer(str(video_path), fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1,
                                pixelformat="yuv420p")

    rec = {k: [] for k in ["q", "dq", "q_target", "tau_cmd", "ee_pos",
                           "ee_quat", "ee_vel", "ee_wrench_est",
                           "gripper_width", "gripper_cmd",
                           "gripper_force_cmd"]}
    rec["verts"] = []
    rec["box_contact_any"] = False
    rec["gripper_contact_any"] = False
    times, kps, coms, bboxes, vols = [], [], [], [], []
    lid_angles, lid_cmds, lid_poses = [], [], []
    grip_con, box_con, lid_con, table_con = [], [], [], []
    contacts = []
    tau_max = 0.0
    contact_f_max = 0.0
    track_err_max = 0.0
    snapshot = None
    snapshot_frame = int(0.55 * F)

    total_steps = int(round(args.duration / dt_sim)) + 1
    next_f = 0
    for step in range(total_steps):
        t = data.time - t0
        if step % n_ctrl == 0:
            pos, grip = interp_waypoints(task.waypoints, t)
            arm.set_targets(pos, grip)
            arm.update(data, DT_CTRL)
            if lid:
                lid.update(data, t)
            tcp, _ = arm.tcp_pose(data)
            track_err_max = max(track_err_max,
                                float(np.linalg.norm(pos - tcp)))
            tau_max = max(tau_max, float(np.abs(arm.tau_cmd(data)).max()))

        while next_f < F and t >= frame_times[next_f] - 1e-9:
            panels = []
            for cam in cam_list:
                renderer.update_scene(data, camera=cam)
                panels.append(renderer.render())
            composite = np.hstack(panels)
            if composite.shape[0] % 2:
                composite = composite[:-1]
            if composite.shape[1] % 2:
                composite = composite[:, :-1]
            writer.append_data(composite)
            if next_f == snapshot_frame:
                snapshot = panels[0].copy()

            times.append(t)
            rec["q"].append(data.qpos[arm.qpos_adr].copy())
            rec["dq"].append(data.qvel[arm.dof_adr].copy())
            rec["q_target"].append(arm.q_target.copy())
            rec["tau_cmd"].append(arm.tau_cmd(data))
            p, q = arm.tcp_pose(data)
            rec["ee_pos"].append(p.copy())
            rec["ee_quat"].append(q.copy())
            rec["ee_vel"].append(arm.tcp_velocity(data))
            rec["ee_wrench_est"].append(arm.estimated_ee_wrench(data))
            rec["gripper_width"].append(arm.gripper_width(data))
            rec["gripper_cmd"].append(arm.grip_cmd)
            rec["gripper_force_cmd"].append(arm.gripper_force_cmd(data))

            v = flex_vertices(data).copy().astype(np.float32)
            rec["verts"].append(v)
            kp = np.vstack([v[kp_idx], v.mean(axis=0, keepdims=True)])
            kps.append(kp.astype(np.float32))
            coms.append(v.mean(axis=0))
            bboxes.append(np.stack([v.min(axis=0), v.max(axis=0)]))
            vols.append(tet_volume(v.astype(np.float64), elem))

            frame_contacts = _scan_flex_contacts(model, data, t, next_f)
            contacts.extend(frame_contacts[:MAX_CONTACTS_PER_FRAME])
            others = {c[5] for c in frame_contacts}
            forces = [np.linalg.norm(c[4]) for c in frame_contacts]
            if forces:
                contact_f_max = max(contact_f_max, max(forces))
            grip_con.append("gripper" in others)
            box_con.append("box" in others)
            lid_con.append("lid" in others)
            table_con.append("table" in others)
            if "box" in others:
                rec["box_contact_any"] = True
            if "gripper" in others:
                rec["gripper_contact_any"] = True

            if lid:
                lid_angles.append(lid.angle(data))
                lid_cmds.append(lid.command(data))
                lp, lq = lid.pose_world(data)
                lid_poses.append(np.concatenate([lp, lq]))
            next_f += 1

        _step()
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f"simulation diverged at t={t:.3f}s")

    writer.close()
    renderer.close()
    if snapshot is None:
        snapshot = panels[0].copy()

    times = np.asarray(times)
    verts = np.asarray(rec["verts"])                     # (F, nvert, 3)
    kp = np.asarray(kps)                                 # (F, 7, 3)
    com = np.asarray(coms)                               # (F, 3)
    bbox = np.asarray(bboxes)                            # (F, 2, 3)
    vols = np.asarray(vols)
    lid_angles = np.asarray(lid_angles) if lid else None

    h0 = max(kp[0, 0, 2], 1e-9)
    comp_ratio = np.clip(1.0 - kp[:, 0, 2] / h0, -1.0, 1.0)
    # deformation excluding rigid translation (rotation NOT removed)
    rel = verts - com[:, None, :]
    deform = np.linalg.norm(rel - rel[0][None], axis=2).max(axis=1)
    inter = task.box_interior
    if inter:
        inside = ((verts[:, :, 0] > inter["x"][0]) &
                  (verts[:, :, 0] < inter["x"][1]) &
                  (verts[:, :, 1] > inter["y"][0]) &
                  (verts[:, :, 1] < inter["y"][1]) &
                  (verts[:, :, 2] < inter["z"][1] + 0.02))
        box_occupancy = inside.mean(axis=1)
    else:
        box_occupancy = None
    if lid:
        lid_len = 2 * abs(task.lid["hinge_pos_world"][0]
                          - task.box_interior["x"][0]) / 2
        lid_clearance = lid_len * np.sin(np.clip(lid_angles, 0.0, None))
    else:
        lid_clearance = None

    if task.squeeze_band:
        b = task.squeeze_band
        band_mask = ((np.abs(verts[0][:, 1] - task.obj.pos[1]) <= b["y_half"])
                     & (verts[0][:, 2] >= b["z_min"]))
        if band_mask.sum() >= 4:
            band = verts[:, band_mask, 0]
            rec["band_ext_x"] = band.max(axis=1) - band.min(axis=1)

    metrics = compute_metrics(task, rec, kp, com, bbox, comp_ratio,
                              lid_angles, tau_max, contact_f_max,
                              track_err_max)
    metrics["max_deformation"] = float(deform.max())

    # --- states npz ---------------------------------------------------------
    states_path = out_dirs["states"] / f"{episode_id}.npz"
    npz = {
        "time": times,
        "softbody_keypoints_world": kp,
        "softbody_keypoint_names": np.array(KEYPOINT_NAMES),
        "object_center_of_mass": com.astype(np.float32),
        "bounding_box_world": bbox.astype(np.float32),
        "volume_proxy_tet_mesh": vols.astype(np.float32),
        "compression_ratio": comp_ratio.astype(np.float32),
        "max_deformation": deform.astype(np.float32),
        "flex_vertex_positions_world": verts,
        "flex_element_indices": elem.astype(np.int32),
        "gripper_contact": np.asarray(grip_con),
        "box_contact": np.asarray(box_con),
        "lid_contact": np.asarray(lid_con),
        "table_contact": np.asarray(table_con),
    }
    if box_occupancy is not None:
        npz["box_occupancy"] = box_occupancy.astype(np.float32)
    if lid:
        npz["lid_angle"] = lid_angles.astype(np.float32)
        npz["lid_command"] = np.asarray(lid_cmds, dtype=np.float32)
        npz["lid_pose_world"] = np.asarray(lid_poses, dtype=np.float32)
        npz["lid_clearance"] = lid_clearance.astype(np.float32)
    for k in ["q", "dq", "q_target", "tau_cmd", "ee_pos", "ee_quat", "ee_vel",
              "ee_wrench_est", "gripper_width", "gripper_cmd",
              "gripper_force_cmd"]:
        npz[f"arm0_{k}"] = np.asarray(rec[k], dtype=np.float32)
    if contacts:
        npz["contact_time"] = np.array([c[0] for c in contacts], dtype=np.float32)
        npz["contact_frame"] = np.array([c[1] for c in contacts], dtype=np.int32)
        npz["contact_pos"] = np.array([c[2] for c in contacts], dtype=np.float32)
        npz["contact_normal"] = np.array([c[3] for c in contacts], dtype=np.float32)
        npz["contact_force"] = np.array([c[4] for c in contacts], dtype=np.float32)
        npz["contact_other"] = np.array([c[5] for c in contacts])
    np.savez_compressed(states_path, **npz)

    first_contact = float(contacts[0][0]) if contacts else None
    obj = task.obj

    meta = {
        "episode_id": episode_id,
        "counterfactual_bundle_id": bundle_id,
        "task_id": task_id,
        "subfamily": task.subfamily,
        "simulator_name": "mujoco",
        "simulator_version": mujoco.__version__,
        "random_seed": seed,
        "physics_variant": variant,
        "dt_sim": dt_sim,
        "dt_control": DT_CTRL,
        "dt_video": 1.0 / args.fps,
        "duration": args.duration,
        "fps": args.fps,
        "n_frames": F,
        "frame_size": [args.width, args.height],
        "camera_names": cam_list,
        "wrist_camera": bool(wrist_cams),
        "video_paths": [str(video_path.relative_to(out_dirs["root"]))],
        "video_layout": "horizontal_composite " + "|".join(cam_list),
        "states_path": str(states_path.relative_to(out_dirs["root"])),
        "scene_xml_path": os.path.relpath(scene_xml_path, out_dirs["root"]),
        "softbody": {
            "object_id": obj.object_id,
            "object_family": OBJECT_FAMILY,
            "material_label": material.label,
            "young_modulus": material.young,
            "poisson_ratio": material.poisson,
            "density": material.density,
            "damping": material.edge_damping,
            "damping_label": material.damping_label,
            "friction": material.friction,
            "restitution": None,           # not implemented; see notes
            "plasticity": False,           # not implemented; see notes
            "mesh_resolution": list(obj.count),
            "grid_resolution": list(obj.count),
            "grid_spacing": list(obj.spacing),
            "num_vertices_or_nodes": obj.num_vertices,
            "num_elements": int(elem.shape[0]),
            "softbody_model_type": SOFTBODY_MODEL_TYPE,
            "plush_proxy": PLUSH_PROXY,
            "contact_solref": [2 * dt_sim, 1.0],
            "object_size_outer": list(obj.outer_size),
            "object_initial_pos": obj.pos.tolist(),
            "total_mass": material.mass_for_volume(obj.volume),
        },
        "parameter_implementation_notes": PARAMETER_IMPLEMENTATION_NOTES,
        "robot": {
            "embodiment": EMBODIMENT,
            "robot_model": "franka_emika_panda (MuJoCo Menagerie)",
            "robot_asset_path": panda_xml,
            "control_mode": "joint_position_servo",
            "action_mode": "cartesian_ee_waypoints_diff_ik",
            "gripper_servo_modification": (
                "actuator8 position-servo gains scaled 12x via MjSpec "
                "(stock kp=100 N/m tendon servo saturates at ~2.4 N pinch, "
                "too weak to visibly squeeze foam; real Panda sustains "
                "~70 N); ctrl->width mapping and forcerange (+-100 N) "
                "unchanged"),
            "arm_id": "arm0",
            "full_robot_state": True,
            "state_arrays_in_npz": [f"arm0_{k}" for k in
                                    ["q", "dq", "q_target", "tau_cmd",
                                     "ee_pos", "ee_quat", "ee_vel",
                                     "ee_wrench_est", "gripper_width",
                                     "gripper_cmd", "gripper_force_cmd"]],
            "ee_wrench_note": ("estimated_ee_wrench is a least-squares "
                               "J^T w = qfrc_constraint solve; approximate"),
            "q_final": rec["q"][-1],
            "ee_pose_world_final": np.concatenate([rec["ee_pos"][-1],
                                                   rec["ee_quat"][-1]]),
        },
        "action": {
            "skill_label": task.skill_label,
            "planned_ee_trajectory": [[w.t, *w.pos, w.grip]
                                      for w in task.waypoints],
            "planned_ee_trajectory_format":
                "[t, x, y, z, gripper(0=closed,1=open,frac=width/0.08)]",
            "executed_ee_trajectory": [[t, *p] for t, p in
                                       zip(times, rec["ee_pos"])],
            "gripper_command_trajectory": [[t, g] for t, g in
                                           zip(times, rec["gripper_cmd"])],
            "endpoint_or_contact_attachment_method": "none_contact_only",
            "press_depth": task.press_depth,
            "squeeze_width_target": task.squeeze_width_target,
            "squeeze_proxy": SQUEEZE_PROXY,
            "push_distance": task.push_distance,
            "push_direction_world": task.push_direction_world,
            "box_pose_world": task.box_pose_world,
            "box_opening_pose_world": task.box_opening_pose_world,
            "box_interior_aabb": task.box_interior,
            "lid_proxy": LID_PROXY if task.lid else None,
            "lid_pose_world_initial": (lid_poses[0].tolist()
                                       if task.lid else None),
            "lid_trajectory_planned": (task.lid["angle_trajectory"]
                                       if task.lid else None),
            "lid_actuation": (dict(kp=task.lid["kp"],
                                   forcerange=task.lid["forcerange"])
                              if task.lid else None),
        },
        "metrics": metrics,
        "contacts_summary": {
            "first_contact_time": first_contact,
            "n_contact_records": len(contacts),
            "sampling": f"per video frame, cap {MAX_CONTACTS_PER_FRAME}/frame,"
                        " flex-internal contacts excluded",
            "arrays_in_npz": (["contact_time", "contact_frame", "contact_pos",
                               "contact_normal", "contact_force",
                               "contact_other"] if contacts else []),
        },
        "notes": task.notes,
    }
    meta = _to_jsonable(meta)
    (out_dirs["metadata"] / f"{episode_id}.json").write_text(
        json.dumps(meta, indent=2))
    return meta, snapshot


# --------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-seeds", type=int, default=2)
    ap.add_argument("--physics-variants", type=int, default=3)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--tasks", nargs="*", default=TASK_IDS, choices=TASK_IDS)
    args = ap.parse_args(argv)

    out_root = Path(args.out).resolve()
    out_dirs = {"root": out_root,
                "videos": out_root / "videos",
                "metadata": out_root / "metadata",
                "states": out_root / "states",
                "contact_sheets": out_root / "contact_sheets",
                "scenes": Path(__file__).parent / "scenes"}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    panda_xml = find_panda_xml()
    print(f"[info] MuJoCo {mujoco.__version__}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"[info] Panda model: {panda_xml}")
    print(f"[info] softbody_model_type: {SOFTBODY_MODEL_TYPE}, "
          f"embodiment: {EMBODIMENT} (full Franka, no proxy arm)")

    episodes, snapshots, failures = [], [], []
    for task_id in args.tasks:
        for seed in range(args.num_seeds):
            for variant in range(args.physics_variants):
                eid = f"{task_id}__seed{seed:03d}__var{variant:02d}"
                tic = _time.time()
                try:
                    meta, snap = run_episode(task_id, seed, variant, args,
                                             panda_xml, out_dirs)
                    episodes.append(meta)
                    snapshots.append(
                        (eid, meta["softbody"]["material_label"],
                         meta["metrics"]["success_label"], snap))
                    print(f"[ok]   {eid}  ({_time.time()-tic:.1f}s)  "
                          f"{meta['metrics']['success_label']}  "
                          f"-> {meta['video_paths'][0]}")
                except Exception as e:  # noqa: BLE001
                    failures.append((eid, f"{type(e).__name__}: {e}"))
                    print(f"[FAIL] {eid}: {type(e).__name__}: {e}")
                    traceback.print_exc()

    jsonl_path = out_root / "episodes.jsonl"
    with open(jsonl_path, "w") as f:
        for m in episodes:
            f.write(json.dumps(m) + "\n")

    parquet_note = ""
    try:
        import pandas as pd  # noqa: PLC0415
        flat = []
        for m in episodes:
            row = {k: v for k, v in m.items() if not isinstance(v, (dict, list))}
            row.update({f"softbody_{k}": v for k, v in m["softbody"].items()
                        if not isinstance(v, (dict, list))})
            row.update({f"metric_{k}": v for k, v in m["metrics"].items()
                        if not isinstance(v, (dict, list))})
            row["embodiment"] = m["robot"]["embodiment"]
            row["skill_label"] = m["action"]["skill_label"]
            flat.append(row)
        pd.DataFrame(flat).to_parquet(out_root / "episodes.parquet")
        parquet_note = str(out_root / "episodes.parquet")
    except Exception as e:  # noqa: BLE001
        parquet_note = f"skipped ({type(e).__name__}: pandas/pyarrow unavailable)"

    sheet_path = out_dirs["contact_sheets"] / "contact_sheet.png"
    if snapshots:
        tiles = [_label_tile(s, f"{eid} [{mat}] {succ}")
                 for eid, mat, succ, s in snapshots]
        cols = min(max(args.physics_variants, 1), 4)
        rows = (len(tiles) + cols - 1) // cols
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (cols * tw, rows * th), (10, 10, 12))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
        sheet.save(sheet_path)

    print("\n" + "=" * 72)
    print(f"Output directory   : {out_root}")
    print(f"Videos generated   : {len(episodes)} ({len(failures)} failures)")
    print(f"Softbody model type: {SOFTBODY_MODEL_TYPE} "
          f"(volumetric tet FEM, plush_proxy={PLUSH_PROXY})")
    print(f"Contact sheet      : {sheet_path}")
    print(f"episodes.jsonl     : {jsonl_path}")
    print(f"parquet            : {parquet_note}")
    print("\nEmbodiment per task (full Franka vs proxy):")
    done_tasks = {m["task_id"] for m in episodes}
    for tid in args.tasks:
        status = (f"{EMBODIMENT} (full Franka arm + real gripper, no proxy)"
                  if tid in done_tasks else "FAILED - see errors above")
        print(f"  {tid:26s} {status}")
    print("\nContact realism:")
    print(f"  squeeze_with_gripper  : real finger-pad contact "
          f"(squeeze_proxy={SQUEEZE_PROXY})")
    print(f"  compress_and_close_lid: real hinged dynamic lid, force-bounded "
          f"position actuator (lid_proxy={LID_PROXY})")
    print("\nPhysically implemented parameters: young_modulus, poisson_ratio "
          "(<=0.40), density (via total mass), friction, damping (as flex "
          "edge damping)")
    print("Approximated / not implemented:")
    for k in ("damping", "restitution", "plasticity", "dt_sim"):
        print(f"  - {k}: {PARAMETER_IMPLEMENTATION_NOTES[k]}")
    if failures:
        print("\nFailures:")
        for eid, err in failures:
            print(f"  {eid}: {err}")
    return 0 if episodes and not failures else (0 if episodes else 1)


if __name__ == "__main__":
    sys.exit(main())
