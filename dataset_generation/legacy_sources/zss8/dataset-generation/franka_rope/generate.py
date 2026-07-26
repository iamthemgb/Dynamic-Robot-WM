"""Scaled dataset generator for the Franka-rope deformable family.

Extends the original preview generator with:
  * per-episode visual **background** domain randomization (see backgrounds.py)
    — plain studio, kitchen, warehouse, workshop, lab, office, patio, garage …
  * five extra dynamics-rich tasks (shake_wave, lift_and_drape, coil_on_table,
    fold_in_half, sweep_aside) on top of the four spec tasks
  * shard/seed/variant CLI so the grid can be fanned out across a SLURM array

Physics are identical across backgrounds — only pixels change — so a model
learns rope dynamics, not a fixed backdrop.

Example (single-process smoke):

  MUJOCO_GL=egl python -m franka_rope.generate \\
      --out /gpfs/radev/scratch/sous/zss8/franka_rope/smoke \\
      --seeds-per-task 1 --variants 2 --width 640 --height 480 \\
      --fps 24 --duration 5.0

Sharded (one array task):

  ... --seeds-per-task 20 --variants 3 --num-shards 8 --shard-index $SLURM_ARRAY_TASK_ID
"""

import argparse
import json
import os
import sys
import time as _time
import traceback
import zlib
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .backgrounds import THEME_NAMES, build_background, choose_theme
from .control import PandaArm, RopeGrasp, interp_waypoints
from .materials import PARAMETER_IMPLEMENTATION_NOTES, PRESETS
from .scene_builder import (ARM_PREFIX, ROPE_PREFIX, compile_episode_model,
                            find_panda_xml, rope_body_ids, rope_geom_ids)
from .tasks import BAR_HEIGHT, TASK_IDS, build_task_spec

DT_SIM = 5.0e-4
DT_CTRL = 1.0e-2
PREROLL = 0.8
MAX_CONTACTS_PER_FRAME = 32
SLIP_THRESHOLD = 0.03
SNAG_THRESHOLD = 0.10

ROPE_MODEL_TYPE = "mujoco_composite_cable"
EMBODIMENT = "single_franka"


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


def _rope_tip(model, data, body_ids):
    last, prev = body_ids[-1], body_ids[-2]
    adr, num = model.body_geomadr[last], model.body_geomnum[last]
    if num < 1:
        return data.xpos[last].copy()
    g = adr
    half = model.geom_size[g, 1]
    z = data.geom_xmat[g].reshape(3, 3)[:, 2]
    c = data.geom_xpos[g]
    t1, t2 = c + half * z, c - half * z
    ref = data.xpos[prev]
    return t1 if np.linalg.norm(t1 - ref) > np.linalg.norm(t2 - ref) else t2


def _scan_rope_contacts(model, data, rope_geoms, t, frame_idx):
    out = []
    force6 = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom[0]), int(con.geom[1])
        in1, in2 = g1 in rope_geoms, g2 in rope_geoms
        if not (in1 or in2):
            continue
        other = g2 if in1 else g1
        if other in rope_geoms:
            other_name = "rope_self"
        else:
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            if not other_name:
                ob = model.geom_bodyid[other]
                other_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, ob)
                              or f"geom_{other}")
        mujoco.mj_contactForce(model, data, i, force6)
        frame = np.array(con.frame).reshape(3, 3)
        out.append((t, frame_idx, con.pos.copy(), frame.T @ force6[:3], other_name))
        if len(out) >= MAX_CONTACTS_PER_FRAME:
            break
    return out


def _label_tile(img, caption, tile_w=420):
    h = int(img.shape[0] * tile_w / img.shape[1])
    tile = Image.fromarray(img).resize((tile_w, h))
    canvas = Image.new("RGB", (tile_w, h + 18), (18, 18, 22))
    canvas.paste(tile, (0, 0))
    ImageDraw.Draw(canvas).text((4, h + 3), caption, fill=(230, 230, 230),
                                font=ImageFont.load_default())
    return canvas


# ------------------------------------------------------------------ metrics

def compute_metrics(task, material, times, centerline, slip_max, track_err_max):
    """Task metrics from recorded trajectories.

    centerline: (F, n_rope, 3) world positions of every rope segment body.
    endpoint0 = grasped/pushed end (segment 0); tip = far end (last segment).
    """
    endpoint0 = centerline[:, 0, :].astype(np.float64)
    tip = centerline[:, -1, :].astype(np.float64)
    mid = centerline[:, centerline.shape[1] // 2, :].astype(np.float64)
    e0, e_final = endpoint0[0], endpoint0[-1]
    disp = float(np.linalg.norm(e_final - e0))
    max_h = float(endpoint0[:, 2].max())
    horiz = float(np.linalg.norm((e_final - e0)[:2]))
    ep_path_len = float(np.linalg.norm(np.diff(endpoint0, axis=0), axis=1).sum())
    slip = bool(slip_max > SLIP_THRESHOLD)
    snag = bool(track_err_max > SNAG_THRESHOLD)

    winding = passage = None
    wave_amp = drape_ok = coil_turns = twirl_radius = sweep_disp = None

    if task.task_id == "wrap_around_post":
        pc = np.asarray(task.post_pose_world["pos"])[:2]
        rel = endpoint0[:, :2] - pc
        ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        winding = float((ang[-1] - ang[0]) / (2 * np.pi))
    if task.task_id == "thread_through_ring":
        rc = np.asarray(task.ring_pose_world["pos"])
        z = endpoint0[:, 2] - rc[2]
        passage = False
        for k in range(1, len(z)):
            if z[k - 1] > 0.0 >= z[k]:
                radial = np.linalg.norm(endpoint0[k, :2] - rc[:2])
                if radial < task.ring_radius:
                    passage = True
        passage = bool(passage and z[-1] < 0.0)
    if task.task_id == "shake_wave":
        wave_amp = float(max(np.ptp(tip[:, 1]), np.ptp(tip[:, 2])))
    if task.task_id == "lift_and_drape":
        # rope must have gone above the bar and the held end ended on the far
        # (+y) side and low -> it now hangs over the bar.
        drape_ok = bool(max_h > BAR_HEIGHT and (e_final[1] - e0[1]) > 0.16
                        and e_final[2] < BAR_HEIGHT)
    if task.task_id == "coil_on_table":
        c = np.asarray(task.coil_center_world)[:2]
        rel = endpoint0[:, :2] - c
        ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        coil_turns = float(abs(ang[-1] - ang[0]) / (2 * np.pi))
    if task.task_id == "twirl_overhead":
        c = np.asarray(task.twirl_center_world)[:2]
        twirl_radius = float(np.linalg.norm(tip[:, :2] - c, axis=1).max())
    if task.task_id == "sweep_aside":
        sweep_disp = float(np.linalg.norm((mid[-1] - mid[0])[:2]))

    if task.task_id == "tug_endpoint":
        ok, fail = max_h > 0.15, "endpoint_not_lifted"
    elif task.task_id == "drag_endpoint":
        ok, fail = horiz > 0.15, "insufficient_drag"
    elif task.task_id == "wrap_around_post":
        ok, fail = winding is not None and abs(winding) >= 0.45, "insufficient_winding"
    elif task.task_id == "thread_through_ring":
        ok, fail = bool(passage), "no_ring_passage"
    elif task.task_id == "shake_wave":
        ok, fail = wave_amp is not None and wave_amp > 0.06, "no_wave_propagated"
    elif task.task_id == "lift_and_drape":
        ok, fail = bool(drape_ok), "rope_not_draped"
    elif task.task_id == "coil_on_table":
        ok, fail = (coil_turns is not None and coil_turns >= 1.0
                    and ep_path_len > 0.5), "insufficient_coiling"
    elif task.task_id == "twirl_overhead":
        ok, fail = (twirl_radius is not None
                    and twirl_radius > 0.12), "rope_did_not_trail"
    elif task.task_id == "sweep_aside":
        ok, fail = sweep_disp is not None and sweep_disp > 0.06, "rope_not_swept"
    else:
        ok, fail = True, "none"
    # A snag (rope caught / arm stuck) fails simple pick-move-place tasks, but
    # several tasks intentionally load the rope against an obstacle or move it
    # fast, so a large TCP tracking error is expected physics, not a failure:
    #   shake_wave      fast servo lag while oscillating
    #   lift_and_drape  rope goes taut over the bar
    #   coil_on_table   tight inward spiral drags the coiling rope
    #   wrap_around_post rope tensions against the post during the orbit
    # Each of these is gated by its own outcome metric (wave amplitude,
    # drape_success, coil_turns, winding), so a genuine stuck-arm still fails
    # there (low winding/turns). The snag_flag stays recorded in metrics.
    SNAG_EXEMPT = {"shake_wave", "lift_and_drape", "coil_on_table",
                   "wrap_around_post", "twirl_overhead"}
    if ok and snag and task.task_id not in SNAG_EXEMPT:
        ok, fail = False, "snag"

    return {
        "endpoint_displacement": disp,
        "endpoint_max_height": max_h,
        "endpoint_horizontal_displacement": horiz,
        "endpoint_path_length": ep_path_len,
        "centerline_chamfer_to_target": None,
        "centerline_chamfer_note": "no target centerline defined for previews",
        "winding_number_around_post": winding,
        "ring_passage_success": passage,
        "wave_tip_amplitude": wave_amp,
        "drape_success": drape_ok,
        "coil_turns": coil_turns,
        "twirl_tip_radius": twirl_radius,
        "sweep_midpoint_displacement": sweep_disp,
        "rope_slip_flag": slip,
        "rope_slip_max_constraint_stretch": float(slip_max),
        "snag_flag": snag,
        "max_tracking_error_while_grasped": float(track_err_max),
        "success_label": "success" if ok else "failure",
        "failure_mode_label": "none" if ok else fail,
    }


# ------------------------------------------------------------------ episode

def _bundle_theme(task_id, seed, forced_theme):
    """Theme is chosen per (task, seed) bundle so material variants of one
    counterfactual bundle share a backdrop; jitter still differs per variant."""
    if forced_theme and forced_theme != "random":
        return forced_theme
    rng = np.random.default_rng([0xB6, zlib.crc32(task_id.encode()), seed])
    return choose_theme(rng)


def run_episode(task_id, seed, variant, args, panda_xml, out_dirs):
    material = PRESETS[variant % len(PRESETS)]
    task = build_task_spec(task_id, seed, args.duration)
    episode_id = f"{task_id}__seed{seed:03d}__var{variant:02d}"
    bundle_id = f"{task_id}__seed{seed:03d}"
    scene_xml_path = out_dirs["scenes"] / f"{episode_id}.xml"

    theme = _bundle_theme(task_id, seed, args.theme)
    bg_rng = np.random.default_rng(
        [0xBACC, zlib.crc32(task_id.encode()), seed, variant])
    background = build_background(theme, bg_rng)

    model, wrist_cams = compile_episode_model(task, material, DT_SIM, panda_xml,
                                              background, scene_xml_path)
    data = mujoco.MjData(model)

    arm = PandaArm(model, ARM_PREFIX, "arm0")
    arm.reset_home(data)
    mujoco.mj_forward(model, data)
    arm.capture_down_orientation(data)

    grasp_log = []
    grasp = RopeGrasp(model, arm, task.grasp)
    body_ids = rope_body_ids(model)
    rope_geoms = rope_geom_ids(model)
    n_rope = len(body_ids)
    kp_idx = {"endpoint_0": 0, "quarter_1": n_rope // 4, "midpoint": n_rope // 2,
              "quarter_3": (3 * n_rope) // 4, "endpoint_1": n_rope - 1}

    n_ctrl = int(round(DT_CTRL / DT_SIM))
    for step in range(int(PREROLL / DT_SIM)):
        if step % n_ctrl == 0:
            pos, grip = interp_waypoints(task.waypoints, 0.0)
            arm.set_targets(pos, grip)
            arm.update(data, DT_CTRL)
        mujoco.mj_step(model, data)
    t0 = data.time

    F = int(round(args.duration * args.fps)) + 1
    frame_times = np.arange(F) / args.fps
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam_list = ["front", "top"] + wrist_cams

    video_mode = getattr(args, "video_mode", "split")

    def _open_writer(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return imageio.get_writer(str(path), fps=args.fps, codec="libx264",
                                  quality=8, macro_block_size=1,
                                  pixelformat="yuv420p")

    if video_mode == "composite":
        # Legacy: one mp4 per episode, all views hstacked side by side.
        video_path = out_dirs["videos"] / f"{episode_id}.mp4"
        writer = _open_writer(video_path)
        view_paths = {}
    else:
        # Split: one mp4 per view, in a per-camera subdirectory so each view is
        # its own standalone dataset of videos (videos/<cam>/<episode_id>.mp4).
        writer = None
        view_paths = {cam: out_dirs["videos"] / cam / f"{episode_id}.mp4"
                      for cam in cam_list}
        view_writers = {cam: _open_writer(p) for cam, p in view_paths.items()}

    rec = {k: [] for k in ["q", "dq", "q_target", "tau_cmd", "ee_pos", "ee_quat",
                           "ee_vel", "gripper_width", "gripper_cmd",
                           "gripper_force_cmd"]}
    times, centerline, tips, contacts = [], [], [], []
    slip_max = 0.0
    track_err_max = 0.0
    snapshot = None
    snapshot_frame = int(0.55 * F)

    total_steps = int(round(args.duration / DT_SIM)) + 1
    next_f = 0
    panels = None
    for step in range(total_steps):
        t = data.time - t0
        if step % n_ctrl == 0:
            pos, grip = interp_waypoints(task.waypoints, t)
            arm.set_targets(pos, grip)
            arm.update(data, DT_CTRL)
            grasp.update(data, t, grasp_log)
            if grasp.active():
                slip_max = max(slip_max, grasp.anchor_error(data))
                tcp, _ = arm.tcp_pose(data)
                track_err_max = max(track_err_max,
                                    float(np.linalg.norm(pos - tcp)))

        while next_f < F and t >= frame_times[next_f] - 1e-9:
            panels = []
            for cam in cam_list:
                renderer.update_scene(data, camera=cam)
                panels.append(renderer.render())
            if video_mode == "composite":
                composite = np.hstack(panels)
                if composite.shape[0] % 2:
                    composite = composite[:-1]
                if composite.shape[1] % 2:
                    composite = composite[:, :-1]
                writer.append_data(composite)
            else:
                for cam, panel in zip(cam_list, panels):
                    if panel.shape[0] % 2:
                        panel = panel[:-1]
                    if panel.shape[1] % 2:
                        panel = panel[:, :-1]
                    view_writers[cam].append_data(panel)
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
            rec["gripper_width"].append(arm.gripper_width(data))
            rec["gripper_cmd"].append(arm.grip_cmd)
            rec["gripper_force_cmd"].append(arm.gripper_force_cmd(data))
            centerline.append(data.xpos[body_ids].copy().astype(np.float32))
            tips.append(_rope_tip(model, data, body_ids))
            contacts.extend(_scan_rope_contacts(model, data, rope_geoms, t, next_f))
            next_f += 1

        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f"simulation diverged at t={t:.3f}s")

    if video_mode == "composite":
        writer.close()
    else:
        for w in view_writers.values():
            w.close()
    renderer.close()
    if snapshot is None:
        snapshot = panels[0].copy()

    times = np.asarray(times)
    centerline = np.asarray(centerline)                    # (F, n_rope, 3)
    tips = np.asarray(tips, dtype=np.float32)
    keypoints = np.stack([centerline[:, kp_idx["endpoint_0"]],
                          centerline[:, kp_idx["quarter_1"]],
                          centerline[:, kp_idx["midpoint"]],
                          centerline[:, kp_idx["quarter_3"]],
                          tips], axis=1)                   # (F, 5, 3)

    metrics = compute_metrics(task, material, times, centerline,
                              slip_max, track_err_max)

    states_path = out_dirs["states"] / f"{episode_id}.npz"
    npz = {
        "time": times,
        "rope_centerline_points_world": centerline,
        "rope_keypoints": keypoints.astype(np.float32),
        "rope_keypoint_names": np.array(["endpoint_0", "quarter_1", "midpoint",
                                         "quarter_3", "endpoint_1"]),
        "endpoint_positions_world": np.stack(
            [centerline[:, 0], tips], axis=1).astype(np.float32),
    }
    for k, v in rec.items():
        npz[f"arm0_{k}"] = np.asarray(v, dtype=np.float32)
    if contacts:
        npz["contact_time"] = np.array([c[0] for c in contacts], dtype=np.float32)
        npz["contact_frame"] = np.array([c[1] for c in contacts], dtype=np.int32)
        npz["contact_pos"] = np.array([c[2] for c in contacts], dtype=np.float32)
        npz["contact_force"] = np.array([c[3] for c in contacts], dtype=np.float32)
        npz["contact_other"] = np.array([c[4] for c in contacts])
    np.savez_compressed(states_path, **npz)

    first_contact = float(contacts[0][0]) if contacts else None
    endpoint_names = [f"{ROPE_PREFIX}B_first", f"{ROPE_PREFIX}B_last"]

    meta = {
        "episode_id": episode_id,
        "counterfactual_bundle_id": bundle_id,
        "task_id": task_id,
        "subfamily": task.subfamily,
        "simulator_name": "mujoco",
        "simulator_version": mujoco.__version__,
        "random_seed": seed,
        "physics_variant": variant,
        "dt_sim": DT_SIM,
        "dt_control": DT_CTRL,
        "dt_video": 1.0 / args.fps,
        "duration": args.duration,
        "fps": args.fps,
        "n_frames": F,
        "frame_size": [args.width, args.height],
        "camera_names": cam_list,
        "wrist_camera": bool(wrist_cams),
        "video_mode": video_mode,
        "video_paths": (
            [str(video_path.relative_to(out_dirs["root"]))]
            if video_mode == "composite"
            else [str(view_paths[cam].relative_to(out_dirs["root"]))
                  for cam in cam_list]),
        "videos_by_camera": (
            None if video_mode == "composite"
            else {cam: str(view_paths[cam].relative_to(out_dirs["root"]))
                  for cam in cam_list}),
        "video_layout": (
            "horizontal_composite " + "|".join(cam_list)
            if video_mode == "composite"
            else "per_view_separate one_mp4_per_camera videos/<camera>/<episode>.mp4"),
        "states_path": str(states_path.relative_to(out_dirs["root"])),
        "scene_xml_path": os.path.relpath(scene_xml_path, out_dirs["root"]),
        "scene": {
            "background_theme": background.theme,
            "background_summary": background.summary,
            "domain_randomization_note": (
                "background is cosmetic (skybox/floor/table-appearance/lights/"
                "backdrop props); table geometry+friction, rope, arm, cameras "
                "and gravity are identical across every theme"),
        },
        "rope": {
            "rope_model_type": ROPE_MODEL_TYPE,
            "rope_length": material.length,
            "rope_radius": material.radius,
            "rope_density": material.density,
            "rope_linear_density": material.linear_density,
            "stretch_stiffness": None,
            "bend_stiffness": material.bend_stiffness_EI,
            "twist_stiffness": material.twist_stiffness_GJ,
            "young_bend_pa": material.young_bend,
            "shear_twist_pa": material.shear_twist,
            "damping": material.joint_damping,
            "rope_table_friction": material.fric_table,
            "rope_gripper_friction": material.fric_grip,
            "rope_post_friction": (material.fric_post
                                   if task_id in ("wrap_around_post",
                                                  "lift_and_drape") else None),
            "rope_ring_friction": (material.fric_ring
                                   if task_id == "thread_through_ring" else None),
            "num_segments": n_rope,
            "endpoint_body_names": endpoint_names,
            "material_label": material.label,
        },
        "parameter_implementation_notes": PARAMETER_IMPLEMENTATION_NOTES,
        "robot": {
            "embodiment": EMBODIMENT,
            "robot_model": "franka_emika_panda (MuJoCo Menagerie)",
            "robot_asset_path": panda_xml,
            "control_mode": "joint_position_servo",
            "action_mode": "cartesian_ee_waypoints_diff_ik",
            "arm_id": "arm0",
            "full_robot_state": True,
            "state_arrays_in_npz": [f"arm0_{k}" for k in rec],
            "q_final": rec["q"][-1],
            "ee_pose_world_final": np.concatenate([rec["ee_pos"][-1],
                                                   rec["ee_quat"][-1]]),
        },
        "action": {
            "skill_label": task.skill_label,
            "planned_ee_trajectory": [[w.t, *w.pos, w.grip]
                                      for w in task.waypoints],
            "planned_ee_trajectory_format": "[t, x, y, z, gripper(0=closed,1=open)]",
            "executed_ee_trajectory": [[t, *p] for t, p in
                                       zip(times, rec["ee_pos"])],
            "gripper_command_trajectory": [[t, g] for t, g in
                                           zip(times, rec["gripper_cmd"])],
            "endpoint_attachment_method": (
                "none_nonprehensile" if task_id == "sweep_aside"
                else "equality_constraint"),
            "grasp_body": task.grasp.endpoint_body,
            "grasp_body_offset_from_endpoint_m": task.grasp_body_offset_m,
            "attachment_note": ("rope endpoint body welded to hand frame via "
                                "connect equality at t_on, released at t_off; "
                                "TODO replace with contact pinch grasp"),
            "release_time": task.release_time,
            "post_pose_world": task.post_pose_world,
            "ring_pose_world": task.ring_pose_world,
            "ring_radius": task.ring_radius,
            "ring_normal_world": task.ring_normal_world,
            "bar_pose_world": task.bar_pose_world,
            "coil_center_world": task.coil_center_world,
            "twirl_center_world": task.twirl_center_world,
            "threading_proxy": task.threading_proxy,
            "grasp_events": grasp_log,
        },
        "metrics": metrics,
        "contacts_summary": {
            "first_contact_time": first_contact,
            "n_contact_records": len(contacts),
            "sampling": f"per video frame, cap {MAX_CONTACTS_PER_FRAME}/frame",
            "arrays_in_npz": ["contact_time", "contact_frame", "contact_pos",
                              "contact_force", "contact_other"] if contacts else [],
        },
        "notes": task.notes,
    }
    meta = _to_jsonable(meta)
    (out_dirs["metadata"] / f"{episode_id}.json").write_text(
        json.dumps(meta, indent=2))
    return meta, snapshot


# --------------------------------------------------------------------- main

def _sharded_bundles(tasks, seeds, num_shards, shard_index):
    """(task, seed) bundles assigned to this shard (round-robin by index)."""
    bundles = [(t, s) for t in tasks for s in range(seeds)]
    if num_shards <= 1:
        return bundles
    return [b for i, b in enumerate(bundles) if i % num_shards == shard_index]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds-per-task", type=int, default=2)
    ap.add_argument("--variants", type=int, default=3,
                    help="material presets per bundle (max %d)" % len(PRESETS))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--video-mode", choices=["split", "composite"],
                    default="split",
                    help="'split' = one mp4 per view under videos/<camera>/ "
                         "(separate per-view datasets); 'composite' = legacy "
                         "single mp4 with all views hstacked side by side")
    ap.add_argument("--tasks", nargs="*", default=TASK_IDS, choices=TASK_IDS)
    ap.add_argument("--theme", default="random",
                    choices=["random"] + THEME_NAMES,
                    help="force one background theme, or 'random' per bundle")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--no-contact-sheet", action="store_true")
    args = ap.parse_args(argv)

    out_root = Path(args.out).resolve()
    out_dirs = {"root": out_root,
                "videos": out_root / "videos",
                "metadata": out_root / "metadata",
                "states": out_root / "states",
                "contact_sheets": out_root / "contact_sheets",
                "scenes": out_root / "scenes"}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    panda_xml = find_panda_xml()
    print(f"[info] MuJoCo {mujoco.__version__}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"[info] Panda model: {panda_xml}")
    print(f"[info] rope_model_type: {ROPE_MODEL_TYPE}, embodiment: {EMBODIMENT}")
    print(f"[info] themes available: {len(THEME_NAMES)}  materials: {len(PRESETS)}")
    print(f"[info] shard {args.shard_index}/{args.num_shards}, "
          f"theme={args.theme}")

    bundles = _sharded_bundles(args.tasks, args.seeds_per_task,
                               args.num_shards, args.shard_index)
    n_var = min(args.variants, len(PRESETS))
    print(f"[info] {len(bundles)} bundles x {n_var} variants "
          f"= {len(bundles) * n_var} episodes this shard")

    episodes, snapshots, failures = [], [], []
    for task_id, seed in bundles:
        for variant in range(n_var):
            eid = f"{task_id}__seed{seed:03d}__var{variant:02d}"
            tic = _time.time()
            try:
                meta, snap = run_episode(task_id, seed, variant, args,
                                         panda_xml, out_dirs)
                episodes.append(meta)
                snapshots.append((eid, meta["scene"]["background_theme"],
                                  meta["rope"]["material_label"],
                                  meta["metrics"]["success_label"], snap))
                print(f"[ok]   {eid}  ({_time.time()-tic:.1f}s)  "
                      f"[{meta['scene']['background_theme']}] "
                      f"{meta['metrics']['success_label']}")
            except Exception as e:  # noqa: BLE001
                failures.append((eid, f"{type(e).__name__}: {e}"))
                print(f"[FAIL] {eid}: {type(e).__name__}: {e}")
                traceback.print_exc()

    tag = (f"_shard{args.shard_index:03d}" if args.num_shards > 1 else "")
    jsonl_path = out_root / f"episodes{tag}.jsonl"
    with open(jsonl_path, "w") as f:
        for m in episodes:
            f.write(json.dumps(m) + "\n")

    # Per-view manifests: one episodes*.jsonl per camera, each record pointing at
    # that single view's mp4, so videos/<cam>/ + episodes<tag>__<cam>.jsonl form a
    # standalone single-view dataset. (Only emitted in split mode.)
    per_view_manifests = []
    if args.video_mode == "split" and episodes:
        cams = list(dict.fromkeys(
            c for m in episodes for c in m.get("camera_names", [])))
        for cam in cams:
            rows = []
            for m in episodes:
                by_cam = m.get("videos_by_camera") or {}
                if cam not in by_cam:
                    continue
                mv = dict(m)
                mv["camera"] = cam
                mv["video_paths"] = [by_cam[cam]]
                rows.append(mv)
            if not rows:
                continue
            vpath = out_root / f"episodes{tag}__{cam}.jsonl"
            with open(vpath, "w") as f:
                for mv in rows:
                    f.write(json.dumps(mv) + "\n")
            per_view_manifests.append((cam, vpath, len(rows)))

    parquet_note = ""
    try:
        import pandas as pd  # noqa: PLC0415
        flat = []
        for m in episodes:
            row = {k: v for k, v in m.items() if not isinstance(v, (dict, list))}
            row["background_theme"] = m["scene"]["background_theme"]
            row.update({f"rope_{k}" if not k.startswith("rope") else k: v
                        for k, v in m["rope"].items()
                        if not isinstance(v, (dict, list))})
            row.update({f"metric_{k}": v for k, v in m["metrics"].items()
                        if not isinstance(v, (dict, list))})
            row["embodiment"] = m["robot"]["embodiment"]
            row["skill_label"] = m["action"]["skill_label"]
            flat.append(row)
        pd.DataFrame(flat).to_parquet(out_root / f"episodes{tag}.parquet")
        parquet_note = str(out_root / f"episodes{tag}.parquet")
    except Exception as e:  # noqa: BLE001
        parquet_note = f"skipped ({type(e).__name__})"

    sheet_path = out_dirs["contact_sheets"] / f"contact_sheet{tag}.png"
    if snapshots and not args.no_contact_sheet:
        tiles = [_label_tile(s, f"{eid[:22]} [{th}/{mat}] {succ}")
                 for eid, th, mat, succ, s in snapshots]
        cols = min(max(n_var, 1), 4)
        rows = (len(tiles) + cols - 1) // cols
        tw, th_ = tiles[0].size
        sheet = Image.new("RGB", (cols * tw, rows * th_), (10, 10, 12))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * tw, (i // cols) * th_))
        sheet.save(sheet_path)

    n_succ = sum(m["metrics"]["success_label"] == "success" for m in episodes)
    print("\n" + "=" * 72)
    print(f"Output directory : {out_root}")
    print(f"Episodes         : {len(episodes)} ok ({len(failures)} failures), "
          f"{n_succ} success")
    themes_used = {}
    for m in episodes:
        t = m["scene"]["background_theme"]
        themes_used[t] = themes_used.get(t, 0) + 1
    print(f"Themes used      : {themes_used}")
    print(f"Contact sheet    : {sheet_path}")
    print(f"episodes.jsonl   : {jsonl_path}")
    print(f"parquet          : {parquet_note}")
    print(f"video_mode       : {args.video_mode}")
    if per_view_manifests:
        for cam, vpath, n in per_view_manifests:
            print(f"view[{cam:>10}]  : {n} eps -> {vpath}")
    if failures:
        print("\nFailures:")
        for eid, err in failures:
            print(f"  {eid}: {err}")
    return 0 if episodes else 1


if __name__ == "__main__":
    sys.exit(main())
