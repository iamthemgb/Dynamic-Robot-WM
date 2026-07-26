"""Generate MP4 previews + metadata for the Franka-cloth kitchen family.

All-in-one mode (small runs, from the directory containing the package):

  MUJOCO_GL=egl python -m mujoco_franka_cloth_kitchen_scene_previews.generate_previews \\
      --out mujoco_franka_cloth_kitchen_scene_previews/outputs/preview_001 \\
      --num-seeds 2 --physics-variants 3 \\
      --width 640 --height 480 --fps 24 --duration 5.0

Dataset mode (SLURM array jobs; episode index -> seed=I//3, variant=I%3):

  ... generate_previews episode --task poke_cloth --episode-index 0 \\
      --dataset-root /path/to/dataset
  ... generate_previews finalize --task poke_cloth --dataset-root /path/to/dataset
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

from .control import GraspManager, PandaArm, interp_waypoints
from .kitchen import (layout_for_variant, plan_rig_placement, resolve_island,
                      transform_task_spec)
from .materials import PRESETS, sample_continuous_material
from .scene_builder import compile_episode_model, find_panda_xml
from .tasks import TASK_IDS, build_task_spec
from .views import caption_frame, caption_text, view_label, view_names

DT_SIM = 5.0e-4
DT_CTRL = 1.0e-2
PREROLL = 0.8          # seconds of unrecorded settling / move-to-start
PREROLL_STAGGER = 0.5  # per-arm start delay during pre-roll: keeps dual arms
                       # from sweeping the shared workspace center at the same
                       # time and wedging wrist-to-wrist before recording
MAX_CONTACTS_PER_FRAME = 32


# --------------------------------------------------------------------- util

def _to_jsonable(x):
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _keypoint_indices(model, cloth_center):
    """Indices of 9 cloth keypoints from initial grid-exact vertex positions:
    4 corners, center, 4 edge midpoints."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    v = data.flexvert_xpos.copy()
    rel = v[:, :2] - np.asarray(cloth_center)[:2]
    x, y = rel[:, 0], rel[:, 1]

    def nearest(tx, ty):
        return int(np.argmin((x - tx) ** 2 + (y - ty) ** 2))

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    kp = {
        "corner_mm": nearest(xmin, ymin), "corner_pm": nearest(xmax, ymin),
        "corner_mp": nearest(xmin, ymax), "corner_pp": nearest(xmax, ymax),
        "center": nearest(0.0, 0.0),
        "edge_mid_xm": nearest(xmin, 0.0), "edge_mid_xp": nearest(xmax, 0.0),
        "edge_mid_ym": nearest(0.0, ymin), "edge_mid_yp": nearest(0.0, ymax),
    }
    return list(kp.keys()), np.array(list(kp.values()), dtype=int)


def _scan_cloth_contacts(model, data, cloth_flex_id, t, frame_idx):
    """Contacts involving the cloth flex at this instant."""
    out = []
    force6 = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        geom = con.geom          # (2,) -1 where the side is a flex element
        flex = con.flex          # (2,)
        sides_cloth = [k for k in range(2)
                       if geom[k] == -1 and flex[k] == cloth_flex_id]
        if not sides_cloth:
            continue
        other = 1 - sides_cloth[0]
        if geom[other] >= 0:
            other_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                            int(geom[other])) or
                          f"geom_{int(geom[other])}")
        elif flex[other] == cloth_flex_id:
            other_name = "cloth_self"
        else:
            other_name = f"flex_{int(flex[other])}"
        mujoco.mj_contactForce(model, data, i, force6)
        frame = np.array(con.frame).reshape(3, 3)
        f_world = frame.T @ force6[:3]
        out.append((t, frame_idx, con.pos.copy(), f_world, other_name))
        if len(out) >= MAX_CONTACTS_PER_FRAME:
            break
    return out


def _label_tile(img, caption, tile_w=420):
    h = int(img.shape[0] * tile_w / img.shape[1])
    tile = Image.fromarray(img).resize((tile_w, h))
    canvas = Image.new("RGB", (tile_w, h + 18), (18, 18, 22))
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, h + 3), caption, fill=(230, 230, 230),
              font=ImageFont.load_default())
    return canvas


# ------------------------------------------------------------------ episode

def run_episode(task_id, seed, variant, args, panda_xml, out_dirs):
    if getattr(args, "continuous_materials", False):
        mat_rng = np.random.default_rng(
            [zlib.crc32(task_id.encode()), seed, variant, 0xC017])
        material = sample_continuous_material(mat_rng)
    else:
        material = PRESETS[variant % len(PRESETS)]
    task = build_task_spec(task_id, seed, args.duration, material.rgba)
    episode_id = f"{task_id}__seed{seed:03d}__var{variant:02d}"
    bundle_id = f"{task_id}__seed{seed:03d}"
    scene_xml_path = out_dirs["scenes"] / f"{episode_id}.xml"

    kitchen_spec = None
    rig_offset, rig_yaw, rig_slide = None, 0.0, 0.0
    if args.kitchen:
        layout_id, style_id = layout_for_variant(variant)
        _scene, counter, tabletop_z = resolve_island(layout_id, style_id)
        # per-seed scene randomization (shared across variants of a bundle)
        scene_rng = np.random.default_rng(
            [zlib.crc32(task_id.encode()), seed, 0x5EED])
        rig_offset, rig_yaw, rig_slide = plan_rig_placement(
            task, counter, tabletop_z, rng=scene_rng)
        lighting_intensity = float(scene_rng.uniform(0.85, 1.15))
        cam_jitter = tuple(scene_rng.normal(0.0, 0.03, size=3))
        task = transform_task_spec(task, rig_offset, rig_yaw)
        kitchen_spec = {"layout_id": layout_id, "style_id": style_id,
                        "seed": seed, "rig_yaw": rig_yaw,
                        "lighting_intensity": lighting_intensity,
                        "cam_jitter": cam_jitter}

    model, _xml, connects, wrist_cams, kitchen_meta = compile_episode_model(
        task, material, DT_SIM, panda_xml, scene_xml_path,
        kitchen_spec=kitchen_spec)
    data = mujoco.MjData(model)

    arms = {a.name: PandaArm(model, a.prefix, a.name) for a in task.arms}
    for arm in arms.values():
        arm.reset_home(data)
    mujoco.mj_forward(model, data)
    for arm in arms.values():
        arm.capture_down_orientation(data)
    if len(task.arms) > 1:
        # separate the (otherwise interpenetrating) dual home TCPs
        for arm in arms.values():
            arm.apply_j1_offset(data, 0.5)
        mujoco.mj_forward(model, data)

    grasp_log = []
    grasps = GraspManager(model, connects, arms)
    kp_names, kp_idx = _keypoint_indices(model, task.cloth.center)
    cloth_flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "cloth")

    # --- pre-roll: settle cloth, servo arms to their first waypoints ------
    # Arms start staggered (arm k after k*PREROLL_STAGGER) so dual arms never
    # cross the workspace center simultaneously mid-convergence.
    n_ctrl = int(round(DT_CTRL / DT_SIM))
    preroll = PREROLL + PREROLL_STAGGER * (len(task.arms) - 1)
    for step in range(int(preroll / DT_SIM)):
        if step % n_ctrl == 0:
            t_pre = step * DT_SIM
            for k, spec_arm in enumerate(task.arms):
                if t_pre < PREROLL_STAGGER * k:
                    continue          # hold home until this arm's start time
                pos, grip = interp_waypoints(spec_arm.waypoints, 0.0)
                arms[spec_arm.name].set_targets(pos, grip)
                arms[spec_arm.name].update(data, DT_CTRL)
        mujoco.mj_step(model, data)
    t0 = data.time

    # --- recorded rollout --------------------------------------------------
    # Frame convention: fps*duration + 1 frames, inclusive of both t=0 and
    # t=duration (e.g. 5.0 s @ 24 FPS -> 121 frames), matching the existing
    # dataset preview format (480p, 24 FPS, 121 frames, ~5 s).
    F = int(round(args.duration * args.fps)) + 1
    frame_times = np.arange(F) / args.fps
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam_list = ["front", "top"] + ([wrist_cams[0]] if wrist_cams else [])

    video_path = out_dirs["videos"] / f"{episode_id}.mp4"
    writer = imageio.get_writer(str(video_path), fps=args.fps,
                                codec="libx264", quality=8,
                                macro_block_size=1, pixelformat="yuv420p")

    # Per-view videos: one captioned clip per camera panel (main/top/wrist).
    view_labels = view_names(cam_list)
    view_video_paths = {
        cam: out_dirs["videos"] / f"{episode_id}__{lbl}.mp4"
        for cam, lbl in zip(cam_list, view_labels)}
    view_writers = {
        cam: imageio.get_writer(str(view_video_paths[cam]), fps=args.fps,
                                codec="libx264", quality=8,
                                macro_block_size=1, pixelformat="yuv420p")
        for cam in cam_list}

    rec = {name: {"q": [], "dq": [], "q_target": [], "tau_cmd": [],
                  "ee_pos": [], "ee_quat": [], "ee_vel": [],
                  "gripper_width": [], "gripper_cmd": [],
                  "gripper_force_cmd": []} for name in arms}
    times, keypoints, vertices, contacts = [], [], [], []
    snapshot = None
    snapshot_frame = int(0.55 * F)

    total_steps = int(round(args.duration / DT_SIM)) + 1
    next_f = 0
    for step in range(total_steps):
        t = data.time - t0
        if step % n_ctrl == 0:
            for spec_arm in task.arms:
                pos, grip = interp_waypoints(spec_arm.waypoints, t)
                arms[spec_arm.name].set_targets(pos, grip)
                arms[spec_arm.name].update(data, DT_CTRL)
            grasps.update(data, t, grasp_log)

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
            for cam, panel in zip(cam_list, panels):
                view_writers[cam].append_data(
                    caption_frame(panel, caption_text(view_label(cam, cam_list.index(cam)))))
            if next_f == snapshot_frame:
                snapshot = panels[0].copy()

            times.append(t)
            for name, arm in arms.items():
                r = rec[name]
                r["q"].append(data.qpos[arm.qpos_adr].copy())
                r["dq"].append(data.qvel[arm.dof_adr].copy())
                r["q_target"].append(arm.q_target.copy())
                r["tau_cmd"].append(arm.tau_cmd(data))
                p, q = arm.tcp_pose(data)
                r["ee_pos"].append(p.copy())
                r["ee_quat"].append(q.copy())
                r["ee_vel"].append(arm.tcp_velocity(data))
                r["gripper_width"].append(arm.gripper_width(data))
                r["gripper_cmd"].append(arm.grip_cmd)
                r["gripper_force_cmd"].append(arm.gripper_force_cmd(data))
            keypoints.append(data.flexvert_xpos[kp_idx].copy())
            vertices.append(data.flexvert_xpos.copy().astype(np.float32))
            contacts.extend(_scan_cloth_contacts(model, data, cloth_flex_id,
                                                 t, next_f))
            next_f += 1

        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f"simulation diverged at t={t:.3f}s")

    writer.close()
    for w in view_writers.values():
        w.close()
    renderer.close()
    if snapshot is None:
        snapshot = panels[0].copy()

    # --- states npz ---------------------------------------------------------
    states_path = out_dirs["states"] / f"{episode_id}.npz"
    npz = {
        "time": np.asarray(times),
        "cloth_keypoints": np.asarray(keypoints, dtype=np.float32),
        "keypoint_names": np.array(kp_names),
        "cloth_vertices": np.asarray(vertices, dtype=np.float32),
    }
    for name, r in rec.items():
        for k, v in r.items():
            npz[f"{name}_{k}"] = np.asarray(v, dtype=np.float32)
    if contacts:
        npz["contact_time"] = np.array([c[0] for c in contacts], dtype=np.float32)
        npz["contact_frame"] = np.array([c[1] for c in contacts], dtype=np.int32)
        npz["contact_pos"] = np.array([c[2] for c in contacts], dtype=np.float32)
        npz["contact_force"] = np.array([c[3] for c in contacts], dtype=np.float32)
        npz["contact_other"] = np.array([c[4] for c in contacts])
    np.savez_compressed(states_path, **npz)

    # --- metadata -----------------------------------------------------------
    area = task.cloth.area
    first_contact = float(contacts[0][0]) if contacts else None
    meta = {
        "episode_id": episode_id,
        "counterfactual_bundle_id": bundle_id,
        "task_id": task_id,
        "subfamily": task.subfamily,
        "simulator_name": "mujoco",
        "simulator_version": mujoco.__version__,
        "random_seed": seed,
        "physics_variant": variant,
        "outcome_branch": task.outcome_branch,
        "branch_note": task.branch_note,
        "dt_sim": DT_SIM,
        "dt_control": DT_CTRL,
        "dt_video": 1.0 / args.fps,
        "duration": args.duration,
        "fps": args.fps,
        "n_frames": F,
        "frame_size": [args.width, args.height],
        "camera_name": "+".join(cam_list),
        "cameras": cam_list,
        "wrist_camera": bool(wrist_cams),
        "video_path": str(video_path.relative_to(out_dirs["root"])),
        "view_videos": {view_label(cam, i): str(view_video_paths[cam].relative_to(out_dirs["root"]))
                        for i, cam in enumerate(cam_list)},
        "states_path": str(states_path.relative_to(out_dirs["root"])),
        "scene_xml_path": os.path.relpath(scene_xml_path, out_dirs["root"]),
        "cloth": {
            "stretch_stiffness": material.stretch_stiffness,
            "bend_stiffness": material.bend_stiffness,
            "shear_stiffness": material.shear_stiffness,
            "density": material.density(area),
            "areal_density": material.areal_density(area),
            "thickness": material.thickness,
            "cloth_table_friction": material.fric_table,
            "cloth_gripper_friction": material.fric_grip,
            "damping": material.edge_damping,
            "material_label": material.label,
            "young": material.young,
            "poisson": material.poisson,
            "mass": material.mass,
            "cloth_grid_resolution": list(task.cloth.count),
            "cloth_size": list(task.cloth.size),
            "cloth_spacing": list(task.cloth.spacing),
            "cloth_rgba": list(task.cloth.rgba),
            "cloth_texture_kind": task.cloth.texture_kind,
            "cloth_rgba2": list(task.cloth.rgba2),
            "cloth_texrepeat": task.cloth.texrepeat,
            "cloth_center_initial": task.cloth.center,
            "tshirt_proxy": task.tshirt_proxy,
            "stiffness_note": ("stretch/bend/shear are thin-shell continuum values "
                               "derived from (young, poisson, thickness); stretch is "
                               "additionally constrained by flex edge equality"),
        },
        "robot": {
            "embodiment": task.embodiment,
            "robot_model": "franka_emika_panda (MuJoCo Menagerie)",
            "robot_model_xml": panda_xml,
            "control_mode": "joint_position_servo",
            "action_mode": "cartesian_ee_waypoints_diff_ik",
            "dual_franka_proxy": task.dual_franka_proxy,
            "full_robot_state": True,
            "arms": [{
                "arm_id": a.name,
                "prefix": a.prefix,
                "base_pos": a.base_pos,
                "base_yaw": a.base_yaw,
                "q_final": rec[a.name]["q"][-1],
                "dq_final": rec[a.name]["dq"][-1],
                "q_target_final": rec[a.name]["q_target"][-1],
                "ee_pose_world_final": np.concatenate(
                    [rec[a.name]["ee_pos"][-1], rec[a.name]["ee_quat"][-1]]),
                "ee_velocity_world_final": rec[a.name]["ee_vel"][-1],
                "gripper_width_final": rec[a.name]["gripper_width"][-1],
                "state_arrays_in_npz": [f"{a.name}_{k}" for k in rec[a.name]],
            } for a in task.arms],
        },
        "action": {
            "skill_label": task.skill_label,
            "planned_ee_trajectory": {
                a.name: [[w.t, *w.pos, w.grip] for w in a.waypoints]
                for a in task.arms},
            "planned_ee_trajectory_format": "[t, x, y, z, gripper(0=closed,1=open)]",
            "executed_ee_trajectory": {
                name: [[t, *p] for t, p in zip(times, r["ee_pos"])]
                for name, r in rec.items()},
            "gripper_command_trajectory": {
                name: [[t, g] for t, g in zip(times, r["gripper_cmd"])]
                for name, r in rec.items()},
            "release_time": task.release_time,
            "fold_line_world": task.fold_line_world,
            "box_pose_world": task.box_pose_world,
            "grasp_mechanism": "equality_connect_proxy",
            "grasp_events": grasp_log,
            "grasp_note": ("proxy grasp: cloth vertex welded to hand body via "
                           "connect equality at t_on, released at t_off; "
                           "TODO replace with real contact grasping"),
        },
        "cloth_keypoints": {
            "names": kp_names,
            "vertex_indices": kp_idx,
            "trajectory_in_npz": "cloth_keypoints",
        },
        "kitchen": None if kitchen_meta is None else {
            **kitchen_meta,
            "rig_offset": rig_offset,
            "rig_yaw": rig_yaw,
            "rig_slide_along_island": rig_slide,
            "lighting_intensity": kitchen_spec["lighting_intensity"],
            "camera_jitter": list(kitchen_spec["cam_jitter"]),
        },
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
    meta_path = out_dirs["metadata"] / f"{episode_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta, snapshot


# ------------------------------------------------------------ dataset CLI

def _dataset_cli(argv):
    """`episode` / `finalize` subcommands for SLURM-array dataset generation.

    episode index -> (seed, variant): seed = idx // variants_per_seed,
    variant = idx % variants_per_seed. Idempotent per episode id.
    """
    ap = argparse.ArgumentParser(prog="generate_previews dataset mode")
    sub = ap.add_subparsers(dest="command", required=True)

    ep = sub.add_parser("episode", help="generate one episode (array job)")
    ep.add_argument("--task", required=True, choices=TASK_IDS)
    ep.add_argument("--episode-index", type=int, required=True)
    ep.add_argument("--dataset-root", required=True)
    ep.add_argument("--variants-per-seed", type=int, default=3)
    ep.add_argument("--width", type=int, default=640)
    ep.add_argument("--height", type=int, default=480)
    ep.add_argument("--fps", type=int, default=24)
    ep.add_argument("--duration", type=float, default=5.0)
    ep.add_argument("--continuous-materials", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="sample cloth physics continuously instead of the "
                         "5 discrete presets (deterministic per episode)")
    ep.add_argument("--kitchen", action=argparse.BooleanOptionalAction,
                    default=True)

    fin = sub.add_parser("finalize", help="aggregate one task's metadata")
    fin.add_argument("--task", required=True, choices=TASK_IDS)
    fin.add_argument("--dataset-root", required=True)
    fin.add_argument("--expected-count", type=int, default=1500)
    fin.add_argument("--variants-per-seed", type=int, default=3)

    args = ap.parse_args(argv)
    root = Path(args.dataset_root).resolve()

    if args.command == "episode":
        out_dirs = {"root": root,
                    "videos": root / "videos",
                    "metadata": root / "metadata",
                    "states": root / "states",
                    "scenes": root / "scenes"}
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        seed = args.episode_index // args.variants_per_seed
        variant = args.episode_index % args.variants_per_seed
        eid = f"{args.task}__seed{seed:03d}__var{variant:02d}"
        panda_xml = find_panda_xml()
        print(f"[info] MuJoCo {mujoco.__version__}, episode {eid}")
        tic = _time.time()
        meta, _snap = run_episode(args.task, seed, variant, args,
                                  panda_xml, out_dirs)
        print(f"[ok]   {eid}  [{meta['outcome_branch']}]  "
              f"({_time.time()-tic:.1f}s)  -> {meta['video_path']}")
        return 0

    # ------------------------------------------------------------ finalize
    metas, missing = [], []
    for idx in range(args.expected_count):
        seed, variant = divmod(idx, args.variants_per_seed)
        mp = root / "metadata" / f"{args.task}__seed{seed:03d}__var{variant:02d}.json"
        if mp.is_file():
            metas.append(json.loads(mp.read_text()))
        else:
            missing.append(idx)
    jsonl_path = root / f"episodes_{args.task}.jsonl"
    with open(jsonl_path, "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")

    # sampled contact sheet (first 16 episodes) instead of thousands of tiles
    sheet_path = root / f"contact_sheet_{args.task}.png"
    tiles = []
    for m in metas[:16]:
        try:
            r = imageio.get_reader(str(root / m["video_path"]))
            snap = r.get_data(int(0.55 * m["n_frames"]))[:, :m["frame_size"][0]]
            r.close()
            tiles.append(_label_tile(
                snap, f"{m['episode_id']} [{m['cloth']['material_label']}]"))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] sheet tile {m['episode_id']}: {e}")
    if tiles:
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (cols * tw, rows * th), (10, 10, 12))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
        sheet.save(sheet_path)

    from collections import Counter
    branch_counts = Counter(m.get("outcome_branch", "unknown") for m in metas)
    print(f"[finalize] {args.task}: {len(metas)}/{args.expected_count} episodes"
          f" -> {jsonl_path}")
    print(f"[finalize] outcome branches: {dict(branch_counts)}")
    if missing:
        print(f"[finalize] MISSING indices ({len(missing)}): {missing[:40]}"
              f"{' ...' if len(missing) > 40 else ''}")
    return 1 if missing else 0


# --------------------------------------------------------------------- main

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("episode", "finalize"):
        return _dataset_cli(argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-seeds", type=int, default=2)
    ap.add_argument("--physics-variants", type=int, default=3)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--tasks", nargs="*", default=TASK_IDS,
                    choices=TASK_IDS, help="subset of tasks to generate")
    ap.add_argument("--kitchen", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="place the rig on a RoboCasa kitchen island "
                         "(--no-kitchen for the bland table scene)")
    args = ap.parse_args(argv)

    out_root = Path(args.out).resolve()
    out_dirs = {"root": out_root,
                "videos": out_root / "videos",
                "metadata": out_root / "metadata",
                "states": out_root / "states",
                "scenes": Path(__file__).parent / "scenes"}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    panda_xml = find_panda_xml()
    print(f"[info] MuJoCo {mujoco.__version__}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    print(f"[info] Panda model: {panda_xml}")

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
                    snapshots.append((eid, meta["cloth"]["material_label"], snap))
                    print(f"[ok]   {eid}  ({_time.time()-tic:.1f}s)  "
                          f"-> {meta['video_path']}")
                except Exception as e:  # noqa: BLE001 - keep generating others
                    failures.append((eid, f"{type(e).__name__}: {e}"))
                    print(f"[FAIL] {eid}: {type(e).__name__}: {e}")
                    traceback.print_exc()

    # consolidated JSONL
    jsonl_path = out_root / "episodes.jsonl"
    with open(jsonl_path, "w") as f:
        for m in episodes:
            f.write(json.dumps(m) + "\n")

    # parquet if pandas+pyarrow available (flat scalar columns only)
    parquet_note = ""
    try:
        import pandas as pd  # noqa: PLC0415
        flat = []
        for m in episodes:
            row = {k: v for k, v in m.items() if not isinstance(v, (dict, list))}
            row.update({f"cloth_{k}": v for k, v in m["cloth"].items()
                        if not isinstance(v, (dict, list))})
            row["embodiment"] = m["robot"]["embodiment"]
            row["dual_franka_proxy"] = m["robot"]["dual_franka_proxy"]
            row["skill_label"] = m["action"]["skill_label"]
            flat.append(row)
        pd.DataFrame(flat).to_parquet(out_root / "episodes.parquet")
        parquet_note = str(out_root / "episodes.parquet")
    except Exception as e:  # noqa: BLE001
        parquet_note = f"skipped ({type(e).__name__}: pandas/pyarrow unavailable)"

    # contact sheet
    sheet_path = out_root / "contact_sheet.png"
    if snapshots:
        tiles = [_label_tile(s, f"{eid} [{mat}]") for eid, mat, s in snapshots]
        cols = min(args.physics_variants, 4) or 1
        rows = (len(tiles) + cols - 1) // cols
        tw, th = tiles[0].size
        sheet = Image.new("RGB", (cols * tw, rows * th), (10, 10, 12))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
        sheet.save(sheet_path)

    # ----------------------------------------------------------- summary
    print("\n" + "=" * 72)
    print(f"Generated {len(episodes)} episodes, {len(failures)} failures")
    print(f"  videos:        {out_dirs['videos']}")
    print(f"  metadata:      {out_dirs['metadata']}")
    print(f"  states:        {out_dirs['states']}")
    print(f"  scenes (xml):  {out_dirs['scenes']}")
    print(f"  episodes.jsonl {jsonl_path}")
    print(f"  parquet:       {parquet_note}")
    print(f"  contact sheet: {sheet_path}")
    by_task = {}
    for m in episodes:
        by_task.setdefault(m["task_id"], m["robot"])
    print("\nEmbodiment summary:")
    for tid in args.tasks:
        if tid in by_task:
            r = by_task[tid]
            proxy = " (PROXY)" if r["dual_franka_proxy"] else " (full Franka)"
            print(f"  {tid:34s} {r['embodiment']}{proxy}")
        else:
            print(f"  {tid:34s} FAILED - see errors above")
    if failures:
        print("\nFailures:")
        for eid, err in failures:
            print(f"  {eid}: {err}")
    return 0 if episodes and not failures else (0 if episodes else 1)


if __name__ == "__main__":
    sys.exit(main())
