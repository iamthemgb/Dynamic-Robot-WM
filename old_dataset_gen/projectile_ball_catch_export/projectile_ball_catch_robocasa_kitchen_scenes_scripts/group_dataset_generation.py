from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from .dataset_generation import BALL_COLORS, FAMILY_SPECS, _flatten_value
from .run_rollout import run_episode
from .scene_builder import _projectile_launch_velocity, sample_episode


SCHEMA = "v3_state_groups"
PHYSICS_REGIME = "nominal_constant_v3"
BRANCH_ORDER = ("success", "spatial_near_miss", "contact_failure", "wrong_action")

DEFAULT_CONFIG = {
    "phase": "unnamed",
    "schema": SCHEMA,
    "siblings_per_group": 4,
    "groups_per_family": 500,
    "branch_mix": {
        "success": 0.50,
        "spatial_near_miss": 0.20,
        "contact_failure": 0.20,
        "wrong_action": 0.10,
    },
    "ic_ranges": {
        "launch_xy_jitter_m": [0.16, 0.10],
        "launch_z_jitter_m": [-0.015, 0.030],
        "catch_xy_jitter_m": [0.025, 0.025],
        "catch_z_jitter_m": [-0.015, 0.020],
        "launch_angle_deg": [55.0, 72.0],
        "speed_scale": [0.98, 1.02],
        "release_time_s": [0.15, 0.40],
        "close_lead_time_offset_s": [-0.003, 0.004],
        "reach_lead_time_offset_s": [-0.010, 0.010],
    },
    "split_percent": {"train": 90, "val": 5, "test": 5},
}


def _rng_from_key(*parts) -> np.random.Generator:
    key = "/".join(str(part) for part in parts)
    digest = hashlib.blake2s(key.encode("utf-8")).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype=np.uint64))


def _load_phase_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    if merged["schema"] != SCHEMA:
        raise ValueError(f"Phase config schema {merged['schema']!r} does not match {SCHEMA!r}.")
    return merged


def group_id_for(family: str, group_index: int) -> str:
    return f"{family}_g{group_index:06d}"


def split_for_group(group_id: str, split_percent: dict) -> str:
    bucket = int.from_bytes(hashlib.blake2s(group_id.encode("utf-8")).digest()[:8], "big") % 100
    train_edge = int(split_percent["train"])
    val_edge = train_edge + int(split_percent["val"])
    if bucket < train_edge:
        return "train"
    if bucket < val_edge:
        return "val"
    return "test"


def _branch_schedule(family: str, config: dict) -> list[str]:
    total = int(config["groups_per_family"]) * int(config["siblings_per_group"])
    mix = config["branch_mix"]
    counts = [int(round(total * float(mix[name]))) for name in BRANCH_ORDER[:-1]]
    counts.append(total - sum(counts))
    if counts[-1] < 0:
        raise ValueError(f"branch_mix ratios sum above 1.0: {mix}")
    branches: list[str] = []
    for name, count in zip(BRANCH_ORDER, counts):
        branches.extend([name] * count)
    order = _rng_from_key(family, config["phase"], "branch-schedule").permutation(total)
    return [branches[int(i)] for i in order]


def _branch_for(family: str, config: dict, group_index: int, ic_index: int) -> str:
    siblings = int(config["siblings_per_group"])
    return _branch_schedule(family, config)[group_index * siblings + ic_index]


def _tags_for_branch(episode_id: str, family: str, branch: str) -> dict:
    failure_mode = "none"
    outcome = "success"
    if branch in {"spatial_near_miss", "contact_failure", "wrong_action"}:
        outcome = "failure"
        failure_mode = branch
    return {
        "episode_id": episode_id,
        "family": family,
        "subfamily": family,
        "branch": branch,
        "outcome": outcome,
        "failure_mode": failure_mode,
    }


def _build_group_context(family: str, group_index: int, *, fps: int, duration: float):
    """Draw everything siblings share: scene, appearance, camera, ball properties."""
    spec = FAMILY_SPECS[family]
    rng = _rng_from_key(family, group_index, "group")
    sample = sample_episode(
        rng,
        "robocasa_kitchen",
        spec["base_seed"],
        fps=fps,
        duration=duration,
        layout_id_override=spec["layout_id"],
        style_id_override=spec["style_id"],
        camera_variant=spec["camera_variant"],
    )
    visual_settings = dict(sample.visual_settings or {})
    visual_settings["dataset_texture_variant"] = int(rng.integers(0, 8))
    visual_settings["dataset_background_variant"] = int(rng.integers(0, 8))
    visual_settings["dataset_asset_color_variant"] = int(rng.integers(0, 8))
    return replace(
        sample,
        ball_color=BALL_COLORS[int(rng.integers(0, len(BALL_COLORS)))],
        visual_settings=visual_settings,
    )


def _build_sibling_sample(
    base_sample,
    *,
    family: str,
    group_index: int,
    ic_index: int,
    branch: str,
    split: str,
    sibling_ids: list[str],
    config: dict,
    width: int,
    height: int,
):
    """Draw this sibling's initial conditions and plan tags; appearance stays the group's."""
    ranges = config["ic_ranges"]
    rng = _rng_from_key(family, group_index, "ic", ic_index)

    launch = np.asarray(base_sample.ball_initial_position, dtype=np.float64).copy()
    launch_jx, launch_jy = (float(v) for v in ranges["launch_xy_jitter_m"])
    launch[:2] += rng.uniform((-launch_jx, -launch_jy), (launch_jx, launch_jy))
    launch[2] += float(rng.uniform(*ranges["launch_z_jitter_m"]))

    catch = np.asarray(base_sample.visual_settings["catch_position_xyz"], dtype=np.float64).copy()
    catch_jx, catch_jy = (float(v) for v in ranges["catch_xy_jitter_m"])
    catch[:2] += rng.uniform((-catch_jx, -catch_jy), (catch_jx, catch_jy))
    catch[2] += float(rng.uniform(*ranges["catch_z_jitter_m"]))

    controller_target_offset = np.zeros(3, dtype=np.float64)
    controller_mode = "ballistic_intercept"
    enable_grasp_capture = True
    close_lead_time_offset_s = float(rng.uniform(*ranges["close_lead_time_offset_s"]))
    reach_lead_time_offset_s = float(rng.uniform(*ranges["reach_lead_time_offset_s"]))

    if branch == "spatial_near_miss":
        lateral = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.09, 0.14))
        controller_target_offset[:2] = (0.0, lateral)
        enable_grasp_capture = False
    elif branch == "contact_failure":
        lateral = float(rng.choice((-1.0, 1.0)) * rng.uniform(0.035, 0.060))
        controller_target_offset[:2] = (0.0, lateral)
        enable_grasp_capture = False
        close_lead_time_offset_s = float(rng.uniform(0.008, 0.020))
        reach_lead_time_offset_s = float(rng.uniform(0.000, 0.020))
    elif branch == "wrong_action":
        controller_mode = "hold_home"
        enable_grasp_capture = False

    velocity = _projectile_launch_velocity(
        launch_position=launch,
        target_position=catch,
        launch_angle_degrees=float(rng.uniform(*ranges["launch_angle_deg"])),
        gravity_z=-9.81,
    )
    velocity *= float(rng.uniform(*ranges["speed_scale"]))
    release_time_s = float(rng.uniform(*ranges["release_time_s"]))

    visual_settings = dict(base_sample.visual_settings or {})
    visual_settings["launch_position_xy"] = [float(launch[0]), float(launch[1])]
    visual_settings["catch_position_xyz"] = [float(catch[0]), float(catch[1]), float(catch[2])]

    group_id = group_id_for(family, group_index)
    episode_id = f"{group_id}_s{ic_index:02d}"
    tags = _tags_for_branch(episode_id, family, branch)
    tags["group"] = {
        "group_id": group_id,
        "sibling_index": ic_index,
        "sibling_ids": list(sibling_ids),
        "split": split,
        "physics_regime": PHYSICS_REGIME,
    }

    return replace(
        base_sample,
        ball_initial_position=tuple(float(v) for v in launch),
        ball_initial_velocity=tuple(float(v) for v in velocity),
        release_time_s=release_time_s,
        catch_center_z=float(catch[2]),
        offscreen_width=width,
        offscreen_height=height,
        visual_settings=visual_settings,
        controller_target_offset=tuple(float(v) for v in controller_target_offset),
        controller_mode=controller_mode,
        enable_grasp_capture=enable_grasp_capture,
        close_lead_time_offset_s=close_lead_time_offset_s,
        reach_lead_time_offset_s=reach_lead_time_offset_s,
        dataset_tags=tags,
    )


def _shared_context_record(base_sample, spec: dict) -> dict:
    visual_settings = base_sample.visual_settings or {}
    return {
        "layout_id": spec["layout_id"],
        "style_id": spec["style_id"],
        "camera_variant": spec["camera_variant"],
        "base_seed": spec["base_seed"],
        "camera_jitter": [float(v) for v in base_sample.camera_jitter],
        "lighting_intensity": float(base_sample.lighting_intensity),
        "floor_material_jitter": float(base_sample.floor_material_jitter),
        "ball_radius_m": float(base_sample.ball_radius),
        "ball_mass_kg": float(base_sample.ball_mass),
        "ball_color": [float(v) for v in base_sample.ball_color],
        "robot_base_position": [float(v) for v in base_sample.robot_base_position],
        "robot_base_yaw": float(base_sample.robot_base_euler[2]),
        "tabletop_height": None if base_sample.tabletop_height is None else float(base_sample.tabletop_height),
        "dataset_texture_variant": visual_settings.get("dataset_texture_variant"),
        "dataset_background_variant": visual_settings.get("dataset_background_variant"),
        "dataset_asset_color_variant": visual_settings.get("dataset_asset_color_variant"),
    }


def _sibling_record(sample, ic_index: int, metadata: dict | None) -> dict:
    tags = sample.dataset_tags or {}
    record = {
        "episode_id": tags.get("episode_id"),
        "ic_index": ic_index,
        "branch": tags.get("branch"),
        "outcome_tag": tags.get("outcome"),
        "controller_mode": sample.controller_mode,
        "enable_grasp_capture": sample.enable_grasp_capture,
        "state_context": {
            "frame": "world",
            "ball_initial_position_m": [float(v) for v in sample.ball_initial_position],
            "ball_initial_velocity_mps": [float(v) for v in sample.ball_initial_velocity],
            "release_time_s": float(sample.release_time_s),
        },
        "catch_target_xyz_m": list((sample.visual_settings or {}).get("catch_position_xyz", [])),
        "paths": {
            "dir": f"s{ic_index:02d}",
            "main": f"s{ic_index:02d}/main.mp4",
            "side": f"s{ic_index:02d}/side.mp4",
            "metadata": f"s{ic_index:02d}/metadata.json",
        },
    }
    if metadata is not None:
        record["gripper_close_command_time_s"] = metadata["action_context"]["gripper_close_command_time_s"]
        record["intercept_position_m"] = metadata["outcomes"]["intercept_position_m"]
        record["catch_success"] = metadata["outcomes"]["catch_success"]
    return record


def _write_json_atomic(path: Path, payload: dict) -> None:
    # PID-unique tmp name: concurrent array tasks may write the same target
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _ensure_generation_config(dataset_root: Path, config: dict, config_path: Path) -> None:
    target = dataset_root / "generation_config.json"
    if target.exists():
        return
    dataset_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(target, {"schema": SCHEMA, "phase_config_path": str(config_path), "phase_config": config})


def _group_dir(dataset_root: Path, family: str, group_id: str) -> Path:
    return dataset_root / family / "groups" / group_id


def _build_group(args: argparse.Namespace):
    config = _load_phase_config(args.phase_config)
    if args.group_index < 0 or args.group_index >= int(config["groups_per_family"]):
        raise ValueError(
            f"group index {args.group_index} outside [0, {config['groups_per_family']}) for phase {config['phase']!r}"
        )
    family = args.family
    spec = FAMILY_SPECS[family]
    group_id = group_id_for(family, args.group_index)
    split = split_for_group(group_id, config["split_percent"])
    siblings = int(config["siblings_per_group"])
    sibling_ids = [f"{group_id}_s{i:02d}" for i in range(siblings)]
    base_sample = _build_group_context(family, args.group_index, fps=args.fps, duration=args.duration)
    samples = [
        _build_sibling_sample(
            base_sample,
            family=family,
            group_index=args.group_index,
            ic_index=i,
            branch=_branch_for(family, config, args.group_index, i),
            split=split,
            sibling_ids=sibling_ids,
            config=config,
            width=args.width,
            height=args.height,
        )
        for i in range(siblings)
    ]
    return config, spec, group_id, split, base_sample, samples


def generate_group(args: argparse.Namespace) -> None:
    config, spec, group_id, split, base_sample, samples = _build_group(args)
    group_dir = _group_dir(args.dataset_root, args.family, group_id)
    _ensure_generation_config(args.dataset_root, config, args.phase_config)

    sibling_records = []
    for ic_index, sample in enumerate(samples):
        output_dir = group_dir / f"s{ic_index:02d}"
        metadata_path = output_dir / "metadata.json"
        done = (
            metadata_path.exists()
            and (output_dir / "main.mp4").exists()
            and (output_dir / "side.mp4").exists()
        )
        if done and not args.overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            print(f"{output_dir} [skipped: already generated]")
        else:
            metadata = run_episode(
                output_dir=output_dir,
                sample=sample,
                width=args.width,
                height=args.height,
                fps=args.fps,
                episode_index=args.group_index * len(samples) + ic_index,
            )
            print(f"{output_dir} catch_success={metadata['outcomes']['catch_success']}")
        sibling_records.append(_sibling_record(sample, ic_index, metadata))

    _write_json_atomic(
        group_dir / "group.json",
        {
            "group_id": group_id,
            "family": args.family,
            "schema": SCHEMA,
            "phase": config["phase"],
            "split": split,
            "physics_regime": PHYSICS_REGIME,
            "sibling_count": len(samples),
            "shared_context": _shared_context_record(base_sample, spec),
            "siblings": sibling_records,
        },
    )
    print(group_dir / "group.json")


def plan_group(args: argparse.Namespace) -> None:
    """Dry run: build and print the group's sibling table without simulating or writing."""
    config, spec, group_id, split, base_sample, samples = _build_group(args)
    payload = {
        "group_id": group_id,
        "family": args.family,
        "phase": config["phase"],
        "split": split,
        "physics_regime": PHYSICS_REGIME,
        "shared_context": _shared_context_record(base_sample, spec),
        "siblings": [_sibling_record(sample, i, None) for i, sample in enumerate(samples)],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    import pandas as pd

    family_dir = args.dataset_root / args.family
    metadata_paths = sorted(family_dir.glob("groups/*/s*/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No episode metadata found under {family_dir / 'groups'}")
    rows = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row = {key: _flatten_value(value) for key, value in metadata.items()}
        row["metadata_json_path"] = str(metadata_path)
        rows.append(row)
    dataframe = pd.DataFrame(rows).sort_values("episode_id").reset_index(drop=True)
    parquet_path = family_dir / "metadata.parquet"
    dataframe.to_parquet(parquet_path, index=False)

    group_jsons = sorted(family_dir.glob("groups/*/group.json"))
    info = {
        "schema": SCHEMA,
        "family": args.family,
        "physics_regime": PHYSICS_REGIME,
        "group_count": len(group_jsons),
        "episode_count": int(len(dataframe)),
        "views": ["main_camera", "side_camera"],
        "fps": 30,
        "resolution": [832, 480],
        "codec": "H.264 MP4",
        "metadata_parquet": str(parquet_path),
    }
    (family_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    splits: dict[str, str] = {}
    for group_json in sorted(args.dataset_root.glob("*/groups/*/group.json")):
        group = json.loads(group_json.read_text(encoding="utf-8"))
        splits[group["group_id"]] = group["split"]
    _write_json_atomic(args.dataset_root / "splits.json", splits)
    print(parquet_path)
    print(args.dataset_root / "splits.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="State-conditional grouped dataset generation (v3): shared scene per group, per-sibling initial conditions, constant nominal physics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_group_args(sub, with_render_args: bool):
        sub.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
        sub.add_argument("--group-index", type=int, required=True)
        sub.add_argument("--phase-config", type=Path, required=True)
        sub.add_argument("--fps", type=int, default=30)
        sub.add_argument("--duration", type=float, default=2.5)
        if with_render_args:
            sub.add_argument("--dataset-root", type=Path, required=True)
            sub.add_argument("--width", type=int, default=832)
            sub.add_argument("--height", type=int, default=480)
            sub.add_argument("--overwrite", action="store_true")
        else:
            sub.set_defaults(width=832, height=480)

    group_parser = subparsers.add_parser("group", help="Generate all siblings of one group (both views each) and write group.json.")
    add_group_args(group_parser, with_render_args=True)

    plan_parser = subparsers.add_parser("plan", help="Dry run: print the group's sibling table without simulating or writing.")
    add_group_args(plan_parser, with_render_args=False)

    finalize_parser = subparsers.add_parser("finalize", help="Aggregate a family into Parquet and refresh dataset-level splits.json.")
    finalize_parser.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
    finalize_parser.add_argument("--dataset-root", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "group":
        generate_group(args)
    elif args.command == "plan":
        plan_group(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
