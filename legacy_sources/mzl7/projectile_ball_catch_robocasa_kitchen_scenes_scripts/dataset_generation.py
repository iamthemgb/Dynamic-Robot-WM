from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .run_rollout import run_episode
from .scene_builder import _projectile_launch_velocity, sample_episode


FAMILY_SPECS = {
    "style020_seed0": {
        "base_seed": 0,
        "layout_id": 11,
        "style_id": 20,
        "camera_variant": None,
    },
    "style020_seed0_opposite_camera": {
        "base_seed": 0,
        "layout_id": 11,
        "style_id": 20,
        "camera_variant": "opposite_counter_side",
    },
    "style055_seed1000": {
        "base_seed": 1000,
        "layout_id": 31,
        "style_id": 55,
        "camera_variant": None,
    },
    "style055_seed1000_opposite_camera": {
        "base_seed": 1000,
        "layout_id": 31,
        "style_id": 55,
        "camera_variant": "opposite_counter_side",
    },
}

BALL_COLORS = (
    (0.88, 0.18, 0.16, 1.0),
    (0.16, 0.72, 0.24, 1.0),
    (0.90, 0.54, 0.10, 1.0),
    (0.14, 0.62, 0.82, 1.0),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset generation helpers for the fixed neutral-style projectile-catch families.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    episode_parser = subparsers.add_parser("episode", help="Generate one episode for a fixed family.")
    episode_parser.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
    episode_parser.add_argument("--episode-index", type=int, required=True)
    episode_parser.add_argument("--episode-count", type=int, default=750)
    episode_parser.add_argument("--dataset-root", type=Path, required=True)
    episode_parser.add_argument("--width", type=int, default=832)
    episode_parser.add_argument("--height", type=int, default=480)
    episode_parser.add_argument("--fps", type=int, default=30)
    episode_parser.add_argument("--duration", type=float, default=2.5)

    finalize_parser = subparsers.add_parser("finalize", help="Aggregate episode JSON metadata into a family Parquet index.")
    finalize_parser.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
    finalize_parser.add_argument("--dataset-root", type=Path, required=True)

    return parser.parse_args()


def _branch_schedule(family: str, episode_count: int) -> list[str]:
    success_count = int(round(episode_count * 0.50))
    near_miss_count = int(round(episode_count * 0.20))
    contact_failure_count = int(round(episode_count * 0.20))
    wrong_action_count = episode_count - success_count - near_miss_count - contact_failure_count
    branches = (
        ["success"] * success_count
        + ["spatial_near_miss"] * near_miss_count
        + ["contact_failure"] * contact_failure_count
        + ["wrong_action"] * wrong_action_count
    )
    seed = sum(ord(char) for char in family) + 17 * episode_count
    order = np.random.default_rng(seed).permutation(episode_count)
    return [branches[int(i)] for i in order]


def _episode_rng(family: str, episode_index: int) -> np.random.Generator:
    seed = (sum(ord(char) for char in family) * 1_000_003 + 97 * episode_index) % (2**32)
    return np.random.default_rng(seed)


def _family_dir(dataset_root: Path, family: str) -> Path:
    return dataset_root / family


def _episode_dir(dataset_root: Path, family: str, episode_index: int) -> Path:
    return _family_dir(dataset_root, family) / "episodes" / f"{family}_{episode_index:06d}"


def _tags_for_branch(family: str, episode_index: int, branch: str) -> dict:
    failure_mode = "none"
    outcome = "success"
    if branch == "spatial_near_miss":
        outcome = "failure"
        failure_mode = "spatial_near_miss"
    elif branch == "contact_failure":
        outcome = "failure"
        failure_mode = "contact_failure"
    elif branch == "wrong_action":
        outcome = "failure"
        failure_mode = "wrong_action"
    return {
        "episode_id": f"{family}_{episode_index:06d}",
        "family": family,
        "subfamily": family,
        "branch": branch,
        "outcome": outcome,
        "failure_mode": failure_mode,
    }


def _build_episode_sample(
    *,
    family: str,
    episode_index: int,
    episode_count: int,
    width: int,
    height: int,
    fps: int,
    duration: float,
):
    spec = FAMILY_SPECS[family]
    rng = _episode_rng(family, episode_index)
    branch = _branch_schedule(family, episode_count)[episode_index]
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

    launch = np.asarray(sample.ball_initial_position, dtype=np.float64).copy()
    launch[:2] += rng.uniform((-0.08, -0.05), (0.08, 0.05))
    launch[2] += float(rng.uniform(-0.015, 0.030))

    catch = np.asarray(sample.visual_settings["catch_position_xyz"], dtype=np.float64).copy()
    catch[:2] += rng.uniform((-0.025, -0.025), (0.025, 0.025))
    catch[2] += float(rng.uniform(-0.015, 0.020))

    controller_target_offset = np.zeros(3, dtype=np.float64)
    controller_mode = "ballistic_intercept"
    enable_grasp_capture = True
    close_lead_time_offset_s = float(rng.uniform(-0.003, 0.004))
    reach_lead_time_offset_s = float(rng.uniform(-0.010, 0.010))

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
        launch_angle_degrees=float(rng.uniform(58.0, 70.0)),
        gravity_z=-9.81,
    )
    velocity *= float(rng.uniform(0.98, 1.02))

    visual_settings = dict(sample.visual_settings or {})
    visual_settings["launch_position_xy"] = [float(launch[0]), float(launch[1])]
    visual_settings["catch_position_xyz"] = [float(catch[0]), float(catch[1]), float(catch[2])]
    visual_settings["dataset_texture_variant"] = int(rng.integers(0, 8))
    visual_settings["dataset_background_variant"] = int(rng.integers(0, 8))
    visual_settings["dataset_asset_color_variant"] = int(rng.integers(0, 8))

    sample = replace(
        sample,
        ball_initial_position=tuple(float(v) for v in launch),
        ball_initial_velocity=tuple(float(v) for v in velocity),
        ball_color=BALL_COLORS[int(rng.integers(0, len(BALL_COLORS)))],
        release_time_s=float(rng.uniform(0.18, 0.34)),
        catch_center_z=float(catch[2]),
        offscreen_width=width,
        offscreen_height=height,
        visual_settings=visual_settings,
        controller_target_offset=tuple(float(v) for v in controller_target_offset),
        controller_mode=controller_mode,
        enable_grasp_capture=enable_grasp_capture,
        close_lead_time_offset_s=close_lead_time_offset_s,
        reach_lead_time_offset_s=reach_lead_time_offset_s,
        dataset_tags=_tags_for_branch(family, episode_index, branch),
    )
    return sample


def generate_episode(args: argparse.Namespace) -> None:
    output_dir = _episode_dir(args.dataset_root, args.family, args.episode_index)
    sample = _build_episode_sample(
        family=args.family,
        episode_index=args.episode_index,
        episode_count=args.episode_count,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
    )
    metadata = run_episode(
        output_dir=output_dir,
        sample=sample,
        width=args.width,
        height=args.height,
        fps=args.fps,
        episode_index=args.episode_index,
    )
    print(output_dir)
    print(metadata["success"])


def _flatten_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def finalize_family(args: argparse.Namespace) -> None:
    family_dir = _family_dir(args.dataset_root, args.family)
    episode_dirs = sorted((family_dir / "episodes").glob(f"{args.family}_*/metadata.json"))
    rows = []
    for metadata_path in episode_dirs:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        row = {key: _flatten_value(value) for key, value in metadata.items()}
        row["metadata_json_path"] = str(metadata_path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No episode metadata found under {family_dir / 'episodes'}")
    dataframe = pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
    parquet_path = family_dir / "metadata.parquet"
    dataframe.to_parquet(parquet_path, index=False)
    info = {
        "family": args.family,
        "episode_count": int(len(dataframe)),
        "views": ["main_camera", "side_camera"],
        "fps": 30,
        "resolution": [832, 480],
        "codec": "H.264 MP4",
        "state_control_frequency_hz": [60, 120],
        "metadata_parquet": str(parquet_path),
    }
    (family_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    print(parquet_path)


def main() -> None:
    args = _parse_args()
    if args.command == "episode":
        generate_episode(args)
        return
    if args.command == "finalize":
        finalize_family(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
