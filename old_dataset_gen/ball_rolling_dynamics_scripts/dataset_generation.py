from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa

from .canonical_writer import (
    SCHEMA_VERSION,
    VideoSpec,
    canonical_camera_name,
    counterfactual_families_schema,
    episode_metadata_table,
    episode_uuid,
    flexible_mapping_rows,
    load_episode_markers,
    sha256_file,
    sha256_json,
    write_episode_marker,
    write_parquet,
    writer_settings,
)
from .run_rollout import (
    DEFAULT_CAMERA_NAMES,
    DEFAULT_CAMERA_STREAMS,
    build_episode_record,
    rollout_episode,
    write_canonical_episode,
)
from .scene_builder import (
    BALL_RADIUS_M,
    ROLLING_HEADING_JITTER_RAD,
    ROLLING_SPEED_RANGE_MPS,
    randomize_initial_velocity,
    sample_episode,
)


FAMILY_SPECS = {
    "rolling_layout38_style42": {
        "base_seed": 38042,
        "layout_id": 38,
        "style_id": 42,
        "camera_variant": "island_rolling_view",
    },
    "rolling_layout48_style41": {
        "base_seed": 48041,
        "layout_id": 48,
        "style_id": 41,
        "camera_variant": "island_rolling_view",
    },
    "rolling_layout51_style34": {
        "base_seed": 51034,
        "layout_id": 51,
        "style_id": 34,
        "camera_variant": "island_rolling_view",
    },
}

FINALIZED_METADATA_ARTIFACTS = (
    "info.json",
    "episodes.parquet",
    "tasks.parquet",
    "cameras.parquet",
    "provenance.parquet",
    "splits.parquet",
    "counterfactual_families.parquet",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset generation for passive ball-rolling-dynamics families.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    episode_parser = subparsers.add_parser("episode", help="Generate one canonical episode for a fixed family.")
    episode_parser.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
    episode_parser.add_argument("--episode-index", type=int, required=True)
    episode_parser.add_argument("--episode-count", type=int, default=750)
    episode_parser.add_argument("--dataset-root", type=Path, required=True)
    episode_parser.add_argument("--width", type=int, default=832)
    episode_parser.add_argument("--height", type=int, default=480)
    episode_parser.add_argument("--fps", type=int, default=30)
    episode_parser.add_argument("--duration", type=float, default=2.5)

    finalize_parser = subparsers.add_parser("finalize", help="Aggregate episode markers into finalized meta/ tables.")
    finalize_parser.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
    finalize_parser.add_argument("--dataset-root", type=Path, required=True)
    finalize_parser.add_argument("--width", type=int, default=832)
    finalize_parser.add_argument("--height", type=int, default=480)

    inspect_parser = subparsers.add_parser("episode-backend", help="Print the scene-resolution backend recorded for one episode.")
    inspect_parser.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
    inspect_parser.add_argument("--episode-index", type=int, required=True)
    inspect_parser.add_argument("--dataset-root", type=Path, required=True)

    return parser.parse_args()


def _episode_seed(family: str, episode_index: int) -> int:
    return (sum(ord(char) for char in family) * 1_000_003 + 97 * episode_index) % (2**32)


def _family_root(dataset_root: Path, family: str) -> Path:
    return dataset_root / family


def _family_config(family: str, args: argparse.Namespace) -> dict:
    spec = FAMILY_SPECS[family]
    return {
        "family": family,
        "base_seed": spec["base_seed"],
        "layout_id": spec["layout_id"],
        "style_id": spec["style_id"],
        "camera_variant": spec["camera_variant"],
        "demo_mode": "ball_rolling_dynamics",
        "randomized_fields": ["ball_initial_velocity"],
        "ball_radius_m": BALL_RADIUS_M,
        "rolling_speed_range_mps": list(ROLLING_SPEED_RANGE_MPS),
        "rolling_heading_jitter_rad": ROLLING_HEADING_JITTER_RAD,
        "width": args.width,
        "height": args.height,
        "fps": getattr(args, "fps", 30),
        "duration": getattr(args, "duration", 2.5),
        "episode_count": getattr(args, "episode_count", None),
    }


def _build_episode_sample(
    *,
    family: str,
    episode_index: int,
    width: int,
    height: int,
    fps: int,
    duration: float,
):
    """Deterministic base scene; only the initial velocity varies per episode."""

    spec = FAMILY_SPECS[family]
    sample = sample_episode(
        "robocasa_kitchen",
        spec["base_seed"],
        fps=fps,
        duration=duration,
        layout_id_override=spec["layout_id"],
        style_id_override=spec["style_id"],
        camera_variant=spec["camera_variant"],
    )
    sample = replace(
        sample,
        offscreen_width=width,
        offscreen_height=height,
        dataset_tags={
            "episode_id": f"{family}_{episode_index:06d}",
            "family": family,
            "subfamily": family,
            "branch": "free_roll",
        },
    )
    velocity_rng = np.random.default_rng(_episode_seed(family, episode_index))
    return randomize_initial_velocity(sample, velocity_rng)


def generate_episode(args: argparse.Namespace) -> None:
    spec = FAMILY_SPECS[args.family]
    root = _family_root(args.dataset_root, args.family)
    video_spec = VideoSpec(width=args.width, height=args.height, fps_num=args.fps)
    config = _family_config(args.family, args)
    config_hash = sha256_json(config)

    sample = _build_episode_sample(
        family=args.family,
        episode_index=args.episode_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
    )
    rollout = rollout_episode(
        sample=sample,
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_names=DEFAULT_CAMERA_NAMES,
    )
    artifacts = write_canonical_episode(
        rollout,
        root=root,
        episode_index=args.episode_index,
        task_index=0,
        camera_streams=DEFAULT_CAMERA_STREAMS,
        video_spec=video_spec,
        overwrite=True,
    )
    record = build_episode_record(
        rollout,
        artifacts=artifacts,
        episode_uuid_text=episode_uuid(args.family, args.episode_index),
        episode_index=args.episode_index,
        task_index=0,
        family=args.family,
        subfamily=args.family,
        scene_seed=spec["base_seed"],
        branch_seed=_episode_seed(args.family, args.episode_index),
        config_hash=config_hash,
    )
    write_episode_marker(root, record, config_hash)

    print(root)
    print(record["objective_metrics"]["initial_speed_mps"])


def _counterfactual_family_rows(records: list[dict]) -> list[dict]:
    """Grouped datasets stamp a shared counterfactual_bundle_id per sibling
    set (group_dataset_generation); bundles with one member are flat episodes
    and emit no row, so this stays empty for ungrouped generation."""

    bundles: dict[str, list[dict]] = {}
    for record in records:
        bundles.setdefault(str(record["counterfactual_bundle_id"]), []).append(record)
    rows = []
    for bundle_id in sorted(bundles):
        members = bundles[bundle_id]
        if len(members) < 2:
            continue
        group_info = (members[0].get("extras") or {}).get("group") or {}
        expected = len(group_info.get("sibling_ids", [])) or len(members)
        rows.append(
            {
                "family_id": bundle_id,
                "relation": "velocity_counterfactual_siblings",
                "split_group_id": str(members[0]["split_group_id"]),
                "expected_member_count": expected,
                "expected_episode_uuids_json": json.dumps(
                    sorted(str(r["episode_uuid"]) for r in members), separators=(",", ":")
                ),
                "intervention_fields_json": json.dumps(["ball_initial_velocity"], separators=(",", ":")),
                "fixed_field_hashes_json": json.dumps(
                    {"config_hash": str(members[0]["config_hash"])}, separators=(",", ":")
                ),
                "table_version": "v1",
            }
        )
    return rows


def _camera_rows(records: list[dict], width: int, height: int) -> list[dict]:
    roles = {
        "observation.images.main": "main_three_quarter_external",
        "observation.images.secondary": "task_specific_secondary",
    }
    camera_poses = {}
    if records:
        camera_poses = records[0].get("extras", {}).get("camera_poses", {}) or {}
    rows = []
    for physical, stream in DEFAULT_CAMERA_STREAMS.items():
        canonical = canonical_camera_name(stream)
        rows.append(
            {
                "camera_id": canonical,
                "stream": canonical,
                "role": roles.get(canonical, "task_specific_secondary"),
                "physical_camera": physical,
                "width": width,
                "height": height,
                "fps": 30,
                "codec": "h264",
                "pixel_format": "yuv420p",
                "calibration": camera_poses.get(physical) or {},
            }
        )
    return rows


def finalize_family(args: argparse.Namespace) -> None:
    root = _family_root(args.dataset_root, args.family)
    records = load_episode_markers(root)
    if not records:
        raise FileNotFoundError(f"No committed episode markers found under {root / '.records'}")
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    spec = FAMILY_SPECS[args.family]
    camera_names = [canonical_camera_name(stream) for stream in DEFAULT_CAMERA_STREAMS.values()]
    simulation_hz = records[0].get("physics", {}).get("simulation_hz")

    write_parquet(meta / "episodes.parquet", episode_metadata_table(records), overwrite=True)
    task_rows = [{"task_index": 0, "family": args.family, "subfamily": args.family}]
    write_parquet(meta / "tasks.parquet", pa.Table.from_pylist(flexible_mapping_rows(task_rows)), overwrite=True)
    write_parquet(
        meta / "cameras.parquet",
        pa.Table.from_pylist(flexible_mapping_rows(_camera_rows(records, args.width, args.height))),
        overwrite=True,
    )
    provenance_rows = [
        {
            "source_generator": "ball_rolling_dynamics_scripts",
            "source_generator_version": "v1",
            "generator_git_commit": "unknown",
            "simulator_name": "mujoco",
            "simulator_version": records[0].get("simulator_version", "unknown"),
            "writer_settings": writer_settings(VideoSpec(width=args.width, height=args.height)),
            "finalized_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    write_parquet(meta / "provenance.parquet", pa.Table.from_pylist(flexible_mapping_rows(provenance_rows)), overwrite=True)
    split_rows = [
        {
            "episode_uuid": record["episode_uuid"],
            "episode_index": int(record["episode_index"]),
            "split_group_id": record["split_group_id"],
            "split": record["split"],
        }
        for record in records
    ]
    write_parquet(meta / "splits.parquet", pa.Table.from_pylist(flexible_mapping_rows(split_rows)), overwrite=True)
    write_parquet(
        meta / "counterfactual_families.parquet",
        pa.Table.from_pylist(_counterfactual_family_rows(records), schema=counterfactual_families_schema()),
        overwrite=True,
    )
    info = {
        "name": f"ball_rolling_dynamics_{args.family}",
        "dataset_uuid": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Passive ball-rolling dynamics on a RoboCasa kitchen island: no robot, "
            "fixed scene and start pose, only the ball's initial velocity varies; "
            "emitted in the canonical dynamic-robot-dataset/v2 rollout format."
        ),
        "coordinate_convention": {
            "units": "SI",
            "handedness": "right",
            "up_axis": "+Z",
            "quaternion_order": "WXYZ",
            "pose_convention": "position_then_quaternion",
            "default_velocity_frame": "world",
        },
        "time_base": {
            "sim_hz": simulation_hz or 240.0,
            "control_hz": simulation_hz or 240.0,
            "video_hz": 30.0,
            "timestamp_dtype": "float64_seconds",
            "video_time_base_num": 1,
            "video_time_base_den": 30,
        },
        "state_features": [
            {
                "name": "object.position",
                "dtype": "float64",
                "unit": "m",
                "shape": [3],
                "frame": "world",
                "description": "Ball center position (object_states table).",
            },
            {
                "name": "object.linear_velocity",
                "dtype": "float64",
                "unit": "m/s",
                "shape": [3],
                "frame": "world",
                "description": "Ball linear velocity (object_states table).",
            },
        ],
        "action_features": [],
        "camera_names": camera_names,
        "camera_roles": {
            name: (
                "main_three_quarter_external"
                if name.endswith(".main")
                else "task_specific_secondary"
            )
            for name in camera_names
        },
        "frame_semantic_fields": ["task_phase", "motion_mode", "active_surface", "contact_role"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": "ball_rolling_dynamics_scripts/v1",
        "extras": {
            "family": args.family,
            "family_spec": {
                "base_seed": spec["base_seed"],
                "layout_id": spec["layout_id"],
                "style_id": spec["style_id"],
                "camera_variant": spec["camera_variant"],
                "demo_mode": "ball_rolling_dynamics",
            },
            "episode_count": len(records),
            "randomized_fields": ["ball_initial_velocity"],
            "rolling_speed_range_mps": list(ROLLING_SPEED_RANGE_MPS),
            "rolling_heading_jitter_rad": ROLLING_HEADING_JITTER_RAD,
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    content_hashes = {name: sha256_file(meta / name) for name in FINALIZED_METADATA_ARTIFACTS}
    completion = {
        "config_hash": records[0].get("config_hash", ""),
        "content_hashes": content_hashes,
        "seal_sha256": None,
    }
    (meta / ".complete.json").write_text(json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(meta / "episodes.parquet")


def episode_backend(args: argparse.Namespace) -> None:
    root = _family_root(args.dataset_root, args.family)
    marker = root / ".records" / f"{episode_uuid(args.family, args.episode_index)}.json"
    record = json.loads(marker.read_text(encoding="utf-8"))["episode"]
    print(record["extras"]["scene_resolution_backend"])


def main() -> None:
    args = _parse_args()
    if args.command == "episode":
        generate_episode(args)
        return
    if args.command == "finalize":
        finalize_family(args)
        return
    if args.command == "episode-backend":
        episode_backend(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
