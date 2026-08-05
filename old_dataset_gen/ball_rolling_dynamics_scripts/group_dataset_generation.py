"""Counterfactual grouped generation for the passive ball-rolling families.

Port of the v3 state-groups mechanism from
``projectile_ball_catch_export/.../group_dataset_generation.py`` (see
``COUNTERFACTUAL_PROJECTILE_CATCH_PLAN.md``), adapted to this pipeline:

* A **group** is M sibling episodes of one family that share everything —
  scene, appearance, camera, ball properties, start pose, physics — and
  differ ONLY in the ball's initial velocity (speed + heading), which is the
  pipeline's entire per-episode variation. The scene is already deterministic
  per family here, so the shared context needs no per-group draws; the group
  is the pairing + split-atomicity unit that makes same-group wrong-state
  negatives and leak-free splits possible.
* **Sibling speeds are stratified** within a group: the effective speed range
  [floor, group cap] is cut into M strata (shrunk by the configured
  separation margin) and assigned to siblings through a group-seeded
  permutation, so pairwise sibling speed separation is guaranteed at
  generation time. The pbc campaign found degenerate near-duplicate siblings
  only at validation time and had to quarantine whole groups; stratification
  removes that failure mode by construction. Headings stay independent
  per-sibling draws. Pooled over a group the speed marginal remains uniform
  on [floor, group cap].
* Episodes are written in the canonical dynamic-robot v2 rollout layout
  (sharded parquet + mp4 + ``.records`` markers) exactly like flat
  generation; groups add ``<family>/groups/<group_id>/group.json`` and stamp
  ``counterfactual_bundle_id`` / ``split_group_id`` / ``split`` into each
  episode record. ``dataset_generation finalize`` then emits a populated
  ``counterfactual_families.parquet``.

Wrong-state negatives for training: for sibling i, condition on the
``state_context`` of another sibling j of the same group. There are no
commands or actions in this passive corpus, so the state context IS the full
conditioning bundle — the swap is internally consistent by construction and
detectable only through the observed dynamics.

RNG discipline (all blake2s-keyed, order-independent):
  group split   <- (dataset) blake2s(group_id) mod 100
  sibling IC    <- (family, group_index, "ic", i)
  strata perm   <- (family, group_index, "strata")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from .canonical_writer import VideoSpec, episode_uuid, sha256_json, write_episode_marker
from .dataset_generation import FAMILY_SPECS
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
    rolling_direction_for_offset,
    rolling_speed_cap_mps,
    sample_episode,
)

SCHEMA = "v1_velocity_groups"
PHYSICS_REGIME = "nominal_constant"
RELATION = "velocity_counterfactual_siblings"

DEFAULT_CONFIG = {
    "phase": "unnamed",
    "schema": SCHEMA,
    "siblings_per_group": 4,
    "groups_per_family": 250,
    "velocity": {
        "speed_range_mps": list(ROLLING_SPEED_RANGE_MPS),
        "heading_jitter_rad": ROLLING_HEADING_JITTER_RAD,
        "stratified_speeds": True,
        "min_speed_separation_mps": 0.04,
    },
    "split_percent": {"train": 90, "val": 5, "test": 5},
}


def _rng_from_key(*parts) -> np.random.Generator:
    key = "/".join(str(part) for part in parts)
    digest = hashlib.blake2s(key.encode("utf-8")).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype=np.uint64))


def _seed_from_key(*parts) -> int:
    key = "/".join(str(part) for part in parts)
    return int.from_bytes(hashlib.blake2s(key.encode("utf-8")).digest()[:4], "big")


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


def episode_index_for(group_index: int, siblings_per_group: int, ic_index: int) -> int:
    return group_index * siblings_per_group + ic_index


def _build_group_context(family: str, *, fps: int, duration: float, width: int, height: int):
    """The deterministic per-family base scene every sibling shares."""

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
    return replace(sample, offscreen_width=width, offscreen_height=height)


def _plan_group_velocities(base_sample, family: str, group_index: int, config: dict) -> tuple[list[dict], dict]:
    """Per-sibling (heading, speed) with guaranteed pairwise speed separation.

    Every sibling draws its own heading; the group speed cap is the tightest
    per-heading runway cap, so every stratified speed is legal for every
    sibling's heading.
    """

    velocity = config["velocity"]
    siblings = int(config["siblings_per_group"])
    jitter = float(velocity["heading_jitter_rad"])
    range_lo, range_hi = (float(v) for v in velocity["speed_range_mps"])

    headings, caps, runways = [], [], []
    for ic_index in range(siblings):
        rng = _rng_from_key(family, group_index, "ic", ic_index)
        heading = float(rng.uniform(-jitter, jitter))
        direction = rolling_direction_for_offset(base_sample, heading)
        cap, runway = rolling_speed_cap_mps(base_sample, direction)
        cap = min(cap, range_hi)
        headings.append(heading)
        caps.append(cap)
        runways.append(runway)

    group_cap = min(caps)
    floor = min(range_lo, group_cap)
    plans = []
    if velocity.get("stratified_speeds", True):
        separation = float(velocity["min_speed_separation_mps"])
        width = (group_cap - floor) / siblings
        if width <= separation:
            raise ValueError(
                f"{group_id_for(family, group_index)}: stratum width {width:.3f} m/s <= "
                f"min separation {separation:.3f} m/s — lower the separation or sibling count"
            )
        strata = [int(v) for v in _rng_from_key(family, group_index, "strata").permutation(siblings)]
        for ic_index in range(siblings):
            rng = _rng_from_key(family, group_index, "speed", ic_index)
            stratum = strata[ic_index]
            lo = floor + stratum * width + separation / 2.0
            hi = floor + (stratum + 1) * width - separation / 2.0
            plans.append(
                {
                    "heading_offset_rad": headings[ic_index],
                    "speed_mps": float(rng.uniform(lo, hi)),
                    "speed_stratum": stratum,
                }
            )
    else:
        strata = None
        for ic_index in range(siblings):
            rng = _rng_from_key(family, group_index, "speed", ic_index)
            plans.append(
                {
                    "heading_offset_rad": headings[ic_index],
                    "speed_mps": float(rng.uniform(floor, caps[ic_index])),
                    "speed_stratum": None,
                }
            )
    for ic_index, plan in enumerate(plans):
        plan["speed_cap_mps"] = caps[ic_index]
        plan["runway_m"] = runways[ic_index]
    return plans, {
        "stratified": bool(velocity.get("stratified_speeds", True)),
        "group_speed_cap_mps": group_cap,
        "group_speed_floor_mps": floor,
        "min_speed_separation_mps": float(velocity["min_speed_separation_mps"]),
        "strata_permutation": strata,
    }


def _build_sibling_sample(base_sample, *, family: str, group_index: int, ic_index: int,
                          plan: dict, split: str, sibling_ids: list[str]):
    group_id = group_id_for(family, group_index)
    episode_id = f"{group_id}_s{ic_index:02d}"
    tags = {
        "episode_id": episode_id,
        "family": family,
        "subfamily": family,
        "branch": "free_roll",
        "group": {
            "group_id": group_id,
            "sibling_index": ic_index,
            "sibling_ids": list(sibling_ids),
            "split": split,
            "physics_regime": PHYSICS_REGIME,
        },
    }
    sample = replace(base_sample, dataset_tags=tags)
    return randomize_initial_velocity(
        sample, None, heading_offset=plan["heading_offset_rad"], speed=plan["speed_mps"]
    )


def _shared_context_record(base_sample, spec: dict) -> dict:
    visual_settings = base_sample.visual_settings or {}
    return {
        "layout_id": spec["layout_id"],
        "style_id": spec["style_id"],
        "camera_variant": spec["camera_variant"],
        "base_seed": spec["base_seed"],
        "demo_mode": "ball_rolling_dynamics",
        "ball_radius_m": float(base_sample.ball_radius),
        "ball_mass_kg": float(base_sample.ball_mass),
        "ball_color": [float(v) for v in base_sample.ball_color],
        "ball_start_position_m": [float(v) for v in base_sample.ball_initial_position],
        "base_direction_xy": list(visual_settings.get("rolling_direction_xy", [])),
        "runway_straight_m": visual_settings.get("rolling_start_distance_m"),
        "tabletop_height_m": float(base_sample.tabletop_height),
        "duration_s": float(base_sample.duration_s),
        "fps": int(base_sample.fps),
        "scene_resolution_backend": visual_settings.get("scene_resolution_backend"),
    }


def _sibling_record(sample, *, ic_index: int, episode_index: int, family: str,
                    plan: dict, record: dict | None) -> dict:
    entry = {
        "episode_id": (sample.dataset_tags or {}).get("episode_id"),
        "sibling_index": ic_index,
        "episode_index": episode_index,
        "episode_uuid": episode_uuid(family, episode_index),
        "state_context": {
            "frame": "world",
            "ball_initial_position_m": [float(v) for v in sample.ball_initial_position],
            "ball_initial_velocity_mps": [float(v) for v in sample.ball_initial_velocity],
            "ball_initial_angular_velocity_rad_s": [
                float(v) for v in sample.ball_initial_angular_velocity
            ],
            "speed_mps": plan["speed_mps"],
            "heading_offset_rad": plan["heading_offset_rad"],
        },
        "speed_cap_mps": plan["speed_cap_mps"],
        "runway_m": plan["runway_m"],
        "speed_stratum": plan["speed_stratum"],
    }
    if record is not None:
        entry["video_paths"] = record["video_paths"]
        entry["frame_data_path"] = record["frame_data_path"]
        entry["roll_metrics"] = {
            key: record["objective_metrics"].get(key)
            for key in (
                "initial_speed_mps",
                "final_speed_mps",
                "travel_distance_m",
                "fell_off_counter",
            )
        }
        entry["scene_resolution_backend"] = record["extras"].get("scene_resolution_backend")
    return entry


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
    _write_json_atomic(
        target, {"schema": SCHEMA, "phase_config_path": str(config_path), "phase_config": config}
    )


def _group_dir(dataset_root: Path, family: str, group_id: str) -> Path:
    return dataset_root / family / "groups" / group_id


def _config_hash_for(family: str, config: dict, args: argparse.Namespace) -> str:
    return sha256_json(
        {
            "schema": SCHEMA,
            "family": family,
            "phase_config": config,
            "ball_radius_m": BALL_RADIUS_M,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "duration": args.duration,
        }
    )


def _build_group(args: argparse.Namespace):
    config = _load_phase_config(args.phase_config)
    if args.group_index < 0 or args.group_index >= int(config["groups_per_family"]):
        raise ValueError(
            f"group index {args.group_index} outside [0, {config['groups_per_family']}) "
            f"for phase {config['phase']!r}"
        )
    family = args.family
    spec = FAMILY_SPECS[family]
    group_id = group_id_for(family, args.group_index)
    split = split_for_group(group_id, config["split_percent"])
    siblings = int(config["siblings_per_group"])
    sibling_ids = [f"{group_id}_s{i:02d}" for i in range(siblings)]
    base_sample = _build_group_context(
        family, fps=args.fps, duration=args.duration, width=args.width, height=args.height
    )
    plans, speed_plan = _plan_group_velocities(base_sample, family, args.group_index, config)
    samples = [
        _build_sibling_sample(
            base_sample,
            family=family,
            group_index=args.group_index,
            ic_index=i,
            plan=plans[i],
            split=split,
            sibling_ids=sibling_ids,
        )
        for i in range(siblings)
    ]
    return config, spec, group_id, split, base_sample, plans, speed_plan, samples


def _group_payload(args, config, spec, group_id, split, base_sample, plans, speed_plan,
                   samples, records) -> dict:
    siblings = []
    for ic_index, sample in enumerate(samples):
        index = episode_index_for(args.group_index, len(samples), ic_index)
        siblings.append(
            _sibling_record(
                sample,
                ic_index=ic_index,
                episode_index=index,
                family=args.family,
                plan=plans[ic_index],
                record=records[ic_index],
            )
        )
    return {
        "group_id": group_id,
        "family": args.family,
        "schema": SCHEMA,
        "phase": config["phase"],
        "split": split,
        "physics_regime": PHYSICS_REGIME,
        "sibling_count": len(samples),
        "speed_plan": speed_plan,
        "shared_context": _shared_context_record(base_sample, spec),
        "siblings": siblings,
    }


def generate_group(args: argparse.Namespace) -> None:
    config, spec, group_id, split, base_sample, plans, speed_plan, samples = _build_group(args)
    _ensure_generation_config(args.dataset_root, config, args.phase_config)
    root = args.dataset_root / args.family
    video_spec = VideoSpec(width=args.width, height=args.height, fps_num=args.fps)
    config_hash = _config_hash_for(args.family, config, args)

    records = []
    for ic_index, sample in enumerate(samples):
        index = episode_index_for(args.group_index, len(samples), ic_index)
        uuid_text = episode_uuid(args.family, index)
        marker = root / ".records" / f"{uuid_text}.json"
        if marker.exists() and not args.overwrite:
            record = json.loads(marker.read_text(encoding="utf-8"))["episode"]
            print(f"{group_id} s{ic_index:02d} episode {index} [skipped: already generated]")
        else:
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
                episode_index=index,
                task_index=0,
                camera_streams=DEFAULT_CAMERA_STREAMS,
                video_spec=video_spec,
                overwrite=True,
            )
            record = build_episode_record(
                rollout,
                artifacts=artifacts,
                episode_uuid_text=uuid_text,
                episode_index=index,
                task_index=0,
                family=args.family,
                subfamily=args.family,
                scene_seed=spec["base_seed"],
                branch_seed=_seed_from_key(args.family, args.group_index, "ic", ic_index),
                config_hash=config_hash,
            )
            record["counterfactual_bundle_id"] = group_id
            record["split_group_id"] = group_id
            record["split"] = split
            record["extras"]["group"] = (sample.dataset_tags or {})["group"]
            write_episode_marker(root, record, config_hash)
            speed = record["objective_metrics"]["initial_speed_mps"]
            print(f"{group_id} s{ic_index:02d} episode {index} speed={speed:.3f} m/s")
        records.append(record)

    group_dir = _group_dir(args.dataset_root, args.family, group_id)
    group_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        group_dir / "group.json",
        _group_payload(args, config, spec, group_id, split, base_sample, plans, speed_plan,
                       samples, records),
    )
    print(group_dir / "group.json")


def plan_group(args: argparse.Namespace) -> None:
    """Dry run: build and print the group's sibling table without simulating or writing."""

    config, spec, group_id, split, base_sample, plans, speed_plan, samples = _build_group(args)
    payload = _group_payload(args, config, spec, group_id, split, base_sample, plans, speed_plan,
                             samples, [None] * len(samples))
    print(json.dumps(payload, indent=2, sort_keys=True))


def finalize_groups(args: argparse.Namespace) -> None:
    """Check group completeness, then run the canonical finalize + dataset splits.json."""

    from .dataset_generation import finalize_family

    family_dir = args.dataset_root / args.family
    group_jsons = sorted(family_dir.glob("groups/*/group.json"))
    if not group_jsons:
        raise FileNotFoundError(f"No group.json found under {family_dir / 'groups'}")
    missing = []
    for group_json in group_jsons:
        group = json.loads(group_json.read_text(encoding="utf-8"))
        for sibling in group["siblings"]:
            marker = family_dir / ".records" / f"{sibling['episode_uuid']}.json"
            if not marker.exists():
                missing.append(f"{group['group_id']}: missing marker for {sibling['episode_id']}")
    if missing:
        raise SystemExit("\n".join(missing[:20]) + f"\n{len(missing)} sibling episode(s) missing")

    finalize_family(
        argparse.Namespace(
            family=args.family, dataset_root=args.dataset_root, width=args.width, height=args.height
        )
    )

    splits: dict[str, str] = {}
    for group_json in sorted(args.dataset_root.glob("*/groups/*/group.json")):
        group = json.loads(group_json.read_text(encoding="utf-8"))
        splits[group["group_id"]] = group["split"]
    _write_json_atomic(args.dataset_root / "splits.json", splits)
    print(f"{len(group_jsons)} groups complete")
    print(args.dataset_root / "splits.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Velocity-counterfactual grouped generation: shared deterministic scene per family, "
            "per-sibling initial velocity, constant nominal physics."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_group_args(sub, with_render_args: bool):
        sub.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
        sub.add_argument("--group-index", type=int, required=True)
        sub.add_argument("--phase-config", type=Path, required=True)
        sub.add_argument("--fps", type=int, default=30)
        sub.add_argument("--duration", type=float, default=2.5)
        sub.add_argument("--width", type=int, default=832)
        sub.add_argument("--height", type=int, default=480)
        if with_render_args:
            sub.add_argument("--dataset-root", type=Path, required=True)
            sub.add_argument("--overwrite", action="store_true")

    group_parser = subparsers.add_parser(
        "group", help="Generate all siblings of one group as canonical episodes and write group.json."
    )
    add_group_args(group_parser, with_render_args=True)

    plan_parser = subparsers.add_parser(
        "plan", help="Dry run: print the group's sibling table without simulating or writing."
    )
    add_group_args(plan_parser, with_render_args=False)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Verify group completeness, build meta/ tables, refresh dataset splits.json.",
    )
    finalize_parser.add_argument("--family", required=True, choices=sorted(FAMILY_SPECS))
    finalize_parser.add_argument("--dataset-root", type=Path, required=True)
    finalize_parser.add_argument("--width", type=int, default=832)
    finalize_parser.add_argument("--height", type=int, default=480)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "group":
        generate_group(args)
    elif args.command == "plan":
        plan_group(args)
    elif args.command == "finalize":
        finalize_groups(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
