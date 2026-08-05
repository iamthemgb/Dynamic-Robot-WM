"""Validation gates for the velocity-counterfactual grouped rolling dataset.

Port of ``projectile_ball_catch_export/.../validate_group_dataset.py`` adapted
to this pipeline: episodes live as canonical ``.records`` markers rather than
per-episode metadata.json, siblings vary only in initial velocity, and there
is no controller, so the state-command consistency gate has no analogue. The
in-frame gate is also dropped deliberately: the zoomed main view lets
fast episodes exit the left frame edge by design (recorded trade-off), so an
in-frame floor would flag correct output. New rolling-specific gates instead:
``on_counter`` (the ball must never leave the island) and ``speed_cap``
(recorded speeds respect the per-heading runway cap).

Hard gates exit 1; soft gates warn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset_generation import FAMILY_SPECS


def _load_grouped_episodes(dataset_root: Path, families: list[str] | None):
    episodes, groups_meta = [], {}
    family_dirs = sorted(
        p for p in dataset_root.iterdir() if p.is_dir() and (p / "groups").is_dir()
    ) if dataset_root.is_dir() else []
    for family_dir in family_dirs:
        if families and family_dir.name not in families:
            continue
        for group_json in sorted(family_dir.glob("groups/*/group.json")):
            group = json.loads(group_json.read_text(encoding="utf-8"))
            groups_meta[group["group_id"]] = {"path": group_json, "group": group}
        for marker in sorted((family_dir / ".records").glob("*.json")):
            record = json.loads(marker.read_text(encoding="utf-8"))["episode"]
            group_info = (record.get("extras") or {}).get("group")
            if not group_info:
                continue  # flat episode in the same root; not part of this dataset
            episodes.append(
                {
                    "path": marker,
                    "family": family_dir.name,
                    "group_id": group_info["group_id"],
                    "split": record.get("split"),
                    "record": record,
                }
            )
    return episodes, groups_meta


def _by_group(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["group_id"]].append(episode)
    return groups


def _velocity(record: dict) -> np.ndarray:
    return np.asarray(record["physics"]["ball_initial_velocity_m_s"], dtype=np.float64)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True)


def gate_group_completeness(groups, groups_meta) -> tuple[bool, str]:
    bad = []
    for group_id, meta in groups_meta.items():
        expected = {s["episode_uuid"] for s in meta["group"]["siblings"]}
        found = {e["record"]["episode_uuid"] for e in groups.get(group_id, [])}
        if found != expected:
            bad.append(
                f"{group_id}: markers {len(found)}/{len(expected)}"
                + (f", unexpected {sorted(found - expected)[:2]}" if found - expected else "")
            )
        if int(meta["group"]["sibling_count"]) != len(expected):
            bad.append(f"{group_id}: sibling_count != sibling list length")
    for group_id in groups:
        if group_id not in groups_meta:
            bad.append(f"{group_id}: markers exist but group.json is missing")
    return not bad, "\n".join(bad[:10]) or f"{len(groups_meta)} groups complete"


SHARED_PHYSICS_KEYS = ("ball_radius_m", "ball_mass_kg")


def gate_shared_context(groups) -> tuple[bool, str]:
    bad = []
    for group_id, episodes in groups.items():
        reference = episodes[0]["record"]
        for episode in episodes[1:]:
            record = episode["record"]
            if record["config_hash"] != reference["config_hash"]:
                bad.append(f"{group_id}: config_hash differs at {episode['path']}")
            for key in SHARED_PHYSICS_KEYS:
                if record["physics"][key] != reference["physics"][key]:
                    bad.append(f"{group_id}: physics.{key} differs at {episode['path']}")
            if record["physics"]["ball_initial_position_m"] != reference["physics"]["ball_initial_position_m"]:
                bad.append(f"{group_id}: start position differs at {episode['path']}")
            if _canonical(record["extras"].get("camera_poses")) != _canonical(
                reference["extras"].get("camera_poses")
            ):
                bad.append(f"{group_id}: camera poses differ at {episode['path']}")
            if record["asset_ids"] != reference["asset_ids"]:
                bad.append(f"{group_id}: asset_ids differ at {episode['path']}")
        for episode in episodes:
            backend = episode["record"]["extras"].get("scene_resolution_backend")
            if backend != "robocasa_native":
                bad.append(f"{group_id}: backend {backend} at {episode['path']}")
    return not bad, "\n".join(bad[:10]) or f"{len(groups)} groups share context exactly"


def gate_ic_diversity(groups, min_separation: float | None, floor: float = 1e-3) -> tuple[bool, str]:
    bad = []
    worst_v, worst_s = math.inf, math.inf
    for group_id, episodes in groups.items():
        if len(episodes) < 2:
            bad.append(f"{group_id}: only {len(episodes)} sibling(s)")
            continue
        velocities = [_velocity(e["record"]) for e in episodes]
        speeds = [float(np.linalg.norm(v[:2])) for v in velocities]
        for i in range(len(velocities)):
            for j in range(i + 1, len(velocities)):
                dv = float(np.linalg.norm(velocities[i] - velocities[j]))
                ds = abs(speeds[i] - speeds[j])
                worst_v, worst_s = min(worst_v, dv), min(worst_s, ds)
                if dv <= floor:
                    bad.append(f"{group_id}: siblings {i},{j} nearly identical velocity (d={dv:.2e})")
                if min_separation is not None and ds < min_separation - 1e-9:
                    bad.append(
                        f"{group_id}: siblings {i},{j} speed separation {ds:.3f} < {min_separation}"
                    )
    detail = f"min pairwise |dv| = {worst_v:.4f} m/s, min pairwise |dspeed| = {worst_s:.4f} m/s"
    return not bad, "\n".join(bad[:10]) or detail


VARIED_PHYSICS_KEYS = {"ball_initial_velocity_m_s", "ball_initial_angular_velocity_rad_s"}


def gate_physics_constancy(episodes) -> tuple[bool, str]:
    def stripped(record):
        physics = json.loads(_canonical(record["physics"]))
        for key in VARIED_PHYSICS_KEYS:
            physics.pop(key, None)
        return _canonical(physics)

    reference = stripped(episodes[0]["record"])
    bad = [str(e["path"]) for e in episodes if stripped(e["record"]) != reference]
    gravity = episodes[0]["record"]["physics"]["gravity_m_s2"]
    if gravity != [0.0, 0.0, -9.81]:
        bad.append(f"gravity is {gravity}, expected [0, 0, -9.81]")
    return not bad, "\n".join(bad[:10]) or f"physics constant across {len(episodes)} episodes"


def gate_on_counter(episodes) -> tuple[bool, str]:
    bad = [
        str(e["path"])
        for e in episodes
        if e["record"]["objective_metrics"].get("fell_off_counter")
    ]
    return not bad, "\n".join(bad[:10]) or f"ball stayed on the island in all {len(episodes)} episodes"


def gate_speed_cap(episodes, tolerance: float = 1.02) -> tuple[bool, str]:
    bad = []
    for episode in episodes:
        randomization = episode["record"]["randomization"]
        speed = randomization.get("rolling_speed_mps")
        cap = randomization.get("rolling_speed_cap_mps")
        if speed is None or cap is None:
            bad.append(f"{episode['path']}: missing recorded speed/cap")
        elif speed > cap * (1.0 + 1e-9):
            bad.append(f"{episode['path']}: speed {speed:.3f} > cap {cap:.3f}")
        runway = randomization.get("rolling_runway_m")
        travel = episode["record"]["objective_metrics"].get("travel_distance_m")
        if runway is not None and travel is not None and travel > 0.92 * runway * tolerance:
            bad.append(f"{episode['path']}: travel {travel:.3f} exceeds 0.92x runway {runway:.3f}")
    return not bad, "\n".join(bad[:10]) or f"speeds within per-heading caps for {len(episodes)} episodes"


def gate_final_divergence(groups, floor_m: float) -> tuple[bool, str]:
    """Measured sibling separation at the last frame (rolling decelerates, so
    use simulated final positions rather than a closed-form prediction)."""

    distances = []
    for episodes in groups.values():
        finals = [
            np.asarray(e["record"]["objective_metrics"]["final_position_m"][:2], dtype=np.float64)
            for e in episodes
        ]
        for i in range(len(finals)):
            for j in range(i + 1, len(finals)):
                distances.append(float(np.linalg.norm(finals[i] - finals[j])))
    if not distances:
        return False, "no sibling pairs found"
    median = float(np.median(distances))
    minimum = float(np.min(distances))
    return median > floor_m, (
        f"median sibling final-position separation = {median:.3f} m "
        f"(min {minimum:.3f}, floor {floor_m})"
    )


def gate_frame_video_integrity(episodes, dataset_root: Path) -> tuple[bool, str]:
    bad = []
    for episode in episodes:
        record = episode["record"]
        expected_frames = round(float(record["duration_s"]) * 30.0)
        if int(record["frame_count"]) != expected_frames:
            bad.append(f"{episode['path']}: frame_count {record['frame_count']} != {expected_frames}")
        family_root = dataset_root / episode["family"]
        for stream, rel in sorted(record["video_paths"].items()):
            if not (family_root / rel).exists():
                bad.append(f"{episode['path']}: missing video {rel}")
        if not (family_root / record["frame_data_path"]).exists():
            bad.append(f"{episode['path']}: missing frame data {record['frame_data_path']}")
    return not bad, "\n".join(bad[:10]) or f"frames + artifacts intact for {len(episodes)} episodes"


def gate_split_integrity(groups, groups_meta, split_percent: dict) -> tuple[bool, str]:
    bad = []
    counts = defaultdict(int)
    train_edge = int(split_percent["train"])
    val_edge = train_edge + int(split_percent["val"])
    for group_id, episodes in groups.items():
        splits = {e["split"] for e in episodes}
        meta = groups_meta.get(group_id)
        if meta is not None:
            splits.add(meta["group"]["split"])
        if len(splits) != 1 or None in splits:
            bad.append(f"{group_id}: inconsistent splits {splits}")
            continue
        split = splits.pop()
        counts[split] += 1
        bucket = int.from_bytes(hashlib.blake2s(group_id.encode("utf-8")).digest()[:8], "big") % 100
        expected = "train" if bucket < train_edge else ("val" if bucket < val_edge else "test")
        if split != expected:
            bad.append(f"{group_id}: split {split} != recomputed {expected}")
    return not bad, "\n".join(bad[:10]) or f"group counts by split: {dict(counts)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation gates for the velocity-counterfactual grouped rolling dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--family", action="append", choices=sorted(FAMILY_SPECS),
                        help="Restrict to one or more families (default: all found).")
    parser.add_argument("--divergence-floor", type=float, default=0.05,
                        help="Median sibling final-position separation floor in metres.")
    args = parser.parse_args()

    episodes, groups_meta = _load_grouped_episodes(args.dataset_root, args.family)
    if not episodes:
        raise SystemExit(f"No grouped episode markers found under {args.dataset_root}")
    groups = _by_group(episodes)

    config_path = args.dataset_root / "generation_config.json"
    split_percent = {"train": 90, "val": 5, "test": 5}
    min_separation = None
    if config_path.exists():
        phase_config = json.loads(config_path.read_text(encoding="utf-8"))["phase_config"]
        split_percent = phase_config["split_percent"]
        velocity = phase_config.get("velocity", {})
        if velocity.get("stratified_speeds", True):
            min_separation = float(velocity.get("min_speed_separation_mps", 0.0)) or None

    gates = [
        ("group_completeness", True, gate_group_completeness(groups, groups_meta)),
        ("shared_context_identity", True, gate_shared_context(groups)),
        ("ic_diversity", True, gate_ic_diversity(groups, min_separation)),
        ("physics_constancy", True, gate_physics_constancy(episodes)),
        ("on_counter", True, gate_on_counter(episodes)),
        ("speed_cap", True, gate_speed_cap(episodes)),
        ("final_divergence", False, gate_final_divergence(groups, args.divergence_floor)),
        ("frame_video_integrity", True, gate_frame_video_integrity(episodes, args.dataset_root)),
        ("split_integrity", True, gate_split_integrity(groups, groups_meta, split_percent)),
    ]

    print(f"dataset: {args.dataset_root}  episodes: {len(episodes)}  groups: {len(groups)}")
    hard_failures = 0
    for name, hard, (passed, detail) in gates:
        label = "PASS" if passed else ("FAIL" if hard else "WARN")
        if not passed and hard:
            hard_failures += 1
        print(f"[{label}] {name} ({'hard' if hard else 'soft'})")
        for line in str(detail).splitlines():
            print(f"    {line}")
    if hard_failures:
        raise SystemExit(f"{hard_failures} hard gate(s) failed")
    print("all hard gates passed")


if __name__ == "__main__":
    main()
