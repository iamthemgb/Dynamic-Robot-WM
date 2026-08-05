from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset_generation import BALL_COLORS


GRAVITY_Z = -9.81
CLOSE_Z_OFFSET = 0.055  # controller closes when the ball descends to catch_center_z + this


def _load_episodes(dataset_root: Path, families: list[str] | None):
    episodes = []
    family_dirs = sorted(p for p in dataset_root.iterdir() if (p / "groups").is_dir()) if dataset_root.is_dir() else []
    for family_dir in family_dirs:
        if families and family_dir.name not in families:
            continue
        for metadata_path in sorted(family_dir.glob("groups/*/s*/metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            group = metadata.get("group") or {}
            episodes.append(
                {
                    "path": metadata_path,
                    "family": family_dir.name,
                    "group_id": group.get("group_id", metadata_path.parent.parent.name),
                    "split": group.get("split"),
                    "metadata": metadata,
                }
            )
    return episodes


def _by_group(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["group_id"]].append(episode)
    return groups


def _state(metadata: dict):
    state = metadata["state_context"]
    p0 = np.asarray(state["ball_initial_position_m"], dtype=np.float64)
    v0 = np.asarray(state["ball_initial_velocity_mps"], dtype=np.float64)
    return p0, v0, float(state["release_time_s"])


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True)


SHARED_KEYS = ("cameras", "ball_color", "robot_base_position", "tabletop_height")
SHARED_VISUAL_KEYS = (
    "kitchen_layout_id",
    "kitchen_style_id",
    "camera_variant",
    "dataset_texture_variant",
    "dataset_background_variant",
    "dataset_asset_color_variant",
)


def gate_shared_context(groups) -> tuple[bool, str]:
    bad = []
    for group_id, episodes in groups.items():
        reference = episodes[0]["metadata"]
        for episode in episodes[1:]:
            metadata = episode["metadata"]
            for key in SHARED_KEYS:
                if _canonical(metadata.get(key)) != _canonical(reference.get(key)):
                    bad.append(f"{group_id}: {key} differs at {episode['path']}")
            for key in SHARED_VISUAL_KEYS:
                if _canonical((metadata.get("visual_settings") or {}).get(key)) != _canonical(
                    (reference.get("visual_settings") or {}).get(key)
                ):
                    bad.append(f"{group_id}: visual_settings.{key} differs at {episode['path']}")
            for key in ("ball_radius_m",):
                if metadata["geometry_context"][key] != reference["geometry_context"][key]:
                    bad.append(f"{group_id}: geometry_context.{key} differs at {episode['path']}")
            if metadata["physics_tokens"]["ball_mass_kg"] != reference["physics_tokens"]["ball_mass_kg"]:
                bad.append(f"{group_id}: ball_mass_kg differs at {episode['path']}")
    return not bad, "\n".join(bad[:10]) or f"{len(groups)} groups share context exactly"


def gate_ic_diversity(groups, floor: float = 1e-3) -> tuple[bool, str]:
    bad = []
    worst = math.inf
    for group_id, episodes in groups.items():
        if len(episodes) < 2:
            bad.append(f"{group_id}: only {len(episodes)} sibling(s)")
            continue
        vectors = []
        for episode in episodes:
            p0, v0, release = _state(episode["metadata"])
            vectors.append(np.r_[p0, v0, release])
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                distance = float(np.linalg.norm(vectors[i] - vectors[j]))
                worst = min(worst, distance)
                if distance <= floor:
                    bad.append(f"{group_id}: siblings {i},{j} nearly identical ICs (d={distance:.2e})")
    return not bad, "\n".join(bad[:10]) or f"min pairwise IC distance = {worst:.4f}"


def gate_physics_constancy(episodes) -> tuple[bool, str]:
    def stripped(metadata):
        tokens = json.loads(_canonical(metadata["physics_tokens"]))
        tokens.pop("ball_mass_kg", None)  # group-level property, varies across groups by design
        tokens.pop("ball_mass_note", None)
        return _canonical(tokens)

    reference = stripped(episodes[0]["metadata"])
    bad = [str(e["path"]) for e in episodes if stripped(e["metadata"]) != reference]
    gravity = episodes[0]["metadata"]["physics_tokens"]["gravity_mps2"]
    if gravity != [0.0, 0.0, GRAVITY_Z]:
        bad.append(f"gravity is {gravity}, expected [0, 0, {GRAVITY_Z}]")
    return not bad, "\n".join(bad[:10]) or f"physics tokens constant across {len(episodes)} episodes"


def gate_divergence(groups, at_time: float, floor_m: float) -> tuple[bool, str]:
    # Under identical gravity, sibling separation at time tau after release is |dp0 + dv0*tau|.
    distances = []
    for episodes in groups.values():
        states = [_state(e["metadata"]) for e in episodes]
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                dp = states[i][0] - states[j][0]
                dv = states[i][1] - states[j][1]
                distances.append(float(np.linalg.norm(dp + dv * at_time)))
    median = float(np.median(distances)) if distances else 0.0
    return median > floor_m, f"median sibling divergence at release+{at_time:.2f}s = {median:.3f} m (floor {floor_m})"


def _descend_time_to_z(p0, v0, target_z: float) -> float | None:
    a = 0.5 * GRAVITY_Z
    b = float(v0[2])
    c = float(p0[2]) - target_z
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_d = math.sqrt(discriminant)
    roots = sorted(root for root in ((-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)) if root >= -1e-9)
    descending = [root for root in roots if b + GRAVITY_Z * root <= 0.0]
    if descending:
        return max(0.0, descending[0])
    return max(0.0, roots[0]) if roots else None


def gate_state_command_consistency(episodes, tolerance: float = 1e-6) -> tuple[bool, str]:
    bad, checked = [], 0
    for episode in episodes:
        metadata = episode["metadata"]
        if metadata.get("branch") not in {"success", "wrong_action"}:
            continue  # near-miss/contact branches add an unrecorded lateral target offset
        intercept = metadata["outcomes"]["intercept_position_m"]
        if intercept is None:
            bad.append(f"{episode['path']}: no intercept recorded")
            continue
        p0, v0, _ = _state(metadata)
        t_star = _descend_time_to_z(p0, v0, float(intercept[2]) + CLOSE_Z_OFFSET)
        if t_star is None:
            bad.append(f"{episode['path']}: no ballistic solution to close height")
            continue
        predicted_xy = p0[:2] + v0[:2] * t_star
        error = float(np.linalg.norm(predicted_xy - np.asarray(intercept[:2], dtype=np.float64)))
        checked += 1
        if error > tolerance:
            bad.append(f"{episode['path']}: intercept mismatch {error:.2e} m")
    return not bad, "\n".join(bad[:10]) or f"{checked} zero-offset episodes match ballistic plan"


def _camera_axes(pos, lookat):
    forward = np.asarray(lookat, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return forward, right, cam_up


def _visible_fraction(metadata: dict, camera: dict, width: int, height: int) -> float:
    p0, v0, release = _state(metadata)
    outcomes = metadata["outcomes"]
    t_end = outcomes.get("first_contact_time_s") or outcomes.get("intercept_time_s") or metadata["duration_sec"]
    t_end = min(float(t_end), float(metadata["duration_sec"]))
    if t_end <= release:
        return 1.0
    fps = float(metadata["fps"])
    forward, right, cam_up = _camera_axes(camera["pos"], camera["lookat"])
    focal = 0.5 * height / math.tan(math.radians(float(camera["fovy"])) / 2.0)
    times = np.arange(math.ceil(release * fps), math.floor(t_end * fps) + 1) / fps
    if times.size == 0:
        return 1.0
    visible = 0
    for t in times:
        tau = t - release
        point = p0 + v0 * tau + 0.5 * np.array([0.0, 0.0, GRAVITY_Z]) * tau * tau
        delta = point - np.asarray(camera["pos"], dtype=np.float64)
        depth = float(np.dot(forward, delta))
        if depth <= 1e-6:
            continue
        u = 0.5 * width + focal * float(np.dot(right, delta)) / depth
        v = 0.5 * height - focal * float(np.dot(cam_up, delta)) / depth
        if 0.0 <= u <= width and 0.0 <= v <= height:
            visible += 1
    return visible / times.size


def gate_in_frame(episodes, width: int, height: int, per_episode_floor: float = 0.8, pass_rate_floor: float = 0.95) -> tuple[bool, str]:
    passing, total, worst = 0, 0, 1.0
    for episode in episodes:
        metadata = episode["metadata"]
        fractions = []
        for view in ("main_camera", "side_camera"):
            camera = (metadata.get("cameras") or {}).get(view)
            if camera is None or "lookat" not in camera:
                continue
            fractions.append(_visible_fraction(metadata, camera, width, height))
        if not fractions:
            continue
        total += 1
        worst = min(worst, min(fractions))
        if min(fractions) >= per_episode_floor:
            passing += 1
    rate = passing / total if total else 1.0
    return rate >= pass_rate_floor, f"{passing}/{total} episodes keep ball ≥{per_episode_floor:.0%} in frame (worst {worst:.2f})"


def gate_split_integrity(groups, split_percent: dict) -> tuple[bool, str]:
    bad = []
    counts = defaultdict(int)
    train_edge = int(split_percent["train"])
    val_edge = train_edge + int(split_percent["val"])
    for group_id, episodes in groups.items():
        splits = {e["split"] for e in episodes}
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


def gate_confound_scan(episodes, warn_r: float = 0.1) -> tuple[bool, str]:
    state_rows, appearance_rows = [], []
    for episode in episodes:
        metadata = episode["metadata"]
        p0, v0, release = _state(metadata)
        state_rows.append([p0[0], p0[1], p0[2], float(np.linalg.norm(v0)), release])
        visual = metadata.get("visual_settings") or {}
        color = [round(float(v), 4) for v in metadata["ball_color"]]
        color_index = next((i for i, c in enumerate(BALL_COLORS) if [round(v, 4) for v in c] == color), -1)
        appearance_rows.append(
            [
                visual.get("dataset_texture_variant", -1),
                visual.get("dataset_background_variant", -1),
                visual.get("dataset_asset_color_variant", -1),
                color_index,
            ]
        )
    state = np.asarray(state_rows, dtype=np.float64)
    appearance = np.asarray(appearance_rows, dtype=np.float64)
    worst = 0.0
    for i in range(state.shape[1]):
        for j in range(appearance.shape[1]):
            if np.std(state[:, i]) < 1e-12 or np.std(appearance[:, j]) < 1e-12:
                continue
            r = float(np.corrcoef(state[:, i], appearance[:, j])[0, 1])
            worst = max(worst, abs(r))
    return worst <= warn_r, f"max |pearson r| between state and appearance = {worst:.3f} (warn > {warn_r})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation gates for the v3 state-conditional grouped dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--family", action="append", help="Restrict to one or more families (default: all found).")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--divergence-time", type=float, default=0.4)
    parser.add_argument("--divergence-floor", type=float, default=0.15)
    args = parser.parse_args()

    episodes = _load_episodes(args.dataset_root, args.family)
    if not episodes:
        raise SystemExit(f"No episode metadata found under {args.dataset_root}")
    groups = _by_group(episodes)

    config_path = args.dataset_root / "generation_config.json"
    split_percent = {"train": 90, "val": 5, "test": 5}
    if config_path.exists():
        split_percent = json.loads(config_path.read_text(encoding="utf-8"))["phase_config"]["split_percent"]

    gates = [
        ("shared_context_identity", True, gate_shared_context(groups)),
        ("ic_diversity", True, gate_ic_diversity(groups)),
        ("physics_constancy", True, gate_physics_constancy(episodes)),
        ("state_trajectory_divergence", False, gate_divergence(groups, args.divergence_time, args.divergence_floor)),
        ("state_command_consistency", True, gate_state_command_consistency(episodes)),
        ("in_frame", False, gate_in_frame(episodes, args.width, args.height)),
        ("split_integrity", True, gate_split_integrity(groups, split_percent)),
        ("confound_scan", False, gate_confound_scan(episodes)),
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
