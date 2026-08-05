"""Reading the projectile-catch state-groups corpus (schema ``v3_state_groups``).

Layout, per family (``pbc_state_groups_p*/<family>/``)::

    groups/<family>_g00NNNN/
        group.json                    shared context + sibling table + split
        s0M/{main.mp4, side.mp4, metadata.json}

Every group holds 4 state siblings sharing scene/appearance/camera/ball;
splits are group-atomic and stamped into both ``group.json`` and each
episode's ``metadata.json`` ``group`` block. Unlike f1_10h there is no
eligibility override: the dataset is gated by
``validate_group_dataset.py`` before a cache is ever built.

Frame convention: episodes render 76 frames (2.5 s @ 30 fps + 1). The Wan
VAE chunks causally as ``1 + (T-1)//4`` and silently drops the tail, so we
read 73 frames -> Tz = 19 exact.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_FRAMES = 73
N_CONTROL = N_FRAMES - 1                       # 72
FPS = 30.0
FAMILIES = ("style020_seed0", "style055_seed1000")


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    family: str
    group_key: str                 # "style020_seed0_g000123" (string form)
    ic_index: int
    branch: str
    split: str
    root: Path                     # .../groups/<group_key>/s0M

    @property
    def video_main(self) -> Path:
        return self.root / "main.mp4"

    @property
    def video_side(self) -> Path:
        return self.root / "side.mp4"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def metadata(self) -> dict:
        return json.loads(self.metadata_path.read_text())


def build_index(dataset_root, families=None):
    """One row per sibling episode across all families.

    Returns a pandas DataFrame with dense ``group_id`` (0..G-1 over the whole
    index), the split from group.json, and the taxonomy columns
    ``prompts.bucket_key`` expects. All rows share one appearance bucket by
    design: the text channel must carry no discriminative signal.
    """
    import pandas as pd

    dataset_root = Path(dataset_root)
    rows = []
    for family in (families or FAMILIES):
        for gjson in sorted((dataset_root / family / "groups").glob(
                "*/group.json")):
            g = json.loads(gjson.read_text())
            if len(g["siblings"]) != g["sibling_count"]:
                raise RuntimeError(f"{gjson}: sibling table incomplete")
            for sib in g["siblings"]:
                root = gjson.parent / sib["paths"]["dir"]
                rows.append({
                    "episode_id": sib["episode_id"],
                    "family": g["family"],
                    "group_key": g["group_id"],
                    "ic_index": int(sib["ic_index"]),
                    "branch": sib["branch"],
                    "split": g["split"],
                    "catch_success": bool(sib["catch_success"]),
                    "root": str(root),
                    # taxonomy for prompts.bucket_key: one bucket for all
                    "leaf": g["family"],
                    "subfamily": "mild_projectile",
                    "variant": "mild_projectile_catch",
                    "tool_type": "franka_hand",
                    "background_style": "robocasa_kitchen",
                })
    if not rows:
        raise FileNotFoundError(f"no group.json under {dataset_root}")
    idx = pd.DataFrame(rows)
    relabel = {g: i for i, g in enumerate(sorted(idx["group_key"].unique()))}
    idx["group_id"] = [relabel[g] for g in idx["group_key"]]
    return idx.sort_values("episode_id").reset_index(drop=True)


def refs_from_index(idx):
    return [EpisodeRef(r.episode_id, r.family, r.group_key, int(r.ic_index),
                       r.branch, r.split, Path(r.root))
            for r in idx.itertuples()]


def read_frames(video_path, n_frames: int = N_FRAMES):
    """-> float32 [3, T, 480, 832] in [-1, 1], the Wan VAE input domain.

    832x480 is rendered natively; no resize. Latents must NOT be
    re-normalised afterwards (WanVAE.encode whitens internally).
    """
    import av
    import torch

    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= n_frames:
            break
    container.close()
    if len(frames) < n_frames:
        raise RuntimeError(f"{video_path}: {len(frames)} < {n_frames}")
    x = torch.from_numpy(np.stack(frames))          # [T, H, W, 3] uint8
    return x.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)


def read_actions(metadata: dict, n_control: int = N_CONTROL):
    """-> float32 [n_control, 9] joint positions (7 arm + 2 finger).

    The v3 pipeline records the planned joint trajectory at frame rate in
    ``action_context.joint_trajectory``; control step t drives frame t -> t+1,
    matching ``cache_wan_latents.control_bin_index``. NOTE: the real Wan DiT
    consumes no dense action stream (``make_real_fm_loss`` ignores it); this
    array exists to satisfy the cache contract and to serve phases 0/2.
    """
    q = np.asarray(
        metadata["action_context"]["joint_trajectory"]["joint_positions_rad"],
        dtype=np.float32)
    if q.shape[0] < n_control:
        raise RuntimeError(f"joint trajectory too short: {q.shape}")
    return q[:n_control]


def impact_frame(metadata: dict, n_frames: int = N_FRAMES):
    """First POST-RELEASE ball contact as a FRAME index (GroupBatcher divides
    by the temporal stride itself), or None.

    ``first_contact_frame`` at or before ``release_frame`` is the ball resting
    on its spawn surface, not a flight impact -- treated as no event. Only
    phases 0/2 consume this; phase 1 never reads events.
    """
    ev = metadata.get("events") or {}
    v = ev.get("first_contact_frame")
    if v is None:
        return None
    release = ev.get("release_frame")
    if release is not None and int(v) <= int(release):
        return None
    return int(min(max(int(v), 0), n_frames - 1))
