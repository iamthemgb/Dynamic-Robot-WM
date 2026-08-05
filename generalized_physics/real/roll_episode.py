"""Reading the ball-rolling velocity-groups corpus (schema ``v1_velocity_groups``).

Layout, per family (``roll_groups_p*/<family>/``) — canonical dynamic-robot
v2 rollouts, sharded, plus group metadata::

    groups/<family>_g00NNNN/group.json    shared context + sibling table + split
    .records/<episode_uuid>.json          full episode record (marker)
    videos/observation.images.main/chunk-XXX/file-NNNNNN.mp4
    object_states/chunk-XXX/file-NNNNNN.parquet   per-frame ball state

Every group holds 4 velocity siblings sharing the family's deterministic
scene; the ONLY per-episode variation is the ball's initial velocity
(speed + heading). Splits are group-atomic, stamped into group.json and every
episode record; the dataset is gated by
``ball_rolling_dynamics_scripts/validate_group_dataset.py`` upstream.

Frame convention: episodes render 75 frames (2.5 s @ 30 fps). The Wan VAE
chunks causally as ``1 + (T-1)//4`` and silently drops the tail, so we read
73 frames -> Tz = 19 exact (same rule as pbc's 76 -> 73).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .pbc_episode import read_frames  # noqa: F401  (re-exported; same domain)

N_FRAMES = 73
N_CONTROL = N_FRAMES - 1                       # 72
FPS = 30.0
FAMILIES = ("rolling_layout38_style42", "rolling_layout48_style41",
            "rolling_layout51_style34")


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    family: str
    group_key: str                 # "rolling_layout38_style42_g000123"
    sibling_index: int
    split: str
    episode_index: int
    episode_uuid: str
    family_root: Path              # <dataset_root>/<family>
    video_main_rel: str            # canonical sharded path, family-relative
    object_states_rel: str

    @property
    def video_main(self) -> Path:
        return self.family_root / self.video_main_rel

    @property
    def object_states_path(self) -> Path:
        return self.family_root / self.object_states_rel

    @property
    def marker_path(self) -> Path:
        return self.family_root / ".records" / f"{self.episode_uuid}.json"

    def record(self) -> dict:
        """The full canonical episode record (camera poses, physics, paths)."""
        return json.loads(self.marker_path.read_text())["episode"]


def build_index(dataset_root, families=None):
    """One row per sibling episode across all families.

    Split and group identity come from group.json (group-atomic by
    construction); artifact paths come from the sibling table, which the
    grouped generator fills from the committed canonical records. All rows
    share one appearance bucket by design: the text channel must carry no
    discriminative signal.
    """
    import pandas as pd

    dataset_root = Path(dataset_root)
    rows = []
    for family in (families or FAMILIES):
        family_root = dataset_root / family
        for gjson in sorted((family_root / "groups").glob("*/group.json")):
            g = json.loads(gjson.read_text())
            if len(g["siblings"]) != g["sibling_count"]:
                raise RuntimeError(f"{gjson}: sibling table incomplete")
            for sib in g["siblings"]:
                marker = (family_root / ".records"
                          / f"{sib['episode_uuid']}.json")
                record = json.loads(marker.read_text())["episode"]
                rows.append({
                    "episode_id": sib["episode_id"],
                    "family": g["family"],
                    "group_key": g["group_id"],
                    "sibling_index": int(sib["sibling_index"]),
                    "split": g["split"],
                    "episode_index": int(sib["episode_index"]),
                    "episode_uuid": sib["episode_uuid"],
                    "root": str(family_root),
                    "video_main": record["video_paths"][
                        "observation.images.main"],
                    "object_states": record["object_states_path"],
                    # taxonomy for prompts.bucket_key: one bucket for all
                    "leaf": g["family"],
                    "subfamily": "rolling_dynamics",
                    "variant": "ball_rolling",
                    "tool_type": "none",
                    "background_style": "robocasa_kitchen",
                })
    if not rows:
        raise FileNotFoundError(f"no group.json under {dataset_root}")
    idx = pd.DataFrame(rows)
    relabel = {g: i for i, g in enumerate(sorted(idx["group_key"].unique()))}
    idx["group_id"] = [relabel[g] for g in idx["group_key"]]
    return idx.sort_values("episode_id").reset_index(drop=True)


def refs_from_index(idx):
    return [EpisodeRef(r.episode_id, r.family, r.group_key,
                       int(r.sibling_index), r.split, int(r.episode_index),
                       r.episode_uuid, Path(r.root), r.video_main,
                       r.object_states)
            for r in idx.itertuples()]
