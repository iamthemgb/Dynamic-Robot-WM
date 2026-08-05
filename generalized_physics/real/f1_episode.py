"""Reading the f1_10h corpus (schema ``dynamic-robot-dataset/v2``).

Layout, per block (12 blocks = 4 leaves x 3 blocks, 1500 episodes each)::

    <leaf>/block-000N/
        meta/{info.json,episodes.parquet,tasks.parquet,splits.parquet,cameras.parquet}
        data/chunk-00C/file-NNNNNN.parquet          60 rows @ 30 Hz
        high_rate/  object_states/                  2401 rows @ 1200 Hz
        events/                                     contact events
        transitions/                                motion-mode changes
        videos/observation.images.{main,secondary}/chunk-00C/file-NNNNNN.mp4

``chunk = episode_index // 1000`` and the file index equals ``episode_index``;
all five parquet streams and both mp4s share that index.

Two traps this module exists to contain:
  * ``episode_index`` restarts at 0 in every block -- only ``episode_uuid`` is
    globally unique, so every row is keyed on (leaf, block, episode_index).
  * ``robot.joint_position`` is length 9 or 15 depending on embodiment, so it
    can never be stacked; only ``action.actuator_command`` (length 8, uniform)
    is safe as a dense array.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import DATASET_ROOT, N_FRAMES

LEAVES = ("F1a", "F1b", "F1c", "F1d")
MAIN_CAM = "observation.images.main"


@dataclass(frozen=True)
class EpisodeRef:
    leaf: str
    block: str
    episode_index: int
    episode_uuid: str
    root: Path                      # .../<leaf>/<block>

    @property
    def chunk(self) -> str:
        return f"chunk-{self.episode_index // 1000:03d}"

    @property
    def stem(self) -> str:
        return f"file-{self.episode_index:06d}"

    def stream(self, name: str) -> Path:
        return self.root / name / self.chunk / f"{self.stem}.parquet"

    def video(self, camera: str = MAIN_CAM) -> Path:
        return self.root / "videos" / camera / self.chunk / f"{self.stem}.mp4"


def _blocks(dataset_root: Path, leaves=None):
    for leaf in (leaves or LEAVES):
        for block in sorted((dataset_root / leaf).glob("block-*")):
            if block.is_dir():
                yield leaf, block


def build_index(dataset_root=None, leaves=None):
    """One row per episode across all requested blocks.

    Returns a pandas DataFrame carrying the episodes.parquet columns we use,
    the parsed randomization/physics JSON blobs, and an ``eligible`` flag.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    dataset_root = Path(dataset_root or DATASET_ROOT)
    keep = ["episode_uuid", "episode_index", "family", "subfamily", "variant",
            "intended_branch", "actual_outcome", "actual_outcome_class",
            "task_success", "failure_mode", "robot_model", "tool_type",
            "frame_count", "duration_s", "event_time_s", "key_event_name",
            "key_event_time_s", "physics_qc_pass", "split", "task_index"]
    frames = []
    for leaf, block in _blocks(dataset_root, leaves):
        t = pq.read_table(block / "meta" / "episodes.parquet")
        cols = [c for c in keep if c in t.column_names]
        df = t.select(cols).to_pandas()
        rnd = [json.loads(s) if s else {}
               for s in t.column("randomization_json").to_pylist()]
        phy = [json.loads(s) if s else {}
               for s in t.column("physics_json").to_pylist()]
        df["background_style"] = [r.get("background_style", "unknown")
                                  for r in rnd]
        df["scene_asset_id"] = [r.get("scene_asset_id") for r in rnd]
        df["object_mass"] = [_param(p, "object_mass") for p in phy]
        df["object_radius"] = [_param(p, "object_radius") for p in phy]
        df["leaf"] = leaf
        df["block"] = block.name
        df["root"] = str(block)
        frames.append(df)

    idx = pd.concat(frames, ignore_index=True)
    idx["eligible"] = eligibility_mask(idx)
    return idx


def _param(physics_json: dict, name: str):
    p = (physics_json or {}).get("parameters", {}).get(name)
    if isinstance(p, dict):
        return p.get("value")
    return p


#: The corpus ships with release_state='blocked', training_eligible=false and
#: every episode listed in qc/manifests/quarantine.jsonl (default_training is
#: empty). The repo's own documented workaround is to consume it by filtering
#: per-episode physics_qc_pass. We apply that plus two tightenings and record
#: the rule verbatim in every cache manifest -- outputs are internal method
#: development, not releasable results.
ELIGIBILITY_RULE = ("physics_qc_pass == True AND actual_outcome != 'invalid' "
                    "AND NOT (leaf == 'F1d' AND tool_type == "
                    "'robotiq_2f85_thick_pad') AND frame_count == 60")


def eligibility_mask(idx):
    degenerate = (idx["leaf"] == "F1d") & (
        idx["tool_type"] == "robotiq_2f85_thick_pad")
    return (idx["physics_qc_pass"].astype(bool)
            & (idx["actual_outcome"] != "invalid")
            & ~degenerate
            & (idx["frame_count"] == 60))


def refs_from_index(idx):
    return [EpisodeRef(r.leaf, r.block, int(r.episode_index), r.episode_uuid,
                       Path(r.root)) for r in idx.itertuples()]


# -- per-episode readers ---------------------------------------------------

def read_frames(ref: EpisodeRef, n_frames: int = N_FRAMES, camera=MAIN_CAM):
    """-> float32 [3, T, 480, 832] in [-1, 1], the Wan VAE input domain.

    No resize: 832x480 is Wan 2.1's native 480p landscape size. The latents
    must NOT be re-normalised afterwards -- WanVAE.encode applies its own
    scale=[mean, 1/std] whitening internally.
    """
    import av
    import torch

    container = av.open(str(ref.video(camera)))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= n_frames:
            break
    container.close()
    if len(frames) < n_frames:
        raise RuntimeError(f"{ref.video(camera)}: {len(frames)} < {n_frames}")
    x = torch.from_numpy(np.stack(frames))          # [T, H, W, 3] uint8
    return x.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)


def read_stream(ref: EpisodeRef, name: str, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(ref.stream(name), columns=columns)


def read_actions(ref: EpisodeRef, n_control: int = N_FRAMES - 1):
    """-> float32 [n_control, 8] actuator commands.

    Control step t drives frame t -> t+1, matching
    ``cache_wan_latents.control_bin_index``.
    """
    t = read_stream(ref, "data", ["action.actuator_command"])
    a = np.stack(t.column("action.actuator_command").to_pylist()[:n_control])
    return a.astype(np.float32)


def read_camera(ref: EpisodeRef, camera=MAIN_CAM):
    """world_to_camera (4x4) and intrinsics for this episode's camera."""
    import pyarrow.parquet as pq
    t = pq.read_table(ref.root / "meta" / "cameras.parquet")
    df = t.to_pandas()
    row = df[(df["camera_name"] == camera)
             & (df.get("episode_index", df.index) == ref.episode_index)]
    if row.empty:
        row = df[df["camera_name"] == camera].iloc[:1]
    r = row.iloc[0]
    return {"world_to_camera": np.asarray(r["world_to_camera"],
                                          dtype=np.float64).reshape(4, 4),
            "intrinsic_matrix": np.asarray(r["intrinsic_matrix"],
                                           dtype=np.float64).reshape(3, 3)}


def impact_frame(ref: EpisodeRef, key_event_time_s=None, fps: float = 30.0,
                 n_frames: int = N_FRAMES):
    """First free_flight -> impact transition, as a FRAME index.

    ``GroupBatcher._gather`` divides this by temporal_stride itself, so what is
    stored must be a frame index, not a latent bin.
    """
    try:
        t = read_stream(ref, "transitions")
    except Exception:
        t = None
    if t is not None and t.num_rows:
        d = t.to_pydict()
        for ts, to in zip(d.get("timestamp", []), d.get("to", [])):
            if str(to) == "impact":
                return int(min(max(round(float(ts) * fps), 0), n_frames - 1))
    if key_event_time_s is not None and np.isfinite(key_event_time_s):
        return int(min(max(round(float(key_event_time_s) * fps), 0),
                       n_frames - 1))
    return None
