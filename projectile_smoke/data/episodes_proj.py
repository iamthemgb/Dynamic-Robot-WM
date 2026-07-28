"""Episode registry for the projectile smoke run.

Data: ONLY projectile_ball_catch_robocasa_kitchen_scenes_rollouts, 4 family
dirs x 750 episodes (style058_seed2350 is a 1-episode preview - skipped).

Keys are "proj/<episode_id>" and cache stems "proj__<episode_id>" so the
latents/t5 precomputed into wan_physics_cache during the cloth-plan work are
reused verbatim (verified 2026-07-15: 3000 latents [16,11,60,104] bf16 +
3000 t5_blind embeddings, keyed exactly this way).

Verified 2026-07-15 (all 3000 episodes): the "opposite_camera twin" premise
of the plan is FALSE for this dataset - no two episodes share physics
(ball_mass + initial velocity are unique per episode across all 4 families).
Splits/bootstrap code still carries a physics-identity cluster id so the
twin-leakage assertion exists and would fire on a re-generated twinned set.
"""

import json
from pathlib import Path

ROOT = Path("/gpfs/radev/project/sous/mzl7/"
            "projectile_ball_catch_robocasa_kitchen_scenes_rollouts")

FAMILIES = [
    "style020_seed0",
    "style020_seed0_opposite_camera",
    "style055_seed1000",
    "style055_seed1000_opposite_camera",
]

CACHE = Path("/gpfs/radev/scratch/sous/mzl7/wan_projectile_smoke_cache")

WAN_MODEL_DIR = Path("/gpfs/radev/scratch/sous/mzl7/wan_models/Wan2.1-T2V-1.3B")

FPS = 30
FRAME_COUNT = 76
NUM_WAN_FRAMES = 41  # 76f@30 -> 41f@16, latent [16, 11, 60, 104]


def episode_key(episode_id):
    return f"proj/{episode_id}"


def file_stem(key):
    return key.replace("/", "__", 1)


def family_of(episode_id):
    for fam in FAMILIES:
        if episode_id.startswith(fam + "_"):
            return fam
    raise ValueError(f"episode {episode_id} not in a known family")


def iter_episodes(families=None):
    """Yield (key, family, metadata_path) sorted by key."""
    for fam in families or FAMILIES:
        ep_root = ROOT / fam / "episodes"
        for meta_path in sorted(ep_root.glob("*/metadata.json")):
            yield episode_key(meta_path.parent.name), fam, meta_path


def load_metadata(meta_path):
    with open(meta_path) as f:
        return json.load(f)


def video_path(meta_path):
    return Path(meta_path).parent / "main.mp4"


def physics_identity(meta):
    """Fingerprint of the sampled physics. Episodes sharing this are 'twins'
    (same sim, different camera) and must never straddle the train/val split.
    On the current dataset every episode is its own singleton cluster."""
    return (round(meta["ball_mass"], 14),
            tuple(round(x, 10) for x in meta["ball_initial_velocity"]),
            tuple(round(x, 10) for x in meta["ball_initial_position"]))
