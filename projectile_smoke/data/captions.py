"""Physics-blind captions for the projectile episodes.

Templates are copied VERBATIM from physics_finetune/data/captions.py (the
cloth-plan code that produced the cached t5_blind embeddings being reused
here), including its "A orange ball" article bug - byte-identical captions
are required so the existing T5 cache stays valid. This script therefore
ALWAYS verifies its output against the cached t5_blind/*.pt caption strings
for every episode and fails loudly on any mismatch or missing file.

Division of labor (deliberate, see plan): outcome/failure lives in the
caption ONLY; ballistics live in the physics tokens ONLY - so the shuffle
gap isolates ballistics.

  python projectile_smoke/data/captions.py [--skip-t5-verify]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projectile_smoke.data.episodes_proj import (  # noqa: E402
    CACHE, file_stem, iter_episodes, load_metadata)

# --- verbatim from physics_finetune/data/captions.py ---------------------
BALL_TEMPLATE = (
    "A {color} ball flies through a kitchen in an arc, and a Franka "
    "robot arm reaches out to catch it in mid-air."
)

BALL_FAILURE_NOTES = {
    "spatial_near_miss": "the gripper closes just beside the ball (near miss)",
    "contact_failure": "the ball slips away without being secured",
    "wrong_action": "the arm reaches toward the wrong spot",
}

PALETTE = {
    "red": (0.75, 0.15, 0.15),
    "orange": (0.85, 0.5, 0.15),
    "yellow": (0.85, 0.8, 0.2),
    "green": (0.2, 0.6, 0.25),
    "teal": (0.2, 0.6, 0.6),
    "blue": (0.2, 0.3, 0.65),
    "purple": (0.5, 0.25, 0.6),
    "pink": (0.85, 0.5, 0.6),
    "brown": (0.45, 0.3, 0.18),
    "gray": (0.5, 0.5, 0.5),
    "white": (0.9, 0.9, 0.9),
    "black": (0.12, 0.12, 0.12),
}


def color_word(rgba):
    r, g, b = rgba[:3]
    return min(PALETTE, key=lambda k: sum((a - c) ** 2 for a, c in zip(PALETTE[k], (r, g, b))))


def build_caption(meta):
    text = BALL_TEMPLATE.format(color=color_word(meta["ball_color"]))
    if meta.get("outcome") != "success":
        note = BALL_FAILURE_NOTES.get(meta.get("failure_mode"),
                                      "the catch attempt fails")
        text += f" However, {note}."
    return text
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(CACHE))
    ap.add_argument("--skip-t5-verify", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, fam, meta_path in iter_episodes():
        meta = load_metadata(meta_path)
        rows.append({
            "key": key,
            "family": fam,
            "episode_id": meta["episode_id"],
            "failure_mode": meta.get("failure_mode", "none"),
            "outcome": meta.get("outcome"),
            "caption": build_caption(meta),
        })

    if not args.skip_t5_verify:
        import torch
        t5_dir = out / "t5_blind"
        mismatched, missing = [], []
        for r in rows:
            p = t5_dir / f"{file_stem(r['key'])}.pt"
            if not p.exists():
                missing.append(r["key"])
                continue
            cached = torch.load(p, map_location="cpu", weights_only=True)
            if cached["caption"] != r["caption"]:
                mismatched.append((r["key"], cached["caption"], r["caption"]))
        assert not missing, f"{len(missing)} episodes missing from t5 cache, e.g. {missing[:5]}"
        assert not mismatched, (
            f"{len(mismatched)} caption mismatches vs cached T5 - the cache "
            f"is stale for these captions and must be re-encoded. First: "
            f"{mismatched[0]}")
        print(f"t5 cache verification: all {len(rows)} captions byte-identical to cache")

    dst = out / "captions_blind.jsonl"
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {dst}")


if __name__ == "__main__":
    main()
