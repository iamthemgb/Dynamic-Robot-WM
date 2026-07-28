"""Per-episode static physics vector (nominally 14-d) + z-score norm stats.

Layout (before the degenerate-dim guard):
  [ log10 ball_mass, ball_radius,
    p0.x, p0.y, p0.z,                # ball_initial_position, MAIN-CAMERA frame
    v0.x, v0.y, v0.z, log10 |v0|,    # ball_initial_velocity + speed
    icpt.x, icpt.y, icpt.z,          # intercept_position, main-camera frame
    ballistic_intercept_time_s, gripper_close_time ]

Positions/velocities are transformed into the per-episode main-camera frame
(camera pose varies per episode; the opposite_camera families make
world-frame vectors visually ambiguous). Camera frame: x right, y up,
z backward (see camera.py).

NO failure_mode one-hot (decided in the plan): outcome lives in the caption
only, so the shuffle gap measures ballistics, not outcome consistency.

Degenerate-dim guard: any dim whose TRAIN std is < 1e-3 x its mean absolute
scale is dropped (z-scoring a near-constant dim amplifies noise into a
full-scale feature). What survives is the final vector; norm_stats.json
records exactly what was dropped and must be eyeballed before training.

Times are in episode time (the ball is released at events.release_frame,
typically 0.17-0.33 s in); ballistics before release are a constant p0.

  python projectile_smoke/data/physics_vec.py
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projectile_smoke.data.camera import world_dir_to_camera, world_to_camera  # noqa: E402
from projectile_smoke.data.episodes_proj import (  # noqa: E402
    CACHE, iter_episodes, load_metadata)

FIELDS = ["log10_ball_mass", "ball_radius",
          "p0_cam_x", "p0_cam_y", "p0_cam_z",
          "v0_cam_x", "v0_cam_y", "v0_cam_z", "log10_speed",
          "icpt_cam_x", "icpt_cam_y", "icpt_cam_z",
          "ballistic_intercept_time_s", "gripper_close_time"]

STD_FLOOR_RATIO = 1e-3  # drop dim if train std < ratio * mean(|x|)


def extract(meta):
    cam = meta["cameras"]["main_camera"]
    pos, lookat = cam["pos"], cam["lookat"]
    p0 = world_to_camera([meta["ball_initial_position"]], pos, lookat)[0]
    v0 = world_dir_to_camera([meta["ball_initial_velocity"]], pos, lookat)[0]
    icpt = world_to_camera([meta["intercept_position"]], pos, lookat)[0]
    speed = float(np.linalg.norm(meta["ball_initial_velocity"]))
    return [math.log10(meta["ball_mass"]), meta["ball_radius"],
            p0[0], p0[1], p0[2],
            v0[0], v0[1], v0[2], math.log10(speed),
            icpt[0], icpt[1], icpt[2],
            meta["controller_result"]["ballistic_intercept_time_s"],
            meta["gripper_close_time"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(CACHE))
    args = ap.parse_args()
    out = Path(args.out_dir)

    with open(out / "splits.json") as f:
        train_eps = set(json.load(f)["train_episodes"])

    raw = {}
    for key, fam, meta_path in iter_episodes():
        vec = extract(load_metadata(meta_path))
        assert all(math.isfinite(x) for x in vec), f"non-finite vector for {key}"
        raw[key] = vec

    train_mat = np.array([raw[k] for k in sorted(train_eps)])
    assert train_mat.shape == (len(train_eps), len(FIELDS))
    mean = train_mat.mean(axis=0)
    std = train_mat.std(axis=0, ddof=1)
    scale = np.abs(train_mat).mean(axis=0)

    keep, dropped = [], []
    for i, name in enumerate(FIELDS):
        ratio = std[i] / max(scale[i], 1e-12)
        (keep if ratio >= STD_FLOOR_RATIO else dropped).append(i)
        print(f"  {name:28s} mean={mean[i]:9.4f} std={std[i]:9.5f} "
              f"scale={scale[i]:9.4f} std/scale={ratio:9.2e} "
              f"{'KEEP' if i in keep else 'DROP (degenerate)'}")
    assert keep, "all dims degenerate?!"
    if dropped:
        print(f"dropped {len(dropped)} degenerate dim(s): "
              f"{[FIELDS[i] for i in dropped]}")
    else:
        print("no degenerate dims dropped (note: plan guessed ball_radius "
              "would fall out; its std/scale is ~1e-2, above the 1e-3 floor)")

    vectors = {k: [v[i] for i in keep] for k, v in raw.items()}
    stats = {
        "fields_all": FIELDS,
        "fields_kept": [FIELDS[i] for i in keep],
        "fields_dropped": [FIELDS[i] for i in dropped],
        "std_floor_ratio": STD_FLOOR_RATIO,
        "mean": [mean[i] for i in keep],
        "std": [max(std[i], 1e-6) for i in keep],
        "n_train": len(train_eps),
        "phys_dim": len(keep),
    }
    with open(out / "physics_vectors.json", "w") as f:
        json.dump(vectors, f)
    with open(out / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"{len(vectors)} vectors, final dim {len(keep)} "
          f"-> {out / 'physics_vectors.json'}")


if __name__ == "__main__":
    main()
