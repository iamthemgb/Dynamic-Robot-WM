"""Build a latent cache from roll_groups (v1_velocity_groups). Two passes,
mirroring ``pbc_encode_cache.py`` so ``cache_io.load_cache`` consumes the
result untouched.

  Pass A (CPU, one task)   -- index all siblings from group.json + canonical
                              markers, derive the 4-key records, fit the
                              normalizer on the train split, build the ROI
                              masks, and write index.parquet / records.jsonl
                              / norm_stats.json / actions.f32.npy /
                              roi.u8.npy / manifest.json.
  Pass B (GPU, array task) -- decode the canonical main video -> VAE.encode
                              -> fp16 into this shard's slice of a
                              preallocated memmap.

Differences from the pbc builder, all deliberate:
  * no actions in the corpus (passive rollouts) -- actions.f32.npy is zeros
    [N, 72, 1] purely to satisfy the cache contract (action_dim=1);
  * no impact events -- the ball is in rolling contact throughout, so
    ``impact_frame`` is -1 everywhere;
  * ROI masks (NEW): per-episode latent-grid masks of the cells the ball
    occupies over time, for the ROI-weighted flow loss and the
    ROI-restricted paired eval (muffling diagnosis, problem 1). Built by
    projecting the recorded per-frame ball position (object_states parquet)
    through the recorded main camera (pos/lookat/fovy, world up=+Z) into
    pixels, then dividing by the VAE spatial stride. ROI resolution follows
    --vae, so each cache dir carries masks matching its latents.

  python -m generalized_physics.real.roll_encode_cache index \
         --dataset-root .../roll_groups_p1 --vae wan21
  python -m generalized_physics.real.roll_encode_cache encode --vae wan21 \
         --shard-id $SLURM_ARRAY_TASK_ID --n-shards 2
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..models.metadata_records import MetadataRegistry
from . import roll_episode as RE
from . import roll_records as RR
from .f1_records import fit_normalizer
from .paths import ARMS, OUT_ROOT

LATENT_SHAPE = {"wan21": (16, 19, 60, 104), "wan22": (48, 19, 30, 52)}
SPATIAL_STRIDE = {"wan21": 8, "wan22": 16}
FRAME_W, FRAME_H = 832, 480
TZ = 19


# ----------------------------------------------------------------- ROI ----

def _camera_axes(cam: dict):
    """forward/right/up rows for the recorded pos/lookat, world up=+Z —
    identical to the generator's own camera construction."""
    pos = np.asarray(cam["pos"], dtype=np.float64)
    forward = np.asarray(cam["lookat"], dtype=np.float64) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return pos, forward, right, cam_up


def _bin_of_frame(f: int) -> int:
    """Wan causal chunking: frame 0 -> bin 0; frames 4k-3..4k -> bin k."""
    return 0 if f == 0 else (f + 3) // 4


def roi_for_episode(record: dict, positions, stride: int, hz: int, wz: int,
                    n_frames: int = RE.N_FRAMES) -> np.ndarray:
    """-> uint8 [TZ, hz, wz]; 1 where the ball (plus one cell of margin)
    projects, unioned over the frames of each latent time bin."""
    cam = record["extras"]["camera_poses"]["main_camera"]
    pos, forward, right, cam_up = _camera_axes(cam)
    focal = 0.5 * FRAME_H / math.tan(math.radians(float(cam["fovy"])) / 2.0)
    radius = float(record["physics"]["ball_radius_m"])

    roi = np.zeros((TZ, hz, wz), dtype=np.uint8)
    ys = np.arange(hz, dtype=np.float64) + 0.5
    xs = np.arange(wz, dtype=np.float64) + 0.5
    for f in range(min(n_frames, len(positions))):
        delta = np.asarray(positions[f], dtype=np.float64) - pos
        depth = float(np.dot(forward, delta))
        if depth <= 1e-6:
            continue
        u = 0.5 * FRAME_W + focal * float(np.dot(right, delta)) / depth
        v = 0.5 * FRAME_H - focal * float(np.dot(cam_up, delta)) / depth
        r_cell = focal * radius / depth / stride + 1.0
        cu, cv = u / stride, v / stride
        if cu < -r_cell or cu > wz + r_cell or cv < -r_cell or cv > hz + r_cell:
            continue                              # fully out of frame
        disk = ((xs[None, :] - cu) ** 2 + (ys[:, None] - cv) ** 2
                <= r_cell * r_cell)
        roi[_bin_of_frame(f)] |= disk.astype(np.uint8)
    return roi


# -------------------------------------------------------------- pass A ----

def run_index(args):
    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stride = SPATIAL_STRIDE[args.vae]
    _c, _tz, hz, wz = LATENT_SHAPE[args.vae]
    t0 = time.time()

    idx = RE.build_index(args.dataset_root)
    groups = idx["group_id"].nunique()
    print(f"index: {len(idx)} episodes in {groups} groups | splits "
          f"{idx['split'].value_counts().to_dict()}", flush=True)

    registry = MetadataRegistry()
    records, rois = [], np.zeros((len(idx), TZ, hz, wz), dtype=np.uint8)
    for i, ref in enumerate(RE.refs_from_index(idx)):
        record = ref.record()
        recs = RR.build_records(record)
        registry.validate(recs)
        records.append(recs)
        states = pd.read_parquet(ref.object_states_path)
        rois[i] = roi_for_episode(record, list(states["object.position"]),
                                  stride, hz, wz)
        if (i + 1) % 250 == 0:
            print(f"  records+roi {i+1}/{len(idx)}  ({time.time()-t0:.0f}s)",
                  flush=True)

    frac = float(rois.mean())
    if not 0.001 <= frac <= 0.05:
        raise RuntimeError(f"ROI fraction {frac:.4f} outside sanity range "
                           "[0.001, 0.05] — projection or stride is wrong")
    np.save(out / "roi.u8.npy", rois)

    idx = idx.reset_index(drop=True)
    idx["impact_frame"] = -1                      # continuous rolling contact
    idx["prompt_id"] = _prompt_ids(idx)

    train = [rs for rs, s in zip(records, idx["split"]) if s == "train"]
    norm, dropped = fit_normalizer(registry, train or records)
    unseen = ({r.key for rs in records for r in rs} - set(norm.stats)
              - set(dropped))
    if unseen:
        raise RuntimeError(f"keys present but not fitted on train: {unseen}")
    weak = {k: s["sigma"] for k, s in norm.stats.items() if s["sigma"] < 1e-6}
    if weak:
        # sigma in (1e-9, 1e-6) survives the constant-drop yet amplifies
        # float noise up to 1e6x through the (sigma + eps) division
        raise RuntimeError(f"near-constant keys would amplify noise: {weak}")
    norm.save(out / "norm_stats.json")
    print(f"normalizer: {len(norm.stats)} keys, dropped constant {dropped}",
          flush=True)

    np.save(out / "actions.f32.npy",
            np.zeros((len(idx), RE.N_CONTROL, 1), dtype=np.float32))

    with open(out / "records.jsonl", "w") as f:
        for rs in records:
            f.write(json.dumps([{"key": r.key, "scope": r.scope,
                                 "unit": r.unit, "value": r.value}
                                for r in rs]) + "\n")
    idx.to_parquet(out / "index.parquet")

    manifest = {
        "dataset": "roll_groups_v1",
        "n_episodes": int(len(idx)), "n_groups": int(groups),
        "n_frames": RE.N_FRAMES, "n_control": RE.N_CONTROL,
        "temporal_stride": 4, "fps": RE.FPS,
        "dataset_root": str(args.dataset_root),
        "action_mean": [0.0], "action_std": [1.0],
        "keys_dropped_constant": dropped,
        "roi": {"vae": args.vae, "spatial_stride": stride,
                "shape": [TZ, hz, wz], "mean_fraction": round(frac, 5)},
        "conditioning": "velocity bundle (v0_cam + log10_speed); scene and "
                        "physics constant per family; no actions",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"pass A done in {time.time()-t0:.0f}s -> {out} "
          f"(roi fraction {frac:.4f})", flush=True)


def _prompt_ids(idx):
    from .prompts import bucket_key
    keys = [bucket_key(r) for r in idx.itertuples()]
    order = {k: i for i, k in enumerate(sorted(set(keys)))}
    return [order[k] for k in keys]


# -------------------------------------------------------------- pass B ----

def run_encode(args):
    import pandas as pd
    import torch

    from . import wan_loader as W

    out = Path(args.out)
    idx = pd.read_parquet(out / "index.parquet")
    n = len(idx)
    shape = (n,) + LATENT_SHAPE[args.vae]
    path = out / "latents.f16.npy"

    if args.shard_id == 0 and not path.exists():
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                  shape=shape)
        print(f"allocated {path} {shape} "
              f"({np.prod(shape)*2/1e9:.1f} GB)", flush=True)
    for _ in range(600):
        if path.exists():
            break
        time.sleep(1)
    z = np.lib.format.open_memmap(path, mode="r+")
    assert z.shape == shape, (z.shape, shape)

    arm = next(a for a in ARMS.values() if a.vae_kind == args.vae)
    vae = W.load_wan_vae(args.vae, arm.vae_path, device="cuda")
    mine = np.arange(n)[args.shard_id::args.n_shards]
    print(f"shard {args.shard_id}/{args.n_shards}: {len(mine)} episodes",
          flush=True)

    t0 = time.time()
    for c, i in enumerate(mine):
        r = idx.iloc[int(i)]
        x = RE.read_frames(Path(r.root) / r.video_main, RE.N_FRAMES)
        with torch.no_grad():
            lat = vae.encode([x.cuda()])[0]
        assert tuple(lat.shape) == LATENT_SHAPE[args.vae], lat.shape
        z[int(i)] = lat.to(torch.float16).cpu().numpy()
        if (c + 1) % 100 == 0:
            rate = (c + 1) / (time.time() - t0)
            print(f"  {c+1}/{len(mine)}  {rate:.2f} ep/s  "
                  f"eta {(len(mine)-c-1)/rate/60:.1f} min", flush=True)
    z.flush()

    done = out / "qc" / f"shard_{args.shard_id:02d}.done"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({
        "shard": args.shard_id, "n": len(mine),
        "seconds": round(time.time() - t0, 1),
        "vae_checkpoint_hash": W.sha256_head(arm.vae_path),
        "latent_shape": list(LATENT_SHAPE[args.vae])}))
    print(f"shard {args.shard_id} done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("index")
    a.add_argument("--dataset-root", required=True)
    a.add_argument("--out", default=str(OUT_ROOT / "cache" / "roll_wan21_vae"))
    a.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    a.set_defaults(fn=run_index)

    b = sub.add_parser("encode")
    b.add_argument("--out", default=str(OUT_ROOT / "cache" / "roll_wan21_vae"))
    b.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    b.add_argument("--shard-id", type=int, default=0)
    b.add_argument("--n-shards", type=int, default=1)
    b.set_defaults(fn=run_encode)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
