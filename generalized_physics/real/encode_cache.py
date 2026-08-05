"""Build a latent cache from f1_10h. Two passes.

  Pass A (CPU, one task)   -- filter, group, stratify, derive records, fit the
                              normalizer on the train split, read actions, and
                              write index.parquet / records.jsonl /
                              norm_stats.json / actions.f32.npy / manifest.json.
  Pass B (GPU, array task) -- decode mp4 -> VAE.encode -> fp16 into this
                              shard's slice of a preallocated memmap.

Splitting them means the expensive GPU pass is idempotent, restartable, and
trivially shardable, while everything order-dependent (grouping, normalizer
statistics) happens exactly once.

  python -m generalized_physics.real.encode_cache index --n 4000
  python -m generalized_physics.real.encode_cache encode --vae wan21 \
         --shard-id $SLURM_ARRAY_TASK_ID --n-shards 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..data.cache_wan_latents import control_bin_index
from ..models.metadata_records import MetadataRegistry
from . import f1_episode as F1
from . import f1_groups as FG
from . import f1_records as FR
from .paths import (ARMS, DATASET_ROOT, FPS, N_FRAMES, OUT_ROOT,
                    TEMPORAL_STRIDE)

N_CONTROL = N_FRAMES - 1                      # 56
LATENT_SHAPE = {"wan21": (16, 15, 60, 104), "wan22": (48, 15, 30, 52)}


# -------------------------------------------------------------- pass A ----

def run_index(args):
    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    idx = F1.build_index(args.dataset_root)
    n_all = len(idx)
    idx = idx[idx.eligible]
    idx, gstats = FG.assign_groups(idx)
    idx = FG.stratified_sample(idx, args.n, seed=args.seed)
    idx, gstats = FG.assign_groups(idx)        # relabel after sampling
    print(f"index: {n_all} -> {len(idx)} episodes | {gstats}", flush=True)

    registry = MetadataRegistry()
    records, actions, impacts = [], [], []
    for i, r in enumerate(idx.itertuples()):
        ref = F1.EpisodeRef(r.leaf, r.block, int(r.episode_index),
                            r.episode_uuid, Path(r.root))
        recs = FR.build_records(ref, r, fps=FPS)
        registry.validate(recs)
        records.append(recs)
        actions.append(F1.read_actions(ref, N_CONTROL))
        impacts.append(F1.impact_frame(ref, r.key_event_time_s, FPS, N_FRAMES))
        if (i + 1) % 250 == 0:
            print(f"  records {i+1}/{len(idx)}  ({time.time()-t0:.0f}s)",
                  flush=True)

    idx = idx.reset_index(drop=True)
    idx["impact_frame"] = [(-1 if v is None else v) for v in impacts]
    idx["prompt_id"] = _prompt_ids(idx)

    train = [rs for rs, s in zip(records, idx["split"]) if s == "train"]
    norm, dropped = FR.fit_normalizer(registry, train or records)
    unseen = ({r.key for rs in records for r in rs} - set(norm.stats)
              - set(dropped))
    if unseen:
        raise RuntimeError(f"keys present but not fitted on train: {unseen}")
    norm.save(out / "norm_stats.json")
    print(f"normalizer: {len(norm.stats)} keys, dropped constant {dropped}",
          flush=True)

    A = np.stack(actions).astype(np.float32)              # [N, 56, 8]
    tr = idx["split"].values == "train"
    mean = A[tr].reshape(-1, A.shape[-1]).mean(0)
    std = A[tr].reshape(-1, A.shape[-1]).std(0) + 1e-6
    np.save(out / "actions.f32.npy", ((A - mean) / std).astype(np.float32))

    with open(out / "records.jsonl", "w") as f:
        for rs in records:
            f.write(json.dumps([{"key": r.key, "scope": r.scope,
                                 "unit": r.unit, "value": r.value}
                                for r in rs]) + "\n")
    idx.to_parquet(out / "index.parquet")

    manifest = {
        "n_episodes": int(len(idx)), "n_frames": N_FRAMES,
        "n_control": N_CONTROL, "temporal_stride": TEMPORAL_STRIDE,
        "fps": FPS, "dataset_root": str(args.dataset_root),
        "group_stats": gstats, "summary": FG.summarize(idx),
        "action_mean": mean.tolist(), "action_std": std.tolist(),
        "keys_dropped_constant": dropped,
        "registry_version": 2,
        "eligibility": {
            "release_state": "blocked", "training_eligible": False,
            "override_rule": F1.ELIGIBILITY_RULE,
            "purpose": "internal method development; NOT a releasable result",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"pass A done in {time.time()-t0:.0f}s -> {out}", flush=True)


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

    # Rank 0 creates the file; the others wait for it rather than racing.
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
        ref = F1.EpisodeRef(r.leaf, r.block, int(r.episode_index),
                            r.episode_uuid, Path(r.root))
        x = F1.read_frames(ref, N_FRAMES)
        with torch.no_grad():
            lat = vae.encode([x.cuda()])[0]
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
    a.add_argument("--dataset-root", default=str(DATASET_ROOT))
    a.add_argument("--out", default=str(OUT_ROOT / "cache" / "wan21_vae"))
    a.add_argument("--n", type=int, default=4000)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=run_index)

    b = sub.add_parser("encode")
    b.add_argument("--out", default=str(OUT_ROOT / "cache" / "wan21_vae"))
    b.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    b.add_argument("--shard-id", type=int, default=0)
    b.add_argument("--n-shards", type=int, default=1)
    b.set_defaults(fn=run_encode)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
