"""Build the rbi cache from the EXISTING zl664 artifacts. CPU-only, no VAE.

Encode parity was verified before this module was written: both pipelines
feed the first 73 frames as float32 [-1, 1] [3,T,480,832] into the same
``wan.modules.vae.WanVAE(z_dim=16, dtype=float32)`` (vendored sources
byte-identical, same checkpoint, hash ``9597dcd3869fda28``), and neither
re-normalizes latents. So the 30 GB latent cache the failed campaign built
(``zl664/physics_training/artifacts/rolling_ball_velocity_4k_v1/cache/
wan21_t2v_1p3b``) is bit-for-bit what ``rbi_encode_cache`` would produce —
this converter just selects the MAIN-view rows (even ``sample_index``) and
re-materializes them under the ``cache_io.load_cache`` contract.

Row ordering is (group_index, action_id, state_id): velocity-pair members
are ADJACENT (rows 0-1, 2-3 of each rbi group), so any even ``--limit``
keeps stage-1 pairs whole; action pairs are rows (0,2) and (1,3), hence
stage-2 smokes need ``--limit % 4 == 0`` (enforced here and in the
campaign).

Two group-id columns are written because the two stages pair differently:

  group_id         dense id of (rbi group x action plan)  -- stage 1: the
                   wrong donor is the OPPOSITE-VELOCITY sibling under an
                   identical action stream (the tex donor rule);
  action_group_id  dense id of (rbi group x velocity state) -- stage 2: the
                   wrong donor is the OPPOSITE-ACTION sibling at identical
                   velocity. Ignored by load_cache; the campaign swaps it in
                   from cache["index"].

  python -m generalized_physics.real.rbi_convert_cache convert
  python -m generalized_physics.real.rbi_convert_cache verify
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..models.metadata_records import MetadataRegistry
from . import rbi_records as RR
from .f1_records import fit_normalizer
from .paths import OUT_ROOT, ROOT

ZL_ART = (ROOT / "zl664" / "physics_training" / "artifacts"
          / "rolling_ball_velocity_4k_v1")
ZL_CONFIG = (ROOT / "zl664" / "physics_training" / "configs"
             / "phase1_rolling_velocity.json")
DEFAULT_OUT = OUT_ROOT / "cache" / "rbi_wan21_vae"

LATENT_SHAPE = (16, 19, 60, 104)
ROI_SHAPE = (19, 60, 104)
N_FRAMES = 73
N_CONTROL = 73                    # frame-aligned actions, one row per frame
SPLIT_MAP = {"validation_id": "val"}
#: constant taxonomy so prompts.bucket_key yields exactly one bucket (the
#: t5_cache fallback path; the primary path repacks the cached embedding).
TAXONOMY = {"leaf": "rbi_interception", "subfamily": "rolling_intercept",
            "variant": "intercept", "tool_type": "franka_hand",
            "background_style": "robocasa_kitchen"}


def _main_rows(index_root, limit=None):
    import pandas as pd

    df = pd.read_parquet(Path(index_root) / "samples.parquet")
    df = df[df["view"] == "main"].copy()
    if (df["sample_index"].to_numpy() % 2 != 0).any():
        raise RuntimeError("main-view sample_index not all even — the "
                           "source index layout changed; do not convert")
    df = (df.sort_values(["group_index", "action_id", "state_id"])
            .reset_index(drop=True))
    if limit is not None:
        if limit % 4:
            raise SystemExit("--limit must be a multiple of 4 so both the "
                             "velocity and the action pairing stay whole")
        df = df.iloc[:limit].copy().reset_index(drop=True)
    return df


def _build_index(df):
    import pandas as pd

    vkey = df["group_id"] + "|" + df["action_id"]
    akey = df["group_id"] + "|" + df["state_id"]
    out = pd.DataFrame({
        "episode_id": df["episode_id"],
        "rbi_group": df["group_id"],
        "rbi_group_id": df["group_index"].astype(np.int64),
        "group_key": vkey,
        "group_id": pd.factorize(vkey)[0].astype(np.int64),
        "action_group_key": akey,
        "action_group_id": pd.factorize(akey)[0].astype(np.int64),
        "cell_id": df["cell_id"],
        "state_id": df["state_id"],
        "action_id": df["action_id"],
        "split": df["split"].map(lambda s: SPLIT_MAP.get(s, s)),
        "contrast_type": df["contrast_type"],
        "catch_success": df["catch_success"],
        "v0_robot_x": df["v0_robot_x"].astype(np.float64),
        "v0_robot_y": df["v0_robot_y"].astype(np.float64),
        "action_plan_sha256": df["action_plan_sha256"],
        "source_sample_index": df["sample_index"].astype(np.int64),
        "impact_frame": -1,
        "prompt_id": 0,
    })
    for k, v in TAXONOMY.items():
        out[k] = v
    return out


def _assert_pairs(idx):
    """Both pairings must be exact 2-member counterfactuals, group-atomic."""
    for col, same, diff in (("group_id", "action_id", "state_id"),
                            ("action_group_id", "state_id", "action_id")):
        for _, sub in idx.groupby(col):
            if len(sub) != 2:
                raise RuntimeError(f"{col}: pair with {len(sub)} members")
            if sub[same].nunique() != 1 or set(sub[diff]) != {"A", "B"}:
                raise RuntimeError(f"{col}: not a clean A/B pair over {diff}")
    for _, sub in idx.groupby("rbi_group_id"):
        if sub["split"].nunique() != 1:
            raise RuntimeError("split not atomic by rbi group")


def run_convert(args):
    t0 = time.time()
    index_root = Path(args.index_root)
    zl_cache = Path(args.zl_cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    src_manifest = json.loads((index_root / "manifest.json").read_text())
    cache_manifest = json.loads((zl_cache / "manifest.json").read_text())
    if int(cache_manifest["n_frames"]) != N_FRAMES:
        raise RuntimeError(f"source cache n_frames "
                           f"{cache_manifest['n_frames']} != {N_FRAMES}")
    complete = np.load(zl_cache / "complete.u8.npy")
    if not bool(np.all(complete == 1)):
        raise RuntimeError("source latent cache is incomplete")

    df = _main_rows(index_root, limit=args.limit)
    idx = _build_index(df)
    _assert_pairs(idx)
    src = idx["source_sample_index"].to_numpy()
    n = len(idx)
    print(f"{n} main-view episodes | {idx['group_id'].nunique()} velocity "
          f"pairs | {idx['action_group_id'].nunique()} action pairs | "
          f"splits {idx['split'].value_counts().to_dict()}", flush=True)

    # records + normalizer (train split), cross-checked against the source
    registry = MetadataRegistry()
    records = [RR.build_records(r) for r in idx.itertuples()]
    for rs in records:
        registry.validate(rs)
    train = [rs for rs, s in zip(records, idx["split"]) if s == "train"]
    norm, dropped = fit_normalizer(registry, train or records)
    if dropped:
        raise RuntimeError(f"constant conditioning keys: {dropped}")
    if args.limit is None:
        zl_norm = json.loads((index_root / "normalizer.json").read_text())
        for k, s in norm.stats.items():
            zs = zl_norm["stats"][k]
            d_mu, d_sig = abs(s["mu"] - zs["mu"]), abs(s["sigma"] - zs["sigma"])
            print(f"normalizer {k}: mu {s['mu']:.8f} (zl {zs['mu']:.8f}) "
                  f"sigma {s['sigma']:.8f} (zl {zs['sigma']:.8f})", flush=True)
            # mu is exact; sigma differs only by the (n-1) of duplicated-vs-
            # deduped fitting populations, bounded well under 1e-4
            if d_mu > 1e-9 or d_sig > 1e-4:
                raise RuntimeError(f"normalizer drift on {k}: "
                                   f"d_mu={d_mu:.2e} d_sigma={d_sig:.2e}")
    norm.save(out / "norm_stats.json")
    with open(out / "records.jsonl", "w") as f:
        for rs in records:
            f.write(json.dumps([{"key": r.key, "scope": r.scope,
                                 "unit": r.unit, "value": r.value}
                                for r in rs]) + "\n")

    # actions: copy verbatim — already normalized upstream (R9)
    acts = np.load(index_root / "actions.f32.npy", mmap_mode="r")
    np.save(out / "actions.f32.npy", np.asarray(acts[src], dtype=np.float32))

    # roi: unpack the per-view packbits masks for the selected rows
    packed = np.load(index_root / "roi_masks_1p3b.packbits.npy", mmap_mode="r")
    rois = np.unpackbits(np.asarray(packed[src]), axis=-1,
                         count=ROI_SHAPE[-1], bitorder="little").astype(np.uint8)
    if rois.shape != (n,) + ROI_SHAPE:
        raise RuntimeError(f"roi shape {rois.shape}")
    frac = float(rois.mean())
    if not 0.005 <= frac <= 0.30:
        raise RuntimeError(f"ROI fraction {frac:.4f} outside [0.005, 0.30] "
                           "— these are fat visibility-aware masks (~0.1), "
                           "not roll's thin tubes; the unpack is wrong")
    np.save(out / "roi.u8.npy", rois)
    print(f"roi mean fraction {frac:.4f}", flush=True)

    # latents: chunked row gather from the source memmap
    z_src = np.load(zl_cache / "latents.f16.npy", mmap_mode="r")
    if tuple(z_src.shape[1:]) != LATENT_SHAPE:
        raise RuntimeError(f"source latent shape {z_src.shape}")
    z_out = np.lib.format.open_memmap(
        out / "latents.f16.npy", mode="w+", dtype=np.float16,
        shape=(n,) + LATENT_SHAPE)
    for i in range(0, n, args.chunk):
        j = min(i + args.chunk, n)
        z_out[i:j] = z_src[src[i:j]]
        print(f"  latents {j}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    z_out.flush()

    # text context: repack the cached umT5 embedding (fp16 -> bf16 lossless
    # in range; max |x| ~ 1.07)
    import torch

    prompt = json.loads(Path(args.zl_config).read_text())["prompt"]
    ctx = torch.load(zl_cache / "prompt_context.pt", map_location="cpu")
    if not torch.isfinite(ctx).all():
        raise RuntimeError("prompt_context.pt has non-finite values")
    (out / "t5").mkdir(exist_ok=True)
    torch.save({"0": ctx.to(torch.bfloat16)}, out / "t5" / "embeddings.pt")
    (out / "t5" / "prompts.json").write_text(json.dumps({"0": prompt},
                                                        indent=2))

    idx.to_parquet(out / "index.parquet")
    manifest = {
        "dataset": "rolling_ball_velocity_4k_v1",
        "view": "main",
        "n_episodes": int(n),
        "n_groups": int(idx["group_id"].nunique()),
        "n_action_groups": int(idx["action_group_id"].nunique()),
        "n_rbi_groups": int(idx["rbi_group_id"].nunique()),
        "n_frames": N_FRAMES, "n_control": N_CONTROL,
        "temporal_stride": 4, "fps": 30.0,
        "action_dim": int(src_manifest["action_dim"]),
        "action_mean": src_manifest["action_mean"],
        "action_std": src_manifest["action_std"],
        "action_note": "copied verbatim from the zl664 index — already "
                       "normalized clip((q-mean)/(3*std), -1, 1) on "
                       "train-split stats; never re-normalize",
        "n_control_note": "control_bin_index(73, 4) emits a final bin == Tz; "
                          "stages 1-2 never consume bin_index, but a future "
                          "student-phase consumer must not window by it",
        "vae_checkpoint_hash": cache_manifest["vae_checkpoint_head_sha256"],
        "dataset_root": src_manifest.get("dataset_root"),
        "roi": {"vae": "wan21", "spatial_stride": 8,
                "shape": list(ROI_SHAPE), "mean_fraction": round(frac, 5),
                "source": src_manifest.get("roi_mask_source")},
        "donor_rule_stage1": "same rbi group x action plan x main view; "
                             "opposite velocity state",
        "donor_rule_stage2": "same rbi group x velocity state x main view; "
                             "opposite action plan",
        "source": {
            "index_root": str(index_root), "zl_cache": str(zl_cache),
            "zl_cache_manifest_sha256": hashlib.sha256(
                (zl_cache / "manifest.json").read_bytes()).hexdigest(),
            "limit": args.limit,
        },
        "conditioning": "velocity bundle (v0_robot_x/y) only; the 2x2 "
                        "action plans are intentionally NOT records (tex "
                        "B1) — the action pathway trains in a separate "
                        "stage against action_group_id pairing",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"convert done in {time.time()-t0:.0f}s -> {out}", flush=True)


def run_verify(args):
    import pandas as pd

    t0 = time.time()
    out = Path(args.out)
    index_root = Path(args.index_root)
    idx = pd.read_parquet(out / "index.parquet")
    manifest = json.loads((out / "manifest.json").read_text())
    n = len(idx)

    # the on-disk mapping must match an independent recompute
    df = _main_rows(index_root, limit=manifest["source"]["limit"])
    expect = _build_index(df)
    for col in ("episode_id", "source_sample_index", "group_id",
                "action_group_id", "split", "state_id", "action_id"):
        if not (idx[col].to_numpy() == expect[col].to_numpy()).all():
            raise RuntimeError(f"index column {col} deviates from recompute")
    _assert_pairs(idx)

    rng = np.random.default_rng(0)
    probes = rng.choice(n, size=min(args.probes, n), replace=False)
    z_out = np.load(out / "latents.f16.npy", mmap_mode="r")
    z_src = np.load(Path(args.zl_cache) / "latents.f16.npy", mmap_mode="r")
    packed = np.load(index_root / "roi_masks_1p3b.packbits.npy", mmap_mode="r")
    roi_out = np.load(out / "roi.u8.npy", mmap_mode="r")
    acts_out = np.load(out / "actions.f32.npy", mmap_mode="r")
    acts_src = np.load(index_root / "actions.f32.npy", mmap_mode="r")
    src = idx["source_sample_index"].to_numpy()
    for k in probes:
        k = int(k)
        if not np.array_equal(z_out[k], z_src[src[k]]):
            raise RuntimeError(f"latent row {k} != source row {src[k]}")
        want = np.unpackbits(np.asarray(packed[src[k]]), axis=-1,
                             count=ROI_SHAPE[-1], bitorder="little")
        if not np.array_equal(roi_out[k], want):
            raise RuntimeError(f"roi row {k} mismatch")
        if not np.array_equal(acts_out[k], acts_src[src[k]]):
            raise RuntimeError(f"actions row {k} mismatch")

    # records alignment: row k's record values must equal the index columns
    with open(out / "records.jsonl") as f:
        raw = [json.loads(line) for line in f]
    for k in probes:
        vals = {r["key"]: r["value"] for r in raw[int(k)]}
        row = idx.iloc[int(k)]
        if (abs(vals["v0_robot_x"] - row.v0_robot_x) > 1e-12
                or abs(vals["v0_robot_y"] - row.v0_robot_y) > 1e-12):
            raise RuntimeError(f"records row {k} misaligned")

    from . import cache_io

    cache, _reg, _norm = cache_io.load_cache(out, split="train", limit=8)
    gid = cache["group_id"].numpy()
    if len(cache["z"]) != 8 or len(np.unique(gid)) != 4:
        raise RuntimeError(f"load_cache round-trip: {len(cache['z'])} rows, "
                           f"{len(np.unique(gid))} groups")
    sub = cache["index"]
    for _, pair in sub.groupby("group_id"):
        if pair["action_id"].nunique() != 1 or set(pair["state_id"]) != {"A", "B"}:
            raise RuntimeError("load_cache pairing broken")
    print(f"verify OK: {n} rows, {len(probes)} bit-exact probes, pairing + "
          f"load_cache round-trip pass ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("convert", run_convert), ("verify", run_verify)):
        a = sub.add_parser(name)
        a.add_argument("--index-root", default=str(ZL_ART / "index"))
        a.add_argument("--zl-cache", default=str(ZL_ART / "cache"
                                                 / "wan21_t2v_1p3b"))
        a.add_argument("--out", default=str(DEFAULT_OUT))
        a.set_defaults(fn=fn)
        if name == "convert":
            a.add_argument("--zl-config", default=str(ZL_CONFIG))
            a.add_argument("--limit", type=int, default=None,
                           help="first N main rows (smoke); multiple of 4")
            a.add_argument("--chunk", type=int, default=256)
        else:
            a.add_argument("--probes", type=int, default=32)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
