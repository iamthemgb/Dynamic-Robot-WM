"""Materialising the on-disk cache as the dict every phase already consumes.

``load_cache`` returns exactly the contract of
``data/counterfactual_dataset.build_cache`` -- z, delta, delta_valid, actions,
bin_index, vae_checkpoint_hash, temporal_stride, group_id, family, records,
events, rec_batch -- so ``GroupBatcher`` and all five phase modules run
untouched.

Latents stay fp16 in host RAM (12 GB at N=4000 for the Wan2.1 VAE) and are
gathered per batch to fp32 on the GPU. Auditing every consumer, the only
operations ever performed on ``cache["z"]`` are ``z[LongTensor]``, ``len(z)``
and ``z.shape[2]``, so a lazy view satisfying those three is a drop-in for a
real tensor -- and keeps a 12 GB array off the accelerator.
"""

import json
from pathlib import Path

import numpy as np
import torch

from ..models.metadata_records import MetadataRegistry, TypedRecord, \
    collate_records
from ..data.normalize_metadata import MetadataNormalizer


class RamLatents:
    """fp16 latents in host RAM, gathered to fp32 on the target device."""

    def __init__(self, array, device="cpu"):
        self.a = array
        self.device = device

    def __len__(self):
        return self.a.shape[0]

    @property
    def shape(self):
        return torch.Size(self.a.shape)

    def __getitem__(self, idx):
        i = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
        out = torch.from_numpy(np.ascontiguousarray(self.a[i]))
        return out.to(device=self.device, dtype=torch.float32)


class DerivedDelta:
    """z_t - z_{t-1} computed on gather; bin 0 zeroed. No extra storage."""

    def __init__(self, latents: RamLatents):
        self.z = latents

    def __len__(self):
        return len(self.z)

    @property
    def shape(self):
        return self.z.shape

    def __getitem__(self, idx):
        z = self.z[idx]
        d = torch.zeros_like(z)
        d[:, :, 1:] = z[:, :, 1:] - z[:, :, :-1]
        return d


def load_cache(cache_dir, device="cpu", split=None, registry=None,
               limit=None):
    """-> (cache dict, registry, normalizer).

    ``limit`` truncates BEFORE the latents are materialised into RAM (the full
    train split is ~7-12 GB), then drops group singletons so
    ``paired_batch``'s >=2-members-per-group contract still holds.
    """
    import pandas as pd

    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    idx = pd.read_parquet(cache_dir / "index.parquet")
    z_all = np.load(cache_dir / "latents.f16.npy", mmap_mode="r")
    actions = np.load(cache_dir / "actions.f32.npy")

    with open(cache_dir / "records.jsonl") as f:
        raw = [json.loads(line) for line in f]

    sel = np.arange(len(idx))
    if split is not None:
        sel = np.nonzero((idx["split"].values == split))[0]
    if limit is not None and limit < len(sel):
        sel = sel[:limit]
    if split is not None or limit is not None:
        # Drop group singletons so paired_batch's >=2-members-per-group
        # contract holds on ANY subset. Splits are group-atomic upstream, but
        # a val/test slice of a corpus with stragglers (or a truncation)
        # must never crash the batcher.
        gid = idx["group_id"].values[sel]
        counts = pd.Series(gid).value_counts()
        sel = sel[np.asarray(counts[gid] >= 2)]
        idx = idx.iloc[sel].reset_index(drop=True)
        raw = [raw[i] for i in sel]

    # GroupBatcher indexes its group list BY group id (`self.groups[gid[k]]`),
    # which silently mis-pairs episodes unless ids are dense 0..G-1. Any
    # subsetting above can leave holes, so always relabel densely.
    gid_raw = idx["group_id"].values
    relabel = {g: i for i, g in enumerate(sorted(set(gid_raw.tolist())))}
    idx = idx.assign(group_id=[relabel[g] for g in gid_raw])

    registry = registry or MetadataRegistry()
    normalizer = MetadataNormalizer.load(registry, cache_dir / "norm_stats.json")
    fitted = set(normalizer.stats)
    records = [[TypedRecord(r["key"], r["scope"], r["unit"], r["value"])
                for r in rs if r["key"] in fitted] for rs in raw]

    # Materialise fp16 latents into RAM once; mmap gathers would otherwise hit
    # NFS on every batch.
    z = np.ascontiguousarray(z_all[sel])
    latents = RamLatents(z, device=device)

    Tz = z.shape[2]
    delta_valid = torch.ones(len(idx), Tz, device=device)
    delta_valid[:, :2] = 0.0                   # causal first-chunk boundary

    n_control = manifest["n_control"]
    cache = {
        "z": latents,
        "delta": DerivedDelta(latents),
        "delta_valid": delta_valid,
        "actions": torch.from_numpy(actions[sel]).to(device),
        "bin_index": _bin_index(n_control, manifest["temporal_stride"]).to(device),
        "vae_checkpoint_hash": manifest.get("vae_checkpoint_hash", "unknown"),
        "temporal_stride": manifest["temporal_stride"],
        "group_id": torch.from_numpy(idx["group_id"].values.astype(np.int64)),
        "family": list(idx["leaf"]),
        "records": records,
        "events": [{"impact": (None if v < 0 else int(v))}
                   for v in idx["impact_frame"].values],
        "rec_batch": _to_device(
            collate_records(records, registry, normalizer), device),
        "prompt_id": torch.from_numpy(
            idx["prompt_id"].values.astype(np.int64)),
        "index": idx,
        "manifest": manifest,
    }
    roi_path = cache_dir / "roi.u8.npy"
    if roi_path.exists():
        # Latent-grid ball masks [N, Tz, Hz, Wz] uint8 (roll_encode_cache);
        # consumed only when cfg.train.roi_lambda > 0 or by ROI-restricted
        # eval — plain caches keep the exact previous cache dict.
        roi_all = np.load(roi_path, mmap_mode="r")
        cache["roi"] = torch.from_numpy(
            np.ascontiguousarray(roi_all[sel])).to(device)
    return cache, registry, normalizer


def _bin_index(n_control, stride):
    from ..data.cache_wan_latents import control_bin_index
    return control_bin_index(n_control, stride)


def _to_device(d, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}


def load_text_ctx(cache_dir, prompt_id_by_row, device="cuda"):
    """-> fn(batch_idx LongTensor) -> list of [L, 4096] bf16 context tensors.

    Embeddings are precomputed by ``t5_cache.py``; the 11.4 GB umT5 encoder is
    never loaded during training.
    """
    blob = torch.load(Path(cache_dir) / "t5" / "embeddings.pt",
                      map_location="cpu")
    table = {int(k): v.to(device=device, dtype=torch.bfloat16)
             for k, v in blob.items()}
    pid = prompt_id_by_row

    def text_ctx(batch_idx):
        rows = batch_idx.cpu().numpy() if torch.is_tensor(batch_idx) \
            else np.asarray(batch_idx)
        return [table[int(pid[int(r)])] for r in rows]

    return text_ctx
