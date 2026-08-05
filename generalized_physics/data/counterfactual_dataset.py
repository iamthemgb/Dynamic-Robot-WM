"""Counterfactual-group dataset over cached latents (plan: data design).

The counterfactual group is the primary training unit: paired correct/wrong
records always come from the SAME group (shared prefix, appearance, action
script; differing physics), so wrong-code training isolates physics rather
than task identity.

build_cache() generates the synthetic families, encodes them with the
frozen VAE, fits the metadata normalizer on the TRAIN split only, and
collates every episode's records into one padded/masked batch tensor so a
row gather yields any episode's (or any same-group wrong variant's)
records.
"""

import numpy as np
import torch

from ..models.metadata_records import MetadataRegistry, collate_records
from .cache_wan_latents import cache_episodes
from .normalize_metadata import MetadataNormalizer
from .synthetic_env import make_dataset


def build_cache(cfg, vae, seed=0, n_groups=None, change_frame=None,
                normalizer=None, registry=None):
    """Returns (cache dict, registry, normalizer). Pass an already-fitted
    normalizer for eval/change-point caches so train statistics are reused."""
    registry = registry or MetadataRegistry()
    groups = make_dataset(
        n_groups or cfg.n_train_groups, seed=seed,
        families=cfg.env.families, change_frame=change_frame,
        n_frames=cfg.env.n_frames, image_size=cfg.env.image_size,
        variants=cfg.variants_per_group, dt=cfg.env.dt)
    episodes = [e for g in groups for e in g]
    cache = cache_episodes(vae, episodes)
    cache["group_id"] = torch.tensor(
        [gi for gi, g in enumerate(groups) for _ in g])
    cache["family"] = [e.family for e in episodes]
    cache["records"] = [e.records for e in episodes]
    cache["events"] = [e.events for e in episodes]

    if normalizer is None:
        normalizer = MetadataNormalizer(registry).fit(cache["records"])
    cache["rec_batch"] = collate_records(cache["records"], registry,
                                         normalizer)
    if change_frame is not None:
        cache["records_after"] = [e.records_after for e in episodes]
        cache["rec_batch_after"] = collate_records(
            cache["records_after"], registry, normalizer)
        cache["change_bin"] = 1 + change_frame // cache["temporal_stride"]
    return cache, registry, normalizer


def _gather_rec(rec_batch, idx):
    return {k: v[idx] for k, v in rec_batch.items()}


class GroupBatcher:
    def __init__(self, cache, seed=0):
        self.c = cache
        self.rng = np.random.default_rng(seed)
        gid = cache["group_id"].numpy()
        self.groups = [np.nonzero(gid == g)[0] for g in np.unique(gid)]

    def _gather(self, idx):
        c = self.c
        i = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        b = {
            "z": c["z"][i], "delta": c["delta"][i],
            "delta_valid": c["delta_valid"][i], "actions": c["actions"][i],
            "bin_index": c["bin_index"], "rec": _gather_rec(c["rec_batch"], i),
            "event_bins": torch.tensor(
                [(c["events"][k]["impact"] or 0) // c["temporal_stride"]
                 for k in i.tolist()]),
            "idx": i,
        }
        if "roi" in c:
            b["roi"] = c["roi"][i]
        return b

    def episode_batch(self, batch_size):
        idx = self.rng.integers(0, len(self.c["z"]), size=batch_size)
        return self._gather(idx)

    def paired_batch(self, batch_size):
        """Adds rec_wrong: the record set of ANOTHER variant of the same
        group (same appearance/script, different physics)."""
        batch = self.episode_batch(batch_size)
        gid = self.c["group_id"].numpy()
        wrong = []
        for k in batch["idx"].tolist():
            members = self.groups[gid[k]]
            wrong.append(int(self.rng.choice(members[members != k])))
        batch["rec_wrong"] = _gather_rec(self.c["rec_batch"],
                                         torch.tensor(wrong))
        return batch

    def window(self, batch, W, end=None):
        """Slice the trailing (or end-anchored) W latent bins plus the
        controls belonging to them, preserving causality."""
        Tz = batch["z"].shape[2]
        end = Tz if end is None else end
        start = max(0, end - W)
        keep = (batch["bin_index"] >= start) & (batch["bin_index"] < end)
        return {
            "z": batch["z"][:, :, start:end],
            "delta": batch["delta"][:, :, start:end],
            "delta_valid": batch["delta_valid"][:, start:end],
            "actions": batch["actions"][:, keep],
            "bin_index": batch["bin_index"][keep] - start,
        }
