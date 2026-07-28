"""Dataset over the smoke cache: latent + T5 embedding + normalized physics.

Unlike the cloth dataset, missing cache entries are a HARD ERROR, not a
silent drop - a quietly shrinking val set shifts every eval statistic.
run verify_cache.py / the precompute scripts if init fails.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

LATENT_SHAPE = (16, 11, 60, 104)


def _stem(key):
    return key.replace("/", "__", 1)


class SmokeDataset(Dataset):

    def __init__(self, cache_dir, split):
        cache = Path(cache_dir)
        self.latent_dir = cache / "latents"
        self.t5_dir = cache / "t5_blind"

        with open(cache / "splits.json") as f:
            splits = json.load(f)
        self.keys = splits[f"{split}_episodes"]
        self.clusters = (splits["val_clusters"] if split == "val" else None)

        with open(cache / "physics_vectors.json") as f:
            vectors = json.load(f)
        with open(cache / "norm_stats.json") as f:
            stats = json.load(f)
        self.phys_dim = stats["phys_dim"]
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)

        missing = [k for k in self.keys
                   if not (self.latent_dir / f"{_stem(k)}.pt").exists()
                   or not (self.t5_dir / f"{_stem(k)}.pt").exists()
                   or k not in vectors]
        assert not missing, (
            f"[dataset:{split}] {len(missing)} episodes missing from cache "
            f"(e.g. {missing[:5]}) - refusing to silently drop them")

        self.phys = {k: (torch.tensor(vectors[k], dtype=torch.float32) - mean) / std
                     for k in self.keys}

    def __len__(self):
        return len(self.keys)

    def phys_matrix(self):
        """[N, phys_dim] normalized vectors in key order (for InfoNCE batches)."""
        return torch.stack([self.phys[k] for k in self.keys])

    def __getitem__(self, idx):
        k = self.keys[idx]
        latent = torch.load(self.latent_dir / f"{_stem(k)}.pt",
                            map_location="cpu", weights_only=True)["latent"]
        assert tuple(latent.shape) == LATENT_SHAPE, (k, latent.shape)
        t5 = torch.load(self.t5_dir / f"{_stem(k)}.pt",
                        map_location="cpu", weights_only=True)["t5"]
        return {"key": k, "latent": latent.float(), "t5": t5.float(),
                "phys": self.phys[k]}


def collate(batch):
    return {
        "keys": [b["key"] for b in batch],
        "latent": torch.stack([b["latent"] for b in batch]),
        "t5": [b["t5"] for b in batch],  # variable length, stays a list
        "phys": torch.stack([b["phys"] for b in batch]),
    }
