"""Blocking cache gate: every split episode must have a latent of the exact
expected shape and a T5 embedding. Fails loudly listing what's missing -
the training dataset intentionally does NOT drop missing episodes silently
(the cloth code did, which can shift val statistics without anyone noticing).

Presence is checked for all 2,700; full tensor loads are spot-checked on a
deterministic random sample (default 40) plus norm-stats sanity.

  python projectile_smoke/data/verify_cache.py [--spot-checks 40]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projectile_smoke.data.episodes_proj import CACHE, file_stem  # noqa: E402

EXPECTED_SHAPE = (16, 11, 60, 104)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(CACHE))
    ap.add_argument("--spot-checks", type=int, default=40)
    args = ap.parse_args()
    cache = Path(args.cache_dir)

    with open(cache / "splits.json") as f:
        splits = json.load(f)
    keys = splits["train_episodes"] + splits["val_episodes"]

    with open(cache / "physics_vectors.json") as f:
        vectors = json.load(f)
    with open(cache / "norm_stats.json") as f:
        stats = json.load(f)
    dim = stats["phys_dim"]
    assert len(stats["mean"]) == dim and len(stats["std"]) == dim
    assert all(s > 0 for s in stats["std"]), "non-positive std in norm_stats"

    missing = []
    for k in keys:
        stem = file_stem(k)
        if not (cache / "latents" / f"{stem}.pt").exists():
            missing.append(f"latent:{k}")
        if not (cache / "t5_blind" / f"{stem}.pt").exists():
            missing.append(f"t5:{k}")
        if k not in vectors:
            missing.append(f"physvec:{k}")
        elif len(vectors[k]) != dim:
            missing.append(f"physdim:{k}")
    assert not missing, (f"{len(missing)} cache entries missing/bad, e.g. "
                         f"{missing[:10]} - run precompute_latents.py / "
                         f"precompute_t5.py / physics_vec.py")

    rng = random.Random(0)
    for k in rng.sample(keys, min(args.spot_checks, len(keys))):
        stem = file_stem(k)
        lat = torch.load(cache / "latents" / f"{stem}.pt",
                         map_location="cpu", weights_only=True)
        assert tuple(lat["latent"].shape) == EXPECTED_SHAPE, \
            f"{k}: latent shape {tuple(lat['latent'].shape)}"
        assert lat["latent"].dtype == torch.bfloat16
        assert torch.isfinite(lat["latent"].float()).all(), f"{k}: non-finite latent"
        t5 = torch.load(cache / "t5_blind" / f"{stem}.pt",
                        map_location="cpu", weights_only=True)
        assert t5["t5"].dim() == 2 and t5["t5"].shape[1] == 4096, t5["t5"].shape
        assert t5["t5"].shape[0] <= 512 - 8, "caption too long for 8 physics tokens"

    print(f"CACHE OK: {len(keys)} episodes complete "
          f"(latents {EXPECTED_SHAPE}, t5, {dim}-d physics vectors); "
          f"{min(args.spot_checks, len(keys))} spot-loads passed")


if __name__ == "__main__":
    main()
