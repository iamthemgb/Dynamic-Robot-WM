"""2,400 train / 300 val episodes, stratified by family x failure_mode.

Sampling unit is the physics-identity CLUSTER (episodes sharing exact sampled
physics, i.e. opposite-camera twins if they existed). A hard assertion - not
convention - fails the script if any cluster straddles the split: twin leakage
would inflate every val number with near-duplicates. On the current dataset
each cluster is a singleton (verified: no shared physics anywhere), so the
assertion is trivially satisfied but stays load-bearing against regenerated
data. Remaining 300 episodes are dropped entirely (epoch-count cap: 2,000
steps x effective batch 8 / 2,400 episodes ~= 6.7 epochs).

  python projectile_smoke/data/splits.py [--seed 0]
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projectile_smoke.data.episodes_proj import (  # noqa: E402
    CACHE, iter_episodes, load_metadata, physics_identity)

N_TRAIN, N_VAL = 2400, 300


def largest_remainder(sizes, total_target, grand_total):
    """Integer per-stratum quotas summing exactly to total_target."""
    quotas_f = {s: n * total_target / grand_total for s, n in sizes.items()}
    quotas = {s: int(q) for s, q in quotas_f.items()}
    short = total_target - sum(quotas.values())
    assert short >= 0
    by_rem = sorted(sizes, key=lambda s: (-(quotas_f[s] - quotas[s]), s))
    for s in by_rem[:short]:
        quotas[s] += 1
    return quotas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(CACHE))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # cluster episodes by exact physics identity
    ident_to_cluster, clusters = {}, {}          # identity -> cid, cid -> [keys]
    cluster_stratum, key_cluster = {}, {}
    for key, fam, meta_path in iter_episodes():
        meta = load_metadata(meta_path)
        ident = physics_identity(meta)
        cid = ident_to_cluster.setdefault(ident, f"c{len(ident_to_cluster):05d}")
        clusters.setdefault(cid, []).append(key)
        key_cluster[key] = cid
        stratum = (fam, meta.get("failure_mode", "none"))
        assert cluster_stratum.setdefault(cid, stratum) == stratum, \
            f"cluster {cid} spans strata - stratify by cluster impossible"

    n_episodes = sum(len(v) for v in clusters.values())
    by_stratum = collections.defaultdict(list)
    for cid, stratum in cluster_stratum.items():
        by_stratum[stratum].append(cid)

    # allocate EPISODE counts per stratum, then greedily take whole clusters
    sizes = {s: sum(len(clusters[c]) for c in cids)
             for s, cids in by_stratum.items()}
    val_quota = largest_remainder(sizes, N_VAL, n_episodes)
    train_quota = largest_remainder(sizes, N_TRAIN, n_episodes)

    train, val, dropped = [], [], []
    for s in sorted(by_stratum):
        cids = sorted(by_stratum[s])
        random.Random(f"{args.seed}:{s}").shuffle(cids)
        got_val = got_train = 0
        for cid in cids:
            eps = sorted(clusters[cid])
            if got_val < val_quota[s]:
                val += eps
                got_val += len(eps)
            elif got_train < train_quota[s]:
                train += eps
                got_train += len(eps)
            else:
                dropped += eps
        assert got_val == val_quota[s] and got_train == train_quota[s], (
            f"stratum {s}: cluster sizes prevented exact quota "
            f"(val {got_val}/{val_quota[s]}, train {got_train}/{train_quota[s]})")

    assert len(val) == N_VAL and len(train) == N_TRAIN, (len(train), len(val))

    # THE assertion: no physics identity crosses the split boundary
    split_of = {k: "train" for k in train}
    split_of.update({k: "val" for k in val})
    for cid, eps in clusters.items():
        splits_hit = {split_of.get(k, "dropped") for k in eps}
        assert len(splits_hit) == 1, (
            f"TWIN LEAKAGE: cluster {cid} straddles splits {splits_hit}: {eps}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = {
        "seed": args.seed,
        "train_episodes": sorted(train),
        "val_episodes": sorted(val),
        "dropped_episodes": sorted(dropped),
        "val_clusters": {k: key_cluster[k] for k in sorted(val)},
        "n_clusters_total": len(clusters),
        "n_val_clusters": len({key_cluster[k] for k in val}),
        "strata_sizes": {f"{f}|{m}": n for (f, m), n in sorted(sizes.items())},
    }
    with open(out / "splits.json", "w") as f:
        json.dump(split, f, indent=1)
    print(f"train {len(train)} / val {len(val)} / dropped {len(dropped)} episodes; "
          f"{split['n_val_clusters']} independent val clusters "
          f"(singletons on current data); twin-leakage assertion PASSED")
    for s in sorted(by_stratum):
        v = val_quota[s]; t = train_quota[s]
        print(f"  {s[0]:38s} {s[1]:18s} val={v:3d} train={t:4d}")


if __name__ == "__main__":
    main()
