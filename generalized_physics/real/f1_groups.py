"""Counterfactual grouping for ``GroupBatcher.paired_batch``.

Phase 1's ranking loss compares the correct record set against a "wrong" one
drawn from the SAME group, so that what differs between them is physics rather
than task identity or appearance. That requires >= 2 members per group.

**f1_10h has no natural groups.** Verified on F1a/block-0000: ``split_group_id``,
``scene_seed``, ``branch_seed`` and ``counterfactual_bundle_id`` are each
1500-unique over 1500 episodes, and ``meta/counterfactual_families.parquet``
has zero rows. So groups are synthesised as appearance buckets: within a bucket
the leaf, subfamily, variant, embodiment and background are fixed, and what
differs is the initial condition -- exactly the quantity the wrong code should
be wrong about.

Caveat to carry forward: ``variant`` and ``tool_type`` are perfectly
confounded in this corpus (``catch_retain``<->``franka_hand`` 750,
``catch_transport``<->``robotiq_2f85_thick_pad`` 750). Both are group keys so
the pairing is unaffected, but a variant effect must never be reported as an
embodiment effect.
"""

import numpy as np

GROUP_KEYS = ("leaf", "subfamily", "variant", "tool_type", "background_style")
MIN_GROUP = 2


def assign_groups(idx, keys=GROUP_KEYS):
    """Add an int64 ``group_id`` column; drop rows in under-filled groups.

    Returns (index_with_group_id, stats dict).
    """
    idx = idx.copy()
    combo = idx[list(keys)].astype(str).agg("|".join, axis=1)
    sizes = combo.value_counts()
    keep = combo.map(sizes) >= MIN_GROUP
    dropped = int((~keep).sum())
    idx = idx[keep].copy()
    combo = combo[keep]
    labels = {c: i for i, c in enumerate(sorted(combo.unique()))}
    idx["group_id"] = combo.map(labels).astype(np.int64)
    stats = {"n_groups": len(labels), "dropped_singletons": dropped,
             "min_size": int(sizes[sizes >= MIN_GROUP].min()),
             "max_size": int(sizes.max()), "keys": list(keys)}
    return idx, stats


def stratified_sample(idx, n, seed=0,
                      strata=("leaf", "tool_type", "actual_outcome")):
    """Take ~n episodes, balanced across strata, preserving whole groups' size.

    Sampling is proportional-with-a-floor rather than strictly equal-quota:
    ``contact_failure`` has only ~430 eligible episodes corpus-wide, so an
    equal quota would either starve the common cells or over-request a rare
    one. Every stratum contributes at least its proportional share.
    """
    if n is None or n >= len(idx):
        return idx.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    frac = n / len(idx)
    picked = []
    for _, grp in idx.groupby(list(strata), sort=True):
        take = min(len(grp), max(1, int(round(len(grp) * frac))))
        picked.append(grp.iloc[rng.permutation(len(grp))[:take]])
    import pandas as pd
    out = pd.concat(picked).sort_values(["leaf", "block", "episode_index"])
    if len(out) > n:
        out = out.iloc[rng.permutation(len(out))[:n]].sort_values(
            ["leaf", "block", "episode_index"])
    return out.reset_index(drop=True)


def summarize(idx):
    g = idx.groupby("group_id").size()
    return {"episodes": int(len(idx)), "groups": int(g.size),
            "median_group": float(g.median()), "min_group": int(g.min()),
            "max_group": int(g.max()),
            "splits": idx["split"].value_counts().to_dict()}
