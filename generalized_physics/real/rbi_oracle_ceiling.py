"""Oracle ceilings for the rbi cache, measured before any training.

Port of ``roll_oracle_ceiling`` to the main-view rbi cache built by
``rbi_convert_cache``, reporting BOTH pairings:

  * velocity pairs (``group_id``): sibling latent MSE full-frame and
    ROI-union-restricted — the upper bound on stage 1's paired
    ``gap_wrong`` in flow-loss units (rbi_phase1_problems_and_fixes.tex,
    problem A1 fix 1);
  * action pairs (``action_group_id``): full-frame only — the bound on
    stage 2's ``gap_wrong_action`` (the arm differs across the pair, so no
    ball-ROI restriction applies).

Report-only by design: the corpus is known to sit ~22x below the passive
reference in the velocity ROI (0.041 vs 0.892, measured 2026-08-03 on the
pooled-view zl664 cache) and the user chose to proceed regardless; this
number goes into the summary so a gate failure is attributable.

CPU-only; reads index.parquet + latents.f16.npy + roi.u8.npy directly.

  python -m generalized_physics.real.rbi_oracle_ceiling \
         [--cache .../cache/rbi_wan21_vae] [--groups 200]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .paths import OUT_ROOT

PASSIVE_REFERENCE = {
    "cache": "mzl7/physics_wan/cache/roll_wan21_vae (passing campaign)",
    "latent_power_mean_sq": 0.655441,
    "ceiling_gap_wrong_fullframe_mean": 0.016213,
    "ceiling_gap_wrong_roi_mean": 0.891632,
    "roi_union_fraction_mean": 0.01373,
}


def _pair_stats(rows, z, roi, id_col, with_roi, max_pairs):
    full, in_roi, roi_frac, power = [], [], [], []
    by_contrast = {}
    cross, prev_ref = [], None
    picked = sorted(rows[id_col].unique())[:max_pairs]
    by_id = {g: sub for g, sub in rows.groupby(id_col)}
    for n, g in enumerate(picked):
        sub = by_id[g]
        if len(sub) != 2:
            raise RuntimeError(f"{id_col} {g}: {len(sub)} members")
        i, j = (int(k) for k in sub.index)
        a = np.asarray(z[i], dtype=np.float32)
        b = np.asarray(z[j], dtype=np.float32)
        power.append(float((np.mean(a * a) + np.mean(b * b)) / 2))
        d2 = (a - b) ** 2                            # [C, Tz, Hz, Wz]
        full.append(float(d2.mean()))
        by_contrast.setdefault(str(sub["contrast_type"].iloc[0]),
                               []).append(float(d2.mean()))
        if with_roi:
            union = np.asarray(roi[i], dtype=bool) | np.asarray(roi[j],
                                                                dtype=bool)
            if union.any():
                in_roi.append(float(d2[:, union].mean()))
                roi_frac.append(float(union.mean()))
        if prev_ref is not None:
            cross.append(float(((a - prev_ref) ** 2).mean()))
        prev_ref = a
        if (n + 1) % 100 == 0:
            print(f"  {id_col}: {n + 1}/{len(picked)}", flush=True)

    out = {
        "pairs_measured": len(picked),
        "latent_power_mean_sq": round(float(np.mean(power)), 6),
        "ceiling_gap_wrong_fullframe": {
            "mean": round(float(np.mean(full)), 6),
            "median": round(float(np.median(full)), 6),
            "min": round(float(np.min(full)), 6),
        },
        "crossgroup_fullframe_mean": (round(float(np.mean(cross)), 6)
                                      if cross else None),
        "fullframe_by_contrast_type": {
            c: round(float(np.mean(v)), 6)
            for c, v in sorted(by_contrast.items())},
    }
    if with_roi and in_roi:
        out["ceiling_gap_wrong_roi"] = {
            "mean": round(float(np.mean(in_roi)), 6),
            "median": round(float(np.median(in_roi)), 6),
        }
        out["roi_union_fraction_mean"] = round(float(np.mean(roi_frac)), 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(OUT_ROOT / "cache"
                                           / "rbi_wan21_vae"))
    ap.add_argument("--groups", type=int, default=200,
                    help="max pairs measured per pairing")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    import pandas as pd

    cache = Path(args.cache)
    idx = pd.read_parquet(cache / "index.parquet")
    manifest = json.loads((cache / "manifest.json").read_text())
    z = np.load(cache / "latents.f16.npy", mmap_mode="r")
    roi = np.load(cache / "roi.u8.npy", mmap_mode="r")

    rows = idx[idx["split"] == args.split]
    report = {
        "cache": str(cache), "split": args.split, "view": "main",
        "donor_rule_stage1": manifest.get("donor_rule_stage1"),
        "donor_rule_stage2": manifest.get("donor_rule_stage2"),
        "velocity_pairs": _pair_stats(rows, z, roi, "group_id",
                                      with_roi=True, max_pairs=args.groups),
        "action_pairs": _pair_stats(rows, z, roi, "action_group_id",
                                    with_roi=False, max_pairs=args.groups),
        "passive_reference": PASSIVE_REFERENCE,
        "note": "velocity_pairs bounds stage 1's paired gap_wrong "
                "(full-frame and ROI-restricted); action_pairs bounds "
                "stage 2's gap_wrong_action (full-frame — the arm differs, "
                "the ball ROI does not apply). Compare against the ~1e-4 "
                "paired-eval noise floor and the passive_reference block. "
                "Report-only: the campaign proceeds regardless.",
    }
    print(json.dumps(report, indent=2))
    out = cache / "oracle_ceiling.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
