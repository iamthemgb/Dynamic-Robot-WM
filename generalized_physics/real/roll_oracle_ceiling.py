"""Oracle ceiling for gap_wrong, measured from the cache before any training.

Under shared (tau, eps), a model that decoded the conditioning perfectly
would predict v = eps - x0(code); conditioning episode i on sibling j's code
then costs exactly mean((1-sigma)^2 ... ) ~ mean((x0_i - x0_j)^2) in flow-MSE
terms. So the sibling latent MSE IS the upper bound on the paired gap — the
"footprint dilution" number from the muffling diagnosis (problem 1), in the
same units as the flow loss. The ROI-restricted variant shows what the
ROI-weighted loss / ROI-restricted eval can see instead.

CPU-only; reads index.parquet + latents.f16.npy + roi.u8.npy directly.

  python -m generalized_physics.real.roll_oracle_ceiling \
         --cache .../cache/roll_wan21_vae [--groups 20]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    import pandas as pd

    cache = Path(args.cache)
    idx = pd.read_parquet(cache / "index.parquet")
    z = np.load(cache / "latents.f16.npy", mmap_mode="r")
    roi = np.load(cache / "roi.u8.npy", mmap_mode="r")

    rows = idx[idx["split"] == args.split]
    by_group = {g: list(sub.index) for g, sub in rows.groupby("group_id")
                if len(sub) >= 2}
    picked = sorted(by_group)[: args.groups]

    full, in_roi, roi_frac, power = [], [], [], []
    prev_first = None
    cross = []
    for g in picked:
        members = by_group[g]
        lat = [np.asarray(z[i], dtype=np.float32) for i in members]
        masks = [np.asarray(roi[i], dtype=bool) for i in members]
        power.append(float(np.mean([np.mean(x * x) for x in lat])))
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                d2 = (lat[a] - lat[b]) ** 2          # [C, Tz, Hz, Wz]
                union = masks[a] | masks[b]          # [Tz, Hz, Wz]
                full.append(float(d2.mean()))
                if union.any():
                    in_roi.append(float(d2[:, union].mean()))
                    roi_frac.append(float(union.mean()))
        if prev_first is not None:
            cross.append(float(((lat[0] - prev_first) ** 2).mean()))
        prev_first = lat[0]

    report = {
        "cache": str(cache), "split": args.split,
        "groups_measured": len(picked),
        "sibling_pairs": len(full),
        "latent_power_mean_sq": round(float(np.mean(power)), 6),
        "ceiling_gap_wrong_fullframe": {
            "mean": round(float(np.mean(full)), 6),
            "median": round(float(np.median(full)), 6),
            "min": round(float(np.min(full)), 6),
        },
        "ceiling_gap_wrong_roi": {
            "mean": round(float(np.mean(in_roi)), 6),
            "median": round(float(np.median(in_roi)), 6),
        },
        "roi_union_fraction_mean": round(float(np.mean(roi_frac)), 5),
        "crossgroup_fullframe_mean": (round(float(np.mean(cross)), 6)
                                      if cross else None),
        "note": "full-frame ceiling bounds the standard paired gap_wrong; "
                "roi ceiling bounds the ROI-restricted gap. Compare against "
                "the ~1e-4 paired-eval noise floor.",
    }
    print(json.dumps(report, indent=2))
    out = cache / "oracle_ceiling.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
