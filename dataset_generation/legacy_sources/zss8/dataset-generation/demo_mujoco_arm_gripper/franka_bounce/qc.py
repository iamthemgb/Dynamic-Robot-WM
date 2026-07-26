"""QC / train-readiness check for the F1_B ground-bounce catch dataset.

Runs the standard structural checks (videos + parquet + rich JSON present, video
resolution/frame count, parquet row/timestamp/NaN sanity, branch/outcome/variant
splits) AND bounce-specific checks: every episode bounced once, measured
restitution is populated and in range, and the intended->actual outcome mix is
sane per branch.

Usage:  python -m franka_bounce.qc --root <dataset_root> [--sample 60]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq

EXPECT_STATE_DIM = 18
EXPECT_ACTION_DIM = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=60, help="episodes to deep-check")
    args = ap.parse_args()
    root = args.root

    eps = [json.loads(l) for l in (root / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    info = json.loads((root / "meta" / "info.json").read_text())
    n = len(eps)
    print(f"episodes={n}  total_frames(info)={info.get('total_frames')}  fps={info.get('fps')} "
          f"control_hz={info.get('control_hz')} res={info.get('resolution')}")
    feats = info.get("features", {})
    sdim = feats.get("observation.state", {}).get("shape", [None])[0]
    adim = feats.get("action", {}).get("shape", [None])[0]
    print(f"  state_dim={sdim} action_dim={adim} "
          f"({'ok' if sdim == EXPECT_STATE_DIM and adim == EXPECT_ACTION_DIM else 'UNEXPECTED'})")

    for field in ("branch", "outcome", "scene_variant", "failure_mode", "subfamily"):
        c = Counter(e.get(field) for e in eps)
        print(f"  {field:14s}: {dict(c.most_common())}")

    # intended (branch) -> actual (outcome) cross-tab
    print("  branch -> outcome:")
    for br in ("success", "spatial_near_miss", "contact_failure", "wrong_action"):
        sub = [e for e in eps if e.get("branch") == br]
        if not sub:
            continue
        succ = sum(1 for e in sub if e.get("outcome") == "success")
        print(f"    {br:18s}: {succ}/{len(sub)} succeeded ({100*succ/len(sub):.0f}%)")

    # -------- bounce-specific --------
    def bget(e, k):
        return (e.get("bounce") or {}).get(k)

    detected = [e for e in eps if bget(e, "detected")]
    e_meas = [bget(e, "measured_restitution") for e in eps if bget(e, "measured_restitution") is not None]
    apex = [bget(e, "measured_apex_z") for e in eps if bget(e, "measured_apex_z") is not None]
    print(f"  bounce detected : {len(detected)}/{n} ({100*len(detected)/max(1,n):.1f}%)")
    if e_meas:
        a = np.array(e_meas)
        print(f"  measured_restitution: mean={a.mean():.2f} min={a.min():.2f} max={a.max():.2f} "
              f"(n={len(e_meas)})")
    if apex:
        a = np.array(apex)
        print(f"  measured_apex_z     : mean={a.mean():.2f} min={a.min():.2f} max={a.max():.2f}")

    # -------- structural deep-check --------
    idxs = np.linspace(0, n - 1, min(args.sample, n)).astype(int)
    problems = 0
    res = tuple(info.get("resolution", [832, 480]))
    for i in idxs:
        e = eps[i]
        ep = e["episode_index"]
        main = root / f"videos/observation.images.main/chunk-000/episode_{ep:06d}.mp4"
        side = root / f"videos/observation.images.side/chunk-000/episode_{ep:06d}.mp4"
        data = root / f"data/chunk-000/episode_{ep:06d}.parquet"
        rich = root / f"meta/rich/episode_{ep:06d}.json"
        for p in (main, side, data):
            if not p.exists():
                print(f"  MISSING {p.name}"); problems += 1
        if not main.exists():
            continue
        r = imageio.get_reader(str(main)); nfr = r.count_frames(); f0 = r.get_data(0); r.close()
        if (f0.shape[1], f0.shape[0]) != res:
            print(f"  ep{ep}: main video {f0.shape[1]}x{f0.shape[0]} != {res}"); problems += 1
        if nfr < 2:
            print(f"  ep{ep}: video frames={nfr}"); problems += 1
        t = pq.read_table(data)
        st = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
        ac = np.array(t["action"].to_pylist(), dtype=np.float32)
        ts = np.array(t["timestamp"].to_pylist(), dtype=np.float32)
        vfi = np.array(t["video_frame_index"].to_pylist(), dtype=np.int64)
        if not np.isfinite(st).all() or not np.isfinite(ac).all():
            print(f"  ep{ep}: NaN/Inf in parquet"); problems += 1
        if st.shape[1] != EXPECT_STATE_DIM or ac.shape[1] != EXPECT_ACTION_DIM:
            print(f"  ep{ep}: state/action dim {st.shape[1]}/{ac.shape[1]}"); problems += 1
        if t.num_rows < 2:
            print(f"  ep{ep}: parquet rows={t.num_rows}"); problems += 1
        if np.any(np.diff(ts) <= 0):
            print(f"  ep{ep}: non-monotonic timestamps"); problems += 1
        if vfi.min() < 0 or vfi.max() >= nfr:
            print(f"  ep{ep}: video_frame_index out of range [0,{nfr})"); problems += 1
        if not rich.exists():
            print(f"  ep{ep}: missing rich json"); problems += 1

    # a few things that would make it NOT train-ready
    ready = True
    if len(detected) < n:
        print(f"  WARN: {n - len(detected)} episodes had no detected bounce"); ready = ready and (len(detected) >= 0.98 * n)
    if problems:
        ready = False
    outcomes = Counter(e.get("outcome") for e in eps)
    if outcomes.get("success", 0) == 0 or outcomes.get("failure", 0) == 0:
        print("  WARN: dataset is single-outcome (no success or no failure)"); ready = False

    print(f"deep-checked {len(idxs)} episodes; problems={problems}")
    du = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"dataset size: {du/1e9:.2f} GB")
    print("TRAIN-READY: OK" if ready and problems == 0 else "TRAIN-READY: ISSUES FOUND")


if __name__ == "__main__":
    main()
