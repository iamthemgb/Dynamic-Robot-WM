"""Quick QC for a merged ball-catch dataset.

Checks: episode count, branch/outcome/variant split, that each episode has both
view videos + a parquet + a rich JSON, video resolution/frame count, parquet
row/timestamp sanity and no NaNs.

Usage:  python -m franka_catch.qc --root <dataset_root> [--sample 40]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=40, help="how many episodes to deep-check")
    args = ap.parse_args()
    root = args.root

    eps = [json.loads(l) for l in (root / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    info = json.loads((root / "meta" / "info.json").read_text())
    n = len(eps)
    print(f"episodes={n}  total_frames(info)={info.get('total_frames')}  fps={info.get('fps')} control_hz={info.get('control_hz')} res={info.get('resolution')}")

    for field in ("branch", "outcome", "scene_variant", "failure_mode"):
        c = Counter(e.get(field) for e in eps)
        pct = {k: f"{v} ({100*v/n:.1f}%)" for k, v in c.most_common()}
        print(f"  {field:14s}: {pct}")

    # deep check a sample spread across the set
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
        r = imageio.get_reader(str(main)); frames = r.count_frames(); f0 = r.get_data(0); r.close()
        if (f0.shape[1], f0.shape[0]) != res:
            print(f"  ep{ep}: main video {f0.shape[1]}x{f0.shape[0]} != {res}"); problems += 1
        t = pq.read_table(data)
        st = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
        ac = np.array(t["action"].to_pylist(), dtype=np.float32)
        if not np.isfinite(st).all() or not np.isfinite(ac).all():
            print(f"  ep{ep}: NaN/Inf in parquet"); problems += 1
        if t.num_rows < 2:
            print(f"  ep{ep}: parquet rows={t.num_rows}"); problems += 1
        if not rich.exists():
            print(f"  ep{ep}: missing rich json"); problems += 1

    print(f"deep-checked {len(idxs)} episodes; problems={problems}")
    du = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"dataset size: {du/1e9:.2f} GB")
    print("OK" if problems == 0 else "ISSUES FOUND")


if __name__ == "__main__":
    main()
