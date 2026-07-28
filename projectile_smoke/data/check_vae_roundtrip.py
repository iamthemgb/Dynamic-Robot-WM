"""GO/NO-GO (GPU): VAE round-trip on cached latents + provenance check.

Decodes N cached latents and compares them frame-by-frame against a fresh
to_wan_frames() read of the CURRENT videos under the project dir. This both
(a) eyeballs VAE quality (side-by-side mp4s) and (b) proves the cached
latents - built before the dataset moved from scratch to project storage -
match today's videos (a stale/foreign latent shows up as garbage PSNR).

  python projectile_smoke/data/check_vae_roundtrip.py [--n 5]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("WAN21_ROOT", "/gpfs/radev/scratch/sous/mzl7/Wan2.1"))
sys.path.insert(0, str(REPO))

from projectile_smoke.mujoco_bridge import to_wan_frames  # noqa: E402
from wan.modules.vae import WanVAE  # noqa: E402

from projectile_smoke.data.episodes_proj import (  # noqa: E402
    CACHE, NUM_WAN_FRAMES, WAN_MODEL_DIR, file_stem, iter_episodes, video_path)

PSNR_MIN = 24.0  # dB, VAE recon of clean renders is typically ~28-35


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out-dir", default=str(CACHE / "vae_roundtrip"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PASS").unlink(missing_ok=True)

    episodes = list(iter_episodes())
    sampled = episodes[:: max(len(episodes) // args.n, 1)][: args.n]

    vae = WanVAE(vae_pth=str(WAN_MODEL_DIR / "Wan2.1_VAE.pth"), device="cuda")

    results, ok = [], True
    for key, fam, meta_path in sampled:
        lat = torch.load(CACHE / "latents" / f"{file_stem(key)}.pt",
                         map_location="cpu", weights_only=True)["latent"]
        with torch.no_grad():
            dec = vae.decode([lat.float().cuda()])[0]  # [C, T, H, W] in [-1,1]
        dec = ((dec.clamp(-1, 1) + 1) * 127.5).round().byte()
        dec = dec.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]

        src = to_wan_frames(str(video_path(meta_path)),
                            num_frames=NUM_WAN_FRAMES)
        assert src.shape == dec.shape, (src.shape, dec.shape)
        val = float(psnr(src, dec))
        ok &= val >= PSNR_MIN
        results.append({"key": key, "psnr_db": round(val, 2)})
        print(f"{key:45s} PSNR {val:6.2f} dB "
              f"{'OK' if val >= PSNR_MIN else 'FAIL'}")

        side = np.concatenate([src, dec], axis=2)  # side-by-side
        iio.imwrite(out_dir / f"{file_stem(key)}_roundtrip.mp4", side, fps=16)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"psnr_min": PSNR_MIN, "results": results, "pass": ok},
                  f, indent=1)
    if ok:
        (out_dir / "PASS").write_text("vae roundtrip passed\n")
        print(f"VAE ROUND-TRIP: PASS (eyeball mp4s in {out_dir})")
    else:
        print("VAE ROUND-TRIP: FAIL - cached latents do not reconstruct "
              "today's videos; re-run precompute_latents.py --overwrite")
        sys.exit(1)


if __name__ == "__main__":
    main()
