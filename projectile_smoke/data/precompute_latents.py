"""VAE-encode main.mp4 of every projectile episode into Wan latents.

76f@30fps 832x480 -> to_wan_frames -> 41f@16fps (no letterbox needed, videos
are native 832x480) -> WanVAE encode -> [16, 11, 60, 104] bf16.

The 3,000 latents already exist in wan_physics_cache (symlinked into the
smoke cache); this script is the idempotent refill for gaps/corruption -
existing files are skipped unless --overwrite.

  python projectile_smoke/data/precompute_latents.py \
      [--shard 0 --num-shards 4] [--overwrite]
"""

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("WAN21_ROOT", "/gpfs/radev/scratch/sous/mzl7/Wan2.1"))
sys.path.insert(0, str(REPO))

from projectile_smoke.mujoco_bridge import to_wan_frames  # noqa: E402
from wan.modules.vae import WanVAE  # noqa: E402

from projectile_smoke.data.episodes_proj import (  # noqa: E402
    CACHE, NUM_WAN_FRAMES, WAN_MODEL_DIR, file_stem, iter_episodes, video_path)

EXPECTED_SHAPE = (16, 11, 60, 104)


def encode_episode(vae, meta_path, num_frames=NUM_WAN_FRAMES):
    frames = to_wan_frames(str(video_path(meta_path)), num_frames=num_frames)
    video = torch.from_numpy(frames).float().div_(127.5).sub_(1.0)
    video = video.permute(3, 0, 1, 2).cuda()  # [C, T, H, W]
    with torch.no_grad():
        latent = vae.encode([video])[0]
    assert tuple(latent.shape) == EXPECTED_SHAPE, latent.shape
    return latent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(CACHE))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir) / "latents"
    out.mkdir(parents=True, exist_ok=True)

    episodes = [e for i, e in enumerate(iter_episodes())
                if i % args.num_shards == args.shard]
    todo = [(k, f, m) for k, f, m in episodes
            if args.overwrite or not (out / f"{file_stem(k)}.pt").exists()]
    print(f"shard {args.shard}/{args.num_shards}: "
          f"{len(todo)}/{len(episodes)} episodes to encode")
    if not todo:
        return

    vae = WanVAE(vae_pth=str(WAN_MODEL_DIR / "Wan2.1_VAE.pth"), device="cuda")

    done = failed = 0
    for key, fam, meta_path in todo:
        dst = out / f"{file_stem(key)}.pt"
        try:
            latent = encode_episode(vae, meta_path)
            tmp = dst.with_suffix(".tmp")
            torch.save({"key": key, "latent": latent.to(torch.bfloat16).cpu()}, tmp)
            tmp.rename(dst)
            done += 1
        except Exception as e:
            failed += 1
            print(f"FAILED {key}: {e}", file=sys.stderr)
        if done % 100 == 0:
            print(f"progress: done={done} failed={failed}", flush=True)

    print(f"finished: done={done} failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
