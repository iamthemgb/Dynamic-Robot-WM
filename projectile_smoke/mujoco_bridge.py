#!/usr/bin/env python
"""Bridge between MuJoCo rollout datasets and Wan2.1.

Converts rollout videos (e.g. 640x480 @ 24fps per-camera mp4s) into
Wan2.1-compatible clips: 832x480, 16 fps, 4n+1 frames. Also extracts
first/last frames for I2V / FLF2V conditioning.

Usage:
  # Prepare one rollout video (writes <out>/clip.mp4, first.png, last.png)
  python mujoco_bridge.py prepare \
      --video ~/scratch/<dataset>/videos/<episode>__main.mp4 \
      --out ~/scratch/wan_inputs/<episode>

  # Prepare every *__main.mp4 in a dataset
  python mujoco_bridge.py prepare-dataset \
      --dataset ~/scratch/franka_cloth_kitchen_dataset_2026-07-07 \
      --cam main --out ~/scratch/wan_inputs/kitchen_main

  # Text-to-video with the local 1.3B checkpoint
  python mujoco_bridge.py t2v --prompt "a Franka arm folding a t-shirt" \
      --out ~/scratch/wan_outputs/test.mp4
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

WAN_ROOT = Path(os.environ.get("WAN21_ROOT", "/gpfs/radev/scratch/sous/mzl7/Wan2.1"))
MODEL_DIR = Path(os.environ.get(
    "WAN_MODEL_DIR",
    Path.home() / "scratch/wan_models/Wan2.1-T2V-1.3B"))
WAN_SIZE = (832, 480)  # (W, H) for the 1.3B 480P models
WAN_FPS = 16


def to_wan_frames(video_path, num_frames=81, src_fps=None):
    """Read a rollout video, resample to WAN_FPS, letterbox to 832x480,
    and trim/pad to a 4n+1 frame count."""
    frames = iio.imread(video_path)  # (T, H, W, 3)
    meta = iio.immeta(video_path)
    src_fps = src_fps or meta.get("fps", 24.0)

    # temporal resample to WAN_FPS
    t_src = len(frames)
    t_dst = max(1, int(round(t_src * WAN_FPS / src_fps)))
    idx = np.linspace(0, t_src - 1, t_dst).round().astype(int)
    frames = frames[idx]

    # trim to 4n+1
    n = min(len(frames), num_frames)
    n = (n - 1) // 4 * 4 + 1
    frames = frames[:n]

    # letterbox to 832x480
    import cv2
    W, H = WAN_SIZE
    out = np.zeros((len(frames), H, W, 3), dtype=np.uint8)
    h, w = frames.shape[1:3]
    scale = min(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    for i, f in enumerate(frames):
        out[i, y0:y0 + nh, x0:x0 + nw] = cv2.resize(f, (nw, nh))
    return out


def cmd_prepare(args):
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = to_wan_frames(Path(args.video).expanduser(), args.frames)
    iio.imwrite(out_dir / "clip.mp4", frames, fps=WAN_FPS)
    iio.imwrite(out_dir / "first.png", frames[0])
    iio.imwrite(out_dir / "last.png", frames[-1])
    print(f"{args.video} -> {out_dir} ({len(frames)} frames @ {WAN_FPS}fps, "
          f"{WAN_SIZE[0]}x{WAN_SIZE[1]})")


def cmd_prepare_dataset(args):
    dataset = Path(args.dataset).expanduser()
    videos = sorted((dataset / "videos").glob(f"*__{args.cam}.mp4"))
    if not videos:
        sys.exit(f"no *__{args.cam}.mp4 videos under {dataset}/videos")
    for v in videos:
        episode = v.stem.replace(f"__{args.cam}", "")
        cmd_prepare(argparse.Namespace(
            video=str(v), out=str(Path(args.out) / episode),
            frames=args.frames))
    print(f"prepared {len(videos)} episodes -> {args.out}")


def cmd_t2v(args):
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(WAN_ROOT / "generate.py"),
        "--task", "t2v-1.3B",
        "--size", f"{WAN_SIZE[0]}*{WAN_SIZE[1]}",
        "--ckpt_dir", str(MODEL_DIR),
        "--prompt", args.prompt,
        "--frame_num", str(args.frames),
        "--sample_steps", str(args.steps),
        "--sample_guide_scale", "6",
        "--sample_shift", "8",
        "--offload_model", "True",
        "--t5_cpu",
        "--save_file", str(out),
    ]
    subprocess.run(cmd, cwd=WAN_ROOT, check=True)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="convert one rollout video")
    sp.add_argument("--video", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--frames", type=int, default=81)
    sp.set_defaults(fn=cmd_prepare)

    sp = sub.add_parser("prepare-dataset", help="convert a whole dataset")
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--cam", default="main")
    sp.add_argument("--out", required=True)
    sp.add_argument("--frames", type=int, default=81)
    sp.set_defaults(fn=cmd_prepare_dataset)

    sp = sub.add_parser("t2v", help="text-to-video with local 1.3B model")
    sp.add_argument("--prompt", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--frames", type=int, default=81)
    sp.add_argument("--steps", type=int, default=50)
    sp.set_defaults(fn=cmd_t2v)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
