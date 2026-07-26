"""Split existing composite previews into per-view (main/top/side) videos.

Each episode's composite ``videos/{episode_id}.mp4`` is a horizontal stack of
camera panels. This reads that composite plus its metadata (for the camera
panel order and per-panel frame size) and writes one captioned clip per view
into the same ``videos/`` folder as ``{episode_id}__{main,top,side}.mp4``.

It also records the split-out clips under a ``view_videos`` key in each
episode's metadata JSON so the dataset stays self-describing.

Usage (from the directory containing the package):

  <venv>/python -m mujoco_franka_cloth_previews.split_views \\
      --out mujoco_franka_cloth_previews/outputs/preview_001
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from .views import caption_frame, caption_text, view_names


def split_episode(meta_path, out_root, fps_default=24, overwrite=False):
    meta = json.loads(Path(meta_path).read_text())
    episode_id = meta["episode_id"]
    cameras = meta["cameras"]
    width = int(meta["frame_size"][0])
    fps = float(meta.get("fps", fps_default))
    videos_dir = out_root / "videos"

    src = videos_dir / f"{episode_id}.mp4"
    if not src.exists():
        return None, f"missing composite {src.name}"

    names = view_names(cameras)
    reader = imageio.get_reader(str(src))
    # Panel width from the actual composite, guarding against a 1px even-crop.
    size = reader.get_meta_data().get("size")
    panel_w = (size[0] // len(names)) if size else width

    writers, rel_paths = {}, {}
    for nm in names:
        dst = videos_dir / f"{episode_id}__{nm}.mp4"
        if dst.exists() and not overwrite:
            reader.close()
            return None, f"exists {dst.name} (use --overwrite)"
        rel_paths[nm] = str(dst.relative_to(out_root))
        writers[nm] = imageio.get_writer(
            str(dst), fps=fps, codec="libx264", quality=8,
            macro_block_size=1, pixelformat="yuv420p")

    n_frames = 0
    for frame in reader:
        for i, nm in enumerate(names):
            panel = frame[:, i * panel_w:(i + 1) * panel_w]
            writers[nm].append_data(caption_frame(panel, caption_text(nm)))
        n_frames += 1
    reader.close()
    for w in writers.values():
        w.close()

    # Record the split-out clips in metadata.
    meta["view_videos"] = {view_labels_key(cameras[i]): rel_paths[names[i]]
                           for i in range(len(names))}
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    return rel_paths, n_frames


def view_labels_key(cam_name):
    # Keyed by the view label so metadata reads {"main": ..., "top": ..., "side": ...}.
    from .views import view_label
    return view_label(cam_name, 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="preview output root (contains videos/ and metadata/)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate per-view clips even if they already exist")
    args = ap.parse_args(argv)

    out_root = Path(args.out).resolve()
    meta_dir = out_root / "metadata"
    metas = sorted(meta_dir.glob("*.json"))
    if not metas:
        raise SystemExit(f"no metadata json found under {meta_dir}")

    ok, skipped = 0, 0
    for mp in metas:
        rel_paths, info = split_episode(mp, out_root, overwrite=args.overwrite)
        if rel_paths is None:
            print(f"[skip] {mp.stem}: {info}")
            skipped += 1
        else:
            views = ", ".join(rel_paths)
            print(f"[ok]   {mp.stem}: {info} frames -> {views}")
            ok += 1

    print("\n" + "=" * 60)
    print(f"Split {ok} episodes into per-view videos ({skipped} skipped)")
    print(f"  videos: {out_root / 'videos'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
