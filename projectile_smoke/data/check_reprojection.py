"""GO/NO-GO (blocking): reproject the analytic ballistic trajectory
p(t) = p0 + v0*t + 0.5*g*t^2 through the per-episode metadata camera onto the
rendered video, on ALL 4 family dirs (every episode has its own camera pose,
and the opposite_camera families exercise a completely different viewpoint).

This is the ONLY check that ties the physics vector to ground-truth pixels:
the recon aux loss decreases even on a garbage vector, so a wrong camera
transform is invisible to every training-side signal.

Two independent measurements per sampled episode:
  1. 3D sanity (camera-independent): analytic ball position at
     gripper_close_time vs metadata close_ball_position (grasped episodes).
     Catches release-time / trajectory-convention errors.
  2. Pixel error: color-based ball detection (chromaticity match inside a
     generous 90 px window around the prediction) vs projected pixel, over
     strictly-in-flight frames. Catches camera-transform errors.

Release-time nuisance: events.release_frame is an integer frame index for a
sim event that occurs between frames (sim step = 1/8 frame), so a sub-frame
offset delta is fitted per episode (bounded to +/-1.2 frames) and errors are
judged on the residual. Diagnosed 2026-07-15: the raw-delta=0 error signature
(error ~ v(t)*delta, largest right after release, sign flip at apex, worst on
the faster style055 launches) is exactly frame quantization; a wrong camera
transform cannot be absorbed by any delta. Both raw and fitted numbers are
reported. NOTE: events.first_contact_frame == 0 means "ball resting on its
launcher at episode start", not the catch - only trusted when > release_frame.

Writes overlay mp4s + summary.json + a PASS marker file that smoke.sbatch
requires. FAILs loudly if any family's median pixel error exceeds threshold.

  python projectile_smoke/data/check_reprojection.py [--per-family 6]
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projectile_smoke.data.camera import (  # noqa: E402
    ballistic_positions, project, world_to_camera)
from projectile_smoke.data.episodes_proj import (  # noqa: E402
    CACHE, FAMILIES, FPS, iter_episodes, load_metadata, video_path)

WINDOW = 90          # px search window around prediction (>> pass threshold)
CHROMA_TOL = 0.14    # normalized-rgb distance
MIN_PIXELS = 8       # blob size below which the frame counts as undetected
MEDIAN_PX_MAX = 8.0  # per-family pass thresholds
P90_PX_MAX = 25.0    # tail tolerates occasional detector outliers
DET_RATE_MIN = 0.5
CLOSE3D_MAX = 0.04   # m, median 3D error at gripper_close_time


def detect_ball(frame, center_uv, ball_rgb):
    """Chromaticity-matched blob centroid near the predicted center, or None.

    Connected components (not a raw centroid) so a same-colored distractor in
    the window can't drag the estimate - the component nearest the prediction
    wins."""
    import cv2
    h, w = frame.shape[:2]
    u, v = int(round(center_uv[0])), int(round(center_uv[1]))
    x0, x1 = max(u - WINDOW, 0), min(u + WINDOW, w)
    y0, y1 = max(v - WINDOW, 0), min(v + WINDOW, h)
    if x1 <= x0 or y1 <= y0:
        return None
    win = frame[y0:y1, x0:x1].astype(np.float64)
    inten = win.sum(axis=2, keepdims=True)
    chroma = win / np.maximum(inten, 1e-6)
    target = np.asarray(ball_rgb, dtype=np.float64)
    target = target / max(target.sum(), 1e-6)
    dist = np.linalg.norm(chroma - target, axis=2)
    mask = ((dist < CHROMA_TOL) & (inten[..., 0] > 60)).astype(np.uint8)
    if mask.sum() < MIN_PIXELS:
        return None
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best, best_d = None, np.inf
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_PIXELS:
            continue
        c = centroids[i]
        d = np.hypot(c[0] - (u - x0), c[1] - (v - y0))
        if d < best_d:
            best, best_d = c, d
    if best is None:
        return None
    return np.array([x0 + best[0], y0 + best[1]])


def temporal_inliers(frames, dets, thresh=12.0, iters=3):
    """Detector-side outlier rejection with no reference to the prediction:
    the true ball pixel track is smooth (near-quadratic over the short arc),
    so detections far from a robust quadratic fit of the track itself are
    detector glitches. Returns a boolean inlier mask."""
    t = np.asarray(frames, dtype=np.float64)
    d = np.asarray(dets, dtype=np.float64)
    keep = np.ones(len(t), dtype=bool)
    if len(t) < 6:
        return keep
    A = np.stack([t * t, t, np.ones_like(t)], axis=1)
    for _ in range(iters):
        res2 = np.zeros(len(t))
        for ax in range(2):
            coef, *_ = np.linalg.lstsq(A[keep], d[keep, ax], rcond=None)
            res2 += (A @ coef - d[:, ax]) ** 2
        res = np.sqrt(res2)
        new = res < max(thresh, 3.0 * np.median(res[keep]))
        if new.sum() < 6 or (new == keep).all():
            break
        keep = new
    return keep


def flight_frames(meta):
    """Frame indices where the ball is strictly ballistic and airborne.

    Ends 2 frames BEFORE any gripper interaction: in contact_failure /
    near_miss episodes the gripper deflects the ball around
    gripper_close_time without this ever appearing in events
    (first_contact_frame==0 is the launcher-rest contact, not the catch).
    Failure modes are induced via controller offsets only - the launch state
    in metadata is exact for every episode (verified in the generator,
    dataset_generation.py)."""
    release = meta["events"]["release_frame"] or 0
    stop = meta["frame_count"] - 1
    fc = meta["events"].get("first_contact_frame")
    if fc is not None and fc > release:
        stop = min(stop, fc - 1)
    t_end = min(meta["controller_result"].get("ballistic_intercept_time_s")
                or 1e9,
                meta.get("gripper_close_time") or 1e9)
    if t_end < 1e9:
        stop = min(stop, int(t_end * FPS) - 2)
    return list(range(release + 1, stop + 1))


def _project_at_delta(meta, frames_idx, delta_frames):
    """Projected pixels for each frame at flight time (fi-release-delta)/fps;
    frames not yet in flight get None."""
    cam = meta["cameras"]["main_camera"]
    release = meta["events"]["release_frame"] or 0
    t_fl = (np.asarray(frames_idx) - release - delta_frames) / meta["fps"]
    valid = t_fl > 0
    pts_w = ballistic_positions(meta, np.clip(t_fl, 0, None))
    pts_c = world_to_camera(pts_w, cam["pos"], cam["lookat"])
    px, depth = project(pts_c, cam["fovy"])
    return px, valid & (depth > 0)


def check_episode(meta, meta_path, out_dir, save_overlay):
    frames_idx = flight_frames(meta)
    if len(frames_idx) < 3:
        return None
    px0, valid0 = _project_at_delta(meta, frames_idx, 0.0)
    video = iio.imread(video_path(meta_path))

    detections = {}
    for j, fi in enumerate(frames_idx):
        if not valid0[j]:
            continue
        det = detect_ball(video[fi], px0[j], meta["ball_color"][:3])
        if det is not None:
            detections[fi] = det
    if len(detections) >= 6:
        fs = sorted(detections)
        inl = temporal_inliers(fs, [detections[f] for f in fs])
        detections = {f: detections[f] for f, k in zip(fs, inl) if k}
    if len(detections) < 3:
        return {"episode_id": meta["episode_id"],
                "n_flight_frames": len(frames_idx), "n_detected": 0,
                "px_errors": [], "raw_median_px": None,
                "delta_frames": None, "close3d_err_m": None}

    def residuals(delta):
        px, valid = _project_at_delta(meta, frames_idx, delta)
        return [float(np.linalg.norm(detections[fi] - px[j]))
                for j, fi in enumerate(frames_idx)
                if valid[j] and fi in detections]

    raw_median = float(np.median(residuals(0.0)))
    # release_frame = round(release_time_s * fps) in the generator => the true
    # sub-frame offset is within +/-0.5 frames; +/-0.8 leaves margin for the
    # pin-release step discretization.
    grid = np.linspace(-0.8, 0.8, 33)
    med = [np.median(residuals(d)) if len(residuals(d)) >= 3 else np.inf
           for d in grid]
    delta = float(grid[int(np.argmin(med))])
    errors = residuals(delta)

    # camera-independent: best analytic match to close_ball_position over the
    # generator's release_time_s sampling range (decoupled from the pixel fit
    # so detector glitches can't inflate it)
    close3d = None
    if meta["controller_result"].get("grasped"):
        cb = np.asarray(meta["controller_result"]["close_ball_position"])
        t_fl = meta["gripper_close_time"] - np.linspace(0.14, 0.38, 121)
        pred = ballistic_positions(meta, t_fl)
        close3d = float(np.linalg.norm(pred - cb, axis=1).min())

    if save_overlay:
        import cv2
        px_d, valid_d = _project_at_delta(meta, frames_idx, delta)
        overlay = video.copy()
        for j, fi in enumerate(frames_idx):
            if not valid_d[j]:
                continue
            u, v = int(round(px_d[j, 0])), int(round(px_d[j, 1]))
            cv2.drawMarker(overlay[fi], (u, v), (255, 0, 0),
                           cv2.MARKER_CROSS, 18, 2)
            if fi in detections:
                du, dv = detections[fi].round().astype(int)
                cv2.circle(overlay[fi], (du, dv), 12, (0, 255, 0), 2)
        dst = out_dir / f"{meta['episode_id']}_overlay.mp4"
        iio.imwrite(dst, overlay, fps=meta["fps"])

    return {"episode_id": meta["episode_id"],
            "n_flight_frames": len(frames_idx),
            "n_detected": len(detections),
            "px_errors": errors,
            "raw_median_px": raw_median,
            "delta_frames": delta,
            "close3d_err_m": close3d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--out-dir", default=str(CACHE / "reprojection_check"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PASS").unlink(missing_ok=True)

    by_family = {f: [] for f in FAMILIES}
    for key, fam, meta_path in iter_episodes():
        by_family[fam].append(meta_path)

    results, ok = {}, True
    for fam in FAMILIES:
        # deterministic spread across the family (and its failure modes)
        paths = by_family[fam]
        step = max(len(paths) // args.per_family, 1)
        sampled = paths[::step][:args.per_family]
        eps, all_err, close3d = [], [], []
        for i, meta_path in enumerate(sampled):
            meta = load_metadata(meta_path)
            r = check_episode(meta, meta_path, out_dir, save_overlay=(i < 3))
            if r is None:
                continue
            eps.append(r)
            all_err += r["px_errors"]
            if r["close3d_err_m"] is not None:
                close3d.append(r["close3d_err_m"])
        n_flight = sum(e["n_flight_frames"] for e in eps)
        det_rate = len(all_err) / max(n_flight, 1)
        med = float(np.median(all_err)) if all_err else float("inf")
        p90 = float(np.percentile(all_err, 90)) if all_err else float("inf")
        med3d = float(np.median(close3d)) if close3d else None
        raw_meds = [e["raw_median_px"] for e in eps if e["raw_median_px"]]
        raw_med = float(np.median(raw_meds)) if raw_meds else None
        deltas = [e["delta_frames"] for e in eps if e["delta_frames"] is not None]
        fam_ok = (med <= MEDIAN_PX_MAX and p90 <= P90_PX_MAX
                  and det_rate >= DET_RATE_MIN
                  and (med3d is None or med3d <= CLOSE3D_MAX))
        ok &= fam_ok
        results[fam] = {"median_px": med, "p90_px": p90,
                        "raw_median_px_delta0": raw_med,
                        "delta_frames": deltas,
                        "det_rate": round(det_rate, 3),
                        "median_close3d_m": med3d,
                        "episodes": eps, "pass": fam_ok}
        print(f"{fam:40s} median={med:6.2f}px (raw {raw_med and round(raw_med, 1)}) "
              f"p90={p90:6.2f}px det={det_rate:4.0%} "
              f"delta[f]={np.mean(deltas) if deltas else 0:+.2f} close3d="
              f"{med3d if med3d is None else round(med3d, 4)} "
              f"{'PASS' if fam_ok else 'FAIL'}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"thresholds": {"median_px": MEDIAN_PX_MAX,
                                  "p90_px": P90_PX_MAX,
                                  "det_rate": DET_RATE_MIN,
                                  "close3d_m": CLOSE3D_MAX},
                   "families": results, "pass": ok}, f, indent=1)
    if ok:
        (out_dir / "PASS").write_text("all families passed\n")
        print(f"REPROJECTION GO/NO-GO: PASS  (overlays in {out_dir})")
    else:
        print("REPROJECTION GO/NO-GO: FAIL - do NOT train; camera transform "
              "or trajectory convention is wrong. Inspect overlays in "
              f"{out_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
