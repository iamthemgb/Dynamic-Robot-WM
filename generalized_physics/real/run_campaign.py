"""Sequence phases 0-3 for one arm against the real Wan backbone.

    python -m generalized_physics.real.run_campaign --arm wan21_t2v_1p3b
    python -m generalized_physics.real.run_campaign --arm wan21_t2v_1p3b \
        --tiny --limit 200          # V7: 5 steps per phase on 200 episodes

One arm per process (both vendored source trees are named ``wan``). Phase
products are threaded in memory exactly as ``run_smoke_e2e`` does; phases 1
and 3 checkpoint/resume via their run dirs, so on requeue the cheap phases
(0, 2) simply rerun (< minutes) while the expensive ones continue where they
stopped.

Per-phase batch sizes differ deliberately: the DiT-free phases train a small
conv-GRU student and take batch 16; the DiT phases run the full Wan backbone
and take batch 1 with gradient accumulation. ``cfg.train`` is a plain mutable
dataclass, so the sequencer sets it between phases -- orchestration, not a
code edit.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from ..config import smoke_config
from ..training import phase0_representation_probe as phase0
from ..training import phase2_student_distillation as phase2
from . import backend, cache_io
from . import phase1_real, phase3_real
from .instrument import PhaseRecorder
from .paths import ARMS


def _gpu_probe_classes(device):
    """Phase 0 constructs its probe directly (not via common builders), so the
    probe lands on CPU while the student is on GPU. Rebinding the class in the
    phase module's namespace keeps the phase file untouched."""
    from ..models.metadata_query_decoder import MetadataQueryDecoder

    class CudaProbe(MetadataQueryDecoder):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.to(device)

    phase0.MetadataQueryDecoder = CudaProbe


def _write_summary(run_dir, phase, metrics, gates, t0, extra=None):
    s = {"phase": phase, "metrics": metrics, "gates": gates,
         "wall_s": round(time.time() - t0, 1),
         "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    s.update(extra or {})
    d = Path(run_dir) / f"phase{phase}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"[phase{phase}] {json.dumps(metrics)}", flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--phases", default="0,1,2,3")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of train episodes (V7 smoke)")
    ap.add_argument("--tiny", action="store_true",
                    help="5 steps per phase; smoke only")
    ap.add_argument("--p0-steps", type=int, default=800)
    ap.add_argument("--p1-steps", type=int, default=800)
    ap.add_argument("--p2-steps", type=int, default=600)
    ap.add_argument("--p3-steps", type=int, default=400)
    ap.add_argument("--grad-accum", type=int, default=2)
    args = ap.parse_args()

    arm = ARMS[args.arm]
    phases = [int(p) for p in args.phases.split(",")]
    device = "cuda"
    run_dir = arm.run_dir
    if args.tiny:
        # keep smoke checkpoints out of the real run dir -- the campaign's
        # resume logic would otherwise pick up a step-5 checkpoint
        run_dir = run_dir.parent.parent / "runs_tiny" / arm.name
        args.p0_steps = args.p1_steps = args.p2_steps = args.p3_steps = 5
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = smoke_config()
    cfg.vae.latent_channels = arm.latent_channels
    cfg.dit.prefix_bins = 2
    cfg.phase0_steps, cfg.phase1_steps = args.p0_steps, args.p1_steps
    cfg.phase2_steps, cfg.phase3_steps = args.p2_steps, args.p3_steps

    print(f"=== {arm.name}: cache {arm.cache_dir} ===", flush=True)
    cache, registry, normalizer = cache_io.load_cache(
        arm.cache_dir, device=device, split="train", limit=args.limit)
    text_ctx = cache_io.load_text_ctx(arm.cache_dir, cache["prompt_id"],
                                      device=device)
    print(f"train episodes: {len(cache['z'])}  Tz={cache['z'].shape[2]}",
          flush=True)

    backend.install_real_backend(arm, cache, text_ctx, cfg, device=device)
    _gpu_probe_classes(device)

    manifest_path = run_dir / "run_manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})

    def mark(phase, summary):
        manifest[f"phase{phase}"] = {
            "done": True, "wall_s": summary["wall_s"],
            "gates": summary["gates"],
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
        manifest_path.write_text(json.dumps(manifest, indent=2))

    p0 = p1 = p2 = None

    if 0 in phases:
        t0 = time.time()
        cfg.train.batch_size = 16
        with PhaseRecorder(run_dir / "phase0", "phase0"):
            p0 = phase0.run(cfg, cache=cache, registry=registry,
                            normalizer=normalizer)
        m = p0["metrics"]
        gates = {"post_event_gain_positive": m["post_event_gain"] > 0,
                 "controls_degrade": m["controls_degrade"]}
        mark(0, _write_summary(run_dir, 0, m, gates, t0))

    if 1 in phases:
        t0 = time.time()
        cfg.train.batch_size = 1
        p1 = phase1_real.run(cfg, cache, registry, normalizer,
                             run_dir / "phase1", steps=cfg.phase1_steps,
                             grad_accum=args.grad_accum)
        m = p1["metrics"]
        gates = {"gap_wrong_positive": m["gap_wrong"] > 0,
                 "gap_null_positive": m["gap_null"] > 0}
        mark(1, _write_summary(run_dir, 1, m, gates, t0))

    if 2 in phases:
        if p1 is None:
            raise SystemExit("phase 2 needs phase 1 in the same process")
        t0 = time.time()
        cfg.train.batch_size = 16
        if p0 is not None:
            # phase 0's evaluation leaves the student in eval mode; cudnn
            # refuses RNN backward outside training mode (the CPU mock never
            # hits cudnn, so the stock chain never noticed)
            p0["student"].train()
        with PhaseRecorder(run_dir / "phase2", "phase2"):
            p2 = phase2.run(cfg, p1,
                            student=None if p0 is None else p0["student"])
        m = p2["metrics"]
        gates = {"controls_degrade":
                 m["query_time_shuffled"] > m["query_loss"]
                 and m["query_action_swapped"] > m["query_loss"]}
        mark(2, _write_summary(run_dir, 2, m, gates, t0))

    if 3 in phases:
        if p1 is None or p2 is None:
            raise SystemExit("phase 3 needs phases 1 and 2 in this process")
        t0 = time.time()
        cfg.train.batch_size = 1
        p3 = phase3_real.run(cfg, p1, p2, run_dir / "phase3",
                             steps=cfg.phase3_steps,
                             grad_accum=args.grad_accum)
        m = p3["metrics"]
        # gap_closure is None when oracle_gap is below the 1e-4 floor
        # (an undefined ratio must gate as a fail, not crash or pass)
        gates = {"gap_closure_positive": (m["gap_closure"] or 0) > 0}
        mark(3, _write_summary(run_dir, 3, m, gates, t0))

    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
