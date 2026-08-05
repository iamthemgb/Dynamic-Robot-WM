"""Phase 1 (oracle Wan state-conditioning) on the pbc_state_groups corpus.

    python -m generalized_physics.real.pbc_run_campaign --arm wan21_t2v_1p3b
    python -m generalized_physics.real.pbc_run_campaign --arm wan22_ti2v_5b \
        --tiny --limit 40           # adapter smoke on the P0 pilot

One arm per process (the vendored source trees are both named ``wan``).
Reuses the f1 campaign machinery wholesale -- ``cache_io``, ``backend``,
``phase1_real`` -- against a ``pbc_``-prefixed cache/run namespace so the
f1_10h artifacts are never touched: ``dataclasses.replace`` on the frozen
ArmSpec redirects ``cache_dir`` to ``cache/pbc_<vae>_vae`` and ``run_dir``
to ``runs/pbc_<arm>``.

v3 gate semantics: conditioning is the state bundle; physics tokens do not
exist in this corpus's records. ``gap_wrong`` compares same-group siblings
(identical scene, different IC) under shared (tau, eps) -- the easiest
causal signal the conditioning pathway could possibly use. 14B is out of
scope for this campaign.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..config import smoke_config
from . import backend, cache_io, phase1_real
from .paths import ARMS

PBC_ARMS = ("wan21_t2v_1p3b", "wan22_ti2v_5b")


def pbc_arm(name, cache_name=None):
    base = ARMS[name]
    return replace(base, name=f"pbc_{base.name}",
                   cache_name=cache_name or f"pbc_{base.cache_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(PBC_ARMS))
    ap.add_argument("--p1-steps", type=int, default=800)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap train episodes (adapter smoke)")
    ap.add_argument("--tiny", action="store_true",
                    help="5 steps; smoke only, separate run dir")
    ap.add_argument("--cache-name", default=None,
                    help="cache dir name under OUT_ROOT/cache (smoke runs)")
    args = ap.parse_args()

    arm = pbc_arm(args.arm, cache_name=args.cache_name)
    device = "cuda"
    run_dir = arm.run_dir
    steps = args.p1_steps
    if args.tiny:
        run_dir = run_dir.parent.parent / "runs_tiny" / arm.name
        steps = 5
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = smoke_config()
    cfg.vae.latent_channels = arm.latent_channels
    cfg.dit.prefix_bins = 2
    cfg.phase1_steps = steps

    print(f"=== {arm.name}: cache {arm.cache_dir} ===", flush=True)
    cache, registry, normalizer = cache_io.load_cache(
        arm.cache_dir, device=device, split="train", limit=args.limit)
    text_ctx = cache_io.load_text_ctx(arm.cache_dir, cache["prompt_id"],
                                      device=device)
    print(f"train episodes: {len(cache['z'])}  Tz={cache['z'].shape[2]}  "
          f"groups={int(cache['group_id'].max()) + 1}", flush=True)

    backend.install_real_backend(arm, cache, text_ctx, cfg, device=device,
                                 action_dim=9)

    t0 = time.time()
    cfg.train.batch_size = 1
    p1 = phase1_real.run(cfg, cache, registry, normalizer,
                         run_dir / "phase1", steps=steps,
                         grad_accum=args.grad_accum)
    m = p1["metrics"]
    gates = {"gap_wrong_positive": m["gap_wrong"] > 0,
             "gap_null_positive": m["gap_null"] > 0}
    summary = {
        "phase": 1, "arm": arm.name, "dataset": "pbc_state_groups_v3",
        "steps": steps, "metrics": m, "gates": gates,
        "wall_s": round(time.time() - t0, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    d = run_dir / "phase1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    manifest_path = run_dir / "run_manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})
    manifest["phase1"] = {"done": True, "gates": gates,
                          "wall_s": summary["wall_s"],
                          "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[phase1 {arm.name}] {json.dumps(m)}", flush=True)
    print(f"gates: {json.dumps(gates)}", flush=True)


if __name__ == "__main__":
    main()
