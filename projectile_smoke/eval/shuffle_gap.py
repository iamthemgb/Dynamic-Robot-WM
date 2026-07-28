"""Paired token-shuffle evaluation - the run's decisive metric.

Val flow-matching loss under three conditions per episode:
  correct  - episode's own physics tokens
  shuffled - a fixed donor episode's tokens (different physics cluster,
             derangement frozen once - a twin's identical physics can never
             be its own 'wrong' donor)
  none     - no physics tokens (style-adaptation control: improves alongside
             the others as the LoRA adapts, so only the paired
             correct-vs-shuffled gap attributes anything to the embedding)

Statistical-power design (the biggest bottleneck of the whole run):
  - Each val episode gets ONE frozen noise tensor and a fixed 4-point sigma
    grid (u before the shift warp), sampled once, stored as a fixture, and
    reused identically across all three conditions and every checkpoint.
  - The gap statistic is the mean PER-EPISODE PAIRED difference
    gap = L(shuffled) - L(correct); pairing cancels scene difficulty and
    noise-draw variance, the dominant variance terms in flow-matching loss.
  - CI via cluster bootstrap over physics-identity clusters (on this dataset
    every cluster is a singleton - verified, no twins exist - so it reduces
    to an episode bootstrap of 300 independent episodes).

Stale-eval canary (cloth run 2103110's bug): consecutive eval summaries must
not be bit-identical; canary_assert fails the run loudly if they are. This
module never caches wrapped weights - the live model is passed in per call,
and the CLI rebuilds the model from the checkpoint on every invocation.

CLI (post-hoc, per checkpoint):
  python projectile_smoke/eval/shuffle_gap.py \
      --ckpt <run>/ckpt_002000 --config projectile_smoke/configs/smoke_lora.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from projectile_smoke.training.dataset import LATENT_SHAPE, SmokeDataset  # noqa: E402

DEFAULT_U_GRID = (0.25, 0.5, 0.75, 0.9)


def shift_sigma(u, shift):
    return shift * u / (1.0 + (shift - 1.0) * u)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def build_fixtures(val_ds, u_grid, noise_seed):
    """Frozen per-episode noise + donor derangement. Deterministic in
    (val split, u_grid, noise_seed)."""
    keys = list(val_ds.keys)
    gen = torch.Generator().manual_seed(noise_seed)
    noise = {k: torch.randn(LATENT_SHAPE, generator=gen,
                            dtype=torch.float32).to(torch.float16)
             for k in keys}

    clusters = val_ds.clusters
    rng = random.Random(noise_seed + 1)
    donors = keys[:]
    rng.shuffle(donors)

    def conflicts():
        return [i for i, (k, d) in enumerate(zip(keys, donors))
                if clusters[k] == clusters[d]]

    for _ in range(200):
        bad = conflicts()
        if not bad:
            break
        if len(bad) >= 2:  # rotate conflicting donors among themselves
            vals = [donors[i] for i in bad]
            for i, v in zip(bad, vals[1:] + vals[:1]):
                donors[i] = v
        else:              # single conflict: swap with a random position
            i, j = bad[0], rng.randrange(len(keys))
            donors[i], donors[j] = donors[j], donors[i]
    assert not conflicts(), "could not build a cluster-safe derangement"

    return {"keys": keys, "u_grid": list(u_grid), "noise_seed": noise_seed,
            "noise": noise, "donor": dict(zip(keys, donors))}


def load_or_build_fixtures(val_ds, cache_dir, u_grid, noise_seed):
    path = Path(cache_dir) / "fixtures_eval.pt"
    if path.exists():
        fx = torch.load(path, map_location="cpu", weights_only=True)
        assert fx["keys"] == list(val_ds.keys), "fixtures stale: val split changed"
        assert fx["u_grid"] == list(u_grid) and fx["noise_seed"] == noise_seed, \
            "fixtures stale: eval config changed - delete fixtures_eval.pt"
        return fx
    fx = build_fixtures(val_ds, u_grid, noise_seed)
    tmp = path.with_suffix(".tmp")
    torch.save(fx, tmp)
    tmp.rename(path)
    print(f"[eval] built fixtures -> {path}")
    return fx


# --------------------------------------------------------------------------
# paired evaluation
# --------------------------------------------------------------------------

def _seq_len(latent):
    c, f, h, w = latent.shape[-4:]
    return f * (h // 2) * (w // 2)


@torch.no_grad()
def paired_eval(model, val_ds, fixtures, device, shift=5.0, batch_size=8,
                max_episodes=None, stride=1):
    """Per-episode losses {correct, shuffled, none}, each averaged over the
    fixed sigma grid. Model is used as passed (live weights); eval-mode is
    set and restored.

    stride>1 evaluates a fixed, family-balanced subset (keys are sorted, so
    striding preserves family proportions) - used for intermediate evals to
    fit the wall-clock budget; the step-0 baseline and final eval run the
    full set. The subset is identical at every step, so the paired trend
    stays comparable across checkpoints."""
    was_training = model.training
    model.eval()
    keys = fixtures["keys"][::stride]
    if max_episodes:
        keys = keys[:max_episodes]
    sigmas = [shift_sigma(u, shift) for u in fixtures["u_grid"]]
    per_ep = {k: {"correct": 0.0, "shuffled": 0.0, "none": 0.0} for k in keys}

    idx_of = {k: i for i, k in enumerate(val_ds.keys)}
    for start in range(0, len(keys), batch_size):
        chunk = keys[start:start + batch_size]
        items = [val_ds[idx_of[k]] for k in chunk]
        x0 = torch.stack([it["latent"] for it in items]).to(device)
        noise = torch.stack([fixtures["noise"][k].float() for k in chunk]).to(device)
        context = [it["t5"].to(device) for it in items]
        phys_correct = torch.stack([it["phys"] for it in items]).to(device)
        phys_shuffled = torch.stack(
            [val_ds.phys[fixtures["donor"][k]] for k in chunk]).to(device)

        for sigma in sigmas:
            x_t = (1.0 - sigma) * x0 + sigma * noise
            v_target = noise - x0
            t = torch.full((len(chunk),), sigma * 1000.0, device=device)
            for cond, phys in [("correct", phys_correct),
                               ("shuffled", phys_shuffled),
                               ("none", None)]:
                with torch.autocast("cuda", torch.bfloat16):
                    pred, _ = model(x=list(x_t), t=t, context=context,
                                    seq_len=_seq_len(x0), phys_vec=phys)
                pred = torch.stack([p.float() for p in pred])
                mse = (pred - v_target.float()).pow(2).flatten(1).mean(dim=1)
                for k, m in zip(chunk, mse):
                    per_ep[k][cond] += m.item() / len(sigmas)

    if was_training:
        model.train()
    return [{"key": k, **v} for k, v in per_ep.items()]


def cluster_bootstrap_ci(per_ep, clusters, iters=2000, seed=0):
    """95% CI of the mean paired gap, resampling physics clusters."""
    by_cluster = {}
    for e in per_ep:
        by_cluster.setdefault(clusters[e["key"]], []).append(
            e["shuffled"] - e["correct"])
    groups = list(by_cluster.values())
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        sample = [g for _ in groups for g in groups[rng.randrange(len(groups))]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def summarize(per_ep, clusters, iters=2000):
    n = len(per_ep)
    gaps = [e["shuffled"] - e["correct"] for e in per_ep]
    lo, hi = cluster_bootstrap_ci(per_ep, clusters, iters=iters)
    return {
        "n_episodes": n,
        "n_clusters": len({clusters[e["key"]] for e in per_ep}),
        "loss_correct": sum(e["correct"] for e in per_ep) / n,
        "loss_shuffled": sum(e["shuffled"] for e in per_ep) / n,
        "loss_none": sum(e["none"] for e in per_ep) / n,
        "gap_paired": sum(gaps) / n,
        "gap_ci95": [lo, hi],
        "gap_positive_significant": lo > 0,
    }


def canary_assert(prev_summary, cur_summary):
    """Stale-eval canary: bit-identical consecutive evals = frozen weights
    (cloth run 2103110 reported identical numbers from step 3k to 10k)."""
    if prev_summary is None:
        return
    same = all(prev_summary[k] == cur_summary[k]
               for k in ("loss_correct", "loss_shuffled", "loss_none"))
    assert not same, (
        "STALE-EVAL CANARY: consecutive checkpoints produced bit-identical "
        "eval losses - the eval is not seeing updated weights. "
        f"prev={prev_summary} cur={cur_summary}")


# --------------------------------------------------------------------------
# CLI: evaluate a saved checkpoint (rebuilds the model fresh every call)
# --------------------------------------------------------------------------

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--prev", default=None,
                    help="shuffle_gap.json of the previous checkpoint (canary)")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda")

    from projectile_smoke.models.lora import (CROSS_ATTN_TARGETS,
                                              DEFAULT_TARGETS, apply_lora,
                                              load_lora_state_dict)
    from projectile_smoke.models.wan_wrapper import PhysicsWan
    from projectile_smoke.models.physics_encoder import ProjectilePhysicsEncoder

    state = torch.load(Path(args.ckpt) / "trainer.pt", map_location="cpu",
                       weights_only=False)  # our own file; contains RNG states
    val_ds = SmokeDataset(cfg["cache_dir"], "val")
    encoder = ProjectilePhysicsEncoder(
        phys_dim=val_ds.phys_dim, num_tokens=cfg["num_tokens"],
        n_freqs=cfg["fourier_freqs"],
        with_contrastive=cfg.get("infonce", {}).get("enabled", False))
    encoder._base_initialized = True  # weights come from the checkpoint
    encoder.load_state_dict(state["encoder"])
    model = PhysicsWan(encoder, model_dir=cfg["model_dir"])
    if state.get("lora"):
        lc = state["config"]["lora"]
        targets = (CROSS_ATTN_TARGETS if lc.get("cross_attn_only", True)
                   else DEFAULT_TARGETS)
        apply_lora(model.dit, targets=targets, rank=lc["rank"], alpha=lc["alpha"])
        load_lora_state_dict(model.dit, state["lora"])
    model.to(device).eval()

    ev = cfg.get("eval", {})
    fixtures = load_or_build_fixtures(
        val_ds, cfg["cache_dir"], ev.get("u_grid", DEFAULT_U_GRID),
        ev.get("noise_seed", 1234))
    per_ep = paired_eval(model, val_ds, fixtures, device, shift=cfg["shift"],
                         batch_size=ev.get("batch_size", 8),
                         max_episodes=args.max_episodes)
    summary = summarize(per_ep, val_ds.clusters,
                        iters=ev.get("bootstrap_iters", 2000))
    summary["ckpt"] = args.ckpt
    summary["step"] = state.get("step")

    if args.prev:
        with open(args.prev) as f:
            canary_assert(json.load(f)["summary"], summary)

    print(json.dumps(summary, indent=2))
    out = Path(args.ckpt) / "shuffle_gap.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "per_episode": per_ep}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
