"""Projectile smoke training: physics encoder (+ small cross-attn LoRA) on
Wan2.1-T2V-1.3B, single GPU, plain PyTorch (no accelerate - debuggable).

The deliverable is loss CURVES, so the loop is built for interpretability:
  - constant LR after linear warmup (LR decay can fake a late-run decrease;
    cosine stays available in the config for the follow-on full run)
  - tanh gate logged EVERY step (train_log.csv) with the step-500 kill rule
    from the cloth failure (gate rising + flat EMA train L_fm = tokens being
    opened but not used - stop and inspect, don't burn queue time)
  - paired 3-condition shuffle-gap eval every eval_interval on frozen
    fixtures + stale-eval canary (see eval/shuffle_gap.py)
  - deterministic data order + saved RNG states => save/resume continues the
    loss from the same value (success criterion 5)

Launch (from the Wan2.1 repo root):
  python projectile_smoke/training/train_smoke.py \
      --config projectile_smoke/configs/smoke_lora.yaml [--resume <ckpt_dir>]
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from projectile_smoke.models.lora import (CROSS_ATTN_TARGETS, DEFAULT_TARGETS,  # noqa: E402
                                          apply_lora, load_lora_state_dict,
                                          lora_state_dict)
from projectile_smoke.models.wan_wrapper import PhysicsWan  # noqa: E402
from projectile_smoke.training.flow_match import (flow_loss, make_targets,  # noqa: E402
                                                  sample_sigmas)

from projectile_smoke.eval.shuffle_gap import (  # noqa: E402
    DEFAULT_U_GRID, canary_assert, load_or_build_fixtures, paired_eval,
    summarize)
from projectile_smoke.models.physics_encoder import ProjectilePhysicsEncoder  # noqa: E402
from projectile_smoke.training.dataset import SmokeDataset, collate  # noqa: E402

EMA_SPAN = 100


def seq_len_of(latent):
    c, f, h, w = latent.shape[-4:]
    return f * (h // 2) * (w // 2)


# --------------------------------------------------------------------------
# construction (also reused by tests/test_overfit.py)
# --------------------------------------------------------------------------

def build_model(cfg, phys_dim, device):
    encoder = ProjectilePhysicsEncoder(
        phys_dim=phys_dim, num_tokens=cfg["num_tokens"],
        n_freqs=cfg["fourier_freqs"],
        with_contrastive=cfg.get("infonce", {}).get("enabled", False))
    model = PhysicsWan(encoder, model_dir=cfg["model_dir"])
    model.freeze_dit()
    model.enable_gradient_checkpointing()
    lora_params = []
    if cfg.get("lora", {}).get("enabled"):
        lc = cfg["lora"]
        targets = (CROSS_ATTN_TARGETS if lc.get("cross_attn_only", True)
                   else DEFAULT_TARGETS)
        lora_params = apply_lora(model.dit, targets=targets,
                                 rank=lc["rank"], alpha=lc["alpha"])
        print(f"LoRA: {sum(p.numel() for p in lora_params) / 1e6:.2f}M params "
              f"({len(lora_params) // 2} wrapped linears)")
    model.to(device)
    return model, encoder, lora_params


def init_base_tokens(encoder, cfg, train_keys):
    """Base tokens match empirical mean/std of OUR captions' T5 embeddings."""
    t5_dir = Path(cfg["cache_dir"]) / "t5_blind"
    stems = sorted(k.replace("/", "__", 1) for k in train_keys)[:64]
    samples = [torch.load(t5_dir / f"{s}.pt", weights_only=True)["t5"].float()
               for s in stems]
    encoder.init_base_from_t5(samples)


def build_optimizer(cfg, encoder, lora_params):
    groups = [{"params": list(encoder.parameters()), "lr": cfg["encoder_lr"]}]
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg["lora"]["lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"],
                            betas=(0.9, 0.95))
    warmup = max(cfg["warmup_steps"], 1)
    schedule = cfg.get("lr_schedule", "constant")
    if schedule == "constant":
        # deliberately NOT cosine: on a 2k-step diagnostic, LR decay flattens
        # the late-run loss on its own; with constant LR any decrease is
        # unambiguously learning.
        def lr_fn(s):
            return min((s + 1) / warmup, 1.0)
    elif schedule == "cosine":  # for the follow-on full run
        total = cfg["max_steps"]

        def lr_fn(s):
            if s < warmup:
                return (s + 1) / warmup
            p = (s - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    else:
        raise ValueError(schedule)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    return opt, sched


# --------------------------------------------------------------------------
# deterministic data order: epoch-wise permutation streams, reconstructable
# from (seed, global sample position) so resume is exact
# --------------------------------------------------------------------------

class EpochPermutation:

    def __init__(self, n, seed):
        self.n, self.seed = n, seed
        self._epoch, self._perm = None, None

    def index_at(self, pos):
        epoch, offset = divmod(pos, self.n)
        if epoch != self._epoch:
            g = torch.Generator().manual_seed(self.seed * 100003 + epoch)
            self._perm = torch.randperm(self.n, generator=g)
            self._epoch = epoch
        return int(self._perm[offset])


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def train_forward(model, batch, cfg, device):
    latent = batch["latent"].to(device)
    sigmas = sample_sigmas(latent.size(0), cfg["shift"], device)
    x_t, v_target, t = make_targets(latent, sigmas)
    context = [u.to(device) for u in batch["t5"]]
    phys = batch["phys"].to(device)
    with torch.autocast("cuda", torch.bfloat16):
        pred, aux_pred = model(x=list(x_t), t=t, context=context,
                               seq_len=seq_len_of(latent), phys_vec=phys)
    loss_fm = flow_loss(pred, v_target)
    loss_aux = torch.nn.functional.mse_loss(aux_pred.float(), phys)
    return loss_fm, loss_aux


# --------------------------------------------------------------------------
# checkpointing (exact-resume: model + opt + sched + EMA + RNG states)
# --------------------------------------------------------------------------

def save_checkpoint(out_dir, step, cfg, encoder, model, optimizer, scheduler,
                    ema_fm, killrule_ref):
    d = Path(out_dir) / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "encoder": encoder.state_dict(),
        "lora": (lora_state_dict(model.dit)
                 if cfg.get("lora", {}).get("enabled") else None),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema_fm": ema_fm,
        "killrule_ref": killrule_ref,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
        "config": cfg,
    }, d / "trainer.pt")
    print(f"saved {d}", flush=True)
    return d


def load_checkpoint(ckpt_dir, encoder, model, optimizer, scheduler,
                    restore_rng=True):
    state = torch.load(Path(ckpt_dir) / "trainer.pt", map_location="cpu",
                       weights_only=False)
    encoder.load_state_dict(state["encoder"])
    if state.get("lora"):
        load_lora_state_dict(model.dit, state["lora"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
    return state


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir (default: auto-resume latest in out_dir)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_ds = SmokeDataset(cfg["cache_dir"], "train")
    val_ds = SmokeDataset(cfg["cache_dir"], "val")
    print(f"train {len(train_ds)} / val {len(val_ds)} episodes, "
          f"phys_dim {train_ds.phys_dim}")

    model, encoder, lora_params = build_model(cfg, train_ds.phys_dim, device)
    init_base_tokens(encoder, cfg, train_ds.keys)
    optimizer, scheduler = build_optimizer(cfg, encoder, lora_params)
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    print(f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M")

    ev = cfg.get("eval", {})
    fixtures = load_or_build_fixtures(
        val_ds, cfg["cache_dir"], ev.get("u_grid", DEFAULT_U_GRID),
        ev.get("noise_seed", 1234))

    infonce = cfg.get("infonce", {})
    phys_matrix = (train_ds.phys_matrix().to(device)
                   if infonce.get("enabled") else None)

    # resume
    step, ema_fm, killrule_ref = 0, None, {}
    ckpts = sorted(out_dir.glob("ckpt_*"), key=lambda p: int(p.name.split("_")[1]))
    resume_dir = Path(args.resume) if args.resume else (ckpts[-1] if ckpts else None)
    if resume_dir is not None:
        state = load_checkpoint(resume_dir, encoder, model, optimizer, scheduler)
        step = state["step"]
        ema_fm = state["ema_fm"]
        killrule_ref = state.get("killrule_ref") or {}
        print(f"resumed from {resume_dir} at step {step}")

    log_path = out_dir / "log.jsonl"
    csv_path = out_dir / "train_log.csv"
    eval_path = out_dir / "eval_history.jsonl"
    if resume_dir is None and csv_path.exists() and len(csv_path.read_text().splitlines()) > 1:
        # fresh start over a dead run that never checkpointed: rotate its
        # logs aside so step numbers don't duplicate and corrupt the curves
        tag = f".orphaned-{int(time.time())}"
        for p in (log_path, csv_path, eval_path):
            if p.exists():
                p.rename(p.with_name(p.name + tag))
        print(f"rotated orphaned logs from an uncheckpointed dead run ({tag})")
    if not csv_path.exists():
        csv_path.write_text("step,loss_fm,loss_aux,loss_nce,ema_fm,gate,lr,sec\n")
    prev_eval = {}  # last summary per stride: canary compares like-with-like
    if eval_path.exists():
        for line in eval_path.read_text().strip().splitlines():
            s = json.loads(line)["summary"]
            prev_eval[s.get("stride", 1)] = s

    def log_line(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def run_eval(at_step):
        nonlocal prev_eval
        t0 = time.time()
        # full 300 episodes at the two verdict points (baseline + final);
        # fixed stride-subset (~100) in between to fit the 6 h budget
        full = at_step in (0, cfg["max_steps"])
        stride = 1 if full else ev.get("subset_stride", 3)
        per_ep = paired_eval(model, val_ds, fixtures, device,
                             shift=cfg["shift"],
                             batch_size=ev.get("batch_size", 8),
                             max_episodes=ev.get("max_episodes"),
                             stride=stride)
        summary = summarize(per_ep, val_ds.clusters,
                            iters=ev.get("bootstrap_iters", 2000))
        summary["step"] = at_step
        summary["stride"] = stride
        summary["eval_sec"] = round(time.time() - t0, 1)
        prev = prev_eval.get(stride)
        if prev is not None and prev.get("step") == summary["step"]:
            prev = None  # same-step re-run (preemption restart): identical
        canary_assert(prev, summary)  # stale-eval canary, fails loudly
        prev_eval[stride] = summary
        with open(eval_path, "a") as f:
            f.write(json.dumps({"summary": summary, "per_episode": per_ep}) + "\n")
        log_line({"eval": summary})
        print(json.dumps(summary), flush=True)

    kill = cfg.get("kill_rule", {})

    def check_kill_rule(gate_val):
        if not kill.get("enabled", True):
            return
        if step == kill.get("ref_step", 100):
            killrule_ref.update({"ema": ema_fm, "gate": gate_val})
        if step == kill.get("at_step", 500) and killrule_ref:
            gate_rising = gate_val > killrule_ref["gate"] + kill.get("min_gate_rise", 0.005)
            fm_flat = ema_fm > killrule_ref["ema"] * (1 - kill.get("min_fm_improve", 0.01))
            if gate_rising and fm_flat:
                save_checkpoint(out_dir, step, cfg, encoder, model, optimizer,
                                scheduler, ema_fm, killrule_ref)
                raise RuntimeError(
                    f"KILL RULE (cloth failure signature): gate rose "
                    f"{killrule_ref['gate']:.4f}->{gate_val:.4f} while EMA "
                    f"train L_fm stayed flat {killrule_ref['ema']:.4f}->"
                    f"{ema_fm:.4f} by step {step}. The DiT is opening the "
                    f"gate without using the tokens - inspect before burning "
                    f"more queue time.")

    perm = EpochPermutation(len(train_ds), cfg["seed"])
    accum = cfg["grad_accum"]
    batch_size = cfg["batch_size"]
    alpha = 2.0 / (EMA_SPAN + 1)
    total_steps = cfg["max_steps"]

    if step == 0 and resume_dir is None:
        run_eval(0)  # baseline: gap should be ~0 at init
        # step-0 ckpt: a preempted/requeued job resumes here instead of
        # re-running the 20-min baseline eval
        save_checkpoint(out_dir, 0, cfg, encoder, model, optimizer,
                        scheduler, ema_fm, killrule_ref)

    model.train()
    print(f"training: steps {step}->{total_steps}, effective batch "
          f"{batch_size * accum}, ~{total_steps * batch_size * accum / len(train_ds):.1f} "
          f"epochs over {len(train_ds)} episodes", flush=True)
    while step < total_steps:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        fm_sum = aux_sum = nce_sum = 0.0
        for micro in range(accum):
            pos = (step * accum + micro) * batch_size
            items = [train_ds[perm.index_at(pos + j)] for j in range(batch_size)]
            batch = collate(items)
            loss_fm, loss_aux = train_forward(model, batch, cfg, device)
            loss = loss_fm + cfg["aux_weight"] * loss_aux
            if phys_matrix is not None:
                g = torch.Generator(device="cpu").manual_seed(
                    cfg["seed"] * 7 + step * accum + micro)
                idx = torch.randperm(len(phys_matrix), generator=g)[:infonce["batch"]]
                loss_nce = encoder.infonce_loss(
                    phys_matrix[idx.to(device)],
                    noise_std=infonce.get("noise_std", 0.02),
                    temperature=infonce.get("temperature", 0.1))
                loss = loss + infonce["weight"] * loss_nce
                nce_sum += loss_nce.item()
            (loss / accum).backward()
            fm_sum += loss_fm.item()
            aux_sum += loss_aux.item()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        step += 1

        fm_step = fm_sum / accum
        aux_step = aux_sum / accum
        ema_fm = fm_step if ema_fm is None else (1 - alpha) * ema_fm + alpha * fm_step
        gate_val = float(torch.tanh(encoder.gate.detach()))
        lr_now = scheduler.get_last_lr()[0]
        sec = time.time() - t0
        with open(csv_path, "a") as f:  # gate logged EVERY step
            f.write(f"{step},{fm_step:.5f},{aux_step:.6f},"
                    f"{nce_sum / accum:.5f},{ema_fm:.5f},{gate_val:.5f},"
                    f"{lr_now:.2e},{sec:.2f}\n")
        if step % cfg["log_interval"] == 0:
            rec = {"step": step, "loss_fm": round(fm_step, 5),
                   "loss_aux": round(aux_step, 6), "ema_fm": round(ema_fm, 5),
                   "gate": round(gate_val, 4), "lr": lr_now,
                   "sec_per_step": round(sec, 2)}
            print(json.dumps(rec), flush=True)
            log_line(rec)

        check_kill_rule(gate_val)

        if step % cfg["eval_interval"] == 0 or step == total_steps:
            run_eval(step)
        if step % cfg["ckpt_interval"] == 0 or step == total_steps:
            save_checkpoint(out_dir, step, cfg, encoder, model, optimizer,
                            scheduler, ema_fm, killrule_ref)

    print("training done; plotting curves")
    try:
        from projectile_smoke.eval.plot_curves import plot_run
        png = plot_run(out_dir)
        print(f"deliverable: {png}")
    except Exception as e:  # plotting must never kill a finished run
        print(f"plotting failed ({e}); run eval/plot_curves.py manually")


if __name__ == "__main__":
    main()
