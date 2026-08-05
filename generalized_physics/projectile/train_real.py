"""Generalized Phase-1 oracle training on the REAL projectile dataset.

The third conditioning arm on wan_projectile_smoke_cache, directly
comparable with the two prior smoke runs (same latents, T5 cache, splits,
frozen eval fixtures, flow convention):

  smoke_lora  : Fourier+MLP -> T5 pad slots + r=8 LoRA (gap 0.0532 @ 2k)
  smoke_adapter: 14-D vector -> projector -> 30 adapters, no LoRA
  THIS RUN    : 14 typed records -> DeepSets metadata teacher -> 64-D belief
                -> shared projector (8 x 256 fixed tokens) -> FOUR quarter-
                depth zero-gated adapters + rank-16 zero-init LoRA on q/v.

Losses (deliberate departure from the plan's eq. 16-19):
    L_T = w_fm * L_FM(conditioned) + w_meta * L_meta          [query decoder]
with the 10% learned-null / 10% code-noise corruption mixture. The
shuffled-code rank hinge is NOT optimized any more — training the
correct-below-wrong margin was circular with the correct/wrong gap used to
certify that the physics vector has been learned. Instead, every
``rank_monitor_interval`` steps (default 10) the wrong-code loss is
recomputed under ``torch.no_grad`` with the SAME (sigma, eps) and the
signed ``rank_gap = L_FM(wrong) - L_FM(correct)`` is logged as a training
diagnostic; the paired eval remains the checking point. A ``w_rank`` key in
the config is ignored with a warning.

The paired eval reports the plan's correct / wrong(shuffled donor) / null
losses under frozen shared noise, with cluster bootstrap; the dataset has
no counterfactual groups, so "wrong" is the frozen donor derangement — the
same definition both prior arms used. At the first and final eval the paired
losses are ALSO computed with LoRA switched off (adapter-only attribution).

Launch (from wan_scripts_new):
  python -m generalized_physics.projectile.train_real \
      --config generalized_physics/projectile/configs/train_real.yaml
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

WAN_REPO = Path("/gpfs/radev/home/mzl7/scratch/Wan2.1")
if str(WAN_REPO) not in sys.path:
    sys.path.insert(0, str(WAN_REPO))

from physics_finetune.training.flow_match import (  # noqa: E402
    flow_loss, make_targets, sample_sigmas, shift_sigma)
from projectile_smoke.eval.shuffle_gap import (  # noqa: E402
    DEFAULT_U_GRID, load_or_build_fixtures, summarize)
from projectile_smoke.training.dataset import SmokeDataset  # noqa: E402

from ..config import PipelineConfig  # noqa: E402
from ..models.metadata_records import MetadataRegistry  # noqa: E402
from ..models.metadata_query_decoder import MetadataQueryDecoder  # noqa: E402
from ..models.metadata_teacher import MetadataTeacher  # noqa: E402
from ..models.physics_projector import PhysicsProjector  # noqa: E402
from ..training.common import dedupe_params  # noqa: E402
from ..wan.adaptive_wan_integration import GeneralizedAdaptiveWan  # noqa: E402
from .records import ProjectileRecordSchema  # noqa: E402

EMA_SPAN = 100


def seq_len_of(latent):
    c, f, h, w = latent.shape[-4:]
    return f * (h // 2) * (w // 2)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def pipeline_cfg(cfg):
    """Map the YAML onto the package's PipelineConfig sections."""
    p = PipelineConfig()
    p.teacher.belief_dim = cfg.get("belief_dim", 64)
    p.projector.k_tokens = cfg.get("k_tokens", 8)
    p.projector.d_phys = cfg.get("d_phys", 256)
    p.adapters.n_adapters = cfg.get("n_adapters", 4)
    p.adapters.heads = cfg.get("adapter_heads", 4)
    p.lora.rank = cfg.get("lora_rank", 16)
    p.lora.alpha = cfg.get("lora_alpha", 16.0)
    p.lora.dropout = cfg.get("lora_dropout", 0.05)
    p.lora.targets = tuple(cfg.get("lora_targets", ["q", "v"]))
    return p


def build_models(cfg, schema, device):
    p = pipeline_cfg(cfg)
    model = GeneralizedAdaptiveWan(p, model_dir=cfg["model_dir"])
    teacher = MetadataTeacher(schema.registry,
                              belief_dim=p.teacher.belief_dim)
    decoder = MetadataQueryDecoder(teacher.embed, p.teacher.belief_dim)
    projector = PhysicsProjector(p.teacher.belief_dim, p.projector.k_tokens,
                                 p.projector.d_phys, cfg.get(
                                     "projector_hidden", 256))
    n = {"teacher": teacher, "decoder": decoder, "projector": projector,
         "adapters": model.physics, "lora": None}
    counts = {k: sum(q.numel() for q in v.parameters())
              for k, v in n.items() if v is not None}
    counts["lora"] = sum(q.numel() for q in model.lora_parameters())
    print("trainable params (M):",
          {k: round(v / 1e6, 3) for k, v in counts.items()},
          f"| {len(model.physics.block_idx)} adapters at blocks "
          f"{model.physics.block_idx} | {len(model.lora_names)} LoRA wraps",
          flush=True)
    for m in (teacher, decoder, projector):
        m.to(device)
    model.to(device)
    return model, teacher, decoder, projector


def build_optimizer(cfg, model, teacher, decoder, projector):
    """Plan LRs: teacher/decoder 3e-4; projector+adapters+LoRA 1e-4;
    zero-init gates at 10x adapter LR without weight decay."""
    wd = cfg.get("weight_decay", 0.01)
    groups = [
        {"params": dedupe_params(list(teacher.parameters())
                                 + list(decoder.parameters())),
         "lr": cfg.get("lr_teacher", 3e-4), "weight_decay": wd},
        {"params": (list(projector.parameters())
                    + model.physics.non_gate_params()
                    + model.lora_parameters()),
         "lr": cfg.get("lr_adapters", 1e-4), "weight_decay": wd},
        {"params": model.physics.gate_params(),
         "lr": cfg.get("lr_adapters", 1e-4) * cfg.get("gate_lr_mult", 10.0),
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    warmup = max(cfg.get("warmup_steps", 100), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 1.0))     # constant after warmup
    return opt, sched


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
# plan losses (eq. 16-19)
# --------------------------------------------------------------------------

def sample_condition(cfg):
    r = float(torch.rand(()))
    if r < cfg.get("p_null", 0.10):
        return "null"
    if r < cfg.get("p_null", 0.10) + cfg.get("p_noise", 0.10):
        return "noisy"
    return "clean"


def train_forward(model, teacher, decoder, projector, schema, batch, cfg,
                  device, phys_matrix, monitor_rank=False):
    """One micro-batch. Returns (loss, parts dict, condition)."""
    latent = batch["latent"].to(device)
    B = latent.size(0)
    sigmas = sample_sigmas(B, cfg["shift"], device)
    x_t, v_target, t = make_targets(latent, sigmas)
    context = [u.to(device) for u in batch["t5"]]

    rec = schema.batch(batch["phys"], device=device)
    b_t = teacher.forward_batch(rec)
    l_meta = decoder.loss(b_t, rec)

    cond = sample_condition(cfg)
    if cond == "null":
        tokens = projector.null_tokens(B)
    elif cond == "noisy":
        tokens = projector(b_t + cfg.get("noise_scale", 0.1)
                           * torch.randn_like(b_t))
    else:
        tokens = projector(b_t)

    with torch.autocast("cuda", torch.bfloat16):
        pred = model(x=list(x_t), t=t, context=context,
                     seq_len=seq_len_of(latent), physics_ctx=tokens)
    l_fm = flow_loss(pred, v_target)
    loss = cfg.get("w_fm", 1.0) * l_fm + cfg.get("w_meta", 0.2) * l_meta

    rank_gap = None
    if monitor_rank and cond == "clean":
        # Diagnostic only, never optimized: wrong code = another training
        # episode's records under the SAME (sigma, eps).  A positive gap
        # means the physics vector is informative to the prediction.
        j = int(torch.randint(len(phys_matrix), (1,)))
        wrong = phys_matrix[j][None].expand(B, -1)
        if torch.allclose(wrong, batch["phys"]):
            j = (j + 1) % len(phys_matrix)
            wrong = phys_matrix[j][None].expand(B, -1)
        with torch.no_grad():
            b_w = teacher.forward_batch(schema.batch(wrong, device=device))
            with torch.autocast("cuda", torch.bfloat16):
                pred_w = model(x=list(x_t), t=t, context=context,
                               seq_len=seq_len_of(latent),
                               physics_ctx=projector(b_w))
            l_fm_w = flow_loss(pred_w, v_target)
        rank_gap = float(l_fm_w) - float(l_fm)

    parts = {"fm": float(l_fm), "meta": float(l_meta), "rank_gap": rank_gap}
    return loss, parts, cond


# --------------------------------------------------------------------------
# paired eval: plan's correct / wrong / null under frozen shared noise
# --------------------------------------------------------------------------

@torch.no_grad()
def paired_eval(model, teacher, decoder, projector, schema, val_ds, fixtures,
                device, shift, batch_size=8, stride=1):
    was_training = model.training
    model.eval(), teacher.eval(), projector.eval()
    keys = fixtures["keys"][::stride]
    sigmas = [shift_sigma(u, shift) for u in fixtures["u_grid"]]
    per_ep = {k: {"correct": 0.0, "shuffled": 0.0, "none": 0.0}
              for k in keys}
    meta_sum, n_meta = 0.0, 0

    idx_of = {k: i for i, k in enumerate(val_ds.keys)}
    for start in range(0, len(keys), batch_size):
        chunk = keys[start:start + batch_size]
        items = [val_ds[idx_of[k]] for k in chunk]
        x0 = torch.stack([it["latent"] for it in items]).to(device)
        noise = torch.stack([fixtures["noise"][k].float()
                             for k in chunk]).to(device)
        context = [it["t5"].to(device) for it in items]
        phys_c = torch.stack([it["phys"] for it in items])
        phys_s = torch.stack([val_ds.phys[fixtures["donor"][k]]
                              for k in chunk])

        rec_c = schema.batch(phys_c, device=device)
        b_c = teacher.forward_batch(rec_c)
        b_s = teacher.forward_batch(schema.batch(phys_s, device=device))
        meta_sum += float(decoder.loss(b_c, rec_c)) * len(chunk)
        n_meta += len(chunk)
        tok = {"correct": projector(b_c), "shuffled": projector(b_s),
               "none": projector.null_tokens(len(chunk))}

        for sigma in sigmas:
            x_t = (1.0 - sigma) * x0 + sigma * noise
            v_target = noise - x0
            t = torch.full((len(chunk),), sigma * 1000.0, device=device)
            for cond in ("correct", "shuffled", "none"):
                with torch.autocast("cuda", torch.bfloat16):
                    pred = model(x=list(x_t), t=t, context=context,
                                 seq_len=seq_len_of(x0),
                                 physics_ctx=tok[cond])
                pred = torch.stack([p.float() for p in pred])
                mse = (pred - v_target.float()).pow(2).flatten(1).mean(1)
                for k, m in zip(chunk, mse):
                    per_ep[k][cond] += m.item() / len(sigmas)

    if was_training:
        model.train(), teacher.train(), projector.train()
    per_ep = [{"key": k, **v} for k, v in per_ep.items()]
    return per_ep, meta_sum / max(n_meta, 1)


def run_eval(step, model, teacher, decoder, projector, schema, val_ds,
             fixtures, cfg, device, out_dir, stride, with_lora_off=False):
    t0 = time.time()
    per_ep, l_meta = paired_eval(
        model, teacher, decoder, projector, schema, val_ds, fixtures,
        device, cfg["shift"], cfg.get("eval_batch_size", 8), stride)
    summary = summarize(per_ep, val_ds.clusters,
                        iters=cfg.get("bootstrap_iters", 2000))
    summary.update(step=step, l_meta_val=l_meta, stride=stride,
                   eval_seconds=round(time.time() - t0))
    out = {"summary": summary}

    if with_lora_off:                       # adapter-only attribution pass
        model.set_lora_enabled(False)
        per_off, _ = paired_eval(
            model, teacher, decoder, projector, schema, val_ds, fixtures,
            device, cfg["shift"], cfg.get("eval_batch_size", 8), stride)
        model.set_lora_enabled(True)
        out["summary_lora_off"] = summarize(
            per_off, val_ds.clusters, iters=cfg.get("bootstrap_iters", 2000))

    line = {k: summary.get(k) for k in
            ("step", "loss_correct", "loss_shuffled", "loss_none",
             "gap_paired", "l_meta_val")}
    if "summary_lora_off" in out:
        line["gap_paired_lora_off"] = out["summary_lora_off"].get(
            "gap_paired")
    print(f"[eval] {json.dumps(line)}", flush=True)
    with open(Path(out_dir) / "eval_log.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    return summary


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def save_checkpoint(out_dir, step, cfg, model, teacher, decoder, projector,
                    opt, sched, ema_fm):
    d = Path(out_dir) / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "config": cfg, "ema_fm": ema_fm,
        "adaptive": model.adaptive_state_dict(),
        "teacher": teacher.state_dict(),
        "decoder": decoder.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "python": random.getstate(),
                "numpy": np.random.get_state()},
    }, d / "trainer.pt")
    print(f"saved {d}", flush=True)


def latest_checkpoint(out_dir):
    cks = sorted(Path(out_dir).glob("ckpt_*/trainer.pt"))
    return cks[-1] if cks else None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda")
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SmokeDataset(cfg["cache_dir"], "train")
    val_ds = SmokeDataset(cfg["cache_dir"], "val")
    registry = MetadataRegistry()
    with open(Path(cfg["cache_dir"]) / "norm_stats.json") as f:
        fields = json.load(f)["fields_kept"]
    schema = ProjectileRecordSchema(fields, registry)
    phys_matrix = train_ds.phys_matrix()
    print(f"train {len(train_ds)} episodes, val {len(val_ds)}, "
          f"{len(fields)} records/episode", flush=True)

    model, teacher, decoder, projector = build_models(cfg, schema, device)
    opt, sched = build_optimizer(cfg, model, teacher, decoder, projector)
    fixtures = load_or_build_fixtures(
        val_ds, cfg["cache_dir"], cfg.get("u_grid", DEFAULT_U_GRID),
        cfg.get("noise_seed", 1234))

    step, ema_fm = 0, None
    ck = latest_checkpoint(out_dir)
    if ck:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_adaptive_state_dict(state["adaptive"])
        teacher.load_state_dict(state["teacher"])
        decoder.load_state_dict(state["decoder"])
        projector.load_state_dict(state["projector"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step, ema_fm = state["step"], state["ema_fm"]
        torch.set_rng_state(state["rng"]["torch"])
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        print(f"resumed from {ck} at step {step}", flush=True)

    if cfg.get("w_rank"):
        print("NOTE: w_rank is set in the config but the rank hinge is "
              "monitor-only now; it is NOT optimized (see module docstring).",
              flush=True)

    csv_path = out_dir / "train_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["step", "loss", "fm", "ema_fm", "rank_gap", "meta", "cond",
                 "gate_mean", "gate_max", "lora_b_norm", "lr", "sec"])

    perm = EpochPermutation(len(train_ds), cfg.get("seed", 0))
    accum = cfg.get("grad_accum", 8)
    bs = cfg.get("batch_size", 1)
    trainable = [p for g in opt.param_groups for p in g["params"]]
    model.train(), teacher.train(), decoder.train(), projector.train()

    if step == 0:
        run_eval(0, model, teacher, decoder, projector, schema, val_ds,
                 fixtures, cfg, device, out_dir,
                 stride=cfg.get("subset_stride", 3), with_lora_off=True)

    monitor_interval = cfg.get("rank_monitor_interval", 10)
    while step < cfg["max_steps"]:
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        parts_acc, conds = {"fm": 0.0, "meta": 0.0}, []
        gap_vals = []
        monitor_step = monitor_interval and (step + 1) % monitor_interval == 0
        for micro in range(accum):
            idx = [perm.index_at((step * accum + micro) * bs + i)
                   for i in range(bs)]
            items = [train_ds[i] for i in idx]
            batch = {"latent": torch.stack([it["latent"] for it in items]),
                     "t5": [it["t5"] for it in items],
                     "phys": torch.stack([it["phys"] for it in items])}
            loss, parts, cond = train_forward(
                model, teacher, decoder, projector, schema, batch, cfg,
                device, phys_matrix,
                monitor_rank=monitor_step and not gap_vals)
            (loss / accum).backward()
            conds.append(cond)
            parts_acc["fm"] += parts["fm"] / accum
            parts_acc["meta"] += parts["meta"] / accum
            if parts["rank_gap"] is not None:
                gap_vals.append(parts["rank_gap"])
        torch.nn.utils.clip_grad_norm_(trainable, cfg.get("grad_clip", 1.0))
        opt.step(), sched.step()
        step += 1

        ema_fm = (parts_acc["fm"] if ema_fm is None else
                  ema_fm + (parts_acc["fm"] - ema_fm) * 2 / (EMA_SPAN + 1))
        g_mean, g_max = model.physics.gate_stats()
        b_norm = float(torch.stack(
            [p.detach().norm() for p in model.lora_parameters()[1::2]]
        ).mean())
        rank_gap = round(sum(gap_vals) / len(gap_vals), 5) if gap_vals else ""
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [step, round(float(parts_acc["fm"] + parts_acc["meta"]), 5),
                 round(parts_acc["fm"], 5), round(ema_fm, 5),
                 rank_gap,
                 round(parts_acc["meta"], 5), "|".join(conds),
                 f"{g_mean:.5f}", f"{g_max:.5f}", f"{b_norm:.5f}",
                 sched.get_last_lr()[1], round(time.time() - t0, 2)])
        if step % cfg.get("log_interval", 10) == 0:
            gap_text = f"{rank_gap}" if gap_vals else "-"
            print(f"step {step} fm {parts_acc['fm']:.4f} ema {ema_fm:.4f} "
                  f"meta {parts_acc['meta']:.4f} "
                  f"gap {gap_text} "
                  f"gates {g_mean:.4f}/{g_max:.4f} loraB {b_norm:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)

        if step % cfg.get("eval_interval", 250) == 0 or \
                step == cfg["max_steps"]:
            final = step == cfg["max_steps"]
            run_eval(step, model, teacher, decoder, projector, schema,
                     val_ds, fixtures, cfg, device, out_dir,
                     stride=cfg.get("subset_stride", 3),
                     with_lora_off=final)
            save_checkpoint(out_dir, step, cfg, model, teacher, decoder,
                            projector, opt, sched, ema_fm)
        elif step % cfg.get("ckpt_interval", 100) == 0:
            # eval-free saves so 6h gpu_devel sessions can resume past step<250
            save_checkpoint(out_dir, step, cfg, model, teacher, decoder,
                            projector, opt, sched, ema_fm)

    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
