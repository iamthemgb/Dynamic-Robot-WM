"""P1 pre-flight (GPU): overfit 4 episodes + save/resume continuity.

If a model can't overfit 4 clips, no 2k-step curve is trustworthy. Setup is
deliberately deterministic - fixed noise per episode, fixed sigma (u=0.5),
the same 4-episode batch every step - so the objective is a fixed function
and the loss must fall fast. Uses the SAME build/save/load code paths as
train_smoke.py, so this also exercises success criterion 5
(checkpoint save -> resume -> loss continues).

Pass criteria:
  - final loss < 0.5 x initial loss (overfit works)
  - first post-resume loss within 15% of the last pre-save loss (resume works)

Run: ~/.venvs/wan21/bin/python projectile_smoke/tests/test_overfit.py \
        --config projectile_smoke/configs/smoke_lora.yaml [--steps 60]
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from projectile_smoke.training.flow_match import flow_loss, shift_sigma  # noqa: E402

from projectile_smoke.training.dataset import SmokeDataset, collate  # noqa: E402
from projectile_smoke.training.train_smoke import (  # noqa: E402
    build_model, build_optimizer, init_base_tokens, load_checkpoint,
    save_checkpoint, seq_len_of)

OVERFIT_LR_SCALE = 10.0  # 4 clips, deterministic objective: crank the LR


def fixed_batch(cfg, device):
    ds = SmokeDataset(cfg["cache_dir"], "train")
    idx = [0, len(ds) // 3, 2 * len(ds) // 3, len(ds) - 1]
    batch = collate([ds[i] for i in idx])
    gen = torch.Generator().manual_seed(7)
    noise = torch.randn(batch["latent"].shape, generator=gen)
    return ds, {
        "latent": batch["latent"].to(device),
        "t5": [u.to(device) for u in batch["t5"]],
        "phys": batch["phys"].to(device),
        "noise": noise.to(device),
    }


def loss_of(model, fb, cfg, device):
    sigma = shift_sigma(0.5, cfg["shift"])
    x0 = fb["latent"]
    x_t = (1 - sigma) * x0 + sigma * fb["noise"]
    v_target = fb["noise"] - x0
    t = torch.full((x0.size(0),), sigma * 1000.0, device=device)
    with torch.autocast("cuda", torch.bfloat16):
        pred, aux = model(x=list(x_t), t=t, context=fb["t5"],
                          seq_len=seq_len_of(x0), phys_vec=fb["phys"])
    fm = flow_loss(pred, v_target)
    return fm + cfg["aux_weight"] * torch.nn.functional.mse_loss(
        aux.float(), fb["phys"])


def build_all(cfg, phys_dim, device, train_keys):
    model, encoder, lora_params = build_model(cfg, phys_dim, device)
    init_base_tokens(encoder, cfg, train_keys)
    optimizer, scheduler = build_optimizer(cfg, encoder, lora_params)
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    return model, encoder, optimizer, scheduler, trainable


def run_steps(model, opt, sched, trainable, fb, cfg, device, n):
    losses = []
    model.train()
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        loss = loss_of(model, fb, cfg, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        opt.step()
        sched.step()
        losses.append(loss.item())
        print(f"  step {len(losses)}: loss {losses[-1]:.4f}", flush=True)
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "projectile_smoke/configs/smoke_lora.yaml"))
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg)
    cfg["encoder_lr"] *= OVERFIT_LR_SCALE
    cfg["warmup_steps"] = 5
    if cfg.get("lora", {}).get("enabled"):
        cfg["lora"] = dict(cfg["lora"], lr=cfg["lora"]["lr"] * OVERFIT_LR_SCALE)
    out_dir = Path(cfg["out_dir"]).parent / "overfit_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("ckpt_*/trainer.pt"):
        old.unlink()
    cfg["out_dir"] = str(out_dir)
    device = torch.device("cuda")
    torch.manual_seed(0)

    ds, fb = fixed_batch(cfg, device)
    model, encoder, opt, sched, trainable = build_all(
        cfg, ds.phys_dim, device, ds.keys)

    half = args.steps // 2
    print(f"phase 1: {half} steps")
    losses1 = run_steps(model, opt, sched, trainable, fb, cfg, device, half)
    save_checkpoint(out_dir, half, cfg, encoder, model, opt, sched,
                    ema_fm=losses1[-1], killrule_ref={})

    print("rebuilding everything from scratch and resuming...")
    del model, encoder, opt, sched, trainable
    torch.cuda.empty_cache()
    model, encoder, opt, sched, trainable = build_all(
        cfg, ds.phys_dim, device, ds.keys)
    state = load_checkpoint(out_dir / f"ckpt_{half:06d}", encoder, model,
                            opt, sched)
    assert state["step"] == half

    print(f"phase 2 (resumed): {args.steps - half} steps")
    losses2 = run_steps(model, opt, sched, trainable, fb, cfg, device,
                        args.steps - half)

    initial, pre_save, post_resume, final = (
        losses1[0], losses1[-1], losses2[0], min(losses2))
    resume_jump = abs(post_resume - pre_save) / pre_save
    print(f"initial {initial:.4f} -> pre-save {pre_save:.4f} | "
          f"post-resume {post_resume:.4f} (jump {resume_jump:.1%}) -> "
          f"best {final:.4f}")

    ok_overfit = final < 0.5 * initial
    ok_resume = resume_jump < 0.15
    print(f"overfit: {'PASS' if ok_overfit else 'FAIL'} "
          f"(final/initial = {final / initial:.2f}, need < 0.50)")
    print(f"resume continuity: {'PASS' if ok_resume else 'FAIL'} "
          f"(jump {resume_jump:.1%}, need < 15%)")
    if not (ok_overfit and ok_resume):
        sys.exit(1)
    print("OVERFIT + RESUME TEST PASSED")


if __name__ == "__main__":
    main()
