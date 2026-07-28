"""One-PNG deliverable: training + eval curves for the smoke run.

Panels:
  1. train L_fm per step (raw, faint) + EMA-100. Read as an INFRA CHECK only:
     at this scale the LoRA lowers train L_fm via style adaptation and
     partial memorization even if the physics tokens are ignored.
  2. L_phys_recon (aux) - should drop 1-2 orders of magnitude early.
  3. tanh gate value per step.
  4. val L_fm: {correct, shuffled, none} + paired gap with bootstrap 95% CI
     band - the actual evidence the embedding is consumed.

  python projectile_smoke/eval/plot_curves.py --run-dir <out_dir>
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_run(run_dir):
    run_dir = Path(run_dir)
    rows = list(csv.DictReader(open(run_dir / "train_log.csv")))
    steps = [int(r["step"]) for r in rows]
    fm = [float(r["loss_fm"]) for r in rows]
    ema = [float(r["ema_fm"]) for r in rows]
    aux = [float(r["loss_aux"]) for r in rows]
    gate = [float(r["gate"]) for r in rows]

    evals = []
    hist = run_dir / "eval_history.jsonl"
    if hist.exists():
        for line in hist.read_text().strip().splitlines():
            evals.append(json.loads(line)["summary"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"projectile smoke run - {run_dir.name}")

    ax = axes[0, 0]
    ax.plot(steps, fm, lw=0.4, alpha=0.35, color="tab:blue", label="L_fm (per step)")
    ax.plot(steps, ema, lw=1.8, color="tab:blue", label="EMA-100")
    ax.set_title("train flow-matching loss (infra check only)")
    ax.set_xlabel("step")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, aux, lw=0.8, color="tab:orange")
    ax.set_yscale("log")
    ax.set_title("L_phys_recon (aux, log scale)")
    ax.set_xlabel("step")

    ax = axes[1, 0]
    ax.plot(steps, gate, lw=1.2, color="tab:green")
    ax.set_title("tanh gate")
    ax.set_xlabel("step")

    ax = axes[1, 1]
    if evals:
        es = [e["step"] for e in evals]
        for cond, color in [("loss_correct", "tab:blue"),
                            ("loss_shuffled", "tab:red"),
                            ("loss_none", "tab:gray")]:
            ax.plot(es, [e[cond] for e in evals], "o-", ms=3, lw=1.2,
                    color=color, label=cond.replace("loss_", "val "))
        ax2 = ax.twinx()
        gap = [e["gap_paired"] for e in evals]
        lo = [e["gap_ci95"][0] for e in evals]
        hi = [e["gap_ci95"][1] for e in evals]
        ax2.axhline(0, color="k", lw=0.6, ls=":")
        ax2.fill_between(es, lo, hi, color="tab:purple", alpha=0.18)
        ax2.plot(es, gap, "s--", ms=4, lw=1.4, color="tab:purple",
                 label="paired gap (right axis)")
        ax2.set_ylabel("gap = L(shuffled) - L(correct)", color="tab:purple")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.set_title("paired shuffle-gap eval (fixed noise/sigma fixtures)")
    ax.set_xlabel("step")

    fig.tight_layout()
    png = run_dir / "curves.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"wrote {png}")
    return png


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    plot_run(ap.parse_args().run_dir)
