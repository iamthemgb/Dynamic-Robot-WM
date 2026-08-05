"""Figures and tables for the physics_wan campaign.

    python -m generalized_physics.real.plots            # all arms found
    python -m generalized_physics.real.plots --arms wan21_t2v_1p3b ...

Reads ``runs/<arm>/phase*/train_log.csv`` + ``eval_log.jsonl`` +
``summary.json``; writes ``figures/*.png`` (with the exact source frame of
each figure under ``figures/data/*.csv``) and ``tables/*.csv``.

Style follows the house template
(/scratch/zl664_yale/GJEPA_muon/scripts/figure_style_common.py); its constants
are inlined here so a CPU-partition job needs nothing outside this repo.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .paths import ARMS, OUT_ROOT

PANEL_FIGSIZE = (7.4, 5.2)
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 18, 16, 10, 12
GRID_ALPHA = 0.18

ARM_LABEL = {"wan21_t2v_1p3b": "Wan2.1 T2V 1.3B",
             "wan22_ti2v_5b": "Wan2.2 TI2V 5B",
             "wan21_t2v_14b": "Wan2.1 T2V 14B"}
ARM_COLOR = {"wan21_t2v_1p3b": "#3567A8",
             "wan22_ti2v_5b": "#4E9151",
             "wan21_t2v_14b": "#D4583C"}
ARM_PARAMS = {"wan21_t2v_1p3b": 1.42, "wan22_ti2v_5b": 5.02,
              "wan21_t2v_14b": 14.33}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.5)
        ax.spines[side].set_color("black")
    ax.tick_params(width=1.0, colors="black", labelsize=TICK_FS)
    ax.grid(alpha=GRID_ALPHA, linewidth=0.8)
    ax.set_axisbelow(True)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_evals(path, phase):
    if not Path(path).exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("phase") == phase:
                out.append(d)
    return out


def ffloat(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def series(rows, col):
    xs, ys = [], []
    for r in rows:
        y = ffloat(r.get(col))
        if y is not None:
            xs.append(int(r["step"]))
            ys.append(y)
    return xs, ys


def dump_frame(fig_dir, name, header, rows):
    d = fig_dir / "data"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save(fig, fig_dir, name):
    fig.savefig(fig_dir / f"{name}.png", dpi=240, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{name}.png")


def fig_phase_loss(arms, runs, fig_dir, phase, col, title, ylabel, name,
                   logy=False, ema_col=None):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    frame = []
    for arm in arms:
        rows = runs[arm].get(phase)
        if not rows:
            continue
        xs, ys = series(rows, col)
        if not xs:
            continue
        ax.plot(xs, ys, color=ARM_COLOR[arm], alpha=0.25, linewidth=0.9)
        if ema_col:
            xe, ye = series(rows, ema_col)
            ax.plot(xe, ye, color=ARM_COLOR[arm], linewidth=2.2,
                    label=ARM_LABEL[arm])
        else:
            ax.plot(*_smooth(xs, ys), color=ARM_COLOR[arm], linewidth=2.2,
                    label=ARM_LABEL[arm])
        frame += [[arm, x, y] for x, y in zip(xs, ys)]
    if not frame:
        return
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold")
    ax.set_xlabel("step", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, fontweight="bold")
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=LEGEND_FS, frameon=False)
    style(ax)
    dump_frame(fig_dir, name, ["arm", "step", col], frame)
    save(fig, fig_dir, name)


def _smooth(xs, ys, k=25):
    if len(ys) <= k:
        return xs, ys
    out = []
    acc = 0.0
    for i, y in enumerate(ys):
        acc += y
        if i >= k:
            acc -= ys[i - k]
        out.append(acc / min(i + 1, k))
    return xs, out


def fig_gap_curves(arms, runs_dir, fig_dir):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    frame = []
    for arm in arms:
        evs = read_evals(runs_dir / arm / "phase1" / "eval_log.jsonl", 1)
        if not evs:
            continue
        xs = [e["step"] for e in evs]
        gw = [e["metrics"]["gap_wrong"] for e in evs]
        gn = [e["metrics"]["gap_null"] for e in evs]
        ax.plot(xs, gw, color=ARM_COLOR[arm], linewidth=2.2, marker="o",
                label=f"{ARM_LABEL[arm]}  gap(wrong)")
        ax.plot(xs, gn, color=ARM_COLOR[arm], linewidth=1.6, marker="s",
                linestyle="--", label=f"{ARM_LABEL[arm]}  gap(null)")
        frame += [[arm, x, w, n] for x, w, n in zip(xs, gw, gn)]
    if not frame:
        return
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Phase 1 — paired conditioning gap", fontsize=TITLE_FS,
                 fontweight="bold")
    ax.set_xlabel("step", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("flow-loss gap (shared $\\sigma,\\epsilon$)",
                  fontsize=LABEL_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS - 2, frameon=False)
    style(ax)
    dump_frame(fig_dir, "fig02_phase1_paired_gap",
               ["arm", "step", "gap_wrong", "gap_null"], frame)
    save(fig, fig_dir, "fig02_phase1_paired_gap")


def fig_scale_summary(arms, runs_dir, fig_dir):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    frame = []
    for arm in arms:
        p = runs_dir / arm / "phase1" / "summary.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())["metrics"]
        x = ARM_PARAMS[arm]
        ax.scatter([x], [m["gap_wrong"]], s=140, color=ARM_COLOR[arm],
                   zorder=3, label=ARM_LABEL[arm])
        frame.append([arm, x, m["gap_wrong"], m["gap_null"]])
    if not frame:
        return
    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Conditioning gap vs backbone size", fontsize=TITLE_FS,
                 fontweight="bold")
    ax.set_xlabel("backbone parameters (B)", fontsize=LABEL_FS,
                  fontweight="bold")
    ax.set_ylabel("final gap(wrong)", fontsize=LABEL_FS, fontweight="bold")
    ax.legend(fontsize=LEGEND_FS, frameon=False)
    style(ax)
    dump_frame(fig_dir, "fig08_scale_summary",
               ["arm", "params_b", "gap_wrong", "gap_null"], frame)
    save(fig, fig_dir, "fig08_scale_summary")


def tables(arms, runs_dir, out_root):
    tdir = out_root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    gate_rows, metric_rows, thr_rows = [], [], []
    for arm in arms:
        for ph in range(5):
            p = runs_dir / arm / f"phase{ph}" / "summary.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            for k, v in s.get("gates", {}).items():
                gate_rows.append([arm, ph, k, v])
            for k, v in s.get("metrics", {}).items():
                if isinstance(v, (int, float)):
                    metric_rows.append([arm, ph, k, v])
            thr_rows.append([arm, ph, s.get("wall_s"), s.get("peak_mem_gb")])
    for name, header, rows in (
            ("gate_table", ["arm", "phase", "gate", "passed"], gate_rows),
            ("final_metrics", ["arm", "phase", "metric", "value"],
             metric_rows),
            ("throughput", ["arm", "phase", "wall_s", "peak_mem_gb"],
             thr_rows)):
        with open(tdir / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"wrote tables/{name}.csv ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=sorted(ARMS))
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    runs_dir = out_root / "runs"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms if (runs_dir / a).exists()]
    print(f"arms with runs: {arms}")

    runs = {arm: {ph: (read_csv(runs_dir / arm / f"phase{ph}" /
                                "train_log.csv")
                       if (runs_dir / arm / f"phase{ph}" /
                           "train_log.csv").exists() else None)
                  for ph in range(5)} for arm in arms}

    fig_phase_loss(arms, runs, fig_dir, 1, "fm",
                   "Phase 1 — flow-matching loss", "L_FM",
                   "fig01_phase1_flow_loss", ema_col="ema_fm")
    fig_gap_curves(arms, runs_dir, fig_dir)
    fig_phase_loss(arms, runs, fig_dir, 0, "loss",
                   "Phase 0 — representation probe", "probe loss",
                   "fig03_phase0_probe", logy=True)
    fig_phase_loss(arms, runs, fig_dir, 2, "loss",
                   "Phase 2 — student distillation", "L_distill + w·L_query",
                   "fig04_phase2_distill", logy=True)
    fig_phase_loss(arms, runs, fig_dir, 3, "distill",
                   "Phase 3 — student substitution", "L_distill",
                   "fig05_phase3_distill")
    fig_scale_summary(arms, runs_dir, fig_dir)
    tables(arms, runs_dir, out_root)


if __name__ == "__main__":
    main()
