"""Per-step logging for the stock phase modules, without editing them.

The DiT-free phases (0, 2, 4) are opaque ``for step in range(...)`` loops with
no logging hooks. Rather than touch five reviewed files, ``PhaseRecorder``
patches, only while active:

  * ``torch.Tensor.backward``     -- records the scalar loss of the iteration
  * ``torch.optim.AdamW.step``    -- closes the iteration: writes the CSV row

Every phase calls ``loss.backward()`` exactly once and ``opt.step()`` exactly
once per iteration (verified in all five modules), so this yields a faithful
per-step trace. Evaluation code runs under ``torch.no_grad`` and calls
neither, so it never pollutes the log.

This is a deliberate, contained monkeypatch: one module, one ``with`` block,
originals restored on exit. Phases 1 and 3 do NOT use it -- they have real
reimplemented loops in ``phase{1,3}_real.py`` with checkpoint/resume.
"""

import csv
import time
from pathlib import Path

import torch

CSV_COLUMNS = ["phase", "step", "wall_s", "loss", "fm", "ema_fm", "rank_gap",
               "meta", "distill", "query", "cond", "gate_mean", "gate_max",
               "lora_b_norm", "grad_norm", "lr_head", "lr_wan", "mem_gb"]


def open_train_log(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "train_log.csv"
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
    return path


def append_row(path, **kv):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([kv.get(c, "") for c in CSV_COLUMNS])


class PhaseRecorder:
    def __init__(self, out_dir, phase: str):
        self.path = open_train_log(out_dir)
        self.phase = phase
        self.step = 0
        self._loss = None
        self._extra = {}
        self._t0 = None
        self._start = None

    def note(self, **kv):
        self._extra.update(kv)

    def __enter__(self):
        self._orig_backward = torch.Tensor.backward
        self._orig_step = torch.optim.AdamW.step
        rec = self

        def backward(tensor, *a, **kw):
            if tensor.dim() == 0:
                rec._loss = float(tensor.detach())
                if rec._t0 is None:
                    rec._t0 = time.time()
            return rec._orig_backward(tensor, *a, **kw)

        def step(opt, *a, **kw):
            out = rec._orig_step(opt, *a, **kw)
            rec.step += 1
            append_row(rec.path, phase=rec.phase, step=rec.step,
                       wall_s=round(time.time() - rec._start, 2),
                       loss=(None if rec._loss is None
                             else round(rec._loss, 6)),
                       lr_head=opt.param_groups[0]["lr"],
                       mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)
                       if torch.cuda.is_available() else "",
                       **rec._extra)
            rec._loss, rec._t0 = None, None
            rec._extra = {}
            return out

        torch.Tensor.backward = backward
        torch.optim.AdamW.step = step
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        torch.Tensor.backward = self._orig_backward
        torch.optim.AdamW.step = self._orig_step
        return False
