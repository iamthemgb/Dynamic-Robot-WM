"""End-to-end CPU smoke: phases 0-4 chained at the given scale.

    python -m generalized_physics.run_smoke_e2e          # tiny (~minutes)
    python -m generalized_physics.run_smoke_e2e smoke    # full smoke config

This validates integration, not learning: at tiny step counts the gates are
not expected to pass, only to be computed. The printed metrics map onto the
plan's gate table (phase 0 controls, phase 1 correct/wrong/null separation,
phase 3 oracle-gap closure, phase 4 recovery profile).
"""

import sys
import time

import torch

from .config import smoke_config, tiny_config
from .training import (phase0_representation_probe, phase1_oracle_wan,
                       phase2_student_distillation,
                       phase3_student_substitution,
                       phase4_sliding_adaptation)


def main(scale="tiny"):
    cfg = tiny_config() if scale == "tiny" else smoke_config()
    torch.set_num_threads(cfg.num_threads)
    t0 = time.time()

    p0 = phase0_representation_probe.run(cfg)
    print(f"[phase0 {time.time() - t0:.0f}s] {p0['metrics']}")

    p1 = phase1_oracle_wan.run(cfg, cache=p0["cache"],
                               registry=p0["registry"],
                               normalizer=p0["normalizer"])
    print(f"[phase1 {time.time() - t0:.0f}s] {p1['metrics']}")

    p2 = phase2_student_distillation.run(cfg, p1, student=p0["student"])
    print(f"[phase2 {time.time() - t0:.0f}s] {p2['metrics']}")

    p3 = phase3_student_substitution.run(cfg, p1, p2)
    print(f"[phase3 {time.time() - t0:.0f}s] {p3['metrics']}")

    p4 = phase4_sliding_adaptation.run(cfg, p1, p3["student"])
    m4 = {w: {k: v for k, v in mw.items() if k != "profile"}
          for w, mw in p4["metrics"].items()}
    print(f"[phase4 {time.time() - t0:.0f}s] {m4}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny")
