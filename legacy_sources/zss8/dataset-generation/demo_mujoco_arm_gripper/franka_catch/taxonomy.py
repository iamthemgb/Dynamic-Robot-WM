"""Success/failure taxonomy for the ball-catch dataset.

Split (per the PI):
  50% clear success
  20% spatial near miss   (hand ends up near, ball just misses the pads)
  20% contact failure     (hand touches the ball but fails to secure it)
  10% wrong action        (arm idles / moves randomly / reaches the wrong way)

``sample_plan`` draws a branch and returns a :class:`~franka_catch.controller.CatchPlan`
plus (family, subfamily, failure_mode) labels for the metadata.
"""
from __future__ import annotations

import numpy as np

from .controller import CatchPlan

FAMILY = "F1"
SUBFAMILY = "F1_A_centered_vertical_drop"

# branch -> probability
BRANCH_SPLIT = {
    "success": 0.50,
    "spatial_near_miss": 0.20,
    "contact_failure": 0.20,
    "wrong_action": 0.10,
}

# branch -> failure_mode label recorded in metadata
FAILURE_MODE = {
    "success": "none",
    "spatial_near_miss": "spatial_near_miss",
    "contact_failure": "contact_slip",
    "wrong_action": "wrong_action",
}


def choose_branch(rng: np.random.Generator) -> str:
    names = list(BRANCH_SPLIT.keys())
    probs = np.array([BRANCH_SPLIT[n] for n in names], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def build_plan(branch: str, rng: np.random.Generator) -> CatchPlan:
    if branch == "success":
        return CatchPlan(branch="success")

    if branch == "spatial_near_miss":
        # Offset the intercept target laterally by more than the retention
        # tolerance (~0.05 m) but only by a near-miss margin.
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = float(rng.uniform(0.065, 0.13))
        return CatchPlan(
            branch="spatial_near_miss",
            lateral_miss=(mag * float(np.cos(ang)), mag * float(np.sin(ang))),
        )

    if branch == "contact_failure":
        mode = str(rng.choice(["wide_gap", "late_close", "shallow"]))
        plan = CatchPlan(branch="contact_failure", branch_mode=mode, disable_latch=True)
        if mode == "wide_gap":
            plan.gap_slip = float(rng.uniform(0.006, 0.013))  # pads too wide -> slip
        elif mode == "late_close":
            plan.close_lead = float(rng.uniform(-0.14, -0.06))  # shut after the ball hits
        elif mode == "shallow":
            # aim a little high so the ball strikes the fingertips and deflects
            plan.wrong_target_offset = (0.0, 0.0, float(rng.uniform(0.05, 0.09)))
            plan.gap_slip = float(rng.uniform(0.003, 0.008))
        return plan

    # wrong_action
    mode = str(rng.choice(["idle", "random", "wrong_way"]))
    plan = CatchPlan(branch="wrong_action", branch_mode=mode, disable_latch=True)
    if mode == "random":
        plan.random_joint_delta = tuple(float(v) for v in rng.uniform(-0.9, 0.9, size=7))
    elif mode == "wrong_way":
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = float(rng.uniform(0.28, 0.5))
        plan.wrong_target_offset = (mag * float(np.cos(ang)), mag * float(np.sin(ang)),
                                    float(rng.uniform(-0.1, 0.15)))
    return plan


def sample_plan(rng: np.random.Generator) -> tuple[str, CatchPlan, str]:
    branch = choose_branch(rng)
    return branch, build_plan(branch, rng), FAILURE_MODE[branch]
