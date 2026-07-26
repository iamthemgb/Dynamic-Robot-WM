from __future__ import annotations

import numpy as np

from .controller import RobotiqCatchPlan


FAMILY = "F1"
SUBFAMILIES = {
    "centered_vertical_drop": {
        "label": "F1_A_centered_vertical_drop",
        "task_index": 0,
        "task": "Catch a centered vertically dropping ball with the Franka arm and Robotiq thick-pad gripper.",
    },
    "direct_projectile_interception": {
        "label": "F1_B_direct_projectile_interception",
        "task_index": 1,
        "task": "Intercept a laterally moving projectile ball with the Franka arm and Robotiq thick-pad gripper.",
    },
}

BRANCH_SPLIT = {
    "success": 0.50,
    "spatial_near_miss": 0.20,
    "contact_failure": 0.20,
    "wrong_action": 0.10,
}

FAILURE_MODE = {
    "success": "none",
    "spatial_near_miss": "spatial_near_miss",
    "contact_failure": "contact_slip",
    "wrong_action": "wrong_action",
}


def branch_sequence(n: int, seed: int) -> list[str]:
    names = list(BRANCH_SPLIT)
    weights = np.asarray([BRANCH_SPLIT[name] for name in names], dtype=np.float64)
    raw = weights / weights.sum() * int(n)
    counts = np.floor(raw).astype(int)
    remainder = int(n) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[int(idx)] += 1

    branches: list[str] = []
    for name, count in zip(names, counts):
        branches.extend([name] * int(count))
    rng = np.random.default_rng(seed)
    rng.shuffle(branches)
    return branches


def build_plan(branch: str, rng: np.random.Generator) -> RobotiqCatchPlan:
    if branch == "success":
        return RobotiqCatchPlan(branch="success")

    if branch == "spatial_near_miss":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = float(rng.uniform(0.075, 0.135))
        return RobotiqCatchPlan(
            branch="spatial_near_miss",
            lateral_miss=(mag * float(np.cos(angle)), mag * float(np.sin(angle))),
            disable_latch=True,
        )

    if branch == "contact_failure":
        mode = str(rng.choice(["wide_gap", "late_close", "shallow"]))
        if mode == "wide_gap":
            return RobotiqCatchPlan(
                branch="contact_failure",
                branch_mode=mode,
                close_fraction=float(rng.uniform(0.58, 0.74)),
                disable_latch=True,
            )
        if mode == "late_close":
            return RobotiqCatchPlan(
                branch="contact_failure",
                branch_mode=mode,
                close_lead_time_s=float(rng.uniform(-0.13, -0.06)),
                disable_latch=True,
            )
        return RobotiqCatchPlan(
            branch="contact_failure",
            branch_mode=mode,
            target_offset=(0.0, 0.0, float(rng.uniform(0.050, 0.090))),
            close_fraction=float(rng.uniform(0.72, 0.88)),
            disable_latch=True,
        )

    mode = str(rng.choice(["idle", "random", "wrong_way"]))
    if mode == "random":
        return RobotiqCatchPlan(
            branch="wrong_action",
            branch_mode=mode,
            random_joint_delta=tuple(float(v) for v in rng.uniform(-0.85, 0.85, size=7)),
            disable_latch=True,
        )
    if mode == "wrong_way":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        mag = float(rng.uniform(0.28, 0.50))
        return RobotiqCatchPlan(
            branch="wrong_action",
            branch_mode=mode,
            target_offset=(mag * float(np.cos(angle)), mag * float(np.sin(angle)), float(rng.uniform(-0.10, 0.14))),
            disable_latch=True,
        )
    return RobotiqCatchPlan(branch="wrong_action", branch_mode="idle", disable_latch=True)

