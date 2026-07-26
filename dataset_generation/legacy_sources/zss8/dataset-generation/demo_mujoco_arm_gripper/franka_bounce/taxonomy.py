"""Success/failure taxonomy for the ground-bounce catch subfamily (F1_B).

Same 50/20/20/10 branch split and CatchPlan machinery as F1_A; only the
subfamily label differs. The plan offsets/timing apply to the *post-bounce*
intercept target chosen reactively by ``BounceCatchController``.
"""
from __future__ import annotations

from franka_catch.taxonomy import (  # noqa: F401  (re-exported)
    BRANCH_SPLIT,
    FAILURE_MODE,
    build_plan,
    choose_branch,
    sample_plan,
)

FAMILY = "F1"
SUBFAMILY = "F1_B_ground_bounce_catch"
