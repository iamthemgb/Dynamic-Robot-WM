"""Retired custom-attachment MuJoCo regression backend.

This package is retained for low-level lifecycle/QC regression tests.  It
mounts synthetic task geometry on the Franka flange and is not a Panda-hand or
Robotiq production generator.  Public execution is blocked in the CLI.
"""

from .backend import NativeMuJoCoBackend
from .evaluators import (
    EVALUATOR_ID,
    EVALUATOR_VERSION,
    OBJECTIVE_THRESHOLD_SET_HASH,
    OBJECTIVE_THRESHOLDS,
    evaluate_saved_native_episode,
    get_objective_evaluator,
)
from .model import FrankaAsset, resolve_franka_asset
from .registration import recompute_native_objective, register_common_objective_evaluator
from .planning import (
    compile_native_plan,
    make_scenario_spec,
    make_scenario_spec_for_subfamily,
    resolve_scenario_alias,
    scenario_family,
    scenario_spec_from_episode_plan,
    scenario_to_episode_plan,
)

__all__ = [
    "FrankaAsset",
    "NativeMuJoCoBackend",
    "EVALUATOR_ID",
    "EVALUATOR_VERSION",
    "OBJECTIVE_THRESHOLD_SET_HASH",
    "OBJECTIVE_THRESHOLDS",
    "compile_native_plan",
    "evaluate_saved_native_episode",
    "get_objective_evaluator",
    "make_scenario_spec",
    "make_scenario_spec_for_subfamily",
    "resolve_franka_asset",
    "recompute_native_objective",
    "register_common_objective_evaluator",
    "resolve_scenario_alias",
    "scenario_family",
    "scenario_spec_from_episode_plan",
    "scenario_to_episode_plan",
]
