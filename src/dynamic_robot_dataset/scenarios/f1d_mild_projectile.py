"""F1d — mild projectile catch and near miss."""
from ._rigid_shared import projectile_randomization_contract, rigid_module

SCENARIO = rigid_module(
    "F1d",
    "falling_catch",
    "mild_projectile",
    fixture_policy="fixtureless free-space interception",
    controller_kind="catch",
    trajectory="jerk_limited_predictive_reach",
    retention_required=True,
    randomization_contract=projectile_randomization_contract("F1d"),
    sampled_projectile_ready_offset_m=(0.0, 0.14, 0.13),
)
