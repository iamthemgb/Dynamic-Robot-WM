"""F2a — direct projectile catch or impulse-consistent deflection."""
from ._rigid_shared import projectile_randomization_contract, rigid_module

SCENARIO = rigid_module(
    "F2a",
    "projectile_rebound",
    "direct_interception",
    fixture_policy="fixtureless direct interception",
    controller_kind="catch_or_deflect",
    trajectory="jerk_limited_predictive_reach",
    retention_required=False,
    randomization_contract=projectile_randomization_contract("F2a"),
    sampled_projectile_ready_offset_m=(0.0, 0.14, 0.13),
)
