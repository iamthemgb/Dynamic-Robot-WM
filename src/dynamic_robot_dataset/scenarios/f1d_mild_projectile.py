"""F1d — mild projectile catch and near miss."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("F1d", "falling_catch", "mild_projectile", fixture_policy="fixtureless free-space interception", controller_kind="catch", trajectory="jerk_limited_predictive_reach", retention_required=True)
