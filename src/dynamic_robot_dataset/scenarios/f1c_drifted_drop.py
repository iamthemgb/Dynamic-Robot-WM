"""F1c — laterally drifted freefall catch."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("F1c", "falling_catch", "drifted_drop", fixture_policy="fixtureless free-space interception", controller_kind="catch", trajectory="jerk_limited_predictive_reach", retention_required=True)
