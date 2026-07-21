"""F1a — centered vertical free-contact catch."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("F1a", "falling_catch", "centered_vertical_drop", fixture_policy="fixtureless free-space interception", controller_kind="catch_or_transport", trajectory="jerk_limited_predictive_reach", retention_required=True)
