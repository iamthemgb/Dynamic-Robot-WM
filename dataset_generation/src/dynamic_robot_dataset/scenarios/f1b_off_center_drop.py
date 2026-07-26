"""F1b — off-center vertical catch and near miss."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("F1b", "falling_catch", "off_center_drop", fixture_policy="fixtureless free-space interception", controller_kind="catch", trajectory="jerk_limited_predictive_reach", retention_required=True)
