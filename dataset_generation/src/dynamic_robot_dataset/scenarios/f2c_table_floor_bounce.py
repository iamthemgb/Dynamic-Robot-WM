"""F2c — calibrated table/floor bounce followed by interception."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("F2c", "projectile_rebound", "table_floor_bounce", fixture_policy="owned grounded calibrated bounce pad", controller_kind="catch_after_rebound", trajectory="jerk_limited_predictive_reach", retention_required=True, hand_orientation="pick_down", settled_aim_correction=True, robotiq_tendon_profile="f2c", interior_joint_margin_rad=0.01)
