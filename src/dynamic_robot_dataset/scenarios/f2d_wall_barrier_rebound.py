"""F2d — wall or angled-barrier rebound followed by interception."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("F2d", "projectile_rebound", "wall_barrier_rebound", fixture_policy="owned grounded calibrated rebound wall", controller_kind="catch_after_rebound", trajectory="jerk_limited_predictive_reach", retention_required=True, hand_orientation="catch_up", settled_aim_correction=True, interior_joint_margin_rad=0.01)
