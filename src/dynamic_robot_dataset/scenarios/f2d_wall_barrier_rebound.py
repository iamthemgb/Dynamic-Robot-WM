"""F2d — wall or angled-barrier rebound followed by interception."""

from ._rigid_shared import rigid_module


# The former +100 mm X timing intervention put the Panda link-6 shell in the
# projectile lane and left the Robotiq target outside the settled-IK gate.
# The fixed -100 mm Y offset is an honest miss away from both the rebound lane
# and the supported wall.  It changes only the declared controller-negative
# branch; nominal and initial-state-negative cases keep their original target.
NEGATIVE_CONTROLLER_OFFSET_M = (0.0, -0.10, 0.0)


SCENARIO = rigid_module(
    "F2d",
    "projectile_rebound",
    "wall_barrier_rebound",
    fixture_policy="owned grounded calibrated rebound wall",
    controller_kind="catch_after_rebound",
    trajectory="jerk_limited_predictive_reach",
    retention_required=True,
    hand_orientation="catch_up",
    settled_aim_correction=True,
    negative_controller_offset_m=NEGATIVE_CONTROLLER_OFFSET_M,
    interior_joint_margin_rad=0.01,
)
