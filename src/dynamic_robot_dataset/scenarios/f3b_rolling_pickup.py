"""F3b — ball rolling on a physical table followed by pickup."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("F3b", "dynamic_handoff", "rolling_pickup", fixture_policy="owned R0 table or audited RoboCasa rolling island", controller_kind="pickup_or_transport", trajectory="jerk_limited_predictive_reach", retention_required=True, hand_orientation="pick_down", robotiq_tendon_profile="pickup")
