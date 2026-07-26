"""P0c — passive table and wall rebound."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("P0c", "passive_no_franka", "bounce_rebound", fixture_policy="grounded rebound table or wall", controller_kind="passive", trajectory="none", retention_required=False)
