"""P0a — passive freefall."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("P0a", "passive_no_franka", "freefall", fixture_policy="grounded support plane", controller_kind="passive", trajectory="none", retention_required=False)
