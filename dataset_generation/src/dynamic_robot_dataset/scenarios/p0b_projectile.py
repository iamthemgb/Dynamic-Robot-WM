"""P0b — passive ballistic projectile."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("P0b", "passive_no_franka", "projectile", fixture_policy="grounded support plane", controller_kind="passive", trajectory="none", retention_required=False)
