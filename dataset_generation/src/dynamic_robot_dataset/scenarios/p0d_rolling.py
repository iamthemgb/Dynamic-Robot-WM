"""P0d — passive straight and slope rolling."""
from ._rigid_shared import rigid_module
SCENARIO = rigid_module("P0d", "passive_no_franka", "rolling", fixture_policy="grounded rolling table or slope", controller_kind="passive", trajectory="none", retention_required=False)
