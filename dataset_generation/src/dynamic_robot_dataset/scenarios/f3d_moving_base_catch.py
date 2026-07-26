"""F3d — moving-base projectile catch contract (not implemented)."""
from ._rigid_shared import blocked_module
SCENARIO = blocked_module("F3d", "dynamic_handoff", "moving_base_catch", "source_mujoco", fixture_policy="owned physical moving robot base", controller_kind="moving_base_catch", implementation_note="net-new scene and moving-base evaluator are pending")
