"""F3a — oscillating-platform handoff contract (not implemented)."""
from ._rigid_shared import blocked_module
SCENARIO = blocked_module("F3a", "dynamic_handoff", "oscillating_platform_handoff", "source_mujoco", fixture_policy="owned physical oscillating platform", controller_kind="receive_or_handoff", implementation_note="net-new scene and independent evaluator are pending")
