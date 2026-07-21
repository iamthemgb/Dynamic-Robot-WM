"""F3c — Genesis current-and-pickup contract (not implemented)."""
from ._rigid_shared import blocked_module
SCENARIO = blocked_module("F3c", "dynamic_handoff", "water_current_pickup", "source_genesis_fluid", fixture_policy="solver-derived current field and buoyant object", controller_kind="fluid_pickup", implementation_note="current scene is pending; pouring and visual-particle fallbacks are forbidden")
