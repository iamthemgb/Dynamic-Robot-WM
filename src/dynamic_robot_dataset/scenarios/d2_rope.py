"""D2 — frictionally grasped rope/cable contract (not implemented)."""
from ._rigid_shared import blocked_module
SCENARIO = blocked_module("D2", "deformable", "rope", "source_mujoco_deformable", fixture_policy="self-contacting rope with frictional finger contact", controller_kind="rope_manipulation", implementation_note="frictional endpoint grasp is pending; equality-connect grasps are forbidden")
