"""D1 — frictionally grasped cloth contract (not implemented)."""
from ._rigid_shared import blocked_module
SCENARIO = blocked_module("D1", "deformable", "cloth", "source_mujoco_deformable", fixture_policy="self-contacting cloth with frictional finger contact", controller_kind="cloth_manipulation", implementation_note="frictional grasp is pending; equality-connect grasps are forbidden")
