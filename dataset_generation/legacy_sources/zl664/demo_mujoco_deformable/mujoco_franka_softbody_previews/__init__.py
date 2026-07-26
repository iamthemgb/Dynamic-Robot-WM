"""Preview pipeline for the Franka plush/sponge/soft-body packing family.

Tasks: press_release, squeeze_with_gripper, push_toward_box_opening,
compress_and_close_lid. Soft body = MuJoCo volumetric flexcomp grid
(softbody_model_type = "mujoco_flexcomp_grid"), full Franka Panda arm
(embodiment = "single_franka").
"""

__version__ = "0.1.0"
