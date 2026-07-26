"""MuJoCo Franka-rope deformable dataset generator (zss8).

Scaled successor to zl664's ``mujoco_franka_rope_previews`` preview pipeline.
Adds per-episode visual background domain randomization (plain studio,
kitchen, warehouse, workshop, lab, office, patio, garage, …) and five extra
dynamics-rich tasks, while keeping physics identical across backgrounds.

Tasks (rope / cable / string manipulation with a full Franka Panda):

  spec:
    1. tug_endpoint         - tug one endpoint up and release
    2. drag_endpoint        - drag one endpoint along a curved path
    3. wrap_around_post     - wrap the rope around one vertical post
    4. thread_through_ring  - thread the rope tail through one ring
  creative:
    5. shake_wave           - oscillate the endpoint -> traveling waves
    6. lift_and_drape       - carry the endpoint over a bar so the rope drapes
    7. coil_on_table        - spiral the endpoint inward to coil the rope
    8. twirl_overhead       - lift the endpoint and circle it so the rope trails
    9. sweep_aside          - non-prehensile sideways push of the rope
"""

__version__ = "0.3.0"  # split per-view videos (videos/<camera>/), per-view manifests
