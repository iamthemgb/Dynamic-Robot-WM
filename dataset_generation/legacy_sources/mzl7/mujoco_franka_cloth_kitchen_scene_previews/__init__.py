"""MuJoCo Franka-cloth deformable preview generator.

Preview pipeline (NOT the final dataset generator) for the first
Franka-cloth deformable dataset family:

  1. poke_cloth                  - single Franka pokes cloth
  2. lift_corner_release         - single Franka lifts one cloth corner, releases
  3. fold_edge_fixed_line        - single Franka folds one cloth edge over a fixed line
  4. dual_franka_tshirt_fold_box - dual Franka folds a T-shirt-like cloth into a shallow box
"""

__version__ = "0.1.0"
