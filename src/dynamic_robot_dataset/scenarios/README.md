# Canonical scenario modules

This is the only operator-facing directory for corpus scenario definitions.
The corpus YAML owns taxonomy and release state; each module here owns the
corresponding executable recipe or an explicit fail-closed contract. Shared
MuJoCo actuation, rendering, writing, QC, and evaluator algorithms remain in
their backend/common packages.

| Leaf | Subfamily | Module | Status |
|---|---|---|---|
| P0a | freefall | `p0a_freefall.py` | implemented |
| P0b | projectile | `p0b_projectile.py` | implemented |
| P0c | bounce_rebound | `p0c_bounce_rebound.py` | implemented |
| P0d | rolling | `p0d_rolling.py` | implemented |
| F1a | centered_vertical_drop | `f1a_centered_vertical_drop.py` | implemented |
| F1b | off_center_drop | `f1b_off_center_drop.py` | implemented |
| F1c | drifted_drop | `f1c_drifted_drop.py` | implemented |
| F1d | mild_projectile | `f1d_mild_projectile.py` | implemented |
| F2a | direct_interception | `f2a_direct_interception.py` | implemented |
| F2b | ramp_launch | `f2b_ramp_launch.py` | implemented, acceptance blocked |
| F2c | table_floor_bounce | `f2c_table_floor_bounce.py` | implemented |
| F2d | wall_barrier_rebound | `f2d_wall_barrier_rebound.py` | implemented, acceptance blocked |
| F2e | multi_surface_rebound | `f2e_multi_surface_rebound.py` | implemented, acceptance blocked |
| F2f | arbitrary_surface_bounce | `f2f_arbitrary_surface_bounce.py` | implemented, admission blocked |
| F3a | oscillating_platform_handoff | `f3a_oscillating_platform_handoff.py` | contract only |
| F3b | rolling_pickup | `f3b_rolling_pickup.py` | implemented |
| F3c | water_current_pickup | `f3c_water_current_pickup.py` | contract only |
| F3d | moving_base_catch | `f3d_moving_base_catch.py` | contract only |
| D1 | cloth | `d1_cloth.py` | contract only |
| D2 | rope | `d2_rope.py` | contract only |

Use `python -m dynamic_robot_dataset.scenarios list` for the registry-bound
inventory and `python -m dynamic_robot_dataset.scenarios show F2b` for one
leaf's variants, embodiments, evaluator, blockers, and normal review command.

The older `families/*/adapter.py` modules are analytical/diagnostic adapters.
They are not canonical rendered MuJoCo scenario implementations and do not
constitute review, pilot, or production evidence.
