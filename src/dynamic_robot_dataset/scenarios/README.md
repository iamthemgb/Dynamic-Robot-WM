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
| F2b | ramp_launch | `f2b_ramp_launch.py` | automated fixed-six pass; 600 Hz selected after dual-rate admission; human review pending |
| F2c | table_floor_bounce | `f2c_table_floor_bounce.py` | implemented |
| F2d | wall_barrier_rebound | `f2d_wall_barrier_rebound.py` | 5/6 automated fixed cases; Robotiq nominal blocked by the hardware-faithful retention/penetration/acceleration tradeoff |
| F2e | multi_surface_rebound | `f2e_multi_surface_rebound.py` | automated fixed-six pass; repaired rendered camera/background evidence is being regenerated; human review pending |
| F2f | arbitrary_surface_bounce | `f2f_arbitrary_surface_bounce.py` | automated fixed-six pass; rendered evidence and human review pending; positive Robotiq barrier catch still blocks pilot activation |
| F3a | oscillating_platform_handoff | `f3a_oscillating_platform_handoff.py` | contract only |
| F3b | rolling_pickup | `f3b_rolling_pickup.py` | implemented |
| F3c | water_current_pickup | `f3c_water_current_pickup.py` | contract only |
| F3d | moving_base_catch | `f3d_moving_base_catch.py` | contract only |
| D1 | cloth | `d1_cloth.py` | contract only |
| D2 | rope | `d2_rope.py` | contract only |

Use `python -m dynamic_robot_dataset.scenarios list` for the registry-bound
inventory and `python -m dynamic_robot_dataset.scenarios show F2b` for one
leaf's variants, embodiments, evaluator, blockers, and normal review command.

This status is deliberately leaf-scoped. The established P0a-d, F1a-d, F2a,
F2c, and F3b implementations have not been changed by the F2b/F2d/F2e/F2f
repair work. Automated fixed-six and timestep admission are preconditions for
rendered review, not human approval, pilot activation, or production release.

F1d and F2a additionally expose versioned projectile initial-state samplers:

```bash
python -m dynamic_robot_dataset.scenarios sample F1d --count 6 --seed 7
python -m dynamic_robot_dataset.scenarios sample F2a --count 6 --seed 7
```

These commands inspect deterministic `sampled_preview` specifications without
simulation. The sampled envelope is training-ineligible until its fixed seeds,
controller behavior, 600/1200 Hz comparison, and human review pass. The
official fixed-review recipes remain unchanged.

The older `families/*/adapter.py` modules are analytical/diagnostic adapters.
They are not canonical rendered MuJoCo scenario implementations and do not
constitute review, pilot, or production evidence.
