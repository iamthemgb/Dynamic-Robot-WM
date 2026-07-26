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
| F2b | ramp_launch | `f2b_ramp_launch.py` | automated fixed-six and dual-rate pass; F2b-only camera/background pre-commit checks pass all six; current-provenance sealed regeneration and human review pending |
| F2c | table_floor_bounce | `f2c_table_floor_bounce.py` | implemented |
| F2d | wall_barrier_rebound | `f2d_wall_barrier_rebound.py` | 5/6 automated fixed cases; Robotiq nominal blocked by the hardware-faithful retention/penetration/acceleration tradeoff |
| F2e | multi_surface_rebound | `f2e_multi_surface_rebound.py` | automated fixed-six pass; prior sealed 6/6 run becomes historical under the 0.20 provenance change; sealed regeneration and human review pending |
| F2f | arbitrary_surface_bounce | `f2f_arbitrary_surface_bounce.py` | automated fixed-six pass; prior sealed 6/6 run becomes historical under the 0.20 provenance change; sealed regeneration and human review pending; positive Robotiq barrier catch still blocks pilot activation |
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
repair work. An explicit 84-case established/non-F2b audit found no behavior
change. Automated fixed-six, timestep admission, and pre-commit rendering are
preconditions for sealed review; they are not human approval, pilot
activation, or production release.

The F2b-only rendered pre-commit check covers all six fixed cases. Every
initial/apex/key/final checkpoint is visible, the minimum full-trajectory
visible-frame fraction is 0.9242, and the minimum key-event target area is 75
pixels against a 64-pixel requirement. Clean R0 removes exactly the two
collision-disabled visual assets that intersect owned fixtures
(`lab_bench_leg_a` and `lab_workbench`); no physics geometry or non-F2b camera
path changes. A new sealed F2b run bound to backend `0.20.0-review` still has
to be generated and reviewed by a human.

The sealed F2e and F2f runs whose names end in `8d2aff0` passed their six
automated rendered cases, but become immutable historical evidence when the
current `0.20.0-review` source commit lands. They must be regenerated with the
current source hashes before human review. The corresponding `8d2aff0` F2b
run and review-suite plan are historical as well; none establishes current
admission.

F2f's positive Robotiq barrier branch remains honestly blocked. In the
bounded hardware-faithful search, the best retained catch penetrated 2.633 mm
(above the 2 mm limit), while the best penetration-compliant contact measured
1.956 mm but did not retain the ball. No runtime change from that search was
admitted.

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
